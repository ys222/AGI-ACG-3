"""
ARC Solver — main orchestration logic.
Combines: task analysis → prompt ensemble → voting → verification → output
"""

from __future__ import annotations
import os, json, time
from typing import List, Dict, Optional, Any
from pathlib import Path

from arc_types import ARCTask, Grid, grids_equal, grid_to_text, task_summary
from task_analyzer import analyze_task, features_to_text
from prompt_builder import build_multishot_ensemble_prompts, SYSTEM_PROMPT
from llm_client import run_ensemble, refine_with_verification, call_llm, extract_grid_from_response


class ARCSolver:
    """
    Full ARC solving pipeline:
      1. Analyze task structure
      2. Build diverse prompt ensemble
      3. Run LLM calls
      4. Ensemble vote
      5. Optional verification pass
      6. Return predictions
    """

    def __init__(self,
                 provider: str = "anthropic",
                 model: str = "claude-opus-4-6",
                 n_ensemble: int = 5,
                 do_verification: bool = True,
                 verify_rounds: int = 1,
                 verbose: bool = True):
        self.provider = provider
        self.model = model
        self.n_ensemble = n_ensemble
        self.do_verification = do_verification
        self.verify_rounds = verify_rounds
        self.verbose = verbose

    def solve_task(self, task: ARCTask) -> Dict[str, Any]:
        """
        Solve all test examples in a task.
        Returns dict: {test_idx: {'prediction': grid, 'confidence': float, ...}}
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(task_summary(task))
            print('='*60)

        # Step 1: Analyze
        features = analyze_task(task)
        if self.verbose:
            print(f"\n[Analysis] Strategy hints: {features['strategy_hints']}")

        results = {}

        for test_idx in range(len(task.test)):
            if self.verbose:
                print(f"\n[Test {test_idx+1}/{len(task.test)}]")

            # Step 2: Build ensemble prompts
            prompts = build_multishot_ensemble_prompts(
                task, test_idx=test_idx,
                features=features,
                n=self.n_ensemble
            )

            # Step 3 & 4: Run + Vote
            ensemble_result = run_ensemble(
                prompts, provider=self.provider, verbose=self.verbose
            )
            prediction = ensemble_result["winner"]
            confidence = ensemble_result["agreement"]

            if self.verbose:
                valid = ensemble_result["valid_count"]
                total = ensemble_result["total"]
                print(f"  Ensemble: {valid}/{total} valid, "
                      f"agreement={confidence:.0%}")

            # Step 5: Verification (if confident enough and got a result)
            if (prediction is not None and
                    self.do_verification and
                    confidence >= 0.4):
                if self.verbose:
                    print(f"  Running verification...")
                prediction = refine_with_verification(
                    task, test_idx, prediction,
                    provider=self.provider,
                    max_rounds=self.verify_rounds
                )

            # Ground truth check (if available)
            gt = task.test[test_idx].output
            is_correct = None
            if gt and gt != [[0]]:
                is_correct = grids_equal(prediction, gt) if prediction else False
                if self.verbose:
                    status = "✅ CORRECT" if is_correct else "❌ WRONG"
                    print(f"  {status}")

            results[test_idx] = {
                "prediction": prediction,
                "confidence": confidence,
                "is_correct": is_correct,
                "ensemble_details": ensemble_result,
                "features": features["strategy_hints"],
            }

        return results

    def solve_file(self, path: str) -> Dict:
        task = ARCTask.from_file(path)
        return self.solve_task(task)

    def solve_dataset(self, data_dir: str,
                      output_path: Optional[str] = None) -> Dict:
        """
        Solve all tasks in a directory. Saves results to JSON.
        """
        task_files = sorted(Path(data_dir).glob("*.json"))
        all_results = {}
        correct = 0
        total = 0
        start = time.time()

        for tf in task_files:
            task = ARCTask.from_file(str(tf))
            task_result = self.solve_task(task)
            all_results[task.task_id] = task_result

            for res in task_result.values():
                if res["is_correct"] is not None:
                    total += 1
                    if res["is_correct"]:
                        correct += 1

        elapsed = time.time() - start
        summary = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0.0,
            "elapsed_s": elapsed,
            "tasks": all_results,
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(self._serialize_results(summary), f, indent=2)
            print(f"\nResults saved to {output_path}")

        print(f"\n{'='*60}")
        print(f"FINAL: {correct}/{total} correct ({100*correct/total if total else 0:.1f}%)")
        print(f"Time: {elapsed:.1f}s")
        return summary

    def _serialize_results(self, results: Dict) -> Dict:
        """Make results JSON-serializable."""
        import copy
        r = copy.deepcopy(results)

        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        return clean(r)


# ── Kaggle submission format ─────────────────────────────────────────────────

def generate_kaggle_submission(solver: ARCSolver,
                                test_dir: str,
                                output_csv: str = "submission.csv"):
    """
    Generate submission.csv in Kaggle ARC format:
    task_id_[attempt_1|attempt_2], output_id, grid
    """
    import csv
    task_files = sorted(Path(test_dir).glob("*.json"))
    rows = []

    for tf in task_files:
        task = ARCTask.from_file(str(tf))
        task_result = solver.solve_task(task)

        for test_idx, res in task_result.items():
            pred = res["prediction"]
            task_id = task.task_id

            if pred:
                # ARC submission: grid as flat string per attempt
                grid_str = "|".join(" ".join(str(c) for c in row) for row in pred)
            else:
                # fallback: all zeros, same size as test input
                inp = task.test[test_idx].input
                H, W = len(inp), len(inp[0]) if inp else (3, 3)
                grid_str = "|".join(" ".join(["0"]*W) for _ in range(H))

            # Two attempts: primary + fallback (repeat or zeros)
            rows.append({
                "output_id": f"{task_id}_{test_idx}",
                "output": grid_str,
            })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["output_id", "output"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Submission saved: {output_csv} ({len(rows)} rows)")
    return output_csv
