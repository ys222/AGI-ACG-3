"""
Evaluator — measures solver performance on ARC tasks.
Supports: exact match, partial credit, per-task breakdown.
"""

from __future__ import annotations
import json
from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np
from arc_types import ARCTask, Grid, grids_equal


def exact_match(pred: Optional[Grid], target: Grid) -> bool:
    if pred is None:
        return False
    return grids_equal(pred, target)


def cell_accuracy(pred: Optional[Grid], target: Grid) -> float:
    """Fraction of cells correct."""
    if pred is None:
        return 0.0
    if len(pred) != len(target):
        return 0.0
    total = sum(len(r) for r in target)
    if total == 0:
        return 1.0
    correct = 0
    for rp, rt in zip(pred, target):
        if len(rp) != len(rt):
            return 0.0
        correct += sum(p == t for p, t in zip(rp, rt))
    return correct / total


def shape_match(pred: Optional[Grid], target: Grid) -> bool:
    if pred is None:
        return False
    return (len(pred) == len(target) and
            (not pred or not target or len(pred[0]) == len(target[0])))


def evaluate_solver_results(results: Dict, tasks: Dict[str, ARCTask]) -> Dict:
    """
    Evaluate a results dict (from ARCSolver.solve_dataset) against ground truth.
    """
    task_scores = {}
    n_exact = 0
    n_shape = 0
    total_cell_acc = 0.0
    n_total = 0

    for task_id, task_result in results.get("tasks", {}).items():
        task = tasks.get(task_id)
        if task is None:
            continue

        task_exact = 0
        task_cell = 0.0
        count = 0

        for test_idx, res in task_result.items():
            idx = int(test_idx)
            if idx >= len(task.test):
                continue
            gt = task.test[idx].output
            pred = res.get("prediction")

            em = exact_match(pred, gt)
            ca = cell_accuracy(pred, gt)
            sm = shape_match(pred, gt)

            task_exact += int(em)
            task_cell += ca
            count += 1

            n_exact += int(em)
            n_shape += int(sm)
            total_cell_acc += ca
            n_total += 1

        task_scores[task_id] = {
            "exact_match": task_exact / count if count else 0,
            "cell_accuracy": task_cell / count if count else 0,
            "n_tests": count,
        }

    return {
        "exact_match_rate": n_exact / n_total if n_total else 0.0,
        "shape_match_rate": n_shape / n_total if n_total else 0.0,
        "mean_cell_accuracy": total_cell_acc / n_total if n_total else 0.0,
        "n_correct": n_exact,
        "n_total": n_total,
        "per_task": task_scores,
    }


def print_evaluation_report(eval_results: Dict):
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"Exact Match Rate:  {eval_results['exact_match_rate']:.1%}  "
          f"({eval_results['n_correct']}/{eval_results['n_total']})")
    print(f"Shape Match Rate:  {eval_results['shape_match_rate']:.1%}")
    print(f"Mean Cell Acc:     {eval_results['mean_cell_accuracy']:.1%}")
    print("\nPer-task breakdown:")
    for tid, ts in sorted(eval_results["per_task"].items(),
                          key=lambda x: -x[1]["exact_match"]):
        em = ts["exact_match"]
        ca = ts["cell_accuracy"]
        marker = "✅" if em == 1.0 else ("🟡" if ca > 0.5 else "❌")
        print(f"  {marker} {tid}: exact={em:.0%} cell_acc={ca:.0%}")


def load_tasks_from_dir(data_dir: str) -> Dict[str, ARCTask]:
    tasks = {}
    for tf in Path(data_dir).glob("*.json"):
        task = ARCTask.from_file(str(tf))
        tasks[task.task_id] = task
    return tasks


def error_analysis(eval_results: Dict, tasks: Dict[str, ARCTask],
                   results: Dict) -> List[Dict]:
    """Identify failure patterns for targeted improvement."""
    failures = []
    for task_id, ts in eval_results["per_task"].items():
        if ts["exact_match"] < 1.0:
            task = tasks.get(task_id)
            task_result = results.get("tasks", {}).get(task_id, {})
            failures.append({
                "task_id": task_id,
                "cell_accuracy": ts["cell_accuracy"],
                "strategy_hints": (list(task_result.values())[0].get("features", [])
                                   if task_result else []),
                "n_train": len(task.train) if task else 0,
            })
    failures.sort(key=lambda x: -x["cell_accuracy"])
    return failures
