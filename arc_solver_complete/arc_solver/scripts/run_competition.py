"""
Competition Runner — solve ARC-AGI-2 or ARC-AGI-3 and generate Kaggle submission.

Usage:
  python run_competition.py --data_dir /path/to/arc/evaluation \
                             --output submission.csv \
                             --provider anthropic \
                             --n_ensemble 5 \
                             --verify

Expects data_dir to contain .json files in ARC format:
  {"train": [{"input": [...], "output": [...]}, ...],
   "test":  [{"input": [...]}]}
"""

import argparse, os, sys, json, time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser(description="ARC-AGI Competition Solver")
    p.add_argument("--data_dir", required=True,
                   help="Directory with ARC task JSON files")
    p.add_argument("--output", default="submission.csv",
                   help="Output submission CSV path")
    p.add_argument("--provider", choices=["anthropic", "openai"],
                   default="anthropic")
    p.add_argument("--model", default="claude-opus-4-6",
                   help="Model name (claude-opus-4-6 / gpt-4o)")
    p.add_argument("--n_ensemble", type=int, default=5,
                   help="Number of ensemble prompts per test")
    p.add_argument("--verify", action="store_true",
                   help="Enable verification pass")
    p.add_argument("--max_tasks", type=int, default=None,
                   help="Limit number of tasks (for testing)")
    p.add_argument("--results_json", default="results.json",
                   help="Save detailed results to JSON")
    return p.parse_args()


def main():
    args = parse_args()

    # Check API key
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    from solver import ARCSolver, generate_kaggle_submission
    from arc_types import ARCTask
    from evaluator import evaluate_solver_results, print_evaluation_report, load_tasks_from_dir

    print(f"\n{'='*60}")
    print(f"ARC-AGI COMPETITION SOLVER")
    print(f"Provider: {args.provider} / Model: {args.model}")
    print(f"Ensemble size: {args.n_ensemble} | Verification: {args.verify}")
    print(f"Data dir: {args.data_dir}")
    print(f"{'='*60}\n")

    solver = ARCSolver(
        provider=args.provider,
        model=args.model,
        n_ensemble=args.n_ensemble,
        do_verification=args.verify,
        verify_rounds=1,
        verbose=True,
    )

    # Discover tasks
    task_files = sorted(Path(args.data_dir).glob("*.json"))
    if args.max_tasks:
        task_files = task_files[:args.max_tasks]

    print(f"Found {len(task_files)} task files\n")

    all_results = {"tasks": {}}
    correct = 0
    total = 0
    start = time.time()

    for tf in task_files:
        task = ARCTask.from_file(str(tf))
        task_result = solver.solve_task(task)
        all_results["tasks"][task.task_id] = {
            str(k): {
                "prediction": v["prediction"],
                "confidence": v["confidence"],
                "is_correct": v["is_correct"],
                "features": v["features"],
            }
            for k, v in task_result.items()
        }

        for res in task_result.values():
            if res["is_correct"] is not None:
                total += 1
                if res["is_correct"]:
                    correct += 1

    elapsed = time.time() - start
    accuracy = correct / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{total} correct ({100*accuracy:.1f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(task_files):.1f}s/task)")

    # Save results
    with open(args.results_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved: {args.results_json}")

    # Generate submission
    generate_kaggle_submission(solver, args.data_dir, args.output)
    print(f"Submission: {args.output}")


if __name__ == "__main__":
    main()
