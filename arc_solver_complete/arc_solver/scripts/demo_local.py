"""
Demo: Run the ARC solver pipeline on sample tasks.
This demo works without an API key — it shows the analysis + prompt generation.
Set ANTHROPIC_API_KEY env var to run live LLM inference.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arc_types import ARCTask, grid_to_text, task_summary
from task_analyzer import analyze_task, features_to_text
from prompt_builder import build_multishot_ensemble_prompts

# ── Sample ARC tasks (built-in, no download needed) ─────────────────────────

SAMPLE_TASKS = {
    "color_swap": {
        "train": [
            {"input": [[1,2],[3,1]], "output": [[2,1],[3,2]]},
            {"input": [[1,1,2],[2,3,1]], "output": [[2,2,1],[1,3,2]]},
        ],
        "test": [
            {"input": [[1,3,2,1],[2,1,1,3]], "output": [[2,3,1,2],[1,2,2,3]]}
        ]
    },
    "tiling_2x2": {
        "train": [
            {"input": [[1,2],[3,4]], "output": [[1,2,1,2],[3,4,3,4],[1,2,1,2],[3,4,3,4]]},
            {"input": [[5,6],[7,8]], "output": [[5,6,5,6],[7,8,7,8],[5,6,5,6],[7,8,7,8]]},
        ],
        "test": [
            {"input": [[1,3],[2,4]], "output": [[1,3,1,3],[2,4,2,4],[1,3,1,3],[2,4,2,4]]}
        ]
    },
    "border_fill": {
        "train": [
            {
                "input": [[0,0,0],[0,1,0],[0,0,0]],
                "output": [[2,2,2],[2,1,2],[2,2,2]]
            },
            {
                "input": [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]],
                "output": [[2,2,2,2],[2,1,1,2],[2,1,1,2],[2,2,2,2]]
            },
        ],
        "test": [
            {
                "input": [[0,0,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,0,0]],
                "output": [[2,2,2,2,2],[2,1,1,1,2],[2,1,0,1,2],[2,1,1,1,2],[2,2,2,2,2]]
            }
        ]
    },
    "gravity_down": {
        "train": [
            {
                "input": [[1,0,0],[0,0,0],[0,0,0]],
                "output": [[0,0,0],[0,0,0],[1,0,0]]
            },
            {
                "input": [[0,2,0],[0,0,0],[0,0,0]],
                "output": [[0,0,0],[0,0,0],[0,2,0]]
            },
            {
                "input": [[1,0,2],[0,0,0],[0,0,0]],
                "output": [[0,0,0],[0,0,0],[1,0,2]]
            },
        ],
        "test": [
            {
                "input": [[3,0,0,4],[0,0,0,0],[0,0,0,0]],
                "output": [[0,0,0,0],[0,0,0,0],[3,0,0,4]]
            }
        ]
    }
}


def run_demo():
    print("="*70)
    print("ARC-AGI SOLVER — DEMO MODE")
    print("(Shows analysis + prompts; set ANTHROPIC_API_KEY for live solving)")
    print("="*70)

    has_api = bool(os.environ.get("ANTHROPIC_API_KEY"))

    for task_id, task_data in SAMPLE_TASKS.items():
        task = ARCTask.from_dict(task_id, task_data)

        print(f"\n{'─'*60}")
        print(task_summary(task))

        # Analysis
        features = analyze_task(task)
        print("\n[ANALYSIS]")
        print(features_to_text(features))

        # Prompts
        prompts = build_multishot_ensemble_prompts(task, test_idx=0, features=features, n=2)
        print(f"\n[PROMPTS GENERATED: {len(prompts)}]")
        for p in prompts:
            print(f"  • [{p['name']}] temperature={p['temperature']}")
            print(f"    Prompt preview: {p['user'][:120].replace(chr(10),' ')}...")

        # Live solving
        if has_api:
            from solver import ARCSolver
            from arc_types import grids_equal

            print(f"\n[SOLVING with Claude API...]")
            solver = ARCSolver(
                provider="anthropic",
                model="claude-opus-4-6",
                n_ensemble=3,
                do_verification=True,
                verbose=True,
            )
            results = solver.solve_task(task)
            for idx, res in results.items():
                pred = res["prediction"]
                gt = task.test[idx].output
                correct = grids_equal(pred, gt) if pred else False
                print(f"\n  Test[{idx}] — {'✅ CORRECT' if correct else '❌ WRONG'}")
                if pred:
                    print("  Prediction:")
                    for row in pred:
                        print("   ", " ".join(str(c) for c in row))
                    print("  Ground truth:")
                    for row in gt:
                        print("   ", " ".join(str(c) for c in row))
        else:
            print("\n[Skipping live API calls — ANTHROPIC_API_KEY not set]")
            print("[Set it and re-run to see live solving results]")

    print("\n" + "="*70)
    print("Demo complete.")
    if not has_api:
        print("\nTo run with live LLM solving:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  python demo_local.py")


if __name__ == "__main__":
    run_demo()
