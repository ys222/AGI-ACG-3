#!/usr/bin/env python3
"""
Quick setup and verification script.
Run this first to check everything is installed and working.

Usage:
  python setup_and_verify.py
  python setup_and_verify.py --api-test      # also test API connection
"""

import sys, os, subprocess, argparse

def check(label, fn):
    try:
        result = fn()
        print(f"  ✅ {label}: {result}")
        return True
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-test", action="store_true",
                        help="Test Claude API connection (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  ARC-AGI Solver — Setup Verification")
    print("="*55)

    # Python version
    print("\n[1] Python version")
    check("Python 3.8+", lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Dependencies
    print("\n[2] Dependencies")
    check("numpy", lambda: __import__("numpy").__version__)
    check("anthropic", lambda: __import__("anthropic").__version__)

    try:
        __import__("openai")
        check("openai (optional)", lambda: __import__("openai").__version__)
    except ImportError:
        print("  ⚠️  openai: not installed (optional, only needed for GPT fallback)")

    # Source modules
    print("\n[3] Source modules")
    src_path = os.path.join(os.path.dirname(__file__), "src")
    sys.path.insert(0, src_path)

    check("arc_types", lambda: "OK" if __import__("arc_types") else "?")
    check("task_analyzer", lambda: "OK" if __import__("task_analyzer") else "?")
    check("prompt_builder", lambda: "OK" if __import__("prompt_builder") else "?")
    check("llm_client", lambda: "OK" if __import__("llm_client") else "?")
    check("solver", lambda: "OK" if __import__("solver") else "?")
    check("evaluator", lambda: "OK" if __import__("evaluator") else "?")

    # Pipeline smoke test
    print("\n[4] Pipeline smoke test (no API)")
    def run_pipeline():
        from arc_types import ARCTask
        from task_analyzer import analyze_task
        from prompt_builder import build_multishot_ensemble_prompts

        task = ARCTask.from_dict("test", {
            "train": [{"input": [[1,0],[0,1]], "output": [[2,0],[0,2]]}],
            "test":  [{"input": [[0,1],[1,0]], "output": [[0,2],[2,0]]}]
        })
        features = analyze_task(task)
        prompts = build_multishot_ensemble_prompts(task, test_idx=0, features=features, n=3)
        assert len(prompts) == 3
        assert features["strategy_hints"]
        return f"{len(prompts)} prompts, hints={features['strategy_hints']}"
    check("Full pipeline", run_pipeline)

    # API key check
    print("\n[5] API key")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        check("ANTHROPIC_API_KEY", lambda: f"set ({api_key[:12]}...)")
    else:
        print("  ⚠️  ANTHROPIC_API_KEY: not set")
        print("     Set it with: export ANTHROPIC_API_KEY=sk-ant-api03-...")

    # Live API test
    if args.api_test:
        print("\n[6] Live Claude API test")
        if not api_key:
            print("  ❌ Cannot test: ANTHROPIC_API_KEY not set")
        else:
            def test_api():
                from llm_client import call_claude
                from prompt_builder import SYSTEM_PROMPT
                resp = call_claude(
                    SYSTEM_PROMPT,
                    "Task: [[1]] → [[2]], [[3]] → [[4]]\nTest: [[5]]\nOutput:",
                    max_tokens=20
                )
                return f"response='{resp.strip()[:30]}'"
            check("Claude API call", test_api)

    print("\n" + "="*55)

    # Install missing deps
    try:
        import anthropic
    except ImportError:
        print("\n⚠️  Missing dependencies. Install with:")
        print("  pip install anthropic numpy")
        print("  # or:")
        print("  pip install -r requirements.txt")

    print("\nAll checks done.")
    print("\nNext steps:")
    print("  1. python scripts/demo_local.py           # test without API")
    print("  2. export ANTHROPIC_API_KEY=sk-ant-...")
    print("  3. python scripts/demo_local.py           # test with API")
    print("  4. Upload src/ to Kaggle and run notebook")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()
