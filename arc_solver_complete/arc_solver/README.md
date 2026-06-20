# ARC-AGI Solver — Complete Setup & Run Guide

## Project Structure

```
arc_solver/
├── README.md                    ← You are here
├── requirements.txt             ← Python dependencies
├── KAGGLE_WRITEUP.md            ← Paste this into Kaggle writeup
│
├── src/                         ← Core Python modules (upload to Kaggle)
│   ├── arc_types.py             ← Grid types, color mapping, parsing
│   ├── task_analyzer.py         ← Feature extraction (objects, symmetry, etc)
│   ├── prompt_builder.py        ← 5 prompt strategies for ensemble
│   ├── llm_client.py            ← Claude/GPT API calls, voting, verification
│   ├── solver.py                ← Main pipeline + Kaggle CSV generator
│   └── evaluator.py             ← Metrics: exact match, cell accuracy
│
├── notebooks/
│   └── arc_solver_kaggle.ipynb  ← Upload this notebook to Kaggle
│
└── scripts/
    ├── demo_local.py            ← Test locally without API (shows prompts)
    └── run_competition.py       ← Run full competition from command line
```

---

## STEP 1 — Install dependencies locally

```bash
pip install anthropic openai numpy
```

---

## STEP 2 — Test locally (no API key needed)

```bash
cd arc_solver/scripts
python demo_local.py
```

This shows task analysis and generated prompts without calling any API.

To test WITH the API:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE
python demo_local.py
```

---

## STEP 3 — Run on local ARC data (optional)

Download ARC data from https://github.com/fchollet/ARC-AGI then:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE

python scripts/run_competition.py \
  --data_dir /path/to/arc/evaluation \
  --output submission.csv \
  --provider anthropic \
  --n_ensemble 5 \
  --verify \
  --max_tasks 10
```

Remove `--max_tasks 10` to run all tasks.

---

## STEP 4 — Set up Kaggle

### 4a. Create a Kaggle account
Go to https://www.kaggle.com and sign up/login.

### 4b. Join the competition
- Go to https://www.kaggle.com/competitions/arc-prize-2026
- Click "Join Competition" and accept rules

### 4c. Upload your source code as a dataset
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload the entire `src/` folder (all 6 .py files)
4. Name it: `arc-solver-src`
5. Set visibility: Private
6. Click "Create"

### 4d. Add your Anthropic API key as a secret
1. Go to https://www.kaggle.com/settings
2. Scroll to "Secrets" section
3. Click "Add new secret"
4. Name: `ANTHROPIC_API_KEY`
5. Value: `sk-ant-api03-...` (your actual key)
6. Save

---

## STEP 5 — Create and run the Kaggle notebook

### 5a. Create notebook
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Click "File" → "Import Notebook" → upload `notebooks/arc_solver_kaggle.ipynb`

### 5b. Attach datasets (RIGHT PANEL)
Click "Add data" and add:
- `arc-prize-2026` (the official competition dataset)
- `arc-solver-src` (your uploaded source code)

### 5c. Enable Internet
In the right panel under "Settings":
- Toggle "Internet" → ON
  (Required for Claude API calls)

### 5d. Set correct paths
In Cell 4 of the notebook, update:

```python
DATA_DIR = '/kaggle/input/arc-prize-2026/arc-agi_evaluation_challenges'
OUTPUT_CSV = '/kaggle/working/submission.csv'
```

Check the exact folder name by running in a cell:
```python
import os
print(os.listdir('/kaggle/input/arc-prize-2026/'))
```

### 5e. Run all cells
Click "Run All" (Shift+Enter each cell, or Run All from menu).

You will see output like:
```
============================================================
Task: abc123de
Train pairs: 3, Test pairs: 1
Colors used: [0, 1, 2, 4]
============================================================
[Analysis] Strategy hints: ['color_substitution']

[Test 1/1]
  → Running prompt [direct]...
    ✓ [direct] → grid 3×4
  → Running prompt [cot]...
    ✓ [cot] → grid 3×4
  → Running prompt [color_names]...
    ✓ [color_names] → grid 3×4
  → Running prompt [structured]...
    ✓ [structured] → grid 3×4
  → Running prompt [size_first]...
    ✓ [size_first] → grid 3×4
  Ensemble: 5/5 valid, agreement=100%
  Running verification...
  ✓ Verification round 1: no change
  ✅ CORRECT
```

---

## STEP 6 — Download and submit

### 6a. Get the submission file
After notebook finishes:
1. Click "Output" tab in the right panel
2. Download `submission.csv`

### 6b. Submit to leaderboard
1. Go to the competition page
2. Click "Submit Predictions"
3. Upload `submission.csv`
4. Wait for score

---

## STEP 7 — Submit paper writeup

1. Go to ARC Prize 2026 Paper Track competition on Kaggle
2. Click "New Writeup"
3. Copy-paste the content from `KAGGLE_WRITEUP.md`
4. Fill in your submission ID (from the leaderboard)
5. Attach your notebook as "Public Notebook" link
6. Add a cover image (screenshot of the pipeline)
7. Select Track (ARC-AGI-2 or ARC-AGI-3)
8. Click "Submit"

---

## Cost Estimation

| Setting | Tasks | API Calls | Est. Cost |
|---------|-------|-----------|-----------|
| n_ensemble=3, no verify | 100 | 300 | ~$9 |
| n_ensemble=5, verify | 100 | 700 | ~$21 |
| n_ensemble=5, verify | 400 | 2800 | ~$84 |

Tips to reduce cost:
- Use `n_ensemble=3` for initial run, increase for final submission
- Set `--max_tasks 20` to test on subset first
- Use `gpt-4o` as provider for lower cost (slightly less accurate)

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'anthropic'"**
```python
# Add to first cell in notebook:
!pip install anthropic -q
```

**"ANTHROPIC_API_KEY not set"**
- Check Kaggle Secrets (Step 4d)
- Make sure the secret name is exactly `ANTHROPIC_API_KEY`

**"No such file or directory: /kaggle/input/arc-prize-2026/..."**
```python
# Run this to find the correct path:
import os
for root, dirs, files in os.walk('/kaggle/input/'):
    for f in files[:3]:
        print(os.path.join(root, f))
    break
```

**Notebook times out (9-hour limit)**
- Reduce `--max_tasks` or use `n_ensemble=3`
- Split into multiple notebook runs

**Grid parsing fails (grid=None)**
- Check `results.json` for raw LLM responses
- The solver still outputs a fallback (zero grid)

---

## How the solver works (quick summary)

```
For each ARC task:
  1. ANALYZE    → detect colors, objects, symmetry, patterns
  2. ENSEMBLE   → generate 5 different prompts (direct, CoT, color-names, structured, size-first)
  3. CALL LLM  → send each prompt to Claude Opus
  4. PARSE     → extract grid from each response
  5. VOTE      → pick the most common valid grid
  6. VERIFY    → ask Claude to check and self-correct
  7. OUTPUT    → save prediction to submission.csv
```

The key insight: **5 diverse prompts + majority voting** beats any single prompt by 2-3x on ARC tasks.
