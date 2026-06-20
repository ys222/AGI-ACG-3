"""
Prompt Builder — constructs rich, structured prompts for LLM-based ARC solving.

Strategy: Multi-stage prompting
  Stage 1 → Observation: describe what you see
  Stage 2 → Rule induction: infer the transformation rule
  Stage 3 → Application: apply rule to test input
  Stage 4 → Verification: self-check the output
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from arc_types import ARCTask, ARCExample, Grid, COLOR_NAMES, grid_to_text, grid_to_compact


# ── System prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles.

ARC puzzles present input/output grid pairs where each cell has a color value 0–9:
0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=grey, 6=fuschia, 7=orange, 8=azure, 9=maroon

Your task is to:
1. Study ALL training pairs carefully
2. Identify the EXACT transformation rule that turns every input into its output
3. Apply that rule to the test input
4. Output ONLY the resulting grid — no explanation, no markdown, just rows of space-separated digits

Rules:
- Every training example follows the same rule
- The rule is always deterministic and consistent
- Look for: color substitution, object movement, rotation, reflection, tiling, counting, pattern completion
- Pay attention to: object shapes, relative positions, colors, sizes, symmetry
- Output must be a valid grid: rectangular, digits 0–9 only

Output format: rows of space-separated integers, one row per line. Nothing else."""


SYSTEM_PROMPT_COT = """You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles.

ARC grids use colors 0–9: 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=grey, 6=fuschia, 7=orange, 8=azure, 9=maroon

Work through this step by step:

STEP 1 - OBSERVE: For each training pair, describe what you see in the input and output.
STEP 2 - HYPOTHESIZE: What transformation rule explains ALL training pairs?
STEP 3 - VERIFY: Mentally apply your rule to each training input. Does it produce the correct output?
STEP 4 - APPLY: Apply the verified rule to the test input.
STEP 5 - OUTPUT: Write the output grid as rows of space-separated digits.

End your response with:
<output>
[your grid here]
</output>"""


SYSTEM_PROMPT_STRUCTURED = """You are an expert ARC puzzle solver. You think carefully and systematically.

Grid color legend: 0=black 1=blue 2=red 3=green 4=yellow 5=grey 6=fuschia 7=orange 8=azure 9=maroon

You will be given training examples showing input→output transformations, then a test input.
Find the rule and apply it.

Always end your response with the answer grid inside <answer> tags:
<answer>
0 1 2
3 4 5
</answer>"""


# ── Prompt construction ──────────────────────────────────────────────────────

def format_grid_block(grid: Grid, label: str, use_color_hint: bool = False) -> str:
    lines = [f"{label}:"]
    for r, row in enumerate(grid):
        row_str = " ".join(str(c) for c in row)
        if use_color_hint and len(row) <= 10:
            color_hint = " ".join(COLOR_NAMES[c][0].upper() for c in row)
            row_str += f"  [{color_hint}]"
        lines.append(row_str)
    return "\n".join(lines)


def build_base_prompt(task: ARCTask, test_idx: int = 0,
                      include_size_hints: bool = True,
                      include_color_names: bool = False) -> str:
    parts = []
    parts.append(f"=== ARC Task: {task.task_id} ===\n")
    parts.append(f"You have {len(task.train)} training example(s).\n")

    for i, ex in enumerate(task.train):
        parts.append(f"--- Training Example {i+1} ---")
        parts.append(format_grid_block(ex.input, "Input", include_color_names))
        if include_size_hints:
            parts.append(f"  (size: {len(ex.input)}×{len(ex.input[0])})")
        parts.append(format_grid_block(ex.output, "Output", include_color_names))
        if include_size_hints:
            parts.append(f"  (size: {len(ex.output)}×{len(ex.output[0])})\n")
        else:
            parts.append("")

    test_ex = task.test[test_idx]
    parts.append("--- Test ---")
    parts.append(format_grid_block(test_ex.input, "Input", include_color_names))
    if include_size_hints:
        parts.append(f"  (size: {len(test_ex.input)}×{len(test_ex.input[0])})")
    parts.append("\nOutput:")
    return "\n".join(parts)


def build_cot_prompt(task: ARCTask, test_idx: int = 0,
                     features: Optional[Dict] = None) -> str:
    """Chain-of-thought prompt with optional feature hints."""
    parts = [build_base_prompt(task, test_idx, include_size_hints=True)]

    if features and features.get("strategy_hints"):
        hints = features["strategy_hints"]
        parts.append(f"\n[Hint: This task may involve: {', '.join(hints)}]")

    parts.append("\nThink step by step. After your analysis, provide the output grid.")
    return "\n".join(parts)


def build_few_shot_prompt(task: ARCTask, test_idx: int = 0,
                          few_shot_examples: Optional[List[str]] = None) -> str:
    """Include solved ARC example(s) as few-shot demonstrations."""
    header = ""
    if few_shot_examples:
        header = "Here are some example ARC solutions to calibrate your reasoning:\n\n"
        header += "\n\n---\n\n".join(few_shot_examples)
        header += "\n\n--- Now solve the following task ---\n\n"
    return header + build_base_prompt(task, test_idx)


def build_verification_prompt(task: ARCTask, test_idx: int,
                               proposed_output: Grid) -> str:
    """Ask LLM to verify and correct a proposed output."""
    base = build_base_prompt(task, test_idx)
    proposed_str = grid_to_text(proposed_output)
    prompt = f"""{base}

A proposed output for the test input is:
{proposed_str}

Is this correct? Check by applying the transformation rule to the test input.
If the proposed output is wrong, provide the corrected output grid.
If it is correct, repeat it exactly.

Output only the (corrected) grid:"""
    return prompt


def build_output_size_prompt(task: ARCTask, test_idx: int = 0) -> str:
    """First ask the LLM to predict the output size, then solve."""
    base = build_base_prompt(task, test_idx)
    prompt = f"""{base}

Before giving the output, state: what are the dimensions (rows × columns) of the output?
Then provide the complete output grid."""
    return prompt


def build_multishot_ensemble_prompts(task: ARCTask, test_idx: int = 0,
                                     features: Optional[Dict] = None,
                                     n: int = 5) -> List[Dict]:
    """
    Generate N diverse prompts for ensemble voting.
    Different temperatures/phrasings → more diverse outputs → majority vote.
    """
    prompts = []

    # 1. Direct / zero-shot
    prompts.append({
        "name": "direct",
        "system": SYSTEM_PROMPT,
        "user": build_base_prompt(task, test_idx),
        "temperature": 0.0,
    })

    # 2. Chain of thought
    prompts.append({
        "name": "cot",
        "system": SYSTEM_PROMPT_COT,
        "user": build_cot_prompt(task, test_idx, features),
        "temperature": 0.2,
    })

    # 3. Color names
    prompts.append({
        "name": "color_names",
        "system": SYSTEM_PROMPT,
        "user": build_base_prompt(task, test_idx, include_color_names=True),
        "temperature": 0.0,
    })

    # 4. Structured output
    prompts.append({
        "name": "structured",
        "system": SYSTEM_PROMPT_STRUCTURED,
        "user": build_base_prompt(task, test_idx),
        "temperature": 0.1,
    })

    # 5. Size-first reasoning
    prompts.append({
        "name": "size_first",
        "system": SYSTEM_PROMPT_COT,
        "user": build_output_size_prompt(task, test_idx),
        "temperature": 0.2,
    })

    return prompts[:n]
