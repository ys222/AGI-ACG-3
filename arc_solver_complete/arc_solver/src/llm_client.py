"""
LLM Client — wraps Anthropic & OpenAI APIs with retry logic,
ensemble voting, and robust output parsing.
"""

from __future__ import annotations
import os, time, re, json
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter
from arc_types import Grid, parse_grid_from_text, grids_equal, grid_to_text


# ── API clients ──────────────────────────────────────────────────────────────

def get_anthropic_client():
    try:
        import anthropic
        return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    except ImportError:
        raise ImportError("Run: pip install anthropic")


def get_openai_client():
    try:
        import openai
        return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    except ImportError:
        raise ImportError("Run: pip install openai")


# ── Single LLM call ──────────────────────────────────────────────────────────

def call_claude(system: str, user: str,
                model: str = "claude-opus-4-6",
                temperature: float = 0.0,
                max_tokens: int = 2048,
                max_retries: int = 3) -> str:
    """Call Claude API with exponential backoff."""
    client = get_anthropic_client()
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return msg.content[0].text
        except Exception as e:
            wait = 2 ** attempt
            print(f"[Claude] Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Claude API failed after max retries")


def call_gpt(system: str, user: str,
             model: str = "gpt-4o",
             temperature: float = 0.0,
             max_tokens: int = 2048,
             max_retries: int = 3) -> str:
    """Call OpenAI API with exponential backoff."""
    client = get_openai_client()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            print(f"[GPT] Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("OpenAI API failed after max retries")


def call_llm(system: str, user: str, provider: str = "anthropic", **kwargs) -> str:
    """Unified LLM call dispatcher."""
    if provider == "anthropic":
        return call_claude(system, user, **kwargs)
    elif provider == "openai":
        return call_gpt(system, user, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Output parsing ───────────────────────────────────────────────────────────

def extract_grid_from_response(response: str) -> Optional[Grid]:
    """
    Extract grid from LLM response. Tries multiple extraction strategies:
    1. <output>...</output> tags
    2. <answer>...</answer> tags
    3. Last fenced code block
    4. Direct grid parsing of full response
    5. Last N lines that look like a grid
    """
    # Strategy 1 & 2: tagged output
    for tag in ["output", "answer"]:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if m:
            grid = parse_grid_from_text(m.group(1))
            if grid:
                return grid

    # Strategy 3: fenced code block
    blocks = re.findall(r"```(?:\w*)?\n([\s\S]*?)```", response)
    for block in reversed(blocks):
        grid = parse_grid_from_text(block)
        if grid:
            return grid

    # Strategy 4: look for contiguous grid-like lines
    lines = response.strip().splitlines()
    # Find runs of digit-only lines
    best_grid = None
    run_start = None
    current_run = []

    for i, line in enumerate(lines):
        clean = line.strip().replace(",", " ").replace("|", " ")
        tokens = clean.split()
        is_grid_line = (len(tokens) >= 1 and
                        all(t.strip(".,;:").isdigit() and 0 <= int(t.strip(".,;:")) <= 9
                            for t in tokens))
        if is_grid_line:
            current_run.append(clean)
        else:
            if len(current_run) >= 1:
                candidate = parse_grid_from_text("\n".join(current_run))
                if candidate and (best_grid is None or
                                  len(candidate) * len(candidate[0]) >
                                  len(best_grid) * len(best_grid[0])):
                    best_grid = candidate
            current_run = []

    if current_run:
        candidate = parse_grid_from_text("\n".join(current_run))
        if candidate and (best_grid is None or
                          len(candidate) * len(candidate[0]) >
                          len(best_grid) * len(best_grid[0])):
            best_grid = candidate

    return best_grid


# ── Ensemble voting ──────────────────────────────────────────────────────────

def grid_to_hashable(grid: Grid) -> Tuple:
    return tuple(tuple(r) for r in grid)


def ensemble_vote(grids: List[Optional[Grid]]) -> Optional[Grid]:
    """Return the most common valid grid among candidates."""
    valid = [g for g in grids if g is not None]
    if not valid:
        return None
    counts = Counter(grid_to_hashable(g) for g in valid)
    best_hash, _ = counts.most_common(1)[0]
    return [list(r) for r in best_hash]


def run_ensemble(prompts: List[Dict], provider: str = "anthropic",
                 verbose: bool = True) -> Dict[str, Any]:
    """
    Run multiple prompts and return ensemble result with metadata.
    prompts: list of {name, system, user, temperature}
    """
    responses = []
    grids = []

    for p in prompts:
        name = p.get("name", "?")
        try:
            if verbose:
                print(f"  → Running prompt [{name}]...")
            resp = call_llm(
                p["system"], p["user"],
                provider=provider,
                temperature=p.get("temperature", 0.0),
                max_tokens=p.get("max_tokens", 2048),
            )
            grid = extract_grid_from_response(resp)
            responses.append({"name": name, "response": resp, "grid": grid, "ok": grid is not None})
            grids.append(grid)
            if verbose:
                status = "✓" if grid else "✗"
                shape = f"{len(grid)}×{len(grid[0])}" if grid else "N/A"
                print(f"    {status} [{name}] → grid {shape}")
        except Exception as e:
            print(f"    ✗ [{name}] ERROR: {e}")
            responses.append({"name": name, "response": "", "grid": None, "ok": False})
            grids.append(None)

    winner = ensemble_vote(grids)
    valid_count = sum(1 for g in grids if g is not None)

    return {
        "winner": winner,
        "all_responses": responses,
        "valid_count": valid_count,
        "total": len(prompts),
        "agreement": _compute_agreement(grids),
    }


def _compute_agreement(grids: List[Optional[Grid]]) -> float:
    valid = [g for g in grids if g is not None]
    if len(valid) < 2:
        return 1.0 if valid else 0.0
    counts = Counter(grid_to_hashable(g) for g in valid)
    top_count = counts.most_common(1)[0][1]
    return top_count / len(valid)


# ── Iterative refinement ─────────────────────────────────────────────────────

def refine_with_verification(task, test_idx: int, initial_grid: Grid,
                              provider: str = "anthropic",
                              max_rounds: int = 2) -> Grid:
    """
    After getting an initial answer, ask the LLM to verify and self-correct.
    """
    from prompt_builder import build_verification_prompt, SYSTEM_PROMPT_COT

    current = initial_grid
    for round_i in range(max_rounds):
        verify_prompt = build_verification_prompt(task, test_idx, current)
        resp = call_llm(SYSTEM_PROMPT_COT, verify_prompt, provider=provider,
                        temperature=0.0)
        refined = extract_grid_from_response(resp)
        if refined and not grids_equal(refined, current):
            print(f"    ↻ Refinement round {round_i+1}: grid changed")
            current = refined
        else:
            print(f"    ✓ Verification round {round_i+1}: no change")
            break
    return current
