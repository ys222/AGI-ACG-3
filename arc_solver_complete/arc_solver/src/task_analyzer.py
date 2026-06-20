"""
Task Analyzer — extracts rich structural features from ARC tasks.
These features feed into prompt construction and strategy selection.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Set
import numpy as np
from collections import Counter
from arc_types import ARCTask, ARCExample, Grid, COLOR_NAMES


# ── Object detection (connected components) ─────────────────────────────────

def get_objects(grid: Grid, background: int = 0, diagonal: bool = False) -> List[Dict]:
    """
    Find all contiguous objects in a grid.
    Returns list of {color, cells, bbox, size}.
    """
    arr = np.array(grid)
    visited = np.zeros_like(arr, dtype=bool)
    H, W = arr.shape
    objects = []

    def neighbors(r, c):
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        if diagonal:
            dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
        return [(r+dr, c+dc) for dr,dc in dirs if 0<=r+dr<H and 0<=c+dc<W]

    for r in range(H):
        for c in range(W):
            if visited[r][c] or arr[r][c] == background:
                continue
            color = int(arr[r][c])
            stack = [(r, c)]
            cells = []
            while stack:
                cr, cc = stack.pop()
                if visited[cr][cc]:
                    continue
                visited[cr][cc] = True
                cells.append((cr, cc))
                for nr, nc in neighbors(cr, cc):
                    if not visited[nr][nc] and arr[nr][nc] == color:
                        stack.append((nr, nc))
            rows = [c[0] for c in cells]
            cols = [c[1] for c in cells]
            objects.append({
                "color": color,
                "color_name": COLOR_NAMES[color],
                "cells": sorted(cells),
                "size": len(cells),
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "height": max(rows)-min(rows)+1,
                "width": max(cols)-min(cols)+1,
            })
    return objects


def detect_background(grid: Grid) -> int:
    """Most frequent color = background."""
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]


# ── Symmetry detection ───────────────────────────────────────────────────────

def has_horizontal_symmetry(grid: Grid) -> bool:
    arr = np.array(grid)
    return np.array_equal(arr, arr[::-1])

def has_vertical_symmetry(grid: Grid) -> bool:
    arr = np.array(grid)
    return np.array_equal(arr, arr[:, ::-1])

def has_rotational_symmetry(grid: Grid, k: int = 2) -> bool:
    arr = np.array(grid)
    return np.array_equal(arr, np.rot90(arr, k))


# ── Transformation inference ─────────────────────────────────────────────────

def infer_color_mapping(ex: ARCExample) -> Dict[int, int]:
    """If transformation is a pure color remap, return the mapping."""
    inp = np.array(ex.input).flatten()
    out = np.array(ex.output).flatten()
    if len(inp) != len(out):
        return {}
    mapping = {}
    for i, o in zip(inp, out):
        i, o = int(i), int(o)
        if i in mapping and mapping[i] != o:
            return {}
        mapping[i] = o
    return mapping


def shapes_preserved(ex: ARCExample) -> bool:
    """Check if input/output have the same shape."""
    return (len(ex.input) == len(ex.output) and
            len(ex.input[0]) == len(ex.output[0]))


def detect_tiling(ex: ARCExample) -> bool:
    """Detect if output is a tiled/repeated version of input."""
    inp = np.array(ex.input)
    out = np.array(ex.output)
    H_i, W_i = inp.shape
    H_o, W_o = out.shape
    if H_o % H_i == 0 and W_o % W_i == 0:
        reps_r = H_o // H_i
        reps_c = W_o // W_i
        tiled = np.tile(inp, (reps_r, reps_c))
        return np.array_equal(tiled, out)
    return False


# ── High-level task feature extraction ──────────────────────────────────────

def analyze_task(task: ARCTask) -> Dict[str, Any]:
    features: Dict[str, Any] = {
        "task_id": task.task_id,
        "n_train": len(task.train),
        "n_test": len(task.test),
        "colors_used": sorted(task.num_colors_used()),
        "n_colors": len(task.num_colors_used()),
        "same_shape_io": task.same_shape_io(),
        "consistent_input_size": task.input_size_consistent(),
        "consistent_output_size": task.output_size_consistent(),
    }

    # Per-example analysis
    ex_features = []
    for ex in task.train:
        inp_bg = detect_background(ex.input)
        out_bg = detect_background(ex.output)
        inp_objs = get_objects(ex.input, inp_bg)
        out_objs = get_objects(ex.output, out_bg)
        ef = {
            "input_shape": ex.input_shape(),
            "output_shape": ex.output_shape(),
            "input_bg": inp_bg,
            "output_bg": out_bg,
            "n_input_objects": len(inp_objs),
            "n_output_objects": len(out_objs),
            "h_sym_input": has_horizontal_symmetry(ex.input),
            "v_sym_input": has_vertical_symmetry(ex.input),
            "h_sym_output": has_horizontal_symmetry(ex.output),
            "v_sym_output": has_vertical_symmetry(ex.output),
            "color_remap": infer_color_mapping(ex),
            "is_tiling": detect_tiling(ex),
            "shape_preserved": shapes_preserved(ex),
        }
        ex_features.append(ef)

    features["examples"] = ex_features

    # Aggregate patterns
    features["all_shape_preserved"] = all(e["shape_preserved"] for e in ex_features)
    features["any_tiling"] = any(e["is_tiling"] for e in ex_features)
    features["has_color_remap"] = all(bool(e["color_remap"]) for e in ex_features)
    features["output_always_h_sym"] = all(e["h_sym_output"] for e in ex_features)
    features["output_always_v_sym"] = all(e["v_sym_output"] for e in ex_features)

    # Strategy hint
    features["strategy_hints"] = _get_strategy_hints(features)
    return features


def _get_strategy_hints(f: Dict) -> List[str]:
    hints = []
    if f["has_color_remap"]:
        hints.append("color_substitution")
    if f["any_tiling"]:
        hints.append("tiling_repetition")
    if f["all_shape_preserved"]:
        hints.append("in_place_transformation")
    if f["output_always_h_sym"] or f["output_always_v_sym"]:
        hints.append("symmetry_completion")
    if f["n_colors"] <= 3:
        hints.append("few_colors_simple_rule")
    if not hints:
        hints.append("complex_spatial_reasoning")
    return hints


def features_to_text(features: Dict) -> str:
    """Convert features to natural language for LLM context."""
    lines = []
    lines.append(f"Task ID: {features['task_id']}")
    lines.append(f"Training examples: {features['n_train']}")
    lines.append(f"Colors used: {[COLOR_NAMES[c] for c in features['colors_used']]}")
    lines.append(f"Input/output same shape: {features['same_shape_io']}")
    lines.append(f"Detected strategy hints: {features['strategy_hints']}")

    for i, ex in enumerate(features["examples"]):
        lines.append(f"\nExample {i+1}:")
        lines.append(f"  Input shape: {ex['input_shape']}, Output shape: {ex['output_shape']}")
        lines.append(f"  Input objects: {ex['n_input_objects']}, Output objects: {ex['n_output_objects']}")
        if ex["color_remap"]:
            remap_str = ", ".join(f"{COLOR_NAMES[k]}→{COLOR_NAMES[v]}"
                                  for k, v in ex["color_remap"].items() if k != v)
            if remap_str:
                lines.append(f"  Color remapping: {remap_str}")
        if ex["is_tiling"]:
            lines.append(f"  ✓ Output is a tiled version of input")
        if ex["h_sym_output"]:
            lines.append(f"  ✓ Output has horizontal symmetry")
        if ex["v_sym_output"]:
            lines.append(f"  ✓ Output has vertical symmetry")

    return "\n".join(lines)
