"""
ARC-AGI Core Types & Utilities
Handles grid representation, color mapping, and task I/O for ARC-AGI-2 and ARC-AGI-3.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import json, copy, numpy as np

# ARC canonical color palette (index → name)
COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "grey", 6: "fuschia", 7: "orange", 8: "azure", 9: "maroon"
}
COLOR_ABBREV = {v: k for k, v in COLOR_NAMES.items()}

Grid = List[List[int]]


@dataclass
class ARCExample:
    input: Grid
    output: Grid

    def input_np(self) -> np.ndarray:
        return np.array(self.input, dtype=np.int8)

    def output_np(self) -> np.ndarray:
        return np.array(self.output, dtype=np.int8)

    def input_shape(self) -> Tuple[int, int]:
        return len(self.input), len(self.input[0]) if self.input else 0

    def output_shape(self) -> Tuple[int, int]:
        return len(self.output), len(self.output[0]) if self.output else 0


@dataclass
class ARCTask:
    task_id: str
    train: List[ARCExample]
    test: List[ARCExample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, task_id: str, data: dict) -> "ARCTask":
        train = [ARCExample(e["input"], e["output"]) for e in data["train"]]
        test_examples = []
        for e in data["test"]:
            out = e.get("output", [[0]])  # output may be hidden
            test_examples.append(ARCExample(e["input"], out))
        return cls(task_id=task_id, train=train, test=test_examples)

    @classmethod
    def from_file(cls, path: str) -> "ARCTask":
        with open(path) as f:
            data = json.load(f)
        task_id = path.split("/")[-1].replace(".json", "")
        return cls.from_dict(task_id, data)

    def num_colors_used(self) -> set:
        colors = set()
        for ex in self.train:
            for row in ex.input:
                colors.update(row)
            for row in ex.output:
                colors.update(row)
        return colors

    def input_size_consistent(self) -> bool:
        shapes = [ex.input_shape() for ex in self.train]
        return len(set(shapes)) == 1

    def output_size_consistent(self) -> bool:
        shapes = [ex.output_shape() for ex in self.train]
        return len(set(shapes)) == 1

    def same_shape_io(self) -> bool:
        return all(ex.input_shape() == ex.output_shape() for ex in self.train)


# ── Grid rendering ──────────────────────────────────────────────────────────

BLOCK = "█"
CELL_W = 2  # chars per cell

def grid_to_text(grid: Grid, use_color_names: bool = False) -> str:
    """Convert grid to a compact ASCII string for LLM prompts."""
    if use_color_names:
        rows = []
        for row in grid:
            rows.append(" ".join(COLOR_NAMES[c] for c in row))
        return "\n".join(rows)
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def grid_to_compact(grid: Grid) -> str:
    """Single-line compact encoding: rows separated by |"""
    return " | ".join(" ".join(str(c) for c in row) for row in grid)


def parse_grid_from_text(text: str) -> Optional[Grid]:
    """Parse LLM output back into a grid. Handles various formats."""
    text = text.strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    grid = []
    for line in lines:
        # strip markdown fences, pipes, brackets
        line = line.strip("|").strip()
        line = line.replace("[", "").replace("]", "").replace(",", " ")
        tokens = line.split()
        row = []
        for t in tokens:
            t = t.strip(".,;:")
            if t.isdigit() and 0 <= int(t) <= 9:
                row.append(int(t))
            elif t.lower() in COLOR_ABBREV:
                row.append(COLOR_ABBREV[t.lower()])
        if row:
            grid.append(row)
    # validate rectangular
    if not grid:
        return None
    width = len(grid[0])
    if all(len(r) == width for r in grid):
        return grid
    return None


def grids_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    return all(ra == rb for ra, rb in zip(a, b))


def task_summary(task: ARCTask) -> str:
    lines = [f"Task: {task.task_id}"]
    lines.append(f"Train pairs: {len(task.train)}, Test pairs: {len(task.test)}")
    lines.append(f"Colors used: {sorted(task.num_colors_used())}")
    for i, ex in enumerate(task.train):
        lines.append(f"  Train[{i}]: input{ex.input_shape()} → output{ex.output_shape()}")
    for i, ex in enumerate(task.test):
        lines.append(f"  Test[{i}]: input{ex.input_shape()}")
    return "\n".join(lines)
