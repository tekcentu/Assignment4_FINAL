"""Labeled grid system — SAP2000-style named X and Y grid lines.

Stored as GUI-side metadata; not part of ``StructuralModel``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class GridLine:
    """A single labeled grid line (one axis).

    Attributes:
        label: Display label (e.g. "A", "B", "1", "2").
        coord: World coordinate along the axis (m).
    """

    label: str
    coord: float


@dataclass(frozen=True)
class GridSystem:
    """Collection of X and Y grid lines.

    The list order doesn't have to be sorted; iterators yield in coordinate
    order. Empty axes are allowed (means "no grid in that direction").
    """

    x_lines: list[GridLine] = field(default_factory=list)
    y_lines: list[GridLine] = field(default_factory=list)

    @classmethod
    def from_spacing(cls, *, x_count: int, x_spacing: float,
                      y_count: int, y_spacing: float,
                      x_origin: float = 0.0, y_origin: float = 0.0,
                      x_label_kind: str = "alpha",
                      y_label_kind: str = "numeric") -> "GridSystem":
        """Convenience: build a regular grid from counts and spacings.

        ``x_label_kind`` and ``y_label_kind`` are "alpha" (A, B, C, ...) or
        "numeric" (1, 2, 3, ...).
        """
        if x_count < 0 or y_count < 0:
            raise ValueError("Grid counts must be non-negative.")
        if x_spacing <= 0 or y_spacing <= 0:
            raise ValueError("Grid spacings must be positive.")
        x_lines = [
            GridLine(_label(i, x_label_kind), x_origin + i * x_spacing)
            for i in range(x_count)
        ]
        y_lines = [
            GridLine(_label(i, y_label_kind), y_origin + i * y_spacing)
            for i in range(y_count)
        ]
        return cls(x_lines=x_lines, y_lines=y_lines)

    def intersections(self) -> Iterator[tuple[GridLine, GridLine, float, float]]:
        """Yield (x_line, y_line, x_coord, y_coord) for every intersection."""
        for xl in self.x_lines:
            for yl in self.y_lines:
                yield xl, yl, xl.coord, yl.coord

    def is_empty(self) -> bool:
        return not self.x_lines and not self.y_lines

    def to_dict(self) -> dict:
        return {
            "x": [{"label": ln.label, "coord": ln.coord} for ln in self.x_lines],
            "y": [{"label": ln.label, "coord": ln.coord} for ln in self.y_lines],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridSystem":
        return cls(
            x_lines=[GridLine(label=d["label"], coord=float(d["coord"]))
                     for d in data.get("x", [])],
            y_lines=[GridLine(label=d["label"], coord=float(d["coord"]))
                     for d in data.get("y", [])],
        )


def _label(i: int, kind: str) -> str:
    if kind == "numeric":
        return str(i + 1)
    # alpha: A, B, ..., Z, AA, AB, ...
    s = ""
    n = i + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        s = chr(ord("A") + rem) + s
    return s
