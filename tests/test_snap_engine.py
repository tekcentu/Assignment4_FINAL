"""Tests for ``structural_analysis.gui_qt.snap.SnapEngine``.

The snap engine is framework-agnostic — these tests don't import Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from structural_analysis.gui_qt.snap import SnapEngine
from structural_analysis.gui_qt.grid import GridLine, GridSystem


# Tiny stub model so we don't need full StructuralModel construction.
@dataclass
class _Node:
    id: int
    x: float
    y: float


@dataclass
class _Elem:
    id: int
    node_i: int
    node_j: int


@dataclass
class _Model:
    nodes: dict
    elements: list


PX_PER_M = 100.0  # 1 m = 100 px (rough zoom level for the tests)


def _model_with(nodes: List[_Node], elements: List[_Elem]) -> _Model:
    return _Model(nodes={n.id: n for n in nodes}, elements=list(elements))


def test_returns_none_when_nothing_in_range():
    eng = SnapEngine(tolerance_px=5.0)
    m = _model_with([_Node(1, 0, 0)], [])
    out = eng.find_snap(
        cursor_x=10.0, cursor_y=10.0,
        px_per_dx=PX_PER_M, px_per_dy=PX_PER_M,
        model=m,
    )
    assert out is None


def test_node_beats_grid_within_tolerance():
    eng = SnapEngine(tolerance_px=10.0)
    m = _model_with([_Node(1, 0.0, 0.0)], [])
    grid = GridSystem(x_lines=[GridLine("A", 0.0)],
                      y_lines=[GridLine("1", 0.0)])
    # Cursor near (0, 0) — both grid intersection and existing node coincide.
    out = eng.find_snap(
        cursor_x=0.02, cursor_y=0.02,
        px_per_dx=PX_PER_M, px_per_dy=PX_PER_M,
        model=m, grid=grid,
    )
    assert out is not None
    assert out.kind == "node"
    assert out.x == 0.0 and out.y == 0.0


def test_midpoint_of_element():
    eng = SnapEngine(tolerance_px=10.0)
    m = _model_with(
        [_Node(1, 0.0, 0.0), _Node(2, 4.0, 0.0)],
        [_Elem(1, 1, 2)],
    )
    # Cursor near the midpoint (2.0, 0.0)
    out = eng.find_snap(
        cursor_x=2.02, cursor_y=0.01,
        px_per_dx=PX_PER_M, px_per_dy=PX_PER_M,
        model=m,
    )
    assert out is not None
    assert out.kind == "midpoint"
    assert out.x == 2.0 and out.y == 0.0
    assert out.object_id == 1


def test_endpoint_beats_midpoint_when_both_in_range():
    eng = SnapEngine(tolerance_px=20.0)
    m = _model_with(
        [_Node(1, 0.0, 0.0), _Node(2, 0.1, 0.0)],   # very short element
        [_Elem(1, 1, 2)],
    )
    # Cursor right on the start endpoint of the very short element
    out = eng.find_snap(
        cursor_x=0.0, cursor_y=0.0,
        px_per_dx=PX_PER_M, px_per_dy=PX_PER_M,
        model=m,
    )
    # node 1 is also at (0,0); the node priority (0) beats endpoint (2).
    assert out is not None
    assert out.kind == "node"
    assert out.object_id == 1


def test_projection_on_element():
    eng = SnapEngine(tolerance_px=10.0)
    m = _model_with(
        [_Node(1, 0.0, 0.0), _Node(2, 10.0, 0.0)],
        [_Elem(1, 1, 2)],
    )
    # Cursor slightly above the element midpoint — should project onto x=3
    out = eng.find_snap(
        cursor_x=3.0, cursor_y=0.05,
        px_per_dx=PX_PER_M, px_per_dy=PX_PER_M,
        model=m,
    )
    assert out is not None
    # Could be project or midpoint depending on tolerance — at x=3 it's project.
    assert out.kind == "project"
    assert abs(out.x - 3.0) < 1e-9
    assert abs(out.y - 0.0) < 1e-9


def test_disabled_kinds_filter():
    eng = SnapEngine(tolerance_px=10.0,
                     enabled_kinds={"node"})  # midpoint disabled
    m = _model_with(
        [_Node(1, 0.0, 0.0), _Node(2, 4.0, 0.0)],
        [_Elem(1, 1, 2)],
    )
    out = eng.find_snap(
        cursor_x=2.0, cursor_y=0.0,
        px_per_dx=PX_PER_M, px_per_dy=PX_PER_M,
        model=m,
    )
    assert out is None  # midpoint suppressed; no node near (2,0)


def test_pixel_tolerance_scales_with_zoom():
    eng = SnapEngine(tolerance_px=10.0)
    m = _model_with([_Node(1, 0.0, 0.0)], [])
    # Zoomed in: 1000 px/m ⇒ 0.05 m offset = 50 px ⇒ outside tolerance
    out_close = eng.find_snap(
        cursor_x=0.05, cursor_y=0.0,
        px_per_dx=1000.0, px_per_dy=1000.0,
        model=m,
    )
    assert out_close is None
    # Zoomed out: 10 px/m ⇒ 0.05 m offset = 0.5 px ⇒ inside tolerance
    out_far = eng.find_snap(
        cursor_x=0.05, cursor_y=0.0,
        px_per_dx=10.0, px_per_dy=10.0,
        model=m,
    )
    assert out_far is not None
    assert out_far.kind == "node"
