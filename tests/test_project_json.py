"""Tests for ``structural_analysis.gui_qt.project_io`` — .spa.json round-trip."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from structural_analysis.file_io import read_input_file
from structural_analysis.gui_qt.grid import GridLine, GridSystem
from structural_analysis.gui_qt.project_io import (
    Project,
    ViewState,
    load_project_json,
    save_project_json,
)
from structural_analysis.main import run_analysis


def _round_trip(project: Project) -> Project:
    fd, tmp = tempfile.mkstemp(suffix=".spa.json")
    os.close(fd)
    try:
        save_project_json(project, tmp)
        return load_project_json(tmp)
    finally:
        os.unlink(tmp)


def test_round_trip_q2a_with_grid():
    model = read_input_file("inputs/q2a_settlement.txt")
    grid = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 4.0),
                 GridLine("C", 12.0), GridLine("D", 15.0)],
        y_lines=[GridLine("1", 0.0), GridLine("2", 4.0)],
    )
    view = ViewState(xlim=(-1, 16), ylim=(-1, 5),
                     snap_kinds=["node", "grid", "midpoint"])
    p = Project(model=model, grid=grid, view=view, title="Q2(a) with grid")

    p2 = _round_trip(p)

    # Grid + view recovered
    assert [ln.label for ln in p2.grid.x_lines] == ["A", "B", "C", "D"]
    assert [ln.label for ln in p2.grid.y_lines] == ["1", "2"]
    assert p2.view.xlim == (-1.0, 16.0)
    assert p2.view.ylim == (-1.0, 5.0)
    assert "midpoint" in p2.view.snap_kinds

    # Model preserves analysis results
    r1 = run_analysis(p.model, verbose=False)
    r2 = run_analysis(p2.model, verbose=False)
    assert r1.status == r2.status == "ok"
    for eid in r1.member_results:
        f1 = np.array(r1.member_results[eid]["f_local"], dtype=float)
        f2 = np.array(r2.member_results[eid]["f_local"], dtype=float)
        np.testing.assert_allclose(f1, f2, atol=1e-9, rtol=1e-9)


def test_round_trip_q2b_thermal():
    model = read_input_file("inputs/q2b_thermal.txt")
    p = Project(model=model, grid=GridSystem(), view=ViewState(),
                title="q2b thermal")
    p2 = _round_trip(p)
    r1 = run_analysis(p.model, verbose=False)
    r2 = run_analysis(p2.model, verbose=False)
    assert r1.status == r2.status == "ok"
    for eid in r1.member_results:
        f1 = np.array(r1.member_results[eid]["f_local"], dtype=float)
        f2 = np.array(r2.member_results[eid]["f_local"], dtype=float)
        np.testing.assert_allclose(f1, f2, atol=1e-9, rtol=1e-9)


def test_loads_blank_grid_when_missing():
    """A project saved without a grid still loads cleanly."""
    model = read_input_file("inputs/q2a_settlement.txt")
    p = Project(model=model)
    p2 = _round_trip(p)
    assert p2.grid.is_empty()
