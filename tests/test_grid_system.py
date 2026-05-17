"""Tests for ``structural_analysis.gui_qt.grid.GridSystem``."""

from __future__ import annotations

import pytest

from structural_analysis.gui_qt.grid import GridLine, GridSystem


def test_intersections_count():
    g = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 4.0), GridLine("C", 8.0)],
        y_lines=[GridLine("1", 0.0), GridLine("2", 3.0)],
    )
    coords = list(g.intersections())
    assert len(coords) == 6
    # ("A","1") at (0, 0)
    xl, yl, xc, yc = coords[0]
    assert (xl.label, yl.label) == ("A", "1")
    assert (xc, yc) == (0.0, 0.0)


def test_from_spacing_alpha_numeric():
    g = GridSystem.from_spacing(
        x_count=3, x_spacing=4.0,
        y_count=2, y_spacing=3.0,
    )
    assert [ln.label for ln in g.x_lines] == ["A", "B", "C"]
    assert [ln.label for ln in g.y_lines] == ["1", "2"]
    assert [ln.coord for ln in g.x_lines] == [0.0, 4.0, 8.0]
    assert [ln.coord for ln in g.y_lines] == [0.0, 3.0]


def test_from_spacing_rejects_invalid():
    with pytest.raises(ValueError):
        GridSystem.from_spacing(x_count=-1, x_spacing=1.0,
                                  y_count=1, y_spacing=1.0)
    with pytest.raises(ValueError):
        GridSystem.from_spacing(x_count=1, x_spacing=0.0,
                                  y_count=1, y_spacing=1.0)


def test_round_trip_via_dict():
    g = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 4.5)],
        y_lines=[GridLine("1", 0.0)],
    )
    g2 = GridSystem.from_dict(g.to_dict())
    assert g2 == g


def test_alphabetic_label_overflow():
    g = GridSystem.from_spacing(x_count=28, x_spacing=1.0,
                                  y_count=0, y_spacing=1.0)
    # A..Z then AA, AB
    labels = [ln.label for ln in g.x_lines]
    assert labels[:3] == ["A", "B", "C"]
    assert labels[25] == "Z"
    assert labels[26] == "AA"
    assert labels[27] == "AB"


def test_is_empty():
    assert GridSystem().is_empty()
    assert not GridSystem(x_lines=[GridLine("A", 0.0)]).is_empty()
