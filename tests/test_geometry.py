"""Tests for the 2-D geometry helpers in gui_common/geometry.py.

The helpers are pure-float, no Qt — fast unit tests.
"""

from __future__ import annotations

import pytest

from structural_analysis.gui_common.geometry import project_point_on_segment


def test_project_midpoint_of_axis_aligned_segment():
    """Cursor directly above the midpoint of a horizontal segment
    projects to the midpoint with t=0.5."""
    px, py, t = project_point_on_segment(
        5.0, 3.0,     # cursor
        0.0, 0.0,     # A
        10.0, 0.0,    # B
    )
    assert (px, py) == pytest.approx((5.0, 0.0))
    assert t == pytest.approx(0.5)


def test_project_onto_tilted_segment():
    """45° segment: cursor perpendicular to midpoint projects there."""
    # Segment from (0,0) to (4,4); cursor at (3, 1) — perpendicular
    # foot is at the midpoint of the segment (2, 2).
    px, py, t = project_point_on_segment(
        3.0, 1.0,
        0.0, 0.0,
        4.0, 4.0,
    )
    assert (px, py) == pytest.approx((2.0, 2.0))
    assert t == pytest.approx(0.5)


def test_project_off_the_start_returns_negative_t():
    """Cursor beyond A returns t < 0 so the caller can detect it."""
    _, _, t = project_point_on_segment(
        -3.0, 0.0,
        0.0, 0.0,
        10.0, 0.0,
    )
    assert t < 0.0


def test_project_off_the_end_returns_t_greater_than_one():
    _, _, t = project_point_on_segment(
        15.0, 0.0,
        0.0, 0.0,
        10.0, 0.0,
    )
    assert t > 1.0


def test_degenerate_zero_length_segment_returns_a_point_and_zero():
    """When A == B the projection is degenerate. The helper must
    return A's coordinates and t = 0 — no division by zero."""
    px, py, t = project_point_on_segment(
        5.0, 5.0,
        2.0, 3.0,
        2.0, 3.0,
    )
    assert (px, py) == (2.0, 3.0)
    assert t == 0.0
