"""Adaptive coordinate ticks — keep canvas axis numbers readable on zoom.

The bug: a fixed ``MultipleLocator(grid_spacing)`` emits one tick per
snap step regardless of zoom, so the coordinate numbers collide the
moment the user zooms out. ``AdaptiveGridLocator`` coarsens the step in a
1-2-5 progression so the on-screen label count is capped, and matplotlib
re-invokes it on every draw (including scroll-zoom, which only repaints).

Pure-logic assertions live up top; the Qt smoke checks the wiring.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.gui_qt.canvas import AdaptiveGridLocator


# ── pure locator logic ───────────────────────────────────────────────────


def _count(locator, vmin, vmax):
    return len(locator.tick_values(vmin, vmax))


def test_zoomed_in_uses_base_spacing():
    # View narrower than max_ticks · base ⇒ ticks land on the raw grid.
    loc = AdaptiveGridLocator(base=0.5, max_ticks=12)
    assert loc.step_for_span(3.0) == pytest.approx(0.5)
    ticks = loc.tick_values(0.0, 3.0)
    assert ticks[0] == pytest.approx(0.0)
    assert all(
        b - a == pytest.approx(0.5) for a, b in zip(ticks, ticks[1:])
    )


def test_never_finer_than_base():
    # Zooming in past one base step must not invent sub-grid coordinates.
    loc = AdaptiveGridLocator(base=0.5, max_ticks=12)
    assert loc.step_for_span(0.7) == pytest.approx(0.5)


def test_zoom_out_caps_label_count():
    loc = AdaptiveGridLocator(base=0.5, max_ticks=12)
    # max_ticks caps the number of *intervals*; an inclusive span adds one
    # boundary label, so the readable cap is max_ticks + 1.
    for span in (3.0, 6.0, 30.0, 120.0, 1000.0, 9999.0):
        assert _count(loc, 0.0, span) <= 13, f"too many ticks across {span}"


def test_step_follows_1_2_5_progression():
    loc = AdaptiveGridLocator(base=1.0, max_ticks=10)
    # span/step must drop to ≤10 with the smallest nice multiple.
    assert loc.step_for_span(10.0) == pytest.approx(1.0)    # 10 ticks
    assert loc.step_for_span(11.0) == pytest.approx(2.0)    # 1→2
    assert loc.step_for_span(21.0) == pytest.approx(5.0)    # 2→5
    assert loc.step_for_span(51.0) == pytest.approx(10.0)   # 5→10
    assert loc.step_for_span(101.0) == pytest.approx(20.0)  # 10→20


def test_ticks_are_round_multiples_of_step():
    loc = AdaptiveGridLocator(base=0.5, max_ticks=12)
    ticks = loc.tick_values(-7.3, 41.8)
    step = loc.step_for_span(41.8 - (-7.3))
    assert ticks, "expected some ticks across the span"
    for t in ticks:
        assert (t / step) == pytest.approx(round(t / step))
    assert min(ticks) >= -7.3 - step
    assert max(ticks) <= 41.8 + step


def test_reversed_interval_is_handled():
    loc = AdaptiveGridLocator(base=0.5, max_ticks=12)
    # Flipped axis (vmin > vmax) must still yield the same tick set.
    assert loc.tick_values(10.0, 0.0) == loc.tick_values(0.0, 10.0)


def test_degenerate_span_does_not_explode():
    loc = AdaptiveGridLocator(base=0.5, max_ticks=12)
    assert _count(loc, 5.0, 5.0) <= 2  # zero-width view, no blow-up


# ── Qt wiring smoke ──────────────────────────────────────────────────────


@pytest.fixture
def qt_app():
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PyQt6 unavailable: {exc}")
    return QApplication.instance() or QApplication([])


def _empty_canvas(qt_app):
    from structural_analysis.gui_qt.canvas import ModelCanvas
    from structural_analysis.model import StructuralModel
    m = StructuralModel(title="ticks")
    return ModelCanvas(None, model_provider=lambda: m)


def test_canvas_installs_adaptive_locator(qt_app):
    canvas = _empty_canvas(qt_app)
    canvas.redraw()
    assert isinstance(
        canvas.ax.xaxis.get_major_locator(), AdaptiveGridLocator
    )
    assert isinstance(
        canvas.ax.yaxis.get_major_locator(), AdaptiveGridLocator
    )


def test_zoom_out_keeps_axis_labels_bounded(qt_app):
    canvas = _empty_canvas(qt_app)
    canvas.redraw()
    # Simulate a hard zoom-out the way scroll-zoom would, then ask the
    # locator (as matplotlib does) for the ticks it would label.
    canvas.ax.set_xlim(-500.0, 500.0)
    loc = canvas.ax.xaxis.get_major_locator()
    assert len(loc()) <= canvas.MAX_AXIS_LABELS + 1
