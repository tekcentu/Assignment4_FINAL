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


# ── plain absolute coordinate labels (no '+1e3' offset notation) ─────────


def _both_axis_formatters(canvas):
    return (canvas.ax.xaxis.get_major_formatter(),
            canvas.ax.yaxis.get_major_formatter())


def test_axis_labels_have_no_offset_notation_after_redraw(qt_app):
    canvas = _empty_canvas(qt_app)
    canvas.redraw()
    for fmt in _both_axis_formatters(canvas):
        # ScalarFormatter only adds a "+1e3" corner label when useOffset
        # is True; disabling it is exactly what keeps the spine readable.
        assert fmt.get_useOffset() is False


def test_axis_labels_have_no_offset_notation_after_zoom(qt_app):
    canvas = _empty_canvas(qt_app)
    canvas.redraw()
    # Pan the view far from the origin — this is where matplotlib would
    # normally engage offset notation and ruin the spine readout.
    canvas.ax.set_xlim(1000.0, 1050.0)
    canvas.ax.set_ylim(2000.0, 2050.0)
    canvas.redraw()
    for fmt in _both_axis_formatters(canvas):
        assert fmt.get_useOffset() is False
    # And the tick labels themselves should be plain absolute numbers,
    # not a "+1e3"-style offset breakdown.
    canvas.fig.canvas.draw()  # force a draw so tick labels populate
    for axis in (canvas.ax.xaxis, canvas.ax.yaxis):
        offset_text = axis.get_offset_text().get_text()
        assert "e" not in offset_text.lower(), (
            f"unexpected scientific offset on spine: {offset_text!r}"
        )


# ── hover/snap repaint must not call full canvas.redraw() ────────────────


def _model_for_hover():
    from structural_analysis.model import StructuralModel, Node, Support, NodalLoad
    from structural_analysis.element import FrameElement2D
    m = StructuralModel(title="hover")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 5.0, 0.0)}
    m.elements = [FrameElement2D(1, 1, 2, E=200e3, A=0.02, I=0.08)]
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    m.nodal_loads = [NodalLoad(node_id=2, fy=-1.0)]
    return m


def test_repaint_hover_marker_does_not_full_redraw(qt_app):
    """Hover-only repaints must update the persistent marker artist
    and call draw_idle — never go through the full redraw() rebuild."""
    from structural_analysis.gui_qt.canvas import ModelCanvas
    m = _model_for_hover()
    canvas = ModelCanvas(None, model_provider=lambda: m)
    canvas.redraw()
    # Count: redraw must NOT be called by the hover path; draw_idle must.
    redraw_calls = {"n": 0}
    idle_calls = {"n": 0}
    real_redraw = canvas.redraw
    real_idle = canvas._mpl_canvas.draw_idle
    canvas.redraw = lambda: (redraw_calls.__setitem__("n", redraw_calls["n"] + 1),  # noqa: E731
                             real_redraw())[1]
    canvas._mpl_canvas.draw_idle = lambda: (idle_calls.__setitem__("n", idle_calls["n"] + 1),  # noqa: E731
                                            real_idle())[1]
    try:
        canvas._hover_xy = (1.0, 2.0)
        for _ in range(10):
            canvas.repaint_hover_marker()
    finally:
        canvas.redraw = real_redraw
        canvas._mpl_canvas.draw_idle = real_idle
    assert redraw_calls["n"] == 0, (
        "hover path must not rebuild the scene"
    )
    assert idle_calls["n"] == 10


def test_hover_marker_artist_is_persistent_across_repaints(qt_app):
    """The hover-marker Line2D must be re-used (set_data) across
    repeated hover paints — not re-created each time, which would
    silently leak artists and bloat ax.lines."""
    from structural_analysis.gui_qt.canvas import ModelCanvas
    m = _model_for_hover()
    canvas = ModelCanvas(None, model_provider=lambda: m)
    canvas.redraw()
    canvas._hover_xy = (1.0, 2.0)
    canvas.repaint_hover_marker()
    first = canvas._hover_marker_artist
    assert first is not None
    n_lines_before = len(canvas.ax.lines)
    for x in (1.5, 2.0, 2.5, 3.0, 3.5):
        canvas._hover_xy = (x, 2.0)
        canvas.repaint_hover_marker()
    # Same artist object, with updated x-data — no new lines added.
    assert canvas._hover_marker_artist is first
    assert len(canvas.ax.lines) == n_lines_before
    xd = list(canvas._hover_marker_artist.get_xdata())
    assert xd[0] == pytest.approx(3.5)


def test_main_window_motion_uses_repaint_hover_not_full_redraw(qt_app):
    """End-to-end: a MainWindow motion event must touch the marker
    artist and NOT trigger a full canvas.redraw()."""
    from structural_analysis.gui_qt.app import MainWindow
    from structural_analysis.gui_qt.canvas import HitResult
    w = MainWindow()
    w._model = _model_for_hover()
    w.canvas.redraw()
    n = {"redraw": 0, "repaint": 0}
    real_redraw = w.canvas.redraw
    real_repaint = w.canvas.repaint_hover_marker
    w.canvas.redraw = lambda: (n.__setitem__("redraw", n["redraw"] + 1),  # noqa: E731
                               real_redraw())[1]
    w.canvas.repaint_hover_marker = lambda: (n.__setitem__("repaint", n["repaint"] + 1),  # noqa: E731
                                             real_repaint())[1]
    try:
        for x in (0.5, 1.0, 1.5, 2.0):
            hit = HitResult(x=x, y=0.0)
            w._on_canvas_motion(hit, cursor_px=(100.0, 100.0))
    finally:
        w.canvas.redraw = real_redraw
        w.canvas.repaint_hover_marker = real_repaint
    assert n["redraw"] == 0, "motion must not trigger full scene rebuild"
    assert n["repaint"] == 4
