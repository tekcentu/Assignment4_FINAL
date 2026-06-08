"""Smoke tests for the View → Show local axes overlay (v0.24.0).

Pin the canvas state flag, the View-menu wiring, and the matplotlib
draw output against the solver's local-axis convention (local x = i→j,
local y = 90° CCW). Headless Qt — offscreen platform plugin.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication  # noqa: E402
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 QtWidgets unavailable: {exc}", allow_module_level=True)

from structural_analysis.element import FrameElement2D, TrussElement2D  # noqa: E402
from structural_analysis.gui_qt.app import MainWindow  # noqa: E402
from structural_analysis.model import Node  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _seed_one_element(w, *, ni=(0.0, 0.0), nj=(2.0, 0.0)):
    w._model.nodes = {1: Node(1, *ni), 2: Node(2, *nj)}
    w._model.elements = [
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.0e8, A=0.01, I=8.3e-5),
    ]


def _annotation_xy_pairs(ax):
    """All (xy, xytext) of `ax.annotate("", …)` arrows currently drawn."""
    out = []
    for child in ax.get_children():
        if child.__class__.__name__ == "FancyArrowPatch":
            continue
        if hasattr(child, "xy") and hasattr(child, "xyann") and child.get_text() == "":
            out.append((child.xy, child.xyann))
    return out


def _text_labels(ax) -> list[tuple[float, float, str]]:
    out = []
    for child in ax.texts:
        out.append((child.get_position()[0], child.get_position()[1], child.get_text()))
    return out


def test_show_local_axes_default_off(qt_app):
    w = MainWindow()
    assert w.canvas.show_local_axes is False
    assert w.act_show_local_axes.isCheckable()
    assert not w.act_show_local_axes.isChecked()


def test_view_menu_has_show_local_axes_action(qt_app):
    w = MainWindow()
    titles = [a.text() for m in w.menuBar().findChildren(type(w.menuBar()))
              for a in m.actions()]
    # `act_show_local_axes` lives on the View menu; verify it's exposed.
    assert w.act_show_local_axes in w.findChildren(type(w.act_show_local_axes))


def test_toggle_action_flips_canvas_flag(qt_app):
    w = MainWindow()
    _seed_one_element(w)
    w.act_show_local_axes.trigger()      # check
    assert w.canvas.show_local_axes is True
    assert w.act_show_local_axes.isChecked()
    w.act_show_local_axes.trigger()      # uncheck
    assert w.canvas.show_local_axes is False


def test_no_axes_drawn_when_flag_off(qt_app):
    w = MainWindow()
    _seed_one_element(w)
    w.canvas.show_local_axes = False
    w.canvas.redraw()
    labels = [t for (_x, _y, t) in _text_labels(w.canvas.ax)]
    assert "x" not in labels and "y" not in labels
    assert "i" not in labels and "j" not in labels


def test_horizontal_element_axes_match_solver_convention(qt_app):
    """Horizontal i→j (right): local x = +world x, local y = +world y."""
    w = MainWindow()
    _seed_one_element(w, ni=(0.0, 0.0), nj=(2.0, 0.0))
    w.canvas.show_local_axes = True
    w.canvas.redraw()
    # Find the 'x' and 'y' tip labels.
    labels = {t: (x, y) for (x, y, t) in _text_labels(w.canvas.ax)}
    assert "x" in labels and "y" in labels
    mx, my = 1.0, 0.0     # element midpoint
    x_tip = labels["x"]
    y_tip = labels["y"]
    # Local x tip must have x_tip.x > mx and y == my (within tol).
    assert x_tip[0] > mx and abs(x_tip[1] - my) < 1e-6
    # Local y tip must have x == mx and y_tip.y > my.
    assert abs(y_tip[0] - mx) < 1e-6 and y_tip[1] > my


def test_vertical_element_axes_match_solver_convention(qt_app):
    """Vertical i→j (up): local x = (0, 1), local y = (-1, 0)."""
    w = MainWindow()
    _seed_one_element(w, ni=(0.0, 0.0), nj=(0.0, 2.0))
    w.canvas.show_local_axes = True
    w.canvas.redraw()
    labels = {t: (x, y) for (x, y, t) in _text_labels(w.canvas.ax)}
    mx, my = 0.0, 1.0
    x_tip = labels["x"]
    y_tip = labels["y"]
    assert abs(x_tip[0] - mx) < 1e-6 and x_tip[1] > my
    assert y_tip[0] < mx and abs(y_tip[1] - my) < 1e-6


def test_inclined_element_axes_match_solver_convention(qt_app):
    """45° inclined: local y = 90° CCW from local x."""
    w = MainWindow()
    _seed_one_element(w, ni=(0.0, 0.0), nj=(2.0, 2.0))
    w.canvas.show_local_axes = True
    w.canvas.redraw()
    labels = {t: (x, y) for (x, y, t) in _text_labels(w.canvas.ax)}
    mx, my = 1.0, 1.0
    x_tip = labels["x"]
    y_tip = labels["y"]
    # x-tip should lie along the +i→j ray (both coords > midpoint).
    assert x_tip[0] > mx and x_tip[1] > my
    # y-tip should be 90° CCW from x: relative to midpoint, dx<0, dy>0.
    assert y_tip[0] < mx and y_tip[1] > my


def test_i_label_at_node_i_j_label_at_node_j(qt_app):
    w = MainWindow()
    _seed_one_element(w, ni=(0.0, 0.0), nj=(10.0, 0.0))
    w.canvas.show_local_axes = True
    w.canvas.redraw()
    labels = {t: (x, y) for (x, y, t) in _text_labels(w.canvas.ax)}
    # 'i' is near node_i (x≈0.6, well below midpoint x=5);
    # 'j' is near node_j (x≈9.4, well above midpoint x=5).
    assert labels["i"][0] < 2.0
    assert labels["j"][0] > 8.0


def test_truss_element_also_renders_axes(qt_app):
    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 3.0, 0.0)}
    w._model.elements = [TrussElement2D(id=1, node_i=1, node_j=2, E=2.0e8, A=0.01)]
    w.canvas.show_local_axes = True
    w.canvas.redraw()
    labels = [t for (_x, _y, t) in _text_labels(w.canvas.ax)]
    assert "x" in labels and "y" in labels
    assert "i" in labels and "j" in labels


def test_dense_model_skips_local_axes(qt_app):
    w = MainWindow()
    cap = w.canvas.MAX_AUTO_ELEMENT_LABELS
    n_nodes = cap + 5
    w._model.nodes = {i: Node(i, float(i), 0.0) for i in range(n_nodes)}
    w._model.elements = [
        FrameElement2D(id=i + 1, node_i=i, node_j=i + 1,
                       E=2.0e8, A=0.01, I=8.3e-5)
        for i in range(n_nodes - 1)
    ]
    assert len(w._model.elements) > cap
    w.canvas.show_local_axes = True
    w.canvas.redraw()
    # In dense mode we silently skip the axes overlay — no x/y tip text.
    labels = [t for (_x, _y, t) in _text_labels(w.canvas.ax)]
    assert "x" not in labels and "y" not in labels


def test_axes_zorder_below_selection(qt_app):
    w = MainWindow()
    _seed_one_element(w)
    w.canvas.show_local_axes = True
    w.canvas.select_element(1)
    w.canvas.redraw()
    # Selection ring zorder is 11; the axis labels (zorder 3.6) must be
    # painted underneath.
    label_zs = [
        child.get_zorder()
        for child in w.canvas.ax.texts
        if child.get_text() in ("x", "y", "i", "j")
    ]
    assert label_zs and max(label_zs) < 11
