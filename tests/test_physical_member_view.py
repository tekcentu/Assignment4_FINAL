"""Tests for PR #49 — Section-Aware Physical Member View (v0.30.3).

Pure-Python tests cover the geometry helpers without any Qt dependency.
Qt offscreen smoke tests follow the fixture-scope skip pattern introduced
in PR #48: PyQt6 is only imported inside the qt_app fixture so the
pure-Python tests collect and run in environments where PyQt6 is absent.
"""

from __future__ import annotations

import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ── Pure-Python geometry helpers ──────────────────────────────────────────────


class _FakeNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_physical_member_polygon_horizontal():
    """Horizontal member (0,0)→(4,0), depth=0.4 → rectangle ±0.2 in y."""
    from structural_analysis.gui_qt.canvas import ModelCanvas
    poly = ModelCanvas._physical_member_polygon(0.0, 0.0, 4.0, 0.0, 0.4)
    assert poly is not None
    assert len(poly) == 4
    ys = [p[1] for p in poly]
    # Two corners at +0.2, two at -0.2
    assert sorted(ys) == pytest.approx([-0.2, -0.2, 0.2, 0.2])
    xs = [p[0] for p in poly]
    assert sorted(xs) == pytest.approx([0.0, 0.0, 4.0, 4.0])


def test_physical_member_polygon_vertical():
    """Vertical member (0,0)→(0,3), depth=0.6 → rectangle ±0.3 in x."""
    from structural_analysis.gui_qt.canvas import ModelCanvas
    poly = ModelCanvas._physical_member_polygon(0.0, 0.0, 0.0, 3.0, 0.6)
    assert poly is not None
    xs = [p[0] for p in poly]
    assert sorted(xs) == pytest.approx([-0.3, -0.3, 0.3, 0.3])


def test_physical_member_polygon_midpoint_on_centerline():
    """Centre of the 4-corner rectangle lies on the input segment midpoint."""
    from structural_analysis.gui_qt.canvas import ModelCanvas
    xi, yi, xj, yj = 1.0, 2.0, 5.0, 4.0
    poly = ModelCanvas._physical_member_polygon(xi, yi, xj, yj, 0.5)
    assert poly is not None
    cx = sum(p[0] for p in poly) / 4
    cy = sum(p[1] for p in poly) / 4
    assert cx == pytest.approx((xi + xj) / 2, abs=1e-9)
    assert cy == pytest.approx((yi + yj) / 2, abs=1e-9)


def test_physical_member_polygon_zero_length_returns_none():
    """Degenerate zero-length member must return None, not raise."""
    from structural_analysis.gui_qt.canvas import ModelCanvas
    result = ModelCanvas._physical_member_polygon(1.0, 1.0, 1.0, 1.0, 0.3)
    assert result is None


def _make_l_frame_model():
    """L-frame: column 1-2 (vertical) + beam 2-3 (horizontal), node 2 shared."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import FrameElement2D
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.nodes[3] = Node(3, 4.0, 3.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4,
                       depth=0.4, rho=7850.0)
    )
    m.elements.append(
        FrameElement2D(id=2, node_i=2, node_j=3, E=2.1e8, A=0.01, I=1e-4,
                       depth=0.3, rho=7850.0)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m


def test_joint_overlap_nodes_shared_node_detected():
    """Node 2 has 2 frames meeting → it appears in the overlap list."""
    from structural_analysis.gui_qt.canvas import ModelCanvas
    m = _make_l_frame_model()
    joints = ModelCanvas._joint_overlap_nodes(m, default_depth=0.2)
    assert len(joints) == 1
    x, y, side = joints[0]
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(3.0)
    # side = max(0.4, 0.3) = 0.4
    assert side == pytest.approx(0.4)


def test_joint_overlap_nodes_single_frame_no_marker():
    """A node touched by only one frame element must NOT get a joint marker."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_qt.canvas import ModelCanvas
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4,
                       depth=0.4, rho=7850.0)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    joints = ModelCanvas._joint_overlap_nodes(m, default_depth=0.2)
    assert joints == []


def test_joint_overlap_nodes_truss_only_no_marker():
    """Truss-only joints must not produce overlap markers."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import TrussElement2D
    from structural_analysis.gui_qt.canvas import ModelCanvas
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.nodes[3] = Node(3, 0.5, 1.0)
    m.elements.append(
        TrussElement2D(id=1, node_i=1, node_j=3, E=2.1e8, A=0.01)
    )
    m.elements.append(
        TrussElement2D(id=2, node_i=2, node_j=3, E=2.1e8, A=0.01)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[2] = Support(node_id=2, ux=True, uy=True, rz=True)
    joints = ModelCanvas._joint_overlap_nodes(m, default_depth=0.2)
    assert joints == []


def test_resolved_default_depth_adaptive():
    """Default depth is ~2% of bbox diagonal, clamped to [0.05, 1.0] m."""
    from structural_analysis.model import StructuralModel, Node
    from structural_analysis.gui_qt.canvas import ModelCanvas, _DEFAULT_VISUAL_DEPTH_FRACTION

    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 10.0, 0.0)
    m.nodes[3] = Node(3, 0.0, 10.0)
    # diag = sqrt(100+100) ≈ 14.14 m  →  0.02 * 14.14 ≈ 0.283 m
    expected = _DEFAULT_VISUAL_DEPTH_FRACTION * math.hypot(10.0, 10.0)
    result = ModelCanvas._resolved_default_depth(m)   # @staticmethod
    assert result == pytest.approx(expected, rel=1e-3)


# ── Qt offscreen smoke tests ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qt_app():
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PyQt6 unavailable: {exc}")
    return QApplication.instance() or QApplication([])


def _make_l_frame_window(qt_app):
    """Return an open MainWindow loaded with the L-frame model."""
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w.show()
    m = _make_l_frame_model()
    w._model = m
    w.canvas._model = lambda: m
    w.canvas.redraw()
    return w


def test_cb_physical_exists_and_unchecked_by_default(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w.show()
    assert hasattr(w, "_cb_physical")
    assert not w._cb_physical.isChecked()
    assert not w.canvas.show_physical_members
    w.close()


def test_toggle_physical_sets_canvas_flag(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w.show()
    w._cb_physical.setChecked(True)
    assert w.canvas.show_physical_members is True
    w._cb_physical.setChecked(False)
    assert w.canvas.show_physical_members is False
    w.close()


def test_physical_view_on_produces_poly_collection(qt_app):
    """With physical view ON, PolyCollection artists must appear on the axes."""
    from matplotlib.collections import PolyCollection
    w = _make_l_frame_window(qt_app)
    w._cb_physical.setChecked(True)
    artists = [a for a in w.canvas.ax.get_children()
               if isinstance(a, PolyCollection)]
    assert len(artists) >= 1, "Expected at least one PolyCollection for frame bodies"
    w.close()


def test_physical_view_off_no_poly_collection(qt_app):
    """With physical view OFF, no PolyCollection should be present."""
    from matplotlib.collections import PolyCollection
    w = _make_l_frame_window(qt_app)
    # Ensure off (default)
    w._cb_physical.setChecked(False)
    artists = [a for a in w.canvas.ax.get_children()
               if isinstance(a, PolyCollection)]
    assert len(artists) == 0


def test_joint_hatch_rectangle_present_when_physical_on(qt_app):
    """Hatched Rectangle marker must appear at the shared L-frame node."""
    from matplotlib.patches import Rectangle
    w = _make_l_frame_window(qt_app)
    w._cb_physical.setChecked(True)
    hatched = [p for p in w.canvas.ax.patches
               if isinstance(p, Rectangle) and p.get_hatch() == "///"]
    assert len(hatched) >= 1, "Expected at least one hatched joint marker"
    w.close()


def test_centerline_dashed_when_physical_on(qt_app):
    """Frame centerline must be thin (linewidth <= 1.0) when physical view is ON."""
    from matplotlib.lines import Line2D
    w = _make_l_frame_window(qt_app)
    w._cb_physical.setChecked(True)
    # Centerline in physical mode: linewidth=1.0, linestyle dashed
    thin_lines = [a for a in w.canvas.ax.lines if isinstance(a, Line2D)
                  and a.get_linewidth() <= 1.0]
    assert len(thin_lines) >= 1, "Expected at least one thin centerline when physical view is on"
    w.close()


def test_centerline_solid_when_physical_off(qt_app):
    """Frame centerline must be thick (linewidth >= 2.0) when physical view is OFF."""
    from matplotlib.lines import Line2D
    w = _make_l_frame_window(qt_app)
    w._cb_physical.setChecked(False)
    thick_lines = [a for a in w.canvas.ax.lines if isinstance(a, Line2D)
                   and a.get_linewidth() >= 1.9]
    assert len(thick_lines) >= 1, "Expected at least one thick centerline when physical view is off"
    w.close()


def test_missing_depth_no_crash(qt_app):
    """Elements with depth=0 must not crash; missing count must be non-negative."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_qt.app import MainWindow
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4,
                       depth=0.0, rho=7850.0)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    w = MainWindow()
    w.show()
    w._model = m
    w.canvas._model = lambda: m
    w._cb_physical.setChecked(True)   # must not raise
    assert w.canvas._physical_members_missing_depth >= 1
    w.close()


def test_physical_view_tooltip_set(qt_app):
    """The Physical members checkbox must carry an informational tooltip."""
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w.show()
    tip = w._cb_physical.toolTip()
    assert "visual only" in tip.lower()
    w.close()
