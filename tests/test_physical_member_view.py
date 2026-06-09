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


def test_physical_member_polygon_horizontal():
    """Horizontal member (0,0)→(4,0), depth=0.4 → rectangle ±0.2 in y."""
    from structural_analysis.gui_common.geometry import physical_member_polygon
    poly = physical_member_polygon(0.0, 0.0, 4.0, 0.0, 0.4)
    assert poly is not None
    assert len(poly) == 4
    ys = [p[1] for p in poly]
    # Two corners at +0.2, two at -0.2
    assert sorted(ys) == pytest.approx([-0.2, -0.2, 0.2, 0.2])
    xs = [p[0] for p in poly]
    assert sorted(xs) == pytest.approx([0.0, 0.0, 4.0, 4.0])


def test_physical_member_polygon_vertical():
    """Vertical member (0,0)→(0,3), depth=0.6 → rectangle ±0.3 in x."""
    from structural_analysis.gui_common.geometry import physical_member_polygon
    poly = physical_member_polygon(0.0, 0.0, 0.0, 3.0, 0.6)
    assert poly is not None
    xs = [p[0] for p in poly]
    assert sorted(xs) == pytest.approx([-0.3, -0.3, 0.3, 0.3])


def test_physical_member_polygon_midpoint_on_centerline():
    """Centre of the 4-corner rectangle lies on the input segment midpoint."""
    from structural_analysis.gui_common.geometry import physical_member_polygon
    xi, yi, xj, yj = 1.0, 2.0, 5.0, 4.0
    poly = physical_member_polygon(xi, yi, xj, yj, 0.5)
    assert poly is not None
    cx = sum(p[0] for p in poly) / 4
    cy = sum(p[1] for p in poly) / 4
    assert cx == pytest.approx((xi + xj) / 2, abs=1e-9)
    assert cy == pytest.approx((yi + yj) / 2, abs=1e-9)


def test_physical_member_polygon_zero_length_returns_none():
    """Degenerate zero-length member must return None, not raise."""
    from structural_analysis.gui_common.geometry import physical_member_polygon
    result = physical_member_polygon(1.0, 1.0, 1.0, 1.0, 0.3)
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


def _build_polys_for_model(m):
    """Helper: build the {elem_id: body_polygon} dict the canvas would build."""
    from structural_analysis.gui_common.geometry import (
        physical_display_thickness, physical_member_polygon,
        resolved_default_depth,
    )
    from structural_analysis.element import FrameElement2D
    default_d = resolved_default_depth(m)
    polys = {}
    for elem in m.elements:
        if not isinstance(elem, FrameElement2D):
            continue
        ni = m.nodes[elem.node_i]
        nj = m.nodes[elem.node_j]
        d = physical_display_thickness(elem, m.sections)
        if d <= 0.0:
            d = default_d
        poly = physical_member_polygon(ni.x, ni.y, nj.x, nj.y, d)
        if poly is not None:
            polys[elem.id] = poly
    return polys


def test_joint_overlap_region_rectangular_when_sections_differ():
    """Beam (depth 0.3) framing into column (depth 0.4) at a corner joint.

    Both body rectangles end at the shared centerline node and extend AWAY
    from it (column goes south, beam goes east).  Their geometric overlap
    is therefore the *quadrant* near the corner:

      column body:  x ∈ [-0.2, 0.2]   (half-width 0.2)
      beam body:    y ∈ [2.85, 3.15]  (half-depth 0.15)
      intersection: x ∈ [0, 0.2], y ∈ [2.85, 3]  → 0.2 × 0.15

    Acceptance: the overlap is *geometry-derived* (not a fixed square)
    and reflects the differing section depths (0.2 ≠ 0.15).
    """
    from structural_analysis.gui_common.geometry import joint_overlap_regions
    m = _make_l_frame_model()       # column depth=0.4, beam depth=0.3
    polys = _build_polys_for_model(m)
    regions = joint_overlap_regions(m, polys)
    assert len(regions) == 1
    _poly, (cx, cy), (w, h), pair = regions[0]
    assert w == pytest.approx(0.2, abs=1e-6)
    assert h == pytest.approx(0.15, abs=1e-6)
    # Centroid (mean of vertices of the corner quadrant).
    assert cx == pytest.approx(0.1, abs=1e-6)
    assert cy == pytest.approx(2.925, abs=1e-6)
    assert pair == (1, 2)


def test_joint_overlap_region_square_when_sections_match():
    """Equal section depths → square intersection (still derived from real
    geometry, not a fixed proxy)."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_common.geometry import joint_overlap_regions
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.nodes[3] = Node(3, 4.0, 3.0)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4,
        depth=0.5, rho=7850.0,
    ))
    m.elements.append(FrameElement2D(
        id=2, node_i=2, node_j=3, E=2.1e8, A=0.01, I=1e-4,
        depth=0.5, rho=7850.0,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    polys = _build_polys_for_model(m)
    regions = joint_overlap_regions(m, polys)
    assert len(regions) == 1
    _poly, _c, (w, h), _pair = regions[0]
    # Both half-widths are 0.25 → square 0.25 × 0.25 corner quadrant.
    assert w == pytest.approx(0.25, abs=1e-6)
    assert h == pytest.approx(0.25, abs=1e-6)


def test_joint_overlap_regions_single_frame_no_region():
    """A node touched by only one frame element must NOT get a region."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_common.geometry import joint_overlap_regions
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4,
        depth=0.4, rho=7850.0,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    polys = _build_polys_for_model(m)
    assert joint_overlap_regions(m, polys) == []


def test_joint_overlap_regions_truss_only_no_region():
    """Truss-only joints must not produce overlap regions."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import TrussElement2D
    from structural_analysis.gui_common.geometry import joint_overlap_regions
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.nodes[3] = Node(3, 0.5, 1.0)
    m.elements.append(TrussElement2D(
        id=1, node_i=1, node_j=3, E=2.1e8, A=0.01,
    ))
    m.elements.append(TrussElement2D(
        id=2, node_i=2, node_j=3, E=2.1e8, A=0.01,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[2] = Support(node_id=2, ux=True, uy=True, rz=True)
    # Trusses are excluded entirely — no polygons → no regions.
    assert joint_overlap_regions(m, {}) == []


# ── polygon_intersection helper ───────────────────────────────────────────────


def test_polygon_intersection_two_unit_squares():
    """Two overlapping axis-aligned unit squares → 0.5×0.5 intersection."""
    from structural_analysis.gui_common.geometry import polygon_intersection
    a = [(0, 0), (1, 0), (1, 1), (0, 1)]
    b = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
    out = polygon_intersection(a, b)
    assert len(out) == 4
    xs = sorted(p[0] for p in out)
    ys = sorted(p[1] for p in out)
    assert xs == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert ys == pytest.approx([0.5, 0.5, 1.0, 1.0])


def test_polygon_intersection_disjoint_returns_empty():
    from structural_analysis.gui_common.geometry import polygon_intersection
    a = [(0, 0), (1, 0), (1, 1), (0, 1)]
    b = [(2, 2), (3, 2), (3, 3), (2, 3)]
    assert polygon_intersection(a, b) == []


def test_polygon_intersection_winding_agnostic():
    """CW and CCW inputs must yield identical bounding boxes."""
    from structural_analysis.gui_common.geometry import polygon_intersection
    a_ccw = [(0, 0), (1, 0), (1, 1), (0, 1)]
    b_cw = [(0.5, 1.5), (1.5, 1.5), (1.5, 0.5), (0.5, 0.5)]
    out = polygon_intersection(a_ccw, b_cw)
    assert len(out) == 4
    xs = [p[0] for p in out]
    ys = [p[1] for p in out]
    assert max(xs) - min(xs) == pytest.approx(0.5)
    assert max(ys) - min(ys) == pytest.approx(0.5)


# ── physical_display_thickness rule ───────────────────────────────────────────


def test_display_thickness_i_section_uses_outer_envelope():
    """I-section: use max(depth, width) — never web thickness."""
    from structural_analysis.model import Section
    from structural_analysis.gui_common.geometry import physical_display_thickness
    sec = Section(
        id=1, material_id=1, A=0.005, I=1e-4,
        depth=0.20, width=0.30,      # b=300, h=200 — flange wider than depth
        shape_type="i_section",
        b=0.30, h=0.20, tf=0.012, tw=0.008,
    )

    class _E:
        section_id = 1
        depth = 0.20
    sections = {1: sec}
    # max(0.20, 0.30) = 0.30; we must NOT pick tw (0.008) or tf (0.012).
    assert physical_display_thickness(_E(), sections) == pytest.approx(0.30)


def test_display_thickness_rectangle_uses_depth():
    from structural_analysis.model import Section
    from structural_analysis.gui_common.geometry import physical_display_thickness
    sec = Section(
        id=1, material_id=1, A=0.06, I=1e-4,
        depth=0.50, width=0.30, shape_type="rectangle",
        b=0.30, h=0.50,
    )

    class _E:
        section_id = 1
        depth = 0.50
    assert physical_display_thickness(_E(), {1: sec}) == pytest.approx(0.50)


def test_display_thickness_manual_falls_back_to_depth():
    from structural_analysis.model import Section
    from structural_analysis.gui_common.geometry import physical_display_thickness
    sec = Section(
        id=1, material_id=1, A=0.01, I=1e-4,
        depth=0.35, width=0.0,         # manual: no width recorded
        shape_type="manual",
    )

    class _E:
        section_id = 1
        depth = 0.35
    assert physical_display_thickness(_E(), {1: sec}) == pytest.approx(0.35)


def test_display_thickness_no_section_uses_elem_depth():
    from structural_analysis.gui_common.geometry import physical_display_thickness

    class _E:
        section_id = None
        depth = 0.22
    assert physical_display_thickness(_E(), {}) == pytest.approx(0.22)


def test_display_thickness_missing_all_returns_zero():
    """No section and elem.depth==0 → 0.0; caller applies adaptive default."""
    from structural_analysis.gui_common.geometry import physical_display_thickness

    class _E:
        section_id = None
        depth = 0.0
    assert physical_display_thickness(_E(), {}) == 0.0


def test_resolved_default_depth_adaptive():
    """Default depth is ~2% of bbox diagonal, clamped to [0.05, 1.0] m."""
    from structural_analysis.model import StructuralModel, Node
    from structural_analysis.gui_common.geometry import (
        resolved_default_depth, PHYSICAL_DEPTH_FRACTION,
    )
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 10.0, 0.0)
    m.nodes[3] = Node(3, 0.0, 10.0)
    # diag = sqrt(100+100) ≈ 14.14 m  →  0.02 * 14.14 ≈ 0.283 m
    expected = PHYSICAL_DEPTH_FRACTION * math.hypot(10.0, 10.0)
    result = resolved_default_depth(m)
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


def test_joint_overlap_polygon_present_when_physical_on(qt_app):
    """Hatched polygon (the geometry-derived joint overlap) must appear at the
    shared L-frame node when physical view is on."""
    from matplotlib.patches import Polygon as MplPolygon
    w = _make_l_frame_window(qt_app)
    w._cb_physical.setChecked(True)
    hatched = [p for p in w.canvas.ax.patches
               if isinstance(p, MplPolygon) and p.get_hatch() == "///"]
    assert len(hatched) >= 1, "Expected at least one hatched joint overlap polygon"
    # Verify it's a real intersection polygon (≥3 vertices), not a fixed square.
    verts = hatched[0].get_xy()
    assert len(verts) >= 3
    w.close()


def test_joint_overlap_debug_label_present_when_physical_on(qt_app):
    """Temporary debug label (w×h @ (cx,cy)) must appear on overlap patch."""
    w = _make_l_frame_window(qt_app)
    w._cb_physical.setChecked(True)
    texts = [t.get_text() for t in w.canvas.ax.texts]
    matches = [t for t in texts if "×" in t and "@" in t]
    assert matches, f"Expected debug label like '0.40×0.30 m @ (...)', got texts={texts}"
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
