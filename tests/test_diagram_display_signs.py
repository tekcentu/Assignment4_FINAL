"""Display-only diagram tests: sign-coloured fills + moment orientation.

Pinned behaviour:

* :func:`_split_segments_by_sign` linearly interpolates the zero crossing
  between adjacent samples of opposite signs and emits one segment per
  single-sign region, with adjacent segments sharing the zero point so
  the rendered fills touch cleanly.
* The element-detail-dialog moment subplot inverts the y-axis (legacy
  ``invert=True``) so positive sagging M draws downward, matching the
  conventional structural display.
* The main canvas ``_draw_diagrams`` flips moment ordinates against the
  perpendicular normal so positive sagging M plots **below** a horizontal
  beam's centerline; shear/axial keep the +normal side.
* Sample values returned by :func:`internal_force_at` are NOT mutated by
  any of these display changes.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.model import (
    StructuralModel, Node, Support, UniformDistributedLoad,
)
from structural_analysis.element import FrameElement2D
from structural_analysis.main import run_analysis
from structural_analysis.gui_qt.element_graphics import (
    _split_segments_by_sign,
    sign_fill_color,
    internal_force_at,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _ss_udl(L=6.0, w=10.0):
    m = StructuralModel(title="ss udl")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    e = FrameElement2D(1, 1, 2, E=200_000.0, A=0.02, I=0.08)
    e.member_loads.append(UniformDistributedLoad(wy=-w))
    m.elements = [e]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=False),
        2: Support(2, ux=False, uy=True, rz=False),
    }
    m.nodal_loads = []
    return m, L, w


def _ff_udl(L=6.0, w=10.0):
    m = StructuralModel(title="ff udl")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    e = FrameElement2D(1, 1, 2, E=200_000.0, A=0.02, I=0.08)
    e.member_loads.append(UniformDistributedLoad(wy=-w))
    m.elements = [e]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=True),
        2: Support(2, ux=True, uy=True, rz=True),
    }
    m.nodal_loads = []
    return m, L, w


def _qt_or_skip():
    """Fixture-style helper: import PyQt6 + matplotlib lazily and skip
    cleanly if either is unavailable. Mirrors the pattern used in
    test_modal_structure_terminology.py."""
    try:
        import PyQt6.QtWidgets as _  # noqa: F401
        from matplotlib.figure import Figure  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt / matplotlib unavailable: {exc}")


# ── 1. _split_segments_by_sign — unit tests ──────────────────────────────


def test_split_inserts_interpolated_zero_crossing_at_midpoint():
    segs = _split_segments_by_sign([0.0, 1.0], [1.0, -1.0])
    assert len(segs) == 2
    (xs0, ys0, s0), (xs1, ys1, s1) = segs
    assert s0 == +1 and s1 == -1
    # The split point is exactly the linear-interpolation midpoint.
    assert xs0[-1] == pytest.approx(0.5)
    assert ys0[-1] == 0.0
    assert xs1[0] == pytest.approx(0.5)
    assert ys1[0] == 0.0
    # The original endpoints survive.
    assert xs0[0] == 0.0 and ys0[0] == 1.0
    assert xs1[-1] == 1.0 and ys1[-1] == -1.0


def test_split_off_center_zero_crossing_interpolation():
    # y goes from +3 at x=0 to -1 at x=4 → zero crossing at x=3.
    segs = _split_segments_by_sign([0.0, 4.0], [3.0, -1.0])
    assert len(segs) == 2
    assert segs[0][0][-1] == pytest.approx(3.0)
    assert segs[1][0][0] == pytest.approx(3.0)


def test_split_no_sign_change_returns_single_segment():
    segs = _split_segments_by_sign([0.0, 1.0, 2.0], [3.0, 2.0, 1.0])
    assert len(segs) == 1
    xs, ys, sign = segs[0]
    assert sign == +1
    assert xs == [0.0, 1.0, 2.0]


def test_split_all_zero_returns_single_positive_segment():
    segs = _split_segments_by_sign([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    assert len(segs) == 1
    assert segs[0][2] == +1


def test_split_two_sign_changes_emits_three_segments():
    segs = _split_segments_by_sign(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.0, -1.0, -2.0, -1.0, 1.0],
    )
    # +,−,− around the first crossing, then − to + → 2 splits → 3 segments.
    signs = [seg[2] for seg in segs]
    assert signs == [+1, -1, +1]
    # Crossings are at x=0.5 and x=3.5.
    assert segs[0][0][-1] == pytest.approx(0.5)
    assert segs[2][0][0] == pytest.approx(3.5)


def test_sign_fill_color_palette():
    assert sign_fill_color(+1) == "#1f77b4"   # blue
    assert sign_fill_color(-1) == "#d24c4c"   # red
    assert sign_fill_color(0) == "#1f77b4"    # zero treated as positive


# ── 2. End-to-end: SS UDL beam moment is all positive ────────────────────


def test_ss_udl_moment_split_has_only_positive_region():
    """Simply-supported beam under downward UDL: M(x) ≥ 0 everywhere,
    so the sign split must produce exactly one positive segment."""
    m, L, w = _ss_udl()
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    from structural_analysis.gui_qt.element_graphics import sample_internal_force
    e = m.elements[0]
    ni, nj = m.nodes[1], m.nodes[2]
    f_local = r.member_results[e.id]["f_local"]
    xs, ys = sample_internal_force(e, ni, nj, f_local, "moment", n_samples=21)
    segs = _split_segments_by_sign(xs, ys)
    signs = [s for _, _, s in segs]
    assert signs == [+1], f"expected only positive moment segment, got {signs}"


def test_ss_udl_shear_split_is_blue_then_red():
    """SS UDL beam shear: starts positive at i, crosses zero at midspan,
    ends negative at j → exactly two segments (blue then red)."""
    m, L, w = _ss_udl()
    r = run_analysis(m, verbose=False)
    from structural_analysis.gui_qt.element_graphics import sample_internal_force
    e = m.elements[0]
    ni, nj = m.nodes[1], m.nodes[2]
    f_local = r.member_results[e.id]["f_local"]
    xs, ys = sample_internal_force(e, ni, nj, f_local, "shear", n_samples=21)
    segs = _split_segments_by_sign(xs, ys)
    signs = [s for _, _, s in segs]
    assert signs == [+1, -1], (
        f"SS UDL shear must split into one positive then one negative region, got {signs}"
    )
    # Crossing is near midspan and exact zero in the boundary samples.
    cross_x = segs[0][0][-1]
    assert cross_x == pytest.approx(L / 2.0, rel=1e-6)
    assert segs[0][1][-1] == 0.0
    assert segs[1][1][0] == 0.0


def test_ff_udl_moment_has_negative_ends_positive_middle():
    """Fixed-fixed UDL beam: hogging at supports, sagging at midspan.
    With the dM/dx=V convention, end moments are negative and midspan is
    positive → segments must be red / blue / red."""
    m, L, w = _ff_udl()
    r = run_analysis(m, verbose=False)
    from structural_analysis.gui_qt.element_graphics import sample_internal_force
    e = m.elements[0]
    ni, nj = m.nodes[1], m.nodes[2]
    f_local = r.member_results[e.id]["f_local"]
    xs, ys = sample_internal_force(e, ni, nj, f_local, "moment", n_samples=41)
    segs = _split_segments_by_sign(xs, ys)
    signs = [s for _, _, s in segs]
    assert signs == [-1, +1, -1], (
        f"fixed-fixed UDL moment must split as red / blue / red, got {signs}"
    )


# ── 3. Detail-dialog moment subplot stays inverted ───────────────────────


def test_detail_dialog_moment_subplot_inverted_and_sign_split():
    """The detail-dialog moment subplot must still invert its y-axis so
    positive (sagging) plots downward, and the V/M panels must contain
    multiple fills (the sign-split renderer creates one fill per
    single-sign segment)."""
    _qt_or_skip()
    from matplotlib.figure import Figure
    from structural_analysis.gui_qt.element_graphics import draw_element_detail

    m, L, w = _ss_udl()
    r = run_analysis(m, verbose=False)
    fig = Figure()
    sec_fig = Figure()
    axes = draw_element_detail(fig, m.elements[0], m, r, section_fig=sec_fig)

    # Moment subplot inverted (legacy invert=True still in effect).
    assert bool(axes.ax_m.yaxis_inverted())

    # SS UDL shear has one positive and one negative region → ≥ 2 PolyCollections.
    # ax.fill_between adds a PolyCollection; matplotlib lists it under .collections.
    v_polys = [c for c in axes.ax_v.collections]
    assert len(v_polys) >= 2, (
        f"shear panel should contain at least one blue and one red fill, "
        f"got {len(v_polys)} collection(s)"
    )


# ── 4. Canvas moment flip: parabola below the member ─────────────────────


def test_canvas_moment_diagram_drawn_below_horizontal_beam():
    """For a horizontal SS UDL beam, the moment polygon's apex must lie
    BELOW the member centerline (y < 0) after the orientation flip."""
    _qt_or_skip()
    from PyQt6.QtWidgets import QApplication
    from structural_analysis.gui_qt.canvas import ModelCanvas

    app = QApplication.instance() or QApplication([])  # noqa: F841

    m, L, w = _ss_udl()
    r = run_analysis(m, verbose=False)

    canvas = ModelCanvas(None, model_provider=lambda: m)
    canvas._result = r
    canvas.diagram_kind = "moment"
    canvas.diagram_scale = 1.0
    canvas.diagram_stations = 21
    canvas._draw_diagrams()

    # Collect all PathPatch / Polygon y-coordinates fed to ax.fill().
    fills_y = []
    for patch in canvas.ax.patches:
        verts = patch.get_path().vertices
        fills_y.extend(verts[:, 1].tolist())

    assert fills_y, "expected at least one filled polygon from the moment diagram"
    # The member centerline is at y=0; positive sagging moment must
    # appear BELOW it after the flip.
    assert min(fills_y) < -1e-6, (
        f"moment fill must extend below y=0; min(fill_y)={min(fills_y):g}"
    )
    # And it must NOT extend significantly above (the SS UDL beam has no
    # hogging region).
    assert max(fills_y) < 1e-6, (
        f"moment fill must not extend above the member; max(fill_y)={max(fills_y):g}"
    )


def test_canvas_shear_diagram_not_flipped_for_horizontal_beam():
    """Shear keeps the original +normal orientation: for the SS UDL beam
    the shear is positive at the i-end, so the fill must extend ABOVE
    the horizontal centerline (y > 0)."""
    _qt_or_skip()
    from PyQt6.QtWidgets import QApplication
    from structural_analysis.gui_qt.canvas import ModelCanvas

    app = QApplication.instance() or QApplication([])  # noqa: F841

    m, L, w = _ss_udl()
    r = run_analysis(m, verbose=False)

    canvas = ModelCanvas(None, model_provider=lambda: m)
    canvas._result = r
    canvas.diagram_kind = "shear"
    canvas.diagram_scale = 1.0
    canvas.diagram_stations = 21
    canvas._draw_diagrams()

    fills_y = []
    for patch in canvas.ax.patches:
        verts = patch.get_path().vertices
        fills_y.extend(verts[:, 1].tolist())
    assert fills_y
    # Positive shear region must be above y=0; negative region below.
    assert max(fills_y) > 1e-6
    assert min(fills_y) < -1e-6


# ── 5. Hover read-out values are NOT mutated by display changes ──────────


def test_internal_force_at_unchanged_by_display_flip():
    """The hover read-out and result-table values must be identical to
    the un-flipped sampled values — display orientation must not leak
    into the numeric API."""
    m, L, w = _ss_udl()
    r = run_analysis(m, verbose=False)
    e = m.elements[0]
    ni, nj = m.nodes[1], m.nodes[2]
    f_local = r.member_results[e.id]["f_local"]
    # Midspan moment of SS UDL = +wL²/8 = +45 kN·m, positive in our convention.
    m_mid = internal_force_at(e, ni, nj, f_local, "moment", L / 2.0)
    assert m_mid == pytest.approx(+w * L ** 2 / 8.0, rel=1e-6)
    # And shear at midspan is exactly zero.
    v_mid = internal_force_at(e, ni, nj, f_local, "shear", L / 2.0)
    assert v_mid == pytest.approx(0.0, abs=1e-6)
