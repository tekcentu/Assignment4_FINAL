"""Regression: shear-diagram display SIDE follows a world-anchored,
sign-based convention (SAP-like), while all numerical values and the
moment diagram are untouched.

Display convention pinned here (``canvas._shear_display_normal`` +
``canvas._draw_diagrams``):

* mostly horizontal member → +V draws ABOVE the member, −V BELOW;
* mostly vertical member   → +V draws to the RIGHT, −V to the LEFT.

The offset is ``signed_V × positive_visual_normal`` — the SIGN of V
alone picks the side, so a sign-changing beam shows its +V region above
and its −V region below. This is display-only: the element local y-axis,
``evaluate_internal_force`` / ``sample_internal_force`` numerical values,
station export, hover read-out, element-detail values, and the moment
diagram are all unchanged.
"""

from __future__ import annotations

import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QSettings  # noqa: E402
    from PyQt6.QtWidgets import QApplication  # noqa: E402
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 unavailable: {exc}", allow_module_level=True)

from structural_analysis.element import FrameElement2D  # noqa: E402
from structural_analysis.file_io import read_input_file  # noqa: E402
from structural_analysis.gui_qt.app import MainWindow  # noqa: E402
from structural_analysis.gui_qt.canvas import (  # noqa: E402
    HitResult, ModelCanvas, _shear_display_normal,
)
from structural_analysis.gui_qt.element_graphics import (  # noqa: E402
    internal_force_at, sample_internal_force,
)
from structural_analysis.main import run_analysis  # noqa: E402
from structural_analysis.model import (  # noqa: E402
    Material, Node, NodalLoad, Section, StructuralModel, Support,
    UniformDistributedLoad,
)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _clean_qsettings():
    s = QSettings("CE4011", "StructuralAnalysis")
    s.remove("units_preset")
    yield
    s.remove("units_preset")


# ── Shared helpers ───────────────────────────────────────────────────────


def _shear_canvas(m: StructuralModel) -> ModelCanvas:
    r = run_analysis(m, verbose=False)
    canvas = ModelCanvas(None, model_provider=lambda: m)
    canvas._result = r
    canvas.diagram_kind = "shear"
    canvas.diagram_scale = 1.0
    canvas.diagram_stations = 21
    canvas._draw_diagrams()
    return canvas


def _patches_with_colour(canvas: ModelCanvas):
    """Per-patch (xmin, xmax, ymin, ymax, 'blue'|'red'|'other')."""
    out = []
    for p in canvas.ax.patches:
        v = p.get_path().vertices
        c = p.get_facecolor()
        col = "blue" if c[2] > 0.5 else "red" if c[0] > 0.5 else "other"
        out.append((float(v[:, 0].min()), float(v[:, 0].max()),
                    float(v[:, 1].min()), float(v[:, 1].max()), col))
    return out


def _horizontal_member(*, simply_supported: bool):
    m = StructuralModel(title="horizontal")
    m.materials[1] = Material(id=1, name="C", E=2.0e8, density=0.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.02, I=8e-4, depth=0.3)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(1, 1, 2, E=2.0e8, A=0.02, I=8e-4, section_id=1)
    e.member_loads.append(UniformDistributedLoad(wy=-10.0))
    m.elements = [e]
    if simply_supported:
        m.supports = {1: Support(1, ux=True, uy=True, rz=False),
                      2: Support(2, ux=False, uy=True, rz=False)}
    else:  # cantilever fixed at the left
        m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    return m


def _vertical_column(tip_fx: float):
    m = StructuralModel(title="vertical")
    m.materials[1] = Material(id=1, name="C", E=2.0e8, density=0.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.02, I=8e-4, depth=0.3)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 0.0, 4.0)}
    e = FrameElement2D(1, 1, 2, E=2.0e8, A=0.02, I=8e-4, section_id=1)
    m.elements = [e]
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    m.nodal_loads = [NodalLoad(node_id=2, fx=tip_fx)]
    return m


# ── 1. The display-normal helper ─────────────────────────────────────────


def test_shear_display_normal_horizontal_is_world_up():
    # Horizontal member (tangent ±x) → +V up.
    assert _shear_display_normal(1.0, 0.0) == (0.0, 1.0)
    assert _shear_display_normal(-1.0, 0.0) == (0.0, 1.0)


def test_shear_display_normal_vertical_is_world_right():
    # Vertical member (tangent ±y) → +V right.
    assert _shear_display_normal(0.0, 1.0) == (1.0, 0.0)
    assert _shear_display_normal(0.0, -1.0) == (1.0, 0.0)


def test_shear_display_normal_diagonal_tie_goes_horizontal():
    # 45° tie: |cx| == |cy| resolves to horizontal (world up).
    s = 2 ** -0.5
    assert _shear_display_normal(s, s) == (0.0, 1.0)


# ── 2. Horizontal cantilever, downward UDL → +V ABOVE ───────────────────


def test_horizontal_cantilever_positive_shear_drawn_above(qt_app):
    m = _horizontal_member(simply_supported=False)
    canvas = _shear_canvas(m)
    patches = _patches_with_colour(canvas)
    # All shear is positive on a left-fixed cantilever with a downward
    # UDL → a single blue lobe entirely ABOVE the member line (y = 0).
    blue = [p for p in patches if p[4] == "blue"]
    assert blue, "expected a positive (blue) shear lobe"
    assert max(p[3] for p in blue) > 1e-6        # extends above
    assert min(p[2] for p in blue) >= -1e-6      # nothing below


# ── 3. Sign-changing SS beam → +V above, −V below ───────────────────────


def test_sign_changing_beam_positive_above_negative_below(qt_app):
    m = _horizontal_member(simply_supported=True)
    canvas = _shear_canvas(m)
    patches = _patches_with_colour(canvas)
    blue = next(p for p in patches if p[4] == "blue")
    red = next(p for p in patches if p[4] == "red")
    # +V region (left half) above y = 0; −V region (right half) below.
    assert blue[3] > 1e-6 and blue[2] >= -1e-6
    assert red[2] < -1e-6 and red[3] <= 1e-6


# ── 4. Vertical column, +x tip load → +V RIGHT ──────────────────────────


def test_vertical_column_positive_shear_drawn_right(qt_app):
    m = _vertical_column(tip_fx=10.0)
    e = m.elements[0]
    r = run_analysis(m, verbose=False)
    _, vs = sample_internal_force(
        e, m.nodes[1], m.nodes[2],
        list(r.member_results[e.id]["f_local"]), "shear", n_samples=3)
    assert vs[0] > 0, "fixture should produce positive V"
    canvas = _shear_canvas(m)
    patches = _patches_with_colour(canvas)
    blue = [p for p in patches if p[4] == "blue"]
    assert blue
    # Column sits on x = 0; positive shear lobes to the RIGHT (x > 0).
    assert max(p[1] for p in blue) > 1e-6
    assert min(p[0] for p in blue) >= -1e-6


# ── 5. Vertical column, −x tip load → −V LEFT ───────────────────────────


def test_vertical_column_negative_shear_drawn_left(qt_app):
    m = _vertical_column(tip_fx=-10.0)
    e = m.elements[0]
    r = run_analysis(m, verbose=False)
    _, vs = sample_internal_force(
        e, m.nodes[1], m.nodes[2],
        list(r.member_results[e.id]["f_local"]), "shear", n_samples=3)
    assert vs[0] < 0, "fixture should produce negative V"
    canvas = _shear_canvas(m)
    patches = _patches_with_colour(canvas)
    red = [p for p in patches if p[4] == "red"]
    assert red
    # Negative shear lobes to the LEFT (x < 0) of the column at x = 0.
    assert min(p[0] for p in red) < -1e-6
    assert max(p[1] for p in red) <= 1e-6


# ── 6. Numerical values + moment diagram untouched ──────────────────────

_EX03_V = {1: 21.7844, 2: 28.2156, 3: -2.2724}


def test_sample_internal_force_values_unchanged():
    m = read_input_file("inputs/example_03_portal_frame_lateral_load.txt")
    r = run_analysis(m, verbose=False)
    for e in m.elements:
        ni, nj = m.nodes[e.node_i], m.nodes[e.node_j]
        f = r.member_results[e.id]["f_local"]
        _, vs = sample_internal_force(e, ni, nj, list(f), "shear",
                                      n_samples=5)
        for v in vs:
            assert v == pytest.approx(_EX03_V[e.id], rel=1e-4)


def test_internal_force_at_values_unchanged():
    m = read_input_file("inputs/example_03_portal_frame_lateral_load.txt")
    r = run_analysis(m, verbose=False)
    for e in m.elements:
        ni, nj = m.nodes[e.node_i], m.nodes[e.node_j]
        f = r.member_results[e.id]["f_local"]
        L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
        v = internal_force_at(e, ni, nj, list(f), "shear", L / 2.0)
        assert v == pytest.approx(_EX03_V[e.id], rel=1e-4)


def test_moment_diagram_unchanged_by_shear_fix(qt_app):
    """The moment diagram still uses the element-local normal (positive
    sagging below the member). Pin the example_03 right-column moment
    fill side — it extends to the right of x = 6, as before the fix."""
    m = read_input_file("inputs/example_03_portal_frame_lateral_load.txt")
    r = run_analysis(m, verbose=False)
    canvas = ModelCanvas(None, model_provider=lambda: m)
    canvas._result = r
    canvas.diagram_kind = "moment"
    canvas.diagram_scale = 1.0
    canvas.diagram_stations = 21
    canvas._draw_diagrams()
    extents = [(float(p.get_path().vertices[:, 0].min()),
                float(p.get_path().vertices[:, 0].max()))
               for p in canvas.ax.patches]
    right_col_M = next((xmin, xmax) for xmin, xmax in extents
                       if xmin >= 6.0 - 1e-6)
    assert right_col_M[1] > 6.05


# ── 7. End-to-end (MainWindow): station export + hover unchanged ────────


def _portal_mainwindow(w: MainWindow):
    m = w._model
    m.materials[1] = Material(id=1, name="C", E=2.0e8, density=0.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.02, I=8e-4, depth=0.3)
    m.nodes = {1: Node(1, 0, 0), 2: Node(2, 6, 0),
               3: Node(3, 0, 4), 4: Node(4, 6, 4)}
    m.elements = [
        FrameElement2D(id=1, node_i=1, node_j=3, E=2e8,
                       A=0.02, I=8e-4, section_id=1),
        FrameElement2D(id=2, node_i=2, node_j=4, E=2e8,
                       A=0.02, I=8e-4, section_id=1),
        FrameElement2D(id=3, node_i=3, node_j=4, E=2e8,
                       A=0.02, I=8e-4, section_id=1),
    ]
    m.supports = {1: Support(node_id=1, ux=True, uy=True, rz=True),
                  2: Support(node_id=2, ux=True, uy=True, rz=True)}
    m.nodal_loads = [NodalLoad(node_id=4, fx=50.0, fy=-20.0),
                     NodalLoad(node_id=3, fy=-20.0)]
    w._run_static_solve(active_only=False)


def test_station_export_V_values_unchanged(qt_app, tmp_path, monkeypatch):
    import csv
    w = MainWindow()
    _portal_mainwindow(w)
    out = tmp_path / "portal.csv"
    import structural_analysis.gui_qt.app as appmod
    monkeypatch.setattr(
        appmod.QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(out), "CSV (*.csv)"))
    w._export_station_results()
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]
    by_elem: dict[str, list[float]] = {}
    for r in rows:
        by_elem.setdefault(r[0], []).append(float(r[3]))
    for e in w._model.elements:
        ni, nj = w._model.nodes[e.node_i], w._model.nodes[e.node_j]
        f = w._result.member_results[e.id]["f_local"]
        _, vs_expected = sample_internal_force(
            e, ni, nj, list(f), "shear", n_samples=21)
        for got, exp in zip(by_elem[str(e.id)], vs_expected):
            assert got == pytest.approx(exp, rel=1e-4, abs=1e-3)


def test_hover_readout_V_value_unchanged(qt_app):
    w = MainWindow()
    _portal_mainwindow(w)
    w.canvas.diagram_kind = "shear"
    beam = w._model.elements[2]
    ni, nj = w._model.nodes[beam.node_i], w._model.nodes[beam.node_j]
    L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
    f = w._result.member_results[beam.id]["f_local"]
    expected_v = internal_force_at(beam, ni, nj, list(f), "shear", L / 2.0)
    hit = HitResult(x=3.0, y=4.0, element_id=3)
    txt = w._diagram_value_text_for_hit(hit)
    assert txt is not None
    m = re.search(r"=\s*([+-]?\d+\.?\d*)", txt)
    assert m, txt
    assert float(m.group(1)) == pytest.approx(expected_v, rel=1e-3, abs=1e-3)


# ── 8. Canonical display direction & node-order invariance ──────────────


from structural_analysis.gui_qt.canvas import (  # noqa: E402
    _canonical_display_direction,
)


def test_canonical_horizontal_left_to_right_is_not_reversed():
    can_tx, can_ty, is_rev = _canonical_display_direction(1.0, 0.0)
    assert (can_tx, can_ty) == (1.0, 0.0)
    assert is_rev is False


def test_canonical_horizontal_right_to_left_is_reversed():
    can_tx, can_ty, is_rev = _canonical_display_direction(-1.0, 0.0)
    assert (can_tx, can_ty) == (1.0, 0.0)
    assert is_rev is True


def test_canonical_vertical_bottom_to_top_is_not_reversed():
    can_tx, can_ty, is_rev = _canonical_display_direction(0.0, 1.0)
    assert (can_tx, can_ty) == (0.0, 1.0)
    assert is_rev is False


def test_canonical_vertical_top_to_bottom_is_reversed():
    can_tx, can_ty, is_rev = _canonical_display_direction(0.0, -1.0)
    assert (can_tx, can_ty) == (0.0, 1.0)
    assert is_rev is True


def _ss_beam_global_udl(*, reversed_order: bool):
    """SS beam with a GLOBAL-coord UDL (physical load independent of
    element node order)."""
    m = StructuralModel(title="ss beam")
    m.materials[1] = Material(id=1, name="C", E=2.0e8, density=0.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.02, I=8e-4, depth=0.3)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    if reversed_order:
        e = FrameElement2D(id=1, node_i=2, node_j=1, E=2.0e8,
                           A=0.02, I=8e-4, section_id=1)
    else:
        e = FrameElement2D(id=1, node_i=1, node_j=2, E=2.0e8,
                           A=0.02, I=8e-4, section_id=1)
    e.member_loads.append(
        UniformDistributedLoad(wy=-10.0, coord_system="global"))
    m.elements = [e]
    m.supports = {1: Support(1, ux=True, uy=True, rz=False),
                  2: Support(2, ux=False, uy=True, rz=False)}
    return m


def _vertical_column_global(*, reversed_order: bool, tip_fx: float):
    m = StructuralModel(title="column")
    m.materials[1] = Material(id=1, name="C", E=2.0e8, density=0.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.02, I=8e-4, depth=0.3)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 0.0, 4.0)}
    if reversed_order:
        e = FrameElement2D(id=1, node_i=2, node_j=1, E=2.0e8,
                           A=0.02, I=8e-4, section_id=1)
    else:
        e = FrameElement2D(id=1, node_i=1, node_j=2, E=2.0e8,
                           A=0.02, I=8e-4, section_id=1)
    m.elements = [e]
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    m.nodal_loads = [NodalLoad(node_id=2, fx=tip_fx)]
    return m


def _patch_centroids(canvas: ModelCanvas):
    """Per-patch (cx, cy, colour) — used to compare visual layout
    between two node-order variants of the same model."""
    out = []
    for p in canvas.ax.patches:
        v = p.get_path().vertices
        c = p.get_facecolor()
        col = "blue" if c[2] > 0.5 else "red" if c[0] > 0.5 else "other"
        out.append((float(v[:, 0].mean()), float(v[:, 1].mean()), col))
    # Sort so the two variants compare element-wise.
    return sorted(out)


def test_horizontal_beam_reversed_node_order_renders_identically(qt_app):
    """A sign-changing SS beam: drawn left→right vs right→left must
    produce the SAME physical visual (same colours, same sides) after
    the canonical-display alignment."""
    a = _patch_centroids(_shear_canvas(_ss_beam_global_udl(
        reversed_order=False)))
    b = _patch_centroids(_shear_canvas(_ss_beam_global_udl(
        reversed_order=True)))
    assert len(a) == len(b)
    for (ax, ay, ac), (bx, by, bc) in zip(a, b):
        assert ac == bc
        assert ax == pytest.approx(bx, abs=1e-6)
        assert ay == pytest.approx(by, abs=1e-6)


def test_vertical_column_reversed_node_order_renders_identically(qt_app):
    """A vertical column with horizontal tip load: drawn bottom→top vs
    top→bottom must produce the same visual (positive shear right)."""
    for tip_fx in (+10.0, -10.0):
        a = _patch_centroids(_shear_canvas(_vertical_column_global(
            reversed_order=False, tip_fx=tip_fx)))
        b = _patch_centroids(_shear_canvas(_vertical_column_global(
            reversed_order=True, tip_fx=tip_fx)))
        assert len(a) == len(b)
        for (ax, ay, ac), (bx, by, bc) in zip(a, b):
            assert ac == bc
            assert ax == pytest.approx(bx, abs=1e-6)
            assert ay == pytest.approx(by, abs=1e-6)


def test_internal_sample_internal_force_values_are_node_order_dependent():
    """Sanity: ``sample_internal_force`` itself IS sensitive to node
    order (the canonical alignment lives in the CANVAS, not in the
    helper) — so numerical V values still reflect the local frame the
    solver used."""
    m_fwd = _ss_beam_global_udl(reversed_order=False)
    m_rev = _ss_beam_global_udl(reversed_order=True)
    r_fwd = run_analysis(m_fwd, verbose=False)
    r_rev = run_analysis(m_rev, verbose=False)
    e_fwd = m_fwd.elements[0]
    e_rev = m_rev.elements[0]
    _, vs_fwd = sample_internal_force(
        e_fwd, m_fwd.nodes[e_fwd.node_i], m_fwd.nodes[e_fwd.node_j],
        list(r_fwd.member_results[e_fwd.id]["f_local"]),
        "shear", n_samples=5)
    _, vs_rev = sample_internal_force(
        e_rev, m_rev.nodes[e_rev.node_i], m_rev.nodes[e_rev.node_j],
        list(r_rev.member_results[e_rev.id]["f_local"]),
        "shear", n_samples=5)
    # Reversed orientation: V values sign-flipped relative to forward
    # (so when read in reverse iteration order they recreate the same
    # physical curve — which the canvas exploits).
    for a, b in zip(vs_fwd, list(reversed(vs_rev))):
        assert a == pytest.approx(b, abs=1e-6)


def test_moment_diagram_reversed_node_order_renders_identically(qt_app):
    """A SS-beam moment diagram: drawn left→right vs right→left must
    render to the SAME side of the member. The user's instruction
    'moment numerical values unchanged, station order reverses when
    member direction is reversed' lands the moment fill on the same
    canonical side regardless of node order."""
    m_fwd = _ss_beam_global_udl(reversed_order=False)
    m_rev = _ss_beam_global_udl(reversed_order=True)
    for m in (m_fwd, m_rev):
        canvas = ModelCanvas(None, model_provider=lambda mm=m: mm)
        canvas._result = run_analysis(m, verbose=False)
        canvas.diagram_kind = "moment"
        canvas.diagram_scale = 1.0
        canvas.diagram_stations = 21
        canvas._draw_diagrams()
        m._centroids = _patch_centroids(canvas)
    assert m_fwd._centroids == m_rev._centroids or all(
        a[2] == b[2]
        and a[0] == pytest.approx(b[0], abs=1e-6)
        and a[1] == pytest.approx(b[1], abs=1e-6)
        for a, b in zip(m_fwd._centroids, m_rev._centroids)
    )


# ── 9. Sanity on the canonical V profile + inclined-member coverage ─────


def test_ss_beam_global_udl_midspan_V_is_zero_and_ends_are_plus_minus_30():
    """Pin the SS beam V profile used by the node-order tests so the
    PR notes can quote it without ambiguity: V at midspan is 0 (the
    classical symmetric-load result), end-station V at i / j is ±30.
    Reversing the node order flips the sign at every station EXCEPT
    midspan, which stays zero."""
    L = 6.0
    w = 10.0      # |wy| of the global UDL
    expected_end = w * L / 2.0     # 30.0

    m_fwd = _ss_beam_global_udl(reversed_order=False)
    m_rev = _ss_beam_global_udl(reversed_order=True)
    for label, m in (("forward", m_fwd), ("reversed", m_rev)):
        r = run_analysis(m, verbose=False)
        e = m.elements[0]
        _, vs = sample_internal_force(
            e, m.nodes[e.node_i], m.nodes[e.node_j],
            list(r.member_results[e.id]["f_local"]),
            "shear", n_samples=5)
        # 5 samples: indices 0 (i-end), 1, 2 (midspan), 3, 4 (j-end).
        assert vs[2] == pytest.approx(0.0, abs=1e-9), (
            f"{label}: midspan V must be 0 for a symmetric SS UDL beam")
        assert abs(vs[0]) == pytest.approx(expected_end, abs=1e-6)
        assert abs(vs[-1]) == pytest.approx(expected_end, abs=1e-6)
        assert vs[0] * vs[-1] < 0, (
            f"{label}: end-station V signs are opposite "
            f"(reactions push opposite ways)")


def test_canonical_handles_diagonal_members_with_45deg_threshold():
    """Inclined members are classified by |cx| ≥ |cy|:

    * ≤ 45° from horizontal → treated as horizontal (canonical = +x);
    * > 45° from horizontal → treated as vertical (canonical = +y).

    This rule covers structural-realistic geometry (most beams and
    columns are within a few degrees of the world axes). Truss
    diagonals at exactly 30°–60° render as horizontal or vertical based
    on which side of the 45° threshold they fall on, and the world-up /
    world-right shear normal is used accordingly. Documenting the
    behaviour here so a future improvement (e.g. a member-perpendicular
    normal for steep diagonals) can be added in a follow-up PR without
    silently changing what users currently see."""
    import math

    for angle_deg in (0, 15, 30, 44):
        rad = math.radians(angle_deg)
        cx, cy = math.cos(rad), math.sin(rad)
        can_tx, can_ty, is_rev = _canonical_display_direction(cx, cy)
        assert (can_tx, can_ty) == (1.0, 0.0)
        assert is_rev is False
        assert _shear_display_normal(can_tx, can_ty) == (0.0, 1.0)

    for angle_deg in (46, 60, 75, 90):
        rad = math.radians(angle_deg)
        cx, cy = math.cos(rad), math.sin(rad)
        can_tx, can_ty, is_rev = _canonical_display_direction(cx, cy)
        assert (can_tx, can_ty) == (0.0, 1.0)
        assert is_rev is False
        assert _shear_display_normal(can_tx, can_ty) == (1.0, 0.0)


def test_diagonal_member_reversed_node_order_renders_identically(qt_app):
    """A 30° inclined truss diagonal modelled with swapped node order
    must still render identically — the canonical-walk alignment
    extends to inclined members via the 45° classification rule."""
    import math

    def diagonal(reversed_order: bool):
        m = StructuralModel(title="diagonal")
        m.materials[1] = Material(id=1, name="C", E=2.0e8, density=0.0)
        m.sections[1] = Section(id=1, name="S", material_id=1,
                                A=0.02, I=8e-4, depth=0.3)
        L = 6.0
        angle = math.radians(30.0)
        m.nodes = {
            1: Node(1, 0.0, 0.0),
            2: Node(2, L * math.cos(angle), L * math.sin(angle)),
        }
        if reversed_order:
            e = FrameElement2D(id=1, node_i=2, node_j=1, E=2.0e8,
                               A=0.02, I=8e-4, section_id=1)
        else:
            e = FrameElement2D(id=1, node_i=1, node_j=2, E=2.0e8,
                               A=0.02, I=8e-4, section_id=1)
        e.member_loads.append(
            UniformDistributedLoad(wy=-10.0, coord_system="global"))
        m.elements = [e]
        m.supports = {1: Support(1, ux=True, uy=True, rz=False),
                      2: Support(2, ux=False, uy=True, rz=False)}
        return m

    a = _patch_centroids(_shear_canvas(diagonal(reversed_order=False)))
    b = _patch_centroids(_shear_canvas(diagonal(reversed_order=True)))
    assert len(a) == len(b)
    for (ax, ay, ac), (bx, by, bc) in zip(a, b):
        assert ac == bc
        assert ax == pytest.approx(bx, abs=1e-6)
        assert ay == pytest.approx(by, abs=1e-6)

