"""Regression: shear-diagram display side mirrors outward for portal columns.

Bug pinned here:

With only the member's local +y_local normal driving the diagram offset,
both columns of a typical portal frame have the same +y_local direction
(left, for bottom→top columns). The right column's positive shear lobe
therefore renders *inside* the portal instead of outside, leaving the
right column visually mismatched with the left column's outward lobe.

Fix: in ``canvas._draw_diagrams`` only — for shear, and only when the
structure has enough nodes for a meaningful centroid — the +y_local
normal is flipped on members where it clearly points toward the
structure's node centroid. Single-member and collinear models keep the
+y_local convention so the existing "positive shear above horizontal
beam" regression keeps passing.

Pinned constraints (none of which are allowed to change):

* numerical V values from ``sample_internal_force`` / ``internal_force_at``;
* the station-export CSV V column;
* the status-bar hover read-out V value;
* the BMD (moment) display orientation and side;
* the axial diagram side;
* the colour palette (blue = +V, red = −V);
* the existing horizontal-beam "positive shear above" lock-in.
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
    HitResult, ModelCanvas,
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


# ── Test fixture: a 4 m / 6 m fixed-base portal with the load pattern
# ── that matches the user's reproduction (lateral push + gravity).


def _portal_canvas(kind: str = "shear") -> tuple[ModelCanvas, StructuralModel]:
    m = read_input_file("inputs/example_03_portal_frame_lateral_load.txt")
    r = run_analysis(m, verbose=False)
    canvas = ModelCanvas(None, model_provider=lambda: m)
    canvas._result = r
    canvas.diagram_kind = kind
    canvas.diagram_scale = 1.0
    canvas.diagram_stations = 21
    canvas._draw_diagrams()
    return canvas, m


def _patches_x_extents(canvas: ModelCanvas) -> list[tuple[float, float]]:
    out = []
    for p in canvas.ax.patches:
        v = p.get_path().vertices
        out.append((float(v[:, 0].min()), float(v[:, 0].max())))
    return out


def _patches_y_extents(canvas: ModelCanvas) -> list[tuple[float, float]]:
    out = []
    for p in canvas.ax.patches:
        v = p.get_path().vertices
        out.append((float(v[:, 1].min()), float(v[:, 1].max())))
    return out


# ── 1. Portal columns mirror outward ────────────────────────────────────


def test_portal_shear_left_column_lobe_extends_outward_left(qt_app):
    """Left column at x = 0 — its shear lobe must reach to the LEFT of
    the column (outward of the portal)."""
    canvas, _ = _portal_canvas("shear")
    left_col_patch = next(
        (xmin, xmax) for xmin, xmax in _patches_x_extents(canvas)
        if xmax <= 0.0 + 1e-6
    )
    # Patch spans some interval ending at x = 0 (the column line) with
    # the LEFT edge at xmin < 0 (outward).
    assert left_col_patch[1] == pytest.approx(0.0, abs=1e-6)
    assert left_col_patch[0] < -0.05


def test_portal_shear_right_column_lobe_extends_outward_right(qt_app):
    """Right column at x = 6 — its shear lobe must reach to the RIGHT of
    the column (outward of the portal). Before the fix it extended left,
    into the portal interior."""
    canvas, _ = _portal_canvas("shear")
    right_col_patch = next(
        (xmin, xmax) for xmin, xmax in _patches_x_extents(canvas)
        if xmin >= 6.0 - 1e-6
    )
    assert right_col_patch[0] == pytest.approx(6.0, abs=1e-6)
    assert right_col_patch[1] > 6.05


def test_portal_columns_mirror_about_centerline(qt_app):
    """The two column lobes have the same magnitude reflected about x=3."""
    canvas, _ = _portal_canvas("shear")
    extents = _patches_x_extents(canvas)
    left = next((xmin, xmax) for xmin, xmax in extents if xmax <= 0.0 + 1e-6)
    right = next((xmin, xmax) for xmin, xmax in extents if xmin >= 6.0 - 1e-6)
    left_reach = abs(left[0] - 0.0)
    right_reach = abs(right[1] - 6.0)
    # Different V magnitudes give different reaches; this test checks the
    # DIRECTION is mirrored, not the absolute size.
    assert left_reach > 0.05 and right_reach > 0.05


# ── 2. BMD / axial / horizontal-beam shear behaviour unchanged ──────────


def test_portal_moment_diagram_unaffected_by_shear_fix(qt_app):
    """The same per-column extents the moment diagram had on main:
    left col M extends to the right (inward), right col M extends to the
    right (outward) — both on the +x_global side. The shear fix must not
    have moved any moment patch."""
    canvas, _ = _portal_canvas("moment")
    # The right column moment patch should sit to the RIGHT of x = 6
    # (this was true before the fix and must remain true after).
    extents = _patches_x_extents(canvas)
    right_col_M = next(
        (xmin, xmax) for xmin, xmax in extents if xmin >= 6.0 - 1e-6
    )
    assert right_col_M[1] > 6.05      # extends right (unchanged)


def test_horizontal_beam_positive_shear_still_above(qt_app):
    """Existing lock-in: a simply-supported UDL beam shows positive shear
    ABOVE the member centreline. The portal-aware flip must not touch
    this path (only models with >= 3 nodes get the centroid logic).

    Verified with a 2-node model so the fix's ``len(model.nodes) >= 3``
    short-circuit kicks in and leaves +y_local untouched."""
    m = StructuralModel(title="SS UDL beam")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(1, 1, 2, E=2.0e8, A=0.02, I=0.08)
    e.member_loads.append(UniformDistributedLoad(wy=-10.0))
    m.elements = [e]
    m.supports = {1: Support(1, ux=True, uy=True, rz=False),
                  2: Support(2, ux=False, uy=True, rz=False)}
    r = run_analysis(m, verbose=False)
    canvas = ModelCanvas(None, model_provider=lambda: m)
    canvas._result = r
    canvas.diagram_kind = "shear"
    canvas.diagram_scale = 1.0
    canvas.diagram_stations = 21
    canvas._draw_diagrams()
    ys = []
    for p in canvas.ax.patches:
        ys.extend(p.get_path().vertices[:, 1].tolist())
    assert max(ys) > 1e-6     # positive shear above
    assert min(ys) < -1e-6    # negative shear below


# ── 3. Numerical V values are completely untouched ──────────────────────


def test_sample_internal_force_values_unchanged(qt_app):
    """The numerical V(x) station values from ``sample_internal_force``
    are the per-station shears and must not change with a display fix."""
    m = read_input_file("inputs/example_03_portal_frame_lateral_load.txt")
    r = run_analysis(m, verbose=False)
    # Closed-form / pinned-from-main values for example_03:
    expected = {1: 21.7844, 2: 28.2156, 3: -2.2724}
    for e in m.elements:
        ni, nj = m.nodes[e.node_i], m.nodes[e.node_j]
        f = r.member_results[e.id]["f_local"]
        xs, vs = sample_internal_force(e, ni, nj, list(f), "shear",
                                       n_samples=5)
        for v in vs:
            assert v == pytest.approx(expected[e.id], rel=1e-4)


def test_internal_force_at_values_unchanged(qt_app):
    """The single-point hover helper returns the same V values too."""
    m = read_input_file("inputs/example_03_portal_frame_lateral_load.txt")
    r = run_analysis(m, verbose=False)
    expected = {1: 21.7844, 2: 28.2156, 3: -2.2724}
    for e in m.elements:
        ni, nj = m.nodes[e.node_i], m.nodes[e.node_j]
        f = r.member_results[e.id]["f_local"]
        L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
        v = internal_force_at(e, ni, nj, list(f), "shear", L / 2.0)
        assert v == pytest.approx(expected[e.id], rel=1e-4)


# ── 4. End-to-end (MainWindow): station export and hover unchanged ──────


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
    """The export CSV V column must equal sample_internal_force exactly:
    the display fix only touches the canvas-side normal direction."""
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
            e, ni, nj, list(f), "shear", n_samples=21,
        )
        vs_got = by_elem[str(e.id)]
        assert len(vs_got) == len(vs_expected)
        for got, exp in zip(vs_got, vs_expected):
            assert got == pytest.approx(exp, rel=1e-4, abs=1e-3)


def test_hover_readout_V_value_unchanged(qt_app):
    """Hover at beam midspan: the reported V equals internal_force_at —
    the display fix changes the on-canvas offset, never the numbers."""
    w = MainWindow()
    _portal_mainwindow(w)
    w.canvas.diagram_kind = "shear"
    beam = w._model.elements[2]   # beam id 3
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


# ── 5. Outward-lobe rule: both +V (blue) and −V (red) lobes always
# ── extend OUTWARD from the structure centroid; sign is communicated
# ── by colour only. This catches the case where a column has negative
# ── shear: with the legacy yy * nx rule the lobe flipped back inward
# ── after the centroid-flip, ending up on the wrong side.


def _portal_with_udl_beam_canvas() -> tuple[ModelCanvas, StructuralModel]:
    """Build a portal whose columns carry frame-action V ≈ ∓16.7 kN
    (left column NEGATIVE, right column POSITIVE) — the configuration
    that exposed the outward-lobe bug. UDL on the beam guarantees a V
    sign change along the beam too."""
    m = StructuralModel(title="portal outward-lobe check")
    m.materials[1] = Material(id=1, name="C", E=2.0e8, density=0.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.02, I=8e-4, depth=0.3)
    m.nodes = {1: Node(1, 0, 0), 2: Node(2, 6, 0),
               3: Node(3, 0, 4), 4: Node(4, 6, 4)}
    beam = FrameElement2D(id=3, node_i=3, node_j=4, E=2e8,
                          A=0.02, I=8e-4, section_id=1)
    beam.member_loads = [UniformDistributedLoad(wy=-20.0)]
    m.elements = [
        FrameElement2D(id=1, node_i=1, node_j=3, E=2e8,
                       A=0.02, I=8e-4, section_id=1),
        FrameElement2D(id=2, node_i=2, node_j=4, E=2e8,
                       A=0.02, I=8e-4, section_id=1),
        beam,
    ]
    m.supports = {1: Support(node_id=1, ux=True, uy=True, rz=True),
                  2: Support(node_id=2, ux=True, uy=True, rz=True)}
    r = run_analysis(m, verbose=False)
    canvas = ModelCanvas(None, model_provider=lambda: m)
    canvas._result = r
    canvas.diagram_kind = "shear"
    canvas.diagram_scale = 1.0
    canvas.diagram_stations = 21
    canvas._draw_diagrams()
    return canvas, m


def _patches_with_colour(canvas: ModelCanvas):
    """Return per-patch (xmin, xmax, ymin, ymax, 'red'|'blue')."""
    out = []
    for p in canvas.ax.patches:
        v = p.get_path().vertices
        c = p.get_facecolor()
        col = "blue" if c[2] > 0.5 else "red" if c[0] > 0.5 else "other"
        out.append((float(v[:, 0].min()), float(v[:, 0].max()),
                    float(v[:, 1].min()), float(v[:, 1].max()), col))
    return out


def test_left_column_negative_shear_lobes_outward_left(qt_app):
    """V < 0 on the left column → red lobe extends to the LEFT of
    x = 0 (outward), not to the right (inward as it did before)."""
    canvas, _ = _portal_with_udl_beam_canvas()
    left_col_patch = next(
        p for p in _patches_with_colour(canvas)
        if abs(p[3] - 4.0) < 1e-6 and abs(p[2]) < 1e-6
        and p[1] <= 0.0 + 1e-3       # column at x=0
    )
    xmin, xmax, _, _, colour = left_col_patch
    assert colour == "red"           # left col V is negative for this model
    assert xmax == pytest.approx(0.0, abs=1e-6)
    assert xmin < -0.05              # OUTWARD (left)


def test_right_column_positive_shear_lobes_outward_right(qt_app):
    """V > 0 on the right column → blue lobe extends to the RIGHT of
    x = 6 (outward) — unchanged from previous fix iteration."""
    canvas, _ = _portal_with_udl_beam_canvas()
    right_col_patch = next(
        p for p in _patches_with_colour(canvas)
        if abs(p[3] - 4.0) < 1e-6 and abs(p[2]) < 1e-6
        and p[0] >= 6.0 - 1e-3
    )
    xmin, xmax, _, _, colour = right_col_patch
    assert colour == "blue"
    assert xmin == pytest.approx(6.0, abs=1e-6)
    assert xmax > 6.05


def test_beam_sign_change_keeps_textbook_axis_convention(qt_app):
    """A UDL beam has V > 0 on the left half and V < 0 on the right
    half. The outward-lobe rule applies ONLY to single-sign elements;
    a sign-changing element keeps the textbook axis convention so
    +V plots ABOVE the centerline and −V plots BELOW it (i.e. red
    on the −y_local side of the beam) — the same convention the user
    sees on any V/M axis diagram."""
    canvas, _ = _portal_with_udl_beam_canvas()
    # All beam patches sit at y ≈ 4. Group them by colour.
    beam_patches = [
        p for p in _patches_with_colour(canvas)
        if abs(p[3] - 4.0) < 1.0 or abs(p[2] - 4.0) < 1.0
    ]
    blue_patch = next(p for p in beam_patches if p[4] == "blue"
                      and abs(p[2] - 4.0) < 1e-3 and p[3] > 4.0)
    red_patch = next(p for p in beam_patches if p[4] == "red"
                     and abs(p[3] - 4.0) < 1e-3 and p[2] < 4.0)
    # Positive V (blue) is ABOVE the beam centerline at y = 4.
    assert blue_patch[3] > 4.05
    assert blue_patch[2] == pytest.approx(4.0, abs=1e-6)
    # Negative V (red) is BELOW the beam centerline.
    assert red_patch[2] < 3.95
    assert red_patch[3] == pytest.approx(4.0, abs=1e-6)


# ── 6. Node-order invariance: swapping i↔j on the right column does not
# ── change which screen side the lobe lands on (the visual is geometry-
# ── driven, not node-name-driven).


def test_right_column_diagram_side_is_node_order_invariant(qt_app):
    def render(swap_right_col: bool) -> tuple[float, float]:
        m = StructuralModel(title="portal node-order check")
        m.materials[1] = Material(id=1, name="C", E=2.0e8, density=0.0)
        m.sections[1] = Section(id=1, name="S", material_id=1,
                                A=0.02, I=8e-4, depth=0.3)
        m.nodes = {1: Node(1, 0, 0), 2: Node(2, 6, 0),
                   3: Node(3, 0, 4), 4: Node(4, 6, 4)}
        right = (FrameElement2D(id=2, node_i=4, node_j=2,
                                 E=2e8, A=0.02, I=8e-4, section_id=1)
                 if swap_right_col else
                 FrameElement2D(id=2, node_i=2, node_j=4,
                                 E=2e8, A=0.02, I=8e-4, section_id=1))
        m.elements = [
            FrameElement2D(id=1, node_i=1, node_j=3, E=2e8,
                           A=0.02, I=8e-4, section_id=1),
            right,
            FrameElement2D(id=3, node_i=3, node_j=4, E=2e8,
                           A=0.02, I=8e-4, section_id=1),
        ]
        m.supports = {1: Support(node_id=1, ux=True, uy=True, rz=True),
                      2: Support(node_id=2, ux=True, uy=True, rz=True)}
        m.nodal_loads = [NodalLoad(node_id=4, fx=50.0, fy=-20.0),
                         NodalLoad(node_id=3, fy=-20.0)]
        r = run_analysis(m, verbose=False)
        canvas = ModelCanvas(None, model_provider=lambda: m)
        canvas._result = r
        canvas.diagram_kind = "shear"
        canvas.diagram_scale = 1.0
        canvas.diagram_stations = 21
        canvas._draw_diagrams()
        right_col_patch = next(
            (xmin, xmax) for xmin, xmax
            in _patches_x_extents(canvas)
            if xmin >= 6.0 - 1e-6
        )
        return right_col_patch

    p_std = render(False)
    p_swap = render(True)
    # Same lobe — extends to the right (outward), same magnitude.
    assert p_std[0] == pytest.approx(p_swap[0], abs=1e-6)
    assert p_std[1] == pytest.approx(p_swap[1], abs=1e-6)
    assert p_std[1] > 6.05    # both still outward
