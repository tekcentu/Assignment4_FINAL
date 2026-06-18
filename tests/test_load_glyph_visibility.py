"""Canvas load-glyph visibility / filtering by active case.

Covers the display-only feature where the existing case/result selector
controls which nodal + member load glyphs are drawn — before AND after a
solve. No solver / model / I/O math is exercised or changed here.

Two layers are tested:

* the Qt-free helpers in ``gui_common.load_view`` (precise filtering /
  combination-factoring logic), and
* the PyQt6 canvas integration (arrow counts on redraw, stale-arrow
  freedom, pre/post-solve result behaviour).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from structural_analysis.gui_common.load_view import (  # noqa: E402
    ACTIVE_CASE,
    ALL,
    HIDE,
    visible_member_loads,
    visible_nodal_loads,
)
from structural_analysis.element import FrameElement2D  # noqa: E402
from structural_analysis.model import (  # noqa: E402
    LoadCase,
    LoadCombination,
    Material,
    NodalLoad,
    Node,
    Section,
    StructuralModel,
    Support,
    UniformDistributedLoad,
)


# ── shared portal-frame fixture (DEFAULT + LIVE nodal & member loads) ──


def _portal_frame() -> StructuralModel:
    """4 m columns, 6 m beam. DEFAULT + LIVE each carry one nodal load
    and one beam UDL so every visibility branch has something to show."""
    m = StructuralModel(title="portal")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3)
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 0.0, 4.0),
        3: Node(3, 6.0, 4.0),
        4: Node(4, 6.0, 0.0),
    }
    col_l = FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, I=8e-5, section_id=1)
    beam = FrameElement2D(id=2, node_i=2, node_j=3, E=2.1e8, A=0.02, I=8e-5, section_id=1)
    col_r = FrameElement2D(id=3, node_i=4, node_j=3, E=2.1e8, A=0.02, I=8e-5, section_id=1)
    m.elements = [col_l, beam, col_r]
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[4] = Support(node_id=4, ux=True, uy=True, rz=True)
    # DEFAULT loads.
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEFAULT"))
    beam.member_loads.append(UniformDistributedLoad(wy=-5.0, load_case="DEFAULT"))
    # LIVE loads.
    m.nodal_loads.append(NodalLoad(node_id=3, fy=-7.0, load_case="LIVE"))
    beam.member_loads.append(UniformDistributedLoad(wy=-3.0, load_case="LIVE"))
    m.load_cases["LIVE"] = LoadCase(name="LIVE")
    return m


def _beam(m: StructuralModel) -> FrameElement2D:
    return m.elements[1]


# ── pure helper: single-case filtering ───────────────────────────────


def test_helper_active_default_shows_only_default():
    m = _portal_frame()
    nodal = visible_nodal_loads(m, "DEFAULT", m.load_combinations, ACTIVE_CASE)
    member = visible_member_loads(_beam(m), "DEFAULT", m.load_combinations, ACTIVE_CASE)
    assert [ld.load_case for ld in nodal] == ["DEFAULT"]
    assert [ld.load_case for ld in member] == ["DEFAULT"]
    assert nodal[0].fy == -10.0
    assert member[0].wy == -5.0


def test_helper_active_live_shows_only_live():
    m = _portal_frame()
    nodal = visible_nodal_loads(m, "LIVE", m.load_combinations, ACTIVE_CASE)
    member = visible_member_loads(_beam(m), "LIVE", m.load_combinations, ACTIVE_CASE)
    assert [ld.load_case for ld in nodal] == ["LIVE"]
    assert [ld.load_case for ld in member] == ["LIVE"]
    assert nodal[0].fy == -7.0
    assert member[0].wy == -3.0


def test_helper_all_and_sum_all_show_everything():
    m = _portal_frame()
    # mode=ALL ignores the active case.
    assert len(visible_nodal_loads(m, "DEFAULT", m.load_combinations, ALL)) == 2
    assert len(visible_member_loads(_beam(m), "DEFAULT", m.load_combinations, ALL)) == 2
    # SUM_ALL selection under the default (active_case) mode shows all too.
    assert len(visible_nodal_loads(m, "SUM_ALL", m.load_combinations, ACTIVE_CASE)) == 2
    assert len(visible_member_loads(_beam(m), "SUM_ALL", m.load_combinations, ACTIVE_CASE)) == 2


def test_helper_hide_shows_nothing():
    m = _portal_frame()
    assert visible_nodal_loads(m, "DEFAULT", m.load_combinations, HIDE) == []
    assert visible_member_loads(_beam(m), "DEFAULT", m.load_combinations, HIDE) == []


# ── pure helper: combination factoring ───────────────────────────────


def test_helper_combination_factors_and_aggregates():
    m = _portal_frame()
    m.load_combinations["COMB"] = LoadCombination(
        name="COMB", terms={"DEFAULT": 1.2, "LIVE": 1.6},
    )
    nodal = visible_nodal_loads(m, "COMB", m.load_combinations, ACTIVE_CASE)
    member = visible_member_loads(_beam(m), "COMB", m.load_combinations, ACTIVE_CASE)
    # Both referenced cases contribute, each factored by its coefficient.
    by_case_n = {ld.load_case: ld.fy for ld in nodal}
    by_case_m = {ld.load_case: ld.wy for ld in member}
    assert by_case_n["DEFAULT"] == pytest.approx(-10.0 * 1.2)
    assert by_case_n["LIVE"] == pytest.approx(-7.0 * 1.6)
    assert by_case_m["DEFAULT"] == pytest.approx(-5.0 * 1.2)
    assert by_case_m["LIVE"] == pytest.approx(-3.0 * 1.6)


def test_helper_combination_drops_unreferenced_case():
    m = _portal_frame()
    m.load_combinations["ONLY_LIVE"] = LoadCombination(
        name="ONLY_LIVE", terms={"LIVE": 1.5},
    )
    nodal = visible_nodal_loads(m, "ONLY_LIVE", m.load_combinations, ACTIVE_CASE)
    assert [ld.load_case for ld in nodal] == ["LIVE"]
    assert nodal[0].fy == pytest.approx(-7.0 * 1.5)


def test_helper_does_not_mutate_model_loads():
    """Combination factoring must produce copies — the stored model loads
    keep their original magnitudes (and identity for non-factored views)."""
    m = _portal_frame()
    m.load_combinations["COMB"] = LoadCombination(
        name="COMB", terms={"DEFAULT": 2.0, "LIVE": 2.0},
    )
    orig_nodal = list(m.nodal_loads)
    orig_member = list(_beam(m).member_loads)
    visible_nodal_loads(m, "COMB", m.load_combinations, ACTIVE_CASE)
    visible_member_loads(_beam(m), "COMB", m.load_combinations, ACTIVE_CASE)
    assert [ld.fy for ld in m.nodal_loads] == [ld.fy for ld in orig_nodal]
    assert [ld.wy for ld in _beam(m).member_loads] == [ld.wy for ld in orig_member]
    # Plain-case / all views return the very same objects (no copying).
    same = visible_nodal_loads(m, "DEFAULT", m.load_combinations, ALL)
    assert all(a is b for a, b in zip(same, m.nodal_loads))


# ── canvas integration ───────────────────────────────────────────────

try:
    from PyQt6.QtWidgets import QApplication  # noqa: E402
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 unavailable: {exc}", allow_module_level=True)

import matplotlib.colors as mcolors  # noqa: E402

from structural_analysis.gui_qt.app import MainWindow  # noqa: E402

_GREEN = mcolors.to_rgba("#2ca02c")   # nodal-load arrows
_PURPLE = mcolors.to_rgba("#9467bd")  # member-load arrows


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _count_arrows(canvas, color) -> int:
    """Count load-arrow glyphs of a given colour. Arrows are drawn via
    ``ax.annotate(... arrowprops=...)``, so each lives as the
    ``arrow_patch`` of an Annotation in ``ax.texts``."""
    target = tuple(round(c, 3) for c in color[:3])
    n = 0
    for ann in list(canvas.ax.texts):
        patch = getattr(ann, "arrow_patch", None)
        if patch is None:
            continue
        ec = patch.get_edgecolor()
        if tuple(round(c, 3) for c in ec[:3]) == target:
            n += 1
    return n


def _install_portal(w: MainWindow) -> None:
    w._model = _portal_frame()
    w.canvas._model = lambda: w._model
    w._refresh_case_selector_combo()


def test_canvas_before_solve_default_shows_only_default_loads(qt_app):
    w = MainWindow()
    _install_portal(w)
    w.canvas.set_active_case("DEFAULT")
    w.canvas.redraw()
    # One DEFAULT nodal (fy) arrow + one DEFAULT UDL strip (6 arrows).
    assert _count_arrows(w.canvas, _GREEN) == 1
    assert _count_arrows(w.canvas, _PURPLE) == 6


def test_canvas_before_solve_live_shows_only_live_loads(qt_app):
    w = MainWindow()
    _install_portal(w)
    w.canvas.set_active_case("LIVE")
    w.canvas.redraw()
    assert _count_arrows(w.canvas, _GREEN) == 1
    assert _count_arrows(w.canvas, _PURPLE) == 6


def test_canvas_before_solve_sum_all_shows_all_loads(qt_app):
    w = MainWindow()
    _install_portal(w)
    w.canvas.set_active_case("SUM_ALL")
    w.canvas.redraw()
    # Both cases' nodal loads (2 green) + both UDL strips (12 purple).
    assert _count_arrows(w.canvas, _GREEN) == 2
    assert _count_arrows(w.canvas, _PURPLE) == 12


def test_canvas_switching_case_leaves_no_stale_arrows(qt_app):
    w = MainWindow()
    _install_portal(w)
    w.canvas.set_active_case("DEFAULT")
    w.canvas.redraw()
    assert _count_arrows(w.canvas, _PURPLE) == 6
    # Switch — count must stay at one strip, not accumulate to 12.
    w.canvas.set_active_case("LIVE")  # triggers its own redraw
    assert _count_arrows(w.canvas, _PURPLE) == 6
    w.canvas.set_active_case("SUM_ALL")
    assert _count_arrows(w.canvas, _PURPLE) == 12
    w.canvas.set_active_case("DEFAULT")
    assert _count_arrows(w.canvas, _PURPLE) == 6


def test_canvas_loads_only_off_shows_all_regardless_of_case(qt_app):
    w = MainWindow()
    _install_portal(w)
    w.canvas.set_active_case("DEFAULT")
    w.canvas.set_active_case_loads_only(False)  # → "all" mode
    assert _count_arrows(w.canvas, _PURPLE) == 12
    assert _count_arrows(w.canvas, _GREEN) == 2


def test_canvas_no_result_pre_solve_does_not_draw_diagrams(qt_app):
    """Selecting a case with no solved result must not produce N/V/M /
    reactions / deformed overlays — only load glyphs update."""
    w = MainWindow()
    _install_portal(w)
    w.canvas.set_active_case("DEFAULT")
    w.canvas.redraw()
    assert w.canvas._result is None
    # Load glyphs are present though.
    assert _count_arrows(w.canvas, _PURPLE) == 6


def test_canvas_after_solve_keeps_result_and_filters_loads(qt_app):
    w = MainWindow()
    _install_portal(w)
    # Reactions share the member-load colour; turn them off so the purple
    # count isolates member-load glyphs (reactions are a separate feature).
    w.canvas.show_reactions = False
    w._refresh_case_selector_combo()
    w._do_solve()
    qt_app.processEvents()
    # DEFAULT: result present, only DEFAULT loads shown.
    w._active_case = "DEFAULT"
    w._push_active_case_to_canvas()
    assert w.canvas._result is not None
    assert _count_arrows(w.canvas, _PURPLE) == 6
    assert _count_arrows(w.canvas, _GREEN) == 1
    # LIVE: distinct result, only LIVE loads shown.
    w._active_case = "LIVE"
    w._push_active_case_to_canvas()
    assert w.canvas._result is not None
    assert _count_arrows(w.canvas, _PURPLE) == 6
    assert _count_arrows(w.canvas, _GREEN) == 1


def test_canvas_combination_shows_factored_effective_glyphs(qt_app):
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd

    w = MainWindow()
    _install_portal(w)
    w.canvas.show_reactions = False  # purple is shared with reaction arrows
    w.execute(AddLoadCombinationCmd(name="COMB", terms={"DEFAULT": 1.2, "LIVE": 1.6}))
    w._do_solve()
    qt_app.processEvents()
    w._active_case = "COMB"
    w._push_active_case_to_canvas()
    # Both referenced cases' glyphs are drawn (factored): 2 nodal + 12 udl.
    assert _count_arrows(w.canvas, _GREEN) == 2
    assert _count_arrows(w.canvas, _PURPLE) == 12
    # The factored magnitudes flow through the helper the canvas uses.
    nodal = visible_nodal_loads(w._model, "COMB", w._model.load_combinations, ACTIVE_CASE)
    by_case = {ld.load_case: ld.fy for ld in nodal}
    assert by_case["DEFAULT"] == pytest.approx(-10.0 * 1.2)
    assert by_case["LIVE"] == pytest.approx(-7.0 * 1.6)


def test_visibility_does_not_change_numeric_results(qt_app):
    """Filtering is display-only: solved member forces / reactions match a
    fresh headless multi-case solve of the same model regardless of the
    canvas selection."""
    import numpy as np

    from structural_analysis.main import run_multi_case_analysis

    w = MainWindow()
    _install_portal(w)
    w._do_solve()
    qt_app.processEvents()
    # Switch selection around (drives the load-visibility path).
    for case in ("LIVE", "SUM_ALL", "DEFAULT"):
        w.canvas.set_active_case(case)
        w.canvas.redraw()
    # Headless reference solve — must equal the GUI's stored case result.
    ref = run_multi_case_analysis(_portal_frame(), verbose=False)
    gui_default = w._multi_result.get("DEFAULT")
    ref_default = ref.get("DEFAULT")
    for elem in _portal_frame().elements:
        g = gui_default.member_results[elem.id]["f_local"]
        r = ref_default.member_results[elem.id]["f_local"]
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), atol=1e-9)
