"""Tests for active-case-aware member-load rendering on the canvas.

The N/V/M correctness fix made diagrams / hover / station export use
``effective_member_loads`` (case-filtered, combination-factored). This
feature brings the canvas load *glyphs* into agreement:

* single active case → only that case's member loads are drawn;
* active combination → the factored *net* load is drawn (e.g.
  ``1.2·DEAD + 1.6·LIVE`` → a single ``-20 kN/m`` glyph);
* SUM_ALL / toggle-off → every assigned load is drawn (legacy view);
* labels stay in internal units (``kN`` / ``kN/m`` / ``kN·m``).

The pure aggregation helper is tested without Qt; the canvas behaviour is
verified offscreen by reading the purple member-load label artist.
"""

from __future__ import annotations

import copy
import os

import pytest

from structural_analysis.model import (
    UniformDistributedLoad, PointLoad,
)


# ── Pure aggregation helper (no Qt) ──────────────────────────────────────


def test_aggregate_sums_udls_of_same_coord_system():
    from structural_analysis.gui_qt.canvas import _aggregate_member_loads
    # Factored copies a combination would feed in: 1.2*(-10), 1.6*(-5).
    loads = [
        UniformDistributedLoad(wy=-12.0, coord_system="local"),
        UniformDistributedLoad(wy=-8.0, coord_system="local"),
    ]
    out = _aggregate_member_loads(loads)
    assert len(out) == 1
    assert isinstance(out[0], UniformDistributedLoad)
    assert out[0].wy == pytest.approx(-20.0)
    assert out[0].coord_system == "local"


def test_aggregate_keeps_different_coord_systems_separate():
    from structural_analysis.gui_qt.canvas import _aggregate_member_loads
    loads = [
        UniformDistributedLoad(wy=-10.0, coord_system="local"),
        UniformDistributedLoad(wy=-5.0, coord_system="gravity"),
    ]
    out = _aggregate_member_loads(loads)
    assert len(out) == 2
    css = {ml.coord_system for ml in out}
    assert css == {"local", "gravity"}


def test_aggregate_sums_point_loads_at_same_position():
    from structural_analysis.gui_qt.canvas import _aggregate_member_loads
    loads = [
        PointLoad(py=-6.0, a=2.0, coord_system="local"),
        PointLoad(py=-8.0, a=2.0, coord_system="local"),
        PointLoad(py=-3.0, a=4.0, coord_system="local"),
    ]
    out = _aggregate_member_loads(loads)
    by_a = {round(ml.a, 3): ml.py for ml in out}
    assert by_a[2.0] == pytest.approx(-14.0)
    assert by_a[4.0] == pytest.approx(-3.0)


def test_aggregate_drops_canceling_loads():
    from structural_analysis.gui_qt.canvas import _aggregate_member_loads
    out = _aggregate_member_loads([
        UniformDistributedLoad(wy=-10.0, coord_system="local"),
        UniformDistributedLoad(wy=10.0, coord_system="local"),
    ])
    assert out == []


# ── Canvas behaviour (offscreen Qt) ──────────────────────────────────────

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QSettings  # noqa: E402
    from PyQt6.QtWidgets import QApplication  # noqa: E402
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 unavailable: {exc}", allow_module_level=True)

from structural_analysis.element import FrameElement2D  # noqa: E402
from structural_analysis.gui_qt.app import MainWindow  # noqa: E402
from structural_analysis.model import (  # noqa: E402
    LoadCase, LoadCombination, Material, Node, Section, Support,
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


def _seed_two_case_beam(w: MainWindow) -> FrameElement2D:
    """Simply-supported beam: UDL -10 in DEAD, UDL -5 in LIVE, plus a
    1.2D+1.6L combination. Solved so every view is available."""
    m = w._model
    m.materials[1] = Material(id=1, name="C", E=2.0e8, density=0.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.02, I=0.08, depth=0.3)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(id=1, node_i=1, node_j=2, E=2.0e8,
                       A=0.02, I=0.08, section_id=1)
    e.member_loads.append(UniformDistributedLoad(wy=-10.0, load_case="DEAD"))
    e.member_loads.append(UniformDistributedLoad(wy=-5.0, load_case="LIVE"))
    m.elements = [e]
    m.supports = {1: Support(node_id=1, ux=True, uy=True, rz=False),
                  2: Support(node_id=2, ux=False, uy=True, rz=False)}
    m.load_cases["DEAD"] = LoadCase(name="DEAD")
    m.load_cases["LIVE"] = LoadCase(name="LIVE")
    m.load_combinations["ULS"] = LoadCombination(
        name="ULS", terms={"DEAD": 1.2, "LIVE": 1.6})
    w._do_solve()
    return e


def _member_load_label(w: MainWindow) -> str:
    """Concatenate every purple member-load label artist on the canvas."""
    w.canvas.redraw()
    parts = []
    for t in w.canvas.ax.texts:
        col = t.get_color()
        if col == "#9467bd" and "UDL" in t.get_text():
            parts.append(t.get_text())
    return " | ".join(parts)


def test_single_case_shows_only_that_cases_member_load(qt_app):
    w = MainWindow()
    _seed_two_case_beam(w)
    w.canvas.set_active_case("DEAD")
    w.canvas.set_active_case_loads_only(True)
    label = _member_load_label(w)
    assert "-10" in label          # DEAD UDL shown
    assert "-5" not in label        # LIVE UDL hidden
    assert "kN/m" in label          # explicit internal unit


def test_other_active_case_hides_first_cases_load(qt_app):
    w = MainWindow()
    _seed_two_case_beam(w)
    w.canvas.set_active_case("LIVE")
    w.canvas.set_active_case_loads_only(True)
    label = _member_load_label(w)
    assert "-5" in label            # LIVE shown
    assert "-10" not in label       # DEAD hidden


def test_combination_shows_factored_net_member_load(qt_app):
    w = MainWindow()
    _seed_two_case_beam(w)
    # 1.2*(-10) + 1.6*(-5) = -20 kN/m net.
    w.canvas.set_active_case("ULS")
    w.canvas.set_active_combination_cases({"DEAD", "LIVE"})
    w.canvas.set_active_case_loads_only(True)
    label = _member_load_label(w)
    assert "-20" in label
    assert "kN/m" in label
    # The unfactored constituents must NOT appear as separate glyphs.
    assert "-10" not in label
    assert label.count("UDL") == 1


def test_toggle_off_shows_all_member_loads(qt_app):
    w = MainWindow()
    _seed_two_case_beam(w)
    w.canvas.set_active_case("DEAD")
    w.canvas.set_active_case_loads_only(False)
    label = _member_load_label(w)
    assert "-10" in label and "-5" in label   # every assigned load drawn


def test_sum_all_shows_all_member_loads(qt_app):
    w = MainWindow()
    _seed_two_case_beam(w)
    w._active_case = "SUM_ALL"
    w._push_active_case_to_canvas()
    w.canvas.set_active_case_loads_only(True)
    label = _member_load_label(w)
    assert "-10" in label and "-5" in label   # unfactored all-load view


def test_labels_stay_internal_units_even_in_kip_preset(qt_app):
    w = MainWindow()
    _seed_two_case_beam(w)
    w.canvas.set_active_case("DEAD")
    w.canvas.set_active_case_loads_only(True)
    w._set_units_preset("kip_ft")
    label = _member_load_label(w)
    assert "kN/m" in label          # member loads never converted in V1
    assert "kip" not in label


def test_rendering_does_not_mutate_model(qt_app):
    w = MainWindow()
    e = _seed_two_case_beam(w)
    before = copy.deepcopy(e.member_loads)
    w.canvas.set_active_case("ULS")
    w.canvas.set_active_combination_cases({"DEAD", "LIVE"})
    w.canvas.set_active_case_loads_only(True)
    _member_load_label(w)
    w.canvas.set_active_case("DEAD")
    _member_load_label(w)
    after = e.member_loads
    assert len(after) == len(before)
    for a, b in zip(before, after):
        assert (a.wy, a.wx, a.load_case, a.coord_system) == \
            (b.wy, b.wx, b.load_case, b.coord_system)


def test_diagrams_unchanged_by_this_feature(qt_app):
    """Sanity: the correctness-fix diagram path still produces the
    factored midspan moment for the combination (this PR is display-only
    and must not perturb N/V/M)."""
    from structural_analysis.gui_qt.element_graphics import (
        sample_internal_force, effective_member_loads,
    )
    w = MainWindow()
    e = _seed_two_case_beam(w)
    w._active_case = "ULS"
    w._push_active_case_to_canvas()
    res = w._result
    ni, nj = w._model.nodes[1], w._model.nodes[2]
    eff = effective_member_loads(e, "ULS", w._model.load_combinations)
    _, ms = sample_internal_force(
        e, ni, nj, list(res.member_results[1]["f_local"]), "moment",
        n_samples=7, member_loads=eff)
    w_eff = 1.2 * 10.0 + 1.6 * 5.0   # 20 kN/m
    assert max(ms) == pytest.approx(w_eff * 6.0 ** 2 / 8.0, rel=1e-6)
