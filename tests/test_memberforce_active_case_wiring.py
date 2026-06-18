"""Regression: active-case-aware member-load N/V/M reconstruction + GUI wiring.

Reproduces the reported bug — a beam carrying member loads in two different
cases (`KAPLAMA`, `KAR`) showed correct `SUM_ALL` diagrams but corrupted
*individual case* diagrams (a one-direction "yamuk"/trapezoid instead of the
expected beam parabola). The root cause: the in-span reconstruction read raw
`elem.member_loads` (all cases) while `f_local` was the single-case result.

The fix routes every diagram surface through
``effective_member_loads(elem, active_case, load_combinations)`` so the span
load matches the displayed result. This file pins both the pure-helper path
and the real GUI path (MainWindow → canvas / station export / hover), because
a passing helper test alone would not catch a wiring regression.
"""

from __future__ import annotations

import os
import re

import pytest

from structural_analysis.model import (
    StructuralModel, Node, Support, UniformDistributedLoad,
    LoadCase, LoadCombination,
)
from structural_analysis.element import FrameElement2D
from structural_analysis.main import run_multi_case_analysis
from structural_analysis.gui_qt.element_graphics import (
    sample_internal_force, effective_member_loads,
)


# ── Part A — pure reconstruction path (KAPLAMA / KAR) ────────────────────


def _ss_two_case_beam(qD=-20.0, qL=-15.0, L=6.0,
                      caseD="KAPLAMA", caseL="KAR"):
    m = StructuralModel(title="two-case beam")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    e = FrameElement2D(1, 1, 2, E=2.0e8, A=0.02, I=0.08)
    e.member_loads = [
        UniformDistributedLoad(wy=qD, load_case=caseD),
        UniformDistributedLoad(wy=qL, load_case=caseL),
    ]
    m.elements = [e]
    m.supports = {1: Support(1, ux=True, uy=True, rz=False),
                  2: Support(2, ux=False, uy=True, rz=False)}
    m.load_cases[caseD] = LoadCase(name=caseD)
    m.load_cases[caseL] = LoadCase(name=caseL)
    m.load_combinations["COMB"] = LoadCombination(
        name="COMB", terms={caseD: 1.0, caseL: 1.0})
    return m, e, L


def _moment(e, ni, nj, res, active, combos, n=7):
    eff = effective_member_loads(e, active, combos)
    f = res.member_results[e.id]["f_local"]
    _, ms = sample_internal_force(e, ni, nj, list(f), "moment",
                                  n_samples=n, member_loads=eff)
    return ms


def test_kaplama_view_excludes_kar():
    m, e, L = _ss_two_case_beam()
    mc = run_multi_case_analysis(m, verbose=False, cases=["KAPLAMA", "KAR"])
    ni, nj = m.nodes[1], m.nodes[2]
    ms = _moment(e, ni, nj, mc.cases["KAPLAMA"], "KAPLAMA",
                 m.load_combinations)
    # Only the −20 KAPLAMA UDL → midspan wL²/8 = 90, not 157.5 (both).
    assert ms[len(ms) // 2] == pytest.approx(20.0 * L ** 2 / 8.0, rel=1e-6)
    assert ms[len(ms) // 2] != pytest.approx(35.0 * L ** 2 / 8.0, rel=1e-3)


def test_kar_view_excludes_kaplama():
    m, e, L = _ss_two_case_beam()
    mc = run_multi_case_analysis(m, verbose=False, cases=["KAPLAMA", "KAR"])
    ni, nj = m.nodes[1], m.nodes[2]
    ms = _moment(e, ni, nj, mc.cases["KAR"], "KAR", m.load_combinations)
    assert ms[len(ms) // 2] == pytest.approx(15.0 * L ** 2 / 8.0, rel=1e-6)


def test_individual_case_shape_is_parabola_not_yamuk():
    """The bug produced an asymmetric one-direction trapezoid. A correct
    simply-supported UDL diagram is a symmetric concave-down parabola with
    zero ends and a strict rise-to-mid / fall-after-mid."""
    m, e, L = _ss_two_case_beam()
    mc = run_multi_case_analysis(m, verbose=False, cases=["KAPLAMA", "KAR"])
    ni, nj = m.nodes[1], m.nodes[2]
    ms = _moment(e, ni, nj, mc.cases["KAPLAMA"], "KAPLAMA",
                 m.load_combinations)
    n = len(ms)
    mid = n // 2
    assert ms[0] == pytest.approx(0.0, abs=1e-6)
    assert ms[-1] == pytest.approx(0.0, abs=1e-6)
    assert ms[mid] == max(ms) and ms[mid] > 0
    for a, b in zip(ms[:mid], ms[1:mid + 1]):
        assert b > a + 1e-9            # strict rise to midspan
    for a, b in zip(ms[mid:-1], ms[mid + 1:]):
        assert b < a - 1e-9            # strict fall after midspan
    # symmetric (a yamuk is not)
    for i in range(mid):
        assert ms[i] == pytest.approx(ms[-1 - i], abs=1e-6)


def test_same_case_two_udls_equal_one_summed_udl():
    # Both UDLs in the SAME case must superpose within that case.
    m1, e1, L = _ss_two_case_beam(qD=-20.0, qL=-15.0,
                                  caseD="KAPLAMA", caseL="KAPLAMA")
    mc1 = run_multi_case_analysis(m1, verbose=False, cases=["KAPLAMA"])
    ni, nj = m1.nodes[1], m1.nodes[2]
    ms_pair = _moment(e1, ni, nj, mc1.cases["KAPLAMA"], "KAPLAMA",
                      m1.load_combinations)

    m2 = StructuralModel(title="summed")
    m2.nodes = {1: Node(1, 0, 0), 2: Node(2, L, 0)}
    e2 = FrameElement2D(1, 1, 2, E=2e8, A=0.02, I=0.08)
    e2.member_loads = [UniformDistributedLoad(wy=-35.0)]
    m2.elements = [e2]
    m2.supports = m1.supports
    from structural_analysis.main import run_analysis
    r2 = run_analysis(m2, verbose=False)
    _, ms_sum = sample_internal_force(e2, m2.nodes[1], m2.nodes[2],
                                      list(r2.member_results[1]["f_local"]),
                                      "moment", n_samples=7)
    assert ms_pair == pytest.approx(ms_sum, rel=1e-9, abs=1e-9)


def test_sum_all_equals_both_and_comb_is_superposition():
    m, e, L = _ss_two_case_beam()
    mc = run_multi_case_analysis(m, verbose=False, cases=["KAPLAMA", "KAR"])
    ni, nj = m.nodes[1], m.nodes[2]
    ms_k = _moment(e, ni, nj, mc.cases["KAPLAMA"], "KAPLAMA",
                   m.load_combinations)
    ms_kar = _moment(e, ni, nj, mc.cases["KAR"], "KAR", m.load_combinations)
    ms_sum = _moment(e, ni, nj, mc.sum_all(), "SUM_ALL", m.load_combinations)
    comb = mc.combination({"KAPLAMA": 1.0, "KAR": 1.0}, name="COMB")
    ms_comb = _moment(e, ni, nj, comb, "COMB", m.load_combinations)
    # SUM_ALL == both summed (midspan 157.5).
    assert ms_sum[len(ms_sum) // 2] == pytest.approx(
        35.0 * L ** 2 / 8.0, rel=1e-6)
    # COMB == station-by-station superposition of the two solved cases.
    for c, a, b in zip(ms_comb, ms_k, ms_kar):
        assert c == pytest.approx(a + b, rel=1e-9, abs=1e-9)
    for c, sm in zip(ms_comb, ms_sum):
        assert c == pytest.approx(sm, rel=1e-9, abs=1e-9)


# ── Part B — end-to-end GUI wiring (portal frame via MainWindow) ─────────

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QSettings  # noqa: E402
    from PyQt6.QtWidgets import QApplication  # noqa: E402
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 unavailable: {exc}", allow_module_level=True)

from structural_analysis.gui_qt.app import MainWindow  # noqa: E402
from structural_analysis.gui_qt.canvas import HitResult  # noqa: E402
from structural_analysis.model import Material, Section  # noqa: E402


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


def _portal(w: MainWindow):
    """2-column (4 m) + beam (6 m) portal; beam carries a −20 DEFAULT UDL
    and a −15 LIVE UDL, with a 1.0D+1.0L combination. Solved."""
    m = w._model
    m.materials[1] = Material(id=1, name="C", E=2.0e8, density=0.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.02, I=8.0e-4, depth=0.3)
    m.nodes = {1: Node(1, 0, 0), 2: Node(2, 6, 0),
               3: Node(3, 0, 4), 4: Node(4, 6, 4)}
    beam = FrameElement2D(id=3, node_i=3, node_j=4, E=2e8,
                          A=0.02, I=8e-4, section_id=1)
    beam.member_loads = [
        UniformDistributedLoad(wy=-20.0, load_case="DEFAULT"),
        UniformDistributedLoad(wy=-15.0, load_case="LIVE"),
    ]
    m.elements = [
        FrameElement2D(id=1, node_i=1, node_j=3, E=2e8,
                       A=0.02, I=8e-4, section_id=1),
        FrameElement2D(id=2, node_i=2, node_j=4, E=2e8,
                       A=0.02, I=8e-4, section_id=1),
        beam,
    ]
    m.supports = {1: Support(node_id=1, ux=True, uy=True, rz=True),
                  2: Support(node_id=2, ux=True, uy=True, rz=True)}
    m.load_cases["DEFAULT"] = LoadCase(name="DEFAULT")
    m.load_cases["LIVE"] = LoadCase(name="LIVE")
    m.load_combinations["COMB"] = LoadCombination(
        name="COMB", terms={"DEFAULT": 1.0, "LIVE": 1.0})
    w._do_solve()
    return beam


def _beam_moment_via_canvas_path(w, beam, active):
    """Replicate exactly what canvas._draw_diagrams computes for the beam:
    effective loads from the canvas's active case + the model combinations,
    sampled through the shared helper."""
    w._active_case = active
    w._push_active_case_to_canvas()
    eff = effective_member_loads(
        beam, w.canvas._active_case, w._model.load_combinations)
    f = w._result.member_results[3]["f_local"]
    ni, nj = w._model.nodes[3], w._model.nodes[4]
    _, ms = sample_internal_force(beam, ni, nj, list(f), "moment",
                                  n_samples=7, member_loads=eff)
    return ms


def test_active_case_is_pushed_to_canvas(qt_app):
    w = MainWindow()
    _portal(w)
    for active in ("DEFAULT", "LIVE", "SUM_ALL", "COMB"):
        w._active_case = active
        w._push_active_case_to_canvas()
        assert w.canvas._active_case == active


def test_portal_default_view_excludes_live(qt_app):
    w = MainWindow()
    beam = _portal(w)
    ms_def = _beam_moment_via_canvas_path(w, beam, "DEFAULT")
    ms_sum = _beam_moment_via_canvas_path(w, beam, "SUM_ALL")
    mid = len(ms_def) // 2
    # DEFAULT midspan must be strictly less than SUM_ALL (LIVE excluded).
    assert ms_def[mid] < ms_sum[mid] - 1.0
    # Symmetric fixed-fixed beam curve: hogging ends, sagging mid — a proper
    # curve, not a one-direction yamuk.
    for i in range(mid):
        assert ms_def[i] == pytest.approx(ms_def[-1 - i], abs=1e-6)
    assert ms_def[0] < 0 < ms_def[mid]      # hogging ends, sagging midspan


def test_portal_live_view_excludes_default(qt_app):
    w = MainWindow()
    beam = _portal(w)
    ms_live = _beam_moment_via_canvas_path(w, beam, "LIVE")
    ms_def = _beam_moment_via_canvas_path(w, beam, "DEFAULT")
    mid = len(ms_live) // 2
    # LIVE (−15) midspan must be smaller than DEFAULT (−20) midspan.
    assert ms_live[mid] < ms_def[mid] - 1.0
    for i in range(mid):
        assert ms_live[i] == pytest.approx(ms_live[-1 - i], abs=1e-6)


def test_portal_comb_is_station_by_station_superposition(qt_app):
    w = MainWindow()
    beam = _portal(w)
    ms_def = _beam_moment_via_canvas_path(w, beam, "DEFAULT")
    ms_live = _beam_moment_via_canvas_path(w, beam, "LIVE")
    ms_comb = _beam_moment_via_canvas_path(w, beam, "COMB")
    ms_sum = _beam_moment_via_canvas_path(w, beam, "SUM_ALL")
    for c, a, b in zip(ms_comb, ms_def, ms_live):
        assert c == pytest.approx(a + b, rel=1e-9, abs=1e-6)
    for c, sm in zip(ms_comb, ms_sum):
        assert c == pytest.approx(sm, rel=1e-9, abs=1e-6)


def test_station_export_uses_active_case(qt_app, tmp_path, monkeypatch):
    import csv
    w = MainWindow()
    beam = _portal(w)
    w._active_case = "DEFAULT"
    w._push_active_case_to_canvas()
    out = tmp_path / "stations.csv"
    import structural_analysis.gui_qt.app as appmod
    monkeypatch.setattr(
        appmod.QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(out), "CSV (*.csv)"))
    w._export_station_results()
    with open(out, newline="", encoding="utf-8") as fh:
        rows = [r for r in list(csv.reader(fh))[1:] if r[0] == "3"]
    csv_m = [float(r[4]) for r in rows]
    # Compare to the DEFAULT-only canvas-path sampling at the same density.
    ni, nj = w._model.nodes[3], w._model.nodes[4]
    eff = effective_member_loads(beam, "DEFAULT", w._model.load_combinations)
    f = w._result.member_results[3]["f_local"]
    _, ms = sample_internal_force(beam, ni, nj, list(f), "moment",
                                  member_loads=eff,
                                  split_discontinuities=True)
    assert len(csv_m) == len(ms)
    for got, exp in zip(csv_m, ms):
        # CSV stores 6 significant figures (".6g"), so compare at that scale.
        assert got == pytest.approx(exp, rel=1e-5, abs=1e-4)
    # And it must NOT match the all-load (SUM_ALL) midspan.
    eff_all = effective_member_loads(beam, "SUM_ALL",
                                     w._model.load_combinations)
    _, ms_all = sample_internal_force(beam, ni, nj, list(f), "moment",
                                      member_loads=eff_all,
                                      split_discontinuities=True)
    assert max(csv_m) != pytest.approx(max(ms_all), rel=1e-3)


def test_hover_readout_uses_active_case(qt_app):
    """The status-bar hover read-out at beam midspan must report the
    active case's moment, not the all-load value."""
    w = MainWindow()
    _portal(w)
    w.canvas.diagram_kind = "moment"
    hit = HitResult(x=3.0, y=4.0, element_id=3)   # beam midspan

    def hover_value(active):
        w._active_case = active
        w._push_active_case_to_canvas()
        txt = w._diagram_value_text_for_hit(hit)
        assert txt is not None
        m = re.search(r"=\s*([+-]?\d+\.?\d*)", txt)
        assert m, txt
        return float(m.group(1))

    v_def = hover_value("DEFAULT")
    v_sum = hover_value("SUM_ALL")
    # DEFAULT-only midspan (~45.2) must differ from SUM_ALL (~79.1).
    assert abs(v_def - v_sum) > 10.0
    assert v_sum > v_def       # all loads → larger sagging midspan
