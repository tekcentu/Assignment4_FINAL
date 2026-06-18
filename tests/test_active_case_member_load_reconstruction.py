"""Regression tests for active-case-aware member-load N/V/M reconstruction."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication
except Exception as exc:  # noqa: BLE001
    QApplication = None
    PYQT_SKIP = f"PyQt6 unavailable: {exc}"
else:
    PYQT_SKIP = ""

from structural_analysis.element import FrameElement2D
from structural_analysis.gui_qt.element_graphics import internal_force_at, sample_internal_force
from structural_analysis.main import run_multi_case_analysis
from structural_analysis.model import (
    LoadCase,
    LoadCombination,
    Material,
    Node,
    Section,
    StructuralModel,
    Support,
    UniformDistributedLoad,
)
from structural_analysis.multi_case_result import SUM_ALL_KEY


def _portal_model(*, default_loads=(-20.0,), live_loads=(-15.0,)):
    m = StructuralModel(title="active case member load portal")
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 6.0, 0.0),
        3: Node(3, 0.0, 4.0),
        4: Node(4, 6.0, 4.0),
    }
    m.materials = {1: Material(id=1, name="steel", E=200_000_000.0, density=0.0)}
    m.sections = {1: Section(id=1, name="frame", material_id=1, A=0.08, I=0.03, depth=0.4)}
    m.elements = [
        FrameElement2D(id=1, node_i=1, node_j=3, E=200_000_000.0, A=0.08, I=0.03, section_id=1),
        FrameElement2D(id=2, node_i=3, node_j=4, E=200_000_000.0, A=0.08, I=0.03, section_id=1),
        FrameElement2D(id=3, node_i=2, node_j=4, E=200_000_000.0, A=0.08, I=0.03, section_id=1),
    ]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=True),
        2: Support(2, ux=True, uy=True, rz=True),
    }
    m.load_cases = {
        "DEFAULT": LoadCase("DEFAULT"),
        "LIVE": LoadCase("LIVE"),
        "KAPLAMA": LoadCase("KAPLAMA"),
        "KAR": LoadCase("KAR"),
    }
    m.load_combinations = {
        "COMB": LoadCombination("COMB", {"DEFAULT": 1.0, "LIVE": 1.0}),
    }
    beam = m.elements[1]
    for wy in default_loads:
        beam.member_loads.append(UniformDistributedLoad(wy=wy, load_case="DEFAULT"))
    for wy in live_loads:
        beam.member_loads.append(UniformDistributedLoad(wy=wy, load_case="LIVE"))
    # User-reported Turkish case names: same beam, different magnitudes.
    beam.member_loads.append(UniformDistributedLoad(wy=-20.0, load_case="KAPLAMA"))
    beam.member_loads.append(UniformDistributedLoad(wy=-6.0, load_case="KAR"))
    return m, beam


def _moment_stations(result, model, elem, n=21):
    ni = model.nodes[elem.node_i]
    nj = model.nodes[elem.node_j]
    f_local = result.member_results[elem.id]["f_local"]
    return sample_internal_force(
        elem,
        ni,
        nj,
        f_local,
        "moment",
        n_samples=n,
        member_loads=result.effective_member_loads.get(elem.id),
    )


def _shear_stations(result, model, elem, n=21):
    ni = model.nodes[elem.node_i]
    nj = model.nodes[elem.node_j]
    f_local = result.member_results[elem.id]["f_local"]
    return sample_internal_force(
        elem,
        ni,
        nj,
        f_local,
        "shear",
        n_samples=n,
        member_loads=result.effective_member_loads.get(elem.id),
    )


def _mid_moment(result, model, elem):
    ni = model.nodes[elem.node_i]
    nj = model.nodes[elem.node_j]
    f_local = result.member_results[elem.id]["f_local"]
    return internal_force_at(
        elem,
        ni,
        nj,
        f_local,
        "moment",
        3.0,
        member_loads=result.effective_member_loads.get(elem.id),
    )


def test_active_case_member_load_reconstruction_keeps_cases_separate_and_combines():
    model, beam = _portal_model(default_loads=(-20.0,), live_loads=(-15.0,))
    mc = run_multi_case_analysis(model, verbose=False, cases=["DEFAULT", "LIVE", "KAPLAMA", "KAR"])
    assert mc.status == "ok", mc.failed_cases

    default = mc.get("DEFAULT")
    live = mc.get("LIVE")
    kaplama = mc.get("KAPLAMA")
    kar = mc.get("KAR")
    sum_all = mc.get(SUM_ALL_KEY)
    comb = mc.combination({"DEFAULT": 1.0, "LIVE": 1.0}, name="COMB")
    assert default and live and kaplama and kar and sum_all and comb

    assert [ld.load_case for ld in default.effective_member_loads[beam.id]] == ["DEFAULT"]
    assert [ld.load_case for ld in live.effective_member_loads[beam.id]] == ["LIVE"]
    assert [ld.load_case for ld in kaplama.effective_member_loads[beam.id]] == ["KAPLAMA"]
    assert [ld.load_case for ld in kar.effective_member_loads[beam.id]] == ["KAR"]

    _xs_d, md = _moment_stations(default, model, beam)
    _xs_l, ml = _moment_stations(live, model, beam)
    _xs_c, mcmb = _moment_stations(comb, model, beam)
    for a, b, c in zip(md, ml, mcmb):
        assert c == pytest.approx(a + b, abs=1e-7)

    # SUM_ALL contains all four named case loads; the DEFAULT+LIVE combination
    # remains the station-by-station superposition of only its referenced cases.
    _xs_s, ms = _moment_stations(sum_all, model, beam)
    _xs_kap, mkap = _moment_stations(kaplama, model, beam)
    _xs_kar, mkar = _moment_stations(kar, model, beam)
    for s, a, b, c, d in zip(ms, md, ml, mkap, mkar):
        assert s == pytest.approx(a + b + c + d, abs=1e-7)

    # Individual UDL case diagrams are proper beam curves: end hogging,
    # midspan sagging, and shear changes sign. The bug produced one-direction
    # trapezoid/yamuk shapes when other raw member loads leaked in.
    for result in (default, live, kaplama, kar):
        _xs, moments = _moment_stations(result, model, beam)
        assert moments[0] < 0.0
        assert moments[len(moments) // 2] > 0.0
        assert moments[-1] < 0.0
        _xs, shears = _shear_stations(result, model, beam)
        assert max(shears) > 0.0
        assert min(shears) < 0.0

    # Before-fix reproduction proof: using the active DEFAULT f_local with raw
    # elem.member_loads (all cases) differs from the corrected active result.
    ni = model.nodes[beam.node_i]
    nj = model.nodes[beam.node_j]
    raw_leak_mid = internal_force_at(
        beam, ni, nj, default.member_results[beam.id]["f_local"], "moment", 3.0
    )
    assert raw_leak_mid != pytest.approx(_mid_moment(default, model, beam), abs=1e-6)


def test_same_case_two_udls_match_one_summed_udl():
    split_model, split_beam = _portal_model(default_loads=(-20.0, -15.0), live_loads=())
    summed_model, summed_beam = _portal_model(default_loads=(-35.0,), live_loads=())
    split = run_multi_case_analysis(split_model, verbose=False, cases=["DEFAULT"]).get("DEFAULT")
    summed = run_multi_case_analysis(summed_model, verbose=False, cases=["DEFAULT"]).get("DEFAULT")
    assert split and summed
    _xs1, m1 = _moment_stations(split, split_model, split_beam)
    _xs2, m2 = _moment_stations(summed, summed_model, summed_beam)
    _xs1v, v1 = _shear_stations(split, split_model, split_beam)
    _xs2v, v2 = _shear_stations(summed, summed_model, summed_beam)
    assert m1 == pytest.approx(m2, abs=1e-7)
    assert v1 == pytest.approx(v2, abs=1e-7)


@pytest.mark.skipif(QApplication is None, reason=PYQT_SKIP)
def test_canvas_hover_detail_and_station_export_share_active_result_path(tmp_path, monkeypatch):
    from structural_analysis.gui_qt.app import MainWindow
    from structural_analysis.gui_qt.dialogs import ElementDetailsDialog

    app = QApplication.instance() or QApplication([])
    _ = app
    win = MainWindow()
    model, beam = _portal_model(default_loads=(-20.0,), live_loads=(-15.0,))
    win._model = model
    win._multi_result = run_multi_case_analysis(model, verbose=False, cases=["DEFAULT", "LIVE"])
    win._active_case = "DEFAULT"
    win._push_active_case_to_canvas()

    default_mid = _mid_moment(win._result, model, beam)
    raw_mid = internal_force_at(
        beam,
        model.nodes[beam.node_i],
        model.nodes[beam.node_j],
        win._result.member_results[beam.id]["f_local"],
        "moment",
        3.0,
    )
    assert default_mid != pytest.approx(raw_mid, abs=1e-6)

    assert win.canvas._result is win._result
    assert win.canvas._result.effective_member_loads[beam.id][0].load_case == "DEFAULT"

    hit = type("Hit", (), {"element_id": beam.id, "x": 3.0, "y": 4.0})()
    win.canvas.diagram_kind = "moment"
    hover = win._diagram_value_text_for_hit(hit)
    assert hover is not None
    assert f"{default_mid:.3g}" in hover

    dlg = ElementDetailsDialog(win, model=model, elem_id=beam.id, result=win._result, multi_result=win._multi_result)
    assert dlg._effective_member_loads_ref[0].load_case == "DEFAULT"

    out = tmp_path / "stations.csv"
    monkeypatch.setattr(
        "structural_analysis.gui_qt.app.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(out), "CSV (*.csv)"),
    )
    win._export_station_results()
    text = out.read_text(encoding="utf-8")
    assert f"{default_mid:.6g}" in text
