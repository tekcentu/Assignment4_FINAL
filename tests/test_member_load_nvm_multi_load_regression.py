"""Regression tests for member-load N/V/M reconstruction with multiple loads.

These pin the ``effective_member_loads`` fix against the multi-load scenarios
that historically corrupted individual case views (e.g. a moment diagram that
collapsed into a one-direction trapezoid). The math is already in
``sample_internal_force`` / ``effective_member_loads``; this file is the
named-scenario regression net.

Covers, per the brief:

* same-case two UDLs == one summed UDL;
* ``DEFAULT`` view excludes ``LIVE`` loads;
* ``LIVE`` view excludes ``DEFAULT`` loads;
* ``DEFAULT + LIVE`` combination == solved-DEFAULT + solved-LIVE;
* individual case moment under multiple loads is the correct symmetric
  parabola (not a one-direction trapezoid);
* station export samples the same corrected values as ``sample_internal_force``.
"""

from __future__ import annotations

import os

import pytest

from structural_analysis.model import (
    StructuralModel, Node, Support, UniformDistributedLoad,
    LoadCase, LoadCombination,
)
from structural_analysis.element import FrameElement2D
from structural_analysis.main import run_analysis, run_multi_case_analysis
from structural_analysis.gui_qt.element_graphics import (
    sample_internal_force, effective_member_loads,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


def _ss_beam(L: float = 6.0):
    """Simply-supported (pin/roller) horizontal beam, no loads yet."""
    m = StructuralModel(title="multi-load reconstruction")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    e = FrameElement2D(1, 1, 2, E=2.0e8, A=0.02, I=0.08)
    m.elements = [e]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=False),
        2: Support(2, ux=False, uy=True, rz=False),
    }
    return m, e, L


def _moment_for_view(model, e, *, view: str, combos: dict | None = None):
    """Solve ``model`` for whatever cases ``view`` needs, then sample the
    moment diagram for that view using the case-consistent effective loads.
    ``view`` is a real case name OR a key in ``combos``."""
    ni, nj = model.nodes[1], model.nodes[2]
    combos = combos or {}
    if view in combos:
        cases = sorted({c for c in combos[view].terms})
        mc = run_multi_case_analysis(model, verbose=False, cases=cases)
        res = mc.combination(combos[view].terms, name=view)
        eff = effective_member_loads(e, view, combos)
    else:
        cases = sorted({getattr(x, "load_case", "DEFAULT")
                        for x in e.member_loads})
        if cases == ["DEFAULT"]:
            res = run_analysis(model, verbose=False)
        else:
            mc = run_multi_case_analysis(model, verbose=False, cases=cases)
            res = mc.cases[view]
        eff = effective_member_loads(e, view, {})
    f = res.member_results[1]["f_local"]
    _, ms = sample_internal_force(
        e, ni, nj, list(f), "moment",
        n_samples=7, member_loads=eff, split_discontinuities=True,
    )
    return ms


# ── 1. Same-case two UDLs == one summed UDL ──────────────────────────────


def test_two_same_case_udls_equal_one_summed_udl():
    m1, e1, L = _ss_beam()
    e1.member_loads.append(UniformDistributedLoad(wy=-20.0))
    e1.member_loads.append(UniformDistributedLoad(wy=-15.0))

    m2, e2, _ = _ss_beam()
    e2.member_loads.append(UniformDistributedLoad(wy=-35.0))

    ms_pair = _moment_for_view(m1, e1, view="DEFAULT")
    ms_sum = _moment_for_view(m2, e2, view="DEFAULT")
    assert ms_pair == pytest.approx(ms_sum, rel=1e-9, abs=1e-9)
    assert ms_pair[len(ms_pair) // 2] == pytest.approx(
        35.0 * L ** 2 / 8.0, rel=1e-6)


# ── 2. DEFAULT view excludes LIVE loads ──────────────────────────────────


def test_default_view_excludes_live_loads():
    m, e, L = _ss_beam()
    e.member_loads.append(UniformDistributedLoad(wy=-20.0, load_case="DEFAULT"))
    e.member_loads.append(UniformDistributedLoad(wy=-15.0, load_case="LIVE"))
    m.load_cases["LIVE"] = LoadCase(name="LIVE")

    ms_def = _moment_for_view(m, e, view="DEFAULT")
    assert ms_def[len(ms_def) // 2] == pytest.approx(
        20.0 * L ** 2 / 8.0, rel=1e-6)
    # If the LIVE contribution had leaked, midspan would shift to 35·L²/8.
    assert ms_def[len(ms_def) // 2] != pytest.approx(
        35.0 * L ** 2 / 8.0, rel=1e-3)


# ── 3. LIVE view excludes DEFAULT loads ──────────────────────────────────


def test_live_view_excludes_default_loads():
    m, e, L = _ss_beam()
    e.member_loads.append(UniformDistributedLoad(wy=-20.0, load_case="DEFAULT"))
    e.member_loads.append(UniformDistributedLoad(wy=-15.0, load_case="LIVE"))
    m.load_cases["LIVE"] = LoadCase(name="LIVE")

    ms_live = _moment_for_view(m, e, view="LIVE")
    assert ms_live[len(ms_live) // 2] == pytest.approx(
        15.0 * L ** 2 / 8.0, rel=1e-6)


# ── 4. Combination == individual-case superposition ──────────────────────


def test_combination_equals_individual_superposition():
    m, e, L = _ss_beam()
    e.member_loads.append(UniformDistributedLoad(wy=-20.0, load_case="DEFAULT"))
    e.member_loads.append(UniformDistributedLoad(wy=-15.0, load_case="LIVE"))
    m.load_cases["LIVE"] = LoadCase(name="LIVE")
    combos = {"COMB": LoadCombination(
        name="COMB", terms={"DEFAULT": 1.0, "LIVE": 1.0})}

    ms_def = _moment_for_view(m, e, view="DEFAULT")
    ms_live = _moment_for_view(m, e, view="LIVE")
    ms_comb = _moment_for_view(m, e, view="COMB", combos=combos)

    # Linear superposition station-by-station.
    for c, d, lv in zip(ms_comb, ms_def, ms_live):
        assert c == pytest.approx(d + lv, rel=1e-9, abs=1e-9)
    # And the closed-form check.
    assert ms_comb[len(ms_comb) // 2] == pytest.approx(
        35.0 * L ** 2 / 8.0, rel=1e-6)


# ── 5. Individual case shape is a correct parabola, not a trapezoid ──────


def test_individual_case_moment_shape_is_correct_parabola_not_trapezoid():
    """Symptom this guards: a one-direction trapezoid would have a
    monotonically non-decreasing left half — a true sagging parabola
    rises strictly past the centre, then drops back to zero. Also
    proves both endpoints stay at 0 (pin/roller)."""
    m, e, L = _ss_beam()
    e.member_loads.append(UniformDistributedLoad(wy=-20.0, load_case="DEFAULT"))
    e.member_loads.append(UniformDistributedLoad(wy=-15.0, load_case="LIVE"))
    m.load_cases["LIVE"] = LoadCase(name="LIVE")

    ms = _moment_for_view(m, e, view="DEFAULT")
    n = len(ms)
    mid = n // 2
    # Pinned ends.
    assert ms[0] == pytest.approx(0.0, abs=1e-6)
    assert ms[-1] == pytest.approx(0.0, abs=1e-6)
    # Concave-down: midspan is the absolute peak.
    assert ms[mid] == max(ms)
    assert ms[mid] > 0.0
    # Strict rise on the left half, strict drop on the right half.
    for a, b in zip(ms[:mid], ms[1:mid + 1]):
        assert b > a + 1e-9
    for a, b in zip(ms[mid:-1], ms[mid + 1:]):
        assert b < a - 1e-9
    # Symmetric within float noise.
    for i in range(mid):
        assert ms[i] == pytest.approx(ms[-1 - i], abs=1e-6)


# ── 6. Station export matches sample_internal_force (Qt-offscreen) ───────


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QSettings  # noqa: E402
    from PyQt6.QtWidgets import QApplication  # noqa: E402
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 unavailable: {exc}", allow_module_level=True)

from structural_analysis.gui_qt.app import MainWindow  # noqa: E402
from structural_analysis.model import (  # noqa: E402
    Material, Section,
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


def _seed_two_same_case_udls(w: MainWindow) -> FrameElement2D:
    m = w._model
    m.materials[1] = Material(id=1, name="C", E=2.0e8, density=0.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.02, I=0.08, depth=0.3)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(id=1, node_i=1, node_j=2, E=2.0e8,
                       A=0.02, I=0.08, section_id=1)
    e.member_loads.append(UniformDistributedLoad(wy=-20.0))
    e.member_loads.append(UniformDistributedLoad(wy=-15.0))
    m.elements = [e]
    m.supports = {1: Support(node_id=1, ux=True, uy=True, rz=False),
                  2: Support(node_id=2, ux=False, uy=True, rz=False)}
    w._run_static_solve(active_only=False)
    return e


def test_station_export_matches_sample_internal_force_with_multiple_loads(
    qt_app, tmp_path, monkeypatch,
):
    """The exported M column must equal what ``sample_internal_force``
    returns for the same active view, even when the member has multiple
    same-case UDLs (the configuration that first surfaced as a buggy
    one-direction trapezoid)."""
    import csv
    w = MainWindow()
    e = _seed_two_same_case_udls(w)
    out = tmp_path / "stations.csv"
    import structural_analysis.gui_qt.app as appmod
    monkeypatch.setattr(
        appmod.QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(out), "CSV (*.csv)"),
    )
    w._export_station_results()

    with open(out, newline="", encoding="utf-8") as fh:
        csv_m = [float(r[4]) for r in list(csv.reader(fh))[1:]]
    ni, nj = w._model.nodes[1], w._model.nodes[2]
    eff = effective_member_loads(e, w._active_case, w._model.load_combinations)
    f = w._result.member_results[1]["f_local"]
    _, ms = sample_internal_force(
        e, ni, nj, list(f), "moment",
        member_loads=eff, split_discontinuities=True,
    )
    assert len(csv_m) == len(ms)
    for got, exp in zip(csv_m, ms):
        assert got == pytest.approx(exp, rel=1e-6, abs=1e-6)
    # Sanity: midspan equals the closed-form 35·L²/8 = 157.5.
    assert max(csv_m) == pytest.approx(35.0 * 6.0 ** 2 / 8.0, rel=1e-3)
