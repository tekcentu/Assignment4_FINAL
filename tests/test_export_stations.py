"""PyQt6 smoke tests for the File → Export station results CSV.

Covers the new export action: it is gated behind a successful solve,
it writes one header row + 21 station rows per element, the N/V/M
columns are scaled to the active Units V1 preset, and the x column
stays in metres (matching the rest of the Units V1 contract).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QSettings  # noqa: E402
    from PyQt6.QtWidgets import QApplication  # noqa: E402
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 unavailable: {exc}", allow_module_level=True)

from structural_analysis.element import FrameElement2D, TrussElement2D  # noqa: E402
from structural_analysis.gui_qt.app import MainWindow  # noqa: E402
from structural_analysis.model import (  # noqa: E402
    Material, Node, Section, Support, NodalLoad, UniformDistributedLoad,
)
from structural_analysis.gui_qt.element_graphics import (  # noqa: E402
    sample_internal_force, effective_member_loads,
)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _clean_qsettings():
    """Other Units V1 tests leave the saved preset behind — reset before
    each export test so MainWindow always starts at the kN-m default."""
    s = QSettings("CE4011", "StructuralAnalysis")
    s.remove("units_preset")
    yield
    s.remove("units_preset")


def _seed_frame_solved(w: MainWindow) -> None:
    m = w._model
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 5.0, 0.0)}
    m.materials = {1: Material(id=1, name="C", E=3e7, density=0.0)}
    m.sections = {1: Section(id=1, name="S", material_id=1,
                             A=0.05, I=1.0e-3, depth=0.3)}
    m.elements = [FrameElement2D(id=1, node_i=1, node_j=2, E=3e7,
                                 A=0.05, I=1.0e-3, section_id=1)]
    m.supports = {1: Support(node_id=1, ux=True, uy=True, rz=True),
                  2: Support(node_id=2, ux=False, uy=True, rz=False)}
    m.nodal_loads = [NodalLoad(node_id=2, fx=10.0)]
    w._run_static_solve(active_only=False)


def _seed_mixed_solved(w: MainWindow) -> None:
    """Frame + truss in one model so the truss-row branch is exercised."""
    m = w._model
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 5.0, 0.0),
        3: Node(3, 5.0, 3.0),
    }
    m.materials = {1: Material(id=1, name="C", E=3e7, density=0.0)}
    m.sections = {1: Section(id=1, name="S", material_id=1,
                             A=0.05, I=1.0e-3, depth=0.3)}
    m.elements = [
        FrameElement2D(id=1, node_i=1, node_j=2, E=3e7,
                       A=0.05, I=1.0e-3, section_id=1),
        TrussElement2D(id=2, node_i=2, node_j=3, E=3e7, A=0.01,
                       section_id=1),
    ]
    m.supports = {
        1: Support(node_id=1, ux=True, uy=True, rz=True),
        3: Support(node_id=3, ux=True, uy=True, rz=False),
    }
    m.nodal_loads = [NodalLoad(node_id=2, fx=10.0, fy=-5.0)]
    w._run_static_solve(active_only=False)


def test_export_action_disabled_before_solve(qt_app):
    w = MainWindow()
    assert not w.act_export_stations.isEnabled()


def test_export_action_enabled_after_solve(qt_app):
    w = MainWindow()
    _seed_frame_solved(w)
    assert w.act_export_stations.isEnabled()


def test_export_writes_header_and_21_stations_per_element(
    qt_app, tmp_path, monkeypatch,
):
    w = MainWindow()
    _seed_frame_solved(w)
    out = tmp_path / "stations.csv"

    import structural_analysis.gui_qt.app as appmod

    def _fake_save(*_a, **_k):
        return (str(out), "CSV (*.csv)")

    monkeypatch.setattr(appmod.QFileDialog, "getSaveFileName", _fake_save)
    w._export_station_results()

    assert out.exists()
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    # Header + 21 stations × 1 element.
    assert len(rows) == 1 + 21
    assert rows[0] == ["Element", "x (m)", "N (kN)", "V (kN)", "M (kN·m)"]
    # First and last x for the 5 m frame.
    assert float(rows[1][1]) == pytest.approx(0.0)
    assert float(rows[-1][1]) == pytest.approx(5.0)


def test_export_no_path_is_noop(qt_app, tmp_path, monkeypatch):
    """Cancelling the save dialog must not raise or write anything."""
    w = MainWindow()
    _seed_frame_solved(w)
    import structural_analysis.gui_qt.app as appmod
    monkeypatch.setattr(
        appmod.QFileDialog, "getSaveFileName",
        lambda *a, **k: ("", ""),
    )
    w._export_station_results()  # must return cleanly, no exception
    assert list(tmp_path.iterdir()) == []


def test_export_headers_and_values_follow_units_preset(
    qt_app, tmp_path, monkeypatch,
):
    """Switching to kip_ft must rename the unit headers and rescale
    every N/V/M cell — x must stay in metres (Units V1 contract)."""
    w = MainWindow()
    _seed_frame_solved(w)
    w._set_units_preset("kip_ft")
    out = tmp_path / "stations_kip.csv"
    import structural_analysis.gui_qt.app as appmod
    monkeypatch.setattr(
        appmod.QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(out), "CSV (*.csv)"),
    )
    w._export_station_results()

    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "Element", "x (m)", "N (kip)", "V (kip)", "M (kip·ft)",
    ]

    # Cross-check N column against the helper: an axial force read from
    # the same f_local should match the row scaled to kip.
    from structural_analysis.gui_qt.element_graphics import (
        sample_internal_force,
    )
    from structural_analysis.gui_common import units as U

    elem = w._model.elements[0]
    mr = w._result.member_results[elem.id]
    xs, ys = sample_internal_force(
        elem, w._model.nodes[1], w._model.nodes[2],
        list(mr["f_local"]), "axial",
    )
    expected_n_first = U.force_to_display(ys[0], "kip_ft")
    assert float(rows[1][2]) == pytest.approx(expected_n_first, rel=1e-5)


def test_export_truss_row_emits_only_axial(qt_app, tmp_path, monkeypatch):
    w = MainWindow()
    _seed_mixed_solved(w)
    out = tmp_path / "mixed.csv"
    import structural_analysis.gui_qt.app as appmod
    monkeypatch.setattr(
        appmod.QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(out), "CSV (*.csv)"),
    )
    w._export_station_results()
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    # Two elements × 21 stations + header.
    assert len(rows) == 1 + 2 * 21
    truss_rows = [r for r in rows[1:] if r[0] == "2"]
    assert len(truss_rows) == 21
    # Every truss row: N populated, V and M blank.
    for r in truss_rows:
        assert r[2] != ""        # N present
        assert r[3] == ""        # no V
        assert r[4] == ""        # no M


def test_station_export_matches_canvas_sampling(qt_app, tmp_path, monkeypatch):
    """The exported M column must equal what the canvas would sample for the
    same active selection — both route through ``effective_member_loads`` +
    ``sample_internal_force`` with the discontinuity split on (the
    member-load reconstruction fix)."""
    import csv
    w = MainWindow()
    m = w._model
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    m.materials = {1: Material(id=1, name="C", E=2e8, density=0.0)}
    m.sections = {1: Section(id=1, name="S", material_id=1,
                             A=0.02, I=0.08, depth=0.3)}
    e = FrameElement2D(id=1, node_i=1, node_j=2, E=2e8,
                       A=0.02, I=0.08, section_id=1)
    e.member_loads.append(UniformDistributedLoad(wy=-10.0))
    m.elements = [e]
    m.supports = {1: Support(node_id=1, ux=True, uy=True, rz=False),
                  2: Support(node_id=2, ux=False, uy=True, rz=False)}
    w._run_static_solve(active_only=False)

    out = tmp_path / "stations.csv"
    import structural_analysis.gui_qt.app as appmod
    monkeypatch.setattr(
        appmod.QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(out), "CSV (*.csv)"),
    )
    w._export_station_results()

    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]   # drop header
    ni, nj = m.nodes[1], m.nodes[2]
    eff = effective_member_loads(e, w._active_case, m.load_combinations)
    f = w._result.member_results[1]["f_local"]
    _, ms = sample_internal_force(
        e, ni, nj, list(f), "moment",
        member_loads=eff, split_discontinuities=True)
    csv_m = [float(r[4]) for r in rows]
    assert len(csv_m) == len(ms)
    for got, exp in zip(csv_m, ms):
        assert got == pytest.approx(exp, rel=1e-6, abs=1e-6)
    # Sanity: a real sagging UDL midspan moment (wL²/8 = 45), not collapsed.
    assert max(csv_m) == pytest.approx(45.0, rel=1e-3)


def test_internal_model_untouched_by_export(qt_app, tmp_path, monkeypatch):
    """The export is read-only — it must not mutate the solved result."""
    import copy
    w = MainWindow()
    _seed_frame_solved(w)
    mr_before = copy.deepcopy(w._result.member_results)
    out = tmp_path / "stations.csv"
    import structural_analysis.gui_qt.app as appmod
    monkeypatch.setattr(
        appmod.QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(out), "CSV (*.csv)"),
    )
    w._export_station_results()
    mr_after = w._result.member_results
    assert mr_after.keys() == mr_before.keys()
    for eid, mr in mr_before.items():
        for k in ("f_local", "d_local", "d_global"):
            assert list(mr[k]) == list(mr_after[eid][k])
    # Sanity: file did get written so we know the export actually ran.
    assert Path(out).exists()
