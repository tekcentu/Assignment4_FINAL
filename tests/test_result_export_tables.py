import copy
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.gui_qt.result_export_data import (
    MEMBER_STATION_HEADERS,
    NODE_RESULT_HEADERS,
    member_station_metadata,
    member_station_rows,
    node_result_rows,
    write_csv,
)
from structural_analysis.main import run_analysis
from structural_analysis.model import (
    Material,
    Node,
    Section,
    StructuralModel,
    Support,
    UniformDistributedLoad,
)
from structural_analysis.multi_case_result import MultiCaseAnalysisResult


FORBIDDEN = ("SAP", "Diff", "Pct", "Percent", "%")


@pytest.fixture(scope="module")
def qt_app():
    widgets = pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
    app = widgets.QApplication.instance() or widgets.QApplication([])
    yield app


def _multi(result):
    return MultiCaseAnalysisResult(
        cases={"DEFAULT": result},
        active_case="DEFAULT",
        requested_cases=["DEFAULT"],
    )


def _ss_beam(*, L=6.0, offset_i=0.0, offset_j=0.0, w=10.0):
    m = StructuralModel(title="export beam")
    m.materials = {1: Material(1, E=200_000.0, alpha=1e-5)}
    m.sections = {1: Section(1, material_id=1, A=0.02, I=0.08, depth=0.3)}
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    elem = FrameElement2D(
        1, 1, 2, E=200_000.0, A=0.02, I=0.08,
        section_id=1, offset_i=offset_i, offset_j=offset_j,
    )
    elem.member_loads.append(UniformDistributedLoad(wy=-w))
    m.elements = [elem]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=False),
        2: Support(2, ux=False, uy=True, rz=False),
    }
    return m


def _node_result_model():
    m = StructuralModel(title="node export")
    m.materials = {1: Material(1, E=200_000.0, alpha=1e-5)}
    m.sections = {1: Section(1, material_id=1, A=0.02, I=0.08, depth=0.3)}
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements = [FrameElement2D(1, 1, 2, E=200_000.0, A=0.02, I=0.08, section_id=1)]
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    m.nodes[2]
    from structural_analysis.model import NodalLoad
    m.nodal_loads = [NodalLoad(node_id=2, fy=-5.0)]
    return m


def test_member_export_headers_are_clean():
    assert MEMBER_STATION_HEADERS == [
        "case_or_combination",
        "element_id",
        "station_index",
        "x_global_member_m",
        "s_flexible_m",
        "x_over_L_total",
        "s_over_L_flex",
        "N_kN",
        "V_kN",
        "M_kN_m",
    ]
    assert not any(
        bad.lower() in h.lower()
        for h in MEMBER_STATION_HEADERS
        for bad in FORBIDDEN
    )


def test_node_export_headers_are_clean():
    assert NODE_RESULT_HEADERS == [
        "case_or_combination",
        "node_id",
        "ux_m",
        "uy_m",
        "rz_rad",
        "Rx_kN",
        "Ry_kN",
        "Mz_kN_m",
    ]
    assert not any(
        bad.lower() in h.lower()
        for h in NODE_RESULT_HEADERS
        for bad in FORBIDDEN
    )


def test_member_station_count_and_no_offset_coordinates():
    m = _ss_beam(L=6.0)
    r = run_analysis(m, verbose=False)
    rows = member_station_rows(
        m, r, [1], case_or_combination="DEFAULT", n_stations=5,
    )
    assert len(rows) == 5
    assert rows[0][2] == 0
    assert rows[0][3] == pytest.approx(0.0)   # x_global_member_m
    assert rows[0][4] == pytest.approx(0.0)   # s_flexible_m
    assert rows[0][5] == pytest.approx(0.0)   # x_over_L_total
    assert rows[0][6] == pytest.approx(0.0)   # s_over_L_flex
    assert rows[-1][2] == 4
    assert rows[-1][3] == pytest.approx(6.0)
    assert rows[-1][4] == pytest.approx(6.0)
    assert rows[-1][5] == pytest.approx(1.0)
    assert rows[-1][6] == pytest.approx(1.0)


def test_member_station_rigid_offset_coordinates_and_metadata():
    m = _ss_beam(L=6.0, offset_i=1.0, offset_j=0.5)
    r = run_analysis(m, verbose=False)
    rows = member_station_rows(
        m, r, [1], case_or_combination="DEFAULT", n_stations=5,
    )
    meta = member_station_metadata(m, m.elements[0])
    assert meta.L_total == pytest.approx(6.0)
    assert meta.offset_i == pytest.approx(1.0)
    assert meta.offset_j == pytest.approx(0.5)
    assert meta.L_flex == pytest.approx(4.5)
    assert meta.x_start == pytest.approx(1.0)
    assert meta.x_end == pytest.approx(5.5)
    assert rows[0][3] == pytest.approx(1.0)
    assert rows[0][4] == pytest.approx(0.0)
    assert rows[0][5] == pytest.approx(1.0 / 6.0)
    assert rows[0][6] == pytest.approx(0.0)
    assert rows[-1][3] == pytest.approx(5.5)
    assert rows[-1][4] == pytest.approx(4.5)
    assert rows[-1][5] == pytest.approx(5.5 / 6.0)
    assert rows[-1][6] == pytest.approx(1.0)


def test_member_station_udl_shear_zero_and_max_moment_near_midspan():
    m = _ss_beam(L=6.0, w=10.0)
    r = run_analysis(m, verbose=False)
    rows = member_station_rows(
        m, r, [1], case_or_combination="DEFAULT", n_stations=21,
    )
    shears = [row[8] for row in rows]
    moments = [row[9] for row in rows]
    xs = [row[3] for row in rows]
    assert min(shears) < 0.0 < max(shears)
    i_max = max(range(len(rows)), key=lambda i: moments[i])
    assert xs[i_max] == pytest.approx(3.0, abs=0.31)


def test_node_rows_match_analysis_result_d_and_reactions():
    m = _node_result_model()
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    rows = node_result_rows(m, r, case_or_combination="DEFAULT")
    by_node = {row[1]: row for row in rows}
    for nid, row in by_node.items():
        em = r.E_map[nid]
        for col, dof in [(2, "ux"), (3, "uy"), (4, "rz")]:
            idx = em[dof]
            expected = float(r.D[idx]) if idx is not None else None
            assert row[col] == expected
        reac = r.reactions.get(nid, {})
        assert row[5] == reac.get("ux")
        assert row[6] == reac.get("uy")
        assert row[7] == reac.get("rz")


def test_clean_csv_export_has_header_and_data_rows_only(tmp_path):
    m = _ss_beam(L=6.0)
    r = run_analysis(m, verbose=False)
    member_rows = member_station_rows(
        m, r, [1], case_or_combination="DEFAULT", n_stations=3,
    )
    out = tmp_path / "member_clean.csv"
    write_csv(out, MEMBER_STATION_HEADERS, member_rows)
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == MEMBER_STATION_HEADERS
    assert len(rows) == 4
    assert all(not cell.startswith("#") for cell in rows[0])
    assert not any("SAP" in h or "Diff" in h for h in rows[0])


def test_member_dialog_tsv_copy_and_csv_export_are_clean(qt_app, tmp_path):
    pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
    from structural_analysis.gui_qt.result_export_tables import (
        MemberStationForceTableDialog,
    )

    m = _ss_beam(L=6.0, offset_i=1.0, offset_j=0.5)
    r = run_analysis(m, verbose=False)
    d = MemberStationForceTableDialog(
        None, model=m, element_ids=[1], multi_result=_multi(r), n_stations=5,
    )
    qt_app.processEvents()
    assert d._table.rowCount() == 5
    assert "Rigid offsets active" in d._note.text()
    text = d.copy_tsv()
    header = text.splitlines()[0].split("\t")
    assert header == MEMBER_STATION_HEADERS
    assert not any("SAP" in h or "Diff" in h for h in header)
    out = tmp_path / "member.csv"
    d.export_csv(out)
    with out.open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.reader(f))
    assert csv_rows[0] == MEMBER_STATION_HEADERS
    assert len(csv_rows) == 6


def test_node_dialog_tsv_copy_and_csv_export_are_clean(qt_app, tmp_path):
    pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
    from structural_analysis.gui_qt.result_export_tables import NodeResultTableDialog

    m = _node_result_model()
    r = run_analysis(m, verbose=False)
    d = NodeResultTableDialog(None, model=m, multi_result=_multi(r))
    qt_app.processEvents()
    assert d._table.rowCount() == len(m.nodes)
    text = d.copy_tsv()
    header = text.splitlines()[0].split("\t")
    assert header == NODE_RESULT_HEADERS
    assert not any("SAP" in h or "Diff" in h for h in header)
    out = tmp_path / "nodes.csv"
    d.export_csv(out)
    with out.open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.reader(f))
    assert csv_rows[0] == NODE_RESULT_HEADERS
    assert len(csv_rows) == len(m.nodes) + 1


def test_result_export_does_not_mutate_solver_results(tmp_path):
    m = _ss_beam(L=6.0, offset_i=1.0, offset_j=0.5)
    r = run_analysis(m, verbose=False)
    D_before = np.array(r.D, copy=True)
    reactions_before = copy.deepcopy(r.reactions)
    f_before = np.array(r.member_results[1]["f_local"], copy=True)
    member_rows = member_station_rows(
        m, r, [1], case_or_combination="DEFAULT", n_stations=21,
    )
    node_rows = node_result_rows(m, r, case_or_combination="DEFAULT")
    write_csv(tmp_path / "member.csv", MEMBER_STATION_HEADERS, member_rows)
    write_csv(tmp_path / "nodes.csv", NODE_RESULT_HEADERS, node_rows)
    assert np.array_equal(r.D, D_before)
    assert reactions_before == r.reactions
    assert np.array_equal(r.member_results[1]["f_local"], f_before)


def test_member_station_rows_reject_truss_element():
    m = _ss_beam()
    m.elements[0] = TrussElement2D(1, 1, 2, E=200_000.0, A=0.02)
    r = run_analysis(_ss_beam(), verbose=False)
    with pytest.raises(ValueError, match="frame elements only"):
        member_station_rows(m, r, [1], case_or_combination="DEFAULT")


def test_csv_writer_rejects_comparison_headers(tmp_path):
    with pytest.raises(ValueError, match="comparison columns"):
        write_csv(tmp_path / "bad.csv", ["N_kN", "SAP_V2"], [[1.0, ""]])


def test_main_window_exposes_node_result_table_action(qt_app):
    from structural_analysis.gui_qt.app import MainWindow

    w = MainWindow()
    run_menu = next(a.menu() for a in w.menuBar().actions() if a.text() == "&Run")
    assert w.act_node_result_table in run_menu.actions()


def test_frame_context_menu_exposes_station_force_table(qt_app, monkeypatch):
    from PyQt6.QtWidgets import QMenu
    from structural_analysis.gui_qt.app import MainWindow

    w = MainWindow()
    m = _ss_beam()
    w._model = m
    captured: dict[str, list[str]] = {}

    def fake_exec(self, _pos):
        captured["labels"] = [a.text() for a in self.actions()]
        return None

    monkeypatch.setattr(QMenu, "exec", fake_exec)
    w.show_element_menu(1)
    labels = captured["labels"]
    assert any("Station Force Table" in label for label in labels)
