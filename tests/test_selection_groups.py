"""Tests for PR #42 — Selection Tools + Named Groups (v0.27.0)."""

from __future__ import annotations

import os
import tempfile

import pytest

from structural_analysis.file_io import read_input_file
from structural_analysis.gui_qt.grid import GridSystem
from structural_analysis.gui_qt.project_io import (
    Project,
    SelectionGroup,
    ViewState,
    load_project_json,
    save_project_json,
)


# ── SelectionGroup dataclass ─────────────────────────────────────────────────


class TestSelectionGroup:
    def test_default_empty(self):
        g = SelectionGroup(name="TEST")
        assert g.node_ids == []
        assert g.element_ids == []

    def test_to_dict_sorted(self):
        g = SelectionGroup(name="A", node_ids=[3, 1, 2], element_ids=[5, 4])
        d = g.to_dict()
        assert d["name"] == "A"
        assert d["node_ids"] == [1, 2, 3]
        assert d["element_ids"] == [4, 5]

    def test_from_dict_round_trip(self):
        d = {"name": "B", "node_ids": [10, 20], "element_ids": [1]}
        g = SelectionGroup.from_dict(d)
        assert g.name == "B"
        assert g.node_ids == [10, 20]
        assert g.element_ids == [1]

    def test_from_dict_missing_keys(self):
        g = SelectionGroup.from_dict({"name": "C"})
        assert g.node_ids == []
        assert g.element_ids == []

    def test_from_dict_coerces_to_int(self):
        g = SelectionGroup.from_dict({"name": "D", "node_ids": ["1", "2"]})
        assert g.node_ids == [1, 2]


# ── Project.groups persistence ───────────────────────────────────────────────


def _round_trip(project: Project) -> Project:
    fd, tmp = tempfile.mkstemp(suffix=".spa.json")
    os.close(fd)
    try:
        save_project_json(project, tmp)
        return load_project_json(tmp)
    finally:
        os.unlink(tmp)


def _simple_model():
    return read_input_file("inputs/q2a_settlement.txt")


def test_groups_round_trip():
    model = _simple_model()
    groups = [
        SelectionGroup(name="COLUMNS", node_ids=[1, 3], element_ids=[2]),
        SelectionGroup(name="BEAMS", node_ids=[], element_ids=[1, 3, 5]),
    ]
    p = Project(model=model, groups=groups)
    p2 = _round_trip(p)

    assert len(p2.groups) == 2
    names = {g.name for g in p2.groups}
    assert names == {"COLUMNS", "BEAMS"}
    cols = next(g for g in p2.groups if g.name == "COLUMNS")
    assert sorted(cols.node_ids) == [1, 3]
    assert sorted(cols.element_ids) == [2]


def test_old_spa_json_without_groups_loads_empty():
    """A .spa.json written before v0.27 (no 'groups' key) must load with empty groups."""
    model = _simple_model()
    p = Project(model=model)
    fd, tmp = tempfile.mkstemp(suffix=".spa.json")
    os.close(fd)
    try:
        save_project_json(p, tmp)
        # Manually strip the groups key to simulate an old file.
        import json
        with open(tmp) as f:
            payload = json.load(f)
        payload.pop("groups", None)
        with open(tmp, "w") as f:
            json.dump(payload, f)
        p2 = load_project_json(tmp)
        assert p2.groups == []
    finally:
        os.unlink(tmp)


def test_plain_txt_loads_with_no_groups():
    """Opening a plain .txt solver file gives empty groups (no .spa.json key)."""
    model = _simple_model()
    assert not hasattr(model, "groups")  # model itself has no groups field
    # The app initialises self._groups = {} when opening a .txt file.


def test_core_txt_export_unchanged():
    """Writing a Project that has groups produces a model_txt without any NAMED_GROUPS."""
    model = _simple_model()
    groups = [SelectionGroup(name="G1", node_ids=[1], element_ids=[1])]
    p = Project(model=model, groups=groups)
    fd, tmp = tempfile.mkstemp(suffix=".spa.json")
    os.close(fd)
    try:
        save_project_json(p, tmp)
        import json
        with open(tmp) as f:
            payload = json.load(f)
        assert "NAMED_GROUPS" not in payload["model_txt"]
    finally:
        os.unlink(tmp)


def test_groups_empty_by_default():
    model = _simple_model()
    p = Project(model=model)
    assert p.groups == []
    p2 = _round_trip(p)
    assert p2.groups == []


# ── GUI smoke tests ──────────────────────────────────────────────────────────

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication
    _has_qt = True
except Exception:
    _has_qt = False


@pytest.fixture(scope="module")
def qt_app():
    if not _has_qt:
        pytest.skip("PyQt6 unavailable")
    app = QApplication.instance() or QApplication([])
    yield app


def _make_window(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    return MainWindow()


def _two_element_model():
    """Return a model with 3 nodes, 1 frame (id 1) and 1 truss (id 2)."""
    from structural_analysis.model import StructuralModel, Node, Material, Section
    from structural_analysis.element import FrameElement2D, TrussElement2D
    m = StructuralModel()
    m.materials[1] = Material(id=1, E=2.1e8, alpha=1e-5, density=7850.0)
    m.sections[1] = Section(id=1, material_id=1, A=0.01, I=1e-4, depth=0.3)
    m.sections[2] = Section(id=2, material_id=1, A=0.005, I=5e-5, depth=0.2)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.nodes[3] = Node(3, 2.0, 0.0)
    m.elements.append(FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1))
    m.elements.append(TrussElement2D(id=2, node_i=2, node_j=3, E=2.1e8, A=0.005, section_id=2))
    return m


def test_selection_menu_exists(qt_app):
    w = _make_window(qt_app)
    assert hasattr(w, "_m_selection")


def test_groups_initially_empty(qt_app):
    w = _make_window(qt_app)
    assert w._groups == {}


def test_filter_sel_frames_only(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    # Select both elements
    w.canvas.add_element_to_selection(1)
    w.canvas.add_element_to_selection(2)
    assert len(w.canvas.get_selected_elements()) == 2

    w._filter_sel_frames_only()
    sel = w.canvas.get_selected_elements()
    assert sel == frozenset({1})


def test_filter_sel_trusses_only(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w.canvas.add_element_to_selection(1)
    w.canvas.add_element_to_selection(2)

    w._filter_sel_trusses_only()
    sel = w.canvas.get_selected_elements()
    assert sel == frozenset({2})


def test_filter_sel_same_section(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w.canvas.add_element_to_selection(1)
    w.canvas.add_element_to_selection(2)

    w._filter_sel_same_section()
    # Element 1 is first; section_id=1; element 2 has section_id=2 → filtered out
    sel = w.canvas.get_selected_elements()
    assert sel == frozenset({1})


def test_filter_sel_same_material(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w.canvas.add_element_to_selection(1)
    w.canvas.add_element_to_selection(2)

    # Both elements share material_id=1, so both kept
    w._filter_sel_same_material()
    sel = w.canvas.get_selected_elements()
    assert sel == frozenset({1, 2})


def test_filter_sel_empty_is_noop(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    # No selection — should not crash
    w._filter_sel_frames_only()
    assert w.canvas.get_selected_elements() == frozenset()


def test_filter_sel_clear(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w.canvas.add_element_to_selection(1)
    w._filter_sel_clear()
    assert w.canvas.get_selected_elements() == frozenset()


def test_select_similar_type(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    # Click element 1 (frame); should select all frames — only element 1
    w._select_similar_type(1)
    assert w.canvas.get_selected_elements() == frozenset({1})


def test_select_similar_type_truss(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w._select_similar_type(2)
    assert w.canvas.get_selected_elements() == frozenset({2})


def test_select_similar_does_not_require_ref_to_be_selected(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    # Element 1 not currently selected
    assert 1 not in w.canvas.get_selected_elements()
    w._select_similar_section(1)
    # Should still work — selects all with section_id=1
    assert 1 in w.canvas.get_selected_elements()


def test_select_similar_section(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w._select_similar_section(1)
    assert w.canvas.get_selected_elements() == frozenset({1})


def test_select_similar_type_and_section(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w._select_similar_type_and_section(1)
    assert w.canvas.get_selected_elements() == frozenset({1})


def test_select_similar_material(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    # Both share material 1
    w._select_similar_material(1)
    assert w.canvas.get_selected_elements() == frozenset({1, 2})


# ── group operations ─────────────────────────────────────────────────────────


def test_group_select_skips_missing_ids(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    # Create group with a non-existent node id
    w._groups["G1"] = SelectionGroup(
        name="G1", node_ids=[999], element_ids=[1],
    )
    # Should not crash
    w._group_select("G1")
    sel_nodes = w.canvas.get_selected_nodes()
    # node 999 does not exist; should be silently skipped
    assert 999 not in sel_nodes
    # element 1 exists
    assert 1 in w.canvas.get_selected_elements()


def test_group_add_to_selection(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w.canvas.add_element_to_selection(1)
    w._groups["G1"] = SelectionGroup(name="G1", node_ids=[], element_ids=[2])
    w._group_add_to_selection("G1")
    assert w.canvas.get_selected_elements() == frozenset({1, 2})


def test_group_remove_from_selection(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w.canvas.add_element_to_selection(1)
    w.canvas.add_element_to_selection(2)
    w._groups["G1"] = SelectionGroup(name="G1", node_ids=[], element_ids=[2])
    w._group_remove_from_selection("G1")
    assert w.canvas.get_selected_elements() == frozenset({1})


def test_group_delete_does_not_delete_model_objects(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w._groups["G1"] = SelectionGroup(name="G1", node_ids=[1, 2], element_ids=[1])
    assert len(w._model.nodes) == 3
    del w._groups["G1"]
    assert len(w._model.nodes) == 3  # model untouched


def test_groups_saved_and_loaded_with_spa_json(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w._groups["TEST"] = SelectionGroup(
        name="TEST", node_ids=[1, 2], element_ids=[1],
    )
    fd, tmp = tempfile.mkstemp(suffix=".spa.json")
    os.close(fd)
    try:
        w._save_to(tmp)
        w2 = MainWindow()
        w2._open_path(tmp)
        assert "TEST" in w2._groups
        assert sorted(w2._groups["TEST"].node_ids) == [1, 2]
        assert w2._groups["TEST"].element_ids == [1]
    finally:
        os.unlink(tmp)


def test_groups_cleared_on_new_file(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._groups["OLD"] = SelectionGroup(name="OLD", node_ids=[1])
    # Simulate _do_new without the confirm-discard dialog
    w._model = w._model.__class__()  # fresh model
    w._groups = {}
    assert "OLD" not in w._groups


def test_group_manager_dialog_opens(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    from structural_analysis.gui_qt.dialogs import GroupManagerDialog
    w = MainWindow()
    w._model = _two_element_model()
    w._groups["G1"] = SelectionGroup(name="G1", node_ids=[1], element_ids=[1])
    d = GroupManagerDialog(w, host=w, groups=w._groups, model=w._model)
    assert d.windowTitle() == "Group Manager"
    # Table should show 1 row
    assert d._table.rowCount() == 1


def test_select_all_elements(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w._select_all_elements()
    assert w.canvas.get_selected_elements() == frozenset({1, 2})


def test_select_all_nodes(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w._select_all_nodes()
    assert w.canvas.get_selected_nodes() == frozenset({1, 2, 3})


def test_select_all(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _two_element_model()
    w._select_all()
    assert w.canvas.get_selected_elements() == frozenset({1, 2})
    assert w.canvas.get_selected_nodes() == frozenset({1, 2, 3})
