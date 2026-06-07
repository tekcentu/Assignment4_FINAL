"""Tests for PR follow-up — group element IDs update after Renumber Elements."""

from __future__ import annotations

import os
import tempfile

import pytest

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


def _window(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    return MainWindow()


def _two_element_model():
    """3 nodes, element id=1 (frame), element id=2 (truss)."""
    from structural_analysis.model import StructuralModel, Node, Material, Section
    from structural_analysis.element import FrameElement2D, TrussElement2D
    m = StructuralModel()
    m.materials[1] = Material(id=1, E=2.1e8, alpha=1e-5, density=7850.0)
    m.sections[1] = Section(id=1, material_id=1, A=0.01, I=1e-4, depth=0.3)
    m.sections[2] = Section(id=2, material_id=1, A=0.005, I=5e-5, depth=0.2)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.nodes[3] = Node(3, 2.0, 0.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1)
    )
    m.elements.append(
        TrussElement2D(id=2, node_i=2, node_j=3, E=2.1e8, A=0.005, section_id=2)
    )
    return m


def _renumber(w, mapping):
    from structural_analysis.gui_common.commands import RenumberElementsCmd
    w.execute(RenumberElementsCmd(mapping=mapping))


# ── core remap tests ─────────────────────────────────────────────────────────


def test_group_element_id_remaps_after_renumber(qt_app):
    """After renumber, group element_ids follow the new IDs."""
    from structural_analysis.gui_qt.project_io import SelectionGroup
    w = _window(qt_app)
    w._model = _two_element_model()
    w._groups["G1"] = SelectionGroup(name="G1", element_ids=[1])
    # Swap element IDs: 1→10, 2→20
    _renumber(w, {1: 10, 2: 20})
    assert w._groups["G1"].element_ids == [10]


def test_multiple_group_elements_all_remap(qt_app):
    from structural_analysis.gui_qt.project_io import SelectionGroup
    w = _window(qt_app)
    w._model = _two_element_model()
    w._groups["ALL"] = SelectionGroup(name="ALL", element_ids=[1, 2])
    _renumber(w, {1: 10, 2: 20})
    assert sorted(w._groups["ALL"].element_ids) == [10, 20]


def test_node_only_group_unaffected(qt_app):
    """node_ids must not be touched by renumber."""
    from structural_analysis.gui_qt.project_io import SelectionGroup
    w = _window(qt_app)
    w._model = _two_element_model()
    w._groups["NODES"] = SelectionGroup(name="NODES", node_ids=[1, 2], element_ids=[])
    _renumber(w, {1: 10, 2: 20})
    assert sorted(w._groups["NODES"].node_ids) == [1, 2]
    assert w._groups["NODES"].element_ids == []


def test_select_group_after_renumber_selects_new_id(qt_app):
    """_group_select must select the physical element under its new ID."""
    from structural_analysis.gui_qt.project_io import SelectionGroup
    w = _window(qt_app)
    w._model = _two_element_model()
    w._groups["G1"] = SelectionGroup(name="G1", element_ids=[1])
    _renumber(w, {1: 10, 2: 20})
    w._group_select("G1")
    assert 10 in w.canvas.get_selected_elements()
    assert 1 not in w.canvas.get_selected_elements()


def test_stale_id_not_in_mapping_kept_as_is(qt_app):
    """A group element_id that is not part of the renumber mapping is kept."""
    from structural_analysis.gui_qt.project_io import SelectionGroup
    w = _window(qt_app)
    w._model = _two_element_model()
    # 999 is a stale / non-existent ID — should survive unchanged
    w._groups["G1"] = SelectionGroup(name="G1", element_ids=[1, 999])
    _renumber(w, {1: 10, 2: 20})
    assert 10 in w._groups["G1"].element_ids
    assert 999 in w._groups["G1"].element_ids


# ── undo / redo ───────────────────────────────────────────────────────────────


def test_undo_renumber_restores_group_element_ids(qt_app):
    from structural_analysis.gui_qt.project_io import SelectionGroup
    w = _window(qt_app)
    w._model = _two_element_model()
    w._groups["G1"] = SelectionGroup(name="G1", element_ids=[1])
    _renumber(w, {1: 10, 2: 20})
    assert w._groups["G1"].element_ids == [10]
    w._do_undo()
    assert w._groups["G1"].element_ids == [1]


def test_redo_renumber_remaps_group_element_ids_again(qt_app):
    from structural_analysis.gui_qt.project_io import SelectionGroup
    w = _window(qt_app)
    w._model = _two_element_model()
    w._groups["G1"] = SelectionGroup(name="G1", element_ids=[1])
    _renumber(w, {1: 10, 2: 20})
    w._do_undo()
    assert w._groups["G1"].element_ids == [1]
    w._do_redo()
    assert w._groups["G1"].element_ids == [10]


def test_multiple_groups_all_remap_on_undo_redo(qt_app):
    from structural_analysis.gui_qt.project_io import SelectionGroup
    w = _window(qt_app)
    w._model = _two_element_model()
    w._groups["A"] = SelectionGroup(name="A", element_ids=[1])
    w._groups["B"] = SelectionGroup(name="B", element_ids=[2])
    _renumber(w, {1: 10, 2: 20})
    assert w._groups["A"].element_ids == [10]
    assert w._groups["B"].element_ids == [20]
    w._do_undo()
    assert w._groups["A"].element_ids == [1]
    assert w._groups["B"].element_ids == [2]
    w._do_redo()
    assert w._groups["A"].element_ids == [10]
    assert w._groups["B"].element_ids == [20]


def test_non_renumber_command_does_not_remap(qt_app):
    """Other commands must not trigger any group remapping."""
    from structural_analysis.gui_qt.project_io import SelectionGroup
    from structural_analysis.gui_common.commands import AddNodeCmd
    w = _window(qt_app)
    w._model = _two_element_model()
    w._groups["G1"] = SelectionGroup(name="G1", element_ids=[1, 2])
    w.execute(AddNodeCmd(node_id=99, x=5.0, y=0.0))
    assert sorted(w._groups["G1"].element_ids) == [1, 2]


# ── persistence ───────────────────────────────────────────────────────────────


def test_save_reopen_after_renumber_preserves_remapped_ids(qt_app):
    """Save → reopen must keep the remapped element IDs, not the originals."""
    from structural_analysis.gui_qt.project_io import SelectionGroup
    w = _window(qt_app)
    w._model = _two_element_model()
    w._groups["G1"] = SelectionGroup(name="G1", element_ids=[1])
    _renumber(w, {1: 10, 2: 20})
    assert w._groups["G1"].element_ids == [10]

    fd, tmp = tempfile.mkstemp(suffix=".spa.json")
    os.close(fd)
    try:
        w._save_to(tmp)
        w2 = _window(qt_app)
        w2._open_path(tmp)
        assert "G1" in w2._groups
        assert w2._groups["G1"].element_ids == [10]
    finally:
        os.unlink(tmp)
