"""Tests for V1 structure-aware modal terminology (v0.30.2).

User-facing modal text says "Structure N" instead of "Component N". Internal
code names (ComponentModalResult, result.components, component_summary, …) are
deliberately unchanged — these tests assert the DISPLAY boundary only.

Kept in its own file (not test_gui_qt_smoke.py) per the known-slow-case
convention.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ── relabel helper (pure-Python) ──────────────────────────────────────────────


def test_relabel_maps_all_surface_forms():
    from structural_analysis.gui_common.results_view import (
        relabel_component_to_structure as relabel,
    )
    assert relabel("Component") == "Structure"
    assert relabel("component") == "structure"
    assert relabel("Components") == "Structures"
    assert relabel("components") == "structures"
    assert relabel("COMPONENT") == "STRUCTURE"
    assert relabel("COMPONENTS") == "STRUCTURES"


def test_relabel_in_sentence_preserves_surrounding_text():
    from structural_analysis.gui_common.results_view import (
        relabel_component_to_structure as relabel,
    )
    src = "Model contains 2 disconnected components. Component 1 skipped."
    out = relabel(src)
    assert out == "Model contains 2 disconnected structures. Structure 1 skipped."
    assert "component" not in out.lower()


def test_relabel_whole_word_only():
    """Substrings like 'componentry'/'subcomponent' must be left untouched."""
    from structural_analysis.gui_common.results_view import (
        relabel_component_to_structure as relabel,
    )
    src = "componentry and subcomponent stay; component changes"
    out = relabel(src)
    assert "componentry" in out
    assert "subcomponent" in out
    assert "structure changes" in out


def test_relabel_empty_and_nomatch_passthrough():
    from structural_analysis.gui_common.results_view import (
        relabel_component_to_structure as relabel,
    )
    assert relabel("") == ""
    assert relabel("no relevant words here") == "no relevant words here"


# ── fixtures for Qt smoke tests ───────────────────────────────────────────────


def _mat_sec(m, mat_id=1, sec_id=1):
    from structural_analysis.model import Material, Section
    m.materials[mat_id] = Material(id=mat_id, E=2.1e8, alpha=1e-5, density=7850.0)
    m.sections[sec_id] = Section(
        id=sec_id, material_id=mat_id, A=0.01, I=1e-4, depth=0.3
    )


def _two_column_model():
    """Two isolated fixed-base columns → two disconnected structures."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import FrameElement2D
    m = StructuralModel()
    _mat_sec(m)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.nodes[3] = Node(3, 4.0, 0.0)
    m.nodes[4] = Node(4, 4.0, 3.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4,
                       rho=7850.0, section_id=1)
    )
    m.elements.append(
        FrameElement2D(id=2, node_i=3, node_j=4, E=2.1e8, A=0.01, I=1e-4,
                       rho=7850.0, section_id=1)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[3] = Support(node_id=3, ux=True, uy=True, rz=True)
    return m


def _two_columns_one_floating():
    """Two columns, the second unsupported → one skipped structure."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import FrameElement2D
    m = StructuralModel()
    _mat_sec(m)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.nodes[3] = Node(3, 4.0, 0.0)
    m.nodes[4] = Node(4, 4.0, 3.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4,
                       rho=7850.0, section_id=1)
    )
    m.elements.append(
        FrameElement2D(id=2, node_i=3, node_j=4, E=2.1e8, A=0.01, I=1e-4,
                       rho=7850.0, section_id=1)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m


def _single_column_model():
    """One fixed-base column → single connected structure (flat dialog)."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import FrameElement2D
    m = StructuralModel()
    _mat_sec(m)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4,
                       rho=7850.0, section_id=1)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m


@pytest.fixture(scope="module")
def qt_app():
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PyQt6 unavailable: {exc}")
    return QApplication.instance() or QApplication([])


def _all_visible_text(dialog) -> str:
    """Concatenate every tree item label + QLabel text in the dialog."""
    from PyQt6.QtWidgets import QLabel
    parts: list[str] = []
    tree = dialog._tree
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        parts.append(top.text(0))
        for j in range(top.childCount()):
            parts.append(top.child(j).text(0))
    for lbl in dialog.findChildren(QLabel):
        parts.append(lbl.text())
    return "\n".join(parts)


# ── Qt smoke tests ────────────────────────────────────────────────────────────


def test_grouped_tree_headers_say_structure(qt_app):
    from structural_analysis.modal import solve_modal
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    r = solve_modal(_two_column_model(), n_modes=4)
    d = ModalResultsDialog(None, r, on_select=lambda i, s: None,
                           on_close=lambda: None)
    tree = d._tree
    labels = [tree.topLevelItem(i).text(0)
              for i in range(tree.topLevelItemCount())]
    assert any(lbl.startswith("Structure 1 —") for lbl in labels)
    assert any(lbl.startswith("Structure 2 —") for lbl in labels)
    d.close()


def test_no_visible_component_word_in_multi_structure_dialog(qt_app):
    from structural_analysis.modal import solve_modal
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    r = solve_modal(_two_column_model(), n_modes=4)
    d = ModalResultsDialog(None, r, on_select=lambda i, s: None,
                           on_close=lambda: None)
    text = _all_visible_text(d)
    assert "component" not in text.lower(), text
    assert "Structure" in text
    d.close()


def test_explanatory_note_present_when_grouped(qt_app):
    from structural_analysis.modal import solve_modal
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    r = solve_modal(_two_column_model(), n_modes=4)
    d = ModalResultsDialog(None, r, on_select=lambda i, s: None,
                           on_close=lambda: None)
    text = _all_visible_text(d)
    assert "Disconnected structures are solved separately." in text
    assert "Modes are shown per structure." in text
    d.close()


def test_skipped_structure_header_says_structure(qt_app):
    from structural_analysis.modal import solve_modal
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    r = solve_modal(_two_columns_one_floating(), n_modes=4)
    d = ModalResultsDialog(None, r, on_select=lambda i, s: None,
                           on_close=lambda: None)
    text = _all_visible_text(d)
    assert "component" not in text.lower(), text
    # The skipped structure still appears with a Structure header.
    tree = d._tree
    skipped = [
        tree.topLevelItem(i).text(0)
        for i in range(tree.topLevelItemCount())
        if "skip" in tree.topLevelItem(i).text(0).lower()
    ]
    assert skipped and all(s.startswith("Structure ") for s in skipped)
    d.close()


def test_single_structure_dialog_flat_and_no_grouping_prose(qt_app):
    from structural_analysis.modal import solve_modal
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    r = solve_modal(_single_column_model(), n_modes=3)
    assert r.components == []
    d = ModalResultsDialog(None, r, on_select=lambda i, s: None,
                           on_close=lambda: None)
    text = _all_visible_text(d)
    # No disconnected-structure note, and no leftover "Component" wording.
    assert "Disconnected structures" not in text
    assert "component" not in text.lower()
    d.close()
