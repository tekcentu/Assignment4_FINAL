"""Tests for PR #45 — Matrix / DOF Inspector (v0.29.0)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Skip the entire module when PyQt6 (or its system libs, e.g. libGL / libEGL)
# is unavailable — the helpers under test live in a Qt-importing module so
# even non-Qt helper tests need PyQt6 importable.
try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401
    _has_qt = True
except Exception:
    _has_qt = False

if not _has_qt:
    pytest.skip("PyQt6 unavailable", allow_module_level=True)

from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


# ── model fixtures ─────────────────────────────────────────────────────────


def _frame_model():
    """Simple pinned-pinned frame: 2 nodes, 1 frame element, 2 supports."""
    from structural_analysis.model import StructuralModel, Node, Material, Section, Support
    from structural_analysis.element import FrameElement2D
    m = StructuralModel()
    m.materials[1] = Material(id=1, E=2.1e8, alpha=1e-5, density=7850.0)
    m.sections[1] = Section(id=1, material_id=1, A=0.01, I=1e-4, depth=0.3)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 3.0, 0.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=False)
    m.supports[2] = Support(node_id=2, ux=False, uy=True, rz=False)
    return m


def _truss_model():
    """Pure-truss: 2 nodes, 1 truss element, 2 pin supports."""
    from structural_analysis.model import StructuralModel, Node, Material, Section, Support
    from structural_analysis.element import TrussElement2D
    m = StructuralModel()
    m.materials[1] = Material(id=1, E=2.1e8, alpha=1e-5, density=7850.0)
    m.sections[1] = Section(id=1, material_id=1, A=0.005, I=5e-5, depth=0.2)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.elements.append(
        TrussElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.005, section_id=1)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=False)
    m.supports[2] = Support(node_id=2, ux=False, uy=True, rz=False)
    return m


def _mixed_model():
    """3 nodes: frame element 1 (nodes 1-2), truss element 2 (nodes 2-3)."""
    from structural_analysis.model import StructuralModel, Node, Material, Section, Support
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
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=False)
    m.supports[3] = Support(node_id=3, ux=False, uy=True, rz=False)
    return m


def _released_frame_model():
    """Frame with a moment release at node j."""
    from structural_analysis.model import StructuralModel, Node, Material, Section, Support
    from structural_analysis.element import FrameElement2D
    m = StructuralModel()
    m.materials[1] = Material(id=1, E=2.1e8, alpha=1e-5, density=0.0)
    m.sections[1] = Section(id=1, material_id=1, A=0.01, I=1e-4, depth=0.3)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 2.0, 0.0)
    m.elements.append(
        FrameElement2D(
            id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4,
            section_id=1, release_j=True,
        )
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[2] = Support(node_id=2, ux=False, uy=True, rz=False)
    return m


def _empty_model():
    from structural_analysis.model import StructuralModel
    return StructuralModel()


def _assemble(model):
    from structural_analysis.assembler import assemble_global_system
    K, F, dofs, _, elem_data = assemble_global_system(model)
    import numpy as np
    free = dofs.free_indices
    Kff = K[np.ix_(free, free)] if free else np.zeros((0, 0))
    return K, F, dofs, Kff, elem_data


# ── helper / formatter tests (no Qt needed) ───────────────────────────────


def test_fmt_zero():
    from structural_analysis.gui_qt.matrix_inspector import _fmt
    assert _fmt(0.0) == "0"
    assert _fmt(1e-40) == "0"
    assert _fmt(-1e-40) == "0"


def test_fmt_scientific():
    from structural_analysis.gui_qt.matrix_inspector import _fmt
    assert _fmt(1234567.0) == "1.235e+06"
    assert _fmt(-0.001) == "-1.000e-03"


def test_build_dof_labels_keys_are_global_indices():
    from structural_analysis.gui_qt.matrix_inspector import _build_dof_labels
    K, F, dofs, Kff, _ = _assemble(_frame_model())
    labels = _build_dof_labels(dofs)
    # Every active global index must have a label
    all_active = {
        idx for m in dofs.active_map.values() for idx in m.values() if idx is not None
    }
    assert set(labels.keys()) == all_active


def test_build_dof_labels_format():
    from structural_analysis.gui_qt.matrix_inspector import _build_dof_labels
    K, F, dofs, Kff, _ = _assemble(_frame_model())
    labels = _build_dof_labels(dofs)
    for key, val in labels.items():
        assert isinstance(key, int)
        assert val.startswith("n") and "." in val


# ── DOF map correctness ───────────────────────────────────────────────────


def test_dof_map_lists_ux_uy_rz_for_frame_nodes():
    K, F, dofs, Kff, _ = _assemble(_frame_model())
    # Frame nodes must have ux, uy, rz in active_map (not None)
    for nid in (1, 2):
        m = dofs.active_map[nid]
        for dof in ("ux", "uy", "rz"):
            assert m.get(dof) is not None, f"node {nid} missing {dof}"


def test_truss_only_model_omits_rz():
    K, F, dofs, Kff, _ = _assemble(_truss_model())
    # Nodes connected only to truss should have no rz DOF
    for nid in (1, 2):
        m = dofs.active_map[nid]
        assert m.get("rz") is None, f"node {nid} should have no rz"


def test_dof_map_free_restrained_status_matches_dofmanager():
    K, F, dofs, Kff, _ = _assemble(_frame_model())
    free_set = set(dofs.free_indices)
    restrained_set = set(dofs.restrained_indices)
    # Every active DOF index is either free or restrained — not both, not neither
    for m in dofs.active_map.values():
        for idx in m.values():
            if idx is not None:
                assert (idx in free_set) != (idx in restrained_set)


def test_kff_size_matches_free_dof_count():
    K, F, dofs, Kff, _ = _assemble(_frame_model())
    nf = len(dofs.free_indices)
    assert Kff.shape == (nf, nf)


# ── element matrix shapes ────────────────────────────────────────────────


def test_frame_element_raw_local_stiffness_is_6x6():
    model = _frame_model()
    elem = model.elements[0]
    k = elem.raw_local_stiffness(model.nodes)
    assert k.shape == (6, 6)


def test_frame_element_transformation_matrix_is_6x6():
    model = _frame_model()
    elem = model.elements[0]
    T = elem.transformation_matrix(model.nodes)
    assert T.shape == (6, 6)


def test_frame_element_global_stiffness_is_6x6():
    model = _frame_model()
    elem = model.elements[0]
    k_global, _ = elem.global_stiffness_and_load(model.nodes)
    assert k_global.shape == (6, 6)


def test_truss_raw_local_stiffness_is_6x6_axial_only():
    model = _truss_model()
    elem = model.elements[0]
    k = elem.raw_local_stiffness(model.nodes)
    assert k.shape == (6, 6)
    # Only axial DOFs (0,3) should be non-zero
    for i in range(6):
        for j in range(6):
            if (i, j) not in ((0, 0), (0, 3), (3, 0), (3, 3)):
                assert k[i, j] == 0.0, f"expected 0 at ({i},{j}), got {k[i,j]}"


def test_frame_element_matrix_labels_are_dof_labels():
    from structural_analysis.gui_qt.matrix_inspector import _elem_dof_labels
    model = _frame_model()
    K, F, dofs, Kff, _ = _assemble(model)
    elem = model.elements[0]
    lbls = _elem_dof_labels(elem, dofs)
    assert len(lbls) == 6
    # Frame element: all 6 should be real labels (not '–'), for ux/uy/rz both ends
    for lbl in lbls:
        assert lbl != "–", f"unexpected inactive DOF label in frame element: {lbls}"
    assert lbls[0].endswith(".ux")
    assert lbls[1].endswith(".uy")
    assert lbls[2].endswith(".rz")
    assert lbls[3].endswith(".ux")
    assert lbls[4].endswith(".uy")
    assert lbls[5].endswith(".rz")


def test_truss_element_dof_labels_mark_rz_as_inactive():
    from structural_analysis.gui_qt.matrix_inspector import _elem_dof_labels
    model = _truss_model()
    K, F, dofs, Kff, _ = _assemble(model)
    elem = model.elements[0]
    lbls = _elem_dof_labels(elem, dofs)
    # rz (index 2 and 5) should be '–' for pure truss
    assert lbls[2] == "–"
    assert lbls[5] == "–"
    # ux, uy (indices 0,1,3,4) should be labelled
    for i in (0, 1, 3, 4):
        assert lbls[i] != "–"


# ── symmetry checks ───────────────────────────────────────────────────────


def test_global_k_symmetric_for_stable_frame_model():
    K, F, dofs, Kff, _ = _assemble(_frame_model())
    asym = np.max(np.abs(K - K.T))
    assert asym < 1e-9, f"K is not symmetric: max|K-Kᵀ| = {asym}"


def test_kff_symmetric_for_stable_frame_model():
    K, F, dofs, Kff, _ = _assemble(_frame_model())
    asym = np.max(np.abs(Kff - Kff.T))
    assert asym < 1e-9, f"Kff is not symmetric: max|Kff-Kffᵀ| = {asym}"


def test_global_k_symmetric_for_mixed_model():
    K, F, dofs, Kff, _ = _assemble(_mixed_model())
    asym = np.max(np.abs(K - K.T))
    assert asym < 1e-9


# ── label–DOF-map consistency ─────────────────────────────────────────────


def test_global_k_labels_cover_all_active_dofs():
    from structural_analysis.gui_qt.matrix_inspector import _build_dof_labels
    K, F, dofs, Kff, _ = _assemble(_frame_model())
    labels = _build_dof_labels(dofs)
    # Number of labelled DOFs == n_total
    assert len(labels) == dofs.n_total


def test_kff_labels_subset_of_dof_labels():
    from structural_analysis.gui_qt.matrix_inspector import _build_dof_labels
    K, F, dofs, Kff, _ = _assemble(_frame_model())
    labels = _build_dof_labels(dofs)
    free_lbls = [labels.get(i) for i in dofs.free_indices]
    assert all(lbl is not None for lbl in free_lbls)
    assert len(free_lbls) == len(dofs.free_indices)


# ── release condensation ──────────────────────────────────────────────────


def test_released_frame_condensed_differs_from_raw():
    model = _released_frame_model()
    elem = model.elements[0]
    k_raw = elem.raw_local_stiffness(model.nodes)
    k_cond, _ = elem.assembled_local_stiffness_and_load(model.nodes)
    diff = float(np.max(np.abs(k_cond - k_raw)))
    assert diff > 1e-10, "Expected condensed stiffness to differ from raw for a released element"


def test_unreleased_frame_condensed_equals_raw():
    model = _frame_model()
    elem = model.elements[0]
    k_raw = elem.raw_local_stiffness(model.nodes)
    k_cond, _ = elem.assembled_local_stiffness_and_load(model.nodes)
    diff = float(np.max(np.abs(k_cond - k_raw)))
    assert diff < 1e-30, "Unreleased element condensed should match raw"


# ── Qt GUI smoke tests ────────────────────────────────────────────────────


def test_inspector_window_opens_on_frame_model(qt_app):
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    assert w.windowTitle() == "Matrix / DOF Inspector"
    w.close()


def test_inspector_has_four_tabs(qt_app):
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    assert w._tabs.count() == 4
    titles = [w._tabs.tabText(i) for i in range(4)]
    assert titles == ["DOF Map", "Element Matrix", "Global K", "Kff"]
    w.close()


def test_inspector_opens_before_analysis(qt_app):
    """Inspector must work even when the model has never been solved."""
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    # No solve called — inspector assembles independently
    w = MatrixDofInspectorWindow(None, lambda: model)
    assert w._tabs.count() == 4
    w.close()


def test_inspector_does_not_mutate_model(qt_app):
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    n_nodes = len(model.nodes)
    n_elems = len(model.elements)
    n_supports = len(model.supports)
    elem_ids = [e.id for e in model.elements]
    w = MatrixDofInspectorWindow(None, lambda: model)
    # After open
    assert len(model.nodes) == n_nodes
    assert len(model.elements) == n_elems
    assert len(model.supports) == n_supports
    assert [e.id for e in model.elements] == elem_ids
    w.close()


def test_inspector_empty_model_shows_placeholder_not_crash(qt_app):
    """Empty model must not raise — each tab shows a friendly placeholder."""
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    w = MatrixDofInspectorWindow(None, lambda: _empty_model())
    assert w._tabs.count() == 4
    w.close()


def test_inspector_default_element_matches_set_selected(qt_app):
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _mixed_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    w.set_selected_element(2)
    w.refresh()
    # After setting element 2 and refreshing, sel_elem_id should be 2
    assert w._sel_elem_id == 2
    w.close()


def test_inspector_truss_model_no_crash(qt_app):
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _truss_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    assert w._tabs.count() == 4
    w.close()


def test_inspector_refresh_does_not_crash(qt_app):
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    w.refresh()   # second call
    assert w._tabs.count() == 4
    w.close()


def test_run_menu_has_matrix_inspector_action(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    assert hasattr(w, "act_matrix_inspector")
    assert w.act_matrix_inspector.text() == "Matrix / &DOF Inspector…"
    w.close()


# ── PR polish: sizing, layout, condition tooltip ──────────────────────────────


def test_window_default_size_is_polished(qt_app):
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    assert w.width() >= 1100 and w.height() >= 800
    mn = w.minimumSize()
    assert mn.width() >= 850 and mn.height() >= 600
    w.close()


def _find_table(widget, type_name="QTableWidget"):
    from PyQt6.QtWidgets import QTableWidget
    if isinstance(widget, QTableWidget):
        return widget
    for child in widget.findChildren(QTableWidget):
        return child
    return None


def test_global_k_table_expands_to_fill(qt_app):
    from PyQt6.QtWidgets import QSizePolicy
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    # Tab index 2 = Global K
    tab = w._tabs.widget(2)
    table = _find_table(tab)
    assert table is not None
    pol = table.sizePolicy()
    assert pol.horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert pol.verticalPolicy() == QSizePolicy.Policy.Expanding
    w.close()


def test_kff_table_expands_to_fill(qt_app):
    from PyQt6.QtWidgets import QSizePolicy
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    # Tab index 3 = Kff
    tab = w._tabs.widget(3)
    table = _find_table(tab)
    assert table is not None
    pol = table.sizePolicy()
    assert pol.horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert pol.verticalPolicy() == QSizePolicy.Policy.Expanding
    w.close()


def test_element_matrix_tab_uses_sub_tabs(qt_app):
    from PyQt6.QtWidgets import QTabWidget
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    tab = w._tabs.widget(1)  # Element Matrix
    sub = tab.findChild(QTabWidget, "elementMatrixSubTabs")
    assert sub is not None
    labels = [sub.tabText(i) for i in range(sub.count())]
    assert "k_local (raw)" in labels
    assert "k_local (condensed)" in labels
    assert any(l.startswith("T") for l in labels)
    assert "k_global" in labels
    w.close()


def test_element_matrix_sub_tabs_present_for_truss(qt_app):
    """All four sub-tabs exist even when condensed == raw."""
    from PyQt6.QtWidgets import QTabWidget
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _truss_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    tab = w._tabs.widget(1)
    sub = tab.findChild(QTabWidget, "elementMatrixSubTabs")
    assert sub is not None
    assert sub.count() == 4
    w.close()


def test_condition_estimate_label_has_tooltip(qt_app):
    from PyQt6.QtWidgets import QLabel
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    # Look across Global K and Kff tabs for the condition-estimate label.
    found = False
    for idx in (2, 3):
        tab = w._tabs.widget(idx)
        for lbl in tab.findChildren(QLabel):
            if "condition estimate" in lbl.text().lower():
                tip = lbl.toolTip().lower()
                assert "condition" in tip and "numerical error" in tip
                assert "near-singular" in tip or "singularity" in tip
                found = True
    assert found, "no condition-estimate label found across K / Kff tabs"
    w.close()


def test_matrix_table_uses_compact_row_height(qt_app):
    from structural_analysis.gui_qt.matrix_inspector import (
        MatrixDofInspectorWindow, _ROW_HEIGHT,
    )
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    tab = w._tabs.widget(2)  # Global K
    table = _find_table(tab)
    assert table is not None
    assert table.verticalHeader().defaultSectionSize() == _ROW_HEIGHT
    w.close()


# ── PR review: initial-open respects canvas selection ────────────────────────


def test_constructor_selected_element_id_is_applied_on_first_open(qt_app):
    """Passing selected_element_id to __init__ must drive the Element Matrix
    tab on the *first* refresh (which runs inside __init__)."""
    from PyQt6.QtWidgets import QComboBox
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _mixed_model()  # has elements 1 (frame) and 2 (truss)
    w = MatrixDofInspectorWindow(None, lambda: model, selected_element_id=2)
    tab = w._tabs.widget(1)
    combo = tab.findChild(QComboBox)
    assert combo is not None
    assert combo.currentData() == 2
    w.close()


def test_constructor_without_selection_defaults_to_first(qt_app):
    from PyQt6.QtWidgets import QComboBox
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _mixed_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    tab = w._tabs.widget(1)
    combo = tab.findChild(QComboBox)
    assert combo is not None
    assert combo.currentData() == 1
    w.close()


def test_main_window_open_inspector_passes_canvas_selection(qt_app):
    """First-open path in MainWindow must pre-seed the selection so the
    Element Matrix tab opens on the canvas-selected element."""
    from PyQt6.QtWidgets import QComboBox
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = _mixed_model()
    w.canvas.select_element(2)
    w._open_matrix_inspector()
    insp = w._matrix_inspector_window
    assert insp is not None
    tab = insp._tabs.widget(1)
    combo = tab.findChild(QComboBox)
    assert combo is not None
    assert combo.currentData() == 2
    insp.close()
    w.close()


# ── PR polish: condition-estimate SVD breakdown ──────────────────────────────


def _find_cond_panel_text(tab) -> str | None:
    """Return the text of the SVD-breakdown panel in a tab, or None."""
    from PyQt6.QtWidgets import QLabel
    for lbl in tab.findChildren(QLabel):
        if lbl.objectName() == "conditionEstimatePanel":
            return lbl.text()
    return None


def test_kff_tab_shows_svd_breakdown(qt_app):
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    text = _find_cond_panel_text(w._tabs.widget(3))   # Kff
    assert text is not None
    assert "Condition estimate" in text
    assert "κ(Kff) = σmax / σmin" in text
    assert "σmax =" in text
    assert "σmin =" in text
    assert " / " in text and " = " in text   # full ratio line
    w.close()


def test_global_k_tab_shows_svd_breakdown(qt_app):
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    text = _find_cond_panel_text(w._tabs.widget(2))   # Global K
    assert text is not None
    assert "Condition estimate" in text
    assert "κ(Kff) = σmax / σmin" in text   # Global K tab reports Kff κ
    assert "σmax =" in text
    assert "σmin =" in text
    w.close()


def test_cond_panel_tooltip_explains_svd(qt_app):
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    from PyQt6.QtWidgets import QLabel
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    for idx in (2, 3):
        for lbl in w._tabs.widget(idx).findChildren(QLabel):
            if lbl.objectName() == "conditionEstimatePanel":
                tip = lbl.toolTip().lower()
                assert "ratio" in tip and "singular value" in tip
                assert "numerical error" in tip
                assert "near-singular" in tip or "singularity" in tip
                w.close()
                return
    w.close()
    pytest.fail("no condition-estimate panel found")


def test_cond_panel_ratio_matches_smax_over_smin(qt_app):
    """κ printed on the last line must equal σmax / σmin (within format)."""
    import re
    from structural_analysis.gui_qt.matrix_inspector import MatrixDofInspectorWindow
    model = _frame_model()
    w = MatrixDofInspectorWindow(None, lambda: model)
    text = _find_cond_panel_text(w._tabs.widget(3))
    assert text is not None
    smax = float(re.search(r"σmax\s*=\s*([\-0-9.eE+]+)", text).group(1))
    smin = float(re.search(r"σmin\s*=\s*([\-0-9.eE+]+)", text).group(1))
    kappa_line = text.splitlines()[-1]
    kappa_val = float(re.search(r"=\s*([\-0-9.eE+]+)\s*$", kappa_line).group(1))
    assert kappa_val == pytest.approx(smax / smin, rel=1e-2)
    w.close()


def test_cond_large_matrix_shows_skipped_message(qt_app, monkeypatch):
    """When n_free > _RANK_GUARD the SVD must be skipped and the user
    sees the documented skip message instead of the breakdown panel."""
    from PyQt6.QtWidgets import QLabel
    from structural_analysis.gui_qt import matrix_inspector as mi
    monkeypatch.setattr(mi, "_RANK_GUARD", 0)   # force skip path
    model = _frame_model()
    w = mi.MatrixDofInspectorWindow(None, lambda: model)
    for idx in (2, 3):
        tab = w._tabs.widget(idx)
        # No breakdown panel
        assert _find_cond_panel_text(tab) is None
        # Documented skip message present
        texts = [lbl.text() for lbl in tab.findChildren(QLabel)]
        assert any("Condition calculation skipped" in t for t in texts), \
            f"skip message missing in tab {idx}: {texts}"
    w.close()
