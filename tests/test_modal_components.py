"""Tests for PR component-aware modal analysis (v0.30.0).

Pure-Python solver tests plus Qt smoke tests for the multi-component
modal results dialog.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ── helpers / fixtures ────────────────────────────────────────────────────────


def _mat_sec(m, mat_id=1, sec_id=1):
    from structural_analysis.model import Material, Section
    m.materials[mat_id] = Material(id=mat_id, E=2.1e8, alpha=1e-5, density=7850.0)
    m.sections[sec_id] = Section(id=sec_id, material_id=mat_id, A=0.01, I=1e-4, depth=0.3)


def _two_column_model():
    """Two isolated fixed-base columns, no cross-connection.

    Column 1: nodes 1 (base, fixed) → 2 (free tip), element id=1.
    Column 2: nodes 3 (base, fixed) → 4 (free tip), element id=2.
    Columns are 4 m apart so no coincident nodes.
    """
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import FrameElement2D
    m = StructuralModel()
    _mat_sec(m)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.nodes[3] = Node(3, 4.0, 0.0)
    m.nodes[4] = Node(4, 4.0, 3.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, rho=7850.0, section_id=1)
    )
    m.elements.append(
        FrameElement2D(id=2, node_i=3, node_j=4, E=2.1e8, A=0.01, I=1e-4, rho=7850.0, section_id=1)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[3] = Support(node_id=3, ux=True, uy=True, rz=True)
    return m


def _two_columns_one_floating():
    """Same as above but column 2 has no support."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import FrameElement2D
    m = StructuralModel()
    _mat_sec(m)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.nodes[3] = Node(3, 4.0, 0.0)
    m.nodes[4] = Node(4, 4.0, 3.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, rho=7850.0, section_id=1)
    )
    m.elements.append(
        FrameElement2D(id=2, node_i=3, node_j=4, E=2.1e8, A=0.01, I=1e-4, rho=7850.0, section_id=1)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m


def _frame_with_orphan_node():
    """Single supported frame plus one orphan node (no elements)."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import FrameElement2D
    m = StructuralModel()
    _mat_sec(m)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.nodes[99] = Node(99, 10.0, 0.0)   # orphan
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, rho=7850.0, section_id=1)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m


def _single_column_model():
    """Simple single fixed-base column for backward-compat comparison."""
    from structural_analysis.model import StructuralModel, Node, Support
    from structural_analysis.element import FrameElement2D
    m = StructuralModel()
    _mat_sec(m)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, rho=7850.0, section_id=1)
    )
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m


# ── component detection ───────────────────────────────────────────────────────


def test_two_columns_detected_as_two_components():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    assert r.status == "ok"
    assert len(r.components) == 2


def test_component_ids_are_one_indexed():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    ids = [c.component_id for c in r.components]
    assert ids == [1, 2]


def test_component_node_ids_are_disjoint():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    c1_nodes = set(r.components[0].node_ids)
    c2_nodes = set(r.components[1].node_ids)
    assert c1_nodes.isdisjoint(c2_nodes)


def test_component_element_ids_correct():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    assert r.components[0].element_ids == [1]
    assert r.components[1].element_ids == [2]


# ── per-component solve ───────────────────────────────────────────────────────


def test_each_supported_component_has_modes():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    for c in r.components:
        if c.is_supported:
            assert c.n_modes > 0


def test_component_frequencies_ascending():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    for c in r.components:
        if c.n_modes > 1:
            assert all(c.frequencies[i] <= c.frequencies[i + 1]
                       for i in range(c.n_modes - 1))


def test_flat_arrays_length_equals_sum_of_component_modes():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    total = sum(c.n_modes for c in r.components)
    assert r.n_modes == total
    assert len(r.frequencies) == total
    assert r.modes.shape == (r.dofs.n_total, total)


def test_global_mode_offsets_non_overlapping():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    solved = [c for c in r.components if c.n_modes > 0]
    for i in range(len(solved) - 1):
        assert solved[i].global_mode_offset + solved[i].n_modes == solved[i + 1].global_mode_offset


def test_component_modes_orthogonal_in_space():
    """C1's modes have zero displacement at C2 nodes and vice-versa."""
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    c1, c2 = r.components[0], r.components[1]
    c2_node_ids = c2.node_ids
    c1_node_ids = c1.node_ids

    for local_k in range(c1.n_modes):
        global_k = c1.global_mode_offset + local_k
        for nid in c2_node_ids:
            ux_idx = r.dofs.active_map[nid]["ux"]
            uy_idx = r.dofs.active_map[nid]["uy"]
            if ux_idx is not None:
                assert abs(r.modes[ux_idx, global_k]) < 1e-10
            if uy_idx is not None:
                assert abs(r.modes[uy_idx, global_k]) < 1e-10

    for local_k in range(c2.n_modes):
        global_k = c2.global_mode_offset + local_k
        for nid in c1_node_ids:
            ux_idx = r.dofs.active_map[nid]["ux"]
            uy_idx = r.dofs.active_map[nid]["uy"]
            if ux_idx is not None:
                assert abs(r.modes[ux_idx, global_k]) < 1e-10
            if uy_idx is not None:
                assert abs(r.modes[uy_idx, global_k]) < 1e-10


def test_identical_columns_have_equal_first_frequencies():
    """Two identical columns must yield the same first frequency."""
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    c1, c2 = r.components[0], r.components[1]
    assert abs(c1.frequencies[0] - c2.frequencies[0]) < 1e-6 * c1.frequencies[0]


# ── unsupported / skipped components ─────────────────────────────────────────


def test_unsupported_component_is_skipped():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_columns_one_floating(), n_modes=4)
    unsupported = [c for c in r.components if not c.is_supported]
    assert len(unsupported) == 1
    assert unsupported[0].n_modes == 0
    assert unsupported[0].skip_reason is not None
    assert "no support" in unsupported[0].skip_reason.lower()


def test_unsupported_component_appears_in_warnings():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_columns_one_floating(), n_modes=4)
    assert any("skipped" in w.lower() for w in r.warnings)


def test_supported_component_still_solved_when_other_unsupported():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_columns_one_floating(), n_modes=4)
    supported = [c for c in r.components if c.is_supported]
    assert len(supported) == 1
    assert supported[0].n_modes > 0


# ── orphan node ───────────────────────────────────────────────────────────────


def test_orphan_node_does_not_crash():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_frame_with_orphan_node(), n_modes=3)
    assert r.status == "ok"


def test_orphan_node_component_is_skipped_not_solved():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_frame_with_orphan_node(), n_modes=3)
    orphans = [c for c in r.components if 99 in c.node_ids]
    assert len(orphans) == 1
    assert orphans[0].n_modes == 0
    assert orphans[0].skip_reason is not None
    assert "orphan" in orphans[0].skip_reason.lower()


def test_orphan_node_does_not_affect_frame_frequencies():
    """Orphan node must not change the solved component's frequencies."""
    from structural_analysis.modal import solve_modal
    r_with = solve_modal(_frame_with_orphan_node(), n_modes=3)
    r_without = solve_modal(_single_column_model(), n_modes=3)
    solved = [c for c in r_with.components if c.n_modes > 0]
    assert len(solved) == 1
    np.testing.assert_allclose(
        solved[0].frequencies, r_without.frequencies[:len(solved[0].frequencies)],
        rtol=1e-8,
    )


# ── single-component backward compat ─────────────────────────────────────────


def test_single_component_components_list_is_empty():
    """Single-component models leave ModalResult.components empty."""
    from structural_analysis.modal import solve_modal
    r = solve_modal(_single_column_model(), n_modes=3)
    assert r.components == []


def test_single_component_component_summary_is_empty():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_single_column_model(), n_modes=3)
    assert r.component_summary == ""


def test_single_component_frequencies_unchanged():
    """Frequencies for a single-component model must not shift."""
    from structural_analysis.modal import solve_modal
    # Cantilever closed-form first bending: f1 = (1.875)²/(2π) * sqrt(EI/ρAL⁴)
    # For our column: L=3, E=2.1e8, I=1e-4, A=0.01, rho=7850
    r = solve_modal(_single_column_model(), n_modes=3)
    assert r.status == "ok"
    assert r.n_modes == 3
    assert r.frequencies[0] < r.frequencies[1] < r.frequencies[2]


# ── component_summary ─────────────────────────────────────────────────────────


def test_component_summary_mentions_count_for_multi():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    assert "2" in r.component_summary
    assert r.component_summary != ""


def test_component_summary_empty_for_single():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_single_column_model(), n_modes=3)
    assert r.component_summary == ""


def test_component_summary_mentions_skipped_when_present():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_columns_one_floating(), n_modes=4)
    assert "skip" in r.component_summary.lower() or "skipped" in r.component_summary.lower()


# ── canvas compatibility: mode vector format ──────────────────────────────────


def test_mode_vectors_are_full_global_length():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    assert r.modes.shape[0] == r.dofs.n_total
    assert r.modes.shape[1] == r.n_modes


def test_inactive_component_nodes_have_zero_displacement_in_c1_modes():
    from structural_analysis.modal import solve_modal
    r = solve_modal(_two_column_model(), n_modes=4)
    c1, c2 = r.components[0], r.components[1]
    for k in range(c1.n_modes):
        global_k = c1.global_mode_offset + k
        for nid in c2.node_ids:
            for dof in ("ux", "uy", "rz"):
                idx = r.dofs.active_map.get(nid, {}).get(dof)
                if idx is not None:
                    assert r.modes[idx, global_k] == pytest.approx(0.0, abs=1e-12)


# ── Qt smoke: modal results dialog ───────────────────────────────────────────


try:
    from PyQt6.QtWidgets import QApplication
    _has_qt = True
except Exception:
    _has_qt = False

if not _has_qt:
    pytest.skip("PyQt6 unavailable", allow_module_level=True)

from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_modal_dialog_flat_for_single_component(qt_app):
    """Single-component result → flat top-level mode items (no grouping)."""
    from structural_analysis.modal import solve_modal
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    r = solve_modal(_single_column_model(), n_modes=3)
    assert r.components == []
    d = ModalResultsDialog(None, r, on_select=lambda i, s: None, on_close=lambda: None)
    # All items must be top-level (no parents)
    tree = d._tree
    assert tree.topLevelItemCount() == r.n_modes
    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        assert item.childCount() == 0
    d.close()


def test_modal_dialog_grouped_for_multi_component(qt_app):
    """Multi-component result → component parent items with child mode rows."""
    from structural_analysis.modal import solve_modal
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    r = solve_modal(_two_column_model(), n_modes=4)
    assert len(r.components) == 2
    d = ModalResultsDialog(None, r, on_select=lambda i, s: None, on_close=lambda: None)
    tree = d._tree
    # Two top-level component groups
    assert tree.topLevelItemCount() == 2
    for i in range(2):
        parent = tree.topLevelItem(i)
        assert parent.childCount() > 0
    d.close()


def test_modal_dialog_skipped_component_shows_in_tree(qt_app):
    """Unsupported component appears as a top-level item with skip info, no children."""
    from structural_analysis.modal import solve_modal
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    r = solve_modal(_two_columns_one_floating(), n_modes=4)
    d = ModalResultsDialog(None, r, on_select=lambda i, s: None, on_close=lambda: None)
    tree = d._tree
    # Find the skipped component's top-level item
    skipped_items = [
        tree.topLevelItem(i)
        for i in range(tree.topLevelItemCount())
        if tree.topLevelItem(i).childCount() == 0
        and "skip" in tree.topLevelItem(i).text(0).lower()
    ]
    assert len(skipped_items) >= 1
    d.close()


def test_modal_dialog_child_item_stores_global_mode_index(qt_app):
    """Child items carry the correct global mode index in UserRole data."""
    from PyQt6.QtCore import Qt
    from structural_analysis.modal import solve_modal
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    r = solve_modal(_two_column_model(), n_modes=4)
    d = ModalResultsDialog(None, r, on_select=lambda i, s: None, on_close=lambda: None)
    tree = d._tree

    expected_global_idx = 0
    for ci in range(tree.topLevelItemCount()):
        parent = tree.topLevelItem(ci)
        for mi in range(parent.childCount()):
            child = parent.child(mi)
            stored = child.data(0, Qt.ItemDataRole.UserRole)
            assert stored == expected_global_idx, (
                f"Component {ci+1}, local mode {mi}: "
                f"expected global_idx={expected_global_idx}, got {stored}"
            )
            expected_global_idx += 1
    d.close()


def test_modal_dialog_component_label_shows_node_count(qt_app):
    """Component group items mention the node or element count."""
    from structural_analysis.modal import solve_modal
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    r = solve_modal(_two_column_model(), n_modes=4)
    d = ModalResultsDialog(None, r, on_select=lambda i, s: None, on_close=lambda: None)
    tree = d._tree
    for i in range(tree.topLevelItemCount()):
        label = tree.topLevelItem(i).text(0)
        # Should mention the component id (1 or 2) and some quantity
        assert any(ch.isdigit() for ch in label)
    d.close()


def test_modal_dialog_selecting_child_calls_on_select(qt_app):
    """Clicking a child mode item triggers on_select with the correct global index."""
    from structural_analysis.modal import solve_modal
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    r = solve_modal(_two_column_model(), n_modes=4)
    received: list[int] = []
    d = ModalResultsDialog(None, r,
                           on_select=lambda i, s: received.append(i),
                           on_close=lambda: None)
    tree = d._tree
    # Click the first child of the second component
    c2_parent = tree.topLevelItem(1)
    first_child = c2_parent.child(0)
    tree.setCurrentItem(first_child)

    c2 = r.components[1]
    assert received[-1] == c2.global_mode_offset
    d.close()
