"""Tests for the GUI command pattern — Stage A: AddMemberCmd.

Stage A (v0.10.0) introduces direct member drawing: the Frame / Truss
tools accept clicks on empty space, auto-creating nodes as a side
effect. The atomic do/undo guarantee lives in
:class:`structural_analysis.gui_common.commands.AddMemberCmd`. These
tests pin down the contract:

  - Two empty clicks → 2 nodes + 1 element.
  - One empty, one snapped → 1 new node + 1 element (existing reused).
  - Two snapped clicks → 0 new nodes + 1 element (legacy behaviour).
  - Within 1e-9 of an existing node → reuse silently (tight-zoom case).
  - Two clicks resolving to the same node → ValueError; model untouched.
  - Zero-length / duplicate element → ValueError; auto-created nodes
    rolled back (atomicity).
  - do → undo restores original counts exactly.
  - do → unrelated AddNodeCmd → undo: unrelated node survives.
"""

from __future__ import annotations

import pytest

from structural_analysis.gui_common.commands import (
    AddMemberCmd,
    AddNodeCmd,
)
from structural_analysis.model import (
    Material, Node, NodalLoad, Section, StructuralModel, Support,
)


# ── fixtures ──────────────────────────────────────────────────


def _model_with_material_and_section() -> StructuralModel:
    m = StructuralModel(title="test")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    return m


def _add_node(m: StructuralModel, x: float, y: float) -> int:
    cmd = AddNodeCmd(x=x, y=y)
    cmd.do(m)
    return cmd.node_id  # type: ignore[return-value]


# ── 1. two empty-space clicks ─────────────────────────────────


def test_two_empty_clicks_create_two_nodes_and_one_element():
    m = _model_with_material_and_section()
    cmd = AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=3.0, y_j=0.0,
        kind="frame", section_id=1,
    )
    cmd.do(m)
    assert sorted(m.nodes.keys()) == [1, 2]
    assert m.nodes[1].x == 0.0 and m.nodes[1].y == 0.0
    assert m.nodes[2].x == 3.0 and m.nodes[2].y == 0.0
    assert len(m.elements) == 1
    elem = m.elements[0]
    assert {elem.node_i, elem.node_j} == {1, 2}
    assert cmd._created_node_i == 1
    assert cmd._created_node_j == 2


# ── 2. one empty + one snapped ────────────────────────────────


def test_one_empty_one_snapped_creates_one_node():
    m = _model_with_material_and_section()
    existing = _add_node(m, 0.0, 0.0)  # node 1
    cmd = AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=4.0, y_j=0.0,
        node_i=existing, node_j=None,
        kind="frame", section_id=1,
    )
    cmd.do(m)
    assert sorted(m.nodes.keys()) == [1, 2]
    assert cmd._created_node_i is None
    assert cmd._created_node_j == 2
    elem = m.elements[0]
    assert {elem.node_i, elem.node_j} == {1, 2}


# ── 3. both snapped → existing behaviour ─────────────────────


def test_two_snapped_clicks_create_zero_new_nodes():
    m = _model_with_material_and_section()
    n1 = _add_node(m, 0.0, 0.0)
    n2 = _add_node(m, 5.0, 0.0)
    cmd = AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=5.0, y_j=0.0,
        node_i=n1, node_j=n2,
        kind="frame", section_id=1,
    )
    cmd.do(m)
    assert sorted(m.nodes.keys()) == [n1, n2]
    assert cmd._created_node_i is None
    assert cmd._created_node_j is None
    assert len(m.elements) == 1


# ── 4. tight-zoom: empty click within 1e-9 of existing → reuse ─


def test_empty_click_within_coincidence_threshold_reuses_existing_node():
    """When the snap engine misses (click outside 10 px snap radius) but
    the world coordinate is within 1e-9 of an existing node, AddMemberCmd
    must reuse that node silently rather than raising the add-time
    coincidence block. This is the tight-zoom case from PR review."""
    m = _model_with_material_and_section()
    n1 = _add_node(m, 0.0, 0.0)
    # Second end is empty (node_j=None) but world coords are 1e-10 from
    # a *new* hypothetical node we want at (3.0, 0.0). Add a node there
    # first so the coincidence check has something to find.
    n2 = _add_node(m, 3.0, 0.0)
    cmd = AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=3.0 + 1e-10, y_j=0.0,  # within 1e-9 of n2
        node_i=n1, node_j=None,
        kind="frame", section_id=1,
    )
    cmd.do(m)
    # No new node created — n2 was reused.
    assert sorted(m.nodes.keys()) == [n1, n2]
    assert cmd._created_node_j is None
    elem = m.elements[0]
    assert {elem.node_i, elem.node_j} == {n1, n2}


# ── 5. both clicks resolve to same node ──────────────────────


def test_two_clicks_resolving_to_same_node_raise_and_leave_model_clean():
    m = _model_with_material_and_section()
    n1 = _add_node(m, 0.0, 0.0)
    cmd = AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=0.0, y_j=0.0,
        node_i=n1, node_j=n1,
        kind="frame", section_id=1,
    )
    with pytest.raises(ValueError, match="same"):
        cmd.do(m)
    assert sorted(m.nodes.keys()) == [n1]
    assert m.elements == []


# ── 6. zero-length empty/empty clicks ────────────────────────


def test_zero_length_empty_empty_raises_and_rolls_back_first_node():
    """Two empty clicks at the same coordinate: first node is created,
    then the second resolves to the same node via coincidence reuse,
    then the same-node check raises. The first node must NOT survive
    on the model (atomicity)."""
    m = _model_with_material_and_section()
    cmd = AddMemberCmd(
        x_i=2.0, y_i=2.0, x_j=2.0, y_j=2.0,
        kind="frame", section_id=1,
    )
    with pytest.raises(ValueError):
        cmd.do(m)
    assert m.nodes == {}
    assert m.elements == []


# ── 7. duplicate element rolls back auto-created node ────────


def test_duplicate_element_rolls_back_auto_created_nodes():
    """An existing element between two nodes; the user re-draws a
    member from a new empty point to one of them. AddElementCmd's
    duplicate-element check raises *after* we auto-created the
    new-end node — atomicity must roll it back."""
    m = _model_with_material_and_section()
    n1 = _add_node(m, 0.0, 0.0)
    n2 = _add_node(m, 3.0, 0.0)
    first = AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=3.0, y_j=0.0,
        node_i=n1, node_j=n2,
        kind="frame", section_id=1,
    )
    first.do(m)
    # Now redraw from a NEW empty point that *coincides with n1*, to n2.
    # The first end will resolve to n1 by coincidence; the second to n2;
    # AddElementCmd then sees a duplicate and raises.
    second = AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=3.0, y_j=0.0,
        node_i=None, node_j=n2,
        kind="frame", section_id=1,
    )
    with pytest.raises(ValueError, match="already connects"):
        second.do(m)
    # No new node added by the failed second command.
    assert sorted(m.nodes.keys()) == [n1, n2]
    assert len(m.elements) == 1


# ── 8. round-trip do → undo ──────────────────────────────────


def test_do_then_undo_restores_original_counts():
    m = _model_with_material_and_section()
    cmd = AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=5.0, y_j=2.0,
        kind="frame", section_id=1,
    )
    cmd.do(m)
    assert len(m.nodes) == 2
    assert len(m.elements) == 1
    cmd.undo(m)
    assert m.nodes == {}
    assert m.elements == []
    # And redo recovers identical state.
    cmd.do(m)
    assert len(m.nodes) == 2
    assert len(m.elements) == 1


# ── 9. unrelated edit between do and undo ────────────────────


def test_undo_preserves_unrelated_node_added_after_draw():
    """If the user draws a member, then adds another unrelated node,
    then undoes the draw, the unrelated node must survive. The
    rollback only touches the nodes this command itself created."""
    m = _model_with_material_and_section()
    cmd_member = AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=4.0, y_j=0.0,
        kind="frame", section_id=1,
    )
    cmd_member.do(m)  # creates nodes 1, 2 and element 1
    cmd_extra = AddNodeCmd(x=10.0, y=10.0)
    cmd_extra.do(m)  # creates node 3, unrelated
    assert sorted(m.nodes.keys()) == [1, 2, 3]

    cmd_member.undo(m)
    # Element gone, nodes 1 & 2 gone, node 3 survives.
    assert m.elements == []
    assert sorted(m.nodes.keys()) == [3]


def test_undo_preserves_auto_created_node_if_later_referenced():
    """After the draw, the user adds a support to one of the auto-
    created nodes. Undoing the draw must remove the element but leave
    the node in place so the support stays valid — the user's later
    work is more important than a clean undo."""
    m = _model_with_material_and_section()
    cmd_member = AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=4.0, y_j=0.0,
        kind="frame", section_id=1,
    )
    cmd_member.do(m)
    auto_i = cmd_member._created_node_i
    assert auto_i is not None
    m.supports[auto_i] = Support(node_id=auto_i, ux=True, uy=True, rz=True)

    cmd_member.undo(m)
    # Element removed.
    assert m.elements == []
    # The supported node stays (preserves the user's later edit).
    assert auto_i in m.nodes
    # The unsupported auto-created node is gone.
    auto_j = cmd_member._created_node_j
    assert auto_j is not None
    assert auto_j not in m.nodes
