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

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.gui_common.commands import (
    AddMemberCmd,
    AddMemberLoadCmd,
    AddNodeCmd,
    BatchDeleteCmd,
    BatchUpdateElementsCmd,
    check_merge_preconditions,
    CLEAR_MATERIAL_OVERRIDE,
    DeleteMemberLoadCmd,
    MergeAdjacentElementsCmd,
    RenumberElementsCmd,
    SetNodalLoadCmd,
    SetSupportCmd,
    UpdateMemberLoadCmd,
    DrawMemberWithSplitsCmd,
    SplitElementCmd,
)
from structural_analysis.model import (
    FrameTemperatureLoad,
    Material,
    NodalLoad,
    PointLoad,
    Section,
    StructuralModel,
    Support,
    TrussTemperatureLoad,
    UniformDistributedLoad,
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


# ── SplitElementCmd (PR #21 Stage C) ─────────────────────────


def _frame_model_one_member(
    *,
    release_i: bool = False,
    release_j: bool = False,
) -> tuple[StructuralModel, int]:
    """Return (model, element_id) with a single 6 m horizontal frame
    from (0,0) to (6,0) and a real material+section. Optional release
    flags so the release-preservation test can flex them."""
    m = _model_with_material_and_section()
    AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=6.0, y_j=0.0,
        kind="frame", section_id=1,
        release_i=release_i, release_j=release_j,
    ).do(m)
    assert len(m.elements) == 1
    return m, m.elements[0].id


def _truss_model_one_member() -> tuple[StructuralModel, int]:
    m = _model_with_material_and_section()
    AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=4.0, y_j=0.0,
        kind="truss", section_id=1,
    ).do(m)
    return m, m.elements[0].id


def test_split_frame_at_midspan_creates_two_children_and_one_new_node():
    m, eid = _frame_model_one_member()
    cmd = SplitElementCmd(element_id=eid, x=3.0, y=0.0)
    cmd.do(m)
    # Parent gone, two children present.
    assert eid not in [e.id for e in m.elements]
    assert len(m.elements) == 2
    # One new node at the midpoint.
    assert len(m.nodes) == 3
    new_node = m.nodes[cmd._created_node_c]
    assert (new_node.x, new_node.y) == (3.0, 0.0)
    # Children chain through the new node.
    a, b = m.elements
    assert a.node_j == new_node.id
    assert b.node_i == new_node.id


def test_split_frame_copies_section_material_overrides_and_inner_end_releases():
    """The user's contract: outer-end releases stay on the matching
    child, inner ends get no release."""
    m, eid = _frame_model_one_member(release_i=True, release_j=True)
    # Add a second material and set it as override on the parent so
    # we can verify it propagates to the children.
    m.materials[2] = Material(id=2, name="Alt", E=1.5e8, density=2400.0)
    parent = m.elements[0]
    parent.material_id_override = 2

    cmd = SplitElementCmd(element_id=eid, x=2.0, y=0.0)
    cmd.do(m)
    a, b = m.elements
    # Property propagation.
    assert a.section_id == parent.section_id == 1
    assert b.section_id == parent.section_id == 1
    assert a.material_id_override == 2
    assert b.material_id_override == 2
    # Inner-end-loses-release: A's outer end (node_i) keeps the
    # release; A's inner end (node_j == C) loses it. Same for B.
    assert isinstance(a, FrameElement2D)
    assert isinstance(b, FrameElement2D)
    assert (a.release_i, a.release_j) == (True, False)
    assert (b.release_i, b.release_j) == (False, True)


def test_split_truss_preserves_kind():
    m, eid = _truss_model_one_member()
    cmd = SplitElementCmd(element_id=eid, x=2.0, y=0.0)
    cmd.do(m)
    a, b = m.elements
    assert isinstance(a, TrussElement2D)
    assert isinstance(b, TrussElement2D)


def test_split_then_undo_restores_parent_and_removes_auto_node():
    m, eid = _frame_model_one_member()
    snapshot_nodes = sorted(m.nodes.keys())
    snapshot_elem_ids = [e.id for e in m.elements]
    snapshot_elem_ends = [(e.node_i, e.node_j) for e in m.elements]

    cmd = SplitElementCmd(element_id=eid, x=3.0, y=0.0)
    cmd.do(m)
    cmd.undo(m)

    assert sorted(m.nodes.keys()) == snapshot_nodes
    assert [e.id for e in m.elements] == snapshot_elem_ids
    assert [(e.node_i, e.node_j) for e in m.elements] == snapshot_elem_ends
    # And redo recovers the split state exactly.
    cmd.do(m)
    assert len(m.elements) == 2


def test_split_too_close_to_endpoint_raises_and_leaves_model_clean():
    m, eid = _frame_model_one_member()
    # ELEMENT_SPLIT_TOL is 1e-6 on parametric t; on a 6 m bar that's
    # 6e-6 m. A click at world-x = 1e-7 puts t ≈ 1.7e-8 — well below
    # the tolerance.
    cmd = SplitElementCmd(element_id=eid, x=1e-7, y=0.0)
    with pytest.raises(ValueError, match="too close to an endpoint"):
        cmd.do(m)
    # Model unchanged.
    assert len(m.elements) == 1
    assert len(m.nodes) == 2


def test_split_off_the_segment_raises_when_t_outside_unit_interval():
    """The strict-interior tolerance also rejects t < 0 and t > 1
    (caller may have asked for a split point past either endpoint).
    The current implementation lumps this under the same 'too close
    to an endpoint' message — accept either wording."""
    m, eid = _frame_model_one_member()
    with pytest.raises(ValueError):
        SplitElementCmd(element_id=eid, x=-1.0, y=0.0).do(m)
    assert len(m.elements) == 1
    with pytest.raises(ValueError):
        SplitElementCmd(element_id=eid, x=7.0, y=0.0).do(m)
    assert len(m.elements) == 1


# The pre-0.12.0 "split blocks on member loads" tests were removed;
# the behavior is now remapping, exercised by the
# "SplitElementCmd: member-load remap" section below
# (``test_split_loaded_frame_*`` / ``test_split_loaded_truss_*``).


def test_split_missing_element_id_raises():
    m = _model_with_material_and_section()
    with pytest.raises(ValueError, match="does not exist"):
        SplitElementCmd(element_id=99, x=0.0, y=0.0).do(m)


def test_split_undo_preserves_auto_node_if_later_attached():
    """If the user splits, then attaches a support / nodal load to
    the auto-created mid node, undoing the split must leave that
    node alive — same defensive rule as AddMemberCmd."""
    m, eid = _frame_model_one_member()
    cmd = SplitElementCmd(element_id=eid, x=3.0, y=0.0)
    cmd.do(m)
    mid = cmd._created_node_c
    assert mid is not None
    # Pretend the user added a support at the mid node.
    m.supports[mid] = Support(node_id=mid, ux=True, uy=True, rz=True)

    cmd.undo(m)
    # Parent restored, children gone — but the mid node survives
    # because the support depended on it.
    assert mid in m.nodes
    assert len(m.elements) == 1
    assert m.elements[0].id == eid


def test_split_rejects_node_id_hint_far_from_projected_point():
    """Defensive guard (PR #21 review): a node_id hint pointing at an
    off-element node must be rejected, not silently used. Without the
    coordinate check, _find_or_create_node would accept the hint by id
    alone and produce geometrically incoherent children."""
    m, eid = _frame_model_one_member()  # frame 1→2 along y=0
    # Add a free node far from the segment.
    free = _add_node(m, 10.0, 10.0)
    n_nodes_before = len(m.nodes)
    elem_ids_before = [e.id for e in m.elements]

    with pytest.raises(ValueError, match="does not lie on element"):
        SplitElementCmd(
            element_id=eid, x=3.0, y=0.0, node_id=free,
        ).do(m)
    # Atomic-rollback: parent intact, free node untouched.
    assert len(m.nodes) == n_nodes_before
    assert [e.id for e in m.elements] == elem_ids_before


# ── DrawMemberWithSplitsCmd (PR #21 follow-up: grouped undo) ──


def _two_parallel_bars() -> tuple[StructuralModel, int, int]:
    """Return (model, lower_id, upper_id): two horizontal 0..6 frames,
    the lower at y=0 and the upper at y=4. Drawing a vertical member
    between their midspans bisects both."""
    m = _model_with_material_and_section()
    AddMemberCmd(x_i=0.0, y_i=0.0, x_j=6.0, y_j=0.0,
                 kind="frame", section_id=1).do(m)
    lower = m.elements[-1].id
    AddMemberCmd(x_i=0.0, y_i=4.0, x_j=6.0, y_j=4.0,
                 kind="frame", section_id=1).do(m)
    upper = m.elements[-1].id
    return m, lower, upper


def test_draw_member_bisecting_two_elements_runs_two_splits_and_one_member():
    """Worst case: both endpoints land on element interiors. One
    composite do() must split both parents and add the connecting
    member; one undo() must reverse all three; redo replays."""
    m, lower, upper = _two_parallel_bars()
    assert (len(m.nodes), len(m.elements)) == (4, 2)

    cmd = DrawMemberWithSplitsCmd(
        split_target_i=(lower, 3.0, 0.0),
        split_target_j=(upper, 3.0, 4.0),
        x_i=3.0, y_i=0.0, x_j=3.0, y_j=4.0,
        kind="frame", section_id=1,
    )
    cmd.do(m)
    # 4 originals + 2 split nodes; 2 parents → 4 children + 1 member.
    assert (len(m.nodes), len(m.elements)) == (6, 5)
    assert lower not in [e.id for e in m.elements]
    assert upper not in [e.id for e in m.elements]
    # The new member connects the two freshly-created split nodes.
    mid_lo = next(n for n in m.nodes.values() if (n.x, n.y) == (3.0, 0.0))
    mid_hi = next(n for n in m.nodes.values() if (n.x, n.y) == (3.0, 4.0))
    member = next(
        e for e in m.elements
        if {e.node_i, e.node_j} == {mid_lo.id, mid_hi.id}
    )
    assert member is not None

    # One undo reverses everything.
    cmd.undo(m)
    assert (len(m.nodes), len(m.elements)) == (4, 2)
    assert lower in [e.id for e in m.elements]
    assert upper in [e.id for e in m.elements]

    # Redo replays identically.
    cmd.do(m)
    assert (len(m.nodes), len(m.elements)) == (6, 5)


def test_draw_member_one_interior_one_node_snap_runs_single_split():
    """One endpoint on an element interior, the other on an existing
    node: exactly one split + one member, reversed in one step."""
    m, _, _ = _two_parallel_bars()
    # Remove the upper bar so only the lower one is interesting, and
    # add a free node to snap the member's far end onto.
    upper = m.elements[-1]
    m.elements.remove(upper)
    free = _add_node(m, 3.0, 5.0)
    lower = m.elements[0].id
    # Now: nodes {0,0; 6,0; 0,4; 6,4; 3,5}; one element (lower bar).
    n_nodes_before = len(m.nodes)
    assert len(m.elements) == 1

    cmd = DrawMemberWithSplitsCmd(
        split_target_i=(lower, 3.0, 0.0),
        split_target_j=None,
        x_i=3.0, y_i=0.0, x_j=3.0, y_j=5.0, node_j_hint=free,
        kind="frame", section_id=1,
    )
    cmd.do(m)
    # One split node added; parent → 2 children + 1 member.
    assert len(m.nodes) == n_nodes_before + 1
    assert len(m.elements) == 3
    mid = next(n for n in m.nodes.values() if (n.x, n.y) == (3.0, 0.0))
    assert any(
        {e.node_i, e.node_j} == {mid.id, free} for e in m.elements
    )

    cmd.undo(m)
    assert len(m.nodes) == n_nodes_before
    assert len(m.elements) == 1
    assert m.elements[0].id == lower


def test_draw_member_with_no_split_targets_degenerates_to_add_member():
    """Both split targets None ⇒ the composite behaves like a plain
    AddMemberCmd (auto-creates the two endpoint nodes)."""
    m = _model_with_material_and_section()
    cmd = DrawMemberWithSplitsCmd(
        split_target_i=None, split_target_j=None,
        x_i=0.0, y_i=0.0, x_j=6.0, y_j=0.0,
        kind="frame", section_id=1,
    )
    cmd.do(m)
    assert (len(m.nodes), len(m.elements)) == (2, 1)
    cmd.undo(m)
    assert (len(m.nodes), len(m.elements)) == (0, 0)


def test_draw_member_rolls_back_first_split_when_second_split_blocked():
    """If the second split is rejected (parent carries an unsupported
    member-load type), the first split is reversed and the model is
    left exactly as it began — no member, no orphaned split node.

    v0.12.0 update: UDL / PointLoad / thermal no longer block; this
    test now uses a synthetic load type to exercise the
    unsupported-type branch of ``_remap_member_loads``.
    """
    from dataclasses import dataclass as _dc

    @_dc
    class _UnsupportedLoad:
        x: float = 0.0

    m, lower, upper = _two_parallel_bars()
    # Block the upper bar with an unsupported member load type.
    m.elements[1].member_loads.append(_UnsupportedLoad())

    cmd = DrawMemberWithSplitsCmd(
        split_target_i=(lower, 3.0, 0.0),
        split_target_j=(upper, 3.0, 4.0),
        x_i=3.0, y_i=0.0, x_j=3.0, y_j=4.0,
        kind="frame", section_id=1,
    )
    with pytest.raises(ValueError, match="not yet supported"):
        cmd.do(m)
    # Fully rolled back: both parents whole, no split nodes, no member.
    assert (len(m.nodes), len(m.elements)) == (4, 2)
    assert lower in [e.id for e in m.elements]
    assert upper in [e.id for e in m.elements]


def test_draw_member_rolls_back_both_splits_when_member_add_fails():
    """If the inner AddMemberCmd raises, every preceding split is
    reversed and the model is untouched.

    Setup: two bars cross at (3,0). Splitting both at that point makes
    the second split reuse the first's freshly-created node (the
    coincidence-dedup rule), so the inner member would connect a node
    to itself → AddMemberCmd raises *after* both splits succeeded.
    The composite must still unwind both."""
    m = _model_with_material_and_section()
    # Two bars crossing at (3,0): one horizontal, one vertical.
    AddMemberCmd(x_i=0.0, y_i=0.0, x_j=6.0, y_j=0.0,
                 kind="frame", section_id=1).do(m)
    horiz = m.elements[-1].id
    AddMemberCmd(x_i=3.0, y_i=-3.0, x_j=3.0, y_j=3.0,
                 kind="frame", section_id=1).do(m)
    vert = m.elements[-1].id
    assert (len(m.nodes), len(m.elements)) == (4, 2)

    cmd = DrawMemberWithSplitsCmd(
        split_target_i=(horiz, 3.0, 0.0),
        split_target_j=(vert, 3.0, 0.0),
        x_i=3.0, y_i=0.0, x_j=3.0, y_j=0.0,
        kind="frame", section_id=1,
    )
    with pytest.raises(ValueError, match="cannot be the same"):
        cmd.do(m)
    # Both splits rolled back: 4 nodes, 2 elements, both parents whole.
    assert (len(m.nodes), len(m.elements)) == (4, 2)
    assert horiz in [e.id for e in m.elements]
    assert vert in [e.id for e in m.elements]


# ── SplitElementCmd: member-load remap (v0.12.0 — Feature B) ──


def test_split_loaded_frame_udl_copies_intensity_to_both_children():
    m, eid = _frame_model_one_member()
    udl = UniformDistributedLoad(wy=-10.0)
    m.elements[0].member_loads.append(udl)
    cmd = SplitElementCmd(element_id=eid, x=3.0, y=0.0)
    cmd.do(m)
    a, b = m.elements
    a_udls = [ld for ld in a.member_loads
              if isinstance(ld, UniformDistributedLoad)]
    b_udls = [ld for ld in b.member_loads
              if isinstance(ld, UniformDistributedLoad)]
    assert len(a_udls) == 1 and a_udls[0].wy == -10.0
    assert len(b_udls) == 1 and b_udls[0].wy == -10.0


def test_split_loaded_frame_point_load_left_of_split_maps_to_child_a():
    m, eid = _frame_model_one_member()
    m.elements[0].member_loads.append(PointLoad(py=-5.0, a=2.0))
    cmd = SplitElementCmd(element_id=eid, x=3.0, y=0.0)  # midspan
    cmd.do(m)
    a, b = m.elements
    a_pts = [ld for ld in a.member_loads if isinstance(ld, PointLoad)]
    b_pts = [ld for ld in b.member_loads if isinstance(ld, PointLoad)]
    assert len(a_pts) == 1
    assert a_pts[0].py == -5.0
    assert a_pts[0].a == 2.0
    assert b_pts == []


def test_split_loaded_frame_point_load_right_of_split_maps_to_child_b_with_shifted_a():
    m, eid = _frame_model_one_member()
    m.elements[0].member_loads.append(PointLoad(py=-5.0, a=4.5))
    cmd = SplitElementCmd(element_id=eid, x=3.0, y=0.0)  # midspan, L1=3
    cmd.do(m)
    a, b = m.elements
    a_pts = [ld for ld in a.member_loads if isinstance(ld, PointLoad)]
    b_pts = [ld for ld in b.member_loads if isinstance(ld, PointLoad)]
    assert a_pts == []
    assert len(b_pts) == 1
    assert b_pts[0].py == -5.0
    assert b_pts[0].a == pytest.approx(1.5)


def test_split_loaded_frame_point_load_at_split_assigns_to_child_a_at_its_full_length():
    m, eid = _frame_model_one_member()
    m.elements[0].member_loads.append(PointLoad(py=-5.0, a=3.0))
    cmd = SplitElementCmd(element_id=eid, x=3.0, y=0.0)
    cmd.do(m)
    a, b = m.elements
    a_pts = [ld for ld in a.member_loads if isinstance(ld, PointLoad)]
    b_pts = [ld for ld in b.member_loads if isinstance(ld, PointLoad)]
    assert b_pts == []
    assert len(a_pts) == 1
    # Child A's actual euclidean length.
    import math as _math
    ni = m.nodes[a.node_i]
    nj = m.nodes[a.node_j]
    L_child_a = _math.hypot(nj.x - ni.x, nj.y - ni.y)
    assert a_pts[0].a == pytest.approx(L_child_a, abs=1e-12)


def test_split_loaded_point_load_just_past_split_routes_to_child_b_not_snapped():
    """A PointLoad placed a finite (non-FP-roundoff) distance past
    the split must route to child B with its true offset, not be
    snapped to child A's endpoint. Regression for codex P2 finding
    on PR #22: with the old ``ELEMENT_SPLIT_TOL * L_parent`` band, on
    a long enough member a load within ``1e-6 * L_parent`` of the
    split was incorrectly snapped to child A's end. The new band is
    a pure FP-roundoff tolerance, so a 1 mm offset routes cleanly to
    B regardless of member length."""
    m, eid = _frame_model_one_member()  # 6 m horizontal frame
    m.elements[0].member_loads.append(PointLoad(py=-5.0, a=3.0 + 1e-3))
    cmd = SplitElementCmd(element_id=eid, x=3.0, y=0.0)
    cmd.do(m)
    a, b = m.elements
    a_pts = [ld for ld in a.member_loads if isinstance(ld, PointLoad)]
    b_pts = [ld for ld in b.member_loads if isinstance(ld, PointLoad)]
    assert a_pts == []
    assert len(b_pts) == 1
    assert b_pts[0].a == pytest.approx(1e-3, abs=1e-12)


def test_split_loaded_frame_thermal_copies_to_both_children():
    m, eid = _frame_model_one_member()
    tload = FrameTemperatureLoad(t_top=20.0, t_bottom=-20.0)
    m.elements[0].member_loads.append(tload)
    cmd = SplitElementCmd(element_id=eid, x=3.0, y=0.0)
    cmd.do(m)
    a, b = m.elements
    a_th = [ld for ld in a.member_loads
            if isinstance(ld, FrameTemperatureLoad)]
    b_th = [ld for ld in b.member_loads
            if isinstance(ld, FrameTemperatureLoad)]
    assert len(a_th) == 1 and a_th[0] == tload
    assert len(b_th) == 1 and b_th[0] == tload


def test_split_loaded_truss_thermal_copies_to_both_children():
    m, eid = _truss_model_one_member()
    tload = TrussTemperatureLoad(delta_T=25.0)
    m.elements[0].member_loads.append(tload)
    cmd = SplitElementCmd(element_id=eid, x=2.0, y=0.0)
    cmd.do(m)
    a, b = m.elements
    a_th = [ld for ld in a.member_loads
            if isinstance(ld, TrussTemperatureLoad)]
    b_th = [ld for ld in b.member_loads
            if isinstance(ld, TrussTemperatureLoad)]
    assert len(a_th) == 1 and a_th[0] == tload
    assert len(b_th) == 1 and b_th[0] == tload


def test_split_loaded_element_with_unsupported_load_type_still_blocks():
    from dataclasses import dataclass as _dc

    @_dc
    class FakeLoad:
        x: float = 0.0

    m, eid = _frame_model_one_member()
    m.elements[0].member_loads.append(FakeLoad())
    snapshot_nodes = sorted(m.nodes.keys())
    snapshot_elem_ids = [e.id for e in m.elements]
    cmd = SplitElementCmd(element_id=eid, x=3.0, y=0.0)
    with pytest.raises(ValueError, match="not yet supported"):
        cmd.do(m)
    # Model untouched (atomic).
    assert sorted(m.nodes.keys()) == snapshot_nodes
    assert [e.id for e in m.elements] == snapshot_elem_ids


def test_split_loaded_element_undo_restores_parent_loads_intact():
    m, eid = _frame_model_one_member()
    udl = UniformDistributedLoad(wy=-7.0)
    pt = PointLoad(py=-3.0, a=2.0)
    m.elements[0].member_loads.extend([udl, pt])
    saved_loads = list(m.elements[0].member_loads)

    cmd = SplitElementCmd(element_id=eid, x=3.0, y=0.0)
    cmd.do(m)
    assert len(m.elements) == 2
    cmd.undo(m)
    assert len(m.elements) == 1
    # The parent's member_loads list should still hold the same
    # entries (by identity or equality).
    restored = m.elements[0].member_loads
    assert list(restored) == saved_loads


def test_split_loaded_element_redo_replays_remapping():
    m, eid = _frame_model_one_member()
    udl = UniformDistributedLoad(wy=-4.0)
    pt_left = PointLoad(py=-2.0, a=1.0)
    pt_right = PointLoad(py=-2.5, a=5.0)
    m.elements[0].member_loads.extend([udl, pt_left, pt_right])

    cmd = SplitElementCmd(element_id=eid, x=3.0, y=0.0)
    cmd.do(m)
    a1, b1 = m.elements

    def _summarize(elem):
        return sorted(
            (
                type(ld).__name__,
                tuple(sorted(ld.__dict__.items()))
                if hasattr(ld, "__dict__")
                else tuple(),
                getattr(ld, "wy", None),
                getattr(ld, "py", None),
                getattr(ld, "a", None),
            )
            for ld in elem.member_loads
        )

    a1_summary = _summarize(a1)
    b1_summary = _summarize(b1)

    cmd.undo(m)
    cmd.do(m)
    a2, b2 = m.elements
    assert _summarize(a2) == a1_summary
    assert _summarize(b2) == b1_summary


# ── batch ops (v0.13.0) ────────────────────────────────────────


def _two_section_model() -> tuple[StructuralModel, list[int]]:
    """Build a 3-element frame with two sections so batch tests can
    distinguish per-element identity. Sections 1 and 2 default to
    material 1; material 2 exists for override tests."""
    m = StructuralModel(title="batch")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.materials[2] = Material(id=2, name="Alu", E=7.0e7, density=2700.0)
    m.sections[1] = Section(id=1, name="S1", material_id=1, A=0.01, I=1e-4, depth=0.3)
    m.sections[2] = Section(id=2, name="S2", material_id=1, A=0.02, I=2e-4, depth=0.4)
    AddMemberCmd(x_i=0.0, y_i=0.0, x_j=2.0, y_j=0.0,
                 kind="frame", section_id=1).do(m)
    AddMemberCmd(x_i=2.0, y_i=0.0, x_j=4.0, y_j=0.0,
                 kind="frame", section_id=2).do(m)
    AddMemberCmd(x_i=4.0, y_i=0.0, x_j=6.0, y_j=0.0,
                 kind="frame", section_id=1).do(m)
    return m, [e.id for e in m.elements]


def test_batch_update_changes_section_on_selected_elements_only():
    m, eids = _two_section_model()
    # Apply section=2 to only the first and third elements.
    target = [eids[0], eids[2]]
    cmd = BatchUpdateElementsCmd(element_ids=target, section_id=2)
    cmd.do(m)
    sections = {e.id: e.section_id for e in m.elements}
    assert sections == {eids[0]: 2, eids[1]: 2, eids[2]: 2}  # 0,2 changed; 1 was already 2


def test_batch_update_undo_restores_original_sections():
    m, eids = _two_section_model()
    before = {e.id: e.section_id for e in m.elements}
    cmd = BatchUpdateElementsCmd(element_ids=eids, section_id=2)
    cmd.do(m)
    assert all(e.section_id == 2 for e in m.elements)
    cmd.undo(m)
    after = {e.id: e.section_id for e in m.elements}
    assert after == before


def test_batch_update_redo_replays_assignment():
    m, eids = _two_section_model()
    cmd = BatchUpdateElementsCmd(element_ids=eids, section_id=2)
    cmd.do(m)
    cmd.undo(m)
    cmd.do(m)
    assert all(e.section_id == 2 for e in m.elements)


def test_batch_update_leave_unchanged_does_not_overwrite_mixed_sections():
    """section_id=None means 'leave each element's section alone'.
    A user updating only the override across elements of varying
    sections must not have their distinct sections clobbered."""
    m, eids = _two_section_model()
    before_sections = {e.id: e.section_id for e in m.elements}
    cmd = BatchUpdateElementsCmd(
        element_ids=eids,
        section_id=None,
        material_override_id=2,
    )
    cmd.do(m)
    after_sections = {e.id: e.section_id for e in m.elements}
    assert after_sections == before_sections  # each section preserved
    assert all(
        getattr(e, "material_id_override", None) == 2
        for e in m.elements
    )


def test_batch_update_clears_material_override_when_sentinel():
    m, eids = _two_section_model()
    # First set an override on all elements.
    BatchUpdateElementsCmd(
        element_ids=eids, material_override_id=2,
    ).do(m)
    assert all(
        getattr(e, "material_id_override", None) == 2 for e in m.elements
    )
    # Now clear it via the sentinel.
    BatchUpdateElementsCmd(
        element_ids=eids, material_override_id=CLEAR_MATERIAL_OVERRIDE,
    ).do(m)
    assert all(
        getattr(e, "material_id_override", None) is None
        for e in m.elements
    )


def test_batch_update_empty_list_is_noop():
    m, _ = _two_section_model()
    snapshot = {e.id: e.section_id for e in m.elements}
    BatchUpdateElementsCmd(element_ids=[], section_id=2).do(m)
    assert {e.id: e.section_id for e in m.elements} == snapshot


def test_batch_update_both_fields_none_is_noop():
    """When both fields say 'leave unchanged' the command is a no-op
    and must not push side effects."""
    m, eids = _two_section_model()
    before = {e.id: e.section_id for e in m.elements}
    BatchUpdateElementsCmd(
        element_ids=eids, section_id=None, material_override_id=None,
    ).do(m)
    assert {e.id: e.section_id for e in m.elements} == before


def test_batch_update_invalid_section_raises_before_mutating():
    m, eids = _two_section_model()
    before = [e.section_id for e in m.elements]
    with pytest.raises(ValueError):
        BatchUpdateElementsCmd(
            element_ids=eids, section_id=99,  # does not exist
        ).do(m)
    assert [e.section_id for e in m.elements] == before


def test_batch_delete_removes_elements_and_nodes_one_undo_step():
    m, eids = _two_section_model()
    node_ids = list(m.nodes.keys())
    # Delete the middle element and the last node.
    cmd = BatchDeleteCmd(node_ids=[node_ids[-1]], element_ids=[eids[1]])
    n_elems_before = len(m.elements)
    n_nodes_before = len(m.nodes)
    cmd.do(m)
    # Middle element gone + last node + its cascade (third element).
    assert len(m.elements) == n_elems_before - 2  # middle gone + cascade
    assert len(m.nodes) == n_nodes_before - 1
    cmd.undo(m)
    assert len(m.elements) == n_elems_before
    assert len(m.nodes) == n_nodes_before


def test_batch_delete_skips_already_cascade_deleted_node():
    """When delete_node cascade already removes a node referenced in
    the batch, the next iteration must skip gracefully (no
    KeyError)."""
    m, eids = _two_section_model()
    node_ids = list(m.nodes.keys())
    # Delete two adjacent nodes — the second one's connected elements
    # were already cascade-removed by the first delete.
    cmd = BatchDeleteCmd(
        node_ids=[node_ids[0], node_ids[1]], element_ids=[],
    )
    cmd.do(m)  # must not raise
    assert node_ids[0] not in m.nodes
    assert node_ids[1] not in m.nodes


def test_batch_delete_empty_lists_is_noop():
    m, _ = _two_section_model()
    n_e = len(m.elements)
    n_n = len(m.nodes)
    BatchDeleteCmd(node_ids=[], element_ids=[]).do(m)
    assert len(m.elements) == n_e
    assert len(m.nodes) == n_n


# ── DeleteMemberLoadCmd (PR #24) ────────────────────────────────────


def _frame_with_three_loads() -> tuple[StructuralModel, int]:
    """One frame element with UDL, PointLoad, FrameTemperatureLoad in
    that order — fixture for the per-row delete tests."""
    m, eid = _frame_model_one_member()
    elem = m.elements[0]
    elem.member_loads.append(UniformDistributedLoad(wy=-10.0))
    elem.member_loads.append(PointLoad(py=-20.0, a=2.0))
    elem.member_loads.append(FrameTemperatureLoad(t_top=10.0, t_bottom=30.0))
    return m, eid


def test_delete_member_load_removes_only_target_index():
    m, eid = _frame_with_three_loads()
    DeleteMemberLoadCmd(elem_id=eid, load_index=1).do(m)
    loads = m.elements[0].member_loads
    assert len(loads) == 2
    assert isinstance(loads[0], UniformDistributedLoad)
    assert isinstance(loads[1], FrameTemperatureLoad)


def test_delete_member_load_undo_restores_at_same_index():
    m, eid = _frame_with_three_loads()
    cmd = DeleteMemberLoadCmd(elem_id=eid, load_index=1)
    cmd.do(m)
    cmd.undo(m)
    loads = m.elements[0].member_loads
    assert len(loads) == 3
    assert isinstance(loads[0], UniformDistributedLoad)
    assert isinstance(loads[1], PointLoad)
    assert loads[1].py == -20.0 and loads[1].a == 2.0
    assert isinstance(loads[2], FrameTemperatureLoad)


def test_delete_member_load_redo_replays():
    m, eid = _frame_with_three_loads()
    cmd = DeleteMemberLoadCmd(elem_id=eid, load_index=0)
    cmd.do(m)
    cmd.undo(m)
    cmd.do(m)
    loads = m.elements[0].member_loads
    assert len(loads) == 2
    assert isinstance(loads[0], PointLoad)
    assert isinstance(loads[1], FrameTemperatureLoad)


def test_delete_member_load_invalid_index_raises_no_mutation():
    m, eid = _frame_with_three_loads()
    snapshot = list(m.elements[0].member_loads)
    with pytest.raises(ValueError):
        DeleteMemberLoadCmd(elem_id=eid, load_index=99).do(m)
    assert m.elements[0].member_loads == snapshot


def test_delete_member_load_negative_index_raises():
    m, eid = _frame_with_three_loads()
    with pytest.raises(ValueError):
        DeleteMemberLoadCmd(elem_id=eid, load_index=-1).do(m)


def test_delete_member_load_unknown_element_raises():
    m, _ = _frame_with_three_loads()
    with pytest.raises(ValueError):
        DeleteMemberLoadCmd(elem_id=9999, load_index=0).do(m)


# ── UpdateMemberLoadCmd (PR #35) ────────────────────────────────────


def test_update_member_load_swaps_in_place():
    """Replace the middle load with a new UDL; the other two rows and the
    list length stay put."""
    m, eid = _frame_with_three_loads()
    new = UniformDistributedLoad(wy=-99.0)
    UpdateMemberLoadCmd(
        elem_id=eid, load_index=1, new_load=new,
    ).do(m)
    loads = m.elements[0].member_loads
    assert len(loads) == 3
    assert isinstance(loads[0], UniformDistributedLoad)
    assert loads[1] is new
    assert isinstance(loads[2], FrameTemperatureLoad)


def test_update_member_load_undo_restores_exact_instance():
    """Undo restores by identity (the EXACT old instance), not just by
    value — matters when the user edits a load that other code already
    holds a reference to."""
    m, eid = _frame_with_three_loads()
    original = m.elements[0].member_loads[1]
    cmd = UpdateMemberLoadCmd(
        elem_id=eid, load_index=1,
        new_load=PointLoad(py=-5.0, a=1.0),
    )
    cmd.do(m)
    cmd.undo(m)
    assert m.elements[0].member_loads[1] is original


def test_update_member_load_redo_replays():
    m, eid = _frame_with_three_loads()
    new = UniformDistributedLoad(wy=-1.0, wx=2.0)
    cmd = UpdateMemberLoadCmd(elem_id=eid, load_index=0, new_load=new)
    cmd.do(m)
    cmd.undo(m)
    cmd.do(m)
    assert m.elements[0].member_loads[0] is new


def test_update_member_load_invalid_index_atomic_no_mutation():
    m, eid = _frame_with_three_loads()
    snapshot = list(m.elements[0].member_loads)
    with pytest.raises(ValueError):
        UpdateMemberLoadCmd(
            elem_id=eid, load_index=99,
            new_load=UniformDistributedLoad(wy=-1.0),
        ).do(m)
    assert m.elements[0].member_loads == snapshot


def test_update_member_load_negative_index_raises():
    m, eid = _frame_with_three_loads()
    with pytest.raises(ValueError):
        UpdateMemberLoadCmd(
            elem_id=eid, load_index=-1,
            new_load=UniformDistributedLoad(wy=-1.0),
        ).do(m)


def test_update_member_load_unknown_element_raises():
    m, _ = _frame_with_three_loads()
    with pytest.raises(ValueError):
        UpdateMemberLoadCmd(
            elem_id=9999, load_index=0,
            new_load=UniformDistributedLoad(wy=-1.0),
        ).do(m)


def test_update_member_load_truss_rejects_frame_thermal():
    """Mirror of AddMemberLoadCmd's element-type guard — atomic refusal."""
    m, eid = _truss_model_one_member()
    m.elements[0].member_loads.append(TrussTemperatureLoad(delta_T=20.0))
    snapshot = list(m.elements[0].member_loads)
    with pytest.raises(ValueError, match="FrameTemperatureLoad"):
        UpdateMemberLoadCmd(
            elem_id=eid, load_index=0,
            new_load=FrameTemperatureLoad(t_top=10.0, t_bottom=30.0),
        ).do(m)
    assert m.elements[0].member_loads == snapshot


def test_update_member_load_frame_rejects_truss_thermal():
    m, eid = _frame_model_one_member()
    m.elements[0].member_loads.append(FrameTemperatureLoad(t_top=10.0, t_bottom=30.0))
    snapshot = list(m.elements[0].member_loads)
    with pytest.raises(ValueError, match="TrussTemperatureLoad"):
        UpdateMemberLoadCmd(
            elem_id=eid, load_index=0,
            new_load=TrussTemperatureLoad(delta_T=20.0),
        ).do(m)
    assert m.elements[0].member_loads == snapshot


def test_delete_member_load_does_not_affect_other_elements():
    m, eid_a = _frame_with_three_loads()
    # Add a second element with its own loads.
    AddMemberCmd(
        x_i=10.0, y_i=0.0, x_j=16.0, y_j=0.0,
        kind="frame", section_id=1,
    ).do(m)
    other = next(e for e in m.elements if e.id != eid_a)
    other.member_loads.append(UniformDistributedLoad(wy=-5.0))
    other.member_loads.append(PointLoad(py=-3.0, a=1.0))
    DeleteMemberLoadCmd(elem_id=eid_a, load_index=0).do(m)
    # Other element's loads untouched.
    assert len(other.member_loads) == 2
    assert isinstance(other.member_loads[0], UniformDistributedLoad)
    assert other.member_loads[0].wy == -5.0
    assert isinstance(other.member_loads[1], PointLoad)


def test_delete_one_thermal_load_keeps_other_thermals():
    """Two FrameTemperatureLoads in a row — delete one, the other
    survives at its new (shifted) index."""
    m, eid = _frame_model_one_member()
    elem = m.elements[0]
    elem.member_loads.append(FrameTemperatureLoad(t_top=10.0, t_bottom=10.0))
    elem.member_loads.append(FrameTemperatureLoad(t_top=20.0, t_bottom=20.0))
    elem.member_loads.append(FrameTemperatureLoad(t_top=30.0, t_bottom=30.0))
    DeleteMemberLoadCmd(elem_id=eid, load_index=1).do(m)
    loads = elem.member_loads
    assert len(loads) == 2
    assert loads[0].t_top == 10.0
    assert loads[1].t_top == 30.0


def test_delete_member_load_then_undo_preserves_identity():
    """The same frozen-dataclass instance is restored on undo —
    important so subsequent AddMemberLoadCmd undo logic (which removes
    by identity) is not confused after a delete+undo."""
    m, eid = _frame_model_one_member()
    pl = PointLoad(py=-7.0, a=1.5)
    m.elements[0].member_loads.append(pl)
    cmd = DeleteMemberLoadCmd(elem_id=eid, load_index=0)
    cmd.do(m)
    cmd.undo(m)
    assert m.elements[0].member_loads[0] is pl


# ── PR-A — load case CRUD commands ───────────────────────────────────


def _model_with_default_case():
    from structural_analysis.model import StructuralModel
    return StructuralModel(title="case-cmds")


def test_add_load_case_undo_restores_dict():
    from structural_analysis.gui_common.commands import AddLoadCaseCmd
    m = _model_with_default_case()
    cmd = AddLoadCaseCmd(name="DEAD")
    cmd.do(m)
    assert "DEAD" in m.load_cases
    cmd.undo(m)
    assert "DEAD" not in m.load_cases
    assert "DEFAULT" in m.load_cases  # untouched


def test_add_load_case_duplicate_raises():
    from structural_analysis.gui_common.commands import AddLoadCaseCmd
    m = _model_with_default_case()
    AddLoadCaseCmd(name="DEAD").do(m)
    with pytest.raises(ValueError, match=r"already exists"):
        AddLoadCaseCmd(name="DEAD").do(m)


def test_rename_load_case_cascades_to_attached_loads():
    from structural_analysis.gui_common.commands import (
        AddLoadCaseCmd, RenameLoadCaseCmd,
    )
    from structural_analysis.model import UniformDistributedLoad
    m, eid = _frame_model_one_member()
    AddLoadCaseCmd(name="DEAD").do(m)
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-10.0, load_case="DEAD")
    )
    RenameLoadCaseCmd(old_name="DEAD", new_name="LIVE").do(m)
    assert "DEAD" not in m.load_cases
    assert "LIVE" in m.load_cases
    assert m.elements[0].member_loads[0].load_case == "LIVE"


def test_rename_load_case_undo_restores_load_attachments():
    from structural_analysis.gui_common.commands import (
        AddLoadCaseCmd, RenameLoadCaseCmd,
    )
    from structural_analysis.model import UniformDistributedLoad
    m, eid = _frame_model_one_member()
    AddLoadCaseCmd(name="DEAD").do(m)
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-10.0, load_case="DEAD")
    )
    cmd = RenameLoadCaseCmd(old_name="DEAD", new_name="LIVE")
    cmd.do(m)
    cmd.undo(m)
    assert "DEAD" in m.load_cases
    assert "LIVE" not in m.load_cases
    assert m.elements[0].member_loads[0].load_case == "DEAD"


def test_rename_default_blocked():
    from structural_analysis.gui_common.commands import RenameLoadCaseCmd
    m = _model_with_default_case()
    with pytest.raises(ValueError, match=r"DEFAULT.*cannot be renamed"):
        RenameLoadCaseCmd(old_name="DEFAULT", new_name="BASE").do(m)


def test_delete_load_case_default_blocked():
    from structural_analysis.gui_common.commands import DeleteLoadCaseCmd
    m = _model_with_default_case()
    with pytest.raises(ValueError, match=r"DEFAULT.*cannot be deleted"):
        DeleteLoadCaseCmd(name="DEFAULT").do(m)


def test_delete_load_case_reassigns_loads_to_default():
    from structural_analysis.gui_common.commands import (
        AddLoadCaseCmd, DeleteLoadCaseCmd,
    )
    from structural_analysis.model import UniformDistributedLoad
    m, eid = _frame_model_one_member()
    AddLoadCaseCmd(name="DEAD").do(m)
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-10.0, load_case="DEAD")
    )
    DeleteLoadCaseCmd(name="DEAD").do(m)
    assert "DEAD" not in m.load_cases
    # Load was reassigned to DEFAULT, not deleted.
    assert m.elements[0].member_loads[0].load_case == "DEFAULT"


def test_delete_load_case_undo_restores_dict_and_loads():
    from structural_analysis.gui_common.commands import (
        AddLoadCaseCmd, DeleteLoadCaseCmd,
    )
    from structural_analysis.model import UniformDistributedLoad
    m, eid = _frame_model_one_member()
    AddLoadCaseCmd(name="DEAD").do(m)
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-10.0, load_case="DEAD")
    )
    cmd = DeleteLoadCaseCmd(name="DEAD")
    cmd.do(m)
    cmd.undo(m)
    assert "DEAD" in m.load_cases
    assert m.elements[0].member_loads[0].load_case == "DEAD"


def test_delete_load_case_resets_self_weight_case_to_default():
    from structural_analysis.gui_common.commands import (
        AddLoadCaseCmd, DeleteLoadCaseCmd, SetSelfWeightCaseCmd,
    )
    m = _model_with_default_case()
    AddLoadCaseCmd(name="DEAD").do(m)
    SetSelfWeightCaseCmd(case_name="DEAD").do(m)
    assert m.self_weight_case == "DEAD"
    DeleteLoadCaseCmd(name="DEAD").do(m)
    assert m.self_weight_case == "DEFAULT"


def test_set_load_case_enabled_toggle_undo():
    from structural_analysis.gui_common.commands import (
        AddLoadCaseCmd, SetLoadCaseEnabledCmd,
    )
    m = _model_with_default_case()
    AddLoadCaseCmd(name="WIND", enabled=True).do(m)
    cmd = SetLoadCaseEnabledCmd(name="WIND", enabled=False)
    cmd.do(m)
    assert m.load_cases["WIND"].enabled is False
    cmd.undo(m)
    assert m.load_cases["WIND"].enabled is True


def test_set_self_weight_case_undo():
    from structural_analysis.gui_common.commands import (
        AddLoadCaseCmd, SetSelfWeightCaseCmd,
    )
    m = _model_with_default_case()
    AddLoadCaseCmd(name="DEAD").do(m)
    cmd = SetSelfWeightCaseCmd(case_name="DEAD")
    cmd.do(m)
    assert m.self_weight_case == "DEAD"
    cmd.undo(m)
    assert m.self_weight_case == "DEFAULT"


def test_set_self_weight_case_unknown_raises():
    from structural_analysis.gui_common.commands import SetSelfWeightCaseCmd
    m = _model_with_default_case()
    with pytest.raises(ValueError, match=r"does not exist"):
        SetSelfWeightCaseCmd(case_name="GHOST").do(m)


# ── PR #29 — load combination CRUD + case cascade ────────────────────


def _model_with_two_cases():
    from structural_analysis.gui_common.commands import AddLoadCaseCmd
    from structural_analysis.model import StructuralModel
    m = StructuralModel(title="comb-cmds")
    AddLoadCaseCmd(name="DEAD").do(m)
    AddLoadCaseCmd(name="LIVE").do(m)
    return m


def test_add_combination_undo_redo():
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    m = _model_with_two_cases()
    cmd = AddLoadCombinationCmd(name="COMB1", terms={"DEAD": 1.2, "LIVE": 1.6})
    cmd.do(m)
    assert "COMB1" in m.load_combinations
    cmd.undo(m)
    assert "COMB1" not in m.load_combinations
    cmd.do(m)
    assert m.load_combinations["COMB1"].terms == {"DEAD": 1.2, "LIVE": 1.6}


def test_add_combination_rejects_unknown_case():
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    m = _model_with_two_cases()
    with pytest.raises(ValueError, match=r"do not exist"):
        AddLoadCombinationCmd(name="C", terms={"GHOST": 1.0}).do(m)


def test_add_combination_rejects_name_colliding_with_case():
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    m = _model_with_two_cases()
    with pytest.raises(ValueError, match=r"already a load-case name"):
        AddLoadCombinationCmd(name="DEAD", terms={"LIVE": 1.0}).do(m)


def test_delete_combination_undo():
    from structural_analysis.gui_common.commands import (
        AddLoadCombinationCmd, DeleteLoadCombinationCmd,
    )
    m = _model_with_two_cases()
    AddLoadCombinationCmd(name="C1", terms={"DEAD": 1.0}).do(m)
    cmd = DeleteLoadCombinationCmd(name="C1")
    cmd.do(m)
    assert "C1" not in m.load_combinations
    cmd.undo(m)
    assert m.load_combinations["C1"].terms == {"DEAD": 1.0}


def test_rename_combination_undo():
    from structural_analysis.gui_common.commands import (
        AddLoadCombinationCmd, RenameLoadCombinationCmd,
    )
    m = _model_with_two_cases()
    AddLoadCombinationCmd(name="C1", terms={"DEAD": 1.0}).do(m)
    cmd = RenameLoadCombinationCmd(old_name="C1", new_name="C2")
    cmd.do(m)
    assert "C1" not in m.load_combinations and "C2" in m.load_combinations
    cmd.undo(m)
    assert "C1" in m.load_combinations and "C2" not in m.load_combinations


def test_set_combination_terms_undo():
    from structural_analysis.gui_common.commands import (
        AddLoadCombinationCmd, SetLoadCombinationTermsCmd,
    )
    m = _model_with_two_cases()
    AddLoadCombinationCmd(name="C1", terms={"DEAD": 1.0}).do(m)
    cmd = SetLoadCombinationTermsCmd(name="C1", terms={"DEAD": 1.2, "LIVE": 1.6})
    cmd.do(m)
    assert m.load_combinations["C1"].terms == {"DEAD": 1.2, "LIVE": 1.6}
    cmd.undo(m)
    assert m.load_combinations["C1"].terms == {"DEAD": 1.0}


def test_rename_case_cascades_into_combination_terms():
    from structural_analysis.gui_common.commands import (
        AddLoadCombinationCmd, RenameLoadCaseCmd,
    )
    m = _model_with_two_cases()
    AddLoadCombinationCmd(name="C1", terms={"DEAD": 1.2, "LIVE": 1.6}).do(m)
    RenameLoadCaseCmd(old_name="DEAD", new_name="PERM").do(m)
    # The combination term keyed by DEAD must now be keyed by PERM.
    assert m.load_combinations["C1"].terms == {"PERM": 1.2, "LIVE": 1.6}


def test_rename_case_cascade_into_combinations_undo():
    from structural_analysis.gui_common.commands import (
        AddLoadCombinationCmd, RenameLoadCaseCmd,
    )
    m = _model_with_two_cases()
    AddLoadCombinationCmd(name="C1", terms={"DEAD": 1.2, "LIVE": 1.6}).do(m)
    cmd = RenameLoadCaseCmd(old_name="DEAD", new_name="PERM")
    cmd.do(m)
    cmd.undo(m)
    assert m.load_combinations["C1"].terms == {"DEAD": 1.2, "LIVE": 1.6}


def test_delete_case_blocked_when_referenced_by_combination():
    from structural_analysis.gui_common.commands import (
        AddLoadCombinationCmd, DeleteLoadCaseCmd,
    )
    m = _model_with_two_cases()
    AddLoadCombinationCmd(name="C1", terms={"DEAD": 1.0}).do(m)
    with pytest.raises(ValueError, match=r"referenced by load combination"):
        DeleteLoadCaseCmd(name="DEAD").do(m)
    # Case is still present (delete was atomic-blocked).
    assert "DEAD" in m.load_cases


def test_delete_case_allowed_after_combination_removed():
    from structural_analysis.gui_common.commands import (
        AddLoadCombinationCmd, DeleteLoadCombinationCmd, DeleteLoadCaseCmd,
    )
    m = _model_with_two_cases()
    AddLoadCombinationCmd(name="C1", terms={"DEAD": 1.0}).do(m)
    DeleteLoadCombinationCmd(name="C1").do(m)
    DeleteLoadCaseCmd(name="DEAD").do(m)  # now allowed
    assert "DEAD" not in m.load_cases


# ── Nodal-load CRUD (v0.20 — PR #30) ───────────────────────────────────


def _model_with_node_and_two_cases():
    """One node + DEAD + LIVE cases (DEFAULT auto-created)."""
    from structural_analysis.gui_common.commands import (
        AddLoadCaseCmd, AddNodeCmd,
    )
    m = StructuralModel(title="nodal-loads")
    AddNodeCmd(x=0.0, y=0.0).do(m)
    AddLoadCaseCmd(name="DEAD").do(m)
    AddLoadCaseCmd(name="LIVE").do(m)
    return m


def test_add_nodal_load_appends_new_row():
    from structural_analysis.gui_common.commands import AddNodalLoadCmd
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fy=-10.0, load_case="DEAD").do(m)
    AddNodalLoadCmd(node_id=1, fy=-20.0, load_case="LIVE").do(m)
    assert len(m.nodal_loads) == 2
    assert [ld.load_case for ld in m.nodal_loads] == ["DEAD", "LIVE"]
    assert [ld.fy for ld in m.nodal_loads] == [-10.0, -20.0]


def test_add_nodal_load_does_not_overwrite_existing_case():
    from structural_analysis.gui_common.commands import AddNodalLoadCmd
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fy=-10.0, load_case="DEAD").do(m)
    AddNodalLoadCmd(node_id=1, fy=-20.0, load_case="LIVE").do(m)
    # The DEAD row must still be present and unchanged.
    dead = [ld for ld in m.nodal_loads if ld.load_case == "DEAD"]
    assert len(dead) == 1
    assert dead[0].fy == -10.0


def test_add_nodal_load_allows_two_rows_same_case():
    from structural_analysis.gui_common.commands import AddNodalLoadCmd
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fx=5.0, load_case="DEAD").do(m)
    AddNodalLoadCmd(node_id=1, fy=-7.0, load_case="DEAD").do(m)
    dead = [ld for ld in m.nodal_loads if ld.load_case == "DEAD"]
    assert len(dead) == 2
    assert dead[0].fx == 5.0 and dead[1].fy == -7.0


def test_add_nodal_load_rejects_zero_row():
    from structural_analysis.gui_common.commands import AddNodalLoadCmd
    m = _model_with_node_and_two_cases()
    with pytest.raises(ValueError, match=r"nothing to add"):
        AddNodalLoadCmd(node_id=1, fx=0.0, fy=0.0, mz=0.0).do(m)
    assert m.nodal_loads == []


def test_add_nodal_load_rejects_unknown_node():
    from structural_analysis.gui_common.commands import AddNodalLoadCmd
    m = _model_with_node_and_two_cases()
    with pytest.raises(ValueError, match=r"Node 99 does not exist"):
        AddNodalLoadCmd(node_id=99, fy=-10.0).do(m)


def test_add_nodal_load_undo_removes_only_added_row():
    from structural_analysis.gui_common.commands import AddNodalLoadCmd
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fy=-10.0, load_case="DEAD").do(m)
    cmd = AddNodalLoadCmd(node_id=1, fy=-20.0, load_case="LIVE")
    cmd.do(m)
    cmd.undo(m)
    # LIVE row gone; DEAD row preserved.
    assert len(m.nodal_loads) == 1
    assert m.nodal_loads[0].load_case == "DEAD"
    assert m.nodal_loads[0].fy == -10.0


def test_edit_nodal_load_row_only_changes_target_row():
    from structural_analysis.gui_common.commands import (
        AddNodalLoadCmd, EditNodalLoadRowCmd,
    )
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fy=-10.0, load_case="DEAD").do(m)
    AddNodalLoadCmd(node_id=1, fy=-20.0, load_case="LIVE").do(m)
    # Edit row 1 (LIVE) → fy = -30; DEAD untouched.
    EditNodalLoadRowCmd(
        row_index=1, fx=0.0, fy=-30.0, mz=0.0, load_case="LIVE",
    ).do(m)
    assert m.nodal_loads[0].load_case == "DEAD" and m.nodal_loads[0].fy == -10.0
    assert m.nodal_loads[1].load_case == "LIVE" and m.nodal_loads[1].fy == -30.0


def test_edit_nodal_load_row_undo_restores_original():
    from structural_analysis.gui_common.commands import (
        AddNodalLoadCmd, EditNodalLoadRowCmd,
    )
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fy=-20.0, load_case="LIVE").do(m)
    cmd = EditNodalLoadRowCmd(
        row_index=0, fx=0.0, fy=-30.0, mz=0.0, load_case="LIVE",
    )
    cmd.do(m)
    cmd.undo(m)
    assert m.nodal_loads[0].fy == -20.0


def test_edit_nodal_load_row_out_of_range():
    from structural_analysis.gui_common.commands import EditNodalLoadRowCmd
    m = _model_with_node_and_two_cases()
    with pytest.raises(ValueError, match=r"out of range"):
        EditNodalLoadRowCmd(
            row_index=0, fx=0.0, fy=-1.0, mz=0.0,
        ).do(m)


def test_edit_nodal_load_row_rejects_zero():
    from structural_analysis.gui_common.commands import (
        AddNodalLoadCmd, EditNodalLoadRowCmd,
    )
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fy=-10.0).do(m)
    with pytest.raises(ValueError, match=r"use Delete"):
        EditNodalLoadRowCmd(
            row_index=0, fx=0.0, fy=0.0, mz=0.0,
        ).do(m)


def test_delete_nodal_load_row_removes_only_target():
    from structural_analysis.gui_common.commands import (
        AddNodalLoadCmd, DeleteNodalLoadRowCmd,
    )
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fy=-10.0, load_case="DEAD").do(m)
    AddNodalLoadCmd(node_id=1, fy=-20.0, load_case="LIVE").do(m)
    AddNodalLoadCmd(node_id=1, fx=5.0, load_case="WIND").do(m)
    # Delete LIVE (index 1). DEAD + WIND stay.
    DeleteNodalLoadRowCmd(row_index=1).do(m)
    assert [ld.load_case for ld in m.nodal_loads] == ["DEAD", "WIND"]


def test_delete_nodal_load_row_undo_restores_at_same_index():
    from structural_analysis.gui_common.commands import (
        AddNodalLoadCmd, DeleteNodalLoadRowCmd,
    )
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fy=-10.0, load_case="DEAD").do(m)
    AddNodalLoadCmd(node_id=1, fy=-20.0, load_case="LIVE").do(m)
    cmd = DeleteNodalLoadRowCmd(row_index=0)
    cmd.do(m)
    cmd.undo(m)
    # Original DEAD-then-LIVE ordering must be preserved.
    assert [ld.load_case for ld in m.nodal_loads] == ["DEAD", "LIVE"]
    assert m.nodal_loads[0].fy == -10.0


def test_delete_nodal_load_row_out_of_range():
    from structural_analysis.gui_common.commands import DeleteNodalLoadRowCmd
    m = _model_with_node_and_two_cases()
    with pytest.raises(ValueError, match=r"out of range"):
        DeleteNodalLoadRowCmd(row_index=0).do(m)


def test_delete_node_cascades_all_nodal_load_rows_and_undo_restores():
    from structural_analysis.gui_common.commands import (
        AddNodalLoadCmd, DeleteNodeCmd,
    )
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fy=-10.0, load_case="DEAD").do(m)
    AddNodalLoadCmd(node_id=1, fy=-20.0, load_case="LIVE").do(m)
    cmd = DeleteNodeCmd(node_id=1)
    cmd.do(m)
    assert m.nodal_loads == []
    assert 1 not in m.nodes
    cmd.undo(m)
    assert 1 in m.nodes
    assert {ld.load_case for ld in m.nodal_loads} == {"DEAD", "LIVE"}
    assert sorted(ld.fy for ld in m.nodal_loads) == [-20.0, -10.0]


def test_rename_case_cascades_to_nodal_load_rows():
    from structural_analysis.gui_common.commands import (
        AddNodalLoadCmd, RenameLoadCaseCmd,
    )
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fy=-10.0, load_case="DEAD").do(m)
    AddNodalLoadCmd(node_id=1, fy=-20.0, load_case="LIVE").do(m)
    RenameLoadCaseCmd(old_name="DEAD", new_name="PERM").do(m)
    cases = sorted(ld.load_case for ld in m.nodal_loads)
    assert cases == ["LIVE", "PERM"]


def test_delete_case_blocked_when_nodal_load_references_it_with_no_reassign():
    from structural_analysis.gui_common.commands import (
        AddNodalLoadCmd, DeleteLoadCaseCmd,
    )
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fy=-10.0, load_case="DEAD").do(m)
    with pytest.raises(ValueError, match=r"referenced by attached loads"):
        DeleteLoadCaseCmd(name="DEAD", reassign_to=None).do(m)


def test_add_nodal_load_undo_is_identity_based_under_non_lifo_mutation():
    """PR #30 reviewer fix (Gemini HIGH / Codex P2): when another
    command inserts a row between AddNodalLoadCmd's do() and undo(),
    the original stale-index fix could either delete the wrong row or
    no-op. Identity tracking removes the row the command actually
    added, even when its position has shifted.
    """
    from structural_analysis.gui_common.commands import (
        AddNodalLoadCmd, DeleteNodalLoadRowCmd,
    )
    m = _model_with_node_and_two_cases()
    # add A at index 0, then B at index 1.
    add_a = AddNodalLoadCmd(node_id=1, fy=-10.0, load_case="DEAD")
    add_b = AddNodalLoadCmd(node_id=1, fy=-20.0, load_case="LIVE")
    add_a.do(m)
    add_b.do(m)
    # Non-LIFO: delete A (the *first* row, index 0). B is now at index 0.
    DeleteNodalLoadRowCmd(row_index=0).do(m)
    assert len(m.nodal_loads) == 1
    assert m.nodal_loads[0].load_case == "LIVE"
    # Identity-based undo removes B, not whatever sits at the stale
    # index 1 (which is past the end now).
    add_b.undo(m)
    assert m.nodal_loads == []


def test_edit_nodal_load_row_undo_is_identity_based_under_non_lifo_mutation():
    """Mirror of the AddNodalLoadCmd identity regression for Edit:
    after an Edit replaces row 1, an unrelated delete of row 0 shifts
    the edited row to index 0. Identity-based undo still restores the
    pre-edit row at the correct (new) position."""
    from structural_analysis.gui_common.commands import (
        AddNodalLoadCmd, DeleteNodalLoadRowCmd, EditNodalLoadRowCmd,
    )
    m = _model_with_node_and_two_cases()
    AddNodalLoadCmd(node_id=1, fy=-10.0, load_case="DEAD").do(m)
    AddNodalLoadCmd(node_id=1, fy=-20.0, load_case="LIVE").do(m)
    edit_b = EditNodalLoadRowCmd(
        row_index=1, fx=0.0, fy=-30.0, mz=0.0, load_case="LIVE",
    )
    edit_b.do(m)
    assert m.nodal_loads[1].fy == -30.0
    # Non-LIFO mutation: delete row 0 (DEAD). The edited LIVE row
    # moves from index 1 to index 0.
    DeleteNodalLoadRowCmd(row_index=0).do(m)
    assert len(m.nodal_loads) == 1
    assert m.nodal_loads[0].fy == -30.0
    # Identity-based undo restores the pre-edit LIVE row in place.
    edit_b.undo(m)
    assert len(m.nodal_loads) == 1
    assert m.nodal_loads[0].fy == -20.0
    assert m.nodal_loads[0].load_case == "LIVE"


# ── renumber / merge (v0.24.0) ────────────────────────────────


def _seed_line_of_frames(n: int) -> StructuralModel:
    """Build a straight horizontal frame chain with N elements (N+1 nodes).
    Section/material identical so merge preconditions can pass."""
    m = StructuralModel(title="renumber")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S1", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    for i in range(n + 1):
        AddNodeCmd(x=float(i), y=0.0).do(m)
    for k in range(n):
        AddMemberCmd(
            x_i=float(k), y_i=0.0, x_j=float(k + 1), y_j=0.0,
            kind="frame", section_id=1,
        ).do(m)
    return m


# ── RenumberElementsCmd ────────────────────────────────────────


def test_renumber_produces_unique_sequential_ids():
    m = _seed_line_of_frames(4)
    ids = [e.id for e in m.elements]
    mapping = {old: new for new, old in enumerate(ids, start=10)}
    RenumberElementsCmd(mapping=mapping).do(m)
    new_ids = [e.id for e in m.elements]
    assert sorted(new_ids) == new_ids        # sorted ascending after do()
    assert len(set(new_ids)) == len(new_ids) # unique
    assert set(new_ids) == set(mapping.values())


def test_renumber_undo_restores_original_ids_and_order():
    m = _seed_line_of_frames(4)
    before_ids = [e.id for e in m.elements]
    before_iter = list(m.elements)  # identity order
    mapping = {old: new for new, old in enumerate(before_ids, start=100)}
    cmd = RenumberElementsCmd(mapping=mapping)
    cmd.do(m)
    cmd.undo(m)
    assert [e.id for e in m.elements] == before_ids
    assert m.elements == before_iter


def test_renumber_redo_replays_assignment():
    m = _seed_line_of_frames(3)
    ids = [e.id for e in m.elements]
    mapping = dict(zip(ids, [9, 8, 7]))
    cmd = RenumberElementsCmd(mapping=mapping)
    cmd.do(m)
    cmd.undo(m)
    cmd.do(m)
    assert sorted(e.id for e in m.elements) == [7, 8, 9]


def test_renumber_member_loads_stay_attached_to_same_physical_element():
    """Loads live on the element object, so renumbering must not detach
    a UDL from the beam that carries it."""
    m = _seed_line_of_frames(3)
    # Drop a UDL on element id 2 (the middle one).
    target = next(e for e in m.elements if e.id == 2)
    target.member_loads.append(UniformDistributedLoad(wy=-5.0))
    ids = [e.id for e in m.elements]
    mapping = dict(zip(ids, [30, 20, 10]))   # 2 → 20
    RenumberElementsCmd(mapping=mapping).do(m)
    # The element that USED to be id 2 should now be id 20 — and its UDL
    # should still be exactly there.
    moved = next(e for e in m.elements if e.id == 20)
    assert moved is target                    # same object
    assert moved.member_loads[0].wy == -5.0


def test_renumber_rejects_non_bijective_mapping():
    m = _seed_line_of_frames(3)
    ids = [e.id for e in m.elements]
    bad = dict(zip(ids, [1, 1, 2]))           # duplicate
    with pytest.raises(ValueError, match="bijective"):
        RenumberElementsCmd(mapping=bad).do(m)


def test_renumber_rejects_mapping_missing_an_element():
    m = _seed_line_of_frames(3)
    ids = [e.id for e in m.elements]
    bad = {ids[0]: 99, ids[1]: 100}           # missing one element
    with pytest.raises(ValueError, match="cover exactly"):
        RenumberElementsCmd(mapping=bad).do(m)


def test_renumber_rejects_non_positive_ids():
    m = _seed_line_of_frames(2)
    ids = [e.id for e in m.elements]
    bad = dict(zip(ids, [0, 1]))
    with pytest.raises(ValueError, match="positive"):
        RenumberElementsCmd(mapping=bad).do(m)


def test_renumber_sorts_elements_by_new_id():
    m = _seed_line_of_frames(4)
    ids = [e.id for e in m.elements]
    mapping = dict(zip(ids, [40, 30, 20, 10]))   # reverse
    RenumberElementsCmd(mapping=mapping).do(m)
    assert [e.id for e in m.elements] == [10, 20, 30, 40]


# ── MergeAdjacentElementsCmd ───────────────────────────────────


def test_merge_unloaded_collinear_frames_succeeds():
    m = _seed_line_of_frames(3)   # 4 nodes (1..4), 3 elements (1..3)
    # Merge at middle node 2 — connects elements 1 (1→2) and 2 (2→3).
    MergeAdjacentElementsCmd(middle_node_id=2).do(m)
    # Node 2 gone, one fewer element.
    assert 2 not in m.nodes
    assert len(m.elements) == 2
    merged = next(e for e in m.elements if e.id == 1)  # lower-id survives
    assert (merged.node_i, merged.node_j) == (1, 3)


def test_merge_preserves_surviving_element_orientation():
    """If the surviving element was outer_S → middle, the merged
    element must be outer_S → other_outer (NOT lower-node-id rule)."""
    m = StructuralModel(title="orient")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S1", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    # Three collinear nodes: 1 at x=0, 2 at x=1 (middle), 3 at x=2.
    # Element 1: 3 → 2 (i=3, j=middle).  Lower-id survivor.
    # Element 2: 1 → 2 (i=1, j=middle).
    for nid, x in ((1, 0.0), (2, 1.0), (3, 2.0)):
        AddNodeCmd(node_id=nid, x=x, y=0.0).do(m)
    AddMemberCmd(x_i=2.0, y_i=0.0, x_j=1.0, y_j=0.0,
                 kind="frame", section_id=1).do(m)  # elem id 1: 3→2
    AddMemberCmd(x_i=0.0, y_i=0.0, x_j=1.0, y_j=0.0,
                 kind="frame", section_id=1).do(m)  # elem id 2: 1→2
    MergeAdjacentElementsCmd(middle_node_id=2).do(m)
    merged = next(e for e in m.elements if e.id == 1)
    # Surviving was outer_S(=3) → middle(=2); merged should be 3 → 1.
    assert (merged.node_i, merged.node_j) == (3, 1)


def test_merge_preserves_outer_releases_through_orientation_swap():
    """The release on the outer end of each incident element must
    survive at the SAME physical end on the merged element."""
    m = StructuralModel(title="rel")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S1", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    for nid, x in ((1, 0.0), (2, 1.0), (3, 2.0)):
        AddNodeCmd(node_id=nid, x=x, y=0.0).do(m)
    # Element 1 (the survivor): outer=1, oriented 1→2; release at outer (=i).
    AddMemberCmd(x_i=0.0, y_i=0.0, x_j=1.0, y_j=0.0,
                 kind="frame", section_id=1).do(m)
    m.elements[0].release_i = True   # release on node 1
    # Element 2: outer=3, oriented 2→3; release at outer (=j).
    AddMemberCmd(x_i=1.0, y_i=0.0, x_j=2.0, y_j=0.0,
                 kind="frame", section_id=1).do(m)
    m.elements[1].release_j = True   # release on node 3
    MergeAdjacentElementsCmd(middle_node_id=2).do(m)
    merged = next(e for e in m.elements if e.id == 1)
    # Surviving was 1→2 (outer at i); merged should be 1→3, with
    # release_i (= old surviving outer) AND release_j (= other outer).
    assert (merged.node_i, merged.node_j) == (1, 3)
    assert merged.release_i is True
    assert merged.release_j is True


def test_merge_unloaded_collinear_trusses_succeeds():
    m = StructuralModel(title="truss")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S1", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    for nid, x in ((1, 0.0), (2, 1.0), (3, 2.0)):
        AddNodeCmd(node_id=nid, x=x, y=0.0).do(m)
    AddMemberCmd(x_i=0.0, y_i=0.0, x_j=1.0, y_j=0.0,
                 kind="truss", section_id=1).do(m)
    AddMemberCmd(x_i=1.0, y_i=0.0, x_j=2.0, y_j=0.0,
                 kind="truss", section_id=1).do(m)
    MergeAdjacentElementsCmd(middle_node_id=2).do(m)
    assert len(m.elements) == 1
    assert isinstance(m.elements[0], TrussElement2D)


def test_merge_blocks_non_collinear():
    m = StructuralModel(title="bend")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S1", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    for nid, xy in ((1, (0.0, 0.0)), (2, (1.0, 0.0)), (3, (1.0, 1.0))):
        AddNodeCmd(node_id=nid, x=xy[0], y=xy[1]).do(m)
    AddMemberCmd(x_i=0.0, y_i=0.0, x_j=1.0, y_j=0.0,
                 kind="frame", section_id=1).do(m)
    AddMemberCmd(x_i=1.0, y_i=0.0, x_j=1.0, y_j=1.0,
                 kind="frame", section_id=1).do(m)
    with pytest.raises(ValueError, match="not collinear"):
        MergeAdjacentElementsCmd(middle_node_id=2).do(m)


def test_merge_blocks_mixed_frame_and_truss():
    m = StructuralModel(title="mix")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S1", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    for nid, x in ((1, 0.0), (2, 1.0), (3, 2.0)):
        AddNodeCmd(node_id=nid, x=x, y=0.0).do(m)
    AddMemberCmd(x_i=0.0, y_i=0.0, x_j=1.0, y_j=0.0,
                 kind="frame", section_id=1).do(m)
    AddMemberCmd(x_i=1.0, y_i=0.0, x_j=2.0, y_j=0.0,
                 kind="truss", section_id=1).do(m)
    with pytest.raises(ValueError, match="frame and a truss"):
        MergeAdjacentElementsCmd(middle_node_id=2).do(m)


def test_merge_blocks_different_section():
    m = _seed_line_of_frames(2)
    m.sections[2] = Section(
        id=2, name="S2", material_id=1, A=0.02, I=2e-4, depth=0.4,
    )
    m.elements[1].section_id = 2  # second element on different section
    with pytest.raises(ValueError, match="different sections"):
        MergeAdjacentElementsCmd(middle_node_id=2).do(m)


def test_merge_blocks_inner_release_at_middle_node():
    m = _seed_line_of_frames(2)
    # Element 1: 1→2 — release at node 2 = release_j.
    m.elements[0].release_j = True
    with pytest.raises(ValueError, match="released end"):
        MergeAdjacentElementsCmd(middle_node_id=2).do(m)


def test_merge_blocks_support_at_middle_node():
    m = _seed_line_of_frames(2)
    SetSupportCmd(
        support=Support(node_id=2, ux=True, uy=True, rz=False),
    ).do(m)
    with pytest.raises(ValueError, match="support"):
        MergeAdjacentElementsCmd(middle_node_id=2).do(m)


def test_merge_blocks_support_settlement_at_middle_node():
    m = _seed_line_of_frames(2)
    m.supports[2] = Support(
        node_id=2, ux=True, uy=True, rz=False,
        settle_ux=0.005, settle_uy=None, settle_rz=None,
    )
    with pytest.raises(ValueError, match="settlement"):
        MergeAdjacentElementsCmd(middle_node_id=2).do(m)


def test_merge_blocks_nodal_load_at_middle_node():
    m = _seed_line_of_frames(2)
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0))
    with pytest.raises(ValueError, match="nodal load"):
        MergeAdjacentElementsCmd(middle_node_id=2).do(m)


def test_merge_blocks_member_loaded_element_with_specified_message():
    m = _seed_line_of_frames(2)
    AddMemberLoadCmd(
        elem_id=1, load=UniformDistributedLoad(wy=-3.0),
    ).do(m)
    with pytest.raises(
        ValueError, match="Merging loaded elements requires load remapping",
    ):
        MergeAdjacentElementsCmd(middle_node_id=2).do(m)


def test_merge_blocks_more_than_two_incident_elements():
    m = _seed_line_of_frames(2)
    # Add a third element going perpendicular off the middle node.
    AddNodeCmd(node_id=99, x=1.0, y=1.0).do(m)
    AddMemberCmd(
        x_i=1.0, y_i=0.0, x_j=1.0, y_j=1.0,
        kind="frame", section_id=1,
    ).do(m)
    with pytest.raises(ValueError, match="exactly 2"):
        MergeAdjacentElementsCmd(middle_node_id=2).do(m)


def test_merge_blocks_node_not_in_model():
    m = _seed_line_of_frames(2)
    with pytest.raises(ValueError, match="not found"):
        MergeAdjacentElementsCmd(middle_node_id=999).do(m)


def test_merge_undo_restores_elements_and_middle_node():
    m = _seed_line_of_frames(3)
    nodes_before = dict(m.nodes)
    elems_before = list(m.elements)
    ids_before = [e.id for e in m.elements]
    cmd = MergeAdjacentElementsCmd(middle_node_id=2)
    cmd.do(m)
    cmd.undo(m)
    assert dict(m.nodes) == nodes_before
    assert m.elements == elems_before
    assert [e.id for e in m.elements] == ids_before


# ── check_merge_preconditions (v0.24.1) ───────────────────────


def test_check_merge_allows_eligible_pair():
    m = _seed_line_of_frames(2)
    ok, reason = check_merge_preconditions(m, 2)
    assert ok is True
    assert reason is None


def test_check_merge_not_exactly_two_elements():
    m = _seed_line_of_frames(2)
    # Node 1 has only one incident element.
    ok, reason = check_merge_preconditions(m, 1)
    assert ok is False
    assert reason is not None
    assert "2" in reason and "found" in reason


def test_check_merge_not_collinear():
    m = StructuralModel(title="bend")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(id=1, name="S1", material_id=1, A=0.01, I=1e-4, depth=0.3)
    AddMemberCmd(x_i=0.0, y_i=0.0, x_j=1.0, y_j=0.0, kind="frame", section_id=1).do(m)
    AddMemberCmd(x_i=1.0, y_i=0.0, x_j=1.0, y_j=1.0, kind="frame", section_id=1).do(m)
    ok, reason = check_merge_preconditions(m, 2)
    assert ok is False
    assert reason is not None
    assert "collinear" in reason


def test_check_merge_different_sections():
    m = _seed_line_of_frames(2)
    m.sections[2] = Section(id=2, name="S2", material_id=1, A=0.02, I=2e-4, depth=0.4)
    m.elements[1].section_id = 2
    ok, reason = check_merge_preconditions(m, 2)
    assert ok is False
    assert reason is not None
    assert "section" in reason


def test_check_merge_support_at_middle_node():
    m = _seed_line_of_frames(2)
    SetSupportCmd(support=Support(node_id=2, ux=True, uy=True, rz=False)).do(m)
    ok, reason = check_merge_preconditions(m, 2)
    assert ok is False
    assert reason is not None
    assert "support" in reason


def test_check_merge_support_settlement_at_middle_node():
    m = _seed_line_of_frames(2)
    from structural_analysis.model import Support as _Support
    m.supports[2] = _Support(
        node_id=2, ux=True, uy=True, rz=False,
        settle_ux=0.005, settle_uy=None, settle_rz=None,
    )
    ok, reason = check_merge_preconditions(m, 2)
    assert ok is False
    assert reason is not None
    assert "settlement" in reason


def test_check_merge_nodal_load_at_middle_node():
    m = _seed_line_of_frames(2)
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0))
    ok, reason = check_merge_preconditions(m, 2)
    assert ok is False
    assert reason is not None
    assert "nodal load" in reason


def test_check_merge_member_loads_gives_reason():
    m = _seed_line_of_frames(2)
    AddMemberLoadCmd(elem_id=1, load=UniformDistributedLoad(wy=-3.0)).do(m)
    ok, reason = check_merge_preconditions(m, 2)
    assert ok is False
    assert reason is not None
    assert "member load" in reason.lower() or "remapping" in reason.lower()


def test_check_merge_and_command_agree_on_eligible_pair():
    """check_merge_preconditions True → command succeeds."""
    m = _seed_line_of_frames(2)
    ok, reason = check_merge_preconditions(m, 2)
    assert ok is True
    MergeAdjacentElementsCmd(middle_node_id=2).do(m)  # must not raise


def test_check_merge_and_command_agree_on_blocked_cases():
    """When precheck returns False, the command must also raise."""
    m = _seed_line_of_frames(2)
    SetSupportCmd(support=Support(node_id=2, ux=True, uy=True, rz=False)).do(m)
    ok, reason = check_merge_preconditions(m, 2)
    assert ok is False
    assert "support" in reason
    with pytest.raises(ValueError, match="support"):
        MergeAdjacentElementsCmd(middle_node_id=2).do(m)
