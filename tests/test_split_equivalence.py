"""Solve-level regression tests for ``SplitElementCmd`` member-load remap.

For each load configuration we solve the analysis once with the parent
element intact, then again on a cloned model after splitting the parent
at midspan via :class:`SplitElementCmd`. The reactions at the support
nodes and the assembled member end-forces must match within tight FP
tolerance — if the remap rules are correct, the post-split structure is
mechanically equivalent to the pre-split one and the solver should be
unable to tell the difference.

Mirrors the helper pattern of ``tests/test_gui_file_writer.py``.
"""

from __future__ import annotations

import copy

import numpy as np

from structural_analysis.gui_common.commands import (
    AddMemberCmd,
    SplitElementCmd,
)
from structural_analysis.main import run_analysis
from structural_analysis.model import (
    FrameTemperatureLoad,
    Material,
    PointLoad,
    Section,
    StructuralModel,
    Support,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)


# ── helpers ────────────────────────────────────────────────────


def _frame_fixed_fixed(load):
    """Horizontal frame 0..6 m on two fixed supports, with a load."""
    m = StructuralModel(title="ff frame")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, alpha=1.2e-5,
                              density=7850.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.01, I=1e-4, depth=0.3)
    AddMemberCmd(x_i=0.0, y_i=0.0, x_j=6.0, y_j=0.0,
                 kind="frame", section_id=1).do(m)
    # The AddMemberCmd auto-created the nodes; fetch their ids.
    elem = m.elements[0]
    m.supports[elem.node_i] = Support(node_id=elem.node_i,
                                      ux=True, uy=True, rz=True)
    m.supports[elem.node_j] = Support(node_id=elem.node_j,
                                      ux=True, uy=True, rz=True)
    elem.member_loads.append(load)
    return m


def _truss_axially_restrained_bar(load):
    """Two-node horizontal truss 0..4 m with BOTH ends fully pinned
    (ux, uy fixed at each end).

    This is deliberately axially restrained (not a roller): a thermal
    load on a fully-restrained bar develops a non-zero axial reaction
    (±E·A·α·ΔT), which is the stronger quantity to check for
    before/after-split equivalence. End displacements are ~0 at both
    fixed ends, so the displacement leg of the equivalence assertion is
    near-vacuous here — the reaction leg carries the test.

    A midspan split introduces a mid node whose transverse (uy) DOF has
    no stiffness on a truss; the caller adds a transverse restraint at
    that mid node post-split to keep the system non-singular. That
    restraint carries zero force for an axial-only thermal bar, so the
    end reactions are unaffected.
    """
    m = StructuralModel(title="truss")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, alpha=1.2e-5,
                              density=7850.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.01, I=1e-4, depth=0.3)
    AddMemberCmd(x_i=0.0, y_i=0.0, x_j=4.0, y_j=0.0,
                 kind="truss", section_id=1).do(m)
    elem = m.elements[0]
    # Both ends fully pinned (ux, uy) → axially restrained.
    m.supports[elem.node_i] = Support(node_id=elem.node_i,
                                      ux=True, uy=True)
    m.supports[elem.node_j] = Support(node_id=elem.node_j,
                                      ux=True, uy=True)
    elem.member_loads.append(load)
    return m


def _solve_pair(model: StructuralModel, split_x: float = 3.0):
    """Solve model once, deep-copy, split via SplitElementCmd at x, solve again.

    Returns (result_before, result_after, model_after).
    """
    result_before = run_analysis(model, verbose=False)
    assert result_before.status == "ok"

    model_after = copy.deepcopy(model)
    elem_id = model_after.elements[0].id
    cmd = SplitElementCmd(element_id=elem_id, x=split_x, y=0.0)
    cmd.do(model_after)
    assert len(model_after.elements) == 2

    result_after = run_analysis(model_after, verbose=False)
    assert result_after.status == "ok"
    return result_before, result_after, model_after


def _reactions_array(result, node_ids):
    rows = []
    for nid in node_ids:
        r = result.reactions.get(nid, {})
        rows.append([r.get("fx", 0.0), r.get("fy", 0.0), r.get("mz", 0.0)])
    return np.array(rows, dtype=float)


def _disp_array(result, node_ids):
    """[ux, uy, rz] displacement per node, read via E_map → D.

    A DOF mapped to ``None`` (e.g. a truss node has no rz) contributes
    0.0 so frame and truss nodes can share the same 3-wide layout.
    """
    D = np.asarray(result.D, dtype=float)
    rows = []
    for nid in node_ids:
        emap = result.E_map.get(nid, {})
        rows.append([
            0.0 if emap.get(k) is None else float(D[emap[k]])
            for k in ("ux", "uy", "rz")
        ])
    return np.array(rows, dtype=float)


# 1e-9 honours the spec's strict tolerance; FE nodal reactions and
# displacements are mesh-exact here, so before/after-split values agree
# to well within this (verified to pass at 1e-12).
_TOL = dict(atol=1e-9, rtol=1e-9)


def _assert_reactions_equivalent(result_before, result_after, node_ids):
    r1 = _reactions_array(result_before, node_ids)
    r2 = _reactions_array(result_after, node_ids)
    np.testing.assert_allclose(r2, r1, **_TOL)


def _assert_displacements_equivalent(result_before, result_after, node_ids):
    d1 = _disp_array(result_before, node_ids)
    d2 = _disp_array(result_after, node_ids)
    np.testing.assert_allclose(d2, d1, **_TOL)


def _assert_equivalent(result_before, result_after, node_ids):
    """Reactions AND nodal displacements at the original nodes match."""
    _assert_reactions_equivalent(result_before, result_after, node_ids)
    _assert_displacements_equivalent(result_before, result_after, node_ids)


# ── tests ──────────────────────────────────────────────────────


def test_split_equivalence_udl_full_length():
    m = _frame_fixed_fixed(UniformDistributedLoad(wy=-10.0))
    end_nodes = [m.elements[0].node_i, m.elements[0].node_j]
    r1, r2, _ = _solve_pair(m, split_x=3.0)
    _assert_equivalent(r1, r2, end_nodes)


def test_split_equivalence_point_load_left_of_split():
    # PointLoad at a = L/3 = 2.0 (left of midspan).
    m = _frame_fixed_fixed(PointLoad(py=-12.0, a=2.0))
    end_nodes = [m.elements[0].node_i, m.elements[0].node_j]
    r1, r2, _ = _solve_pair(m, split_x=3.0)
    _assert_equivalent(r1, r2, end_nodes)


def test_split_equivalence_point_load_right_of_split():
    # PointLoad at a = 2L/3 = 4.0 (right of midspan).
    m = _frame_fixed_fixed(PointLoad(py=-12.0, a=4.0))
    end_nodes = [m.elements[0].node_i, m.elements[0].node_j]
    r1, r2, _ = _solve_pair(m, split_x=3.0)
    _assert_equivalent(r1, r2, end_nodes)


def test_split_equivalence_point_load_exactly_at_split():
    # PointLoad at a = L/2 = 3.0 (exactly at midspan).
    m = _frame_fixed_fixed(PointLoad(py=-12.0, a=3.0))
    end_nodes = [m.elements[0].node_i, m.elements[0].node_j]
    r1, r2, _ = _solve_pair(m, split_x=3.0)
    _assert_equivalent(r1, r2, end_nodes)


def test_split_equivalence_frame_thermal_axial_only():
    m = _frame_fixed_fixed(FrameTemperatureLoad(t_top=20.0, t_bottom=20.0))
    end_nodes = [m.elements[0].node_i, m.elements[0].node_j]
    r1, r2, _ = _solve_pair(m, split_x=3.0)
    _assert_equivalent(r1, r2, end_nodes)


def test_split_equivalence_frame_thermal_bending_only():
    m = _frame_fixed_fixed(FrameTemperatureLoad(t_top=20.0, t_bottom=-20.0))
    end_nodes = [m.elements[0].node_i, m.elements[0].node_j]
    r1, r2, _ = _solve_pair(m, split_x=3.0)
    _assert_equivalent(r1, r2, end_nodes)


def test_split_equivalence_frame_thermal_combined():
    m = _frame_fixed_fixed(FrameTemperatureLoad(t_top=20.0, t_bottom=-10.0))
    end_nodes = [m.elements[0].node_i, m.elements[0].node_j]
    r1, r2, _ = _solve_pair(m, split_x=3.0)
    _assert_equivalent(r1, r2, end_nodes)


def test_split_equivalence_truss_thermal():
    """Truss thermal equivalence. Single horizontal bar with both ends
    pinned (ux, uy fixed). Splitting introduces a mid node whose
    y-DOF would otherwise float (truss has no rotational stiffness),
    so we add a transverse roller at the mid node after the split to
    keep the system non-singular. The pre-split model is solved as-is
    (no mid node exists); the equivalence is on the support
    reactions at the two end nodes, which are unaffected by the mid
    transverse restraint (transverse equilibrium at the mid node is
    trivially zero either way for an axial-only thermal bar).
    """
    m = _truss_axially_restrained_bar(TrussTemperatureLoad(delta_T=25.0))
    end_nodes = [m.elements[0].node_i, m.elements[0].node_j]

    result_before = run_analysis(m, verbose=False)
    assert result_before.status == "ok"

    model_after = copy.deepcopy(m)
    elem_id = model_after.elements[0].id
    cmd = SplitElementCmd(element_id=elem_id, x=2.0, y=0.0)
    cmd.do(model_after)
    assert len(model_after.elements) == 2
    # Add a transverse roller at the new mid node to keep the
    # post-split system non-singular (truss y-DOF at an interior
    # node).
    mid_nid = cmd._created_node_c
    assert mid_nid is not None
    model_after.supports[mid_nid] = Support(node_id=mid_nid,
                                            ux=False, uy=True)
    result_after = run_analysis(model_after, verbose=False)
    assert result_after.status == "ok"
    _assert_equivalent(result_before, result_after, end_nodes)


# ── displacement-teeth: simply-supported beam (non-vacuous) ──
#
# The fixed-fixed fixtures above pin every end DOF, so their
# displacement check is near-vacuous (all ~0). These use a
# simply-supported span (pin + roller) where the original end nodes
# genuinely DISPLACE — end rotations under transverse load, and axial
# extension under thermal — so the displacement leg of the equivalence
# assertion has real teeth.


def _frame_simply_supported(load):
    """Horizontal frame 0..6 m: pin (ux, uy) at left, roller (uy) at
    right. End rotations are non-zero under transverse load and the
    right node translates freely under axial thermal expansion."""
    m = StructuralModel(title="ss frame")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, alpha=1.2e-5,
                              density=7850.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.01, I=1e-4, depth=0.3)
    AddMemberCmd(x_i=0.0, y_i=0.0, x_j=6.0, y_j=0.0,
                 kind="frame", section_id=1).do(m)
    elem = m.elements[0]
    m.supports[elem.node_i] = Support(node_id=elem.node_i, ux=True, uy=True)
    m.supports[elem.node_j] = Support(node_id=elem.node_j, uy=True)
    elem.member_loads.append(load)
    return m


def test_split_equivalence_udl_displacements_nonvacuous():
    """Simply-supported UDL: end rotations are non-zero; they must
    match before/after split (proves the displacement check bites)."""
    m = _frame_simply_supported(UniformDistributedLoad(wy=-10.0))
    end_nodes = [m.elements[0].node_i, m.elements[0].node_j]
    r1, r2, _ = _solve_pair(m, split_x=3.0)
    # Guard: the displacement we're comparing is actually non-trivial.
    d = _disp_array(r1, end_nodes)
    assert np.max(np.abs(d[:, 2])) > 1e-4  # end rotations rz
    _assert_equivalent(r1, r2, end_nodes)


def test_split_equivalence_thermal_axial_displacements_nonvacuous():
    """Simply-supported axial thermal: the roller end translates
    (free expansion); ux there must match before/after split."""
    m = _frame_simply_supported(
        FrameTemperatureLoad(t_top=15.0, t_bottom=15.0))
    end_nodes = [m.elements[0].node_i, m.elements[0].node_j]
    r1, r2, _ = _solve_pair(m, split_x=3.0)
    d = _disp_array(r1, end_nodes)
    assert np.max(np.abs(d[:, 0])) > 1e-4  # axial translation ux
    _assert_equivalent(r1, r2, end_nodes)
