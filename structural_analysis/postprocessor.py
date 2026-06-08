"""
Postprocessor: member end forces, support reactions, equilibrium check.
"""

from __future__ import annotations

import numpy as np

from .assembler import DofManager
from .model import StructuralModel


def compute_member_forces(
    model: StructuralModel,
    D: np.ndarray,
    dofs: DofManager,
    elem_data: dict,
) -> dict[int, dict]:
    """Compute member end forces in local coordinates for each element.

    Args:
        model: The structural model.
        D: Full displacement vector (all DOFs).
        dofs: DOF manager for index lookups.
        elem_data: Per-element data from assembly (contains mappings).

    Returns:
        Dict mapping element ID to a dict with keys:
            "f_local": 6-element array [N_i, V_i, M_i, N_j, V_j, M_j].
            "d_local": 6-element array of local displacements.
            "d_global": 6-element array of global element displacements.
    """
    results: dict[int, dict] = {}

    for elem in model.elements:
        ed = elem_data[elem.id]
        mapping = ed["mapping"]

        # Extract global displacements for this element
        u_global_elem = np.zeros(6)
        node_dof_keys = [
            (elem.node_i, "ux"), (elem.node_i, "uy"), (elem.node_i, "rz"),
            (elem.node_j, "ux"), (elem.node_j, "uy"), (elem.node_j, "rz"),
        ]
        for local_idx, global_idx in enumerate(mapping):
            if global_idx is not None:
                u_global_elem[local_idx] = D[global_idx]
            else:
                # For truss θ DOFs: check if node has an active rz from
                # another element (frame) — if so, use that displacement
                nid, dof_name = node_dof_keys[local_idx]
                fallback = dofs.index(nid, dof_name)
                if fallback is not None and dof_name != "rz":
                    u_global_elem[local_idx] = D[fallback]

        # Transient loads added directly to global F (currently:
        # self-weight) leave a raw local fixed-end vector in elem_data so
        # the K·d − p recovery sees the same load on both sides of the
        # solve. Trusses don't get one — their self-weight is lumped at
        # uy DOFs, so no member distributed-load term exists.
        p_extra = ed.get("self_weight_p_local")
        d_local, q_local = elem.local_displacement_and_end_forces(
            model.nodes, u_global_elem, p_extra_local=p_extra,
        )

        results[elem.id] = {
            "f_local": q_local,
            "d_local": d_local,
            "d_global": u_global_elem,
        }

    return results


def compute_reactions(
    model: StructuralModel,
    K: np.ndarray,
    D: np.ndarray,
    F: np.ndarray,
    dofs: DofManager,
) -> dict[int, dict[str, float]]:
    """Compute support reactions from R = K·D − F at restrained DOFs.

    Args:
        model: The structural model.
        K: Global stiffness matrix (n × n).
        D: Full displacement vector (n,).
        F: Global load vector (n,).
        dofs: DOF manager for index lookups.

    Returns:
        Dict mapping supported node_id to {"ux": Rx, "uy": Ry, "rz": Mz}.
        Only restrained DOFs are included per node.
    """
    R_vec = K @ D - F
    reactions: dict[int, dict[str, float]] = {}

    for nid in model.node_ids:
        sup = model.support_for(nid)
        if not (sup.ux or sup.uy or sup.rz):
            continue

        r: dict[str, float] = {}
        for dof, flag in [("ux", sup.ux), ("uy", sup.uy), ("rz", sup.rz)]:
            idx = dofs.index(nid, dof)
            if idx is not None and flag:
                r[dof] = float(R_vec[idx])
        if r:
            reactions[nid] = r

    return reactions


def equilibrium_check(
    model: StructuralModel,
    member_results: dict[int, dict],
    dofs: DofManager,
) -> tuple[float, list[str]]:
    """Verify equilibrium at free nodes by summing element end forces.

    Args:
        model: The structural model.
        member_results: Per-element results from compute_member_forces().
        dofs: DOF manager (used to identify support vs free nodes).

    Returns:
        Tuple (max_residual, messages) where max_residual is the largest
        equilibrium error at any free node, and messages is a list of
        warning strings for nodes exceeding the tolerance.
    """
    from .element import _rotation_matrix_6x6

    messages: list[str] = []
    max_res = 0.0

    # Sum element contributions at each node in global coords
    node_forces: dict[int, np.ndarray] = {nid: np.zeros(3) for nid in model.node_ids}

    for elem in model.elements:
        mr = member_results[elem.id]
        q_local = mr["f_local"]
        ed_L, ed_c, ed_s = elem.length_cos_sin(model.nodes)
        R = _rotation_matrix_6x6(ed_c, ed_s)
        q_global = R.T @ q_local

        node_forces[elem.node_i] += q_global[0:3]
        node_forces[elem.node_j] += q_global[3:6]

    # Check at free nodes
    for nid in model.node_ids:
        sup = model.support_for(nid)
        if sup.ux or sup.uy or sup.rz:
            continue  # skip support nodes

        applied = np.zeros(3)
        for load in model.nodal_loads:
            if load.node_id == nid:
                applied += np.array([load.fx, load.fy, load.mz])

        residual = np.linalg.norm(node_forces[nid] - applied)
        max_res = max(max_res, residual)
        if residual > 1e-3:
            messages.append(
                f"WARNING: Equilibrium residual at node {nid}: {residual:.4e}"
            )

    return max_res, messages
