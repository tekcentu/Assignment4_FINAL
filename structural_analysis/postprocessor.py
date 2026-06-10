"""
Postprocessor: member end forces, support reactions, equilibrium check.

Dimension-generic (v0.32): every loop that used to hard-code the 2D
per-node DOF triple ('ux', 'uy', 'rz') now reads ``dofs.dof_names``
and the per-element ``dof_keys()`` map, so the same code serves the
legacy 2D pipeline and the 6-DOF-per-node 3D pipeline. Force recovery
always goes through ``elem_data[eid]["element"]`` — in 3D mode that is
the promoted space element whose stiffness was actually assembled.
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
        elem_data: Per-element data from assembly (contains mappings
            and the solve-time element objects).

    Returns:
        Dict mapping element ID to a dict with keys:
            "f_local": local end-force array — 6 entries
                [N_i, V_i, M_i, N_j, V_j, M_j] for 2D elements, 12
                ([N, Vy, Vz, T, My, Mz] per end) for a space frame.
            "d_local": local displacement array (same size).
            "d_global": global element displacement array (same size).
            "f_local_inplane": 3D frames only — the in-plane
                [N, Vy, Mz] per-end 6-vector, sign-compatible with the
                2D convention, used by the N/V/M diagram overlays.
    """
    results: dict[int, dict] = {}

    for eid, ed in elem_data.items():
        elem = ed["element"]
        mapping = ed["mapping"]

        # Extract global displacements for this element
        u_global_elem = np.zeros(len(mapping))
        node_dof_keys = elem.dof_keys()
        for local_idx, global_idx in enumerate(mapping):
            if global_idx is not None:
                u_global_elem[local_idx] = D[global_idx]
            else:
                # For truss θ DOFs: check if node has an active rotation
                # from another element (frame) — if so, use that
                # displacement. Released frame rotations stay excluded
                # (they are recovered by back-substitution instead).
                nid, dof_name = node_dof_keys[local_idx]
                fallback = dofs.index(nid, dof_name)
                if fallback is not None and not dof_name.startswith("r"):
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

        entry = {
            "f_local": q_local,
            "d_local": d_local,
            "d_global": u_global_elem,
        }
        if len(q_local) == 12:
            # In-plane slice for the 2D-convention consumers (N/V/M
            # diagram overlays, element detail dialog): local DOF
            # order is [ux, uy, uz, rx, ry, rz] per end, so
            # (N, Vy, Mz) live at indices (0, 1, 5) and (6, 7, 11).
            entry["f_local_inplane"] = q_local[[0, 1, 5, 6, 7, 11]]
        results[eid] = entry

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
        Dict mapping supported node_id to {dof_name: reaction}. Only
        restrained DOFs are included per node; in 3D mode the dict can
        additionally carry "uz" / "rx" / "ry" entries.
    """
    R_vec = K @ D - F
    reactions: dict[int, dict[str, float]] = {}

    for nid in model.node_ids:
        sup = model.support_for(nid)
        if not sup.any_restrained:
            continue

        r: dict[str, float] = {}
        for dof in dofs.dof_names:
            if not getattr(sup, dof):
                continue
            idx = dofs.index(nid, dof)
            if idx is not None:
                r[dof] = float(R_vec[idx])
        if r:
            reactions[nid] = r

    return reactions


def equilibrium_check(
    model: StructuralModel,
    member_results: dict[int, dict],
    dofs: DofManager,
    elem_data: dict | None = None,
) -> tuple[float, list[str]]:
    """Verify equilibrium at free nodes by summing element end forces.

    Args:
        model: The structural model.
        member_results: Per-element results from compute_member_forces().
        dofs: DOF manager (used to identify support vs free nodes and
            the per-node DOF name order).
        elem_data: Optional per-element assembly data. When given, the
            solve-time elements stored there are used (required for the
            3D pipeline, where promoted elements own the rotation
            math); when None, ``model.elements`` is used (legacy 2D
            call sites).

    Returns:
        Tuple (max_residual, messages) where max_residual is the largest
        equilibrium error at any free node, and messages is a list of
        warning strings for nodes exceeding the tolerance.
    """
    messages: list[str] = []
    max_res = 0.0

    if elem_data is not None:
        elements = [ed["element"] for ed in elem_data.values()]
    else:
        elements = list(model.elements)

    dof_pos = {name: k for k, name in enumerate(dofs.dof_names)}
    n_per_node = len(dofs.dof_names)

    # Sum element contributions at each node in global coords
    node_forces: dict[int, np.ndarray] = {
        nid: np.zeros(n_per_node) for nid in model.node_ids
    }

    for elem in elements:
        q_local = member_results[elem.id]["f_local"]
        R = elem.transformation_matrix(model.nodes)
        q_global = R.T @ q_local
        for li, (nid, dname) in enumerate(elem.dof_keys()):
            node_forces[nid][dof_pos[dname]] += q_global[li]

    # Check at free nodes
    load_attrs_by_dof = {
        "ux": "fx", "uy": "fy", "uz": "fz",
        "rx": "mx", "ry": "my", "rz": "mz",
    }
    for nid in model.node_ids:
        sup = model.support_for(nid)
        if sup.any_restrained:
            continue  # skip support nodes

        applied = np.zeros(n_per_node)
        for load in model.nodal_loads:
            if load.node_id == nid:
                for dname, pos in dof_pos.items():
                    applied[pos] += getattr(
                        load, load_attrs_by_dof[dname], 0.0,
                    )

        residual = np.linalg.norm(node_forces[nid] - applied)
        max_res = max(max_res, residual)
        if residual > 1e-3:
            messages.append(
                f"WARNING: Equilibrium residual at node {nid}: {residual:.4e}"
            )

    return max_res, messages
