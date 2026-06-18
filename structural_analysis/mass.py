"""Global mass-matrix assembly for free-vibration analysis.

Mirrors :func:`structural_analysis.assembler.assemble_global_system` but
only builds the global mass matrix M from the elements'
:meth:`Element2D.consistent_mass_local` (default) or
:meth:`Element2D.lumped_mass_local` (v0.9.2 lumped-translational
comparison mode). The same :class:`DofManager` that built the stiffness
DOF order is reused, so M is guaranteed to share K's row/column
ordering.

Unit system. Material density is stored on :class:`Material` and mirrored
to the element as ``rho`` in kg/m³ — the unit users naturally write. The
element-level mass routines convert to the consistent kN-m-s system used
by the static stiffness pipeline (mass in Mg = kN·s²/m) before emitting
their local 6×6 matrices, so the eigenvalue problem ``(K − ω²·M)·φ = 0``
yields ``ω`` directly in rad/s and ``f = ω/(2π)`` directly in Hz.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

import numpy as np

from .assembler import DofManager
from .model import ModalMassSource, STANDARD_GRAVITY, StructuralModel


MassFormulation = Literal["consistent", "lumped"]


@contextmanager
def _self_weight_only_context(model: StructuralModel):
    """Temporarily remove all manual loads and enable self-weight.

    Used by the load-case-to-mass path to build a force vector that
    contains *only* the density-driven self-weight contribution.  This
    vector is subtracted from the full-case F before converting manual
    loads to mass, preventing double-counting when element self-mass is
    also enabled.
    """
    saved_nodal = model.nodal_loads
    saved_member: dict[int, list | None] = {}
    for elem in model.elements:
        saved_member[id(elem)] = getattr(elem, "member_loads", None)
    saved_sw = model.include_self_weight
    try:
        model.nodal_loads = []
        for elem in model.elements:
            if saved_member[id(elem)] is not None:
                elem.member_loads = []
        model.include_self_weight = True
        yield model
    finally:
        model.nodal_loads = saved_nodal
        for elem in model.elements:
            ml = saved_member[id(elem)]
            if ml is not None:
                elem.member_loads = ml
        model.include_self_weight = saved_sw


def assemble_mass_matrix(
    model: StructuralModel,
    dofs: DofManager,
    *,
    formulation: MassFormulation = "lumped",
) -> np.ndarray:
    """Assemble the global mass matrix.

    Args:
        model: The structural model.
        dofs: DOF manager whose ordering matches the one used for K.
        formulation: ``"lumped"`` (default — translational-only, half
            the bar mass at each end on ux and uy, zero on rz; the
            only formulation exposed by the user-facing modal workflow
            in the final-submission build) or ``"consistent"`` (internal
            helper kept for low-level tests and the diagnostic Joint
            Masses inspector — energy-consistent Hermite-cubic element
            mass, carries rotational inertia on rz DOFs). The lumped path
            aid; with it the global M is singular on every rz row/col
            and the modal solver in :func:`solve_modal` condenses those
            DOFs out (Guyan reduction).

    Returns:
        ``n_total × n_total`` numpy array — global mass matrix M in
        consistent kN-m-s units (Mg per translational diagonal entry).

    Raises:
        ValueError: If ``formulation`` is not one of the recognised
            options.
    """
    if formulation not in ("consistent", "lumped"):
        raise ValueError(
            f"Unknown mass formulation {formulation!r}; "
            "expected 'consistent' or 'lumped'."
        )
    n = dofs.n_total
    M = np.zeros((n, n))
    for elem in model.elements:
        if formulation == "consistent":
            m_local = elem.consistent_mass_local(model.nodes)
        else:
            m_local = elem.lumped_mass_local(model.nodes)
        R = elem.transformation_matrix(model.nodes)
        m_global = R.T @ m_local @ R
        mapping = dofs.element_dof_map(elem)
        for a, I in enumerate(mapping):
            if I is None:
                continue
            for b, J in enumerate(mapping):
                if J is None:
                    continue
                M[I, J] += m_global[a, b]
    # Symmetrise numerically tiny asymmetries from the rotation product.
    M = 0.5 * (M + M.T)
    return M


def assemble_mass_matrix_with_source(
    model: StructuralModel,
    dofs: DofManager,
    *,
    formulation: MassFormulation = "lumped",
    source: ModalMassSource | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the global mass matrix using a :class:`ModalMassSource`.

    Extends :func:`assemble_mass_matrix` with optional joint-mass and
    load-case-to-mass contributions.  When *source* is ``None`` or
    equivalent to the default (self-mass only, no joint masses, no load
    cases), the result is numerically identical to
    ``assemble_mass_matrix(model, dofs, formulation=formulation)``.

    Args:
        model: The structural model.
        dofs: DOF manager whose ordering matches the one used for K.
        formulation: Mass formulation forwarded to
            :func:`assemble_mass_matrix` for the self-mass branch.
        source: Mass source settings.  ``None`` → safe default
            (self-mass only).

    Returns:
        ``(M, info_lines)`` where *M* is the ``n × n`` mass matrix and
        *info_lines* is a list of informational strings suitable for
        ``ModalResult.warnings``.

    Raises:
        ValueError: If a load case referenced in ``source.load_case_factors``
            is not present in ``model.load_cases``.
        KeyError: If a joint-mass node id is not in ``model.nodes``.
    """
    if source is None:
        source = ModalMassSource()

    info: list[str] = []

    # ── self-mass branch ──────────────────────────────────────────────────
    if source.include_self_mass:
        M = assemble_mass_matrix(model, dofs, formulation=formulation)
    else:
        M = np.zeros((dofs.n_total, dofs.n_total))

    # ── joint-mass branch ─────────────────────────────────────────────────
    if source.include_joint_masses and model.joint_masses:
        for node_id, jm in model.joint_masses.items():
            if node_id not in model.nodes:
                raise KeyError(
                    f"JointMass references node {node_id}, which is not in "
                    "the model."
                )
            nm = dofs.active_map.get(node_id)
            if nm is None:
                info.append(
                    f"JointMass at node {node_id}: node has no active DOFs; "
                    "mass ignored."
                )
                continue
            ux_idx = nm.get("ux")
            uy_idx = nm.get("uy")
            # Convert kg → Mg (consistent kN-m-s system).
            if ux_idx is not None:
                M[ux_idx, ux_idx] += jm.mx / 1000.0
            if uy_idx is not None:
                M[uy_idx, uy_idx] += jm.my / 1000.0

    # ── load-case-to-mass branch ──────────────────────────────────────────
    if source.include_load_cases and source.load_case_factors:
        # Lazy import avoids any potential import-order issue; filter helper
        # lives in main.py which has a larger import footprint.
        from .main import filter_loads_to_case
        from .assembler import assemble_global_system

        for case_name, mult in source.load_case_factors.items():
            if mult <= 0.0:
                continue
            if case_name not in model.load_cases:
                raise ValueError(
                    f"MODAL_MASS_SOURCE: load_case_factors references case "
                    f"{case_name!r}, which is not in model.load_cases."
                )
            # Build F_case: manual loads for this case, + possibly SW.
            with filter_loads_to_case(model, case_name):
                _, F_case, dofs_lc, _, _ = assemble_global_system(model)

            # Anti-double-count gate: if element self-mass is enabled AND
            # include_self_weight is True AND this IS the self_weight_case,
            # the density-driven gravity force is already captured by the
            # self-mass branch.  Subtract it so only manual superimposed
            # dead loads are converted to mass.
            if (source.include_self_mass
                    and model.include_self_weight
                    and case_name == model.self_weight_case):
                with _self_weight_only_context(model):
                    _, F_sw, _, _, _ = assemble_global_system(model)
                F_convert = F_case - F_sw
                info.append(
                    f"Load-case mass ({case_name}): generated self-weight "
                    "excluded from mass conversion (element density already "
                    "provides self-mass)."
                )
            else:
                F_convert = F_case

            # |Fy| at each free node → scalar nodal mass added to both
            # translational DOFs (gravity loads represent scalar mass that
            # resists lateral and vertical inertia equally).
            for node_id in model.nodes:
                nm = dofs_lc.active_map.get(node_id)
                if nm is None:
                    continue
                uy_idx_lc = nm.get("uy")
                ux_idx_lc = nm.get("ux")
                if uy_idx_lc is None:
                    continue
                fy_abs = abs(float(F_convert[uy_idx_lc]))
                if fy_abs == 0.0:
                    continue
                nodal_mass = mult * fy_abs / STANDARD_GRAVITY  # Mg

                # Look up indices in the *passed-in* dofs (same ordering).
                nm2 = dofs.active_map.get(node_id)
                if nm2 is None:
                    continue
                ux_idx2 = nm2.get("ux")
                uy_idx2 = nm2.get("uy")
                if ux_idx2 is not None:
                    M[ux_idx2, ux_idx2] += nodal_mass
                if uy_idx2 is not None:
                    M[uy_idx2, uy_idx2] += nodal_mass

    return M, info
