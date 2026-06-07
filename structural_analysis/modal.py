"""Free-vibration modal analysis.

Solves the generalised eigenvalue problem ``(K − ω²·M)·φ = 0`` on the
free-DOF partition of the global system. Returns a :class:`ModalResult`
carrying natural frequencies (Hz), periods (s), angular frequencies
(rad/s) and full-length mode-shape vectors (with zeros at restrained
DOFs so they can be drawn directly on top of the model geometry).

Two mass formulations are supported (v0.9.2). ``"consistent"`` is the
default and is unchanged from v0.9.1. ``"lumped"`` is a comparison aid:
it assembles a translational-only mass matrix (zero on every rotational
DOF) and statically condenses the massless DOFs out of the eigenproblem
(Guyan reduction). With lumped mass this condensation is mathematically
exact — massless DOFs carry zero kinetic energy and therefore move
quasi-statically with the mass DOFs — so the recovered mode shapes
remain valid full-DOF vectors.

A dedicated result object is used instead of overloading
:class:`AnalysisResult` (which stays bound to static analysis) — the
GUI dispatches on the result type.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as _dc_replace
from typing import Literal

import numpy as np
import scipy.linalg

from .assembler import DofManager, _connectivity_components, assemble_global_system
from .mass import MassFormulation, assemble_mass_matrix, assemble_mass_matrix_with_source
from .model import ModalMassSource, StructuralModel, Support


# Relative tolerance used to classify free DOFs as "mass-bearing" vs
# "massless" when condensing on the lumped path. The classifier
# compares each diagonal of M_ff against this fraction of the largest
# diagonal — so a numerically dirty 1e-14 entry from R.T M R doesn't
# get mistaken for real mass.
_LUMPED_MASS_REL_TOL: float = 1e-12

LUMPED_COMPARISON_NOTE: str = (
    "Lumped translational mass is a comparison aid. Agreement with "
    "external software depends on matching units, density/mass source, "
    "section properties, mesh, boundary conditions, restraints, and "
    "mass formulation."
)


@dataclass
class ComponentModalResult:
    """Modal result for a single connected component.

    When a model contains multiple disconnected but individually stable
    structures, :func:`solve_modal` solves each component separately and
    returns one :class:`ComponentModalResult` per component inside
    :attr:`ModalResult.components`.

    Mode vectors are stored in the **global DOF order** (length
    ``n_total``) with zeros at DOFs belonging to other components, so
    the canvas can display them without any extra bookkeeping.
    """

    component_id: int           # 1-indexed
    node_ids: list[int]         # sorted list of node IDs in this component
    element_ids: list[int]      # sorted list of element IDs whose both nodes are in this component
    is_supported: bool          # at least one support restraint in this component
    is_singular: bool           # Kff rank < n_comp_free_dofs (detected via SVD)
    skip_reason: str | None     # None when successfully solved
    n_modes: int
    frequencies: np.ndarray
    periods: np.ndarray
    omegas: np.ndarray
    global_mode_offset: int     # first column index in ModalResult.modes for this component


@dataclass
class ModalResult:
    """Output of a free-vibration analysis.

    Attributes:
        status: ``"ok"`` or ``"error"``.
        title: Model title.
        warnings: Validation warnings carried over from the assembler.
        n_modes: Number of returned modes (may be less than requested
            when there are fewer free DOFs).
        frequencies: 1-D array of natural frequencies in Hz, ascending.
        periods: 1-D array of natural periods in s (∞ where f = 0).
        omegas: 1-D array of natural angular frequencies in rad/s.
        modes: ``n_total × n_modes`` array. Each column is a mode shape
            in the full DOF order produced by :class:`DofManager`, with
            restrained entries explicitly set to zero — so each column
            can be re-indexed by ``DofManager.active_map`` to recover
            ``(ux, uy, rz)`` at every node.
        normalisation: ``"mass"`` (mass-orthonormal) or ``"max"``
            (max-absolute-component = 1).
        dofs: The :class:`DofManager` used so callers can map mode
            entries back to (node_id, dof) tuples.
        mass_formulation: ``"consistent"`` (default) or ``"lumped"`` —
            records which modal mass matrix produced these frequencies
            so the GUI / report can display it.
        components: Per-component results when the model has >1 connected
            component; empty list for single-component models (flat-array
            fields are the authoritative data in both cases).
        component_summary: Human-readable summary of component-solve
            status; empty string for single-component models.
    """

    status: str = "ok"
    title: str = ""
    warnings: list[str] = field(default_factory=list)
    n_modes: int = 0
    frequencies: np.ndarray = field(default_factory=lambda: np.zeros(0))
    periods: np.ndarray = field(default_factory=lambda: np.zeros(0))
    omegas: np.ndarray = field(default_factory=lambda: np.zeros(0))
    modes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    normalisation: str = "mass"
    dofs: DofManager | None = None
    mass_formulation: MassFormulation = "consistent"
    mass_source_summary: str = "self-mass only"
    components: list[ComponentModalResult] = field(default_factory=list)
    component_summary: str = ""


def _solve_modal_condensed(
    K_ff: np.ndarray,
    M_ff: np.ndarray,
    n_modes_request: int,
) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    """Static-condensation eigensolver for singular (lumped) M_ff.

    Partitions free DOFs into mass-bearing (``m``) and massless (``r``)
    using a relative tolerance on ``diag(M_ff)``. Solves the condensed
    problem ``(K_mm − K_mr K_rr⁻¹ K_rm) u_m = ω² M_mm u_m`` and then
    reconstructs ``u_r = −K_rr⁻¹ K_rm u_m`` so the returned eigenvectors
    are full free-DOF vectors and can drop straight into
    ``modes_full[free_idx, :]`` without changes to downstream code.

    Returns:
        ``(eigvals, eigvecs_free, n_modes, info_messages)`` where
        ``eigvecs_free`` is ``n_free × n_modes``.

    Raises:
        ValueError: If no mass-bearing free DOF is detected, or if the
            massless-stiffness block ``K_rr`` is singular / ill-
            conditioned (a model whose massless DOFs are not rigidly
            coupled to the mass DOFs cannot be condensed). The error
            text is suitable for surfacing to the user.
    """
    n_free = K_ff.shape[0]
    diag_M = np.diag(M_ff)
    diag_max = float(np.max(diag_M)) if n_free else 0.0
    if diag_max <= 0.0:
        raise ValueError(
            "Lumped modal: every free DOF is massless. Check that "
            "elements connected to free DOFs have positive ρ and A."
        )
    tol = _LUMPED_MASS_REL_TOL * diag_max
    mass_local = np.where(diag_M > tol)[0]
    rest_local = np.where(diag_M <= tol)[0]
    if mass_local.size == 0:
        raise ValueError(
            "Lumped modal: no mass-bearing free DOFs after applying "
            "tolerance. Check ρ, A, and supports."
        )

    K_mm = K_ff[np.ix_(mass_local, mass_local)]
    M_mm = M_ff[np.ix_(mass_local, mass_local)]

    if rest_local.size == 0:
        # No condensation needed — every free DOF carries mass.
        eigvals, eigvecs_mm = scipy.linalg.eigh(K_mm, M_mm)
        eigvecs_free = np.zeros((n_free, eigvecs_mm.shape[1]))
        eigvecs_free[mass_local, :] = eigvecs_mm
        n_avail = eigvals.size
        n_modes = max(1, min(n_modes_request, n_avail))
        return eigvals[:n_modes], eigvecs_free[:, :n_modes], n_modes, []

    K_rr = K_ff[np.ix_(rest_local, rest_local)]
    K_rm = K_ff[np.ix_(rest_local, mass_local)]
    # K_ff is symmetric, so K_mr is exactly K_rm.T — skip a redundant
    # np.ix_ slice (per gemini PR-19 review).
    K_mr = K_rm.T

    # K_rr⁻¹ K_rm via solve (never inv).
    try:
        # K_rr is a principal submatrix of the SPD K_ff, hence itself
        # symmetric positive-definite — use Cholesky via assume_a="pos"
        # (~2× faster and more stable than Bunch-Kaufman "sym"). Per
        # gemini PR-19 review.
        K_rr_sym = 0.5 * (K_rr + K_rr.T)
        Krr_inv_Krm = scipy.linalg.solve(
            K_rr_sym, K_rm, assume_a="pos",
        )
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ValueError(
            "Lumped modal: massless-DOF stiffness block K_rr is "
            "singular or ill-conditioned, so the rotational DOFs "
            "cannot be condensed out. The structure may have an "
            "unconstrained rotational DOF (e.g. a hinge with no "
            "translational mass attached). Either pin / restrain the "
            "DOF or use the Consistent mass formulation."
        ) from exc

    K_cond = K_mm - K_mr @ Krr_inv_Krm
    K_cond = 0.5 * (K_cond + K_cond.T)

    eigvals, eigvecs_mm = scipy.linalg.eigh(K_cond, M_mm)

    n_avail = eigvals.size
    n_modes = max(1, min(n_modes_request, n_avail))
    eigvals = eigvals[:n_modes]
    eigvecs_mm = eigvecs_mm[:, :n_modes]

    # u_r = −K_rr⁻¹ K_rm u_m — quasi-static recovery on the massless
    # block so the returned mode shapes are full free-DOF vectors.
    eigvecs_rr = -Krr_inv_Krm @ eigvecs_mm

    eigvecs_free = np.zeros((n_free, n_modes))
    eigvecs_free[mass_local, :] = eigvecs_mm
    eigvecs_free[rest_local, :] = eigvecs_rr

    info: list[str] = []
    if n_modes_request > n_avail:
        info.append(
            f"Lumped modal: requested {n_modes_request} modes, returned "
            f"{n_modes} (number of mass-bearing free DOFs)."
        )
    return eigvals, eigvecs_free, n_modes, info


def _build_mass_source_summary(
    source: ModalMassSource,
    model: StructuralModel,
    info: list[str],
) -> str:
    """Build the human-readable mass-source summary line for ModalResult."""
    parts: list[str] = []
    if source.include_self_mass:
        parts.append("self-mass")
    if source.include_joint_masses and model.joint_masses:
        n = len(model.joint_masses)
        parts.append(f"joint masses ({n} {'entry' if n == 1 else 'entries'})")
    if source.include_load_cases and source.load_case_factors:
        active = {k: v for k, v in source.load_case_factors.items() if v > 0.0}
        if active:
            terms = ", ".join(
                f"{k}×{v:g}" for k, v in sorted(active.items())
            )
            parts.append(f"cases ({terms})")
    if not parts:
        return "no mass sources active"
    return " + ".join(parts)


def _solve_components(
    model: StructuralModel,
    K: np.ndarray,
    M: np.ndarray,
    dofs: DofManager,
    n_modes_request: int,
    normalisation: str,
    mass_formulation: str,
) -> tuple[list[ComponentModalResult], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Solve modal eigenproblem independently per connected component.

    Returns:
        ``(components, freqs_all, periods_all, omegas_all, modes_all, warnings)``
        where the ``*_all`` arrays are the concatenated flat results in
        component order (C1 modes, then C2 modes, …) ready to drop into
        :class:`ModalResult`.

    Each component's mode vectors are embedded into the full global DOF
    space (length ``n_total``) with zeros at DOFs outside that component.
    """
    raw_components = _connectivity_components(model)
    n_total = dofs.n_total
    free_set = set(dofs.free_indices)
    supported_nodes = set(model.supports.keys())

    # Build element lookup: node_id → set of element IDs
    elem_by_node: dict[int, list[int]] = {nid: [] for nid in model.nodes}
    for elem in model.elements:
        elem_by_node.setdefault(elem.node_i, []).append(elem.id)
        elem_by_node.setdefault(elem.node_j, []).append(elem.id)

    components: list[ComponentModalResult] = []
    all_freqs: list[np.ndarray] = []
    all_periods: list[np.ndarray] = []
    all_omegas: list[np.ndarray] = []
    all_modes: list[np.ndarray] = []
    extra_warnings: list[str] = []
    global_offset = 0

    for cid, node_set in enumerate(raw_components, start=1):
        node_ids = sorted(node_set)

        # Elements whose both endpoints are in this component
        elem_ids = sorted(
            e.id for e in model.elements
            if e.node_i in node_set and e.node_j in node_set
        )

        # Orphan node: belongs to no element
        if not elem_ids:
            components.append(ComponentModalResult(
                component_id=cid,
                node_ids=node_ids,
                element_ids=[],
                is_supported=any(n in supported_nodes for n in node_ids),
                is_singular=False,
                skip_reason="orphan node — no elements",
                n_modes=0,
                frequencies=np.zeros(0),
                periods=np.zeros(0),
                omegas=np.zeros(0),
                global_mode_offset=global_offset,
            ))
            extra_warnings.append(
                f"Component {cid} (node {node_ids[0]}): orphan node — skipped."
            )
            continue

        # Support check
        is_supported = any(n in supported_nodes for n in node_ids)
        if not is_supported:
            components.append(ComponentModalResult(
                component_id=cid,
                node_ids=node_ids,
                element_ids=elem_ids,
                is_supported=False,
                is_singular=False,
                skip_reason="no supports (unsupported disconnected component)",
                n_modes=0,
                frequencies=np.zeros(0),
                periods=np.zeros(0),
                omegas=np.zeros(0),
                global_mode_offset=global_offset,
            ))
            extra_warnings.append(
                f"Component {cid} ({len(node_ids)} nodes, {len(elem_ids)} elements): "
                f"no supports — modal analysis skipped for this component."
            )
            continue

        # Collect free DOF indices for this component
        comp_free: list[int] = []
        for nid in node_ids:
            m = dofs.active_map.get(nid, {})
            for idx in m.values():
                if idx is not None and idx in free_set:
                    comp_free.append(idx)
        comp_free = sorted(set(comp_free))

        if not comp_free:
            components.append(ComponentModalResult(
                component_id=cid,
                node_ids=node_ids,
                element_ids=elem_ids,
                is_supported=True,
                is_singular=False,
                skip_reason="no free DOFs (fully restrained)",
                n_modes=0,
                frequencies=np.zeros(0),
                periods=np.zeros(0),
                omegas=np.zeros(0),
                global_mode_offset=global_offset,
            ))
            continue

        # Extract submatrices
        K_comp = K[np.ix_(comp_free, comp_free)]
        M_comp = M[np.ix_(comp_free, comp_free)]

        # Check for mass
        M_comp_diag = np.diag(M_comp)
        if not np.any(M_comp_diag > 0.0):
            components.append(ComponentModalResult(
                component_id=cid,
                node_ids=node_ids,
                element_ids=elem_ids,
                is_supported=True,
                is_singular=False,
                skip_reason="no mass on free DOFs",
                n_modes=0,
                frequencies=np.zeros(0),
                periods=np.zeros(0),
                omegas=np.zeros(0),
                global_mode_offset=global_offset,
            ))
            extra_warnings.append(
                f"Component {cid} ({len(node_ids)} nodes): "
                f"no mass on free DOFs — modal analysis skipped."
            )
            continue

        # Solve
        _diag_max = float(np.max(M_comp_diag))
        _has_massless = np.any(
            M_comp_diag <= _LUMPED_MASS_REL_TOL * max(1.0, _diag_max)
        )
        is_singular = False
        comp_warn: list[str] = []
        try:
            if mass_formulation == "lumped" or _has_massless:
                eigvals, modes_comp_free, n_comp_modes, comp_warn = (
                    _solve_modal_condensed(K_comp, M_comp, n_modes_request)
                )
            else:
                eigvals_all, eigvecs_all = scipy.linalg.eigh(K_comp, M_comp)
                n_avail = eigvals_all.size
                n_comp_modes = max(1, min(n_modes_request, n_avail))
                eigvals = eigvals_all[:n_comp_modes]
                modes_comp_free = eigvecs_all[:, :n_comp_modes].copy()
        except ValueError as exc:
            # Singular Kff within this component
            is_singular = True
            components.append(ComponentModalResult(
                component_id=cid,
                node_ids=node_ids,
                element_ids=elem_ids,
                is_supported=True,
                is_singular=True,
                skip_reason=f"singular stiffness: {exc}",
                n_modes=0,
                frequencies=np.zeros(0),
                periods=np.zeros(0),
                omegas=np.zeros(0),
                global_mode_offset=global_offset,
            ))
            extra_warnings.append(
                f"Component {cid} ({len(node_ids)} nodes): "
                f"singular or near-singular stiffness — modal analysis skipped."
            )
            continue

        eigvals = np.maximum(eigvals, 0.0)
        comp_omegas = np.sqrt(eigvals)
        comp_freqs = comp_omegas / (2.0 * np.pi)
        with np.errstate(divide="ignore"):
            comp_periods = np.where(
                comp_freqs > 0.0,
                1.0 / np.where(comp_freqs == 0.0, 1.0, comp_freqs),
                np.inf,
            )

        if normalisation == "max":
            for k in range(modes_comp_free.shape[1]):
                peak = float(np.max(np.abs(modes_comp_free[:, k])))
                if peak > 0.0:
                    modes_comp_free[:, k] /= peak

        # Embed into full-DOF global space
        comp_free_arr = np.array(comp_free, dtype=int)
        modes_global = np.zeros((n_total, n_comp_modes))
        modes_global[comp_free_arr, :] = modes_comp_free

        components.append(ComponentModalResult(
            component_id=cid,
            node_ids=node_ids,
            element_ids=elem_ids,
            is_supported=True,
            is_singular=False,
            skip_reason=None,
            n_modes=n_comp_modes,
            frequencies=comp_freqs,
            periods=comp_periods,
            omegas=comp_omegas,
            global_mode_offset=global_offset,
        ))
        all_freqs.append(comp_freqs)
        all_periods.append(comp_periods)
        all_omegas.append(comp_omegas)
        all_modes.append(modes_global)
        extra_warnings.extend(comp_warn)
        global_offset += n_comp_modes

    if all_modes:
        freqs_all = np.concatenate(all_freqs)
        periods_all = np.concatenate(all_periods)
        omegas_all = np.concatenate(all_omegas)
        modes_all = np.concatenate(all_modes, axis=1)
    else:
        freqs_all = np.zeros(0)
        periods_all = np.zeros(0)
        omegas_all = np.zeros(0)
        modes_all = np.zeros((n_total, 0))

    return components, freqs_all, periods_all, omegas_all, modes_all, extra_warnings


def solve_modal(
    model: StructuralModel,
    n_modes: int = 6,
    normalisation: str = "mass",
    *,
    mass_formulation: MassFormulation = "consistent",
    mass_source: ModalMassSource | None = None,
) -> ModalResult:
    """Run a free-vibration analysis and return a :class:`ModalResult`.

    The static stiffness assembler is reused to build K and to obtain
    the active :class:`DofManager`; the mass assembler builds M with the
    same DOF ordering. The eigenvalue problem is restricted to the
    free-DOF block (:attr:`DofManager.free_indices`) — boundary
    conditions are imposed via DOF restraint exactly as in the static
    solve, and prescribed settlements are ignored (free vibration
    studies displacements about the rest state).

    Args:
        model: The structural model to analyse.
        n_modes: Maximum number of modes to return. Capped at the number
            of free DOFs (consistent) or mass-bearing free DOFs
            (lumped).
        normalisation: ``"mass"`` (default — eigenvectors satisfy
            ``φᵀ·M·φ = 1``) or ``"max"`` (each mode scaled so the
            largest absolute entry is 1).
        mass_formulation: ``"consistent"`` (default — unchanged from
            v0.9.1) or ``"lumped"`` (translational-only mass matrix,
            rotational DOFs condensed out via Guyan reduction).

    Returns:
        A populated :class:`ModalResult`.

    Raises:
        ValueError: If no element carries a positive material density
            (so the global mass matrix would be zero and no vibration
            problem is defined), if ``normalisation`` or
            ``mass_formulation`` is unknown, or if the lumped path
            cannot be condensed (no mass-bearing free DOFs / singular
            massless-DOF stiffness block).
    """
    if normalisation not in ("mass", "max"):
        raise ValueError(
            f"Unknown normalisation {normalisation!r}; expected 'mass' or 'max'."
        )
    if mass_formulation not in ("consistent", "lumped"):
        raise ValueError(
            f"Unknown mass formulation {mass_formulation!r}; "
            "expected 'consistent' or 'lumped'."
        )

    # Resolve effective mass source (None → safe default = density only).
    effective_source = mass_source if mass_source is not None else ModalMassSource()

    # Pre-flight check: at least one mass source must be able to contribute.
    # (The old density-only check is relaxed: joint masses or load-case mass
    # also satisfy the requirement.)
    _has_density = any(getattr(e, "rho", 0.0) > 0.0 for e in model.elements)
    _has_joint = (
        effective_source.include_joint_masses and bool(model.joint_masses)
    )
    _has_lc = (
        effective_source.include_load_cases
        and any(v > 0.0 for v in effective_source.load_case_factors.values())
    )
    if not (
        (effective_source.include_self_mass and _has_density)
        or _has_joint
        or _has_lc
    ):
        raise ValueError(
            "Modal analysis requires mass. Enable at least one of: element "
            "material density (include_self_mass), joint masses "
            "(include_joint_masses with entries in the model), or load-case "
            "mass (include_load_cases with a positive factor). Configure in "
            "Run → Modal mass source…"
        )

    # ── Pre-detect components (before assembly) ───────────────────────────
    # Detect connected components so we can add dummy supports to floating
    # components and remove orphan nodes from the assembly model. This
    # prevents assemble_global_system() from raising for those cases,
    # while keeping the original model clean for per-component classification.
    raw_comps = _connectivity_components(model)
    use_per_component = len(raw_comps) > 1

    assembly_model = model
    if use_per_component:
        _real_supported = set(model.supports.keys())
        _orphan_nids: set[int] = set()
        _unsupported_nids: set[int] = set()
        for _comp_nodes in raw_comps:
            _is_orphan = not any(
                e.node_i in _comp_nodes and e.node_j in _comp_nodes
                for e in model.elements
            )
            if _is_orphan:
                _orphan_nids |= _comp_nodes
            elif not any(n in _real_supported for n in _comp_nodes):
                _unsupported_nids |= _comp_nodes

        if _orphan_nids or _unsupported_nids:
            _new_nodes = {
                nid: n for nid, n in model.nodes.items()
                if nid not in _orphan_nids
            }
            _new_supports = dict(model.supports)
            for nid in _unsupported_nids:
                _new_supports[nid] = Support(node_id=nid, ux=True, uy=True, rz=False)
            assembly_model = _dc_replace(model, nodes=_new_nodes, supports=_new_supports)

    K, _F, dofs, warnings, _elem_data = assemble_global_system(assembly_model)
    M, mass_info = assemble_mass_matrix_with_source(
        assembly_model, dofs,
        formulation=mass_formulation,
        source=effective_source,
    )

    free = list(dofs.free_indices)
    if not free:
        raise ValueError(
            "Modal analysis has no free DOFs — the structure is fully "
            "restrained. Release at least one support DOF before running modal."
        )

    # Verify assembled M has non-zero diagonal on free DOFs (guards against
    # e.g. density=0 + joint masses only on restrained nodes).
    M_ff_diag = np.diag(M)[free]
    if not np.any(M_ff_diag > 0.0):
        raise ValueError(
            "The assembled mass matrix has no non-zero diagonals on free "
            "DOFs.  Check that mass sources are connected to free (not "
            "fully restrained) nodes."
        )

    summary = _build_mass_source_summary(effective_source, model, mass_info)
    combined_base = list(warnings) + mass_info
    if mass_formulation == "lumped":
        combined_base.append(LUMPED_COMPARISON_NOTE)

    # ── Multi-component path ──────────────────────────────────────────────

    if use_per_component:
        components, freqs, periods, omegas, modes_full, comp_warns = (
            _solve_components(
                model, K, M, dofs, n_modes, normalisation, mass_formulation,
            )
        )

        solved = [c for c in components if c.skip_reason is None]
        if not solved:
            skipped_reasons = "; ".join(
                f"C{c.component_id}: {c.skip_reason}" for c in components
            )
            raise ValueError(
                f"Modal analysis: all components were skipped. {skipped_reasons}"
            )

        n_modes_returned = sum(c.n_modes for c in components)
        combined_warnings = combined_base + comp_warns

        n_supported = sum(1 for c in components if c.is_supported)
        n_skipped = sum(1 for c in components if c.skip_reason is not None)
        comp_summary_parts = [
            f"Model contains {len(components)} disconnected component"
            f"{'s' if len(components) != 1 else ''}. "
            f"Modal analysis solved each supported component separately."
        ]
        if n_skipped:
            comp_summary_parts.append(
                f"{n_skipped} component{'s' if n_skipped != 1 else ''} skipped "
                f"(see warnings)."
            )
        component_summary = " ".join(comp_summary_parts)

        return ModalResult(
            status="ok",
            title=model.title,
            warnings=combined_warnings,
            n_modes=n_modes_returned,
            frequencies=freqs,
            periods=periods,
            omegas=omegas,
            modes=modes_full,
            normalisation=normalisation,
            dofs=dofs,
            mass_formulation=mass_formulation,
            mass_source_summary=summary,
            components=components,
            component_summary=component_summary,
        )

    # ── Single-component path (existing behaviour, unchanged) ─────────────
    K_ff = K[np.ix_(free, free)]
    M_ff = M[np.ix_(free, free)]

    _diag_M_ff = np.diag(M_ff)
    _has_massless = np.any(_diag_M_ff <= _LUMPED_MASS_REL_TOL * max(1.0, float(np.max(_diag_M_ff))))

    extra_warnings: list[str] = []
    if mass_formulation == "lumped" or _has_massless:
        eigvals, modes_free, n_modes_returned, extra_warnings = (
            _solve_modal_condensed(K_ff, M_ff, n_modes)
        )
    else:
        eigvals_all, eigvecs_all = scipy.linalg.eigh(K_ff, M_ff)
        n_avail = eigvals_all.size
        n_modes_returned = max(1, min(n_modes, n_avail))
        eigvals = eigvals_all[:n_modes_returned]
        modes_free = eigvecs_all[:, :n_modes_returned].copy()

    eigvals = np.maximum(eigvals, 0.0)
    omegas = np.sqrt(eigvals)

    freqs = omegas / (2.0 * np.pi)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0.0, 1.0 / np.where(freqs == 0.0, 1.0, freqs), np.inf)

    if normalisation == "max":
        for k in range(modes_free.shape[1]):
            peak = float(np.max(np.abs(modes_free[:, k])))
            if peak > 0.0:
                modes_free[:, k] = modes_free[:, k] / peak

    n_total = dofs.n_total
    modes_full = np.zeros((n_total, n_modes_returned))
    free_idx = np.array(free, dtype=int)
    modes_full[free_idx, :] = modes_free

    combined_warnings = combined_base + extra_warnings

    return ModalResult(
        status="ok",
        title=model.title,
        warnings=combined_warnings,
        n_modes=n_modes_returned,
        frequencies=freqs,
        periods=periods,
        omegas=omegas,
        modes=modes_full,
        normalisation=normalisation,
        dofs=dofs,
        mass_formulation=mass_formulation,
        mass_source_summary=summary,
        components=[],
        component_summary="",
    )
