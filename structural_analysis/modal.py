"""Free-vibration modal analysis.

Solves the generalised eigenvalue problem ``(K − ω²·M)·φ = 0`` on the
free-DOF partition of the global system. Returns a :class:`ModalResult`
carrying natural frequencies (Hz), periods (s), angular frequencies
(rad/s) and full-length mode-shape vectors (with zeros at restrained
DOFs so they can be drawn directly on top of the model geometry).

A dedicated result object is used instead of overloading
:class:`AnalysisResult` (which stays bound to static analysis) — the
GUI dispatches on the result type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg

from .assembler import DofManager, assemble_global_system
from .mass import assemble_mass_matrix
from .model import StructuralModel


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


def solve_modal(
    model: StructuralModel,
    n_modes: int = 6,
    normalisation: str = "mass",
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
            of free DOFs.
        normalisation: ``"mass"`` (default — eigenvectors satisfy
            ``φᵀ·M·φ = 1``) or ``"max"`` (each mode scaled so the
            largest absolute entry is 1).

    Returns:
        A populated :class:`ModalResult`.

    Raises:
        ValueError: If no element carries a positive material density
            (so the global mass matrix would be zero and no vibration
            problem is defined), or if ``normalisation`` is unknown.
    """
    if normalisation not in ("mass", "max"):
        raise ValueError(
            f"Unknown normalisation {normalisation!r}; expected 'mass' or 'max'."
        )

    if not any(getattr(elem, "rho", 0.0) > 0.0 for elem in model.elements):
        raise ValueError(
            "Modal analysis requires at least one element whose material "
            "carries a positive density. Set density on the Material "
            "(kg/m³) before running modal."
        )

    K, _F, dofs, warnings, _elem_data = assemble_global_system(model)
    M = assemble_mass_matrix(model, dofs)

    free = list(dofs.free_indices)
    if not free:
        raise ValueError(
            "Modal analysis has no free DOFs — the structure is fully "
            "restrained. Release at least one support DOF before running modal."
        )
    K_ff = K[np.ix_(free, free)]
    M_ff = M[np.ix_(free, free)]

    eigvals, eigvecs = scipy.linalg.eigh(K_ff, M_ff)

    # Small negative numerical noise on near-rigid-body modes.
    eigvals = np.maximum(eigvals, 0.0)
    omegas = np.sqrt(eigvals)
    n_avail = len(omegas)
    n_modes = max(1, min(n_modes, n_avail))

    omegas = omegas[:n_modes]
    freqs = omegas / (2.0 * np.pi)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0.0, 1.0 / np.where(freqs == 0.0, 1.0, freqs), np.inf)
    modes_free = eigvecs[:, :n_modes].copy()

    if normalisation == "max":
        for k in range(modes_free.shape[1]):
            peak = float(np.max(np.abs(modes_free[:, k])))
            if peak > 0.0:
                modes_free[:, k] = modes_free[:, k] / peak

    n_total = dofs.n_total
    modes_full = np.zeros((n_total, n_modes))
    free_idx = np.array(free, dtype=int)
    modes_full[free_idx, :] = modes_free

    return ModalResult(
        status="ok",
        title=model.title,
        warnings=list(warnings),
        n_modes=n_modes,
        frequencies=freqs,
        periods=periods,
        omegas=omegas,
        modes=modes_full,
        normalisation=normalisation,
        dofs=dofs,
    )
