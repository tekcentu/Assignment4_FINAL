"""
Solver: partitioned K·D = F with SVD-based singularity detection.
Supports prescribed displacements (support settlement).

Partitioned system:
    [K_ff  K_fs] [D_f]   [F_f]
    [K_sf  K_ss] [D_s] = [F_s + R]      (D_s prescribed, R = reactions)

Solve:
    K_ff · D_f = F_f - K_fs · D_s
    R = K_sf · D_f + K_ss · D_s - F_s

When D_s = 0 (no settlement) this reduces to the standard case.
"""

from __future__ import annotations

import numpy as np

from .assembler import DofManager


def solve_system(
    K: np.ndarray,
    F: np.ndarray,
    dofs: DofManager,
    D_prescribed: np.ndarray | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """Solve the partitioned system with optional support settlements.

    Args:
        K: Full global stiffness matrix, ndarray (n × n).
        F: Full global load vector, ndarray (n,).
        dofs: DofManager with free/restrained index lists.
        D_prescribed: Optional full-length array of prescribed displacements.
            Non-zero entries at restrained DOFs represent support settlements.
            If None or all zeros, the classical solve applies.

    Returns:
        Tuple (D, residual, warnings) where:
            D: Full displacement vector, ndarray (n,).
            residual: float — ||K_ff · D_f − (F_f − K_fs · D_s)||.
            warnings: list[str] — any warnings or errors encountered.
    """
    warnings: list[str] = []
    n = K.shape[0]

    if D_prescribed is None:
        D_prescribed = np.zeros(n)

    if not dofs.free_indices:
        warnings.append("No free DOFs — structure is fully restrained.")
        # Return prescribed displacements as the full solution
        return D_prescribed.copy(), 0.0, warnings

    free = dofs.free_indices
    restrained = dofs.restrained_indices

    Kff = K[np.ix_(free, free)]
    Ff = F[free]

    # ── Account for prescribed settlements: effective load = F_f − K_fs · D_s ──
    has_settlement = bool(restrained) and np.any(D_prescribed[restrained] != 0.0)
    if has_settlement:
        Kfs = K[np.ix_(free, restrained)]
        Ds = D_prescribed[restrained]
        Ff_eff = Ff - Kfs @ Ds
    else:
        Ff_eff = Ff

    # ── SVD rank check on K_ff ──
    try:
        u, s, vh = np.linalg.svd(Kff, full_matrices=False)
    except np.linalg.LinAlgError:
        warnings.append("ERROR: SVD failed on K_ff.")
        return np.full(n, np.nan), float("inf"), warnings

    tol = max(Kff.shape) * np.max(s) * 1e-12
    rank = int(np.sum(s > tol))

    if rank < Kff.shape[0]:
        null_vec = vh[-1]
        dominant = np.argsort(np.abs(null_vec))[::-1][:3]
        dominant_labels = [
            dofs.labels[free[i]]
            for i in dominant
            if abs(null_vec[i]) > 1e-6
        ]
        warnings.append(
            f"ERROR: Singular stiffness matrix (rank {rank}/{Kff.shape[0]}). "
            f"Structure is unstable. "
            f"Mechanism DOFs: {', '.join(dominant_labels) if dominant_labels else 'undetermined'}."
        )
        return np.full(n, np.nan), float("inf"), warnings

    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    if cond > 1e12:
        warnings.append(
            f"WARNING: Ill-conditioned K (cond ≈ {cond:.3e}). Results may be unreliable."
        )

    try:
        Df = np.linalg.solve(Kff, Ff_eff)
    except np.linalg.LinAlgError:
        warnings.append("ERROR: Solve failed — singular matrix.")
        return np.full(n, np.nan), float("inf"), warnings

    # Build full D vector: prescribed at restrained, solved at free
    D = D_prescribed.copy()
    D[free] = Df

    residual = float(np.linalg.norm(Kff @ Df - Ff_eff))
    if residual > 1e-3:
        warnings.append(f"WARNING: Large residual = {residual:.4e}.")

    return D, residual, warnings
