"""Global mass-matrix assembly for free-vibration analysis.

Mirrors :func:`structural_analysis.assembler.assemble_global_system` but
only builds the global mass matrix M from the elements'
:meth:`Element2D.consistent_mass_local`. The same :class:`DofManager`
that built the stiffness DOF order is reused, so M is guaranteed to
share K's row/column ordering.

Unit system. Material density is stored on :class:`Material` and mirrored
to the element as ``rho`` in kg/m³ — the unit users naturally write. The
element-level mass routines convert to the consistent kN-m-s system used
by the static stiffness pipeline (mass in Mg = kN·s²/m) before emitting
their local 6×6 matrices, so the eigenvalue problem ``(K − ω²·M)·φ = 0``
yields ``ω`` directly in rad/s and ``f = ω/(2π)`` directly in Hz.
"""

from __future__ import annotations

import numpy as np

from .assembler import DofManager
from .model import StructuralModel


def assemble_mass_matrix(
    model: StructuralModel,
    dofs: DofManager,
) -> np.ndarray:
    """Assemble the global consistent mass matrix.

    Args:
        model: The structural model.
        dofs: DOF manager whose ordering matches the one used for K.

    Returns:
        ``n_total × n_total`` numpy array — global mass matrix M in
        consistent kN-m-s units (Mg per translational diagonal entry).
    """
    n = dofs.n_total
    M = np.zeros((n, n))
    for elem in model.elements:
        m_local = elem.consistent_mass_local(model.nodes)
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
