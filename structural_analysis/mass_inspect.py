"""Assembled-joint-mass inspection helper (v0.9.1).

Builds a per-node summary of the assembled global mass matrix produced
by :func:`structural_analysis.mass.assemble_mass_matrix`. This is a
*diagnostic* view of M — the modal solver continues to use the full
mass matrix and is unaffected by anything in this module.

Two equivalent-mass summary methods are offered, mirroring SAP2000's
"Assembled Joint Masses" table:

  * ``"diagonal"`` — display ``M[i, i]`` at each DOF row.
  * ``"row_sum"`` — display ``Σ_j M[i, j]`` at each DOF row. For
    consistent element mass this is the lumped-equivalent total mass
    that translates with that DOF; commonly used for SAP-style joint
    mass cross-checks against hand calculations.

Units. The mass matrix is assembled in kN·s²/m (= Mg) for translational
DOFs and kN·m·s² (= Mg·m²) for rotational DOFs. This module multiplies
by 1000 to return kg / kg·m², which is what the GUI table displays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .assembler import DofManager
from .mass import assemble_mass_matrix
from .model import StructuralModel


_DOFS: tuple[str, str, str] = ("ux", "uy", "rz")
_NOT_ACTIVE: str = "not_active"
_RESTRAINED: str = "restrained"

Method = Literal["diagonal", "row_sum"]


@dataclass
class NodeMassRow:
    """One row of the Assembled Joint Masses table.

    Attributes:
        node_id: The node identifier.
        values: Per-DOF cell, keyed by ``"ux"``, ``"uy"``, ``"rz"``.
            A ``float`` when the DOF exists in the assembled matrix
            (regardless of restraint) — kg for ``ux``/``uy``, kg·m² for
            ``rz``. Otherwise the sentinel ``"not_active"``.
        restrained: Per-DOF bool. ``True`` iff the DOF is restrained by
            the model's supports. False for inactive DOFs.
    """

    node_id: int
    values: dict[str, float | str]
    restrained: dict[str, bool]

    def notes(self) -> str:
        """Render a human-readable Notes field describing this row.

        Lists each DOF that is restrained or not active. Empty string
        when every DOF is active and free.
        """
        parts: list[str] = []
        for dof in _DOFS:
            v = self.values[dof]
            if v == _NOT_ACTIVE:
                parts.append(f"{dof} not active")
            elif self.restrained[dof]:
                parts.append(f"{dof} restrained")
        return ", ".join(parts)


@dataclass
class JointMassReport:
    """Per-node + global summary of the assembled mass matrix.

    Attributes:
        rows: One :class:`NodeMassRow` per model node, in
            ``model.node_ids`` order.
        method: ``"diagonal"`` or ``"row_sum"``.
        formulation: Always ``"Consistent element mass"`` in v0.9.1.
        n_free_dofs: Number of modal-free DOFs (= ``len(dofs.free_indices)``).
        n_total_dofs: Total assembled DOFs (free + restrained).
        totals_kg: Column totals — sum of every numeric cell in each
            DOF column. ``ux`` and ``uy`` in kg, ``rz`` in kg·m².
    """

    rows: list[NodeMassRow]
    method: Method
    formulation: str
    n_free_dofs: int
    n_total_dofs: int
    totals_kg: dict[str, float]


def joint_mass_table(
    model: StructuralModel,
    *,
    method: Method = "row_sum",
) -> JointMassReport:
    """Build the Assembled Joint Masses table for ``model``.

    This is purely diagnostic — it assembles a fresh copy of M via the
    existing :func:`assemble_mass_matrix` and reads from it. No state
    is cached or attached to the model.

    Args:
        model: The structural model to summarise.
        method: ``"row_sum"`` (default, SAP-style equivalent mass) or
            ``"diagonal"`` (raw ``M[i,i]``).

    Returns:
        A populated :class:`JointMassReport`.

    Raises:
        ValueError: If ``method`` is not one of the recognised summaries.
    """
    if method not in ("diagonal", "row_sum"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'diagonal' or 'row_sum'."
        )

    dofs = DofManager.from_model(model)
    M = assemble_mass_matrix(model, dofs)
    restrained_set = set(dofs.restrained_indices)

    rows: list[NodeMassRow] = []
    totals = {dof: 0.0 for dof in _DOFS}

    for nid in model.node_ids:
        values: dict[str, float | str] = {}
        restrained: dict[str, bool] = {}
        for dof in _DOFS:
            i = dofs.active_map[nid].get(dof)
            if i is None:
                values[dof] = _NOT_ACTIVE
                restrained[dof] = False
                continue
            if method == "diagonal":
                m_consistent = float(M[i, i])
            else:
                m_consistent = float(np.sum(M[i, :]))
            # Mg → kg (and Mg·m² → kg·m² for rz).
            m_kg = m_consistent * 1000.0
            values[dof] = m_kg
            restrained[dof] = i in restrained_set
            totals[dof] += m_kg
        rows.append(NodeMassRow(node_id=nid, values=values, restrained=restrained))

    return JointMassReport(
        rows=rows,
        method=method,
        formulation="Consistent element mass",
        n_free_dofs=len(dofs.free_indices),
        n_total_dofs=dofs.n_total,
        totals_kg=totals,
    )
