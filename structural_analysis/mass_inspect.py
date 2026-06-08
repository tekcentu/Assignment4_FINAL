"""Assembled-joint-mass inspection helper (v0.9.1).

Builds a per-node summary of the assembled global mass matrix produced
by :func:`structural_analysis.mass.assemble_mass_matrix`. This is a
*diagnostic* view of M — the modal solver continues to use the full
mass matrix and is unaffected by anything in this module.

Two equivalent-mass summary methods are offered, mirroring SAP2000's
"Assembled Joint Masses" table:

  * ``"diagonal"`` — display ``M[i, i]`` at each DOF row.
  * ``"row_sum"`` — display ``Σ_{j in same DOF block} M[i, j]`` at
    each DOF row. The sum is restricted to columns of the same DOF
    type (ux-row over ux-columns, uy-row over uy-columns, rz-row over
    rz-columns) so per-cell units stay physical: kg for translational
    rows, kg·m² for rotational rows. Mixing across blocks would add
    Mg + Mg·m (translational ↔ rotational coupling terms) and yield
    non-physical numbers at the cell level. The block-restricted sum
    still preserves the SAP-style identity Σ_node M_translational =
    total translational mass of the structure.

Units. The mass matrix is assembled in kN·s²/m (= Mg) for translational
DOFs and kN·m·s² (= Mg·m²) for rotational DOFs. This module multiplies
by 1000 to return kg / kg·m², which is what the GUI table displays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .assembler import DofManager
from .mass import MassFormulation, assemble_mass_matrix
from .model import StructuralModel


_DOFS: tuple[str, str, str] = ("ux", "uy", "rz")
_NOT_ACTIVE: str = "not_active"

Method = Literal["diagonal", "row_sum"]

_FORMULATION_LABEL: dict[MassFormulation, str] = {
    "consistent": "Consistent element mass",
    "lumped": "Lumped translational mass",
}


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
        warning: Optional non-fatal advisory string when the model
            assembled successfully but the resulting M is degenerate —
            e.g. every element contributes zero mass because both ρ
            and A are zero, or some legacy fixtures omit the density
            column. ``None`` when the report is healthy.
    """

    rows: list[NodeMassRow]
    method: Method
    formulation: str
    n_free_dofs: int
    n_total_dofs: int
    totals_kg: dict[str, float]
    warning: str | None = None


_ZERO_MASS_TOL: float = 1e-12  # kg threshold for "element contributes nothing"
_MAX_IDS_IN_WARNING: int = 10  # truncate element-ID list to keep banner short


def _zero_mass_element_ids(model: StructuralModel) -> list[int]:
    """Return IDs of elements whose ρ·A is non-positive (zero contribution).

    An element with ρ ≤ 0 *or* A ≤ 0 produces a 6×6 zero element-mass
    matrix (see ``FrameElement2D.consistent_mass_local`` /
    ``TrussElement2D.consistent_mass_local`` — both guard on
    ``m_bar = ρ·A ≤ 0`` and short-circuit to zeros). We don't try to
    distinguish "ρ=0" from "A=0" in the warning; both are equally
    valid causes and the user fixes them in the same Materials /
    Sections dialogs.
    """
    bad: list[int] = []
    for elem in model.elements:
        rho = float(getattr(elem, "rho", 0.0))
        A = float(getattr(elem, "A", 0.0))
        if rho * A <= 0.0:
            bad.append(int(elem.id))
    return bad


def _build_warning(model: StructuralModel) -> str | None:
    """Build the non-fatal advisory text for a :class:`JointMassReport`.

    Returns ``None`` when every element contributes a positive mass.
    Returns an "all-zero" message when *every* element is zero-mass
    (typical of legacy ``inputs/*.txt`` files that lack a density
    column). Returns a "some elements" message — with up to
    ``_MAX_IDS_IN_WARNING`` element IDs and an ``…(+N more)`` suffix
    — when only a subset is degenerate.
    """
    if not model.elements:
        return None  # empty model is its own kind of empty — no row to warn on
    bad = _zero_mass_element_ids(model)
    if not bad:
        return None
    if len(bad) == len(model.elements):
        return (
            "All assembled element mass contributions are zero. "
            "Check material density ρ (kg/m³) and section area A. "
            "Legacy files may have ρ = 0."
        )
    shown = bad[:_MAX_IDS_IN_WARNING]
    suffix = (
        f" …(+{len(bad) - _MAX_IDS_IN_WARNING} more)"
        if len(bad) > _MAX_IDS_IN_WARNING
        else ""
    )
    id_list = ", ".join(str(eid) for eid in shown) + suffix
    return (
        "Some elements have zero mass contribution. "
        f"Check ρ and A for elements: {id_list}."
    )


def joint_mass_table(
    model: StructuralModel,
    *,
    method: Method = "row_sum",
    mass_formulation: MassFormulation = "consistent",
) -> JointMassReport:
    """Build the Assembled Joint Masses table for ``model``.

    This is purely diagnostic — it assembles a fresh copy of M via the
    existing :func:`assemble_mass_matrix` and reads from it. No state
    is cached or attached to the model and the modal solver is never
    invoked.

    Args:
        model: The structural model to summarise.
        method: ``"row_sum"`` (default — block-restricted row sum) or
            ``"diagonal"`` (raw ``M[i,i]``).
        mass_formulation: ``"consistent"`` (default — energy-consistent
            Hermite-cubic mass) or ``"lumped"`` (translational-only
            mass, zero on every rotational DOF — comparison aid).

    Returns:
        A populated :class:`JointMassReport`.

    Raises:
        ValueError: If ``method`` or ``mass_formulation`` is not one of
            the recognised summaries.
    """
    if method not in ("diagonal", "row_sum"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'diagonal' or 'row_sum'."
        )
    if mass_formulation not in ("consistent", "lumped"):
        raise ValueError(
            f"Unknown mass formulation {mass_formulation!r}; "
            "expected 'consistent' or 'lumped'."
        )

    dofs = DofManager.from_model(model)
    M = assemble_mass_matrix(model, dofs, formulation=mass_formulation)
    restrained_set = set(dofs.restrained_indices)

    # Precompute the per-DOF-block row sums once (O(n²) numpy work)
    # instead of recomputing inside the per-node Python loop. For
    # method="row_sum" we restrict each row's sum to columns of the
    # same DOF type so per-cell units stay physical — see module
    # docstring for the unit-mixing rationale.
    block_row_sums: dict[str, np.ndarray] | None = None
    if method == "row_sum":
        block_row_sums = {}
        for dof in _DOFS:
            # DofManager.active_map[nid] may store inactive DOFs as
            # explicit ``None`` rather than omitting the key — filter
            # on the value, not key presence.
            col_idx = np.fromiter(
                (
                    idx for nid in model.node_ids
                    if (idx := dofs.active_map[nid].get(dof)) is not None
                ),
                dtype=np.intp,
            )
            if col_idx.size == 0:
                block_row_sums[dof] = np.zeros(dofs.n_total)
            else:
                block_row_sums[dof] = M[:, col_idx].sum(axis=1)

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
                # block_row_sums is populated above when method=="row_sum".
                m_consistent = float(block_row_sums[dof][i])  # type: ignore[index]
            # Mg → kg (and Mg·m² → kg·m² for rz).
            m_kg = m_consistent * 1000.0
            values[dof] = m_kg
            restrained[dof] = i in restrained_set
            totals[dof] += m_kg
        rows.append(NodeMassRow(node_id=nid, values=values, restrained=restrained))

    return JointMassReport(
        rows=rows,
        method=method,
        formulation=_FORMULATION_LABEL[mass_formulation],
        n_free_dofs=len(dofs.free_indices),
        n_total_dofs=dofs.n_total,
        totals_kg=totals,
        warning=_build_warning(model),
    )
