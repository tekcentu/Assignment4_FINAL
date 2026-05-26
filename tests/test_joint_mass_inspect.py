"""Tests for v0.9.1 Assembled Joint Masses inspection.

Covers ``structural_analysis.mass_inspect.joint_mass_table``:
  - row-sum equivalent translational totals match ρ·A·L
  - consistent mass produces nonzero rotational diagonals on frames
  - restrained DOFs are flagged in Notes but still carry a number
  - the inspection helper has no side effect on modal frequencies
"""

from __future__ import annotations

import numpy as np
import pytest

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.mass_inspect import joint_mass_table
from structural_analysis.modal import solve_modal
from structural_analysis.model import (
    Material, Node, Section, StructuralModel, Support,
)


# ── helpers ────────────────────────────────────────────────────


def _single_frame_model(L: float = 5.0, rho: float = 7850.0, A: float = 0.01):
    m = StructuralModel(title="joint-mass-frame")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, L, 0.0)
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=rho)
    m.sections[1] = Section(id=1, name="S", material_id=1, A=A, I=1e-4, depth=0.3)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=A, I=1e-4, rho=rho, depth=0.3, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m, L, rho, A


def _single_truss_model(L: float = 4.0, rho: float = 7850.0, A: float = 0.005):
    m = StructuralModel(title="joint-mass-truss")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, L, 0.0)
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=rho)
    m.sections[1] = Section(id=1, name="T", material_id=1, A=A, I=0.0, depth=0.05)
    m.elements.append(TrussElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=A, rho=rho, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True)
    m.supports[2] = Support(node_id=2, uy=True)
    return m, L, rho, A


# ── 1. row-sum equivalent total mass == ρ·A·L ─────────────────


def test_row_sum_translational_total_matches_rho_A_L():
    """Σ M[i, :] over translational DOFs equals the bar's mass (kg)."""
    m, L, rho, A = _single_frame_model()
    report = joint_mass_table(m, method="row_sum")

    expected_kg = rho * A * L  # ρ kg/m³ · A m² · L m → kg
    assert report.totals_kg["ux"] == pytest.approx(expected_kg, rel=1e-9)
    assert report.totals_kg["uy"] == pytest.approx(expected_kg, rel=1e-9)


# ── 2. consistent mass has nonzero rotational rz on frame ─────


def test_consistent_mass_has_nonzero_rz_on_frame_diagonal():
    """FrameElement2D's consistent mass diagonal at θ DOFs is positive."""
    m, _, _, _ = _single_frame_model()
    report = joint_mass_table(m, method="diagonal")

    rows_by_id = {row.node_id: row for row in report.rows}
    for nid in (1, 2):
        rz = rows_by_id[nid].values["rz"]
        assert isinstance(rz, float)
        assert rz > 0.0, f"node {nid} rz diagonal expected > 0, got {rz}"


def test_truss_only_model_reports_rz_not_active():
    """A pure-truss node has no assembled rotational DOF — must read as
    ``not active``, not as zero."""
    m, _, _, _ = _single_truss_model()
    report = joint_mass_table(m, method="row_sum")
    for row in report.rows:
        assert row.values["rz"] == "not_active"
        assert "rz not active" in row.notes()


# ── 3. restrained DOFs are flagged but still numeric ──────────


def test_restrained_dofs_are_flagged_but_keep_numeric_mass():
    """Per the design: restrained DOFs that exist in M still show a
    number; ``Notes`` flags the restraint. Totals include them."""
    m, _, rho, A = _single_frame_model()
    report = joint_mass_table(m, method="row_sum")

    fixed_row = next(r for r in report.rows if r.node_id == 1)
    # All three DOFs at node 1 are restrained.
    assert fixed_row.restrained == {"ux": True, "uy": True, "rz": True}
    notes = fixed_row.notes()
    assert "ux restrained" in notes
    assert "uy restrained" in notes
    assert "rz restrained" in notes
    # Numeric cells, not sentinels.
    for dof in ("ux", "uy", "rz"):
        assert isinstance(fixed_row.values[dof], float)
    # And those numbers participate in the column totals (row-sum
    # method): node-1 ux + node-2 ux = ρ·A·L.
    other_row = next(r for r in report.rows if r.node_id == 2)
    assert fixed_row.values["ux"] + other_row.values["ux"] == pytest.approx(
        rho * 0.01 * 5.0, rel=1e-9,
    )


# ── 4. inspection has no side effect on modal frequencies ─────


def test_joint_mass_inspection_does_not_affect_modal_frequencies():
    """Running joint_mass_table between two modal solves must yield
    identical frequencies — the helper assembles a fresh M and never
    touches model state."""
    m, _, _, _ = _single_frame_model()
    # Make node 2 modal-free so there's something to vibrate.
    # (node 1 stays fully fixed.)

    r_before = solve_modal(m, n_modes=3)
    _ = joint_mass_table(m, method="row_sum")
    _ = joint_mass_table(m, method="diagonal")
    r_after = solve_modal(m, n_modes=3)

    assert r_before.n_modes == r_after.n_modes
    np.testing.assert_allclose(
        r_before.frequencies, r_after.frequencies, rtol=0.0, atol=1e-12,
    )
    np.testing.assert_allclose(
        r_before.periods, r_after.periods, rtol=0.0, atol=1e-12,
    )


# ── extras: method validation & free-DOF count ────────────────


def test_unknown_method_raises():
    m, _, _, _ = _single_frame_model()
    with pytest.raises(ValueError, match="Unknown method"):
        joint_mass_table(m, method="lumped")  # type: ignore[arg-type]


def test_active_modal_dof_count_matches_dof_manager():
    """``n_free_dofs`` in the report equals ``len(DofManager.free_indices)``
    so the footer always agrees with the modal solver's view of the
    model."""
    from structural_analysis.assembler import DofManager

    m, _, _, _ = _single_frame_model()
    dofs = DofManager.from_model(m)
    report = joint_mass_table(m)
    assert report.n_free_dofs == len(dofs.free_indices)
    assert report.n_total_dofs == dofs.n_total
