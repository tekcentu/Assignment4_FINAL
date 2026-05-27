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


def test_row_sum_is_block_restricted_to_same_dof_type():
    """Row-sum must restrict to columns of the same DOF type so a uy
    row never picks up rz coupling columns (which would mix kg with
    kg·m). Verified against a hand calculation on the symmetric beam
    consistent-mass matrix: uy row at one node sums to ρAL/2 when
    restricted to uy columns; an unrestricted sum would instead give
    (210 + 9L)/420 · ρAL — see PR #18 review.
    """
    m, L, rho, A = _single_frame_model(L=5.0)
    report = joint_mass_table(m, method="row_sum")

    rows_by_id = {row.node_id: row for row in report.rows}
    expected_per_node = rho * A * L / 2.0  # ρAL/2
    for nid in (1, 2):
        uy = rows_by_id[nid].values["uy"]
        ux = rows_by_id[nid].values["ux"]
        assert isinstance(uy, float)
        assert isinstance(ux, float)
        # Symmetric model — both nodes should carry ρAL/2 translational
        # mass in both ux and uy under block-restricted row-sum.
        assert uy == pytest.approx(expected_per_node, rel=1e-9)
        assert ux == pytest.approx(expected_per_node, rel=1e-9)


def test_row_sum_rotational_cell_is_physical_kg_m2():
    """rz row, restricted to rz columns, must yield a non-negative
    physical kg·m² value. Cross-block sums could go negative because
    of the -3L² off-diagonal in the consistent mass — that's the
    reviewer's concern, locked down here."""
    m, L, rho, A = _single_frame_model(L=5.0)
    report = joint_mass_table(m, method="row_sum")

    for row in report.rows:
        rz = row.values["rz"]
        assert isinstance(rz, float)
        assert rz >= 0.0, f"rz block-row-sum must be ≥ 0, got {rz}"

    # Hand value: rz-row over rz-columns is (4L² - 3L²)·coef = L²·coef
    # where coef = ρ_consistent · A · L / 420 (kg/m³ → Mg/m³ → ×1000 = kg).
    # So per-node rz = L² · ρ A L / 420 (in kg·m²).
    expected_per_node = (L * L) * rho * A * L / 420.0
    for row in report.rows:
        assert row.values["rz"] == pytest.approx(expected_per_node, rel=1e-9)


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


# ── 5. degenerate-mass warning banner ─────────────────────────


def test_warning_all_zero_density_model():
    """Every element has ρ=0 (typical of legacy inputs/*.txt fixtures
    that omit the density column) → the report carries the global
    'all assembled element mass contributions are zero' warning, and
    the table still renders (rows present, just zero)."""
    m, _, _, _ = _single_frame_model(rho=0.0)
    report = joint_mass_table(m, method="row_sum")
    assert report.warning is not None
    assert "All assembled element mass contributions are zero" in report.warning
    assert "ρ" in report.warning  # explicit density hint
    assert "A" in report.warning  # explicit area hint
    # Table still populated, just zeros.
    assert len(report.rows) == 2
    for row in report.rows:
        for dof in ("ux", "uy", "rz"):
            v = row.values[dof]
            assert v == "not_active" or v == 0.0


def test_warning_partial_zero_mass_lists_offending_element_ids():
    """A mixed model (one frame with ρ·A > 0, one with A=0) → the
    warning starts with 'Some elements have zero mass contribution'
    and contains the bad element's ID."""
    m = StructuralModel(title="mixed")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 5.0, 0.0)
    m.nodes[3] = Node(3, 10.0, 0.0)
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(id=1, name="S1", material_id=1, A=0.01, I=1e-4, depth=0.3)
    m.sections[2] = Section(id=2, name="S2-zero", material_id=1, A=0.0, I=1e-4, depth=0.3)
    m.elements.append(FrameElement2D(
        id=10, node_i=1, node_j=2,
        E=2.1e8, A=0.01, I=1e-4, rho=7850.0, depth=0.3, section_id=1,
    ))
    m.elements.append(FrameElement2D(
        id=11, node_i=2, node_j=3,
        E=2.1e8, A=0.0, I=1e-4, rho=7850.0, depth=0.3, section_id=2,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)

    report = joint_mass_table(m, method="row_sum")
    assert report.warning is not None
    assert report.warning.startswith("Some elements have zero mass contribution")
    assert "11" in report.warning  # bad element ID is listed
    # The good element's ID shouldn't be in the list.
    after_colon = report.warning.split(":", 1)[1]
    assert "10" not in after_colon


def test_warning_healthy_model_is_none():
    """A model with ρ·A > 0 on every element → no warning."""
    m, _, _, _ = _single_frame_model()
    report = joint_mass_table(m, method="row_sum")
    assert report.warning is None


# ── 6. inspection-only contract (regression lock) ─────────────


def test_joint_mass_table_never_invokes_solve_modal(monkeypatch):
    """`joint_mass_table` must read M directly — it must never trigger
    a modal eigenvalue solve. Lock this down so a future refactor that
    accidentally calls into modal raises here first.
    """
    import structural_analysis.modal as modal_mod

    def _boom(*args, **kwargs):
        raise AssertionError(
            "solve_modal was invoked from the inspection path"
        )

    monkeypatch.setattr(modal_mod, "solve_modal", _boom)
    m, _, _, _ = _single_frame_model()
    # Must not raise.
    report = joint_mass_table(m, method="row_sum")
    assert report.warning is None


# ── 7. v0.9.2 mass-formulation pass-through ──────────────────


def test_joint_mass_table_lumped_zeros_rotational_cells():
    """v0.9.2: with mass_formulation="lumped", every rz cell is 0.0,
    and the report's formulation label flips."""
    m, L, rho, A = _single_frame_model(L=5.0)
    report = joint_mass_table(m, method="row_sum", mass_formulation="lumped")
    assert report.formulation == "Lumped translational mass"

    # rz cell on the free node (node 2) is 0.0 under lumped.
    free_row = next(r for r in report.rows if r.node_id == 2)
    rz = free_row.values["rz"]
    assert isinstance(rz, float)
    assert rz == pytest.approx(0.0, abs=1e-12)

    # Translational totals still match ρ·A·L (translational mass is
    # preserved in both formulations).
    expected_kg = rho * A * L
    assert report.totals_kg["ux"] == pytest.approx(expected_kg, rel=1e-9)
    assert report.totals_kg["uy"] == pytest.approx(expected_kg, rel=1e-9)
    # And rz column total is zero.
    assert report.totals_kg["rz"] == pytest.approx(0.0, abs=1e-12)


def test_joint_mass_table_unknown_formulation_raises():
    m, _, _, _ = _single_frame_model()
    with pytest.raises(ValueError, match="Unknown mass formulation"):
        joint_mass_table(m, mass_formulation="hrz")  # type: ignore[arg-type]
