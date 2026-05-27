"""v0.9.2 tests — translational-lumped mass assembly path.

Covers:
  - FrameElement2D.lumped_mass_local: diagonal pattern, ρ·A·L conservation,
    zero rotational mass, isotropic to rotation
  - TrussElement2D.lumped_mass_local: same diagonal, zero rotational
  - assemble_mass_matrix(formulation="lumped"): total translational
    mass matches Σ ρ·A·L; rotational diagonals are zero everywhere
  - default formulation is unchanged (consistent path is byte-for-byte
    identical to v0.9.1)
  - unknown formulation raises
"""

from __future__ import annotations

import numpy as np
import pytest

from structural_analysis.assembler import DofManager
from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.mass import assemble_mass_matrix
from structural_analysis.model import (
    Material, Node, Section, StructuralModel, Support,
)


# ── helpers ────────────────────────────────────────────────────


def _frame_at_angle(L: float, theta_deg: float, *, rho: float = 7850.0, A: float = 0.01):
    """Single-element frame whose axis makes ``theta_deg`` with global x."""
    th = np.deg2rad(theta_deg)
    m = StructuralModel(title=f"frame-{theta_deg}")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, L * np.cos(th), L * np.sin(th))
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=rho)
    m.sections[1] = Section(id=1, name="S", material_id=1, A=A, I=1e-4, depth=0.3)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=A, I=1e-4, rho=rho, depth=0.3, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m, L, rho, A


# ── element-level lumped mass ─────────────────────────────────


def test_frame_lumped_mass_local_has_expected_diag_pattern():
    """Each diagonal translational entry equals ρ·A·L/2 (in Mg);
    rotational entries are zero; off-diagonals are zero."""
    m, L, rho, A = _frame_at_angle(L=4.0, theta_deg=0.0)
    elem = m.elements[0]
    M_local = elem.lumped_mass_local(m.nodes)
    assert M_local.shape == (6, 6)

    expected_half = (rho / 1000.0) * A * L / 2.0  # Mg
    for idx in (0, 1, 3, 4):
        assert M_local[idx, idx] == pytest.approx(expected_half, rel=1e-12)
    # Rotational entries zero.
    for idx in (2, 5):
        assert M_local[idx, idx] == 0.0
    # All off-diagonals zero.
    off = M_local - np.diag(np.diag(M_local))
    assert np.allclose(off, 0.0)


def test_frame_lumped_mass_total_equals_rho_A_L():
    m, L, rho, A = _frame_at_angle(L=4.0, theta_deg=0.0)
    elem = m.elements[0]
    M_local = elem.lumped_mass_local(m.nodes)
    total_translational = (
        M_local[0, 0] + M_local[1, 1] + M_local[3, 3] + M_local[4, 4]
    )
    # Half mass on each end on both ux and uy → 4·(m/2) = 2m. But the
    # *physical* mass is m, not 2m. The 2m sum is what you get if you
    # sum the matrix; the conservation property the lumped model
    # preserves is "ux total = m" AND separately "uy total = m".
    expected_kg = rho * A * L  # m in kg
    expected_Mg = expected_kg / 1000.0
    # Per-direction total: ux sum at both ends = m; uy sum at both ends = m.
    ux_sum = M_local[0, 0] + M_local[3, 3]
    uy_sum = M_local[1, 1] + M_local[4, 4]
    assert ux_sum == pytest.approx(expected_Mg, rel=1e-12)
    assert uy_sum == pytest.approx(expected_Mg, rel=1e-12)
    # Sanity on the sum above.
    assert total_translational == pytest.approx(2.0 * expected_Mg, rel=1e-12)


def test_frame_lumped_mass_is_rotation_invariant():
    """Because the translational block is m/2 · I₂, R.T M R = M for
    any element orientation. Verified by comparing global lumped M
    on two beams of identical geometry but different angles."""
    # Build a horizontal frame and a tilted frame separately, transform
    # each element-local matrix to global, and compare diagonals.
    for theta in (0.0, 30.0, 45.0, 90.0):
        m, _, _, _ = _frame_at_angle(L=4.0, theta_deg=theta)
        elem = m.elements[0]
        M_local = elem.lumped_mass_local(m.nodes)
        R = elem.transformation_matrix(m.nodes)
        M_global = R.T @ M_local @ R
        # Diagonals on ux/uy slots must equal the local diagonal.
        for idx in (0, 1, 3, 4):
            assert M_global[idx, idx] == pytest.approx(
                M_local[idx, idx], rel=1e-12,
            )
        # Rotational diagonals still zero.
        for idx in (2, 5):
            assert M_global[idx, idx] == pytest.approx(0.0, abs=1e-15)


def test_truss_lumped_mass_local_has_expected_diag_pattern():
    m = StructuralModel(title="truss")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 3.0, 0.0)
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(id=1, name="T", material_id=1, A=0.005, I=0.0, depth=0.05)
    elem = TrussElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=0.005, rho=7850.0, section_id=1,
    )
    m.elements.append(elem)
    M_local = elem.lumped_mass_local(m.nodes)
    expected_half = (7850.0 / 1000.0) * 0.005 * 3.0 / 2.0
    for idx in (0, 1, 3, 4):
        assert M_local[idx, idx] == pytest.approx(expected_half, rel=1e-12)
    for idx in (2, 5):
        assert M_local[idx, idx] == 0.0


def test_lumped_mass_short_circuits_on_zero_rho_A():
    """ρ·A ≤ 0 → 6×6 zeros, parallel to the consistent path."""
    m, _, _, _ = _frame_at_angle(L=4.0, theta_deg=0.0, rho=0.0)
    elem = m.elements[0]
    M_local = elem.lumped_mass_local(m.nodes)
    assert np.allclose(M_local, 0.0)


# ── global assembly ───────────────────────────────────────────


def test_assemble_mass_lumped_zeros_every_rz_diagonal():
    """assemble_mass_matrix(formulation="lumped") must produce zero
    rotational mass on every rz DOF — that is the entire point of the
    lumped path."""
    m, _, _, _ = _frame_at_angle(L=4.0, theta_deg=30.0)
    # Free end (node 2) to give some rz DOFs.
    dofs = DofManager.from_model(m)
    M_lumped = assemble_mass_matrix(m, dofs, formulation="lumped")
    M_consistent = assemble_mass_matrix(m, dofs, formulation="consistent")

    # Every rz DOF in the active map → zero on lumped diagonal.
    rz_count_checked = 0
    for nid, dof_map in dofs.active_map.items():
        rz_idx = dof_map.get("rz")
        if rz_idx is None:
            continue
        assert M_lumped[rz_idx, rz_idx] == pytest.approx(0.0, abs=1e-15)
        rz_count_checked += 1
    assert rz_count_checked > 0, "fixture should have at least one rz DOF"
    # Consistent still carries rz mass for comparison.
    consistent_rz_mass = sum(
        float(M_consistent[dofs.active_map[nid]["rz"], dofs.active_map[nid]["rz"]])
        for nid in dofs.active_map
        if dofs.active_map[nid].get("rz") is not None
    )
    assert consistent_rz_mass > 0.0


def test_assemble_mass_lumped_per_direction_total_matches_rho_A_L():
    """Σ_node M_lumped[ux_i, ux_i] over the global M equals ρ·A·L
    (total bar mass). Same for uy."""
    m, L, rho, A = _frame_at_angle(L=4.0, theta_deg=0.0)
    dofs = DofManager.from_model(m)
    M_lumped = assemble_mass_matrix(m, dofs, formulation="lumped")

    ux_total = 0.0
    uy_total = 0.0
    for nid in m.node_ids:
        ux = dofs.active_map[nid].get("ux")
        uy = dofs.active_map[nid].get("uy")
        if ux is not None:
            ux_total += float(M_lumped[ux, ux])
        if uy is not None:
            uy_total += float(M_lumped[uy, uy])

    # Mg → kg
    expected_kg = rho * A * L
    assert ux_total * 1000.0 == pytest.approx(expected_kg, rel=1e-12)
    assert uy_total * 1000.0 == pytest.approx(expected_kg, rel=1e-12)


def test_assemble_mass_default_is_consistent_and_unchanged():
    """Backward compat: calling assemble_mass_matrix(model, dofs)
    without the new kwarg must return exactly the consistent M."""
    m, _, _, _ = _frame_at_angle(L=4.0, theta_deg=0.0)
    dofs = DofManager.from_model(m)
    M_default = assemble_mass_matrix(m, dofs)
    M_explicit = assemble_mass_matrix(m, dofs, formulation="consistent")
    np.testing.assert_allclose(M_default, M_explicit, rtol=0.0, atol=0.0)


def test_assemble_mass_unknown_formulation_raises():
    m, _, _, _ = _frame_at_angle(L=4.0, theta_deg=0.0)
    dofs = DofManager.from_model(m)
    with pytest.raises(ValueError, match="Unknown mass formulation"):
        assemble_mass_matrix(m, dofs, formulation="hrz")  # type: ignore[arg-type]
