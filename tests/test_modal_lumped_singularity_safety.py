"""Regression: lumped-mass modal solve does NOT crash on free DOFs with
zero rotational mass — the textbook ``scipy.linalg.LinAlgError: The
leading minor of order ... of B is not positive definite`` failure mode
is avoided by static (Guyan) condensation in
``modal._solve_modal_condensed`` (it partitions free DOFs into mass-
bearing and massless using a tolerance on ``diag(M_ff)``, then solves a
positive-definite condensed system on the mass-bearing block only).

These tests pin three behaviours so a future refactor cannot silently
reintroduce the crash, the regularization-by-stealth, or the unwanted
"diagonal is a mass formulation" confusion:

1.  A free rotational DOF with zero lumped mass DOES NOT crash modal
    solve and returns the expected number of physical modes (the
    mass-bearing DOF count).
2.  The condensation path is exercised (not a plain ``eigh`` call on a
    singular B) — verified via the mode count: a cantilever with one
    free node has 3 free DOFs but only 2 mass-bearing translational
    DOFs under lumped mass, so a working condensation yields 2 modes
    while a plain ``eigh`` would raise.
3.  No tiny mass / regularization is silently added to the assembled
    mass matrix — the lumped-formulation diagonal stays exactly zero on
    rotational entries.

The first few physical modes from the lumped path are also compared
against the consistent path; we don't expect bit-equality (different
mass distributions) but the lumped fundamental must be the same order
of magnitude as the consistent fundamental — proving the condensation
is not silently dropping or scrambling the physical modes.
"""

from __future__ import annotations

import pytest

from structural_analysis.element import FrameElement2D
from structural_analysis.mass import assemble_mass_matrix
from structural_analysis.assembler import DofManager
from structural_analysis.model import (
    Material, Node, Section, StructuralModel, Support,
)
from structural_analysis.modal import solve_modal


def _free_cantilever_with_rotational_dof():
    """Horizontal cantilever fixed at node 1, free at node 2 — node 2's
    free rz DOF has zero lumped mass (lumped is translational-only).
    Under the consistent formulation this same model has full rank, so
    the two paths give comparable fundamentals."""
    m = StructuralModel(title="lumped rz=0 cantilever")
    m.materials[1] = Material(id=1, name="C", E=2.0e8, density=7850.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.02, I=8.0e-4, depth=0.3)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 5.0, 0.0)}
    m.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.0e8,
        A=0.02, I=8.0e-4, section_id=1, rho=7850.0,
    )]
    m.supports = {1: Support(node_id=1, ux=True, uy=True, rz=True)}
    return m


def test_lumped_modal_does_not_crash_on_zero_rotational_mass():
    """The user's exact failure mode (eigh raising on a singular B
    because a free rz DOF has zero lumped mass) must NOT happen — the
    condensation path absorbs the massless rotational DOF and the
    solver returns a clean ModalResult."""
    m = _free_cantilever_with_rotational_dof()
    result = solve_modal(m, n_modes=3, mass_formulation="lumped")
    # Lumped on this fixture has 2 mass-bearing free DOFs (ux, uy at
    # node 2) and 1 massless (rz at node 2). After condensation: 2 modes.
    assert result.mass_formulation == "lumped"
    assert len(result.omegas) == 2
    assert all(om > 0 for om in result.omegas)


def test_lumped_mass_keeps_zero_rotational_diagonal_no_regularization():
    """The fix must NOT silently add tiny mass to rz DOFs. Confirm that
    the assembled M still has exactly zero rotational diagonal entries
    under the lumped formulation — the modal solver works *around* the
    singularity, it does NOT *hide* it with regularization."""
    m = _free_cantilever_with_rotational_dof()
    dofs = DofManager.from_model(m)
    M = assemble_mass_matrix(m, dofs, formulation="lumped")
    # Look up the free rz DOF index at node 2.
    eq = dofs.index(node_id=2, dof="rz")
    assert eq is not None and 0 <= eq < M.shape[0]
    # Lumped is translational-only: EXACTLY zero (not a tiny "regularized"
    # value) on rz. This is the contract the condensation path relies on.
    assert M[eq, eq] == 0.0


def test_consistent_modal_unchanged_default_path():
    """Backward-compat: consistent (the default) is unaffected — returns
    3 modes (all DOFs mass-bearing) and round-trips its tag."""
    m = _free_cantilever_with_rotational_dof()
    r_default = solve_modal(m, n_modes=3)
    r_explicit = solve_modal(m, n_modes=3, mass_formulation="consistent")
    assert r_default.mass_formulation == "consistent"
    assert r_explicit.mass_formulation == "consistent"
    assert len(r_default.omegas) == 3
    for a, b in zip(r_default.omegas, r_explicit.omegas):
        assert a == pytest.approx(b, rel=1e-12)


def test_lumped_first_mode_is_same_order_as_consistent_first_mode():
    """Sanity: the condensation result is *physical*, not garbage. The
    fundamental ω from the lumped path should be within a factor of 2 of
    the consistent fundamental on this fixture (lumped is lower because
    translational mass is concentrated at the tip — guarded loosely so
    different fixtures wouldn't flake)."""
    m = _free_cantilever_with_rotational_dof()
    om_c = solve_modal(m, n_modes=1, mass_formulation="consistent").omegas[0]
    om_l = solve_modal(m, n_modes=1, mass_formulation="lumped").omegas[0]
    assert 0.25 * om_c <= om_l <= 4.0 * om_c
