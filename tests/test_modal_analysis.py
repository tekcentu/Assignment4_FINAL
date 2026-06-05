"""Verification tests for the free-vibration modal analyser.

Three closed-form / textbook cases (clamped-free beam, simply-supported
beam, two-DOF shear building) and two protective tests (density-zero
error and mode orthogonality).

All beam cases use kN-m engineering units (E in kN/m², I in m⁴, A in m²,
L in m). Material density is supplied in kg/m³; the modal module
converts internally so that frequencies come out in Hz.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.element import FrameElement2D
from structural_analysis.model import (
    Node, StructuralModel, Support,
)
from structural_analysis.modal import solve_modal


# ── helpers ──────────────────────────────────────────────────────


def _uniform_beam(n_elems: int, L: float, E: float, A: float, I: float,
                  rho: float, *, fix_left: bool, fix_right: bool) -> StructuralModel:
    """Build a horizontal beam meshed into ``n_elems`` equal frame elements.

    ``fix_left`` / ``fix_right`` flag whether to fully clamp that end
    (ux, uy, rz all restrained). Intermediate nodes are free.
    """
    m = StructuralModel(title=f"beam-n{n_elems}")
    xs = np.linspace(0.0, L, n_elems + 1)
    m.nodes = {i + 1: Node(i + 1, float(x), 0.0) for i, x in enumerate(xs)}
    m.elements = []
    for k in range(n_elems):
        e = FrameElement2D(
            id=k + 1,
            node_i=k + 1,
            node_j=k + 2,
            E=E, A=A, I=I, rho=rho,
        )
        m.elements.append(e)
    if fix_left:
        m.supports[1] = Support(1, ux=True, uy=True, rz=True)
    if fix_right:
        last = n_elems + 1
        # If the right end is simply supported (pin) we only restrain uy
        # (the LH support pins ux for the SS-beam case).
        m.supports[last] = Support(last, ux=True, uy=True, rz=True) if fix_left and fix_right \
            else Support(last, ux=False, uy=True, rz=False)
    return m


# ── 1. clamped-free beam — first three transverse frequencies ────


def test_clamped_free_beam_first_three_frequencies():
    """Cantilever Euler-Bernoulli beam.

    f_n = (β_n L)² · √(EI / (ρ A L⁴)) / (2π)
    with β_n L = 1.8751, 4.6941, 7.8548 for n = 1, 2, 3.
    """
    L, E, A, I, rho = 5.0, 200e6, 0.005, 1.0e-5, 7850.0  # kN/m², m², m⁴, kg/m³
    # Build the cantilever with a fine mesh so FE error is small.
    model = _uniform_beam(n_elems=16, L=L, E=E, A=A, I=I, rho=rho,
                          fix_left=True, fix_right=False)
    r = solve_modal(model, n_modes=6, normalisation="mass")
    assert r.status == "ok"
    assert r.n_modes >= 3
    # Closed-form. EI uses kN-m² units; ρA in kg/m → divide by 1000 to
    # get Mg/m so the radicand is in m⁴/s² → root in m²/s.
    EI = E * I
    mu = (rho / 1000.0) * A           # Mg/m  (= kN·s²/m²)
    betaL = np.array([1.8751, 4.6941, 7.8548])
    expected = (betaL ** 2) * np.sqrt(EI / (mu * L ** 4)) / (2.0 * np.pi)
    for n in range(3):
        rel = abs(r.frequencies[n] - expected[n]) / expected[n]
        assert rel < 0.01, (
            f"mode {n + 1}: got {r.frequencies[n]:.4f} Hz, "
            f"expected ~{expected[n]:.4f} Hz, rel err {rel:.4%}"
        )


# ── 2. simply-supported beam frequencies ─────────────────────────


def test_simply_supported_beam_first_two_frequencies():
    """Simply-supported (pin-roller) beam.

    f_n = (n π / L)² · √(EI / (ρA)) / (2π)
    """
    L, E, A, I, rho = 6.0, 200e6, 0.01, 8.0e-5, 7850.0
    model = _uniform_beam(n_elems=20, L=L, E=E, A=A, I=I, rho=rho,
                          fix_left=False, fix_right=False)
    # Pin at left, roller at right (only uy at right).
    model.supports[1] = Support(1, ux=True, uy=True, rz=False)
    last = len(model.nodes)
    model.supports[last] = Support(last, ux=False, uy=True, rz=False)
    r = solve_modal(model, n_modes=4, normalisation="mass")
    EI = E * I
    mu = (rho / 1000.0) * A
    expected = []
    for n in (1, 2):
        omega = (n * np.pi / L) ** 2 * np.sqrt(EI / mu)
        expected.append(omega / (2.0 * np.pi))
    for n in range(2):
        rel = abs(r.frequencies[n] - expected[n]) / expected[n]
        assert rel < 0.01, (
            f"SS mode {n + 1}: got {r.frequencies[n]:.4f} Hz, "
            f"expected ~{expected[n]:.4f} Hz, rel err {rel:.4%}"
        )


# ── 3. two-DOF shear-building topology ───────────────────────────


def test_two_dof_shear_building_eigenpair():
    """Two-storey shear-building idealisation.

    We build a vertical stick of two clamped-clamped columns. The base
    is clamped, and at each "floor" node we restrain uy and rz so the
    only free DOFs are the two horizontal sways — that is exactly the
    two-DOF shear-building topology used in textbooks (e.g. Chopra
    *Dynamics of Structures*, §10).

    The textbook closed-form ``ω² = (3 ± √5)/2 · k/m`` assumes a
    *lumped* mass concentrated at each floor. The consistent-mass FE
    formulation distributes the element mass across both ends through
    the Hermite shape functions, so the FE-distributed M_ff matrix
    differs from the textbook diagonal m·I — and so do the absolute
    frequencies (typically ~30 % higher for this configuration). To
    keep the test exact rather than approximate, we therefore:

      1. assemble K_ff and M_ff from the same FE pipeline,
      2. compute the analytical closed-form eigenvalues of that 2×2
         generalised eigenproblem (a quadratic in ω²), and
      3. assert that :func:`solve_modal` reproduces those values.

    The mode shapes are also checked qualitatively: mode 1 is in-phase
    (both floors swaying together with the upper floor moving more),
    mode 2 is out-of-phase. These two qualitative facts are
    distribution-independent — they're true for both lumped-mass and
    consistent-mass shear buildings.
    """
    from structural_analysis.assembler import assemble_global_system
    from structural_analysis.mass import assemble_mass_matrix

    m_floor = 2000.0   # kg
    k_storey = 5.0e4   # kN/m
    h = 3.0            # storey height, m
    E = 200e6          # kN/m²
    I = k_storey * h ** 3 / (12.0 * E)
    A = 1.0e-3

    # Density chosen so element mass m̄·L equals m_floor (in kg).
    # rho_consistent = rho_user / 1000 (Mg/m³); m̄ = rho_consistent · A;
    # m̄·L = rho_user · A · L / 1000 (Mg). Want m̄·L = m_floor / 1000 (Mg)
    # → rho_user = m_floor / (A · L) [kg/m³].
    rho_user = m_floor / (A * h)

    model = StructuralModel(title="2-DOF shear building")
    model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 0.0, h),
        3: Node(3, 0.0, 2 * h),
    }
    model.elements = [
        FrameElement2D(1, 1, 2, E=E, A=A, I=I, rho=rho_user),
        FrameElement2D(2, 2, 3, E=E, A=A, I=I, rho=rho_user),
    ]
    model.supports = {
        1: Support(1, ux=True, uy=True, rz=True),
        2: Support(2, ux=False, uy=True, rz=True),
        3: Support(3, ux=False, uy=True, rz=True),
    }

    K, _F, dofs, _w, _ed = assemble_global_system(model)
    M = assemble_mass_matrix(model, dofs)
    free = np.array(dofs.free_indices, dtype=int)
    Kff = K[np.ix_(free, free)]
    Mff = M[np.ix_(free, free)]
    assert Kff.shape == (2, 2)

    # Sanity: the FE stiffness must be the shear-building stiffness
    # matrix [[2k, -k], [-k, k]] (with k = k_storey) — this is a direct
    # consequence of stacking two clamped-clamped 12EI/L³ columns.
    np.testing.assert_allclose(
        Kff, np.array([[2 * k_storey, -k_storey], [-k_storey, k_storey]]),
        rtol=1e-9,
    )

    # Closed-form eigenvalues of the symmetric 2×2 generalised problem
    # det(K − ω²·M) = 0. Let x = ω². Then
    #   (K11 − x·M11)(K22 − x·M22) − (K12 − x·M12)² = 0
    K11, K22, K12 = Kff[0, 0], Kff[1, 1], Kff[0, 1]
    M11, M22, M12 = Mff[0, 0], Mff[1, 1], Mff[0, 1]
    a = M11 * M22 - M12 ** 2
    b = -(K11 * M22 + K22 * M11 - 2.0 * K12 * M12)
    c = K11 * K22 - K12 ** 2
    disc = b * b - 4.0 * a * c
    assert disc > 0.0
    x1 = (-b - np.sqrt(disc)) / (2.0 * a)
    x2 = (-b + np.sqrt(disc)) / (2.0 * a)
    omegas_expected = np.sqrt([x1, x2])
    freqs_expected = omegas_expected / (2.0 * np.pi)

    r = solve_modal(model, n_modes=2, normalisation="mass")
    assert r.n_modes == 2
    np.testing.assert_allclose(r.frequencies, freqs_expected, rtol=1e-9)
    np.testing.assert_allclose(r.omegas, omegas_expected, rtol=1e-9)

    # Mode-shape check on the two free DOFs (node 2 UX, node 3 UX).
    phi1 = r.modes[free, 0]
    phi2 = r.modes[free, 1]
    # Mode 1: both floors in-phase, upper floor displaces more.
    assert phi1[0] * phi1[1] > 0.0, "mode 1 should be in-phase"
    assert abs(phi1[1]) > abs(phi1[0]), "mode 1: upper floor must move more than lower"
    # Mode 2: floors out-of-phase.
    assert phi2[0] * phi2[1] < 0.0, "mode 2 should be out-of-phase"


# ── 4. mode orthogonality ────────────────────────────────────────


def test_modes_are_mass_orthogonal():
    """For mass-normalised modes: φᵢᵀ · M · φⱼ should be δ_ij."""
    L, E, A, I, rho = 4.0, 200e6, 0.005, 5.0e-5, 7850.0
    model = _uniform_beam(n_elems=10, L=L, E=E, A=A, I=I, rho=rho,
                          fix_left=True, fix_right=False)
    r = solve_modal(model, n_modes=5, normalisation="mass")
    # Rebuild M with the same DOF manager to check orthogonality.
    from structural_analysis.assembler import assemble_global_system
    from structural_analysis.mass import assemble_mass_matrix
    _K, _F, dofs, _w, _ed = assemble_global_system(model)
    M = assemble_mass_matrix(model, dofs)
    free = np.array(dofs.free_indices, dtype=int)
    M_ff = M[np.ix_(free, free)]
    phi = r.modes[free, :]
    gram = phi.T @ M_ff @ phi
    eye = np.eye(r.n_modes)
    assert np.allclose(gram, eye, atol=1e-8), (
        f"mass-orthogonality violated, max off-diagonal "
        f"{np.max(np.abs(gram - eye)):.3e}"
    )


# ── 5. density-zero error ────────────────────────────────────────


def test_density_zero_model_raises():
    """A model whose materials all carry density = 0 must refuse the
    modal solve with a clear error."""
    L, E, A, I = 4.0, 200e6, 0.005, 5.0e-5
    model = _uniform_beam(n_elems=4, L=L, E=E, A=A, I=I, rho=0.0,
                          fix_left=True, fix_right=False)
    with pytest.raises(ValueError, match="density"):
        solve_modal(model)


# ── 6. mass_source kwarg — backward-compat & equivalence ────────


def test_solve_modal_default_source_matches_legacy():
    """Explicit mass_source=None and source=ModalMassSource() both give
    the same frequencies as the old API (no mass_source argument)."""
    from structural_analysis.model import ModalMassSource

    L, E, A, I, rho = 5.0, 200e6, 0.005, 1.0e-5, 7850.0
    model = _uniform_beam(n_elems=8, L=L, E=E, A=A, I=I, rho=rho,
                          fix_left=True, fix_right=False)

    r_old = solve_modal(model, n_modes=4)
    r_none = solve_modal(model, n_modes=4, mass_source=None)
    r_default = solve_modal(model, n_modes=4, mass_source=ModalMassSource())

    np.testing.assert_array_almost_equal(r_old.frequencies, r_none.frequencies, decimal=8)
    np.testing.assert_array_almost_equal(r_old.frequencies, r_default.frequencies, decimal=8)


def test_density_zero_with_joint_mass_succeeds():
    """A ρ=0 model with a joint mass should produce a valid ModalResult
    (the old density-only guard must not fire)."""
    from structural_analysis.model import JointMass, Material, ModalMassSource, Section

    L, E, A, I = 2.0, 200e6, 1e-2, 1e-4
    model = StructuralModel()
    model.materials[1] = Material(id=1, E=E, density=0.0)
    model.sections[1] = Section(id=1, material_id=1, A=A, I=I, depth=0.1)
    model.nodes[1] = Node(1, 0.0, 0.0)
    model.nodes[2] = Node(2, L, 0.0)
    model.supports[1] = Support(1, ux=True, uy=True, rz=True)
    model.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=E, A=A, I=I,
        alpha=0.0, depth=0.1, rho=0.0, section_id=1,
    ))
    model.joint_masses[2] = JointMass(node_id=2, mx=500.0, my=500.0)
    src = ModalMassSource(include_self_mass=False, include_joint_masses=True)

    r = solve_modal(model, n_modes=2, mass_source=src)
    assert r.status == "ok"
    assert r.n_modes >= 1
    assert np.all(r.frequencies >= 0.0)


def test_modal_result_has_mass_source_summary():
    """ModalResult.mass_source_summary must be a non-empty string."""
    L, E, A, I, rho = 2.0, 200e6, 5e-3, 1e-5, 7850.0
    model = _uniform_beam(n_elems=4, L=L, E=E, A=A, I=I, rho=rho,
                          fix_left=True, fix_right=False)
    r = solve_modal(model, n_modes=2)
    assert isinstance(r.mass_source_summary, str)
    assert r.mass_source_summary  # non-empty
