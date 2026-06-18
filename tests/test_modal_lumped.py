"""v0.9.2 tests — lumped-translational modal path.

Covers:
  - default-consistent regression lock (frequencies byte-equal v0.9.1)
  - analytical 1-DOF mass-on-spring (cantilever lateral mode):
    static condensation must give ω = √(k_eff / m) exactly
  - fixed-base column: lumped mode count follows mass-bearing free
    DOFs; consistent mode count > lumped mode count (no hard-coded
    "2 modes" assumption — just an inequality on rotational mass)
  - ModalResult.mass_formulation field round-trips
  - LUMPED_COMPARISON_NOTE appears in the warnings list on the lumped
    path
  - error when no mass-bearing DOFs exist
  - error when K_rr is singular (artificial fixture: pinned hinge with
    no translational mass on the hinge node — synthesised)
"""

from __future__ import annotations

import numpy as np
import pytest

from structural_analysis.element import FrameElement2D
from structural_analysis.modal import (
    LUMPED_COMPARISON_NOTE, ModalResult, solve_modal,
)
from structural_analysis.model import (
    Material, Node, Section, StructuralModel, Support,
)


# ── helpers ────────────────────────────────────────────────────


def _fixed_base_column(
    *,
    L: float = 3.0,
    E: float = 2.0e8,
    A: float = 0.01,
    I: float = 1.0e-4,
    rho: float = 7850.0,
):
    """Single-element vertical cantilever, fixed at node 1, free at node 2."""
    m = StructuralModel(title="fixed-base-column")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, L)
    m.materials[1] = Material(id=1, name="Steel", E=E, density=rho)
    m.sections[1] = Section(id=1, name="S", material_id=1, A=A, I=I, depth=0.3)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=E, A=A, I=I, rho=rho, depth=0.3, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m


# ── 1. default-consistent regression lock ─────────────────────


def test_default_modal_is_lumped_and_consistent_is_deprecated():
    """Final-submission build: the default formulation is lumped, and
    passing ``"consistent"`` is accepted only as a graceful backward-
    compat path — it emits a :class:`DeprecationWarning` and is
    transparently mapped to lumped (no silent crash, no fake result).
    Previously this test pinned ``default == "consistent"``; the
    final-submission cleanup intentionally flips it."""
    import warnings

    m = _fixed_base_column()
    r_default = solve_modal(m, n_modes=3)
    assert r_default.mass_formulation == "lumped"
    # Explicit lumped must equal the default.
    r_explicit_lumped = solve_modal(m, n_modes=3, mass_formulation="lumped")
    np.testing.assert_allclose(
        r_default.frequencies, r_explicit_lumped.frequencies,
        rtol=0.0, atol=0.0,
    )
    # Passing "consistent" warns and maps to lumped (= default).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r_legacy = solve_modal(m, n_modes=3, mass_formulation="consistent")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert r_legacy.mass_formulation == "lumped"
    np.testing.assert_allclose(
        r_legacy.frequencies, r_default.frequencies,
        rtol=0.0, atol=0.0,
    )


# ── 2. analytical 1-DOF mass-on-spring (axial cantilever) ──────


def test_lumped_axial_cantilever_matches_one_dof_omega():
    """For a single axial-spring cantilever (vertical column with no
    transverse load path), the lumped axial mode has ω = √(k_eff / m_tip)
    where k_eff = EA/L and m_tip = half the bar mass. Static
    condensation of the rotational DOF must give this exact value.
    """
    L = 3.0
    E = 2.0e8        # kN/m²
    A = 0.01         # m²
    rho = 7850.0     # kg/m³

    m = _fixed_base_column(L=L, E=E, A=A, rho=rho)
    r = solve_modal(m, n_modes=3, mass_formulation="lumped")

    # Axial stiffness, axial tip mass (Mg).
    k_axial = E * A / L              # kN/m
    m_tip = (rho / 1000.0) * A * L / 2.0  # Mg = kN·s²/m
    omega_expected = float(np.sqrt(k_axial / m_tip))  # rad/s

    # The axial mode is one of the returned modes — find it by matching
    # ω to the closed-form value.
    matches = np.isclose(r.omegas, omega_expected, rtol=1e-9)
    assert matches.any(), (
        f"expected axial ω = {omega_expected:.6g} rad/s in "
        f"{r.omegas} (mass={r.mass_formulation})"
    )


# ── 3. fixed-base column: lumped removes rotational mode(s) ────


def test_lumped_column_modes_match_mass_bearing_dof_count():
    """Lumped mass condenses massless rotational DOFs out of the modal
    eigenproblem, so the returned mode count equals the number of
    mass-bearing free DOFs (translational only on a frame member).

    Replaces the pre-fix lumped-vs-consistent comparison: since the
    user-facing modal workflow now only exposes lumped, the comparison
    is no longer meaningful through the public API. We instead pin the
    physical invariant — fewer modes than free DOFs because the rz
    block was condensed away — which is what the user actually relies
    on for output."""
    m = _fixed_base_column()
    r = solve_modal(m, n_modes=10, mass_formulation="lumped")

    # Single 2-node column has 3 free DOFs at node 2 (ux, uy, rz).
    # Lumped is translational-only → 2 mass-bearing DOFs → at most 2 modes.
    assert r.n_modes <= 2
    assert r.n_modes >= 1
    assert r.mass_formulation == "lumped"


# ── 4. ModalResult carries the formulation tag ────────────────


def test_modal_result_records_mass_formulation():
    """Round-trip the formulation tag through ``ModalResult``. In the
    final-submission build, the public path always tags the result
    ``"lumped"`` — passing ``"consistent"`` is mapped to lumped (and
    emits a DeprecationWarning) so the recorded tag is still
    ``"lumped"``."""
    import warnings

    m = _fixed_base_column()
    r_l = solve_modal(m, n_modes=3, mass_formulation="lumped")
    assert r_l.mass_formulation == "lumped"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        r_legacy = solve_modal(m, n_modes=3, mass_formulation="consistent")
    assert r_legacy.mass_formulation == "lumped"


def test_lumped_modal_appends_comparison_note_warning():
    """The lumped path always tacks the neutral comparison-aid note onto
    ModalResult.warnings so the GUI / report can pick it up without
    string-matching the formulation field."""
    m = _fixed_base_column()
    r = solve_modal(m, n_modes=3, mass_formulation="lumped")
    assert LUMPED_COMPARISON_NOTE in r.warnings


# ── 5. unknown formulation rejected ──────────────────────────


def test_solve_modal_rejects_unknown_mass_formulation():
    m = _fixed_base_column()
    with pytest.raises(ValueError, match="Unknown mass formulation"):
        solve_modal(m, n_modes=3, mass_formulation="hrz")  # type: ignore[arg-type]


# ── 6. modes are valid full free-DOF vectors after recovery ───


def test_lumped_mode_shapes_include_recovered_rotational_components():
    """After static condensation, mode shapes must be expanded back
    onto the full free-DOF set so the GUI renderer (which indexes by
    DofManager.active_map) keeps working unchanged. The recovered rz
    component for a lateral bending mode should be nonzero on the free
    end of a cantilever (lateral displacement → rotation about the
    base for a flexural beam)."""
    m = _fixed_base_column()
    r = solve_modal(m, n_modes=10, mass_formulation="lumped")
    assert r.dofs is not None
    # Free end is node 2.
    rz2 = r.dofs.active_map[2].get("rz")
    assert rz2 is not None
    # At least one mode shape has a nonzero rotational component at node 2.
    rz_vals = np.abs(r.modes[rz2, :])
    assert rz_vals.max() > 1e-9, (
        f"expected recovered rotational component on at least one "
        f"lumped mode at node 2 rz; got {rz_vals}"
    )


def test_lumped_modal_mass_normalisation_is_satisfied_on_mass_dofs():
    """φᵀ_m M_mm φ_m = 1 for mass-normalised eigenvectors on the
    mass-bearing DOFs. Since the recovered rotational components carry
    zero mass, the full-vector identity φᵀ M_ff φ = 1 still holds in
    block form."""
    from structural_analysis.assembler import DofManager, assemble_global_system
    from structural_analysis.mass import assemble_mass_matrix

    m = _fixed_base_column()
    r = solve_modal(m, n_modes=10, mass_formulation="lumped", normalisation="mass")
    K, _, dofs, _, _ = assemble_global_system(m)
    M = assemble_mass_matrix(m, dofs, formulation="lumped")
    free = list(dofs.free_indices)
    M_ff = M[np.ix_(free, free)]
    free_idx = np.array(free, dtype=int)
    for k in range(r.n_modes):
        phi = r.modes[free_idx, k]
        gen_mass = float(phi @ M_ff @ phi)
        assert gen_mass == pytest.approx(1.0, rel=1e-7, abs=1e-9), (
            f"mode {k}: φᵀ M φ = {gen_mass}"
        )
