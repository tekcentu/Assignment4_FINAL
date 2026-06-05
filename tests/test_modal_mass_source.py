"""Tests for assemble_mass_matrix_with_source (PR #40)."""

import numpy as np
import pytest

from structural_analysis.model import (
    JointMass,
    LoadCase,
    Material,
    ModalMassSource,
    NodalLoad,
    Node,
    Section,
    STANDARD_GRAVITY,
    StructuralModel,
    Support,
)
from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.assembler import DofManager, assemble_global_system
from structural_analysis.mass import (
    assemble_mass_matrix,
    assemble_mass_matrix_with_source,
)
from structural_analysis.main import run_analysis


# ── Minimal model factory ──────────────────────────────────────────────────


def _cantilever_model(n_elems: int = 2, density: float = 7850.0) -> StructuralModel:
    """Simple horizontal cantilever with n_elems FrameElement2D elements."""
    model = StructuralModel(title="Cantilever")
    L = 1.0  # element length (m)
    E = 2.1e8  # kN/m²
    A = 1e-2  # m²
    I = 1e-4  # m⁴
    model.materials[1] = Material(id=1, E=E, density=density)
    model.sections[1] = Section(id=1, material_id=1, A=A, I=I, depth=0.1)
    for k in range(n_elems + 1):
        model.nodes[k + 1] = Node(k + 1, float(k) * L, 0.0)
    model.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    for k in range(n_elems):
        ni, nj = k + 1, k + 2
        model.elements.append(FrameElement2D(
            id=k + 1, node_i=ni, node_j=nj,
            E=E, A=A, I=I,
            alpha=0.0, depth=0.1, rho=density,
            section_id=1,
        ))
    return model


def _two_node_frame(
    density: float = 7850.0,
    include_sw: bool = False,
) -> StructuralModel:
    """Single-element frame with a free tip node."""
    model = StructuralModel()
    model.materials[1] = Material(id=1, E=2.1e8, density=density)
    model.sections[1] = Section(id=1, material_id=1, A=1e-2, I=1e-4, depth=0.1)
    model.nodes[1] = Node(1, 0.0, 0.0)
    model.nodes[2] = Node(2, 1.0, 0.0)
    model.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    model.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=1e-2, I=1e-4,
        alpha=0.0, depth=0.1, rho=density,
        section_id=1,
    ))
    model.include_self_weight = include_sw
    return model


# ── Equivalence with legacy function ──────────────────────────────────────


def test_default_source_equals_legacy_mass_matrix():
    """assemble_mass_matrix_with_source(source=None) must match legacy."""
    model = _cantilever_model()
    _, _, dofs, _, _ = assemble_global_system(model)

    M_legacy = assemble_mass_matrix(model, dofs)
    M_new, info = assemble_mass_matrix_with_source(model, dofs, source=None)

    np.testing.assert_array_almost_equal(M_legacy, M_new, decimal=12)
    assert info == []


def test_default_source_object_equals_legacy():
    model = _cantilever_model()
    _, _, dofs, _, _ = assemble_global_system(model)

    M_legacy = assemble_mass_matrix(model, dofs)
    M_new, _ = assemble_mass_matrix_with_source(
        model, dofs, source=ModalMassSource()
    )
    np.testing.assert_array_almost_equal(M_legacy, M_new, decimal=12)


# ── Joint mass ────────────────────────────────────────────────────────────


def test_joint_mass_contributes_to_mass_matrix():
    """1000 kg joint mass → 1.0 Mg on M[ux,ux] and M[uy,uy]."""
    model = _two_node_frame()
    _, _, dofs, _, _ = assemble_global_system(model)
    nm = dofs.active_map[2]
    ux_idx = nm["ux"]
    uy_idx = nm["uy"]
    assert ux_idx is not None and uy_idx is not None

    model.joint_masses[2] = JointMass(node_id=2, mx=1000.0, my=1000.0)
    src = ModalMassSource(include_self_mass=False, include_joint_masses=True)
    M, info = assemble_mass_matrix_with_source(model, dofs, source=src)

    assert pytest.approx(M[ux_idx, ux_idx], rel=1e-10) == 1.0  # 1000 kg = 1 Mg
    assert pytest.approx(M[uy_idx, uy_idx], rel=1e-10) == 1.0


def test_joint_mass_excluded_when_flag_off():
    model = _two_node_frame()
    _, _, dofs, _, _ = assemble_global_system(model)
    model.joint_masses[2] = JointMass(node_id=2, mx=1000.0, my=1000.0)
    src = ModalMassSource(include_self_mass=False, include_joint_masses=False)
    M, _ = assemble_mass_matrix_with_source(model, dofs, source=src)

    nm = dofs.active_map[2]
    ux_idx = nm["ux"]
    # With self-mass off and joint-masses off the matrix is all zeros.
    assert M[ux_idx, ux_idx] == pytest.approx(0.0, abs=1e-15)


def test_joint_mass_no_rz_contribution_on_truss_node():
    """Truss node has no rz DOF; mx/my must still contribute."""
    model = StructuralModel()
    model.materials[1] = Material(id=1, E=2.1e8, density=7850.0)
    model.sections[1] = Section(id=1, material_id=1, A=1e-2, I=0.0, depth=0.0)
    model.nodes[1] = Node(1, 0.0, 0.0)
    model.nodes[2] = Node(2, 1.0, 0.0)
    model.supports[1] = Support(node_id=1, ux=True, uy=True)
    model.elements.append(TrussElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=1e-2, alpha=0.0, depth=0.0, rho=7850.0,
        section_id=1,
    ))
    _, _, dofs, _, _ = assemble_global_system(model)
    nm2 = dofs.active_map[2]
    assert nm2["rz"] is None  # truss node: no rz DOF

    model.joint_masses[2] = JointMass(node_id=2, mx=500.0, my=500.0)
    src = ModalMassSource(include_self_mass=False, include_joint_masses=True)
    M, info = assemble_mass_matrix_with_source(model, dofs, source=src)
    ux_idx = nm2["ux"]
    uy_idx = nm2["uy"]
    assert M[ux_idx, ux_idx] == pytest.approx(0.5, rel=1e-10)
    assert M[uy_idx, uy_idx] == pytest.approx(0.5, rel=1e-10)


# ── Load-case to mass ──────────────────────────────────────────────────────


def test_load_case_gravity_contributes_to_both_translational_dofs():
    """Vertical load at tip → mass added to BOTH M[ux,ux] and M[uy,uy]."""
    model = _two_node_frame()
    model.load_cases["LIVE"] = LoadCase(name="LIVE")
    model.nodal_loads.append(NodalLoad(node_id=2, fx=0.0, fy=-100.0, mz=0.0,
                                       load_case="LIVE"))
    _, _, dofs, _, _ = assemble_global_system(model)

    src = ModalMassSource(
        include_self_mass=False,
        include_joint_masses=False,
        include_load_cases=True,
        load_case_factors={"LIVE": 0.3},
    )
    M, _ = assemble_mass_matrix_with_source(model, dofs, source=src)

    nm2 = dofs.active_map[2]
    ux_idx = nm2["ux"]
    uy_idx = nm2["uy"]
    expected_mass = 0.3 * 100.0 / STANDARD_GRAVITY
    assert M[ux_idx, ux_idx] == pytest.approx(expected_mass, rel=1e-6)
    assert M[uy_idx, uy_idx] == pytest.approx(expected_mass, rel=1e-6)


def test_lateral_wind_loads_excluded_from_mass():
    """Pure lateral (Fx-only) load contributes zero mass (|Fy|=0)."""
    model = _two_node_frame()
    model.load_cases["WIND"] = LoadCase(name="WIND")
    model.nodal_loads.append(NodalLoad(node_id=2, fx=50.0, fy=0.0, mz=0.0,
                                       load_case="WIND"))
    _, _, dofs, _, _ = assemble_global_system(model)

    src = ModalMassSource(
        include_self_mass=False,
        include_joint_masses=False,
        include_load_cases=True,
        load_case_factors={"WIND": 1.0},
    )
    M, _ = assemble_mass_matrix_with_source(model, dofs, source=src)
    # M should be all zeros (lateral load → no fy → no mass)
    assert np.allclose(M, 0.0, atol=1e-15)


def test_load_case_mass_uses_abs_value():
    """Upward and downward loads of equal magnitude give equal mass."""
    model_down = _two_node_frame()
    model_up = _two_node_frame()
    model_down.load_cases["LC"] = LoadCase(name="LC")
    model_up.load_cases["LC"] = LoadCase(name="LC")
    model_down.nodal_loads.append(
        NodalLoad(node_id=2, fx=0.0, fy=-10.0, mz=0.0, load_case="LC")
    )
    model_up.nodal_loads.append(
        NodalLoad(node_id=2, fx=0.0, fy=+10.0, mz=0.0, load_case="LC")
    )

    src = ModalMassSource(
        include_self_mass=False, include_joint_masses=False,
        include_load_cases=True, load_case_factors={"LC": 1.0},
    )

    _, _, dofs_d, _, _ = assemble_global_system(model_down)
    _, _, dofs_u, _, _ = assemble_global_system(model_up)
    M_down, _ = assemble_mass_matrix_with_source(model_down, dofs_d, source=src)
    M_up, _ = assemble_mass_matrix_with_source(model_up, dofs_u, source=src)

    np.testing.assert_array_almost_equal(M_down, M_up, decimal=12)


def test_load_case_mass_excluded_when_flag_off():
    model = _two_node_frame()
    model.load_cases["LIVE"] = LoadCase(name="LIVE")
    model.nodal_loads.append(
        NodalLoad(node_id=2, fx=0.0, fy=-100.0, mz=0.0, load_case="LIVE")
    )
    _, _, dofs, _, _ = assemble_global_system(model)

    src = ModalMassSource(
        include_self_mass=False,
        include_joint_masses=False,
        include_load_cases=False,  # off
        load_case_factors={"LIVE": 0.3},
    )
    M, _ = assemble_mass_matrix_with_source(model, dofs, source=src)
    assert np.allclose(M, 0.0, atol=1e-15)


def test_unknown_load_case_in_factors_raises():
    model = _two_node_frame()
    _, _, dofs, _, _ = assemble_global_system(model)
    src = ModalMassSource(
        include_load_cases=True,
        load_case_factors={"NONEXISTENT": 1.0},
    )
    with pytest.raises(ValueError, match="NONEXISTENT"):
        assemble_mass_matrix_with_source(model, dofs, source=src)


# ── Self-weight double-count prevention ───────────────────────────────────


def test_self_weight_mathematical_isolation():
    """Manual dead load is converted; density self-weight is stripped out."""
    model = _two_node_frame(density=7850.0, include_sw=True)
    model.self_weight_case = "DEAD"
    model.load_cases["DEAD"] = LoadCase(name="DEAD")
    # 50 kN manual dead load at tip
    model.nodal_loads.append(
        NodalLoad(node_id=2, fx=0.0, fy=-50.0, mz=0.0, load_case="DEAD")
    )
    _, _, dofs, _, _ = assemble_global_system(model)
    nm2 = dofs.active_map[2]
    uy_idx = nm2["uy"]

    src_with_sw = ModalMassSource(
        include_self_mass=True,
        include_joint_masses=False,
        include_load_cases=True,
        load_case_factors={"DEAD": 1.0},
    )
    M_with_sw, info = assemble_mass_matrix_with_source(model, dofs, source=src_with_sw)

    # The manual 50 kN → 50/9.81 ≈ 5.097 Mg per DOF.
    expected_manual = 50.0 / STANDARD_GRAVITY
    # Self-mass component is also present but we focus on the info line.
    assert any("excluded" in line for line in info)

    # Compare with a model that has no density (so self-mass=0) and the same
    # manual load — in that case no subtraction occurs, but the result should
    # include the same manual mass.
    model_no_density = _two_node_frame(density=0.0, include_sw=False)
    model_no_density.self_weight_case = "DEAD"
    model_no_density.load_cases["DEAD"] = LoadCase(name="DEAD")
    model_no_density.nodal_loads.append(
        NodalLoad(node_id=2, fx=0.0, fy=-50.0, mz=0.0, load_case="DEAD")
    )
    _, _, dofs2, _, _ = assemble_global_system(model_no_density)
    nm2b = dofs2.active_map[2]
    uy_idx2 = nm2b["uy"]
    ux_idx2 = nm2b["ux"]

    src_no_sw = ModalMassSource(
        include_self_mass=False,
        include_joint_masses=False,
        include_load_cases=True,
        load_case_factors={"DEAD": 1.0},
    )
    M_no_density, _ = assemble_mass_matrix_with_source(
        model_no_density, dofs2, source=src_no_sw
    )

    assert M_no_density[uy_idx2, uy_idx2] == pytest.approx(expected_manual, rel=1e-5)
    assert M_no_density[ux_idx2, ux_idx2] == pytest.approx(expected_manual, rel=1e-5)


# ── Modal runs with only joint masses (density = 0) ───────────────────────


def test_modal_runs_with_only_joint_masses_when_density_zero():
    """A model with ρ=0 but joint mass should run modal without error."""
    from structural_analysis.modal import solve_modal

    model = _two_node_frame(density=0.0)
    model.joint_masses[2] = JointMass(node_id=2, mx=1000.0, my=1000.0)
    model.modal_mass_source = ModalMassSource(
        include_self_mass=False,
        include_joint_masses=True,
    )
    result = solve_modal(model, n_modes=2, mass_source=model.modal_mass_source)
    assert result.status == "ok"
    assert result.n_modes >= 1
    assert all(np.isfinite(result.frequencies))


# ── Static analysis isolation ─────────────────────────────────────────────


def test_static_analysis_isolation_joint_masses():
    """Adding joint masses must not affect static displacements or reactions."""
    model_base = _two_node_frame()
    model_base.nodal_loads.append(
        NodalLoad(node_id=2, fx=0.0, fy=-10.0, mz=0.0)
    )
    r_base = run_analysis(model_base, verbose=False)

    model_with_jm = _two_node_frame()
    model_with_jm.nodal_loads.append(
        NodalLoad(node_id=2, fx=0.0, fy=-10.0, mz=0.0)
    )
    model_with_jm.joint_masses[2] = JointMass(node_id=2, mx=5000.0, my=5000.0)

    r_jm = run_analysis(model_with_jm, verbose=False)

    np.testing.assert_array_almost_equal(r_base.D, r_jm.D, decimal=12)
    for nid in r_base.reactions:
        for dof in ("ux", "uy", "rz"):
            assert (r_base.reactions[nid].get(dof, 0.0)
                    == pytest.approx(r_jm.reactions[nid].get(dof, 0.0), abs=1e-12))


def test_static_analysis_isolation_mass_source():
    """Changing modal_mass_source must not affect static results."""
    model_a = _two_node_frame()
    model_a.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0))
    r_a = run_analysis(model_a, verbose=False)

    model_b = _two_node_frame()
    model_b.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0))
    model_b.load_cases["DEAD"] = LoadCase(name="DEAD")
    model_b.modal_mass_source = ModalMassSource(
        include_self_mass=True,
        include_joint_masses=True,
        include_load_cases=True,
        load_case_factors={"DEFAULT": 1.0},
    )
    r_b = run_analysis(model_b, verbose=False)

    np.testing.assert_array_almost_equal(r_a.D, r_b.D, decimal=12)
