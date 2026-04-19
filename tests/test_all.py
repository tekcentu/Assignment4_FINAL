"""
CE 4011 Assignment 3 — Comprehensive Test Suite

Test categories:
    UT  = Unit tests          (element-level)
    IT  = Interface tests     (assembly-level)
    RT  = Regression tests    (full analysis vs closed-form / A2 reference)

Run:  python -m pytest tests/test_all.py -v
"""

import sys, os
import numpy as np
import pytest
from numpy.testing import assert_allclose

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad,
    TrussTemperatureLoad, FrameTemperatureLoad,
)
from structural_analysis.element import (
    FrameElement2D, TrussElement2D, _length_cos_sin, _rotation_matrix_6x6,
)
from structural_analysis.assembler import (
    DofManager, assemble_global_system, validate_model,
    _connectivity_components,
)
from structural_analysis.solver import solve_system
from structural_analysis.postprocessor import compute_member_forces, compute_reactions
from structural_analysis.main import run_analysis


TOL = 1e-6


# ================================================================
#  Helpers — standard models
# ================================================================

def _portal_frame() -> StructuralModel:
    """4-node portal frame from Assignment 2 course notes."""
    m = StructuralModel(title="Portal Frame")
    m.nodes = {1: Node(1,0,0), 2: Node(2,0,3), 3: Node(3,4,3), 4: Node(4,4,0)}
    m.materials = {1: Material(1, 200000.0, 0.02, 0.08),
                   2: Material(2, 200000.0, 0.01, 0.01)}
    m.elements = [
        FrameElement2D(id=1, node_i=1, node_j=2, E=200000, A=0.02, I=0.08),
        FrameElement2D(id=2, node_i=2, node_j=3, E=200000, A=0.02, I=0.08),
        FrameElement2D(id=3, node_i=4, node_j=3, E=200000, A=0.02, I=0.08),
        FrameElement2D(id=4, node_i=1, node_j=3, E=200000, A=0.01, I=0.01),
    ]
    m.supports = {1: Support(1, True, True, False), 4: Support(4, False, True, False)}
    m.nodal_loads = [NodalLoad(2, 10, -10, 0), NodalLoad(3, 10, -10, 0)]
    return m


def _cantilever(P: float = 12.0, L: float = 4.0) -> StructuralModel:
    """Fixed-free cantilever beam with tip load P."""
    m = StructuralModel(title="Cantilever")
    m.nodes = {1: Node(1,0,0), 2: Node(2,L,0)}
    m.materials = {1: Material(1, 200000.0, 0.02, 0.08)}
    m.elements = [FrameElement2D(id=1, node_i=1, node_j=2, E=200000, A=0.02, I=0.08)]
    m.supports = {1: Support(1, True, True, True)}
    m.nodal_loads = [NodalLoad(2, 0, -P, 0)]
    return m


NODES_H6 = {1: Node(1, 0, 0), 2: Node(2, 6, 0)}  # horizontal L=6


# ================================================================
#  UT-1 — Element geometry
# ================================================================
class TestUTGeometry:
    def test_horizontal(self):
        L, c, s = _length_cos_sin(Node(1,0,0), Node(2,4,0))
        assert abs(L - 4) < TOL and abs(c - 1) < TOL and abs(s) < TOL

    def test_vertical(self):
        L, c, s = _length_cos_sin(Node(1,0,0), Node(2,0,3))
        assert abs(L - 3) < TOL and abs(c) < TOL and abs(s - 1) < TOL

    def test_inclined_345(self):
        L, c, s = _length_cos_sin(Node(1,0,0), Node(2,4,3))
        assert abs(L - 5) < TOL and abs(c - 0.8) < TOL and abs(s - 0.6) < TOL

    def test_zero_length_raises(self):
        with pytest.raises(ValueError):
            _length_cos_sin(Node(1,2,3), Node(2,2,3))


# ================================================================
#  UT-2 — Frame stiffness
# ================================================================
class TestUTFrameStiffness:
    def test_symmetry(self):
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08)
        k = e.raw_local_stiffness(NODES_H6)
        assert_allclose(k, k.T, atol=1e-10)

    def test_diagonal_positive(self):
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08)
        k = e.raw_local_stiffness(NODES_H6)
        assert all(k[i,i] >= 0 for i in range(6))

    def test_known_values_A2_element1(self):
        """Verify against course notes: L=3, A=0.02, I=0.08, E=200000."""
        nodes = {1: Node(1,0,0), 2: Node(2,0,3)}
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08)
        k = e.raw_local_stiffness(nodes)
        assert abs(k[0,0] - 1333.33) < 0.01
        assert abs(k[1,1] - 7111.11) < 0.01
        assert abs(k[2,2] - 21333.33) < 0.01

    def test_rigid_body_modes(self):
        """Frame k must have exactly 3 zero eigenvalues (rigid-body)."""
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08)
        eigvals = np.sort(np.linalg.eigvalsh(e.raw_local_stiffness(NODES_H6)))
        assert abs(eigvals[0]) < TOL
        assert abs(eigvals[1]) < TOL
        assert abs(eigvals[2]) < TOL
        assert eigvals[3] > 0

    def test_closed_form_match(self):
        """Full k matrix matches analytical formula."""
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08)
        k = e.raw_local_stiffness(NODES_H6)
        L = 6.0; EA_L = 200000*0.02/L; EI = 200000*0.08
        expected = np.array([
            [EA_L,0,0,-EA_L,0,0],
            [0,12*EI/L**3,6*EI/L**2,0,-12*EI/L**3,6*EI/L**2],
            [0,6*EI/L**2,4*EI/L,0,-6*EI/L**2,2*EI/L],
            [-EA_L,0,0,EA_L,0,0],
            [0,-12*EI/L**3,-6*EI/L**2,0,12*EI/L**3,-6*EI/L**2],
            [0,6*EI/L**2,2*EI/L,0,-6*EI/L**2,4*EI/L],
        ])
        assert_allclose(k, expected, atol=1e-10)


# ================================================================
#  UT-3 — Truss stiffness
# ================================================================
class TestUTTrussStiffness:
    def test_only_axial(self):
        e = TrussElement2D(1, 1, 2, E=200000, A=0.01)
        k = e.raw_local_stiffness(NODES_H6)
        EA_L = 200000 * 0.01 / 6.0
        assert abs(k[0,0] - EA_L) < TOL
        assert abs(k[3,3] - EA_L) < TOL
        for i in [1,2,4,5]:
            for j in range(6):
                assert abs(k[i,j]) < TOL

    def test_symmetry(self):
        e = TrussElement2D(1, 1, 2, E=200000, A=0.01)
        k = e.raw_local_stiffness(NODES_H6)
        assert_allclose(k, k.T, atol=1e-10)

    def test_assembly_indices_suppress_rotation(self):
        e = TrussElement2D(1, 1, 2, E=200000, A=0.01)
        idx = e.assembly_local_indices()
        assert idx == [0, 1, None, 3, 4, None]


# ================================================================
#  UT-4 — Moment release (Schur complement condensation)
# ================================================================
class TestUTMomentRelease:
    def test_release_start_zeros_dof2(self):
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08, release_i=True)
        k, _ = e.assembled_local_stiffness_and_load(NODES_H6)
        assert_allclose(k[2, :], 0, atol=TOL)
        assert_allclose(k[:, 2], 0, atol=TOL)

    def test_release_end_zeros_dof5(self):
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08, release_j=True)
        k, _ = e.assembled_local_stiffness_and_load(NODES_H6)
        assert_allclose(k[5, :], 0, atol=TOL)
        assert_allclose(k[:, 5], 0, atol=TOL)

    def test_release_both_axial_only(self):
        """Both released → only axial stiffness remains (like truss)."""
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08,
                           release_i=True, release_j=True)
        k, _ = e.assembled_local_stiffness_and_load(NODES_H6)
        EA_L = 200000 * 0.02 / 6.0
        assert abs(k[0,0] - EA_L) < TOL
        assert abs(k[3,3] - EA_L) < TOL

    def test_symmetry_preserved(self):
        for ri, rj in [(True,False), (False,True), (True,True)]:
            e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08,
                               release_i=ri, release_j=rj)
            k, _ = e.assembled_local_stiffness_and_load(NODES_H6)
            assert_allclose(k, k.T, atol=1e-10)

    def test_schur_complement_independent_verify(self):
        """Verify condensation against manual partition formula."""
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08,
                           release_i=True, release_j=True,
                           member_loads=[UniformDistributedLoad(wy=-10)])
        k_full = e.raw_local_stiffness(NODES_H6)
        p_full = e.local_consistent_load(NODES_H6)

        released = [2, 5]
        retained = [0, 1, 3, 4]
        kaa = k_full[np.ix_(retained, retained)]
        kab = k_full[np.ix_(retained, released)]
        kba = k_full[np.ix_(released, retained)]
        kbb = k_full[np.ix_(released, released)]
        pa, pb = p_full[retained], p_full[released]

        expected_k = np.zeros((6,6))
        expected_p = np.zeros(6)
        expected_k[np.ix_(retained, retained)] = kaa - kab @ np.linalg.inv(kbb) @ kba
        expected_p[retained] = pa - kab @ np.linalg.inv(kbb) @ pb

        k_c, p_c = e.assembled_local_stiffness_and_load(NODES_H6)
        assert_allclose(k_c, expected_k, atol=1e-10)
        assert_allclose(p_c, expected_p, atol=1e-10)


# ================================================================
#  UT-5 — Fixed-end forces (consistent load vector)
# ================================================================
class TestUTConsistentLoad:
    def test_udl_equilibrium(self):
        """V_i + V_j = w·L for UDL."""
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08,
                           member_loads=[UniformDistributedLoad(wy=-10)])
        p = e.local_consistent_load(NODES_H6)
        L = 6.0
        assert abs(p[1] + p[4] - (-10)*L) < TOL

    def test_udl_moment_equilibrium(self):
        """Moment equilibrium about start: M_i + M_j + V_j·L − w·L²/2 = 0."""
        L = 6.0; w = -10.0
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08,
                           member_loads=[UniformDistributedLoad(wy=w)])
        p = e.local_consistent_load(NODES_H6)
        residual = p[2] + p[5] + p[4]*L - w*L**2/2
        assert abs(residual) < TOL

    def test_midspan_point_load_symmetric(self):
        """Midspan point load: shears equal, moments antisymmetric."""
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08,
                           member_loads=[PointLoad(py=-20, a=3.0)])
        p = e.local_consistent_load(NODES_H6)
        assert abs(p[1] - p[4]) < TOL      # symmetric shears
        assert abs(p[2] + p[5]) < TOL       # antisymmetric moments

    def test_combined_udl_and_point_load(self):
        """UDL + midspan point load (from ChatGPT Codex regression)."""
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08,
                           member_loads=[UniformDistributedLoad(wy=-10),
                                         PointLoad(py=-12, a=3.0)])
        p = e.local_consistent_load(NODES_H6)
        expected = np.array([0, -36, -39, 0, -36, 39])
        assert_allclose(p, expected, atol=1e-10)

    def test_udl_on_released_element_condensed(self):
        """UDL on both-released element: condensed p should give wL/2 shears."""
        L = 6.0; w = -10.0
        e = FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08,
                           release_i=True, release_j=True,
                           member_loads=[UniformDistributedLoad(wy=w)])
        _, p_c = e.assembled_local_stiffness_and_load(NODES_H6)
        # With both moments released, should act like simply-supported
        assert abs(p_c[1] - w*L/2) < TOL
        assert abs(p_c[4] - w*L/2) < TOL
        assert abs(p_c[2]) < TOL
        assert abs(p_c[5]) < TOL


# ================================================================
#  IT-1 — Equation numbering and DofManager
# ================================================================
class TestITEquationNumbering:
    def test_portal_frame_E_matrix(self):
        """E matrix matches A2 course notes."""
        model = _portal_frame()
        dofs = DofManager.from_model(model)
        E = dofs.e_matrix_for_display(model)
        assert E[1] == [0, 0, 1]   # pin
        assert E[2] == [2, 3, 4]   # free
        assert E[3] == [5, 6, 7]   # free
        assert E[4] == [8, 0, 9]   # roller

    def test_truss_node_no_rz(self):
        """Pure truss node should NOT get an Rz DOF."""
        m = StructuralModel(title="Truss")
        m.nodes = {1: Node(1,0,0), 2: Node(2,3,4), 3: Node(3,6,0)}
        m.elements = [
            TrussElement2D(1, 1, 2, E=200000, A=0.01),
            TrussElement2D(2, 2, 3, E=200000, A=0.01),
        ]
        m.supports = {1: Support(1,True,True), 3: Support(3,True,True)}
        dofs = DofManager.from_model(m)
        # Node 2 should have ux, uy but NO rz
        assert dofs.active_map[2]["rz"] is None

    def test_mixed_frame_truss_node_gets_rz(self):
        """A node connected to both frame and truss should have Rz."""
        m = StructuralModel(title="Mixed")
        m.nodes = {1: Node(1,0,0), 2: Node(2,4,0), 3: Node(3,4,3)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08),
            TrussElement2D(2, 2, 3, E=200000, A=0.01),
        ]
        m.supports = {1: Support(1,True,True,True), 3: Support(3,True,True)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        dofs = DofManager.from_model(m)
        # Node 2 connects to frame elem → should have Rz
        assert dofs.active_map[2]["rz"] is not None
        # Node 3 connects only to truss → no Rz
        assert dofs.active_map[3]["rz"] is None


# ================================================================
#  IT-2 — Assembly
# ================================================================
class TestITAssembly:
    def test_K_symmetry(self):
        K, F, dofs, w, _ = assemble_global_system(_portal_frame())
        assert_allclose(K, K.T, atol=1e-10)

    def test_K_size(self):
        K, F, dofs, w, _ = assemble_global_system(_portal_frame())
        assert K.shape == (dofs.n_total, dofs.n_total)

    def test_load_vector_values(self):
        K, F, dofs, w, _ = assemble_global_system(_portal_frame())
        # Node 2 Tx = +10 kN
        idx_2_ux = dofs.index(2, "ux")
        assert abs(F[idx_2_ux] - 10.0) < TOL

    def test_released_beam_load_assembly(self):
        """Both-released beam + UDL: K should have only axial stiffness."""
        m = StructuralModel(title="Released beam")
        m.nodes = {1: Node(1,0,0), 2: Node(2,6,0)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08,
                                      release_i=True, release_j=True,
                                      member_loads=[UniformDistributedLoad(wy=-10)])]
        m.supports = {1: Support(1,True,True), 2: Support(2,False,True)}
        K, F, dofs, w, _ = assemble_global_system(m)
        # With both rotations released, no Rz DOFs should exist
        assert dofs.active_map[1]["rz"] is None
        assert dofs.active_map[2]["rz"] is None


# ================================================================
#  IT-3 — Validation
# ================================================================
class TestITValidation:
    def test_disconnected_component_raises(self):
        """Floating sub-structure should raise ValueError."""
        m = StructuralModel(title="Disconnected")
        m.nodes = {1: Node(1,0,0), 2: Node(2,4,0), 3: Node(3,10,0), 4: Node(4,14,0)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08),
            FrameElement2D(2, 3, 4, E=200000, A=0.02, I=0.08),
        ]
        m.supports = {1: Support(1,True,True,True)}  # only component 1 supported
        with pytest.raises(ValueError, match="no supports"):
            assemble_global_system(m)

    def test_isolated_node_raises(self):
        m = StructuralModel(title="Isolated")
        m.nodes = {1: Node(1,0,0), 2: Node(2,4,0), 3: Node(3,8,0)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08)]
        m.supports = {1: Support(1,True,True,True)}
        with pytest.raises(ValueError, match="Isolated"):
            assemble_global_system(m)


# ================================================================
#  RT-1 — Portal frame (A2 reference)
# ================================================================
class TestRTPortalFrame:
    def test_displacements_match_course_notes(self):
        """Displacements within 0.5% of A2 course reference (3 sig figs)."""
        r = run_analysis(_portal_frame(), verbose=False)
        assert r.status == "ok"
        D = r.D
        dofs_map = r.E_map
        # Reference from A2 report (independently verified against SAP2000)
        ref = {
            (1, "rz"): -1.150e-2,
            (2, "ux"): +3.130e-2,
            (2, "uy"): +5.450e-4,
            (2, "rz"): -8.040e-3,
            (3, "ux"): +3.580e-2,
            (3, "uy"): -1.870e-2,
            (3, "rz"): -3.400e-3,
            (4, "ux"): +2.560e-2,
            (4, "rz"): -3.400e-3,
        }
        for (nid, dof), expected in ref.items():
            idx = dofs_map[nid][dof]
            if idx is not None and abs(expected) > 1e-10:
                actual = D[idx]
                assert abs((actual - expected)/expected) < 0.005, \
                    f"Node {nid} {dof}: {actual:.6e} vs ref {expected:.3e}"

    def test_residual(self):
        r = run_analysis(_portal_frame(), verbose=False)
        assert r.residual < 1e-8

    def test_reactions_equilibrium(self):
        r = run_analysis(_portal_frame(), verbose=False)
        # Pin at node 1: Rx ≈ −20, Ry ≈ −5
        assert abs(r.reactions[1]["ux"] - (-20.0)) < 0.1
        assert abs(r.reactions[1]["uy"] - (-5.0)) < 0.1


# ================================================================
#  RT-2 — Cantilever (PL³/3EI)
# ================================================================
class TestRTCantilever:
    def test_tip_deflection(self):
        """Cantilever tip deflection = PL³/3EI (closed-form)."""
        P, L, E, I = 12.0, 4.0, 200000.0, 0.08
        r = run_analysis(_cantilever(P, L), verbose=False)
        assert r.status == "ok"
        idx_uy = r.E_map[2]["uy"]
        expected = -P * L**3 / (3 * E * I)
        assert_allclose(r.D[idx_uy], expected, atol=1e-9)

    def test_tip_rotation(self):
        P, L, E, I = 12.0, 4.0, 200000.0, 0.08
        r = run_analysis(_cantilever(P, L), verbose=False)
        idx_rz = r.E_map[2]["rz"]
        expected = -P * L**2 / (2 * E * I)
        assert_allclose(r.D[idx_rz], expected, atol=1e-9)

    def test_reactions(self):
        r = run_analysis(_cantilever(12.0, 4.0), verbose=False)
        assert_allclose(r.reactions[1]["uy"], 12.0, atol=1e-10)
        assert_allclose(r.reactions[1]["rz"], 48.0, atol=1e-10)


# ================================================================
#  RT-3 — Simply-supported beam (PL³/48EI)
# ================================================================
class TestRTSimplySupportedBeam:
    def test_midspan_deflection(self):
        m = StructuralModel(title="SS Beam")
        m.nodes = {1: Node(1,0,0), 2: Node(2,3,0), 3: Node(3,6,0)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08),
            FrameElement2D(2, 2, 3, E=200000, A=0.02, I=0.08),
        ]
        m.supports = {1: Support(1,True,True,False), 3: Support(3,False,True,False)}
        m.nodal_loads = [NodalLoad(2, 0, -100, 0)]
        r = run_analysis(m, verbose=False)
        assert r.status == "ok"
        P, L, E, I = 100, 6, 200000, 0.08
        expected = P * L**3 / (48 * E * I)
        idx = r.E_map[2]["uy"]
        assert abs(abs(r.D[idx]) - expected) / expected < 0.001


# ================================================================
#  RT-4 — Two-bar truss (analytical)
# ================================================================
class TestRTTwoBarTruss:
    def test_apex_displacement_analytical(self):
        """Symmetric 2-bar truss: δ = PL/(2EA·sin²α)."""
        m = StructuralModel(title="2-Bar Truss")
        m.nodes = {1: Node(1,0,0), 2: Node(2,3,4), 3: Node(3,6,0)}
        m.materials = {1: Material(1,200000,0.01,0)}
        m.elements = [
            TrussElement2D(1, 1, 2, E=200000, A=0.01),
            TrussElement2D(2, 2, 3, E=200000, A=0.01),
        ]
        m.supports = {1: Support(1,True,True), 3: Support(3,True,True)}
        m.nodal_loads = [NodalLoad(2, 0, -12, 0)]
        r = run_analysis(m, verbose=False)
        assert r.status == "ok"
        P, E, A, L = 12.0, 200000.0, 0.01, 5.0
        sin_a = 4.0/5.0
        expected_uy = -P * L / (2 * E * A * sin_a**2)
        idx = r.E_map[2]["uy"]
        assert_allclose(r.D[idx], expected_uy, atol=1e-9)


# ================================================================
#  RT-5 — Mechanism detection
# ================================================================
class TestRTMechanism:
    def test_rollers_only_detected(self):
        """Beam with only roller supports → mechanism."""
        m = StructuralModel(title="Mechanism")
        m.nodes = {1: Node(1,0,0), 2: Node(2,4,0), 3: Node(3,8,0)}
        m.materials = {1: Material(1,200000,0.02,0.0004)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.0004),
            FrameElement2D(2, 2, 3, E=200000, A=0.02, I=0.0004),
        ]
        m.supports = {1: Support(1,False,True), 3: Support(3,False,True)}
        m.nodal_loads = [NodalLoad(2, 10, -20, 0)]
        r = run_analysis(m, verbose=False)
        assert r.status == "error"
        # Should identify UX DOFs in mechanism
        assert any("UX" in w for w in r.warnings)

    def test_portal_rollers_mechanism(self):
        """Portal frame on rollers — sway mechanism."""
        m = StructuralModel(title="Portal Rollers")
        m.nodes = {1: Node(1,0,0), 2: Node(2,0,4), 3: Node(3,6,4), 4: Node(4,6,0)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08),
            FrameElement2D(2, 2, 3, E=200000, A=0.02, I=0.08),
            FrameElement2D(3, 3, 4, E=200000, A=0.02, I=0.08),
        ]
        m.supports = {1: Support(1,False,True), 4: Support(4,False,True)}
        m.nodal_loads = [NodalLoad(2, 10, 0, 0)]
        r = run_analysis(m, verbose=False)
        assert r.status == "error"


# ================================================================
#  RT-6 — Internal hinge (moment = 0 at released end)
# ================================================================
class TestRTInternalHinge:
    def test_hinge_moment_zero(self):
        """Moment at released end must be zero."""
        m = StructuralModel(title="Hinge")
        m.nodes = {1: Node(1,0,0), 2: Node(2,0,3), 3: Node(3,4,3), 4: Node(4,4,0)}
        m.materials = {1: Material(1,200000,0.02,0.0004)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.0004),
            FrameElement2D(2, 2, 3, E=200000, A=0.02, I=0.0004, release_j=True),
            FrameElement2D(3, 4, 3, E=200000, A=0.02, I=0.0004),
        ]
        m.supports = {1: Support(1,True,True,True), 4: Support(4,True,True,False)}
        m.nodal_loads = [NodalLoad(2, 10, -10, 0)]
        r = run_analysis(m, verbose=False)
        assert r.status == "ok"
        # Element 2 M_j (released end) should be ~0
        assert abs(r.member_results[2]["f_local"][5]) < 0.01

    def test_hinge_with_member_load_moment_zero(self):
        """UDL on released element: M at released end still zero."""
        m = StructuralModel(title="Hinge+UDL")
        m.nodes = {1: Node(1,0,0), 2: Node(2,6,0)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08,
                           release_j=True,
                           member_loads=[UniformDistributedLoad(wy=-10)]),
        ]
        m.supports = {1: Support(1,True,True,True), 2: Support(2,False,True,False)}
        r = run_analysis(m, verbose=False)
        assert r.status == "ok"
        # M_j should be ~0
        assert abs(r.member_results[1]["f_local"][5]) < 0.01

    def test_propped_cantilever_asymmetric(self):
        """Propped cantilever (asymmetric) — catches sign bugs."""
        L, P = 6.0, 20.0
        m = StructuralModel(title="Propped cantilever")
        m.nodes = {1: Node(1,0,0), 2: Node(2,L,0)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08,
                           release_j=True),  # hinge at end = propped cantilever
        ]
        m.supports = {1: Support(1,True,True,True), 2: Support(2,False,True,False)}
        m.nodal_loads = [NodalLoad(2, 0, -P, 0)]
        r = run_analysis(m, verbose=False)
        assert r.status == "ok"
        # Reactions: R1y + R2y = P
        total_ry = r.reactions[1].get("uy",0) + r.reactions[2].get("uy",0)
        assert abs(total_ry - P) < 0.01


# ================================================================
#  RT-7 — Mixed frame + truss
# ================================================================
class TestRTMixedFrameTruss:
    def test_frame_truss_hybrid(self):
        m = StructuralModel(title="Hybrid")
        m.nodes = {1: Node(1,0,0), 2: Node(2,0,4), 3: Node(3,6,4), 4: Node(4,6,0)}
        m.materials = {1: Material(1,200000,0.02,0.0004),
                       2: Material(2,200000,0.005,0.0001)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.0004),
            FrameElement2D(2, 2, 3, E=200000, A=0.02, I=0.0004),
            FrameElement2D(3, 4, 3, E=200000, A=0.02, I=0.0004),
            TrussElement2D(4, 1, 3, E=200000, A=0.005),
            TrussElement2D(5, 2, 4, E=200000, A=0.005),
        ]
        m.supports = {1: Support(1,True,True,False), 4: Support(4,True,True,False)}
        m.nodal_loads = [NodalLoad(2, 15, 0, 0)]
        r = run_analysis(m, verbose=False)
        assert r.status == "ok"
        assert r.residual < 1e-8
        assert not np.any(np.isnan(r.D))


# ================================================================
#  RT-8 — Q3 Case (a): Portal on rollers (mechanism)
# ================================================================
class TestRTQ3aCaseMechanism:
    def test_portal_rollers_is_unstable(self):
        """Portal frame with only roller supports → sway mechanism."""
        m = StructuralModel(title="Q3a")
        m.nodes = {1: Node(1,0,0), 2: Node(2,0,4), 3: Node(3,6,4), 4: Node(4,6,0)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08),
            FrameElement2D(2, 2, 3, E=200000, A=0.02, I=0.08),
            FrameElement2D(3, 4, 3, E=200000, A=0.02, I=0.08),
        ]
        m.supports = {1: Support(1, False, True), 4: Support(4, False, True)}
        m.nodal_loads = [NodalLoad(2, 10, 0, 0)]
        r = run_analysis(m, verbose=False)
        assert r.status == "error"
        assert any("UX" in w for w in r.warnings)

    def test_portal_rollers_does_not_crash(self):
        """Program must not crash — returns error status gracefully."""
        m = StructuralModel(title="Q3a")
        m.nodes = {1: Node(1,0,0), 2: Node(2,0,4), 3: Node(3,6,4), 4: Node(4,6,0)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08),
            FrameElement2D(2, 2, 3, E=200000, A=0.02, I=0.08),
            FrameElement2D(3, 4, 3, E=200000, A=0.02, I=0.08),
        ]
        m.supports = {1: Support(1, False, True), 4: Support(4, False, True)}
        m.nodal_loads = [NodalLoad(2, 10, 0, 0)]
        r = run_analysis(m, verbose=False)
        # Must return an AnalysisResult, not raise an exception
        assert hasattr(r, "status")


# ================================================================
#  RT-9 — Q3 Case (b): Disconnected floating member
# ================================================================
class TestRTQ3bDisconnected:
    def test_floating_component_detected(self):
        """Unsupported disconnected component → error before assembly."""
        m = StructuralModel(title="Q3b")
        m.nodes = {1: Node(1,0,0), 2: Node(2,0,4), 3: Node(3,5,0), 4: Node(4,5,4)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08),
            FrameElement2D(2, 3, 4, E=200000, A=0.02, I=0.08),
        ]
        m.supports = {1: Support(1, True, True, True)}
        m.nodal_loads = [NodalLoad(2, 10, -5, 0)]
        r = run_analysis(m, verbose=False)
        assert r.status == "error"
        assert any("Disconnected" in w or "no supports" in w for w in r.warnings)


# ================================================================
#  RT-10 — Q3 Case (c): Two independent supported sub-structures
# ================================================================
class TestRTQ3cDisconnectedSupported:
    def test_warns_about_components(self):
        """Two separate supported beams → warning, solves anyway."""
        m = StructuralModel(title="Q3c")
        m.nodes = {1: Node(1,0,0), 2: Node(2,4,0), 3: Node(3,6,0), 4: Node(4,10,0)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08),
            FrameElement2D(2, 3, 4, E=200000, A=0.02, I=0.08),
        ]
        m.supports = {
            1: Support(1, True, True), 2: Support(2, False, True),
            3: Support(3, True, True), 4: Support(4, False, True),
        }
        m.nodal_loads = [NodalLoad(2, 0, -20, 0)]
        r = run_analysis(m, verbose=False)
        assert r.status == "ok"
        assert any("disconnected" in w.lower() or "component" in w.lower() for w in r.warnings)


# ================================================================
#  RT-11 — Q3 Case (d): Two-bar truss (analytical)
# ================================================================
class TestRTQ3dTruss:
    def test_apex_displacement(self):
        """Two-bar truss apex deflection = PL/(2EA sin²α)."""
        m = StructuralModel(title="Q3d")
        m.nodes = {1: Node(1,0,0), 2: Node(2,3,4), 3: Node(3,6,0)}
        m.materials = {1: Material(1,200000,0.01,0.0001)}
        m.elements = [
            TrussElement2D(1, 1, 2, E=200000, A=0.01),
            TrussElement2D(2, 2, 3, E=200000, A=0.01),
        ]
        m.supports = {1: Support(1, True, True), 3: Support(3, True, True)}
        m.nodal_loads = [NodalLoad(2, 0, -12, 0)]
        r = run_analysis(m, verbose=False)
        assert r.status == "ok"
        # Analytical: δ = PL/(2EA sin²α), sin α = 4/5, L = 5
        expected_uy = -12 * 5 / (2 * 200000 * 0.01 * (4/5)**2)
        idx = r.E_map[2]["uy"]
        assert abs(r.D[idx] - expected_uy) < 1e-9

    def test_truss_no_rz_dof(self):
        """Pure truss nodes should have no Rz DOF (auto-omitted)."""
        m = StructuralModel(title="Q3d")
        m.nodes = {1: Node(1,0,0), 2: Node(2,3,4), 3: Node(3,6,0)}
        m.materials = {1: Material(1,200000,0.01,0.0001)}
        m.elements = [
            TrussElement2D(1, 1, 2, E=200000, A=0.01),
            TrussElement2D(2, 2, 3, E=200000, A=0.01),
        ]
        m.supports = {1: Support(1, True, True), 3: Support(3, True, True)}
        from structural_analysis.assembler import DofManager
        dofs = DofManager.from_model(m)
        assert dofs.active_map[2]["rz"] is None  # apex has no Rz


# ================================================================
#  RT-12 — Q3 Case (e): Two-span beam with internal hinge
# ================================================================
class TestRTQ3eInternalHinge:
    def test_hinge_moment_zero_both_sides(self):
        """Moment at internal hinge (node 3) must be zero from both elements."""
        m = StructuralModel(title="Q3e")
        m.nodes = {1: Node(1,0,0), 2: Node(2,5,0), 3: Node(3,8,0), 4: Node(4,12,0)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08),
            FrameElement2D(2, 2, 3, E=200000, A=0.02, I=0.08, release_j=True),
            FrameElement2D(3, 3, 4, E=200000, A=0.02, I=0.08, release_i=True,
                           member_loads=[PointLoad(py=-15, a=2.0)]),
        ]
        m.supports = {1: Support(1,True,True), 2: Support(2,False,True), 4: Support(4,False,True)}
        m.elements[0].member_loads = [UniformDistributedLoad(wy=-10)]
        r = run_analysis(m, verbose=False)
        assert r.status == "ok"
        # Elem 2 M_j (hinge side) = 0
        assert abs(r.member_results[2]["f_local"][5]) < 0.01
        # Elem 3 M_i (hinge side) = 0
        assert abs(r.member_results[3]["f_local"][2]) < 0.01

    def test_total_reactions_equal_load(self):
        """Sum of vertical reactions = total applied load."""
        m = StructuralModel(title="Q3e")
        m.nodes = {1: Node(1,0,0), 2: Node(2,5,0), 3: Node(3,8,0), 4: Node(4,12,0)}
        m.materials = {1: Material(1,200000,0.02,0.08)}
        m.elements = [
            FrameElement2D(1, 1, 2, E=200000, A=0.02, I=0.08,
                           member_loads=[UniformDistributedLoad(wy=-10)]),
            FrameElement2D(2, 2, 3, E=200000, A=0.02, I=0.08, release_j=True),
            FrameElement2D(3, 3, 4, E=200000, A=0.02, I=0.08, release_i=True,
                           member_loads=[PointLoad(py=-15, a=2.0)]),
        ]
        m.supports = {1: Support(1,True,True), 2: Support(2,False,True), 4: Support(4,False,True)}
        r = run_analysis(m, verbose=False)
        total_ry = sum(r.reactions[nid].get("uy", 0) for nid in r.reactions)
        total_load = 10*5 + 15  # UDL on 5m + point load 15 kN
        assert abs(total_ry - total_load) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ================================================================
#  ASSIGNMENT 4: Unit tests for new capabilities
# ================================================================

class TestA4UnitThermalFEF:
    """UT-A4-1: Unit test for thermal fixed-end force formulas."""

    def test_uniform_thermal_axial_fef(self):
        """Uniform ΔT should produce axial FEF N_T = E·A·α·ΔT on both ends.

        Verification: fully clamped bar of L=4, A=0.01, E=2e8 kN/m², α=1.2e-5,
        ΔT=+50°C. Expected N_T = 2e8 · 0.01 · 1.2e-5 · 50 = 1200 kN.

        Uses FrameTemperatureLoad with t_top=t_bottom=ΔT (pure uniform heating).
        α is stored on the element (read from Material in production).
        Tolerance: 1e-9 (machine precision).
        """
        E, A, alpha, dT, L = 2e8, 0.01, 1.2e-5, 50.0, 4.0
        nodes = {1: Node(1, 0, 0), 2: Node(2, L, 0)}
        elem = FrameElement2D(
            id=1, node_i=1, node_j=2, E=E, A=A, I=1e-4,
            alpha=alpha,
            member_loads=[FrameTemperatureLoad(t_top=dT, t_bottom=dT)],
        )
        p = elem.local_consistent_load(nodes)
        N_T_expected = E * A * alpha * dT
        assert_allclose(p[0], +N_T_expected, atol=1e-9)
        assert_allclose(p[3], -N_T_expected, atol=1e-9)
        # No flexural contribution
        assert_allclose(p[1:3], 0.0, atol=1e-9)
        assert_allclose(p[4:6], 0.0, atol=1e-9)

    def test_thermal_gradient_moment_fef(self):
        """Pure gradient (t_bot − t_top, zero mean) should produce M_T = E·I·α·ΔT/h.

        Verification: beam L=4, E=3e7 kN/m², I = 0.4·0.8³/12, α=8e-6,
        depth=0.8 m. Uses t_top=-25, t_bottom=+25 → zero mean, pure gradient
        ΔT_diff = 50°C.
        Expected M_T = 3e7 · I · 8e-6 · 50 / 0.8.
        Tolerance: 1e-9.
        """
        E, alpha, dTg, h, L = 3e7, 8e-6, 50.0, 0.8, 4.0
        I_val = 0.4 * 0.8**3 / 12
        nodes = {1: Node(1, 0, 0), 2: Node(2, L, 0)}
        elem = FrameElement2D(
            id=1, node_i=1, node_j=2, E=E, A=0.32, I=I_val,
            alpha=alpha, depth=h,
            member_loads=[FrameTemperatureLoad(t_top=-25.0, t_bottom=+25.0)],
        )
        p = elem.local_consistent_load(nodes)
        M_T_expected = E * I_val * alpha * dTg / h
        assert_allclose(-p[2], M_T_expected, atol=1e-9)  # p[2] = -M_T
        assert_allclose(p[5], M_T_expected, atol=1e-9)   # p[5] = +M_T
        # No axial contribution from pure gradient
        assert_allclose(p[0], 0.0, atol=1e-9)
        assert_allclose(p[3], 0.0, atol=1e-9)

    def test_truss_cannot_carry_frame_thermal(self):
        """TrussElement2D must reject FrameTemperatureLoad with TypeError.

        Explicit validation replaces the prior silent-ignore behavior.
        """
        import pytest as _pytest
        nodes = {1: Node(1, 0, 0), 2: Node(2, 4, 0)}
        elem = TrussElement2D(
            id=1, node_i=1, node_j=2, E=2e8, A=0.01, alpha=1.2e-5,
            member_loads=[FrameTemperatureLoad(t_top=0, t_bottom=50)],
        )
        with _pytest.raises(TypeError, match="FrameTemperatureLoad"):
            elem.local_consistent_load(nodes)

    def test_frame_cannot_carry_truss_thermal(self):
        """FrameElement2D must reject TrussTemperatureLoad with TypeError."""
        import pytest as _pytest
        nodes = {1: Node(1, 0, 0), 2: Node(2, 4, 0)}
        elem = FrameElement2D(
            id=1, node_i=1, node_j=2, E=2e8, A=0.01, I=1e-4, alpha=1.2e-5,
            member_loads=[TrussTemperatureLoad(delta_T=50)],
        )
        with _pytest.raises(TypeError, match="TrussTemperatureLoad"):
            elem.local_consistent_load(nodes)


# ================================================================
#  ASSIGNMENT 4: Integration tests for new capabilities
# ================================================================

class TestA4IntegrationSupportSettlement:
    """IT-A4-1: End-to-end integration test for support settlement.

    Structure: propped cantilever with fixed end at node 1 and roller
    at node 2 with a prescribed vertical settlement Δ = −10 mm.
    No external loads are applied.

    Closed-form (beam table): for a propped cantilever with support
    settlement Δ at the roller, the support reaction is
        R_B = 3·E·I·Δ / L³
    and the fixed-end moment is
        M_A = 3·E·I·Δ / L²

    Verification tolerance: 1e-6 relative.
    """

    def test_settlement_propped_cantilever(self):
        E, A, I, L = 2e8, 0.01, 1e-4, 4.0
        delta = -0.010  # 10 mm settlement downward
        m = StructuralModel(title="Propped cantilever settlement")
        m.nodes = {1: Node(1, 0, 0), 2: Node(2, L, 0)}
        m.materials = {1: Material(1, E, A, I)}
        m.elements = [FrameElement2D(
            id=1, node_i=1, node_j=2, E=E, A=A, I=I
        )]
        m.supports = {
            1: Support(1, True, True, True),
            2: Support(2, False, True, False, settle_uy=delta),
        }
        r = run_analysis(m, verbose=False)
        assert r.status == "ok"
        # 1. Prescribed settlement shows up in the displacement field
        idx_uy2 = r.E_map[2]["uy"]
        assert abs(r.D[idx_uy2] - delta) < 1e-12
        # 2. Reaction at B matches closed form
        R_B_expected = 3 * E * I * delta / L**3
        R_B = r.reactions[2]["uy"]
        assert abs((R_B - R_B_expected) / R_B_expected) < 1e-6
        # 3. Fixed-end moment at A matches closed form (magnitude)
        M_A_expected = 3 * E * I * delta / L**2
        assert abs(abs(r.reactions[1]["rz"]) - abs(M_A_expected)) < 1e-6


class TestA4IntegrationClampedThermalBar:
    """IT-A4-2: End-to-end integration test for thermal loading.

    Structure: fully clamped bar (fixed at both ends) heated by ΔT = +50°C.
    Expected:
        - Zero displacements (both ends restrained).
        - Reactions: ±N_T at the two ends, with N_T = E·A·α·ΔT.
        - Member internal axial force: N = −N_T (compression).

    This confirms the whole pipeline (assembly, solve, post-processing)
    handles thermal loading end-to-end.

    Verification tolerance: 1e-9 absolute.
    """

    def test_clamped_thermal_bar_reactions(self):
        E, A, alpha, dT, L = 2e8, 0.01, 1.2e-5, 50.0, 4.0
        m = StructuralModel(title="Clamped thermal bar")
        m.nodes = {1: Node(1, 0, 0), 2: Node(2, L, 0)}
        m.materials = {1: Material(1, E, A, 1e-4, alpha=alpha)}
        m.elements = [FrameElement2D(
            id=1, node_i=1, node_j=2, E=E, A=A, I=1e-4,
            alpha=alpha,
            member_loads=[FrameTemperatureLoad(t_top=dT, t_bottom=dT)],
        )]
        m.supports = {
            1: Support(1, True, True, True),
            2: Support(2, True, True, True),
        }
        r = run_analysis(m, verbose=False)
        assert r.status == "ok"
        N_T = E * A * alpha * dT  # = 1200 kN

        # 1. Displacements must all be zero
        assert np.max(np.abs(r.D)) < 1e-9

        # 2. Reactions: node 1 pushes right (+N_T), node 2 pushes left (−N_T)
        #    Because heating wants to expand — supports resist with compression
        assert abs(r.reactions[1]["ux"] - (-N_T)) < 1e-6
        assert abs(r.reactions[2]["ux"] - (+N_T)) < 1e-6

        # 3. Member internal axial force = −N_T (compression)
        assert abs(r.member_results[1]["f_local"][0] - (-N_T)) < 1e-6

