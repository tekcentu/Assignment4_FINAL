"""Rigid end offsets — stiffness transformation and solver integration.

Transformation order under test (documented in FrameElement2D):
    k_flex(L_flex) → T_offset (joint coords) → release condensation → R.

Key guarantees:

* zero offsets give BIT-IDENTICAL stiffness and results (legacy path);
* k_joint = Tᵀ·k_flex·T stays symmetric;
* nodal loads remain at the analytical joints — the rigid-arm lever
  effect is carried by the transformation, not by moving loads;
* reversed i/j orientation with swapped offsets is physically identical.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.model import (
    StructuralModel, Node, Support, NodalLoad,
)
from structural_analysis.element import FrameElement2D
from structural_analysis.main import run_analysis

E, A, I = 200_000.0, 0.02, 0.08


def _disp(r, node_id: int, dof: str) -> float:
    """Nodal displacement from the flat solution vector via E_map."""
    idx = r.E_map[node_id][dof]
    assert idx is not None, f"node {node_id} dof {dof} is restrained"
    return float(r.D[idx])


def _cantilever(L=6.0, *, offset_i=0.0, offset_j=0.0, P=10.0):
    """Fixed at node 1, transverse nodal load −P at node 2 (the joint)."""
    m = StructuralModel(title="cantilever")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    m.elements = [FrameElement2D(
        1, 1, 2, E=E, A=A, I=I, offset_i=offset_i, offset_j=offset_j,
    )]
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    m.nodal_loads = [NodalLoad(node_id=2, fy=-P)]
    return m


# ── transformation matrix ────────────────────────────────────────────────


def test_offset_transform_identity_when_zero():
    e = FrameElement2D(1, 1, 2, E=E, A=A, I=I)
    assert np.array_equal(e._offset_transform(), np.eye(6))


def test_offset_transform_entries():
    e = FrameElement2D(1, 1, 2, E=E, A=A, I=I, offset_i=0.4, offset_j=0.3)
    T = e._offset_transform()
    expected = np.eye(6)
    expected[1, 2] = 0.4
    expected[4, 5] = -0.3
    assert np.allclose(T, expected)


def test_joint_stiffness_symmetric():
    nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    for ei, ej in [(0.0, 0.0), (0.5, 0.0), (0.0, 0.7), (0.45, 0.85)]:
        e = FrameElement2D(1, 1, 2, E=E, A=A, I=I,
                           offset_i=ei, offset_j=ej)
        k = e.joint_local_stiffness(nodes)
        assert np.max(np.abs(k - k.T)) < 1e-9 * max(1.0, np.max(np.abs(k)))


def test_zero_offsets_bit_identical_stiffness():
    nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(1, 1, 2, E=E, A=A, I=I)
    assert np.array_equal(
        e.joint_local_stiffness(nodes), e.raw_local_stiffness(nodes),
    )


# ── zero-offset solver regression ────────────────────────────────────────


def test_zero_offset_results_identical_to_legacy():
    m0 = _cantilever()
    r0 = run_analysis(m0, verbose=False)
    assert r0.status == "ok"
    # Hand value: tip deflection of a 6 m cantilever under 10 kN:
    # δ = PL³/3EI = 10·216/(3·16000) = 0.045 m downward.
    tip = _disp(r0, 2, "uy")
    assert tip == pytest.approx(-10 * 6**3 / (3 * E * I), rel=1e-9)


# ── nodal load + rigid arm: closed forms ─────────────────────────────────


def test_cantilever_offset_at_support_hand_calc():
    """Rigid zone at the SUPPORT: the flexible cantilever is the
    remaining L_f = L − e span, loaded at its tip → δ = P·L_f³/3EI."""
    e_off = 1.0
    P = 10.0
    L = 6.0
    m = _cantilever(L=L, offset_i=e_off, P=P)
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    Lf = L - e_off
    assert _disp(r, 2, "uy") == pytest.approx(
        -P * Lf**3 / (3 * E * I), rel=1e-9,
    )


def test_cantilever_offset_at_tip_lever_arm_hand_calc():
    """Rigid zone at the TIP: nodal load stays at the joint; the rigid
    arm transfers it to the flexible face as shear P AND moment P·e.

    δ_face = P·L_f³/3EI + (P·e)·L_f²/2EI
    θ_face = P·L_f²/2EI + (P·e)·L_f/EI
    δ_joint = δ_face + e·θ_face        (rigid-arm kinematics)

    This is the test that proves the lever-arm effect is captured —
    naively shortening the element would give only P·L_f³/3EI.
    """
    e_off = 1.0
    P = 10.0
    L = 6.0
    m = _cantilever(L=L, offset_j=e_off, P=P)
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    Lf = L - e_off
    EI = E * I
    d_face = P * Lf**3 / (3 * EI) + (P * e_off) * Lf**2 / (2 * EI)
    th_face = P * Lf**2 / (2 * EI) + (P * e_off) * Lf / EI
    d_joint = d_face + e_off * th_face
    assert _disp(r, 2, "uy") == pytest.approx(-d_joint, rel=1e-9)
    # Sanity: stiffer than the no-offset cantilever, softer than naive
    # shortening would suggest at the FACE but the joint adds arm terms.
    assert abs(_disp(r, 2, "uy")) < P * L**3 / (3 * EI)


def test_offsets_stiffen_structure():
    r_plain = run_analysis(_cantilever(), verbose=False)
    r_off = run_analysis(_cantilever(offset_i=1.0), verbose=False)
    assert abs(_disp(r_off, 2, "uy")) < abs(_disp(r_plain, 2, "uy"))


def test_member_results_include_face_displacements_for_offsets():
    e_off = 1.0
    m = _cantilever(L=6.0, offset_j=e_off, P=10.0)
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    mr = r.member_results[1]
    d_joint = mr["d_local"]
    d_face = mr["d_local_face"]
    expected = m.elements[0]._offset_transform() @ d_joint
    assert np.allclose(d_face, expected)
    assert d_face[4] == pytest.approx(d_joint[4] - e_off * d_joint[5])
    assert not np.allclose(d_face, d_joint)


# ── reference comparison: near-rigid stub model ──────────────────────────


def test_offset_matches_near_rigid_stub_reference():
    """Offset element vs an explicit 2-element model whose support-side
    stub is quasi-rigid (EI and EA scaled ×1e8). Same tip load; joint
    displacements must agree to engineering tolerance."""
    e_off = 1.0
    P = 10.0
    L = 6.0
    m_off = _cantilever(L=L, offset_i=e_off, P=P)
    r_off = run_analysis(m_off, verbose=False)

    m_ref = StructuralModel(title="stub reference")
    m_ref.nodes = {
        1: Node(1, 0.0, 0.0),
        3: Node(3, e_off, 0.0),
        2: Node(2, L, 0.0),
    }
    stiff = 1e8
    m_ref.elements = [
        FrameElement2D(1, 1, 3, E=E * stiff, A=A, I=I),
        FrameElement2D(2, 3, 2, E=E, A=A, I=I),
    ]
    m_ref.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    m_ref.nodal_loads = [NodalLoad(node_id=2, fy=-P)]
    r_ref = run_analysis(m_ref, verbose=False)
    assert r_off.status == r_ref.status == "ok"
    assert _disp(r_off, 2, "uy") == pytest.approx(
        _disp(r_ref, 2, "uy"), rel=1e-6,
    )


def test_release_plus_offset_matches_stub_reference():
    """Hinge at the loaded joint end + offset at the support end,
    propped-cantilever configuration, vs the near-rigid-stub model.
    Verifies the condensation-after-offset ordering."""
    e_off = 0.8
    L = 6.0
    m_off = StructuralModel(title="hinge+offset")
    m_off.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    m_off.elements = [FrameElement2D(
        1, 1, 2, E=E, A=A, I=I, offset_i=e_off, release_j=True,
    )]
    m_off.supports = {
        1: Support(1, ux=True, uy=True, rz=True),
        2: Support(2, ux=False, uy=True, rz=False),
    }
    m_off.nodal_loads = [NodalLoad(node_id=2, fx=5.0)]
    r_off = run_analysis(m_off, verbose=False)

    m_ref = StructuralModel(title="hinge+stub reference")
    m_ref.nodes = {
        1: Node(1, 0.0, 0.0),
        3: Node(3, e_off, 0.0),
        2: Node(2, L, 0.0),
    }
    m_ref.elements = [
        # Quasi-rigid stub: scale section properties only (scaling E
        # and A together makes the system ill-conditioned).
        FrameElement2D(1, 1, 3, E=E, A=A * 1e8, I=I * 1e8),
        FrameElement2D(2, 3, 2, E=E, A=A, I=I, release_j=True),
    ]
    m_ref.supports = {
        1: Support(1, ux=True, uy=True, rz=True),
        2: Support(2, ux=False, uy=True, rz=False),
    }
    m_ref.nodal_loads = [NodalLoad(node_id=2, fx=5.0)]
    r_ref = run_analysis(m_ref, verbose=False)
    assert r_off.status == r_ref.status == "ok"
    assert _disp(r_off, 2, "ux") == pytest.approx(
        _disp(r_ref, 2, "ux"), rel=1e-6,
    )


# ── orientation ──────────────────────────────────────────────────────────


def test_reversed_orientation_with_swapped_offsets_equivalent():
    """Same physical member drawn j→i with offsets swapped must produce
    the same joint displacement under the same physical load."""
    e_off = 1.0
    P = 10.0
    L = 6.0
    m_fwd = _cantilever(L=L, offset_j=e_off, P=P)
    r_fwd = run_analysis(m_fwd, verbose=False)

    m_rev = StructuralModel(title="reversed")
    m_rev.nodes = {1: Node(1, L, 0.0), 2: Node(2, 0.0, 0.0)}
    # Element runs from the free end (node 1 at x=L) to the support
    # (node 2 at x=0): the rigid zone at the free PHYSICAL end is now
    # at the element's i-end.
    m_rev.elements = [FrameElement2D(
        1, 1, 2, E=E, A=A, I=I, offset_i=e_off,
    )]
    m_rev.supports = {2: Support(2, ux=True, uy=True, rz=True)}
    m_rev.nodal_loads = [NodalLoad(node_id=1, fy=-P)]
    r_rev = run_analysis(m_rev, verbose=False)
    assert r_fwd.status == r_rev.status == "ok"
    assert _disp(r_rev, 1, "uy") == pytest.approx(
        _disp(r_fwd, 2, "uy"), rel=1e-12,
    )


# ── equilibrium + modal direction ────────────────────────────────────────


def test_equilibrium_check_passes_with_offsets():
    m = _cantilever(offset_i=0.7, offset_j=0.4)
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    # Reactions balance the applied load.
    total_ry = sum(rx.get("uy", 0.0) for rx in r.reactions.values())
    assert total_ry == pytest.approx(10.0, rel=1e-9)


def test_modal_frequency_increases_with_offsets():
    """Offsets shorten the flexible span → stiffer → higher f₁. Mass
    stays full-length Hermitian (documented V1 behaviour)."""
    from structural_analysis.modal import solve_modal

    def beam(off):
        m = StructuralModel(title="modal")
        m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
        m.elements = [FrameElement2D(
            1, 1, 2, E=E, A=A, I=I, rho=7850.0, offset_i=off,
        )]
        m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
        return m

    r0 = solve_modal(beam(0.0), n_modes=1)
    r1 = solve_modal(beam(1.0), n_modes=1)
    assert r1.frequencies[0] > r0.frequencies[0]
