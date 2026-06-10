"""
Core 3D-FEM tests: space-frame stiffness, local axes, promotion,
planar equivalence with the 2D pipeline, and textbook benchmarks.
"""

import numpy as np
import pytest

from structural_analysis.assembler import (
    DofManager, model_is_3d, prepare_solve_elements,
)
from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.element3d import (
    FrameElement3D, TrussElement3D, local_axes, promote_element_to_3d,
)
from structural_analysis.main import run_analysis
from structural_analysis.model import (
    Node, NodalLoad, StructuralModel, Support,
    UniformDistributedLoad, PointLoad, FrameTemperatureLoad,
    TrussTemperatureLoad,
)


E = 200e6      # kN/m²
G = 80e6       # kN/m²
A = 0.01       # m²
IY = 2e-5      # m⁴
IZ = 4e-5      # m⁴
J = 3e-5       # m⁴
L = 3.0        # m


def _space_frame(eid=1, ni=1, nj=2, **kw):
    defaults = dict(E=E, A=A, Iy=IY, Iz=IZ, J=J, G=G)
    defaults.update(kw)
    return FrameElement3D(id=eid, node_i=ni, node_j=nj, **defaults)


def _cantilever_model(**elem_kw):
    """Cantilever along +X, fully fixed at node 1."""
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0, 0.0)
    m.nodes[2] = Node(2, L, 0.0, 0.0)
    m.elements.append(_space_frame(**elem_kw))
    m.supports[1] = Support(1, True, True, True,
                            uz=True, rx=True, ry=True)
    return m


def _disp(result, nid, dof):
    idx = result.E_map[nid][dof]
    return 0.0 if idx is None else float(result.D[idx])


# ── local axes convention ───────────────────────────────────────


def test_local_axes_match_2d_for_xy_plane_member():
    ni, nj = Node(1, 0.0, 0.0), Node(2, 3.0, 4.0)
    _, lam = local_axes(ni, nj)
    c, s = 0.6, 0.8
    np.testing.assert_allclose(lam[0], [c, s, 0.0], atol=1e-14)
    np.testing.assert_allclose(lam[1], [-s, c, 0.0], atol=1e-14)
    np.testing.assert_allclose(lam[2], [0.0, 0.0, 1.0], atol=1e-14)


def test_local_axes_right_handed_for_global_z_member():
    ni, nj = Node(1, 0.0, 0.0, 0.0), Node(2, 0.0, 0.0, 5.0)
    _, lam = local_axes(ni, nj)
    # Right-handed and orthonormal even in the degenerate orientation.
    np.testing.assert_allclose(lam @ lam.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(lam[0], lam[1]), lam[2],
                               atol=1e-12)


def test_local_axes_roll_rotates_section():
    ni, nj = Node(1, 0.0, 0.0, 0.0), Node(2, 2.0, 0.0, 0.0)
    _, lam0 = local_axes(ni, nj)
    _, lam90 = local_axes(ni, nj, roll=np.pi / 2)
    np.testing.assert_allclose(lam90[1], lam0[2], atol=1e-12)
    np.testing.assert_allclose(lam90[2], -lam0[1], atol=1e-12)


# ── stiffness matrix sanity ────────────────────────────────────


def test_space_frame_stiffness_symmetric_with_rigid_body_nullspace():
    nodes = {1: Node(1, 0.5, 1.0, -0.7), 2: Node(2, 3.1, 2.0, 1.3)}
    k, _ = _space_frame().global_stiffness_and_load(nodes)
    np.testing.assert_allclose(k, k.T, atol=1e-6)
    svals = np.linalg.svd(k, compute_uv=False)
    tol = max(svals) * 1e-9
    assert int(np.sum(svals > tol)) == 6  # 12 DOFs − 6 rigid-body modes


def test_space_frame_reduces_to_2d_stiffness_in_plane():
    nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 3.0, 4.0)}
    e2 = FrameElement2D(id=1, node_i=1, node_j=2, E=E, A=A, I=IZ)
    e3 = _space_frame()
    k2, _ = e2.global_stiffness_and_load(nodes)
    k3, _ = e3.global_stiffness_and_load(nodes)
    # In-plane global DOFs of the 12-DOF matrix: ux, uy, rz per node.
    ip = [0, 1, 5, 6, 7, 11]
    np.testing.assert_allclose(k3[np.ix_(ip, ip)], k2, rtol=1e-12)


# ── cantilever benchmarks ──────────────────────────────────────


def test_cantilever_tip_deflection_in_plane():
    m = _cantilever_model()
    P = 10.0
    m.nodal_loads.append(NodalLoad(2, fy=-P))
    m.force_3d = True
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    assert _disp(r, 2, "uy") == pytest.approx(-P * L**3 / (3 * E * IZ),
                                              rel=1e-9)


def test_cantilever_tip_deflection_out_of_plane():
    m = _cantilever_model()
    P = 10.0
    m.nodal_loads.append(NodalLoad(2, fz=-P))
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    assert _disp(r, 2, "uz") == pytest.approx(-P * L**3 / (3 * E * IY),
                                              rel=1e-9)


def test_cantilever_torsion():
    m = _cantilever_model()
    T = 5.0
    m.nodal_loads.append(NodalLoad(2, mx=T))
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    assert _disp(r, 2, "rx") == pytest.approx(T * L / (G * J), rel=1e-9)
    # Reaction torque balances the applied torque.
    assert r.reactions[1]["rx"] == pytest.approx(-T, rel=1e-9)


def test_cantilever_axial():
    m = _cantilever_model()
    P = 100.0
    m.nodal_loads.append(NodalLoad(2, fx=P))
    m.force_3d = True
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    assert _disp(r, 2, "ux") == pytest.approx(P * L / (E * A), rel=1e-9)


def test_cantilever_udl_local_z():
    m = _cantilever_model()
    w = -7.0
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=0.0, wz=w),
    )
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    assert _disp(r, 2, "uz") == pytest.approx(w * L**4 / (8 * E * IY),
                                              rel=1e-9)
    # Total vertical-z reaction balances the load.
    assert r.reactions[1]["uz"] == pytest.approx(-w * L, rel=1e-9)


def test_cantilever_point_load_z_at_midspan():
    m = _cantilever_model()
    P, a = -12.0, L / 2
    m.elements[0].member_loads.append(PointLoad(py=0.0, pz=P, a=a))
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    # δ(L) = P·a²·(3L − a)/(6EI) for a cantilever point load at a.
    expected = P * a**2 * (3 * L - a) / (6 * E * IY)
    assert _disp(r, 2, "uz") == pytest.approx(expected, rel=1e-9)


# ── grillage benchmark (bending + torsion coupling) ────────────


def test_grillage_tip_deflection_with_torsion_coupling():
    """Right-angle grillage in the XZ plane, vertical (Y) tip load.

    δ_tip = FL³/3EI(m2 about its vertical-bending axis)
          + FL³/3EI(m1) + F·L³/GJ(m1)
    """
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0, 0.0)
    m.nodes[2] = Node(2, L, 0.0, 0.0)
    m.nodes[3] = Node(3, L, 0.0, L)
    m.elements.append(_space_frame(eid=1, ni=1, nj=2))
    m.elements.append(_space_frame(eid=2, ni=2, nj=3))
    m.supports[1] = Support(1, True, True, True,
                            uz=True, rx=True, ry=True)
    F = -10.0
    m.nodal_loads.append(NodalLoad(3, fy=F))
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    # Member 1 (along X): vertical load bends about local z (Iz);
    # member 2 (along Z): local ẑ is global Y ⇒ bends about local y (Iy).
    expected = (F * L**3 / (3 * E * IY)
                + F * L**3 / (3 * E * IZ)
                + F * L**3 / (G * J))
    assert _disp(r, 3, "uy") == pytest.approx(expected, rel=1e-9)


# ── space truss benchmark ──────────────────────────────────────


def test_space_truss_tripod():
    """Symmetric tripod, vertical load at apex: N = −P·L/(3h)."""
    m = StructuralModel()
    h, rad = 4.0, 2.0
    m.nodes[1] = Node(1, 0.0, h, 0.0)  # apex
    for k in range(3):
        ang = 2 * np.pi * k / 3
        m.nodes[2 + k] = Node(2 + k, rad * np.cos(ang), 0.0,
                              rad * np.sin(ang))
        m.elements.append(TrussElement3D(
            id=1 + k, node_i=2 + k, node_j=1, E=E, A=A,
        ))
        m.supports[2 + k] = Support(2 + k, True, True, False, uz=True)
    P = 30.0
    m.nodal_loads.append(NodalLoad(1, fy=-P))
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    bar_L = float(np.sqrt(h * h + rad * rad))
    expected_N = -P * bar_L / (3 * h)  # compression
    for eid in (1, 2, 3):
        f = r.member_results[eid]["f_local"]
        # j-end axial force is +tension under the q = k·d − p convention.
        assert f[3] == pytest.approx(expected_N, rel=1e-9)
    # Vertical reactions sum to P.
    total_ry = sum(rx.get("uy", 0.0) for rx in r.reactions.values())
    assert total_ry == pytest.approx(P, rel=1e-9)


# ── planar equivalence: 2D model promoted to the 3D pipeline ───


def _portal_frame_2d():
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 0.0, 3.0)
    m.nodes[3] = Node(3, 4.0, 3.0)
    m.nodes[4] = Node(4, 4.0, 0.0)
    col1 = FrameElement2D(id=1, node_i=1, node_j=2, E=E, A=A, I=IZ,
                          alpha=1.2e-5, depth=0.4)
    beam = FrameElement2D(id=2, node_i=2, node_j=3, E=E, A=A, I=IZ,
                          alpha=1.2e-5, depth=0.4, release_j=True)
    col2 = FrameElement2D(id=3, node_i=3, node_j=4, E=E, A=A, I=IZ,
                          alpha=1.2e-5, depth=0.4)
    beam.member_loads.append(UniformDistributedLoad(wy=-10.0))
    beam.member_loads.append(FrameTemperatureLoad(t_top=10.0,
                                                  t_bottom=30.0))
    col1.member_loads.append(PointLoad(py=5.0, a=1.5))
    m.elements += [col1, beam, col2]
    m.supports[1] = Support(1, True, True, True)
    m.supports[4] = Support(4, True, True, False,
                            settle_uy=-0.01)
    m.nodal_loads.append(NodalLoad(2, fx=12.0, mz=4.0))
    return m


def test_planar_model_solves_identically_through_3d_pipeline():
    m2 = _portal_frame_2d()
    r2 = run_analysis(m2, verbose=False)
    assert r2.status == "ok"

    m3 = _portal_frame_2d()
    # The 6-DOF solve needs the out-of-plane rigid-body modes
    # restrained — physically honest, and the in-plane response stays
    # bit-identical to the 2D pipeline.
    m3.supports[1] = Support(1, True, True, True,
                             uz=True, rx=True, ry=True)
    m3.supports[4] = Support(4, True, True, False,
                             settle_uy=-0.01, uz=True)
    m3.force_3d = True
    assert model_is_3d(m3)
    r3 = run_analysis(m3, verbose=False)
    assert r3.status == "ok"

    for nid in m2.nodes:
        for dof in ("ux", "uy", "rz"):
            assert _disp(r3, nid, dof) == pytest.approx(
                _disp(r2, nid, dof), rel=1e-9, abs=1e-14,
            ), f"node {nid} {dof}"
        # Out-of-plane response of a planar model must be exactly zero.
        for dof in ("uz", "rx", "ry"):
            assert _disp(r3, nid, dof) == pytest.approx(0.0, abs=1e-12)

    for nid, r2_react in r2.reactions.items():
        for dof, val in r2_react.items():
            assert r3.reactions[nid][dof] == pytest.approx(
                val, rel=1e-9, abs=1e-12,
            )

    for eid in (1, 2, 3):
        f2 = r2.member_results[eid]["f_local"]
        f3 = r3.member_results[eid]["f_local_inplane"]
        np.testing.assert_allclose(f3, f2, rtol=1e-9, atol=1e-9)


def test_planar_truss_thermal_matches_2d_pipeline():
    def build():
        m = StructuralModel()
        m.nodes[1] = Node(1, 0.0, 0.0)
        m.nodes[2] = Node(2, 2.0, 0.0)
        e = TrussElement2D(id=1, node_i=1, node_j=2, E=E, A=A,
                           alpha=1.2e-5)
        e.member_loads.append(TrussTemperatureLoad(delta_T=40.0))
        m.elements.append(e)
        m.supports[1] = Support(1, True, True, False)
        m.supports[2] = Support(2, False, True, False)
        return m

    r2 = run_analysis(build(), verbose=False)
    m3 = build()
    m3.supports[1] = Support(1, True, True, False, uz=True)
    m3.supports[2] = Support(2, False, True, False, uz=True)
    m3.force_3d = True
    r3 = run_analysis(m3, verbose=False)
    assert r2.status == r3.status == "ok"
    assert _disp(r3, 2, "ux") == pytest.approx(_disp(r2, 2, "ux"),
                                               rel=1e-12)


# ── detection / promotion mechanics ────────────────────────────


def test_model_is_3d_detection_triggers():
    m = _portal_frame_2d()
    assert not model_is_3d(m)
    m.nodes[5] = Node(5, 1.0, 1.0, 2.0)
    assert model_is_3d(m)

    m = _portal_frame_2d()
    m.supports[1] = Support(1, True, True, True, uz=True)
    assert model_is_3d(m)

    m = _portal_frame_2d()
    m.nodal_loads.append(NodalLoad(3, fz=1.0))
    assert model_is_3d(m)

    m = _portal_frame_2d()
    m.elements[1].member_loads.append(
        UniformDistributedLoad(wy=0.0, wz=-1.0))
    assert model_is_3d(m)


def test_promotion_maps_section_properties():
    m = _portal_frame_2d()
    elem = m.elements[0]
    p = promote_element_to_3d(elem, m)
    assert isinstance(p, FrameElement3D)
    assert p.Iz == elem.I and p.Iy == elem.I
    assert p.J == pytest.approx(2 * elem.I)  # polar fallback
    assert p.G == pytest.approx(elem.E / 2.0)  # ν = 0 identity
    assert p.member_loads is elem.member_loads


def test_promotion_rejects_rigid_offsets_in_3d():
    m = _portal_frame_2d()
    m.elements[0].offset_i = 0.3
    m.force_3d = True
    r = run_analysis(m, verbose=False)
    assert r.status == "error"
    assert any("rigid end offsets" in w for w in r.warnings)


def test_truss_promotion():
    m = StructuralModel()
    t = TrussElement2D(id=1, node_i=1, node_j=2, E=E, A=A)
    p = promote_element_to_3d(t, m)
    assert isinstance(p, TrussElement3D)
    assert p.E == E and p.A == A


def test_prepare_solve_elements_2d_passthrough():
    m = _portal_frame_2d()
    is_3d, elems = prepare_solve_elements(m)
    assert not is_3d
    assert elems[0] is m.elements[0]


# ── 3D DOF manager behaviour ───────────────────────────────────


def test_dof_manager_3d_suppresses_rotations_at_truss_nodes():
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 1.0, 1.0)
    m.elements.append(TrussElement3D(id=1, node_i=1, node_j=2, E=E, A=A))
    m.supports[1] = Support(1, True, True, False, uz=True)
    dofs = DofManager.from_model(m)
    assert dofs.is_3d
    for nid in (1, 2):
        for rot in ("rx", "ry", "rz"):
            assert dofs.active_map[nid][rot] is None
        for tr in ("ux", "uy", "uz"):
            assert dofs.active_map[nid][tr] is not None


def test_dof_manager_3d_released_end_suppresses_rz_only():
    m = _cantilever_model(release_j=True)
    dofs = DofManager.from_model(m)
    assert dofs.is_3d
    assert dofs.active_map[2]["rz"] is None
    assert dofs.active_map[2]["rx"] is not None
    assert dofs.active_map[2]["ry"] is not None


# ── 3D support settlement ──────────────────────────────────────


def test_3d_settlement_rigid_translation():
    m = _cantilever_model()
    delta = -0.02
    m.supports[1] = Support(1, True, True, True,
                            uz=True, rx=True, ry=True,
                            settle_uz=delta)
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    assert _disp(r, 2, "uz") == pytest.approx(delta, rel=1e-12)
    np.testing.assert_allclose(r.member_results[1]["f_local"],
                               np.zeros(12), atol=1e-9)


# ── load-type guards on 3D elements ────────────────────────────


def test_3d_frame_rejects_truss_temperature():
    m = _cantilever_model()
    m.elements[0].member_loads.append(TrussTemperatureLoad(delta_T=10.0))
    with pytest.raises(TypeError, match="TrussTemperatureLoad"):
        m.elements[0].local_consistent_load(m.nodes)


def test_3d_truss_rejects_transverse_loads():
    t = TrussElement3D(id=1, node_i=1, node_j=2, E=E, A=A)
    t.member_loads.append(UniformDistributedLoad(wy=-5.0))
    nodes = {1: Node(1, 0, 0, 0), 2: Node(2, 1, 0, 0)}
    with pytest.raises(TypeError, match="transverse"):
        t.local_consistent_load(nodes)
