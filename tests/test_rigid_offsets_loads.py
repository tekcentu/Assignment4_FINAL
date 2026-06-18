"""Rigid end offsets — member-load handling and diagram conventions.

Load semantics (v0.41 — UDL full-length fix):

* a UDL is applied over the FULL analytical member, including the rigid
  end zones: ``ΣR = w · L_total`` (the rigid zones carry their share of
  the load straight to their joints — see
  ``FrameElement2D.local_consistent_load``);
* point / thermal member loads still act on the FLEXIBLE span only;
  point-load station ``a`` stays measured from analytical node i and
  must land inside the flexible span — a load in a rigid zone is
  rejected, never silently relocated;
* self-weight is still applied over the flexible span (documented;
  out of scope for the UDL fix);
* diagrams are SAMPLED on the flexible span (display footprint
  unchanged), but the values integrate the UDL from x = 0 so they are
  consistent with the full-length load; ``dM/dx = V`` holds.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.model import (
    StructuralModel, Node, Support, UniformDistributedLoad, PointLoad, FrameTemperatureLoad,
)
from structural_analysis.element import FrameElement2D
from structural_analysis.main import run_analysis
from structural_analysis.gui_qt.element_graphics import (
    sample_internal_force, internal_force_at, diagram_domain,
)

E, A, I = 200_000.0, 0.02, 0.08


def _ss_beam(L=6.0, *, offset_i=0.0, offset_j=0.0):
    m = StructuralModel(title="ss offsets")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    m.elements = [FrameElement2D(
        1, 1, 2, E=E, A=A, I=I, offset_i=offset_i, offset_j=offset_j,
    )]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=False),
        2: Support(2, ux=False, uy=True, rz=False),
    }
    m.nodal_loads = []
    return m


# ── UDL on the flexible span ─────────────────────────────────────────────


def test_udl_reactions_equal_full_member_total():
    """A UDL acts over the FULL analytical member (incl. rigid zones) ⇒
    ΣR = w·L_total, split evenly on the symmetric configuration. (This
    replaces the pre-fix ``…equal_flexible_span_total`` which encoded the
    old ΣR = w·L_flex behaviour.)"""
    L, e_off, w = 6.0, 1.0, 10.0
    m = _ss_beam(L, offset_i=e_off, offset_j=e_off)
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    total = sum(rx.get("uy", 0.0) for rx in r.reactions.values())
    assert total == pytest.approx(w * L, rel=1e-9)
    assert r.reactions[1]["uy"] == pytest.approx(w * L / 2, rel=1e-9)


def test_udl_midspan_moment_static_equivalent():
    """Supports at the joints, UDL w over the FULL member [0, L]: this is
    a plain simply-supported beam, so the midspan moment is the textbook
    w·L²/8 (the rigid zones don't change the support reactions of a
    determinate beam)."""
    L, e_off, w = 6.0, 1.0, 10.0
    m = _ss_beam(L, offset_i=e_off, offset_j=e_off)
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    r = run_analysis(m, verbose=False)
    elem = m.elements[0]
    ni, nj = m.nodes[1], m.nodes[2]
    f_local = r.member_results[1]["f_local"]
    m_mid = internal_force_at(elem, ni, nj, f_local, "moment", L / 2)
    assert m_mid == pytest.approx(w * L ** 2 / 8.0, rel=1e-9)


# ── point loads ──────────────────────────────────────────────────────────


def test_point_load_inside_rigid_zone_rejected_by_solver():
    L = 6.0
    m = _ss_beam(L, offset_i=1.0)
    m.elements[0].member_loads.append(PointLoad(py=-10.0, a=0.5))
    with pytest.raises(ValueError, match="rigid end zone"):
        m.elements[0].local_consistent_load(m.nodes)


def test_point_load_on_flexible_span_static_equivalent():
    """Central point load with symmetric offsets: reactions P/2 at the
    joints, midspan moment P·L/4 (lever measured from the JOINT — the
    rigid arms transfer the support reactions without bending)."""
    L, e_off, P = 6.0, 1.0, 10.0
    m = _ss_beam(L, offset_i=e_off, offset_j=e_off)
    m.elements[0].member_loads.append(PointLoad(py=-P, a=L / 2))
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    assert r.reactions[1]["uy"] == pytest.approx(P / 2, rel=1e-9)
    elem = m.elements[0]
    f_local = r.member_results[1]["f_local"]
    m_mid = internal_force_at(
        elem, m.nodes[1], m.nodes[2], f_local, "moment", L / 2,
    )
    assert m_mid == pytest.approx(P * L / 4, rel=1e-6)


def test_point_load_station_measured_from_node_i():
    """a is from node i, NOT from the offset face: a load at a=2.0 with
    offset_i=1.0 sits 1.0 m into the flexible span — verify via the
    shear jump location in the sampled diagram."""
    L, e_off, P = 6.0, 1.0, 10.0
    m = _ss_beam(L, offset_i=e_off)
    m.elements[0].member_loads.append(PointLoad(py=-P, a=2.0))
    r = run_analysis(m, verbose=False)
    elem = m.elements[0]
    f_local = r.member_results[1]["f_local"]
    v_before = internal_force_at(
        elem, m.nodes[1], m.nodes[2], f_local, "shear", 1.99,
    )
    v_after = internal_force_at(
        elem, m.nodes[1], m.nodes[2], f_local, "shear", 2.01,
    )
    assert v_before - v_after == pytest.approx(P, rel=1e-6)


# ── thermal ──────────────────────────────────────────────────────────────


def test_thermal_axial_force_with_offsets():
    """Fixed-fixed bar, uniform heating: rigid zones are axially rigid,
    so the restrained axial force stays N = E·A·α·ΔT."""
    L, dT, alpha = 6.0, 30.0, 1.2e-5
    m = StructuralModel(title="thermal offsets")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    m.elements = [FrameElement2D(
        1, 1, 2, E=E, A=A, I=I, alpha=alpha, depth=0.2,
        offset_i=0.8, offset_j=0.5,
    )]
    m.elements[0].member_loads.append(
        FrameTemperatureLoad(t_top=dT, t_bottom=dT))
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=True),
        2: Support(2, ux=True, uy=True, rz=True),
    }
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    N_expected = E * A * alpha * dT
    f_local = r.member_results[1]["f_local"]
    assert abs(f_local[0]) == pytest.approx(N_expected, rel=1e-9)


# ── self-weight ──────────────────────────────────────────────────────────


def test_self_weight_total_uses_flexible_span():
    from structural_analysis.model import STANDARD_GRAVITY
    L, e_off, rho = 6.0, 1.0, 7850.0
    m = _ss_beam(L, offset_i=e_off, offset_j=e_off)
    m.elements[0].rho = rho
    m.include_self_weight = True
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    Lf = L - 2 * e_off
    w_self = rho * A * STANDARD_GRAVITY / 1000.0
    total = sum(rx.get("uy", 0.0) for rx in r.reactions.values())
    assert total == pytest.approx(w_self * Lf, rel=1e-9)


# ── diagrams ─────────────────────────────────────────────────────────────


def test_diagram_domain_is_flexible_span():
    L, ei, ej = 6.0, 1.0, 0.5
    m = _ss_beam(L, offset_i=ei, offset_j=ej)
    elem = m.elements[0]
    x0, x1 = diagram_domain(elem, m.nodes[1], m.nodes[2])
    assert x0 == pytest.approx(ei)
    assert x1 == pytest.approx(L - ej)


def test_sampled_stations_cover_flexible_span_only():
    L, ei, ej, w = 6.0, 1.0, 0.5, 10.0
    m = _ss_beam(L, offset_i=ei, offset_j=ej)
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    r = run_analysis(m, verbose=False)
    elem = m.elements[0]
    f_local = r.member_results[1]["f_local"]
    xs, ys = sample_internal_force(
        elem, m.nodes[1], m.nodes[2], f_local, "moment", n_samples=21,
    )
    assert xs[0] == pytest.approx(ei)
    assert xs[-1] == pytest.approx(L - ej)
    # No station inside a rigid zone.
    assert all(ei - 1e-9 <= x <= L - ej + 1e-9 for x in xs)


def test_dM_dx_equals_V_on_flexible_span_with_offsets():
    L, ei, ej, w = 8.0, 1.0, 0.5, 6.0
    m = _ss_beam(L, offset_i=ei, offset_j=ej)
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    r = run_analysis(m, verbose=False)
    elem = m.elements[0]
    f_local = r.member_results[1]["f_local"]
    n = 201
    xs_m, ms = sample_internal_force(
        elem, m.nodes[1], m.nodes[2], f_local, "moment", n_samples=n)
    xs_v, vs = sample_internal_force(
        elem, m.nodes[1], m.nodes[2], f_local, "shear", n_samples=n)
    dx = xs_m[1] - xs_m[0]
    for i in range(10, n - 10, 20):
        dmdx = (ms[i + 1] - ms[i - 1]) / (2 * dx)
        assert dmdx == pytest.approx(vs[i], abs=1e-6)


def test_face_moment_includes_rigid_zone_udl():
    """With the full-length UDL the i-face moment is
    −M_i + V_i·e_i + ½·w_y·e_i² — the rigid zone now carries its share of
    the distributed load, so the reconstruction integrates the UDL from
    x = 0 (this replaces the pre-fix ``…carries_linearly_from_joint``
    which assumed the rigid zone was unloaded)."""
    L, ei, w = 6.0, 1.0, 10.0
    m = _ss_beam(L, offset_i=ei)
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    r = run_analysis(m, verbose=False)
    elem = m.elements[0]
    f_local = r.member_results[1]["f_local"]
    V_i = float(f_local[1])
    M_i = float(f_local[2])
    wy = -w   # local +y intensity for a downward UDL on a horizontal beam
    m_face = internal_force_at(
        elem, m.nodes[1], m.nodes[2], f_local, "moment", ei,
    )
    assert m_face == pytest.approx(
        -M_i + V_i * ei + 0.5 * wy * ei ** 2, abs=1e-9)


def test_zero_offset_diagrams_unchanged():
    """Without offsets the domain is [0, L] and the formulas reduce to
    the legacy ones — endpoint stations at exactly 0 and L."""
    L, w = 6.0, 10.0
    m = _ss_beam(L)
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    r = run_analysis(m, verbose=False)
    elem = m.elements[0]
    f_local = r.member_results[1]["f_local"]
    xs, ys = sample_internal_force(
        elem, m.nodes[1], m.nodes[2], f_local, "moment", n_samples=7,
    )
    assert xs[0] == 0.0
    assert xs[-1] == pytest.approx(L)
    mid = (len(ys) - 1) // 2
    assert ys[mid] == pytest.approx(w * L**2 / 8, rel=1e-9)
