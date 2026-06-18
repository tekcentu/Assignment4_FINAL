"""Rigid-offset UDL full-length load-vector mechanics (v0.41).

Pins the fix that a uniform distributed load on a frame member with
rigid end offsets contributes over the FULL analytical member length —
not only the flexible span — with the rigid-zone portions transferred to
the joints as direct force/moment resultants.

These tests are the merge-blocker checks requested in the review:
force equilibrium, moment equilibrium, correct local→global transform
(proved by an inclined member), no-offset regression, axial + transverse
components, load cases / combinations, and diagram/station consistency.

Nothing here touches solver local axes, transformation matrices, member
load projection, file format, or user-facing load input semantics — the
only behaviour change is that the equivalent nodal load vector now spans
the full member.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from structural_analysis.model import (
    StructuralModel, Node, Support, UniformDistributedLoad, LoadCase,
    LoadCombination,
)
from structural_analysis.element import FrameElement2D
from structural_analysis.main import run_analysis, run_multi_case_analysis
from structural_analysis.gui_qt.element_graphics import (
    sample_internal_force, internal_force_at,
)

E, A, I = 200_000.0, 0.02, 0.08  # noqa: E741  (I = moment of inertia)


def _ss_beam(L, *, offset_i=0.0, offset_j=0.0):
    m = StructuralModel(title="udl full-length")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    m.elements = [FrameElement2D(
        1, 1, 2, E=E, A=A, I=I, offset_i=offset_i, offset_j=offset_j)]
    m.supports = {1: Support(1, ux=True, uy=True, rz=False),
                  2: Support(2, ux=False, uy=True, rz=False)}
    m.nodal_loads = []
    return m


# ── Test 1 — horizontal beam, asymmetric offsets ────────────────────────


def test_horizontal_beam_asymmetric_offsets_full_length_reactions():
    """L=10, offset_i=3, offset_j=1, downward UDL w=10 over the full
    member → ΣR = w·L_total = 100, and by load symmetry (uniform over
    [0,10], supports at the joints) R_left = R_right = 50. This checks
    force AND moment equilibrium simultaneously."""
    L, ei, ej, w = 10.0, 3.0, 1.0, 10.0
    m = _ss_beam(L, offset_i=ei, offset_j=ej)
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    R_left = r.reactions[1]["uy"]
    R_right = r.reactions[2]["uy"]
    assert R_left + R_right == pytest.approx(w * L, rel=1e-9)   # ΣF
    assert R_right * L == pytest.approx((w * L) * (L / 2), rel=1e-9)  # ΣM
    assert R_left == pytest.approx(50.0, rel=1e-9)
    assert R_right == pytest.approx(50.0, rel=1e-9)


def test_local_load_vector_force_and_moment_equilibrium():
    """The element local joint load vector itself is statically
    equivalent to the full-member UDL: ΣFy = w·L_total and
    ΣM_about_i = w·L_total²/2."""
    L, ei, ej, w = 10.0, 3.0, 2.0, 10.0
    m = _ss_beam(L, offset_i=ei, offset_j=ej)
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    p = m.elements[0].local_consistent_load(m.nodes)
    wy = -w
    assert p[1] + p[4] == pytest.approx(wy * L, abs=1e-9)          # ΣFy
    assert p[2] + p[5] + L * p[4] == pytest.approx(
        wy * L ** 2 / 2.0, abs=1e-6)                                # ΣM_i


# ── Test 2 — inclined member, global equilibrium (transform proof) ──────


def test_inclined_member_with_offsets_global_equilibrium():
    """A 45° fixed cantilever with non-zero offsets under a GLOBAL
    vertical UDL must satisfy global equilibrium. This FAILS if p_rigid
    is left unrotated or double-rotated — the world-frame check is the
    transform-correctness proof for inclined members."""
    ang = math.radians(45.0)
    L = 8.0
    x2, y2 = L * math.cos(ang), L * math.sin(ang)
    w = 10.0
    m = StructuralModel(title="inclined offsets")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, x2, y2)}
    m.elements = [FrameElement2D(
        1, 1, 2, E=E, A=A, I=I, offset_i=2.0, offset_j=1.5)]
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-w, coord_system="global"))
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}  # cantilever
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    R = r.reactions[1]
    total_Fy = w * L                       # global vertical, full length
    assert R["uy"] == pytest.approx(total_Fy, abs=1e-6)   # ΣFy
    assert R["ux"] == pytest.approx(0.0, abs=1e-6)        # ΣFx
    # ΣM about node 1: reaction moment balances the load moment. The
    # global-vertical UDL resultant (−w·L) acts at the member midpoint.
    x_centroid = x2 / 2.0
    m_load = (-total_Fy) * x_centroid
    assert R["rz"] + m_load == pytest.approx(0.0, abs=1e-6)


def test_inclined_member_30deg_global_equilibrium():
    """Same check at 30° (different cos/sin split) — guards against a
    transform that only happens to work at 45°."""
    ang = math.radians(30.0)
    L = 9.0
    x2, y2 = L * math.cos(ang), L * math.sin(ang)
    w = 7.5
    m = StructuralModel(title="inclined 30")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, x2, y2)}
    m.elements = [FrameElement2D(
        1, 1, 2, E=E, A=A, I=I, offset_i=1.0, offset_j=2.0)]
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-w, coord_system="global"))
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    r = run_analysis(m, verbose=False)
    R = r.reactions[1]
    assert R["uy"] == pytest.approx(w * L, abs=1e-6)
    assert R["ux"] == pytest.approx(0.0, abs=1e-6)
    assert R["rz"] + (-w * L) * (x2 / 2.0) == pytest.approx(0.0, abs=1e-6)


# ── Test 3 — no-offset regression ───────────────────────────────────────


def test_no_offset_udl_unchanged():
    """offset_i = offset_j = 0 must be byte-identical to the legacy
    behaviour: ΣR = w·L, R = w·L/2 each, midspan M = w·L²/8."""
    L, w = 6.0, 10.0
    m = _ss_beam(L)
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    r = run_analysis(m, verbose=False)
    assert r.reactions[1]["uy"] == pytest.approx(w * L / 2, rel=1e-12)
    assert r.reactions[2]["uy"] == pytest.approx(w * L / 2, rel=1e-12)
    # Local vector identical to the hand formula.
    p = m.elements[0].local_consistent_load(m.nodes)
    expected = np.array([0, -w*L/2, -w*L**2/12, 0, -w*L/2, w*L**2/12])
    assert np.allclose(p, expected, atol=1e-12)


def test_no_offset_local_vector_matches_zero_offset_limit():
    """A vanishing offset converges to the no-offset vector (continuity:
    p_rigid → 0 as offsets → 0)."""
    L, w = 6.0, 10.0
    m0 = _ss_beam(L)
    m0.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    p0 = m0.elements[0].local_consistent_load(m0.nodes)
    m_eps = _ss_beam(L, offset_i=1e-7, offset_j=1e-7)
    m_eps.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    p_eps = m_eps.elements[0].local_consistent_load(m_eps.nodes)
    assert np.allclose(p0, p_eps, atol=1e-4)


# ── Test 4 — axial and transverse UDL components ────────────────────────


def test_axial_udl_with_offsets_full_length():
    """A local-axis axial UDL (wx) on a horizontal member with offsets
    contributes over the full length: a fixed-fixed bar develops the
    full restrained axial resultant ΣN-input = wx·L_total."""
    L, ei, ej, wx = 8.0, 2.0, 1.0, 5.0
    m = StructuralModel(title="axial udl")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    m.elements = [FrameElement2D(
        1, 1, 2, E=E, A=A, I=I, offset_i=ei, offset_j=ej)]
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=0.0, wx=wx, coord_system="local"))
    m.supports = {1: Support(1, ux=True, uy=True, rz=True),
                  2: Support(2, ux=True, uy=True, rz=True)}
    r = run_analysis(m, verbose=False)
    Rx_total = r.reactions[1]["ux"] + r.reactions[2]["ux"]
    assert Rx_total == pytest.approx(-wx * L, rel=1e-9)
    # local-vector axial balance
    p = m.elements[0].local_consistent_load(m.nodes)
    assert p[0] + p[3] == pytest.approx(wx * L, abs=1e-9)


def test_transverse_udl_with_offsets_full_length():
    """Transverse UDL (wy) with offsets — force + moment equilibrium of
    the local vector (the workhorse case)."""
    L, ei, ej, wy = 8.0, 2.0, 1.0, -12.0
    m = _ss_beam(L, offset_i=ei, offset_j=ej)
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=wy))
    p = m.elements[0].local_consistent_load(m.nodes)
    assert p[1] + p[4] == pytest.approx(wy * L, abs=1e-9)
    assert p[2] + p[5] + L * p[4] == pytest.approx(wy * L ** 2 / 2, abs=1e-6)


# ── Test 5 — active load cases and combinations ─────────────────────────


def test_rigid_offset_udl_load_cases_and_combination():
    """DEFAULT UDL and LIVE UDL on the same offset member, plus a
    1.0·DEFAULT + 1.0·LIVE combination. Each case includes only its own
    full-member UDL; the combination is the post-processing
    superposition of the solved cases."""
    L, ei, ej = 10.0, 3.0, 1.0
    wD, wL = 10.0, 6.0
    m = _ss_beam(L, offset_i=ei, offset_j=ej)
    e = m.elements[0]
    e.member_loads.append(UniformDistributedLoad(wy=-wD, load_case="DEFAULT"))
    e.member_loads.append(UniformDistributedLoad(wy=-wL, load_case="LIVE"))
    m.load_cases["DEFAULT"] = LoadCase(name="DEFAULT")
    m.load_cases["LIVE"] = LoadCase(name="LIVE")
    m.load_combinations["COMB"] = LoadCombination(
        name="COMB", terms={"DEFAULT": 1.0, "LIVE": 1.0})
    mc = run_multi_case_analysis(m, verbose=False, cases=["DEFAULT", "LIVE"])

    rD = mc.cases["DEFAULT"]
    rL = mc.cases["LIVE"]
    # Each case: full-member total of its own UDL only.
    assert sum(v["uy"] for v in rD.reactions.values()) == pytest.approx(
        wD * L, rel=1e-9)
    assert sum(v["uy"] for v in rL.reactions.values()) == pytest.approx(
        wL * L, rel=1e-9)

    comb = mc.combination({"DEFAULT": 1.0, "LIVE": 1.0}, name="COMB")
    # Combination == station-by-station superposition of the two cases.
    for nid in comb.reactions:
        assert comb.reactions[nid]["uy"] == pytest.approx(
            rD.reactions[nid]["uy"] + rL.reactions[nid]["uy"], rel=1e-9)
    assert sum(v["uy"] for v in comb.reactions.values()) == pytest.approx(
        (wD + wL) * L, rel=1e-9)


# ── Test 6 — diagram / station-export consistency ───────────────────────


def test_diagram_consistent_with_full_member_udl():
    """The sampled flexible-span diagram values match the textbook
    simply-supported beam under the full-length UDL (supports at the
    joints): V(x) = w·L/2 − w·x, M(x) = w·L/2·x − w·x²/2. The rigid-zone
    load is reflected in the values (integrated from x = 0), not omitted."""
    L, ei, ej, w = 10.0, 3.0, 1.0, 10.0
    m = _ss_beam(L, offset_i=ei, offset_j=ej)
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    r = run_analysis(m, verbose=False)
    elem = m.elements[0]
    ni, nj = m.nodes[1], m.nodes[2]
    f = list(r.member_results[1]["f_local"])
    R = w * L / 2.0
    for x in (ei, 4.0, 5.0, 6.0, L - ej):     # stations on the flexible span
        v = internal_force_at(elem, ni, nj, f, "shear", x)
        mom = internal_force_at(elem, ni, nj, f, "moment", x)
        assert v == pytest.approx(R - w * x, abs=1e-6)
        assert mom == pytest.approx(R * x - w * x ** 2 / 2.0, abs=1e-6)


def test_station_export_matches_sample_with_offsets():
    """Station export and ``sample_internal_force`` use the same path, so
    the moment column equals the helper sampling on the flexible span."""
    L, ei, ej, w = 10.0, 3.0, 1.0, 10.0
    m = _ss_beam(L, offset_i=ei, offset_j=ej)
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-w))
    r = run_analysis(m, verbose=False)
    elem = m.elements[0]
    ni, nj = m.nodes[1], m.nodes[2]
    f = list(r.member_results[1]["f_local"])
    xs, ms = sample_internal_force(elem, ni, nj, f, "moment", n_samples=11)
    R = w * L / 2.0
    for x, mom in zip(xs, ms):
        assert mom == pytest.approx(R * x - w * x ** 2 / 2.0, abs=1e-6)
