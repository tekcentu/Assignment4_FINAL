"""Regression tests for frame N/V/M reconstruction under member loads.

The shared reconstruction in ``gui_qt/element_graphics.py`` anchors the
diagram to the solved local end forces ``f_local`` (which the multi-case
layer filters per case and scales by combination factors) but historically
rebuilt the in-span load term from the live, unfactored, all-cases
``elem.member_loads``. For any combination or multi-case view the two were
inconsistent, so the parabola no longer matched its own end forces — e.g. a
DEAD-only view of a beam that also carries a LIVE UDL showed midspan M = 0
instead of wL²/8.

These tests pin the fix: callers pass the case-consistent span loads via
``effective_member_loads`` + the ``member_loads=`` kwarg, so the diagram and
station export agree with the displayed result.
"""

from __future__ import annotations

import pytest

from structural_analysis.model import (
    StructuralModel, Node, Support, UniformDistributedLoad, PointLoad,
    LoadCombination,
)
from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.main import run_analysis, run_multi_case_analysis
from structural_analysis.gui_qt.element_graphics import (
    sample_internal_force, internal_force_at, effective_member_loads,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _ss_beam(L=6.0):
    """Simply-supported (pin/roller) horizontal beam, no loads yet."""
    m = StructuralModel(title="nvm reconstruction")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    e = FrameElement2D(1, 1, 2, E=2.0e8, A=0.02, I=0.08)
    m.elements = [e]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=False),
        2: Support(2, ux=False, uy=True, rz=False),
    }
    m.nodal_loads = []
    return m, e, L


def _multi(m, cases):
    return run_multi_case_analysis(m, verbose=False, cases=cases)


# ── 1. Single full-span UDL: correct shape ───────────────────────────────


def test_single_udl_moment_and_shear_shape():
    m, e, L = _ss_beam()
    w = 10.0
    e.member_loads.append(UniformDistributedLoad(wy=-w))  # DEFAULT case
    r = run_analysis(m, verbose=False)
    f = r.member_results[1]["f_local"]
    ni, nj = m.nodes[1], m.nodes[2]

    xs, ms = sample_internal_force(e, ni, nj, f, "moment", n_samples=7)
    assert ms[0] == pytest.approx(0.0, abs=1e-6)
    assert ms[-1] == pytest.approx(0.0, abs=1e-6)
    assert ms[len(ms) // 2] == pytest.approx(w * L ** 2 / 8.0, rel=1e-6)

    _, vs = sample_internal_force(e, ni, nj, f, "shear", n_samples=7)
    assert vs[0] == pytest.approx(+w * L / 2.0, abs=1e-6)
    assert vs[-1] == pytest.approx(-w * L / 2.0, abs=1e-6)
    assert vs[len(vs) // 2] == pytest.approx(0.0, abs=1e-6)


# ── 2. Two UDLs == one summed UDL; factored combinations ─────────────────


def test_two_udls_equal_one_summed_udl_via_combination():
    """The regression that failed before the fix: DEAD-only must NOT
    include the LIVE UDL, and a factored combination must use factored
    member loads."""
    m, e, L = _ss_beam()
    e.member_loads.append(UniformDistributedLoad(wy=-10.0, load_case="DEAD"))
    e.member_loads.append(UniformDistributedLoad(wy=-10.0, load_case="LIVE"))
    mc = _multi(m, ["DEAD", "LIVE"])
    ni, nj = m.nodes[1], m.nodes[2]

    # DEAD-only combination: span load must be just the DEAD UDL.
    res_d = mc.combination({"DEAD": 1.0}, name="C_DEAD")
    eff_d = effective_member_loads(
        e, "C_DEAD",
        {"C_DEAD": LoadCombination(name="C_DEAD", terms={"DEAD": 1.0})},
    )
    _, ms_d = sample_internal_force(
        e, ni, nj, res_d.member_results[1]["f_local"], "moment",
        n_samples=7, member_loads=eff_d)
    assert ms_d[len(ms_d) // 2] == pytest.approx(10.0 * L ** 2 / 8.0, rel=1e-6)

    # Equivalent single UDL of w = 20 over the same beam: midspan must match
    # the SUM_ALL view (DEAD+LIVE, both factor 1).
    res_sum = mc.sum_all()
    _, ms_sum = sample_internal_force(
        e, ni, nj, res_sum.member_results[1]["f_local"], "moment",
        n_samples=7, member_loads=effective_member_loads(e, "SUM_ALL", {}))
    assert ms_sum[len(ms_sum) // 2] == pytest.approx(
        20.0 * L ** 2 / 8.0, rel=1e-6)

    # Factored 1.2D + 1.6L: w_eff = 28.
    combos = {"ULS": LoadCombination(
        name="ULS", terms={"DEAD": 1.2, "LIVE": 1.6})}
    res_u = mc.combination({"DEAD": 1.2, "LIVE": 1.6}, name="ULS")
    eff_u = effective_member_loads(e, "ULS", combos)
    _, ms_u = sample_internal_force(
        e, ni, nj, res_u.member_results[1]["f_local"], "moment",
        n_samples=7, member_loads=eff_u)
    w_eff = 1.2 * 10.0 + 1.6 * 10.0
    assert ms_u[len(ms_u) // 2] == pytest.approx(w_eff * L ** 2 / 8.0, rel=1e-6)


def test_dead_only_does_not_include_live_contribution_regression():
    """Direct guard on the reported symptom: without the fix the DEAD-only
    midspan moment collapsed to ~0 because the span load summed DEAD+LIVE
    while the end forces were DEAD-only."""
    m, e, L = _ss_beam()
    e.member_loads.append(UniformDistributedLoad(wy=-10.0, load_case="DEAD"))
    e.member_loads.append(UniformDistributedLoad(wy=-10.0, load_case="LIVE"))
    mc = _multi(m, ["DEAD", "LIVE"])
    ni, nj = m.nodes[1], m.nodes[2]
    combos = {"C_DEAD": LoadCombination(name="C_DEAD", terms={"DEAD": 1.0})}
    res_d = mc.combination({"DEAD": 1.0}, name="C_DEAD")
    f = res_d.member_results[1]["f_local"]

    # The buggy path (raw elem.member_loads = DEAD+LIVE) gives ~0 at midspan.
    _, ms_bug = sample_internal_force(e, ni, nj, f, "moment", n_samples=7)
    assert ms_bug[len(ms_bug) // 2] == pytest.approx(0.0, abs=1e-6)

    # The fixed path (case-consistent loads) gives the correct 45.
    eff = effective_member_loads(e, "C_DEAD", combos)
    _, ms_fix = sample_internal_force(
        e, ni, nj, f, "moment", n_samples=7, member_loads=eff)
    assert ms_fix[len(ms_fix) // 2] == pytest.approx(45.0, rel=1e-6)


# ── 3. Anchored to solved end forces (not zero) ──────────────────────────


def test_diagram_anchors_to_solved_end_forces_fixed_fixed():
    """Fixed-fixed UDL beam has non-zero end moments; the diagram must
    start at -M_i / V_i, never at zero."""
    m, e, L = _ss_beam()
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=True),
        2: Support(2, ux=True, uy=True, rz=True),
    }
    w = 12.0
    e.member_loads.append(UniformDistributedLoad(wy=-w))
    r = run_analysis(m, verbose=False)
    f = r.member_results[1]["f_local"]
    V_i, M_i = float(f[1]), float(f[2])
    ni, nj = m.nodes[1], m.nodes[2]

    m0 = internal_force_at(e, ni, nj, f, "moment", 0.0)
    v0 = internal_force_at(e, ni, nj, f, "shear", 0.0)
    assert M_i != pytest.approx(0.0, abs=1e-6)   # fixed end => real moment
    assert m0 == pytest.approx(-M_i, rel=1e-9)   # anchored, not zero
    assert v0 == pytest.approx(V_i, rel=1e-9)
    # Closed-form fixed-fixed: end moment = wL²/12, end shear = wL/2.
    assert abs(M_i) == pytest.approx(w * L ** 2 / 12.0, rel=1e-6)


# ── 4. Point load: shear jump + moment kink ──────────────────────────────


def test_point_load_creates_shear_jump_and_moment_kink():
    m, e, L = _ss_beam(L=10.0)
    P = 10.0
    a = L / 2.0
    e.member_loads.append(PointLoad(py=-P, a=a))
    r = run_analysis(m, verbose=False)
    f = r.member_results[1]["f_local"]
    ni, nj = m.nodes[1], m.nodes[2]

    # Shear just left vs just right of the load differs by P.
    v_left = internal_force_at(e, ni, nj, f, "shear", a - 1e-6)
    v_right = internal_force_at(e, ni, nj, f, "shear", a + 1e-6)
    assert (v_left - v_right) == pytest.approx(P, abs=1e-4)
    # Moment is continuous (kink, not jump) and peaks at the load = PL/4.
    m_mid = internal_force_at(e, ni, nj, f, "moment", a)
    assert m_mid == pytest.approx(P * L / 4.0, rel=1e-4)


# ── 5. Vertical-jump stations are duplicated and not removed ─────────────


def test_split_discontinuities_keeps_duplicate_x_at_point_load():
    m, e, L = _ss_beam(L=10.0)
    P = 10.0
    a = L / 2.0
    e.member_loads.append(PointLoad(py=-P, a=a))
    r = run_analysis(m, verbose=False)
    f = r.member_results[1]["f_local"]
    ni, nj = m.nodes[1], m.nodes[2]

    xs, vs = sample_internal_force(
        e, ni, nj, f, "shear", n_samples=11, split_discontinuities=True)
    # Two stations sit at x == a (left limit + right limit).
    at_a = [i for i, x in enumerate(xs) if x == pytest.approx(a, abs=1e-9)]
    assert len(at_a) == 2, f"expected duplicate stations at x={a}, got {at_a}"
    i, j = at_a
    assert abs(vs[i] - vs[j]) == pytest.approx(P, abs=1e-4)  # vertical jump

    # Without the opt-in the legacy single-station output is preserved.
    xs0, _ = sample_internal_force(e, ni, nj, f, "shear", n_samples=11)
    assert len(xs0) == 11


# (6. Station-export ⇄ canvas cross-check lives in tests/test_export_stations.py
#     where the PyQt6 offscreen harness fixtures are already set up.)


# ── 7. Truss stays axial-only ────────────────────────────────────────────


def test_truss_shear_moment_return_none():
    m = StructuralModel(title="truss")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    t = TrussElement2D(1, 1, 2, E=2e8, A=0.01)
    m.elements = [t]
    ni, nj = m.nodes[1], m.nodes[2]
    f = [5.0, 0.0, 0.0, -5.0, 0.0, 0.0]
    assert sample_internal_force(t, ni, nj, f, "shear")[0] is None
    assert sample_internal_force(t, ni, nj, f, "moment")[0] is None
    xs, ys = sample_internal_force(t, ni, nj, f, "axial", n_samples=5)
    assert xs is not None and len(xs) == 5


# ── 8. No member loads: unchanged (byte-compatible default path) ─────────


def test_no_member_load_default_path_unchanged():
    m, e, L = _ss_beam()
    m.nodal_loads = []
    # Pin/roller with a nodal load at node 2 instead of a member load.
    from structural_analysis.model import NodalLoad
    m.nodal_loads = [NodalLoad(node_id=2, fx=15.0)]
    r = run_analysis(m, verbose=False)
    f = r.member_results[1]["f_local"]
    ni, nj = m.nodes[1], m.nodes[2]
    # Passing the effective (empty) loads or omitting must be identical.
    eff = effective_member_loads(e, "DEFAULT", {})
    a, ya = sample_internal_force(e, ni, nj, f, "axial", n_samples=9)
    b, yb = sample_internal_force(
        e, ni, nj, f, "axial", n_samples=9, member_loads=eff)
    assert a == b and ya == yb
