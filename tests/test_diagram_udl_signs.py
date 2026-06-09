"""Regression tests for V/M diagram signs under uniform distributed loads.

The existing ``test_diagram_signs.py`` only exercises a mid-span point load.
The diagram-reconstruction code in ``gui_qt/element_graphics.py`` historically
carried the wrong sign on the UDL terms inside ``shear()`` and ``moment()``:
a simply-supported 6 m beam with w = -10 kN/m produced V = +30 → +90 instead
of the textbook V = +30 → -30 (crossing zero at midspan).

These tests pin the correct UDL behaviour so the regression cannot return.
The chosen convention is ``dM/dx = V`` (already pinned by
``test_diagram_signs.py``), with ``V_i, M_i`` taken straight from the solver's
``q_local`` recovery (``+y_local`` shear at the i-end, CCW moment at the i-end).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.model import (
    StructuralModel, Node, Support, UniformDistributedLoad, PointLoad,
)
from structural_analysis.element import FrameElement2D
from structural_analysis.main import run_analysis
from structural_analysis.gui_qt.element_graphics import sample_internal_force


# ── Helpers ──────────────────────────────────────────────────────────────


def _beam(node_i_xy=(0.0, 0.0), node_j_xy=(6.0, 0.0),
          E=200_000.0, A=0.02, I=0.08):
    m = StructuralModel(title="UDL diagram test")
    m.nodes = {
        1: Node(1, node_i_xy[0], node_i_xy[1]),
        2: Node(2, node_j_xy[0], node_j_xy[1]),
    }
    e = FrameElement2D(1, 1, 2, E=E, A=A, I=I)
    m.elements = [e]
    m.nodal_loads = []
    return m, e


def _simply_supported(L=6.0, w=10.0):
    """Pin at left, roller at right; downward UDL w (so wy = -w)."""
    m, e = _beam(node_j_xy=(L, 0.0))
    e.member_loads.append(UniformDistributedLoad(wy=-w))
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=False),
        2: Support(2, ux=False, uy=True, rz=False),
    }
    return m, L, w


def _fixed_fixed(L=6.0, w=10.0):
    """Both ends fully fixed; downward UDL w."""
    m, e = _beam(node_j_xy=(L, 0.0))
    e.member_loads.append(UniformDistributedLoad(wy=-w))
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=True),
        2: Support(2, ux=True, uy=True, rz=True),
    }
    return m, L, w


def _solve_and_sample(model, kind, n_samples=21):
    r = run_analysis(model, verbose=False)
    assert r.status == "ok", r.status
    e = model.elements[0]
    ni = model.nodes[e.node_i]
    nj = model.nodes[e.node_j]
    f_local = r.member_results[e.id]["f_local"]
    xs, ys = sample_internal_force(e, ni, nj, f_local, kind, n_samples=n_samples)
    return xs, ys, f_local


# ── 1. Simply-supported UDL beam — closed form ───────────────────────────


def test_simply_supported_udl_shear_crosses_zero_at_midspan():
    """V is linear, |V| = wL/2 at ends with opposite signs; zero at midspan."""
    m, L, w = _simply_supported(L=6.0, w=10.0)
    xs, vs, _ = _solve_and_sample(m, "shear", n_samples=7)
    assert vs[0] == pytest.approx(+w * L / 2.0, abs=1e-6)
    assert vs[-1] == pytest.approx(-w * L / 2.0, abs=1e-6)
    mid = (len(vs) - 1) // 2
    assert vs[mid] == pytest.approx(0.0, abs=1e-6)
    # Strict monotonic decrease (linear).
    for a, b in zip(vs, vs[1:]):
        assert b < a + 1e-9


def test_simply_supported_udl_moment_zero_at_ends_peak_at_midspan():
    """M = 0 at pinned ends; M_max = wL²/8 sagging (positive in our convention)."""
    m, L, w = _simply_supported(L=6.0, w=10.0)
    xs, ms, _ = _solve_and_sample(m, "moment", n_samples=21)
    assert ms[0] == pytest.approx(0.0, abs=1e-6)
    assert ms[-1] == pytest.approx(0.0, abs=1e-6)
    mid = (len(ms) - 1) // 2
    assert ms[mid] == pytest.approx(w * L ** 2 / 8.0, rel=1e-6)
    # Parabolic and concave-down: midspan is the max.
    assert ms[mid] > ms[mid - 1] > 0
    assert ms[mid] > ms[mid + 1] > 0


# ── 2. Fixed-fixed UDL beam — end moments and shear span ─────────────────


def test_fixed_fixed_udl_shear_span_equals_wL():
    """V(0) - V(L) = w·L (total transverse load), with V crossing zero at midspan."""
    m, L, w = _fixed_fixed(L=6.0, w=10.0)
    xs, vs, _ = _solve_and_sample(m, "shear", n_samples=7)
    assert vs[0] - vs[-1] == pytest.approx(w * L, abs=1e-6)
    mid = (len(vs) - 1) // 2
    assert vs[mid] == pytest.approx(0.0, abs=1e-6)
    assert vs[0] == pytest.approx(+w * L / 2.0, abs=1e-6)
    assert vs[-1] == pytest.approx(-w * L / 2.0, abs=1e-6)


def test_fixed_fixed_udl_end_moments_textbook():
    """|M_end| = wL²/12 hogging; midspan = wL²/24 sagging."""
    m, L, w = _fixed_fixed(L=6.0, w=10.0)
    xs, ms, _ = _solve_and_sample(m, "moment", n_samples=21)
    m_end_expected = w * L ** 2 / 12.0
    m_mid_expected = w * L ** 2 / 24.0
    # End moments are hogging — negative in our convention.
    assert ms[0] == pytest.approx(-m_end_expected, rel=1e-5)
    assert ms[-1] == pytest.approx(-m_end_expected, rel=1e-5)
    # Midspan sagging, positive.
    mid = (len(ms) - 1) // 2
    assert ms[mid] == pytest.approx(+m_mid_expected, rel=1e-5)


# ── 3. SAP-like portal — top beam under UDL ──────────────────────────────


def test_portal_top_beam_shear_changes_sign_regression():
    """Two pinned columns + one top beam under downward UDL.

    Regression for the +30 → +90 bug: the top beam's shear MUST change sign
    across the span (V at i and V at j must have opposite signs and the
    diagram must contain a zero crossing).
    """
    L_beam = 6.0
    H = 4.0
    w = 10.0
    E, A, I = 200_000.0, 0.02, 0.08
    m = StructuralModel(title="Portal UDL regression")
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 0.0, H),
        3: Node(3, L_beam, H),
        4: Node(4, L_beam, 0.0),
    }
    col_l = FrameElement2D(1, 1, 2, E=E, A=A, I=I)
    beam = FrameElement2D(2, 2, 3, E=E, A=A, I=I)
    col_r = FrameElement2D(3, 3, 4, E=E, A=A, I=I)
    beam.member_loads.append(UniformDistributedLoad(wy=-w))
    m.elements = [col_l, beam, col_r]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=False),
        4: Support(4, ux=True, uy=True, rz=False),
    }
    m.nodal_loads = []

    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    ni = m.nodes[beam.node_i]
    nj = m.nodes[beam.node_j]
    f_local = r.member_results[beam.id]["f_local"]

    xs, vs = sample_internal_force(beam, ni, nj, f_local, "shear", n_samples=21)
    assert vs[0] * vs[-1] < 0.0, (
        f"top beam shear must change sign across span; got V_i={vs[0]}, V_j={vs[-1]} "
        "(regression: pre-fix the diagram went +30 → +90 without crossing zero)."
    )
    # And a zero crossing must exist among sampled points.
    sign_changes = sum(
        1 for a, b in zip(vs, vs[1:]) if a * b <= 0.0
    )
    assert sign_changes >= 1


# ── 4. Reversed i/j orientation — physical shape unchanged ───────────────


def test_simply_supported_udl_reversed_orientation_same_extrema():
    """Swap node coordinates so j is at x=0 and i is at x=L (member points in
    -x_global). The local frame rotates 180°, so V_local labels flip sign,
    but the **physical** extrema (|V_max|, |M_max|) and zero crossings must
    be identical."""
    # Forward orientation: i at (0,0), j at (6,0).
    m1, L, w = _simply_supported(L=6.0, w=10.0)
    _, vs_fwd, _ = _solve_and_sample(m1, "shear", n_samples=21)
    _, ms_fwd, _ = _solve_and_sample(m1, "moment", n_samples=21)

    # Reversed orientation: same physical beam, i at (6,0), j at (0,0).
    m2 = StructuralModel(title="reversed")
    m2.nodes = {1: Node(1, 6.0, 0.0), 2: Node(2, 0.0, 0.0)}
    e = FrameElement2D(1, 1, 2, E=200_000.0, A=0.02, I=0.08)
    e.member_loads.append(UniformDistributedLoad(wy=-w))
    m2.elements = [e]
    m2.supports = {
        1: Support(1, ux=False, uy=True, rz=False),  # roller at physical right (=i)
        2: Support(2, ux=True, uy=True, rz=False),   # pin at physical left (=j)
    }
    m2.nodal_loads = []
    _, vs_rev, _ = _solve_and_sample(m2, "shear", n_samples=21)
    _, ms_rev, _ = _solve_and_sample(m2, "moment", n_samples=21)

    # Magnitudes of extrema match.
    assert max(abs(v) for v in vs_rev) == pytest.approx(
        max(abs(v) for v in vs_fwd), abs=1e-6,
    )
    assert max(ms_rev) == pytest.approx(max(ms_fwd), rel=1e-6)
    # Both still cross zero in shear and stay sagging-positive in moment.
    assert any(a * b <= 0.0 for a, b in zip(vs_rev, vs_rev[1:]))
    assert min(ms_rev) == pytest.approx(0.0, abs=1e-6)


# ── 5. dM/dx = V holds with combined UDL + interior point load ───────────


def test_dM_dx_equals_V_with_udl_and_point_load():
    """Combined UDL + interior point load. Central-difference of M must match
    V at sampled interior points (away from the point-load discontinuity)."""
    L = 8.0
    w = 6.0
    P = 12.0
    m, e = _beam(node_j_xy=(L, 0.0))
    e.member_loads.append(UniformDistributedLoad(wy=-w))
    e.member_loads.append(PointLoad(py=-P, a=L / 2.0))
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=False),
        2: Support(2, ux=False, uy=True, rz=False),
    }

    n = 201
    xs_m, ms, _ = _solve_and_sample(m, "moment", n_samples=n)
    xs_v, vs, _ = _solve_and_sample(m, "shear", n_samples=n)
    dx = xs_m[1] - xs_m[0]

    # Stay clear of x=0, x=L, and x=L/2 (the point-load discontinuity).
    quarter = (n - 1) // 4
    for i in (quarter // 2, quarter, quarter + quarter // 2,
              n - 1 - quarter - quarter // 2,
              n - 1 - quarter, n - 1 - quarter // 2):
        dmdx = (ms[i + 1] - ms[i - 1]) / (2.0 * dx)
        assert abs(dmdx - vs[i]) < 1e-5, (
            f"dM/dx ({dmdx}) != V ({vs[i]}) at x={xs_m[i]:.3f} — "
            f"UDL terms in shear and moment are inconsistent."
        )


# ── 6. End-force-only diagrams unchanged when no member loads ────────────


def test_no_member_loads_diagrams_are_end_force_only():
    """With no member loads, V(x) ≡ V_i and M(x) is linear with slope V_i.

    This guards the back-compat path: the UDL term must drop out cleanly
    (both old and new formula reduce to V_i when w=0, but pin the property
    explicitly so the back-compat is regression-tested)."""
    L = 5.0
    m, e = _beam(node_j_xy=(L, 0.0))
    # End-moment-loaded cantilever-like setup, via nodal load at j.
    from structural_analysis.model import NodalLoad
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    m.nodal_loads = [NodalLoad(node_id=2, fy=-7.0)]
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    ni = m.nodes[1]
    nj = m.nodes[2]
    f_local = r.member_results[1]["f_local"]
    V_i = float(f_local[1])
    M_i = float(f_local[2])

    xs, vs = sample_internal_force(e, ni, nj, f_local, "shear", n_samples=11)
    xms, ms = sample_internal_force(e, ni, nj, f_local, "moment", n_samples=11)
    # V(x) ≡ V_i and M(x) = -M_i + V_i·x.
    for x, v in zip(xs, vs):
        assert v == pytest.approx(V_i, abs=1e-9)
    for x, mom in zip(xms, ms):
        assert mom == pytest.approx(-M_i + V_i * x, abs=1e-9)
