"""Regression test for the canvas internal-force diagram functions.

The q2a/q2b validation cases used elsewhere in the suite exercise distributed
loads and end moments but not mid-span point loads. Gemini's PR review pointed
out that, for the diagram functions in :mod:`structural_analysis.gui_qt.canvas`,
differentiating the moment expression with respect to x gives a point-load
contribution with the **opposite sign** of the point-load term in the shear
expression — i.e. ``dM/dx ≠ V`` once any in-span point load is present.

This test pins down the expected behaviour on a simply-supported beam with a
single mid-span point load (textbook closed-form available), and asserts that:

* end moments are zero (pin + roller);
* mid-span moment magnitude equals ``P·L/4``;
* shear magnitude in each half equals ``P/2`` and the two halves have
  opposite signs;
* the differential identity ``dM/dx = V`` holds at sampled interior points
  in each half (so the point-load sign convention agrees across the two
  functions).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.model import (
    StructuralModel, Node, Support, NodalLoad, PointLoad,
)
from structural_analysis.element import FrameElement2D
from structural_analysis.main import run_analysis
from structural_analysis.gui_qt.canvas import _diagram_ordinates


def _simply_supported_beam_central_point_load(P: float = 10.0, L: float = 10.0):
    """10 m horizontal beam, pin at left, roller at right, P (positive) acting
    downward at mid-span via a member-level PointLoad (py = -P)."""
    E, A, I = 200_000.0, 0.02, 0.08
    m = StructuralModel(title="SS beam, central point load")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    elem = FrameElement2D(1, 1, 2, E=E, A=A, I=I)
    elem.member_loads.append(PointLoad(py=-P, a=L / 2.0))
    m.elements = [elem]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=False),
        2: Support(2, ux=False, uy=True, rz=False),
    }
    m.nodal_loads = []
    return m, P, L


def _diagrams(model, kind, n_samples=101):
    """Solve the model and return (xs, ys) from _diagram_ordinates for the
    first element, for the requested kind ('shear' or 'moment')."""
    r = run_analysis(model, verbose=False)
    assert r.status == "ok"
    elem = model.elements[0]
    ni = model.nodes[elem.node_i]
    nj = model.nodes[elem.node_j]
    f_local = r.member_results[elem.id]["f_local"]
    return _diagram_ordinates(elem, ni, nj, f_local, kind, n_samples=n_samples)


def test_simply_supported_beam_central_point_load_endpoints_and_peaks():
    model, P, L = _simply_supported_beam_central_point_load()

    xs_m, ys_m = _diagrams(model, "moment", n_samples=101)
    assert xs_m is not None and ys_m is not None

    # Pin + roller: no end moments.
    assert abs(ys_m[0]) < 1e-6
    assert abs(ys_m[-1]) < 1e-6

    # Mid-span moment magnitude equals P·L/4 = 25.0 for P=10, L=10.
    mid_idx = (len(xs_m) - 1) // 2
    assert abs(abs(ys_m[mid_idx]) - P * L / 4.0) < 1e-3

    xs_v, ys_v = _diagrams(model, "shear", n_samples=101)
    quarter = (len(xs_v) - 1) // 4
    three_quarter = 3 * (len(xs_v) - 1) // 4

    # |V| = P/2 = 5 in each half; opposite signs across the load.
    assert abs(abs(ys_v[quarter]) - P / 2.0) < 1e-6
    assert abs(abs(ys_v[three_quarter]) - P / 2.0) < 1e-6
    assert ys_v[quarter] * ys_v[three_quarter] < 0.0


def test_simply_supported_beam_central_point_load_dM_dx_equals_V():
    model, P, L = _simply_supported_beam_central_point_load()

    n = 101
    xs_m, ys_m = _diagrams(model, "moment", n_samples=n)
    xs_v, ys_v = _diagrams(model, "shear", n_samples=n)
    dx = xs_m[1] - xs_m[0]

    # Sample indices well away from x = 0, x = L, and x = L/2 (the load
    # discontinuity) so a central difference is well-defined.
    n_quarter = (n - 1) // 4
    for i in (n_quarter // 2, n_quarter, n_quarter + n_quarter // 2,
              n - 1 - n_quarter - n_quarter // 2,
              n - 1 - n_quarter, n - 1 - n_quarter // 2):
        dmdx = (ys_m[i + 1] - ys_m[i - 1]) / (2.0 * dx)
        assert abs(dmdx - ys_v[i]) < 1e-6, (
            f"dM/dx ({dmdx}) != V ({ys_v[i]}) at x={xs_m[i]:.3f} — "
            f"point-load sign is inconsistent between shear and moment."
        )
