"""Feature-coverage tests for the combined example_09/10/11 inputs.

The parametrized sweep in ``test_example_inputs.py`` already proves every
bundled example parses, static-solves, and (where it has mass) runs modal.
These tests additionally assert that the three new "combined" examples
actually *demonstrate the features they advertise* — predefined load
cases + combinations, rigid end offsets, diagonal bracing, and modal —
so a future edit that quietly strips a feature is caught.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.file_io import read_input_file
from structural_analysis.main import run_analysis, run_multi_case_analysis
from structural_analysis.modal import solve_modal

_INPUTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "inputs")


def _load(name: str):
    return read_input_file(os.path.join(_INPUTS, name))


def _has_offset(model) -> bool:
    return any(
        isinstance(e, FrameElement2D)
        and (e.offset_i > 0.0 or e.offset_j > 0.0)
        for e in model.elements
    )


def _has_truss_diagonal(model) -> bool:
    return any(isinstance(e, TrussElement2D) for e in model.elements)


def _has_density(model) -> bool:
    return any(e.rho > 0.0 for e in model.elements)


# ── example_09 — cases + combinations + bracing + modal ───────────────────


def test_example_09_has_cases_combos_bracing_and_modal():
    m = _load("example_09_braced_frame_load_combos_modal.txt")
    # ≥3 user load cases (DEAD/LIVE/WIND) plus the auto DEFAULT.
    assert {"DEAD", "LIVE", "WIND"} <= set(m.load_cases)
    assert len(m.load_combinations) >= 1
    assert _has_truss_diagonal(m), "example_09 should have diagonal TRUSS braces"
    assert _has_density(m), "example_09 should carry mass for modal"

    mc = run_multi_case_analysis(m, verbose=False)
    assert not mc.failed_cases, f"cases failed: {mc.failed_cases}"
    assert {"DEAD", "LIVE", "WIND"} <= set(mc.cases)
    # A predefined combination must compute from the solved cases.
    comb = mc.combination({"DEAD": 1.2, "LIVE": 1.6}, name="ULS")
    assert comb is not None

    r = solve_modal(m, n_modes=3)
    assert r.status == "ok"
    assert float(r.frequencies[0]) > 0.0


# ── example_10 — rigid offsets + cases + combination (static) ──────────────


def test_example_10_has_rigid_offsets_and_combo():
    m = _load("example_10_portal_rigid_offsets_combos.txt")
    assert _has_offset(m), "example_10 should have rigid end offsets on frames"
    assert len(m.load_combinations) >= 1
    assert {"DEAD", "LIVE"} <= set(m.load_cases)

    result = run_analysis(m, verbose=False)
    assert result.status == "ok"
    assert result.residual < 1e-6

    mc = run_multi_case_analysis(m, verbose=False)
    assert mc.combination({"DEAD": 1.2, "LIVE": 1.6}, name="ULS") is not None


def test_example_10_is_static_only():
    """Zero-density material → modal must refuse (no silent garbage)."""
    m = _load("example_10_portal_rigid_offsets_combos.txt")
    assert not _has_density(m)
    with pytest.raises(ValueError, match="density"):
        solve_modal(m)


# ── example_11 — every feature at once ────────────────────────────────────


def test_example_11_combines_all_features():
    m = _load("example_11_combined_all_features.txt")
    assert {"DEAD", "LIVE", "WIND"} <= set(m.load_cases)
    assert len(m.load_combinations) >= 2
    assert _has_offset(m), "kitchen-sink should have rigid offsets"
    assert _has_truss_diagonal(m), "kitchen-sink should have TRUSS diagonals"
    assert _has_density(m), "kitchen-sink should carry mass for modal"

    mc = run_multi_case_analysis(m, verbose=False)
    assert not mc.failed_cases, f"cases failed: {mc.failed_cases}"
    assert mc.combination({"DEAD": 1.2, "LIVE": 1.6}, name="ULS") is not None

    # Modal must solve even with the rigid offsets + bracing present.
    r = solve_modal(m, n_modes=3)
    assert r.status == "ok"
    assert r.n_modes >= 3
    assert float(r.frequencies[0]) > 0.0


def test_example_11_rigid_offsets_are_frame_only():
    """Sanity: no TRUSS element carries an offset (parser forbids it).

    Rigid offsets are a FRAME-only feature — TrussElement2D inherits from
    Element2D, which does not define offset fields — so a truss must not
    even have the attributes. hasattr is the strongest expression of that
    invariant (it also catches a future field accidentally added to the
    base/truss class)."""
    m = _load("example_11_combined_all_features.txt")
    for e in m.elements:
        if isinstance(e, TrussElement2D):
            assert not hasattr(e, "offset_i")
            assert not hasattr(e, "offset_j")
