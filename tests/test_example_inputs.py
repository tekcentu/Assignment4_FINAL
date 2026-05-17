"""Smoke tests for the bundled inputs/example_*.txt files.

Each example must round-trip through the parser, solve cleanly through
:func:`run_analysis`, and (where modal is applicable) produce at least
one positive natural frequency. New examples added to ``inputs/`` will
be picked up automatically — they only need to live under ``inputs/``
and match the ``example_*.txt`` glob.
"""

import os
import sys
from glob import glob

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.file_io import read_input_file
from structural_analysis.main import run_analysis
from structural_analysis.modal import solve_modal


def _example_paths() -> list[str]:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(here, "inputs", "example_*.txt")
    return sorted(glob(pattern))


@pytest.mark.parametrize("path", _example_paths(),
                          ids=lambda p: os.path.basename(p))
def test_example_static_solve(path: str):
    """Every bundled example must load and static-solve to a small
    equilibrium residual."""
    model = read_input_file(path)
    assert model.elements, f"{path}: parsed model has no elements"
    result = run_analysis(model, verbose=False)
    assert result.status == "ok", f"{path}: status = {result.status}"
    assert result.residual < 1e-6, f"{path}: residual {result.residual:.2e}"


@pytest.mark.parametrize("path", _example_paths(),
                          ids=lambda p: os.path.basename(p))
def test_example_modal_when_applicable(path: str):
    """If an example carries positive density on any element and has at
    least one free DOF, modal must return ``status == "ok"`` with
    strictly positive frequencies. Otherwise the solver must refuse
    with a clear ValueError (no silent garbage answer)."""
    model = read_input_file(path)
    has_mass = any(getattr(e, "rho", 0.0) > 0.0 for e in model.elements)
    if not has_mass:
        with pytest.raises(ValueError, match="density"):
            solve_modal(model)
        return
    try:
        r = solve_modal(model, n_modes=3)
    except ValueError as e:
        # Acceptable when every DOF is restrained — Example 7 is the
        # canonical case (fixed-fixed beam, thermal-only demo).
        assert "no free DOFs" in str(e), f"{path}: unexpected error {e!r}"
        return
    assert r.status == "ok"
    assert r.n_modes >= 1
    assert float(r.frequencies[0]) > 0.0, (
        f"{path}: f1 = {float(r.frequencies[0])} Hz, expected > 0"
    )
