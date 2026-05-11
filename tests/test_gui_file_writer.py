"""Round-trip tests for ``structural_analysis.gui.file_writer.write_input_file``.

These guard against silent regressions in the serialiser: every common
language construct (materials with α/depth, frame & truss elements, moment
releases, supports with settlement, nodal loads, member UDLs, point loads,
truss & frame thermal loads) should write to disk and parse back to a model
that solves identically.
"""

from __future__ import annotations

import tempfile
import os

import numpy as np
import pytest

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.file_io import read_input_file
from structural_analysis.gui.file_writer import write_input_file
from structural_analysis.main import run_analysis
from structural_analysis.model import (
    FrameTemperatureLoad,
    Material,
    NodalLoad,
    PointLoad,
    StructuralModel,
    Support,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)


def _round_trip(model: StructuralModel) -> StructuralModel:
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        write_input_file(model, tmp)
        return read_input_file(tmp)
    finally:
        os.unlink(tmp)


def _assert_results_equal(m1: StructuralModel, m2: StructuralModel) -> None:
    r1 = run_analysis(m1, verbose=False)
    r2 = run_analysis(m2, verbose=False)
    assert r1.status == r2.status == "ok"
    assert set(r1.member_results.keys()) == set(r2.member_results.keys())
    for eid in r1.member_results:
        f1 = np.array(r1.member_results[eid]["f_local"], dtype=float)
        f2 = np.array(r2.member_results[eid]["f_local"], dtype=float)
        np.testing.assert_allclose(f1, f2, atol=1e-9, rtol=1e-9)


def test_round_trip_q2a_settlement():
    m = read_input_file("inputs/q2a_settlement.txt")
    m2 = _round_trip(m)
    _assert_results_equal(m, m2)


def test_round_trip_q2b_thermal():
    m = read_input_file("inputs/q2b_thermal.txt")
    m2 = _round_trip(m)
    _assert_results_equal(m, m2)


def test_round_trip_synthetic_kitchen_sink():
    """Hand-built model exercising every load type, releases, settlement, alpha."""
    from structural_analysis.model import Node

    m = StructuralModel(title="kitchen sink")
    m.materials[1] = Material(1, E=2.0e8, A=0.01, I=1e-4, alpha=1.2e-5, depth=0.3)
    m.materials[2] = Material(2, E=2.0e8, A=1e-4, I=1e-12)

    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 4.0, 0.0)
    m.nodes[3] = Node(3, 8.0, 0.0)
    m.nodes[4] = Node(4, 4.0, 3.0)

    mat1 = m.materials[1]
    mat2 = m.materials[2]
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=mat1.E, A=mat1.A, I=mat1.I,
        alpha=mat1.alpha, depth=mat1.depth,
        release_i=False, release_j=True,
        member_loads=[UniformDistributedLoad(wy=-5.0)],
    ))
    m.elements.append(FrameElement2D(
        id=2, node_i=2, node_j=3, E=mat1.E, A=mat1.A, I=mat1.I,
        alpha=mat1.alpha, depth=mat1.depth,
        member_loads=[PointLoad(py=-3.0, a=2.0),
                       FrameTemperatureLoad(t_top=10.0, t_bottom=-10.0)],
    ))
    m.elements.append(TrussElement2D(
        id=3, node_i=4, node_j=2, E=mat2.E, A=mat2.A,
        member_loads=[TrussTemperatureLoad(delta_T=25.0)],
    ))

    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[3] = Support(node_id=3, ux=False, uy=True, settle_uy=-0.001)
    m.supports[4] = Support(node_id=4, ux=True, uy=True)

    m.nodal_loads.append(NodalLoad(node_id=2, fx=10.0, fy=0.0, mz=0.0))

    m2 = _round_trip(m)
    _assert_results_equal(m, m2)


def test_writer_raises_when_element_material_unrecoverable():
    """If an element's E/A/I doesn't match any material, the writer must fail loudly."""
    m = StructuralModel(title="orphan element")
    from structural_analysis.model import Node
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.materials[1] = Material(1, E=1.0, A=1.0, I=1.0)
    # Element references properties that don't match material 1.
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=999.0, A=999.0, I=999.0,
    ))
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with pytest.raises(ValueError, match="does not match any material"):
            write_input_file(m, tmp)
    finally:
        os.unlink(tmp)
