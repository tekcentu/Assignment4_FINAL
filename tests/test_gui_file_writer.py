"""Round-trip tests for ``structural_analysis.gui_common.file_writer.write_input_file``.

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
from structural_analysis.gui_common.file_writer import write_input_file
from structural_analysis.main import run_analysis
from structural_analysis.model import (
    FrameTemperatureLoad,
    Material,
    NodalLoad,
    PointLoad,
    Section,
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
    m.materials[1] = Material(id=1, name="steel", E=2.0e8, alpha=1.2e-5)
    m.materials[2] = Material(id=2, name="steel-truss", E=2.0e8)
    m.sections[1] = Section(id=1, name="50x50", material_id=1,
                             A=0.01, I=1e-4, depth=0.3)
    m.sections[2] = Section(id=2, name="rod", material_id=2,
                             A=1e-4, I=1e-12)

    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 4.0, 0.0)
    m.nodes[3] = Node(3, 8.0, 0.0)
    m.nodes[4] = Node(4, 4.0, 3.0)

    mat1, sec1 = m.materials[1], m.sections[1]
    mat2, sec2 = m.materials[2], m.sections[2]
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=mat1.E, A=sec1.A, I=sec1.I,
        alpha=mat1.alpha, depth=sec1.depth, section_id=sec1.id,
        release_i=False, release_j=True,
        member_loads=[UniformDistributedLoad(wy=-5.0)],
    ))
    m.elements.append(FrameElement2D(
        id=2, node_i=2, node_j=3, E=mat1.E, A=sec1.A, I=sec1.I,
        alpha=mat1.alpha, depth=sec1.depth, section_id=sec1.id,
        member_loads=[PointLoad(py=-3.0, a=2.0),
                       FrameTemperatureLoad(t_top=10.0, t_bottom=-10.0)],
    ))
    m.elements.append(TrussElement2D(
        id=3, node_i=4, node_j=2, E=mat2.E, A=sec2.A, section_id=sec2.id,
        member_loads=[TrussTemperatureLoad(delta_T=25.0)],
    ))

    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[3] = Support(node_id=3, ux=False, uy=True, settle_uy=-0.001)
    m.supports[4] = Support(node_id=4, ux=True, uy=True)

    m.nodal_loads.append(NodalLoad(node_id=2, fx=10.0, fy=0.0, mz=0.0))

    m2 = _round_trip(m)
    _assert_results_equal(m, m2)


def test_writer_raises_when_element_has_no_section_id():
    """Elements built outside the model layer have section_id=None.
    The writer must refuse to serialise them rather than silently dropping
    the section assignment."""
    m = StructuralModel(title="orphan element")
    from structural_analysis.model import Node
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.materials[1] = Material(id=1, E=1.0)
    m.sections[1] = Section(id=1, material_id=1, A=1.0, I=1.0)
    # No section_id assigned (default None).
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=1.0, A=1.0, I=1.0,
    ))
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with pytest.raises(ValueError, match="no section_id assigned"):
            write_input_file(m, tmp)
    finally:
        os.unlink(tmp)


def test_writer_raises_when_section_id_dangles():
    """If section_id points at a section that's been deleted, fail loudly."""
    m = StructuralModel(title="dangling section_id")
    from structural_analysis.model import Node
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.materials[1] = Material(id=1, E=1.0)
    # Note: no section with id 99
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=1.0, A=1.0, I=1.0, section_id=99,
    ))
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with pytest.raises(ValueError, match="references section 99"):
            write_input_file(m, tmp)
    finally:
        os.unlink(tmp)


def test_writer_rejects_whitespace_in_material_or_section_name():
    """Names round-trip as single whitespace-delimited tokens, so the writer
    must refuse names containing spaces rather than silently truncate on
    reload."""
    from structural_analysis.model import Node
    base = StructuralModel(title="ws names")
    base.nodes[1] = Node(1, 0.0, 0.0)
    base.nodes[2] = Node(2, 1.0, 0.0)
    base.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=1.0, A=1.0, I=1.0, section_id=1,
    ))
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        m_bad_mat = StructuralModel(title=base.title)
        m_bad_mat.nodes = dict(base.nodes)
        m_bad_mat.elements = list(base.elements)
        m_bad_mat.materials[1] = Material(id=1, name="A 36 steel", E=1.0)
        m_bad_mat.sections[1] = Section(id=1, material_id=1, A=1.0, I=1.0)
        with pytest.raises(ValueError, match="Material 1 name"):
            write_input_file(m_bad_mat, tmp)

        m_bad_sec = StructuralModel(title=base.title)
        m_bad_sec.nodes = dict(base.nodes)
        m_bad_sec.elements = list(base.elements)
        m_bad_sec.materials[1] = Material(id=1, name="A36", E=1.0)
        m_bad_sec.sections[1] = Section(
            id=1, name="W 12x26", material_id=1, A=1.0, I=1.0,
        )
        with pytest.raises(ValueError, match="Section 1 name"):
            write_input_file(m_bad_sec, tmp)
    finally:
        os.unlink(tmp)


# ── helpers for PR #40 tests ────────────────────────────────────────────


def _make_cantilever() -> StructuralModel:
    """Two-node single-element cantilever for I/O round-trip tests."""
    from structural_analysis.model import Node
    m = StructuralModel(title="cantilever")
    m.materials[1] = Material(id=1, E=2.1e8, alpha=0.0)
    m.sections[1] = Section(id=1, material_id=1, A=1e-2, I=1e-4, depth=0.1)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.nodes[3] = Node(3, 2.0, 0.0)
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=1e-2, I=1e-4,
        alpha=0.0, depth=0.1, section_id=1,
    ))
    m.elements.append(FrameElement2D(
        id=2, node_i=2, node_j=3, E=2.1e8, A=1e-2, I=1e-4,
        alpha=0.0, depth=0.1, section_id=1,
    ))
    return m


# ── JOINT_MASSES round-trip (PR #40) ────────────────────────────────────


def test_joint_masses_round_trip(tmp_path):
    """JOINT_MASSES block survives write→read with values intact."""
    from structural_analysis.model import JointMass, Node
    model = _make_cantilever()
    model.joint_masses[2] = JointMass(node_id=2, mx=500.0, my=750.5)
    model.joint_masses[3] = JointMass(node_id=3, mx=1000.0, my=1000.0)
    path = str(tmp_path / "jm_test.txt")
    write_input_file(model, path)
    m2 = read_input_file(path)
    assert 2 in m2.joint_masses
    assert m2.joint_masses[2].mx == pytest.approx(500.0)
    assert m2.joint_masses[2].my == pytest.approx(750.5)
    assert m2.joint_masses[3].mx == pytest.approx(1000.0)


def test_joint_masses_zero_values_round_trip(tmp_path):
    """Nodes with all-zero JointMass values write and read back."""
    from structural_analysis.model import JointMass
    model = _make_cantilever()
    model.joint_masses[2] = JointMass(node_id=2, mx=0.0, my=0.0)
    path = str(tmp_path / "jm_zero.txt")
    write_input_file(model, path)
    m2 = read_input_file(path)
    assert 2 in m2.joint_masses
    assert m2.joint_masses[2].mx == 0.0
    assert m2.joint_masses[2].my == 0.0


def test_writer_omits_joint_masses_block_when_empty(tmp_path):
    """Writer must NOT emit JOINT_MASSES when dict is empty."""
    model = _make_cantilever()
    path = str(tmp_path / "no_jm.txt")
    write_input_file(model, path)
    with open(path) as f:
        text = f.read()
    assert "JOINT_MASSES" not in text


# ── MODAL_MASS_SOURCE round-trip (PR #40) ───────────────────────────────


def test_modal_mass_source_round_trip(tmp_path):
    """MODAL_MASS_SOURCE block survives write→read."""
    from structural_analysis.model import LoadCase, ModalMassSource
    model = _make_cantilever()
    model.load_cases["LIVE"] = LoadCase(name="LIVE")
    model.modal_mass_source = ModalMassSource(
        include_self_mass=True,
        include_joint_masses=False,
        include_load_cases=True,
        load_case_factors={"LIVE": 0.3},
    )
    path = str(tmp_path / "mms_test.txt")
    write_input_file(model, path)
    m2 = read_input_file(path)
    assert m2.modal_mass_source.include_self_mass is True
    assert m2.modal_mass_source.include_joint_masses is False
    assert m2.modal_mass_source.include_load_cases is True
    assert m2.modal_mass_source.load_case_factors.get("LIVE") == pytest.approx(0.3)


def test_writer_omits_modal_mass_source_block_when_default(tmp_path):
    """Writer must NOT emit MODAL_MASS_SOURCE when source is default."""
    from structural_analysis.model import ModalMassSource
    model = _make_cantilever()
    model.modal_mass_source = ModalMassSource()  # default
    path = str(tmp_path / "no_mms.txt")
    write_input_file(model, path)
    with open(path) as f:
        text = f.read()
    assert "MODAL_MASS_SOURCE" not in text


def test_legacy_file_loads_with_empty_joint_masses_and_default_mms(tmp_path):
    """An old input file (no JOINT_MASSES / MODAL_MASS_SOURCE) loads with
    safe defaults: empty joint_masses dict and default ModalMassSource."""
    from structural_analysis.model import ModalMassSource
    model = _make_cantilever()
    path = str(tmp_path / "legacy.txt")
    write_input_file(model, path)

    # Strip any JOINT_MASSES / MODAL_MASS_SOURCE lines (shouldn't exist
    # with default model, but this makes the intent explicit).
    with open(path) as f:
        content = f.read()
    assert "JOINT_MASSES" not in content
    assert "MODAL_MASS_SOURCE" not in content

    m2 = read_input_file(path)
    assert m2.joint_masses == {}
    assert m2.modal_mass_source.is_default()


def test_joint_masses_unknown_node_id_raises(tmp_path):
    """JOINT_MASSES referencing a non-existent node must raise ValueError."""
    from structural_analysis.model import JointMass
    model = _make_cantilever()
    # Write manually to put a bad node_id in
    path = str(tmp_path / "bad_jm.txt")
    write_input_file(model, path)
    with open(path) as f:
        original = f.read()
    # Inject a JOINT_MASSES block with a bad node id
    injected = original.rstrip() + "\nJOINT_MASSES 1\n9999 mx=100.0\n"
    with open(path, "w") as f:
        f.write(injected)
    with pytest.raises(ValueError, match="9999"):
        read_input_file(path)


def test_joint_masses_negative_value_raises(tmp_path):
    """JOINT_MASSES with a negative mass value must raise ValueError."""
    from structural_analysis.model import JointMass
    model = _make_cantilever()
    path = str(tmp_path / "neg_jm.txt")
    write_input_file(model, path)
    with open(path) as f:
        original = f.read()
    injected = original.rstrip() + "\nJOINT_MASSES 1\n2 mx=-100.0\n"
    with open(path, "w") as f:
        f.write(injected)
    with pytest.raises(ValueError):
        read_input_file(path)


def test_modal_mass_source_unknown_key_raises(tmp_path):
    """MODAL_MASS_SOURCE with an unrecognised key must raise ValueError."""
    model = _make_cantilever()
    path = str(tmp_path / "bad_mms.txt")
    write_input_file(model, path)
    with open(path) as f:
        original = f.read()
    injected = original.rstrip() + "\nMODAL_MASS_SOURCE 1\nunknown_key=true\n"
    with open(path, "w") as f:
        f.write(injected)
    with pytest.raises(ValueError, match="Unknown MODAL_MASS_SOURCE key"):
        read_input_file(path)


def test_modal_mass_source_unknown_case_raises(tmp_path):
    """MODAL_MASS_SOURCE referencing a non-existent case must raise ValueError."""
    model = _make_cantilever()
    path = str(tmp_path / "bad_case_mms.txt")
    write_input_file(model, path)
    with open(path) as f:
        original = f.read()
    injected = (
        original.rstrip()
        + "\nMODAL_MASS_SOURCE 2\ninclude_load_cases=true\n"
        "case_factor:NONEXISTENT=1.0\n"
    )
    with open(path, "w") as f:
        f.write(injected)
    with pytest.raises(ValueError, match="NONEXISTENT"):
        read_input_file(path)
