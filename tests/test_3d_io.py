"""
3D file-format tests: NODES z column, SUPPORTS3D, LOADS3D, member-load
z components, force_3d option, writer round-trips, and the shipped
3D example input.
"""

import os

import pytest

from structural_analysis.file_io import read_input_file
from structural_analysis.gui_common.file_writer import write_input_file
from structural_analysis.main import run_from_file
from structural_analysis.model import (
    Material, Node, NodalLoad, Section, StructuralModel, Support,
    UniformDistributedLoad, PointLoad,
)
from structural_analysis.element import FrameElement2D

INPUTS = os.path.join(os.path.dirname(__file__), "..", "inputs")


def _write(tmp_path, text):
    p = tmp_path / "model.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_nodes_z_column_parses(tmp_path):
    path = _write(tmp_path, """
TITLE
z column
NODES 2
1  0.0  0.0
2  1.0  2.0  3.5
MATERIALS 1
1  0.01  1e-4  200000000.0
ELEMENTS 1
1  1  2  1  FRAME
""")
    m = read_input_file(path)
    assert m.nodes[1].z == 0.0
    assert m.nodes[2].z == 3.5


def test_supports3d_parses_flags_and_settlements(tmp_path):
    path = _write(tmp_path, """
TITLE
s3d
NODES 2
1  0.0  0.0  0.0
2  1.0  0.0  1.0
MATERIALS 1
1  0.01  1e-4  200000000.0
ELEMENTS 1
1  1  2  1  FRAME
SUPPORTS3D 1
1  1 1 1 0 1 1  0.0 -0.01 0.002 0.0 0.0 0.0
""")
    m = read_input_file(path)
    s = m.supports[1]
    assert (s.ux, s.uy, s.uz, s.rx, s.ry, s.rz) == (
        True, True, True, False, True, True)
    assert s.settle_uy == -0.01
    assert s.settle_uz == 0.002
    assert s.settle_rx is None


def test_supports3d_short_row_rejected(tmp_path):
    path = _write(tmp_path, """
TITLE
bad
NODES 1
1  0.0  0.0
MATERIALS 0
SUPPORTS3D 1
1  1 1 1
""")
    with pytest.raises(ValueError, match="SUPPORTS3D"):
        read_input_file(path)


def test_loads3d_parses_with_case(tmp_path):
    path = _write(tmp_path, """
TITLE
l3d
NODES 2
1  0.0  0.0  0.0
2  1.0  0.0  1.0
MATERIALS 1
1  0.01  1e-4  200000000.0
ELEMENTS 1
1  1  2  1  FRAME
LOADS3D 1
2  1.0  -2.0  3.0  0.5  -0.5  4.0  case=WIND
""")
    m = read_input_file(path)
    ld = m.nodal_loads[0]
    assert (ld.fx, ld.fy, ld.fz) == (1.0, -2.0, 3.0)
    assert (ld.mx, ld.my, ld.mz) == (0.5, -0.5, 4.0)
    assert ld.load_case == "WIND"
    assert "WIND" in m.load_cases


def test_member_loads_z_components_parse(tmp_path):
    path = _write(tmp_path, """
TITLE
member z
NODES 2
1  0.0  0.0
2  4.0  0.0
MATERIALS 1
1  0.01  1e-4  200000000.0
ELEMENTS 1
1  1  2  1  FRAME
MEMBER_UDL 1
1  0.0  -5.0  -2.5
MEMBER_POINT_LOADS 1
1  2.0  0.0  -8.0  -3.0  global
""")
    m = read_input_file(path)
    udl, pl = m.elements[0].member_loads
    assert udl.wz == -2.5
    assert pl.pz == -3.0
    assert pl.coord_system == "global"


def test_force_3d_analysis_option(tmp_path):
    path = _write(tmp_path, """
TITLE
forced
NODES 2
1  0.0  0.0
2  4.0  0.0
MATERIALS 1
1  0.01  1e-4  200000000.0
ELEMENTS 1
1  1  2  1  FRAME
ANALYSIS_OPTIONS 1
force_3d=true
""")
    m = read_input_file(path)
    assert m.force_3d is True


def _model_3d():
    m = StructuralModel()
    m.title = "roundtrip-3d"
    m.nodes[1] = Node(1, 0.0, 0.0, 0.0)
    m.nodes[2] = Node(2, 3.0, 0.0, 0.0)
    m.nodes[3] = Node(3, 3.0, 0.0, 3.0)
    m.materials[1] = Material(id=1, E=200e6, nu=0.25)
    m.sections[1] = Section(id=1, material_id=1, A=0.01, I=4e-5,
                            depth=0.4, J=3e-5)
    e1 = FrameElement2D(id=1, node_i=1, node_j=2, E=200e6, A=0.01,
                        I=4e-5, section_id=1)
    e2 = FrameElement2D(id=2, node_i=2, node_j=3, E=200e6, A=0.01,
                        I=4e-5, section_id=1)
    e1.member_loads.append(UniformDistributedLoad(wy=-1.0, wz=-2.0))
    e2.member_loads.append(PointLoad(py=0.0, pz=-5.0, a=1.0))
    m.elements += [e1, e2]
    m.supports[1] = Support(1, True, True, True, uz=True, rx=True,
                            ry=True, settle_uz=-0.004)
    m.nodal_loads.append(NodalLoad(3, fy=-10.0, fz=2.0, mx=1.5))
    return m


def test_3d_model_roundtrips_through_writer(tmp_path):
    m = _model_3d()
    path = str(tmp_path / "rt.txt")
    write_input_file(m, path)
    m2 = read_input_file(path)

    assert m2.nodes[3].z == 3.0
    s = m2.supports[1]
    assert s.uz and s.rx and s.ry
    assert s.settle_uz == -0.004
    ld = m2.nodal_loads[0]
    assert (ld.fy, ld.fz, ld.mx) == (-10.0, 2.0, 1.5)
    udl = m2.elements[0].member_loads[0]
    assert (udl.wy, udl.wz) == (-1.0, -2.0)
    pl = m2.elements[1].member_loads[0]
    assert pl.pz == -5.0

    # And the round-tripped model solves identically.
    from structural_analysis.main import run_analysis
    r1 = run_analysis(m, verbose=False)
    r2 = run_analysis(m2, verbose=False)
    assert r1.status == r2.status == "ok"
    for nid in m.nodes:
        for dof in ("ux", "uy", "uz", "rx", "ry", "rz"):
            i1 = r1.E_map[nid][dof]
            i2 = r2.E_map[nid][dof]
            v1 = 0.0 if i1 is None else r1.D[i1]
            v2 = 0.0 if i2 is None else r2.D[i2]
            assert v1 == pytest.approx(v2, rel=1e-12, abs=1e-15)


def test_2d_writer_output_has_no_3d_blocks(tmp_path):
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 4.0, 0.0)
    m.materials[1] = Material(id=1, E=200e6)
    m.sections[1] = Section(id=1, material_id=1, A=0.01, I=4e-5)
    m.elements.append(FrameElement2D(id=1, node_i=1, node_j=2, E=200e6,
                                     A=0.01, I=4e-5, section_id=1))
    m.supports[1] = Support(1, True, True, True)
    m.nodal_loads.append(NodalLoad(2, fy=-1.0))
    path = str(tmp_path / "plain.txt")
    write_input_file(m, path)
    text = open(path, encoding="utf-8").read()
    assert "SUPPORTS3D" not in text
    assert "LOADS3D" not in text
    # node rows keep the legacy 3-column shape
    assert "1  0.0  0.0\n" in text


def test_example_3d_grillage_input_solves():
    result = run_from_file(
        os.path.join(INPUTS, "example_3d_grillage.txt"), verbose=False,
    )
    assert result.status == "ok"
    E, Iz, G, J, L, F = 200e6, 4e-5, 80e6, 3e-5, 3.0, -10.0
    expected = (2 * F * L**3 / (3 * E * Iz)) + F * L**3 / (G * J)
    uy3 = result.D[result.E_map[3]["uy"]]
    assert uy3 == pytest.approx(expected, rel=1e-9)
