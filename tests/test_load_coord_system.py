"""PR #25 — Local vs Global mechanical member-load coordinate system.

Pins down the new physics (axial wx/px components + global-frame
projection to local axes) against analytical answers, and the
backward-compat contract (existing local-y-only tests behave the same).

Sign convention reminder:
    p_local = [N_i, V_i, M_i, N_j, V_j, M_j]
    q_local = K · d - p_local                       (action of node on element)
    Internal axial N(x) = -N_i - wx_local · x       (tension positive)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from structural_analysis.element import (
    FrameElement2D,
    TrussElement2D,
    _project_load_to_local,
)
from structural_analysis.model import (
    Material,
    Node,
    PointLoad,
    Section,
    StructuralModel,
    Support,
    UniformDistributedLoad,
)


TOL = 1e-9


# ── helpers ──────────────────────────────────────────────────────────


def _horizontal_frame(L: float = 6.0) -> tuple[FrameElement2D, dict]:
    nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    e = FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.0e8, A=0.02, I=8e-5,
    )
    return e, nodes


def _vertical_column(L: float = 6.0) -> tuple[FrameElement2D, dict]:
    # i at the bottom, j at the top — local +x points upward (+Y global).
    nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 0.0, L)}
    e = FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.0e8, A=0.02, I=8e-5,
    )
    return e, nodes


def _diagonal_45(L: float = 6.0) -> tuple[FrameElement2D, dict]:
    d = L / (2 ** 0.5)
    nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, d, d)}
    e = FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.0e8, A=0.02, I=8e-5,
    )
    return e, nodes


# ── _project_load_to_local helper ─────────────────────────────────────


def test_project_local_pass_through():
    assert _project_load_to_local(1.0, 2.0, "local", 0.5, 0.7) == (1.0, 2.0)


def test_project_global_horizontal_member_identity():
    # θ = 0 → c=1, s=0 → global == local.
    wx_l, wy_l = _project_load_to_local(3.0, -5.0, "global", 1.0, 0.0)
    assert abs(wx_l - 3.0) < TOL
    assert abs(wy_l - (-5.0)) < TOL


def test_project_global_vertical_column_swaps_components():
    # θ = 90° → c=0, s=1 → global +Y becomes local +x.
    wx_l, wy_l = _project_load_to_local(0.0, -10.0, "global", 0.0, 1.0)
    assert abs(wx_l - (-10.0)) < TOL
    assert abs(wy_l - 0.0) < TOL


def test_project_global_45_splits_components_evenly():
    c = s = 1.0 / (2 ** 0.5)
    wx_l, wy_l = _project_load_to_local(0.0, -10.0, "global", c, s)
    # global -Y on a +45° member produces equal axial (-) and transverse (-).
    assert abs(wx_l - (-10.0 * s)) < TOL
    assert abs(wy_l - (-10.0 * c)) < TOL


def test_project_invalid_coord_system_raises():
    with pytest.raises(ValueError):
        _project_load_to_local(1.0, 2.0, "polar", 1.0, 0.0)


# ── Backward-compat: existing local-only behavior unchanged ──────────


def test_local_only_udl_matches_pre_v015_formula():
    """A local wy-only UDL must produce the exact 6-vector that v0.14.0 did."""
    e, nodes = _horizontal_frame(6.0)
    e.member_loads.append(UniformDistributedLoad(wy=-10.0))
    p = e.local_consistent_load(nodes)
    L = 6.0
    expected = np.array([0, -10 * L / 2, -10 * L**2 / 12,
                         0, -10 * L / 2, +10 * L**2 / 12])
    assert_allclose(p, expected, atol=1e-10)


def test_local_only_point_load_matches_pre_v015_formula():
    e, nodes = _horizontal_frame(6.0)
    e.member_loads.append(PointLoad(py=-20.0, a=3.0))
    p = e.local_consistent_load(nodes)
    # Midspan: equal shears, antisymmetric moments. Cubic Hermite gives
    # py*0.5 at each end and ±py*L/8 moments.
    assert abs(p[1] - p[4]) < TOL
    assert abs(p[2] + p[5]) < TOL
    assert abs(p[1] - (-20.0 * 0.5)) < TOL


# ── New axial physics: local wx / px ─────────────────────────────────


def test_local_wx_distributes_axial_half_to_each_end():
    """Axial UDL: half the total goes to each node along +x_local."""
    e, nodes = _horizontal_frame(6.0)
    e.member_loads.append(UniformDistributedLoad(wy=0.0, wx=4.0))
    p = e.local_consistent_load(nodes)
    L = 6.0
    expected = np.array([4.0 * L / 2, 0, 0, 4.0 * L / 2, 0, 0])
    assert_allclose(p, expected, atol=1e-10)


def test_local_px_distributes_by_lever_rule():
    """Axial PointLoad: linear shape-function distribution."""
    e, nodes = _horizontal_frame(6.0)
    L = 6.0
    a = 2.0
    e.member_loads.append(PointLoad(py=0.0, a=a, px=12.0))
    p = e.local_consistent_load(nodes)
    expected = np.array([
        12.0 * (L - a) / L, 0, 0,
        12.0 * a / L,       0, 0,
    ])
    assert_allclose(p, expected, atol=1e-10)


def test_combined_local_wx_and_wy_sums_axial_and_transverse():
    e, nodes = _horizontal_frame(6.0)
    e.member_loads.append(UniformDistributedLoad(wy=-10.0, wx=4.0))
    p = e.local_consistent_load(nodes)
    L = 6.0
    expected = np.array([
        4.0 * L / 2,         # axial i
        -10 * L / 2,         # shear i
        -10 * L**2 / 12,     # moment i
        4.0 * L / 2,         # axial j
        -10 * L / 2,         # shear j
        +10 * L**2 / 12,     # moment j
    ])
    assert_allclose(p, expected, atol=1e-10)


# ── Global frame: inclined-member transformation ──────────────────────


def test_global_udl_on_horizontal_member_matches_local():
    """θ=0 → global is identical to local; no axial component."""
    e_g, nodes_g = _horizontal_frame(6.0)
    e_g.member_loads.append(
        UniformDistributedLoad(wy=-10.0, coord_system="global")
    )
    e_l, nodes_l = _horizontal_frame(6.0)
    e_l.member_loads.append(UniformDistributedLoad(wy=-10.0))
    assert_allclose(
        e_g.local_consistent_load(nodes_g),
        e_l.local_consistent_load(nodes_l),
        atol=1e-10,
    )


def test_global_udl_on_vertical_column_produces_only_axial_fems():
    """Global -Y UDL on +Y column → axial-only fixed-end forces.

    The transformation gives (wx_l, wy_l) = (-10, 0); the FEM is
    (wx_l * L/2) at each end and zero shear/moment."""
    e, nodes = _vertical_column(6.0)
    e.member_loads.append(
        UniformDistributedLoad(wy=-10.0, coord_system="global")
    )
    p = e.local_consistent_load(nodes)
    L = 6.0
    # Local axial only at both ends. Local shear/moment must be ~0.
    assert abs(p[0] - (-10.0 * L / 2)) < TOL
    assert abs(p[3] - (-10.0 * L / 2)) < TOL
    assert abs(p[1]) < TOL
    assert abs(p[4]) < TOL
    assert abs(p[2]) < TOL
    assert abs(p[5]) < TOL


def test_global_udl_on_45_member_splits_axial_and_transverse():
    """A global -Y UDL on a +45° brace gives equal-magnitude axial and
    transverse FEM contributions, both negative."""
    L = 6.0
    e, nodes = _diagonal_45(L)
    e.member_loads.append(
        UniformDistributedLoad(wy=-10.0, coord_system="global")
    )
    p = e.local_consistent_load(nodes)
    s = 1.0 / (2 ** 0.5)
    expected_wx_l = -10.0 * s   # = -10 * sin θ
    expected_wy_l = -10.0 * s   # = -10 * cos θ (cos 45 = sin 45)
    # Axial half-and-half.
    assert abs(p[0] - expected_wx_l * L / 2) < 1e-9
    assert abs(p[3] - expected_wx_l * L / 2) < 1e-9
    # Transverse: standard wL/2 + ±wL²/12.
    assert abs(p[1] - expected_wy_l * L / 2) < 1e-9
    assert abs(p[4] - expected_wy_l * L / 2) < 1e-9
    assert abs(p[2] - expected_wy_l * L**2 / 12) < 1e-9
    assert abs(p[5] - (-expected_wy_l * L**2 / 12)) < 1e-9


def test_global_point_load_on_vertical_column_produces_only_axial_fems():
    e, nodes = _vertical_column(6.0)
    e.member_loads.append(PointLoad(
        py=-15.0, a=2.0, coord_system="global",
    ))
    p = e.local_consistent_load(nodes)
    # Local: (px_l, py_l) = (-15, 0). Axial split by lever rule.
    L = 6.0
    a = 2.0
    assert abs(p[0] - (-15.0 * (L - a) / L)) < TOL
    assert abs(p[3] - (-15.0 * a / L)) < TOL
    assert abs(p[1]) < TOL
    assert abs(p[4]) < TOL
    assert abs(p[2]) < TOL
    assert abs(p[5]) < TOL


# ── Static equilibrium: full solve with global loads ─────────────────


def _solve(model: StructuralModel):
    from structural_analysis.main import run_analysis
    return run_analysis(model)


def _model_with_one_member(elem: FrameElement2D, nodes: dict[int, Node]):
    """Build a minimal solvable model around one element + its nodes,
    fully fixed at both ends so the loaded element sees real reactions
    rather than getting absorbed by a flexible boundary."""
    m = StructuralModel(title="t")
    m.materials[1] = Material(id=1, name="Steel", E=2.0e8, density=0.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    elem.section_id = 1
    elem.alpha = 0.0
    elem.depth = 0.3
    m.nodes = dict(nodes)
    m.elements = [elem]
    for nid in m.nodes:
        m.supports[nid] = Support(node_id=nid, ux=True, uy=True, rz=True)
    return m


def test_solve_global_udl_vertical_column_axial_reactions_match_total_load():
    """End-to-end: a 6 m fixed-fixed column with global -Y UDL of 10 kN/m
    must develop axial reactions summing to -60 kN in global Y."""
    L = 6.0
    e, nodes = _vertical_column(L)
    e.member_loads.append(
        UniformDistributedLoad(wy=-10.0, coord_system="global")
    )
    m = _model_with_one_member(e, nodes)
    r = _solve(m)
    assert r.status == "ok"
    # Reactions: each node's Ry component, summed.
    total_Ry = sum(
        rxn.get("uy", 0.0) for rxn in r.reactions.values()
    )
    assert abs(total_Ry - (10.0 * L)) < 1e-6, total_Ry


def test_solve_local_wy_horizontal_beam_unchanged_equilibrium():
    """Local-only path must remain identical: 6 m fixed-fixed beam with
    wy=-10 develops shear reactions summing to wy*L."""
    L = 6.0
    e, nodes = _horizontal_frame(L)
    e.member_loads.append(UniformDistributedLoad(wy=-10.0))
    m = _model_with_one_member(e, nodes)
    r = _solve(m)
    assert r.status == "ok"
    total_Ry = sum(
        rxn.get("uy", 0.0) for rxn in r.reactions.values()
    )
    assert abs(total_Ry - (10.0 * L)) < 1e-6


# ── Truss rejection unchanged ────────────────────────────────────────


def test_truss_still_rejects_global_udl():
    nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    e = TrussElement2D(id=1, node_i=1, node_j=2, E=2.0e8, A=0.02)
    e.member_loads.append(
        UniformDistributedLoad(wy=-10.0, coord_system="global")
    )
    with pytest.raises(TypeError):
        e.local_consistent_load(nodes)


def test_truss_still_rejects_global_pointload():
    nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    e = TrussElement2D(id=1, node_i=1, node_j=2, E=2.0e8, A=0.02)
    e.member_loads.append(PointLoad(
        py=-15.0, a=2.0, coord_system="global",
    ))
    with pytest.raises(TypeError):
        e.local_consistent_load(nodes)


# ── Member-force recovery includes axial fixed-end effects ───────────


def test_member_force_recovery_includes_axial_fixed_end_for_local_wx():
    """For a fixed-fixed bar with axial local wx, the end forces N_i and
    N_j must reflect the consistent-load formula. Specifically, with
    both ends fixed and only an axial wx, K·d=0 so q_local = -p_local,
    giving N_i = N_j = -wx·L/2."""
    L = 6.0
    e, nodes = _horizontal_frame(L)
    e.member_loads.append(UniformDistributedLoad(wy=0.0, wx=5.0))
    m = _model_with_one_member(e, nodes)
    r = _solve(m)
    assert r.status == "ok"
    forces = r.member_results[e.id]["f_local"]
    expected_N_i = -5.0 * L / 2
    expected_N_j = -5.0 * L / 2
    assert abs(forces[0] - expected_N_i) < 1e-6
    assert abs(forces[3] - expected_N_j) < 1e-6


# ── N diagram reflects wx / px (PR #25 element_graphics update) ──────


def test_axial_diagram_constant_when_no_axial_member_loads():
    """Backward compat: an element with only transverse loads still
    produces a CONSTANT N(x) trace equal to -N_i, matching v0.14.0."""
    from structural_analysis.gui_qt.element_graphics import (
        sample_internal_force,
    )
    e, nodes = _horizontal_frame(6.0)
    e.member_loads.append(UniformDistributedLoad(wy=-10.0))
    f_local = np.array([3.0, -30.0, 0.0, -3.0, -30.0, 0.0])
    xs, ys = sample_internal_force(
        e, nodes[1], nodes[2], f_local, "axial", n_samples=5,
    )
    # All samples equal -N_i = -3.0.
    for y in ys:
        assert abs(y - (-3.0)) < TOL


def test_axial_diagram_linear_with_local_wx():
    """N(x) = -N_i - wx · x for a local axial UDL."""
    from structural_analysis.gui_qt.element_graphics import (
        sample_internal_force,
    )
    L = 6.0
    e, nodes = _horizontal_frame(L)
    e.member_loads.append(UniformDistributedLoad(wy=0.0, wx=4.0))
    # With both ends fixed and axial wx=4, N_i = -wx·L/2 = -12.
    N_i = -12.0
    f_local = np.array([N_i, 0.0, 0.0, -N_i, 0.0, 0.0])
    xs, ys = sample_internal_force(
        e, nodes[1], nodes[2], f_local, "axial", n_samples=7,
    )
    for x, y in zip(xs, ys):
        expected = -N_i - 4.0 * x
        assert abs(y - expected) < 1e-9


def test_axial_diagram_jumps_at_px_point():
    """A local axial PointLoad must produce a step in N(x) at x = a."""
    from structural_analysis.gui_qt.element_graphics import (
        evaluate_internal_force,
    )
    L = 6.0
    e, nodes = _horizontal_frame(L)
    e.member_loads.append(PointLoad(py=0.0, a=3.0, px=10.0))
    # Pretend f_local with N_i = 0 for clarity.
    f_local = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _, fn = evaluate_internal_force(
        e, nodes[1], nodes[2], f_local, "axial",
    )
    # Just before x = 3: N = 0; just after: N = -10.
    assert abs(fn(2.9999) - 0.0) < 1e-6
    assert abs(fn(3.0001) - (-10.0)) < 1e-6


# ── Split / remap preserves new fields ───────────────────────────────


def test_split_preserves_coord_system_and_wx_on_both_udl_children():
    from structural_analysis.gui_common.commands import SplitElementCmd

    m = StructuralModel(title="t")
    m.materials[1] = Material(id=1, name="Steel", E=2.0e8, density=0.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.0e8, A=0.02, I=8e-5, section_id=1,
    )
    e.member_loads.append(UniformDistributedLoad(
        wy=-10.0, wx=4.0, coord_system="global",
    ))
    m.elements = [e]
    SplitElementCmd(element_id=1, x=3.0, y=0.0).do(m)
    a, b = sorted(m.elements, key=lambda el: el.id)
    for child in (a, b):
        assert len(child.member_loads) == 1
        ld = child.member_loads[0]
        assert isinstance(ld, UniformDistributedLoad)
        assert ld.wx == 4.0
        assert ld.wy == -10.0
        assert ld.coord_system == "global"


def test_split_preserves_px_and_coord_system_on_point_load():
    from structural_analysis.gui_common.commands import SplitElementCmd

    m = StructuralModel(title="t")
    m.materials[1] = Material(id=1, name="Steel", E=2.0e8, density=0.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.0e8, A=0.02, I=8e-5, section_id=1,
    )
    e.member_loads.append(PointLoad(
        py=-15.0, a=4.0, px=8.0, coord_system="global",
    ))
    m.elements = [e]
    SplitElementCmd(element_id=1, x=3.0, y=0.0).do(m)
    children = sorted(m.elements, key=lambda el: el.id)
    # PointLoad at a=4 > L1=3 routes to child B with a-=L1 = 1.0.
    found = False
    for child in children:
        for ld in child.member_loads:
            if isinstance(ld, PointLoad):
                assert ld.py == -15.0
                assert ld.px == 8.0
                assert ld.coord_system == "global"
                if abs(ld.a - 1.0) < 1e-9:
                    found = True
    assert found, "PointLoad with shifted a was not found on either child"


# ── File I/O round-trip ──────────────────────────────────────────────


def _roundtrip(model: StructuralModel) -> StructuralModel:
    import os
    import tempfile

    from structural_analysis.file_io import read_input_file
    from structural_analysis.gui_common.file_writer import write_input_file

    fd, path = tempfile.mkstemp(suffix=".spa.txt")
    os.close(fd)
    try:
        write_input_file(model, path)
        return read_input_file(path)
    finally:
        os.unlink(path)


def _seed_model_with_loaded_member(load) -> StructuralModel:
    m = StructuralModel(title="t")
    m.materials[1] = Material(id=1, name="Steel", E=2.0e8, density=0.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.0e8, A=0.02, I=8e-5, section_id=1,
    )
    e.member_loads.append(load)
    m.elements = [e]
    return m


def test_roundtrip_local_only_udl_byte_stable_default_token():
    """Writer must NOT emit a coord-system token for local loads (the
    default) so legacy .spa.txt files stay diff-free."""
    import os
    import tempfile

    from structural_analysis.gui_common.file_writer import write_input_file

    m = _seed_model_with_loaded_member(UniformDistributedLoad(wy=-10.0))
    fd, path = tempfile.mkstemp(suffix=".spa.txt")
    os.close(fd)
    try:
        write_input_file(m, path)
        with open(path, "r") as f:
            content = f.read()
        # The UDL line must NOT mention "local" or "global".
        udl_lines = [ln for ln in content.splitlines()
                      if ln.strip().startswith("1  ")
                      and "0.0" in ln and "MEMBER" not in ln]
        # Find the actual UDL row (under MEMBER_UDL header).
        rows = content.splitlines()
        idx = next(i for i, r in enumerate(rows) if r.startswith("MEMBER_UDL"))
        udl_row = rows[idx + 1]
        assert "local" not in udl_row
        assert "global" not in udl_row
    finally:
        os.unlink(path)


def test_roundtrip_global_udl_emits_and_preserves_token():
    m = _seed_model_with_loaded_member(
        UniformDistributedLoad(wy=-10.0, wx=4.0, coord_system="global")
    )
    m2 = _roundtrip(m)
    ld = m2.elements[0].member_loads[0]
    assert ld.coord_system == "global"
    assert ld.wx == 4.0
    assert ld.wy == -10.0


def test_roundtrip_global_pointload_preserves_token():
    m = _seed_model_with_loaded_member(
        PointLoad(py=-15.0, a=2.0, px=5.0, coord_system="global")
    )
    m2 = _roundtrip(m)
    ld = m2.elements[0].member_loads[0]
    assert isinstance(ld, PointLoad)
    assert ld.coord_system == "global"
    assert ld.px == 5.0
    assert ld.py == -15.0
    assert ld.a == 2.0


def test_reader_accepts_legacy_three_column_udl_no_token():
    """A legacy ``MEMBER_UDL`` row with the three pre-v0.15.0 columns
    (eid, wx, wy) and no trailing token must parse as coord_system="local"."""
    import os
    import tempfile

    from structural_analysis.file_io import read_input_file

    body = (
        "TITLE\nlegacy\nNODES 2\n"
        "1 0 0\n2 6 0\n"
        "MATERIALS 1\n1  0.02  8e-5  2e8\n"
        "ELEMENTS 1\n1 1 2 1 FRAME\n"
        "SUPPORTS 1\n1 1 1 1\n"
        "LOADS 0\n"
        "MEMBER_UDL 1\n1  0.0  -10.0\n"
    )
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            f.write(body)
        m = read_input_file(tmp)
        ld = m.elements[0].member_loads[0]
        assert ld.coord_system == "local"
        assert ld.wx == 0.0
        assert ld.wy == -10.0
    finally:
        os.unlink(tmp)


def test_reader_rejects_unknown_coord_system_token():
    """Typos must not silently degrade to local."""
    import os
    import tempfile

    from structural_analysis.file_io import read_input_file

    body = (
        "TITLE\nbad\nNODES 2\n"
        "1 0 0\n2 6 0\n"
        "MATERIALS 1\n1  0.02  8e-5  2e8\n"
        "ELEMENTS 1\n1 1 2 1 FRAME\n"
        "SUPPORTS 1\n1 1 1 1\n"
        "LOADS 0\n"
        "MEMBER_UDL 1\n1  0.0  -10.0  globle\n"  # typo
    )
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            f.write(body)
        with pytest.raises(ValueError):
            read_input_file(tmp)
    finally:
        os.unlink(tmp)


# ── PR #26 — Gravity coord_system token ──────────────────────────────


def test_gravity_udl_horizontal_beam_projects_to_local_y_negative():
    """Gravity magnitude wy=+10 on a horizontal beam → local y = -10
    (force points in -Y, which equals -local-y on a horizontal beam)."""
    e, nodes = _horizontal_frame(6.0)
    e.member_loads.append(
        UniformDistributedLoad(wy=10.0, coord_system="gravity")
    )
    p = e.local_consistent_load(nodes)
    L = 6.0
    # Pure transverse: -10 in local y. p[1] = p[4] = -10*L/2 = -30.
    assert abs(p[1] - (-10.0 * L / 2)) < TOL
    assert abs(p[4] - (-10.0 * L / 2)) < TOL
    # No axial.
    assert abs(p[0]) < TOL
    assert abs(p[3]) < TOL


def test_gravity_udl_vertical_column_projects_to_local_axial_compression():
    """Gravity wy=+10 on a +Y column (c=0, s=1) → local x = -10,
    local y = 0. Axial FEMs only, both negative (compression at top
    and bottom)."""
    e, nodes = _vertical_column(6.0)
    e.member_loads.append(
        UniformDistributedLoad(wy=10.0, coord_system="gravity")
    )
    p = e.local_consistent_load(nodes)
    L = 6.0
    # Pure axial: -10 in local x. p[0] = p[3] = -10*L/2 = -30.
    assert abs(p[0] - (-10.0 * L / 2)) < TOL
    assert abs(p[3] - (-10.0 * L / 2)) < TOL
    # No transverse.
    assert abs(p[1]) < TOL
    assert abs(p[4]) < TOL
    assert abs(p[2]) < TOL
    assert abs(p[5]) < TOL


def test_gravity_udl_45_member_splits_evenly():
    """Gravity wy=+10 on a +45° brace → local x = -10/√2,
    local y = -10/√2."""
    L = 6.0
    e, nodes = _diagonal_45(L)
    e.member_loads.append(
        UniformDistributedLoad(wy=10.0, coord_system="gravity")
    )
    p = e.local_consistent_load(nodes)
    expected = -10.0 / (2 ** 0.5)
    # Axial and transverse equal magnitude (both compression / -y).
    assert abs(p[0] - expected * L / 2) < 1e-9
    assert abs(p[3] - expected * L / 2) < 1e-9
    assert abs(p[1] - expected * L / 2) < 1e-9
    assert abs(p[4] - expected * L / 2) < 1e-9


def test_gravity_pointload_vertical_column_axial_only():
    e, nodes = _vertical_column(6.0)
    e.member_loads.append(PointLoad(
        py=15.0, a=2.0, coord_system="gravity",
    ))
    p = e.local_consistent_load(nodes)
    L = 6.0
    a = 2.0
    # py=+15 gravity on +Y column → local px = -15, local py = 0.
    # Lever rule on axial:
    assert abs(p[0] - (-15.0 * (L - a) / L)) < TOL
    assert abs(p[3] - (-15.0 * a / L)) < TOL
    # No transverse.
    assert abs(p[1]) < TOL
    assert abs(p[4]) < TOL
    assert abs(p[2]) < TOL
    assert abs(p[5]) < TOL


def test_gravity_udl_with_nonzero_wx_raises():
    with pytest.raises(ValueError, match="gravity"):
        UniformDistributedLoad(wy=10.0, wx=3.0, coord_system="gravity")


def test_gravity_pointload_with_nonzero_px_raises():
    with pytest.raises(ValueError, match="gravity"):
        PointLoad(py=10.0, a=1.0, px=3.0, coord_system="gravity")


def test_invalid_coord_system_token_raises():
    with pytest.raises(ValueError):
        UniformDistributedLoad(wy=10.0, coord_system="cartesian")


def test_solve_gravity_udl_horizontal_beam_total_Ry_equals_wL():
    """Positive gravity magnitude on a horizontal fixed-fixed beam
    must produce upward total vertical reaction equal to wy·L."""
    L = 6.0
    e, nodes = _horizontal_frame(L)
    e.member_loads.append(
        UniformDistributedLoad(wy=10.0, coord_system="gravity")
    )
    m = _model_with_one_member(e, nodes)
    r = _solve(m)
    assert r.status == "ok"
    total_Ry = sum(rxn.get("uy", 0.0) for rxn in r.reactions.values())
    assert abs(total_Ry - 10.0 * L) < 1e-6, total_Ry


def test_split_preserves_gravity_coord_system_on_both_children():
    from structural_analysis.gui_common.commands import SplitElementCmd

    m = StructuralModel(title="t")
    m.materials[1] = Material(id=1, name="Steel", E=2.0e8, density=0.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.0e8, A=0.02, I=8e-5, section_id=1,
    )
    e.member_loads.append(
        UniformDistributedLoad(wy=10.0, coord_system="gravity")
    )
    m.elements = [e]
    SplitElementCmd(element_id=1, x=3.0, y=0.0).do(m)
    a, b = sorted(m.elements, key=lambda el: el.id)
    for child in (a, b):
        assert len(child.member_loads) == 1
        ld = child.member_loads[0]
        assert isinstance(ld, UniformDistributedLoad)
        assert ld.coord_system == "gravity"
        assert ld.wy == 10.0
        assert ld.wx == 0.0


def test_file_roundtrip_gravity_token_preserved():
    m = _seed_model_with_loaded_member(
        UniformDistributedLoad(wy=10.0, coord_system="gravity")
    )
    m2 = _roundtrip(m)
    ld = m2.elements[0].member_loads[0]
    assert ld.coord_system == "gravity"
    assert ld.wy == 10.0


def test_reader_accepts_gravity_token():
    """Pre-v0.16 files never had this token; PR #26 adds it to the
    allowlist."""
    import os
    import tempfile

    from structural_analysis.file_io import read_input_file

    body = (
        "TITLE\ngravity\nNODES 2\n"
        "1 0 0\n2 6 0\n"
        "MATERIALS 1\n1  0.02  8e-5  2e8\n"
        "ELEMENTS 1\n1 1 2 1 FRAME\n"
        "SUPPORTS 1\n1 1 1 1\n"
        "LOADS 0\n"
        "MEMBER_UDL 1\n1  0.0  10.0  gravity\n"
    )
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            f.write(body)
        m = read_input_file(tmp)
        ld = m.elements[0].member_loads[0]
        assert ld.coord_system == "gravity"
        assert ld.wy == 10.0
    finally:
        os.unlink(tmp)


# ── PR #26 — projection helper: gravity handling ─────────────────────


def test_project_gravity_on_horizontal_member_pure_transverse():
    wx_l, wy_l = _project_load_to_local(0.0, 10.0, "gravity", 1.0, 0.0)
    # Gravity mag = 10. Global components: (0, -10). On horizontal:
    # local = global → wy_l = -10, wx_l = 0.
    assert abs(wx_l - 0.0) < TOL
    assert abs(wy_l - (-10.0)) < TOL


def test_project_gravity_on_vertical_column_pure_axial():
    wx_l, wy_l = _project_load_to_local(0.0, 10.0, "gravity", 0.0, 1.0)
    # Gravity mag = 10 on +Y column. Global components: (0, -10).
    # T·(0,-10) for c=0,s=1: local x = s*(-10) = -10, local y = c*(-10) = 0.
    assert abs(wx_l - (-10.0)) < TOL
    assert abs(wy_l - 0.0) < TOL
