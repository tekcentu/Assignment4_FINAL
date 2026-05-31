"""Tests for PR #31 model validation (pure logic, no Qt).

Covers:

* ``validate_model`` issue collection (orphans, unsupported
  disconnected components, single-truss free-end mechanism);
* false-positive guards for triangulated truss configurations and
  for frames that stabilise an otherwise free node;
* ``cases_with_loads`` active-case filtering, including self-weight
  attribution to ``model.self_weight_case`` when self-weight is on.
"""

from __future__ import annotations

import math

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.gui_common.validation import (
    ModelValidationResult,
    ValidationIssue,
    cases_with_loads,
    validate_model,
)
from structural_analysis.model import (
    LoadCase,
    Material,
    NodalLoad,
    Node,
    Section,
    StructuralModel,
    Support,
    UniformDistributedLoad,
)


# ── tiny model factories ────────────────────────────────────────────


def _seed_materials_sections(m: StructuralModel) -> None:
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )


def _frame(nid_i: int, nid_j: int, elem_id: int = 1) -> FrameElement2D:
    return FrameElement2D(
        id=elem_id, node_i=nid_i, node_j=nid_j,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
    )


def _truss(nid_i: int, nid_j: int, elem_id: int = 1) -> TrussElement2D:
    return TrussElement2D(
        id=elem_id, node_i=nid_i, node_j=nid_j,
        E=2.1e8, A=0.02, section_id=1,
    )


def _basic_supported_cantilever() -> StructuralModel:
    """1 ← fixed, 2 free, one frame element."""
    m = StructuralModel(title="t")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(_frame(1, 2))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m


# ── orphan node detection ────────────────────────────────────────────


def test_orphan_node_is_detected_as_warning():
    m = _basic_supported_cantilever()
    m.nodes[3] = Node(3, 10.0, 10.0)  # disconnected island
    res = validate_model(m)
    orphans = [
        i for i in res.issues
        if i.severity == "warning" and 3 in i.node_ids
        and "not connected" in i.message
    ]
    assert len(orphans) == 1


def test_connected_node_not_flagged_as_orphan():
    m = _basic_supported_cantilever()
    res = validate_model(m)
    msgs = [i.message for i in res.issues]
    assert not any("Node 1 is not connected" in m_ for m_ in msgs)
    assert not any("Node 2 is not connected" in m_ for m_ in msgs)


def test_orphan_does_not_block_solve_by_itself():
    """An orphan node alone is a warning, not an error, so a model
    that is otherwise valid is not flagged as having errors."""
    m = _basic_supported_cantilever()
    m.nodes[99] = Node(99, 20.0, 20.0)
    res = validate_model(m)
    assert not res.has_errors
    assert res.has_warnings


# ── disconnected unsupported components ──────────────────────────────


def test_disconnected_unsupported_component_is_error():
    m = StructuralModel(title="two-blobs")
    _seed_materials_sections(m)
    # Blob A: supported.
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 4.0, 0.0)
    m.elements.append(_frame(1, 2, elem_id=1))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    # Blob B: floating, no supports.
    m.nodes[5] = Node(5, 10.0, 10.0)
    m.nodes[6] = Node(6, 14.0, 10.0)
    m.elements.append(_frame(5, 6, elem_id=2))
    res = validate_model(m)
    errs = [
        i for i in res.issues
        if i.severity == "error" and "no supports" in i.message
    ]
    assert len(errs) == 1
    assert set(errs[0].node_ids) == {5, 6}
    assert 2 in errs[0].element_ids


def test_supported_component_not_flagged():
    m = _basic_supported_cantilever()
    res = validate_model(m)
    # Supported single-component frame: no "no supports" error.
    msgs = [i.message for i in res.issues if i.severity == "error"]
    assert not any("no supports" in s for s in msgs)


def test_disconnected_but_supported_component_is_ok():
    """Two independent supported sub-frames are individually stable
    and must not be flagged."""
    m = StructuralModel(title="two-stable-blobs")
    _seed_materials_sections(m)
    # Blob A
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 4.0, 0.0)
    m.elements.append(_frame(1, 2, elem_id=1))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    # Blob B — also supported.
    m.nodes[5] = Node(5, 10.0, 0.0)
    m.nodes[6] = Node(6, 14.0, 0.0)
    m.elements.append(_frame(5, 6, elem_id=2))
    m.supports[5] = Support(node_id=5, ux=True, uy=True, rz=True)
    res = validate_model(m)
    assert not res.has_errors


# ── truss free-end mechanism ────────────────────────────────────────


def test_single_truss_free_end_is_error():
    """Truss from supported node 1 to free node 2: node 2 has an
    unconstrained transverse DOF — should be flagged as error."""
    m = StructuralModel(title="single-truss")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(_truss(1, 2, elem_id=1))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if i.severity == "error" and "unconstrained transverse DOF" in i.message
    ]
    assert len(mech) == 1
    assert mech[0].node_ids == [2]
    assert 1 in mech[0].element_ids


def test_single_truss_supported_end_is_ok():
    """If both endpoints are supported, no mechanism."""
    m = StructuralModel(title="t")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(_truss(1, 2, elem_id=1))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[2] = Support(node_id=2, ux=True, uy=True, rz=False)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
    ]
    assert not mech


def test_two_collinear_trusses_free_node_is_error():
    """Two collinear trusses meeting at a free midpoint span only 1-D
    → transverse mechanism remains."""
    m = StructuralModel(title="collinear")
    _seed_materials_sections(m)
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 4.0, 0.0),  # free midpoint, both bars horizontal
        3: Node(3, 8.0, 0.0),
    }
    m.elements.append(_truss(1, 2, elem_id=1))
    m.elements.append(_truss(2, 3, elem_id=2))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[3] = Support(node_id=3, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
        and 2 in i.node_ids
    ]
    assert len(mech) == 1


def test_two_noncollinear_trusses_free_node_is_ok():
    """Two non-collinear trusses at a free node span 2-D and provide
    translational stability — no false positive."""
    m = StructuralModel(title="v-truss")
    _seed_materials_sections(m)
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 4.0, 0.0),
        3: Node(3, 2.0, 3.0),  # apex of V
    }
    m.elements.append(_truss(1, 3, elem_id=1))
    m.elements.append(_truss(2, 3, elem_id=2))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[2] = Support(node_id=2, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
    ]
    assert not mech, (
        f"two non-collinear trusses must not trigger a mechanism "
        f"warning, got: {[i.message for i in mech]}"
    )


def test_three_truss_triangulated_node_is_ok():
    """Three trusses meeting at one free node in non-collinear
    directions — classic triangulated joint; do not flag."""
    m = StructuralModel(title="tri")
    _seed_materials_sections(m)
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 4.0, 0.0),
        3: Node(3, 2.0, 3.0),
        4: Node(4, 2.0, 1.0),  # interior free node
    }
    m.elements.append(_truss(1, 4, elem_id=1))
    m.elements.append(_truss(2, 4, elem_id=2))
    m.elements.append(_truss(3, 4, elem_id=3))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[2] = Support(node_id=2, ux=True, uy=True, rz=True)
    m.supports[3] = Support(node_id=3, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
    ]
    assert not mech


def test_frame_element_stabilises_free_end_not_flagged():
    """A free node with a frame member is rotationally + transversely
    stabilised — no truss-mechanism warning."""
    m = _basic_supported_cantilever()  # 1 fixed → 2 free, frame
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
    ]
    assert not mech


def test_mixed_truss_and_frame_at_free_node_not_flagged():
    """If a node has at least one frame element, transverse stiffness
    is present even if other incident elements are trusses."""
    m = StructuralModel(title="mixed")
    _seed_materials_sections(m)
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 4.0, 0.0),
        3: Node(3, 8.0, 0.0),
    }
    m.elements.append(_frame(1, 2, elem_id=1))
    m.elements.append(_truss(2, 3, elem_id=2))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[3] = Support(node_id=3, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
        and 2 in i.node_ids
    ]
    assert not mech


# ── basic sanity (formerly "fatal" list) ─────────────────────────────


def test_missing_materials_is_error():
    m = StructuralModel(title="empty")
    res = validate_model(m)
    errs = [i.message for i in res.issues if i.severity == "error"]
    assert any("No materials" in s for s in errs)


def test_no_supports_is_warning_not_error():
    m = StructuralModel(title="empty-supports")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 1.0, 0.0)}
    m.elements.append(_frame(1, 2))
    res = validate_model(m)
    # "no supports" itself is a warning per the legacy contract.
    no_sup_warn = [
        i for i in res.issues
        if i.severity == "warning" and "No supports" in i.message
    ]
    assert no_sup_warn
    # But disconnected-component check will turn this into an error
    # because no node carries any restraint.  That's correct — model
    # IS unstable.
    no_sup_err = [
        i for i in res.issues
        if i.severity == "error" and "no supports" in i.message
    ]
    assert no_sup_err


# ── format_report ────────────────────────────────────────────────────


def test_format_report_empty_when_no_issues():
    assert ModelValidationResult().format_report() == ""


def test_format_report_groups_warnings_and_errors():
    r = ModelValidationResult(issues=[
        ValidationIssue(severity="warning", message="W1"),
        ValidationIssue(severity="error", message="E1"),
    ])
    text = r.format_report()
    assert "Validation warnings:" in text
    assert "W1" in text
    assert "Validation errors:" in text
    assert "E1" in text
    assert "blocked" in text.lower()


# ── cases_with_loads filtering ───────────────────────────────────────


def _multi_case_model() -> StructuralModel:
    m = StructuralModel(title="cases")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(_frame(1, 2))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.load_cases["DEAD"] = LoadCase(name="DEAD")
    m.load_cases["LIVE"] = LoadCase(name="LIVE")
    m.load_cases["WIND"] = LoadCase(name="WIND")
    m.load_cases["THERMAL"] = LoadCase(name="THERMAL")
    # Disable DEFAULT so empty-DEFAULT doesn't pollute the result.
    m.load_cases["DEFAULT"].enabled = False
    return m


def test_cases_with_loads_returns_only_loaded_cases():
    m = _multi_case_model()
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEAD"))
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-5.0, load_case="LIVE"),
    )
    active = cases_with_loads(m)
    assert active == ["DEAD", "LIVE"]


def test_cases_with_loads_excludes_empty_cases():
    """WIND and THERMAL are enabled but carry no loads → not active."""
    m = _multi_case_model()
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEAD"))
    active = cases_with_loads(m)
    assert "WIND" not in active
    assert "THERMAL" not in active
    assert "DEAD" in active


def test_cases_with_loads_includes_member_load_case():
    m = _multi_case_model()
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-2.0, load_case="WIND"),
    )
    active = cases_with_loads(m)
    assert active == ["WIND"]


def test_cases_with_loads_includes_self_weight_case():
    """Self-weight enabled + assigned to DEAD makes DEAD active even
    when no manual loads exist."""
    m = _multi_case_model()
    m.include_self_weight = True
    m.self_weight_case = "DEAD"
    active = cases_with_loads(m)
    assert "DEAD" in active
    # LIVE has nothing tied to it.
    assert "LIVE" not in active


def test_cases_with_loads_self_weight_disabled_does_not_count():
    m = _multi_case_model()
    m.include_self_weight = False
    m.self_weight_case = "DEAD"
    active = cases_with_loads(m)
    assert "DEAD" not in active


def test_cases_with_loads_excludes_disabled_cases():
    """A disabled case with loads attached still doesn't count — the
    user has explicitly told the solver to skip it."""
    m = _multi_case_model()
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEAD"))
    m.load_cases["DEAD"].enabled = False
    active = cases_with_loads(m)
    assert "DEAD" not in active


def test_cases_with_loads_returns_empty_when_no_loads():
    m = _multi_case_model()
    active = cases_with_loads(m)
    assert active == []


def test_cases_with_loads_settlement_makes_all_enabled_cases_active():
    """Support settlement isn't case-tagged, so any non-zero
    prescribed displacement makes every enabled case active —
    otherwise the legacy ``q2a_settlement.txt`` example (zero applied
    loads, two cm of node-E settlement) would be filtered out and
    "Solve All" would refuse to run."""
    m = _multi_case_model()
    # No loads — only a settlement.
    m.supports[1] = Support(
        node_id=1, ux=True, uy=True, rz=True,
        settle_uy=-0.002,
    )
    active = cases_with_loads(m)
    # Every enabled case becomes active because the settlement is
    # case-independent and would drive each solve.
    assert "DEAD" in active and "LIVE" in active
    assert "WIND" in active and "THERMAL" in active


def test_cases_with_loads_zero_settlement_does_not_count():
    """A settle_uy of 0.0 is the same as no settlement — must NOT
    flag every case as active."""
    m = _multi_case_model()
    m.supports[1] = Support(
        node_id=1, ux=True, uy=True, rz=True,
        settle_uy=0.0,
    )
    assert cases_with_loads(m) == []


def test_cases_with_loads_combines_all_sources():
    """Nodal + member + self-weight, three different cases — all three
    must surface in the active list."""
    m = _multi_case_model()
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEAD"))
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-2.0, load_case="LIVE"),
    )
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-1.0, load_case="WIND"),
    )
    m.include_self_weight = True
    m.self_weight_case = "DEAD"
    active = cases_with_loads(m)
    assert set(active) == {"DEAD", "LIVE", "WIND"}
