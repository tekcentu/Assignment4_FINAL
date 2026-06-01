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


def _double_pin_frame(nid_i: int, nid_j: int, elem_id: int = 1) -> FrameElement2D:
    """Frame element with releases at both ends — equivalent to a truss after condensation."""
    return FrameElement2D(
        id=elem_id, node_i=nid_i, node_j=nid_j,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_i=True, release_j=True,
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


# ── double-pinned frame (hinge/release) mechanism detection ──────────


def test_double_pin_frame_free_end_is_error():
    """A frame with releases at both ends (double-pin) connecting a supported
    node to a free node is mechanically a truss — the free node has an
    unconstrained transverse DOF and must be flagged as an error."""
    m = StructuralModel(title="double-pin")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(_double_pin_frame(1, 2, elem_id=1))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if i.severity == "error" and "unconstrained transverse DOF" in i.message
    ]
    assert len(mech) == 1
    assert mech[0].node_ids == [2]
    assert 1 in mech[0].element_ids


def test_single_pin_at_free_end_only_is_valid():
    """A frame with a release at only the free end (release_j=True) still
    carries transverse shear at that node — must NOT be flagged."""
    m = StructuralModel(title="single-pin-free")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_j=True,  # pin at the free end only
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
    ]
    assert not mech, (
        f"single-pin at free end must not trigger mechanism: "
        f"{[i.message for i in mech]}"
    )


def test_single_pin_at_supported_end_is_error():
    """A frame with release_i=True (pin at the supported far end) leaves the
    free node N with a singular stiffness — the element can rotate as a rigid
    body about the pin at M regardless of M's rz support, because the pin
    decouples element rotation from node rotation.  Must be flagged as error."""
    m = StructuralModel(title="single-pin-support")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_i=True,  # pin at the supported end → rigid-body rotation about node 1
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if i.severity == "error" and "unconstrained transverse DOF" in i.message
    ]
    assert len(mech) == 1, (
        f"single-pin at supported end must be a mechanism error "
        f"(rz support at M doesn't prevent rigid-body rotation via the pin): "
        f"{[i.message for i in res.issues]}"
    )
    assert mech[0].node_ids == [2]


def test_two_collinear_double_pin_frames_at_free_midnode_is_error():
    """Two collinear double-pinned frames at a free midpoint are both
    axial-only (truss equivalents) — transverse mechanism remains."""
    m = StructuralModel(title="collinear-dpin")
    _seed_materials_sections(m)
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 4.0, 0.0),  # free midpoint
        3: Node(3, 8.0, 0.0),
    }
    m.elements.append(_double_pin_frame(1, 2, elem_id=1))
    m.elements.append(_double_pin_frame(2, 3, elem_id=2))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[3] = Support(node_id=3, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
        and 2 in i.node_ids
    ]
    assert len(mech) == 1


def test_double_pin_frame_mixed_with_truss_collinear_is_error():
    """One double-pin frame and one truss, both collinear, at a free node —
    both are axial-only and collinear, so the transverse mechanism persists."""
    m = StructuralModel(title="mixed-axial-collinear")
    _seed_materials_sections(m)
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 4.0, 0.0),  # free node
        3: Node(3, 8.0, 0.0),
    }
    m.elements.append(_double_pin_frame(1, 2, elem_id=1))
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


def test_double_pin_frame_with_noncollinear_truss_is_valid():
    """A double-pin frame + a non-collinear truss at a free node: directions
    span 2-D so translational stability exists — must NOT be flagged."""
    m = StructuralModel(title="dpin-noncollinear")
    _seed_materials_sections(m)
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 4.0, 0.0),  # free apex
        3: Node(3, 4.0, 3.0),
    }
    # double-pin horizontal + truss vertical → non-collinear
    m.elements.append(_double_pin_frame(1, 2, elem_id=1))
    m.elements.append(_truss(3, 2, elem_id=2))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[3] = Support(node_id=3, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
        and 2 in i.node_ids
    ]
    assert not mech, (
        f"non-collinear axial-only members must not trigger mechanism: "
        f"{[i.message for i in mech]}"
    )


# ── single-release rigid-body-rotation mechanism detection ───────────


def test_single_release_at_far_supported_end_is_error():
    """Frame E from M (translation-only support: ux+uy, no rz) to free N,
    with release at M side (pin at M).  E can rotate as a rigid body about
    M → mechanism at N, must be flagged."""
    m = StructuralModel(title="single-release")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    # release_i=True → pin at node 1 (M), full connection at node 2 (N)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_i=True,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=False)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if i.severity == "error" and "unconstrained transverse DOF" in i.message
    ]
    assert len(mech) == 1
    assert mech[0].node_ids == [2]
    assert 1 in mech[0].element_ids


def test_single_release_at_far_end_rz_support_still_error():
    """Even when the far supported node has rz=True, the pin in the element
    decouples element rotation from node rotation — mechanism persists."""
    m = StructuralModel(title="single-release-rz")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_i=True,  # pin at node 1 — rz support at 1 doesn't help
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if i.severity == "error" and "unconstrained transverse DOF" in i.message
    ]
    assert len(mech) == 1
    assert mech[0].node_ids == [2]


def test_single_release_at_free_end_only_is_not_flagged_by_rigid_body_check():
    """Pin at the free end (release_j=True, release_i=False) — the element is
    rigidly connected to the supported far end, so no rigid-body rotation is
    possible.  Must NOT be flagged by the new check."""
    m = StructuralModel(title="pin-at-free-end")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_j=True,  # pin at free node 2 only
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=False)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if i.severity == "error" and "unconstrained transverse DOF" in i.message
    ]
    assert not mech, (
        f"pin at free end only must not trigger rigid-body check: "
        f"{[i.message for i in mech]}"
    )


def test_normal_cantilever_not_flagged_by_rigid_body_check():
    """Frame with no releases at all — standard fixed cantilever.
    The rigid-body check must not flag it."""
    m = _basic_supported_cantilever()
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
    ]
    assert not mech


def test_single_release_two_incident_elements_not_flagged():
    """Free node N with TWO incident elements (one single-release, one normal)
    — multi-element topology; rigid-body check is deferred and must not fire."""
    m = StructuralModel(title="two-elem")
    _seed_materials_sections(m)
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 4.0, 0.0),  # free node with 2 incident elements
        3: Node(3, 8.0, 0.0),
    }
    # E1: pin at node 1 (far, supported), full connection at node 2 (free)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_i=True,
    ))
    # E2: normal frame element — provides bending coupling at node 2
    m.elements.append(FrameElement2D(
        id=2, node_i=2, node_j=3,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=False)
    m.supports[3] = Support(node_id=3, ux=True, uy=True, rz=True)
    res = validate_model(m)
    rigid_mech = [
        i for i in res.issues
        if "rigid body" in i.message.lower() or (
            "unconstrained transverse DOF" in i.message and 2 in i.node_ids
        )
    ]
    assert not rigid_mech, (
        f"two-element topology must not trigger single-element rigid-body "
        f"check: {[i.message for i in rigid_mech]}"
    )


def test_released_beam_between_two_supports_is_valid():
    """A frame element with releases at BOTH ends, between two supported
    nodes (no free node) — a simply-supported beam modelled with end pins.
    Neither endpoint is free, so no mechanism check should fire."""
    m = StructuralModel(title="released-beam-between-supports")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(_double_pin_frame(1, 2, elem_id=1))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=False)
    m.supports[2] = Support(node_id=2, ux=True, uy=True, rz=False)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
    ]
    assert not mech, (
        f"double-pin beam between two supports must not be flagged: "
        f"{[i.message for i in mech]}"
    )


def test_internal_hinge_continuous_beam_not_flagged():
    """Gerber-style continuous beam: nodes 1 and 3 supported, node 2 is
    an internal hinge between two collinear frame elements (release_j at
    elem 1, release_i at elem 2).  Node 2 has TWO incident elements
    contributing translational stiffness — must NOT be flagged."""
    m = StructuralModel(title="internal-hinge")
    _seed_materials_sections(m)
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 4.0, 0.0),  # internal hinge (free node)
        3: Node(3, 8.0, 0.0),
    }
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_j=True,
    ))
    m.elements.append(FrameElement2D(
        id=2, node_i=2, node_j=3,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_i=True,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[3] = Support(node_id=3, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
    ]
    assert not mech, (
        f"internal hinge between two stable supports must not be flagged: "
        f"{[i.message for i in mech]}"
    )


# ── corbel / indirect-support release mechanism ──────────────────────


def _corbel_column_model(
    *,
    release_at_column_side: bool = False,
    release_at_free_end: bool = False,
) -> StructuralModel:
    """Column 1→3→2 (fixed at node 1) + corbel 3→4.

    Node 1: fixed support.
    Node 3: column junction — NOT directly supported but part of the
            stable column connected to the fixed base.
    Node 2: column top (free).
    Node 4: corbel tip (free leaf — only one incident element).

    ``release_at_column_side=True`` sets ``release_i=True`` on the corbel
    element at the node-3 (column-junction) side.
    ``release_at_free_end=True`` sets ``release_j=True`` at the tip (node 4).
    """
    m = StructuralModel(title="corbel-column")
    _seed_materials_sections(m)
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        3: Node(3, 0.0, 2.0),
        2: Node(2, 0.0, 4.0),
        4: Node(4, 2.0, 2.0),
    }
    m.elements.append(_frame(1, 3, elem_id=1))
    m.elements.append(_frame(3, 2, elem_id=2))
    m.elements.append(FrameElement2D(
        id=3, node_i=3, node_j=4,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_i=release_at_column_side,
        release_j=release_at_free_end,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m


def test_corbel_indirect_junction_release_is_error():
    """Column 1-3-2 (fixed at 1) + corbel 3→4 with release_i at the column
    junction (node 3).  Node 3 is NOT directly supported — it is stabilised
    indirectly via the column.  Node 4 is the free leaf.

    The element can rotate as a rigid body about the pin at node 3, so node 4
    has an unconstrained transverse DOF and must be flagged."""
    m = _corbel_column_model(release_at_column_side=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if i.severity == "error" and "unconstrained transverse DOF" in i.message
    ]
    assert len(mech) == 1, (
        f"expected 1 mechanism error, got {len(mech)}: "
        f"{[i.message for i in mech]}"
    )
    assert mech[0].node_ids == [4], f"problem node must be 4, got {mech[0].node_ids}"
    assert 3 in mech[0].element_ids, f"element 3 (corbel) must be listed"


def test_corbel_without_release_is_valid():
    """Same column + corbel geometry, but no moment release on the corbel.
    Normal fixed-base frame cantilever — must NOT be flagged."""
    m = _corbel_column_model(release_at_column_side=False)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
    ]
    assert not mech, (
        f"unreleased corbel must not trigger mechanism check: "
        f"{[i.message for i in mech]}"
    )


def test_corbel_free_end_release_only_is_not_flagged():
    """Corbel 3→4 with release_j=True (pin at the free tip, node 4) and
    a rigid connection at the column-side node 3.  No rigid-body rotation is
    possible — the base is stiff.  Must NOT be flagged by this check."""
    m = _corbel_column_model(release_at_free_end=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if "unconstrained transverse DOF" in i.message
    ]
    assert not mech, (
        f"free-end-only release must not trigger rigid-body check: "
        f"{[i.message for i in mech]}"
    )


def test_corbel_reverse_orientation_end_release_is_error():
    """Same corbel mechanism expressed with the REVERSED element orientation:
    element drawn from free tip (node 4 = node_i) → column junction (node 3 = node_j),
    with release_j=True (END release at node_j=3, the column side).

    Equivalent to a user clicking the free tip first and the column junction
    second, then checking 'Moment release at end (j)' in the dialog.
    Must be detected identically to the 3→4 START case."""
    m = StructuralModel(title="corbel-reverse")
    _seed_materials_sections(m)
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        3: Node(3, 0.0, 2.0),
        2: Node(2, 0.0, 4.0),
        4: Node(4, 2.0, 2.0),
    }
    m.elements.append(_frame(1, 3, elem_id=1))
    m.elements.append(_frame(3, 2, elem_id=2))
    # Corbel: drawn free-tip → column-junction, release at end (node_j=3)
    m.elements.append(FrameElement2D(
        id=3, node_i=4, node_j=3,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_i=False, release_j=True,  # END release = pin at node_j=3
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    res = validate_model(m)
    mech = [
        i for i in res.issues
        if i.severity == "error" and "unconstrained transverse DOF" in i.message
    ]
    assert len(mech) == 1, (
        f"reversed-orientation corbel (4→3 END) must still be detected: "
        f"{[i.message for i in mech]}"
    )
    assert mech[0].node_ids == [4], f"problem node must be 4, got {mech[0].node_ids}"
    assert 3 in mech[0].element_ids


def test_corbel_text_format_start_release_is_error():
    """Parse a model in the native .txt file format that matches the GUI-exported
    model: column 1-3-2 (fixed at 1) + corbel '3 3 4 1 FRAME START'.

    This is the exact text a user would get from File→Export or from the
    model_txt field in a .spa.json.  The parsed StructuralModel must carry
    release_i=True on the corbel, and validate_model must flag it."""
    import tempfile, os
    from structural_analysis.file_io import read_input_file

    model_txt = """\
TITLE
Corbel column mechanism test

NODES 4
1  0.0  0.0
2  0.0  4.0
3  0.0  2.0
4  2.0  2.0

MATERIALS 1
1  210000000.0  1.2e-05  7850.0  Steel

SECTIONS 1
1  1  0.02  8e-05  0.3  S

ELEMENTS 3
1  1  3  1  FRAME
2  3  2  1  FRAME
3  3  4  1  FRAME START

SUPPORTS 1
1  1  1  1

LOADS 1
4  0.0  -10.0  0.0
"""
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    try:
        os.write(fd, model_txt.encode())
        os.close(fd)
        model = read_input_file(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    # Verify the file was parsed correctly.
    corbel = next(e for e in model.elements if e.node_i == 3 and e.node_j == 4)
    assert getattr(corbel, "release_i", False), (
        "file_io must set release_i=True for 'FRAME START'"
    )
    assert not getattr(corbel, "release_j", False), (
        "release_j must remain False for a START-only release"
    )

    # Now run the same validate_model that the Solve button calls.
    res = validate_model(model)
    mech = [
        i for i in res.issues
        if i.severity == "error" and "unconstrained transverse DOF" in i.message
    ]
    assert len(mech) == 1, (
        f"text-format corbel with 'FRAME START' must be detected: "
        f"{[i.message for i in mech]}"
    )
    assert 4 in mech[0].node_ids
    assert corbel.id in mech[0].element_ids


# ── basic sanity (formerly "fatal" list) ─────────────────────────────


def test_missing_materials_is_error():
    m = StructuralModel(title="empty")
    res = validate_model(m)
    errs = [i.message for i in res.issues if i.severity == "error"]
    assert any("No materials" in s for s in errs)


def test_element_referencing_missing_node_does_not_crash():
    """Regression (Gemini PR #31 HIGH finding): an element whose
    ``node_j`` doesn't exist in ``model.nodes`` used to crash
    ``_find_truss_mechanisms`` with a ``KeyError`` because the
    geometry checks dereferenced the invalid id directly.  The
    validator must instead surface the broken reference as a basic-
    sanity error and stop before geometry checks run."""
    m = StructuralModel(title="bad")
    _seed_materials_sections(m)
    m.nodes = {1: Node(1, 0.0, 0.0)}  # only node 1 exists
    # Truss references node 999 which is NOT in model.nodes — would
    # have crashed _bar_direction_from_node previously.
    m.elements.append(_truss(1, 999, elem_id=1))
    # Must not raise.
    res = validate_model(m)
    errs = [i.message for i in res.issues if i.severity == "error"]
    assert any("missing end node 999" in s for s in errs)
    # And no mechanism check should have run (those would crash too).
    assert not any("transverse DOF" in s for s in errs)


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
