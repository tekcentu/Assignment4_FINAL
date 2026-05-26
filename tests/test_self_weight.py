"""Tests for v0.9.0 self-weight generation.

Covers the static-analysis self-weight path:
  - off by default (no behaviour change on existing fixtures)
  - vertical-equilibrium of reactions when on
  - inclined-frame projection through R.T @ p_local
  - truss endpoints lumped at uy DOFs
  - the loads are never persisted into the model
  - the effective-material override drives ρ for self-weight
  - .txt / .spa.json round-trip the include_self_weight flag
  - unknown ANALYSIS_OPTIONS keys raise ValueError
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.file_io import read_input_file
from structural_analysis.gui_common.file_writer import write_input_file
from structural_analysis.main import run_analysis
from structural_analysis.model import (
    STANDARD_GRAVITY,
    Material,
    Node,
    NodalLoad,
    Section,
    StructuralModel,
    Support,
    UniformDistributedLoad,
)


# ── helpers ────────────────────────────────────────────────────


def _steel_model_with_one_horizontal_beam(
    rho: float = 7850.0,
    A: float = 0.01,
    I: float = 1e-4,
    L: float = 5.0,
) -> StructuralModel:
    m = StructuralModel(title="cantilever-self-weight")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, L, 0.0)
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8,
                              alpha=1.2e-5, density=rho)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=A, I=I, depth=0.3)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=m.materials[1].E, A=A, I=I,
        alpha=m.materials[1].alpha, rho=rho,
        depth=0.3, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m


def _total_weight_kN(elem_rho, elem_A, elem_L) -> float:
    return elem_rho * elem_A * elem_L * STANDARD_GRAVITY / 1000.0


# ── 1. backward compat: flag off → no change ──────────────────


def test_flag_off_leaves_horizontal_cantilever_with_zero_reactions():
    m = _steel_model_with_one_horizontal_beam()
    # Default: include_self_weight is False.
    assert m.include_self_weight is False
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    # No applied loads → no support reactions.
    rxn = r.reactions[1]
    assert abs(rxn["ux"]) < 1e-9
    assert abs(rxn["uy"]) < 1e-9
    assert abs(rxn["rz"]) < 1e-9


# ── 2. horizontal cantilever: vertical reaction == total weight ──


def test_self_weight_on_horizontal_cantilever_reaction_equals_total_weight():
    m = _steel_model_with_one_horizontal_beam()
    m.include_self_weight = True
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"

    elem = m.elements[0]
    expected_weight_kN = _total_weight_kN(elem.rho, elem.A, 5.0)
    rxn = r.reactions[1]
    # Self-weight acts downward; the upward support reaction must
    # balance the total bar weight.
    assert rxn["uy"] == pytest.approx(expected_weight_kN, rel=1e-9)
    # Horizontal beam, gravity in -Y → no horizontal reaction.
    assert abs(rxn["ux"]) < 1e-9


# ── 3. matches a UDL of the same magnitude on a horizontal frame ──


def test_self_weight_equivalent_to_explicit_udl_on_horizontal_beam():
    """A horizontal frame member with self-weight on must produce the
    same internal forces as the same model with self-weight off and
    an explicit UDL of magnitude ``-ρ·A·g/1000`` in local +y.

    This pins both the sign convention and the equivalent-nodal-load
    form against the existing UDL FEF code path.
    """
    m_sw = _steel_model_with_one_horizontal_beam()
    m_sw.include_self_weight = True
    r_sw = run_analysis(m_sw, verbose=False)

    m_udl = _steel_model_with_one_horizontal_beam()
    w = -m_udl.elements[0].rho * m_udl.elements[0].A * STANDARD_GRAVITY / 1000.0
    m_udl.elements[0].member_loads.append(UniformDistributedLoad(wy=w))
    r_udl = run_analysis(m_udl, verbose=False)

    # Reactions at the fixed support must match.
    for dof in ("ux", "uy", "rz"):
        assert r_sw.reactions[1][dof] == pytest.approx(
            r_udl.reactions[1][dof], rel=1e-9, abs=1e-9,
        )


# ── 4. inclined frame: vertical equilibrium ────────────────────


def test_self_weight_inclined_frame_satisfies_vertical_equilibrium():
    """45°-inclined cantilever with self-weight on. Global vertical
    reactions must sum to the total bar weight regardless of the
    local axial-vs-transverse split that ``R.T @ p_local`` produces.
    """
    L_diag = 5.0
    dx = L_diag * (2 ** 0.5) / 2.0  # = L_diag · cos 45°
    dy = dx
    rho, A = 7850.0, 0.01

    m = StructuralModel(title="inclined-frame")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, dx, dy)
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=rho)
    m.sections[1] = Section(id=1, material_id=1, A=A, I=1e-4, depth=0.3)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=m.materials[1].E, A=A, I=1e-4, rho=rho,
        depth=0.3, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.include_self_weight = True

    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    expected = _total_weight_kN(rho, A, L_diag)
    # Sum of all uy reactions == total weight (gravity in -Y → all
    # weight reacted upward at the single fixed support).
    total_uy = sum(rxn.get("uy", 0.0) for rxn in r.reactions.values())
    assert total_uy == pytest.approx(expected, rel=1e-9)
    # No net horizontal external load.
    total_ux = sum(rxn.get("ux", 0.0) for rxn in r.reactions.values())
    assert abs(total_ux) < 1e-9


# ── 5. truss: half the bar weight lumped at each endpoint ──────


def test_self_weight_truss_lumped_at_endpoints_stable_model():
    """Stable horizontal truss bar: pin (node 1) + roller (node 2,
    uy restrained). Self-weight on. Sum of uy reactions equals the
    total bar weight; each support carries exactly half.
    """
    rho, A, L = 7850.0, 0.005, 4.0
    m = StructuralModel(title="truss-self-weight")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, L, 0.0)
    m.materials[1] = Material(id=1, E=2.1e8, density=rho)
    m.sections[1] = Section(id=1, material_id=1, A=A, I=0.0)
    m.elements.append(TrussElement2D(
        id=1, node_i=1, node_j=2,
        E=m.materials[1].E, A=A, rho=rho, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True)
    m.supports[2] = Support(node_id=2, uy=True)
    m.include_self_weight = True

    r = run_analysis(m, verbose=False)
    assert r.status == "ok"

    expected_total = _total_weight_kN(rho, A, L)
    half = expected_total / 2.0
    assert r.reactions[1]["uy"] == pytest.approx(half, rel=1e-9)
    assert r.reactions[2]["uy"] == pytest.approx(half, rel=1e-9)
    total_uy = sum(rxn.get("uy", 0.0) for rxn in r.reactions.values())
    assert total_uy == pytest.approx(expected_total, rel=1e-9)


def test_self_weight_truss_does_not_attach_member_loads():
    """Truss self-weight must go through global F at uy DOFs — not as
    a member-load object on the element. (TrussElement2D rejects
    member loads with TypeError.)"""
    rho, A, L = 7850.0, 0.005, 4.0
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, L, 0.0)
    m.materials[1] = Material(id=1, E=2.1e8, density=rho)
    m.sections[1] = Section(id=1, material_id=1, A=A, I=0.0)
    m.elements.append(TrussElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=A, rho=rho, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True)
    m.supports[2] = Support(node_id=2, uy=True)
    m.include_self_weight = True
    run_analysis(m, verbose=False)
    # The truss element must still have an empty member_loads list —
    # self-weight is solve-only, never persisted.
    assert m.elements[0].member_loads == []


# ── 6. not persisted to model loads ────────────────────────────


def test_self_weight_not_persisted_to_model_loads():
    m = _steel_model_with_one_horizontal_beam()
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-3.0))
    initial_nodal = list(m.nodal_loads)
    initial_member_loads = list(m.elements[0].member_loads)
    m.include_self_weight = True
    run_analysis(m, verbose=False)
    # Lists are not mutated by the solve.
    assert m.nodal_loads == initial_nodal
    assert m.elements[0].member_loads == initial_member_loads


# ── 7. effective-material override drives ρ ────────────────────


def test_self_weight_uses_effective_material_density_override():
    """Override material has a different density than the section's
    default. Self-weight reaction must reflect the *override* density.
    Pins the integration with the v0.8.x override field.
    """
    L, A = 5.0, 0.01
    default_rho = 7850.0  # steel-ish
    override_rho = 2500.0  # concrete-ish (denser/lighter than steel — checks the path)

    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, L, 0.0)
    m.materials[1] = Material(id=1, name="default", E=2.1e8, density=default_rho)
    m.materials[2] = Material(id=2, name="override", E=2.1e8, density=override_rho)
    m.sections[1] = Section(id=1, material_id=1, A=A, I=1e-4, depth=0.3)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=m.materials[2].E, A=A, I=1e-4,
        rho=override_rho,                 # element snapshot — set by
        depth=0.3,                          # commands/file_io from the
        section_id=1,                       # effective material
        material_id_override=2,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.include_self_weight = True

    r = run_analysis(m, verbose=False)
    expected = _total_weight_kN(override_rho, A, L)
    assert r.reactions[1]["uy"] == pytest.approx(expected, rel=1e-9)
    # Sanity: would have been wrong if we'd read the section default.
    not_expected = _total_weight_kN(default_rho, A, L)
    assert abs(r.reactions[1]["uy"] - not_expected) > 1e-3


# ── 8. .txt round-trip preserves include_self_weight ───────────


def test_txt_round_trip_preserves_include_self_weight_true():
    m = _steel_model_with_one_horizontal_beam()
    m.include_self_weight = True
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.txt")
        write_input_file(m, path)
        with open(path, "r") as f:
            text = f.read()
        assert "ANALYSIS_OPTIONS" in text
        assert "include_self_weight=true" in text
        reloaded = read_input_file(path)
    assert reloaded.include_self_weight is True


def test_txt_round_trip_preserves_include_self_weight_false_no_block():
    m = _steel_model_with_one_horizontal_beam()
    assert m.include_self_weight is False
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.txt")
        write_input_file(m, path)
        with open(path, "r") as f:
            text = f.read()
        # Default-off model must emit no ANALYSIS_OPTIONS block at all,
        # so existing fixtures stay byte-identical on round-trip.
        assert "ANALYSIS_OPTIONS" not in text
        reloaded = read_input_file(path)
    assert reloaded.include_self_weight is False


# ── 9. .spa.json round-trip preserves include_self_weight ─────


def test_spa_json_round_trip_preserves_include_self_weight():
    try:
        from structural_analysis.gui_qt.project_io import (
            Project,
            load_project_json,
            save_project_json,
        )
    except ImportError:
        pytest.skip("project_io requires GUI deps; not available here.")

    m = _steel_model_with_one_horizontal_beam()
    m.include_self_weight = True
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.spa.json")
        save_project_json(Project(model=m, title=m.title), path)
        project = load_project_json(path)
    assert project.model.include_self_weight is True


# ── 10. unknown ANALYSIS_OPTIONS key surfaces ─────────────────


def test_unknown_analysis_options_key_raises():
    text = (
        "TITLE\nx\n\n"
        "NODES 2\n1  0.0  0.0\n2  1.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  0.0  0.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
        "ANALYSIS_OPTIONS 1\nfoo=bar\n"
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bad.txt")
        with open(path, "w") as f:
            f.write(text)
        with pytest.raises(ValueError, match="foo"):
            read_input_file(path)


# ── 11. self-weight + moment releases (PR #17 review fix) ─────


def _released_beam_model(release_i: bool, release_j: bool):
    """Horizontal pinned-fixed beam with optional moment releases at
    each end. Identical to the helper above except for releases and a
    second roller support so the model is still stable when both ends
    are pinned/released."""
    rho, A, I, L = 7850.0, 0.01, 1e-4, 5.0
    m = StructuralModel(title="released-self-weight")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, L, 0.0)
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8,
                              alpha=1.2e-5, density=rho)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=A, I=I, depth=0.3)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=A, I=I, alpha=1.2e-5, rho=rho,
        depth=0.3, section_id=1,
        release_i=release_i, release_j=release_j,
    ))
    # Fixed at i, roller at j → stable for any release combo.
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.supports[2] = Support(node_id=2, ux=False, uy=True, rz=False)
    return m, rho, A, L


def test_self_weight_with_release_matches_equivalent_udl_with_release():
    """Self-weight on a released-end frame must produce the SAME
    reactions and end forces as an explicit UDL of the same magnitude
    on the same released-end frame. Without Schur condensation of the
    self-weight FEF, the released-end moment term would be silently
    dropped at assembly and the reactions would diverge from the
    UDL-path solution (which routes through the existing
    ``assembled_local_stiffness_and_load`` condensation)."""
    m_sw, rho, A, _ = _released_beam_model(release_i=False, release_j=True)
    m_sw.include_self_weight = True
    r_sw = run_analysis(m_sw, verbose=False)

    m_udl, _, _, _ = _released_beam_model(release_i=False, release_j=True)
    w = -rho * A * STANDARD_GRAVITY / 1000.0
    m_udl.elements[0].member_loads.append(UniformDistributedLoad(wy=w))
    r_udl = run_analysis(m_udl, verbose=False)

    for nid in (1, 2):
        # Roller nodes only carry the constrained DOFs in the dict.
        assert r_sw.reactions[nid].keys() == r_udl.reactions[nid].keys()
        for dof, val_sw in r_sw.reactions[nid].items():
            assert val_sw == pytest.approx(
                r_udl.reactions[nid][dof], rel=1e-9, abs=1e-9,
            ), f"reaction mismatch at node {nid} dof {dof}"
    # Recovered ``f_local`` is intentionally NOT compared: self-weight
    # is added to global F, never attached as a member load, so the
    # element's ``q = K·d − p_full`` recovery uses ``p_full = 0`` (vs.
    # ``p_full = p_UDL`` for the UDL model). That separate asymmetry is
    # tracked outside this PR; what the reviewer flagged — release-end
    # moment FEF dropped from F at assembly — is what this test pins.


def test_self_weight_released_end_has_zero_moment():
    """The released end's moment in the recovered local end-force
    vector must be ~0 (that is the definition of a moment release)."""
    m, _, _, _ = _released_beam_model(release_i=False, release_j=True)
    m.include_self_weight = True
    r = run_analysis(m, verbose=False)
    f_local = r.member_results[1]["f_local"]
    # q_local = [N_i, V_i, M_i, N_j, V_j, M_j]; release_j → q[5] ≈ 0.
    assert abs(f_local[5]) < 1e-8

    m2, _, _, _ = _released_beam_model(release_i=True, release_j=False)
    m2.include_self_weight = True
    r2 = run_analysis(m2, verbose=False)
    f2 = r2.member_results[1]["f_local"]
    # release_i → q[2] ≈ 0.
    assert abs(f2[2]) < 1e-8


def test_self_weight_with_release_still_satisfies_vertical_equilibrium():
    """ΣFy_reactions = total bar weight even with one end released —
    proves condensation redistributes the released moment FEF without
    leaking vertical load."""
    m, rho, A, L = _released_beam_model(release_i=False, release_j=True)
    m.include_self_weight = True
    r = run_analysis(m, verbose=False)
    W = _total_weight_kN(rho, A, L)
    sum_uy = sum(
        rxn["uy"] for rxn in r.reactions.values() if "uy" in rxn
    )
    assert sum_uy == pytest.approx(W, rel=1e-9, abs=1e-9)


def test_unknown_analysis_options_bool_value_raises():
    text = (
        "TITLE\nx\n\n"
        "NODES 2\n1  0.0  0.0\n2  1.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  0.0  0.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
        "ANALYSIS_OPTIONS 1\ninclude_self_weight=maybe\n"
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bad.txt")
        with open(path, "w") as f:
            f.write(text)
        with pytest.raises(ValueError, match="boolean"):
            read_input_file(path)
