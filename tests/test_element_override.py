"""Tests for the element-level material override (PR #16).

Covers:

1. ``test_old_model_no_override_uses_section_default`` — a saved-on-disk
   model with no override token loads with ``material_id_override is None``
   on every element and inherits E/α/ρ from the section's default material.
2. ``test_element_override_changes_effective_E_alpha_rho`` — placing an
   override on an element re-pulls E, α, **and** ρ from the override
   material; sibling elements on the same section stay on the section
   default.
3. ``test_round_trip_preserves_material_override_txt_and_json`` — writing
   a model with one overridden element to ``.txt`` (and ``.spa.json``)
   and reading it back preserves the override; un-overridden elements
   write **no** trailing token (byte-equal to the pre-PR output line).
4. ``test_material_edit_propagates_only_to_effective_users`` — editing a
   material refreshes only the elements whose *effective* material id
   matches; overridden elements pointing at a different material are
   untouched.
5. ``test_section_edit_propagates_geometry_but_not_overrides`` — editing
   a section always updates A / I / depth on every element on that
   section, but only refreshes E / α / ρ on the non-overridden ones.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from structural_analysis.element import FrameElement2D
from structural_analysis.file_io import read_input_file
from structural_analysis.gui_common.commands import (
    AddOrUpdateMaterialCmd,
    AddOrUpdateSectionCmd,
    DeleteMaterialCmd,
    UpdateElementCmd,
)
from structural_analysis.gui_common.file_writer import write_input_file
from structural_analysis.model import (
    Material,
    Node,
    Section,
    StructuralModel,
    Support,
    effective_material,
)


def _two_material_model() -> StructuralModel:
    """Two materials (M1, M2) and one section S using M1 as default.

    Two frame elements sit on S: elem 1 and elem 2. Both span from a
    pinned support to a free node, so they are independent and any
    override on one does not silently propagate through shared DOFs.
    """
    m = StructuralModel(title="override test")
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 5.0, 0.0),
        3: Node(3, 0.0, 5.0),
        4: Node(4, 5.0, 5.0),
    }
    m.materials = {
        1: Material(id=1, name="M1", E=2.0e8, alpha=1.0e-5,
                    density=7850.0, nu=0.30),
        2: Material(id=2, name="M2", E=3.0e7, alpha=2.0e-5,
                    density=2400.0, nu=0.20),
    }
    m.sections = {
        1: Section(id=1, name="S", material_id=1,
                   A=0.01, I=1.0e-4, depth=0.1, width=0.1),
    }
    m.elements = [
        FrameElement2D(
            id=1, node_i=1, node_j=2,
            E=m.materials[1].E, A=m.sections[1].A, I=m.sections[1].I,
            alpha=m.materials[1].alpha, depth=m.sections[1].depth,
            rho=m.materials[1].density,
            section_id=1,
        ),
        FrameElement2D(
            id=2, node_i=3, node_j=4,
            E=m.materials[1].E, A=m.sections[1].A, I=m.sections[1].I,
            alpha=m.materials[1].alpha, depth=m.sections[1].depth,
            rho=m.materials[1].density,
            section_id=1,
        ),
    ]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=True),
        3: Support(3, ux=True, uy=True, rz=True),
    }
    return m


def _round_trip_txt(model: StructuralModel) -> tuple[StructuralModel, str]:
    """Round-trip ``model`` via the .txt writer/reader and return both the
    reconstructed model and the written-on-disk .txt text (so tests can
    inspect the token format)."""
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        write_input_file(model, tmp)
        with open(tmp, "r", encoding="utf-8") as f:
            text = f.read()
        m2 = read_input_file(tmp)
    finally:
        os.unlink(tmp)
    return m2, text


# ── 1. Old model with no override uses section default ────────────


def test_old_model_no_override_uses_section_default():
    """Every element in inputs/q2a_settlement.txt loads with no override,
    and its E/α/ρ match the resolved section-default material."""
    m = read_input_file("inputs/q2a_settlement.txt")
    assert m.elements, "fixture must have at least one element"
    for elem in m.elements:
        assert elem.material_id_override is None, (
            f"elem {elem.id}: expected no override, got "
            f"{elem.material_id_override!r}"
        )
        sec = m.sections[elem.section_id]
        default_mat = m.materials[sec.material_id]
        assert elem.E == pytest.approx(default_mat.E)
        assert elem.alpha == pytest.approx(default_mat.alpha)
        # rho on file_io-loaded elements should track density too.
        assert elem.rho == pytest.approx(default_mat.density)
        # And the resolver returns the same Material object.
        assert effective_material(m, elem) is default_mat


# ── 2. Override changes E, α, ρ on the element ─────────────────────


def test_element_override_changes_effective_E_alpha_rho():
    m = _two_material_model()
    # Apply override on element 1 — switch to M2.
    UpdateElementCmd(
        elem_id=1, section_id=1, kind="frame",
        material_override_id=2,
    ).do(m)

    e1 = next(e for e in m.elements if e.id == 1)
    e2 = next(e for e in m.elements if e.id == 2)
    M1 = m.materials[1]
    M2 = m.materials[2]

    # Overridden element pulls from M2.
    assert e1.material_id_override == 2
    assert e1.E == pytest.approx(M2.E)
    assert e1.alpha == pytest.approx(M2.alpha)
    assert e1.rho == pytest.approx(M2.density)

    # Geometry (A, I, depth) still from section S.
    assert e1.A == pytest.approx(m.sections[1].A)
    assert e1.I == pytest.approx(m.sections[1].I)
    assert e1.depth == pytest.approx(m.sections[1].depth)

    # Sibling on the same section stays on M1 — no spillover.
    assert e2.material_id_override is None
    assert e2.E == pytest.approx(M1.E)
    assert e2.alpha == pytest.approx(M1.alpha)
    assert e2.rho == pytest.approx(M1.density)


# ── 3. Round-trip preserves the override (txt + json) ──────────────


def test_round_trip_preserves_material_override_txt_and_json():
    m = _two_material_model()
    UpdateElementCmd(
        elem_id=1, section_id=1, kind="frame",
        material_override_id=2,
    ).do(m)

    m2, written_txt = _round_trip_txt(m)

    # Reconstructed element keeps the override id.
    e1b = next(e for e in m2.elements if e.id == 1)
    e2b = next(e for e in m2.elements if e.id == 2)
    assert e1b.material_id_override == 2
    assert e2b.material_id_override is None

    # The non-overridden element's ELEMENTS line must NOT carry the
    # trailing token (byte-compat with pre-PR output).
    lines = [ln for ln in written_txt.splitlines()
             if ln and not ln.startswith("#")]
    elem_lines = []
    in_block = False
    for ln in lines:
        if ln.startswith("ELEMENTS"):
            in_block = True
            continue
        if in_block:
            # Stop at next block header or blank.
            if not ln.startswith(("1 ", "2 ")):
                break
            elem_lines.append(ln.strip())
    assert any("material_override_id=2" in ln for ln in elem_lines), (
        f"expected the overridden element line to carry the token; "
        f"got {elem_lines!r}"
    )
    assert any(("material_override_id" not in ln) for ln in elem_lines), (
        "the non-overridden element line must NOT carry the token"
    )

    # And the post-roundtrip E/α/ρ resolution matches what we set up.
    M2 = m.materials[2]
    M1 = m.materials[1]
    assert e1b.E == pytest.approx(M2.E)
    assert e1b.alpha == pytest.approx(M2.alpha)
    assert e1b.rho == pytest.approx(M2.density)
    assert e2b.E == pytest.approx(M1.E)
    assert e2b.alpha == pytest.approx(M1.alpha)
    assert e2b.rho == pytest.approx(M1.density)

    # JSON path — project_io stores the canonical .txt verbatim, so if
    # .txt round-trips correctly, .spa.json round-trips for free. Verify
    # by walking the same code path the GUI uses.
    from structural_analysis.gui_qt.project_io import (
        Project, save_project_json, load_project_json,
    )
    fd, jpath = tempfile.mkstemp(suffix=".spa.json")
    os.close(fd)
    try:
        save_project_json(Project(model=m), jpath)
        proj = load_project_json(jpath)
    finally:
        os.unlink(jpath)
    e1c = next(e for e in proj.model.elements if e.id == 1)
    e2c = next(e for e in proj.model.elements if e.id == 2)
    assert e1c.material_id_override == 2
    assert e2c.material_id_override is None


# ── 4. Material edit propagates only to effective users ────────────


def test_material_edit_propagates_only_to_effective_users():
    """Three elements on section S (default M1):
       A: no override
       B: override = M2
       C: override = M1 (explicitly the section default)

    Editing M1's E must refresh A and C but leave B untouched.
    Editing M2's E must refresh B but leave A and C untouched.
    """
    m = _two_material_model()
    # Add a third element + node pair.
    m.nodes[5] = Node(5, 0.0, 10.0)
    m.nodes[6] = Node(6, 5.0, 10.0)
    m.supports[5] = Support(5, ux=True, uy=True, rz=True)
    m.elements.append(FrameElement2D(
        id=3, node_i=5, node_j=6,
        E=m.materials[1].E, A=m.sections[1].A, I=m.sections[1].I,
        alpha=m.materials[1].alpha, depth=m.sections[1].depth,
        rho=m.materials[1].density,
        section_id=1,
    ))
    # B (id 2) → override M2; C (id 3) → override M1.
    UpdateElementCmd(
        elem_id=2, section_id=1, kind="frame",
        material_override_id=2,
    ).do(m)
    UpdateElementCmd(
        elem_id=3, section_id=1, kind="frame",
        material_override_id=1,
    ).do(m)

    elem_a = next(e for e in m.elements if e.id == 1)
    elem_b = next(e for e in m.elements if e.id == 2)
    elem_c = next(e for e in m.elements if e.id == 3)
    assert elem_a.material_id_override is None
    assert elem_b.material_id_override == 2
    assert elem_c.material_id_override == 1

    # Edit M1: bump E by 10×.
    new_M1_E = m.materials[1].E * 10.0
    new_M1 = Material(id=1, name="M1", E=new_M1_E,
                      alpha=m.materials[1].alpha,
                      density=m.materials[1].density,
                      nu=m.materials[1].nu)
    AddOrUpdateMaterialCmd(material=new_M1).do(m)

    assert elem_a.E == pytest.approx(new_M1_E), (
        "A (no override, section default M1) must follow M1 edits"
    )
    assert elem_c.E == pytest.approx(new_M1_E), (
        "C (explicit override = M1) must also follow M1 edits"
    )
    assert elem_b.E == pytest.approx(m.materials[2].E), (
        "B (override = M2) must NOT follow M1 edits"
    )

    # Now edit M2: bump E by 10×.
    new_M2_E = m.materials[2].E * 10.0
    new_M2 = Material(id=2, name="M2", E=new_M2_E,
                      alpha=m.materials[2].alpha,
                      density=m.materials[2].density,
                      nu=m.materials[2].nu)
    AddOrUpdateMaterialCmd(material=new_M2).do(m)

    assert elem_b.E == pytest.approx(new_M2_E), (
        "B (override = M2) must follow M2 edits"
    )
    assert elem_a.E == pytest.approx(new_M1_E), (
        "A must NOT follow M2 edits"
    )
    assert elem_c.E == pytest.approx(new_M1_E), (
        "C must NOT follow M2 edits"
    )


# ── 5. Section edit propagates geometry but not override material ──


def test_section_edit_propagates_geometry_but_not_overrides():
    """Two elements on section S (default M1), one with override = M2.

    Edit S: change A by 2× AND switch S's default material to M2.
    Expect: both elements' A doubles (geometry follows section), but the
    overridden element's E stays at M2's E (override didn't change)
    while the non-overridden element's E switches to M2's E (it now
    follows S's new default).
    """
    m = _two_material_model()
    # Element 1 → override = M2. Element 2 stays section-default.
    UpdateElementCmd(
        elem_id=1, section_id=1, kind="frame",
        material_override_id=2,
    ).do(m)
    elem_overridden = next(e for e in m.elements if e.id == 1)
    elem_default = next(e for e in m.elements if e.id == 2)

    old_A = m.sections[1].A
    new_A = old_A * 2.0
    # New S: doubled A, switched default to M2.
    new_S = Section(
        id=1, name="S", material_id=2,
        A=new_A, I=m.sections[1].I, depth=m.sections[1].depth,
        width=m.sections[1].width,
    )
    AddOrUpdateSectionCmd(section=new_S).do(m)

    # Geometry: both elements' A doubles.
    assert elem_overridden.A == pytest.approx(new_A)
    assert elem_default.A == pytest.approx(new_A)

    # Material: overridden element stays on its override (M2). The
    # non-overridden element now follows S's new default (also M2).
    M2 = m.materials[2]
    assert elem_overridden.E == pytest.approx(M2.E)
    assert elem_default.E == pytest.approx(M2.E)
    assert elem_overridden.material_id_override == 2
    assert elem_default.material_id_override is None


# ── PR #16 review-feedback regression tests ───────────────────


def test_delete_material_blocked_by_element_override():
    """DeleteMaterialCmd must refuse to delete a material that is
    referenced *only* through ``elem.material_id_override`` (no section
    points at it). Pre-fix, the check was section-only and would have
    silently deleted M2, leaving elem 1's override dangling and the
    next file write/load broken."""
    m = _two_material_model()
    # Override elem 1 onto M2. No section references M2 — the only
    # remaining tether to M2 is the element override.
    UpdateElementCmd(elem_id=1, section_id=1, kind="frame",
                     material_override_id=2).do(m)
    assert m.elements[0].material_id_override == 2
    assert all(s.material_id != 2 for s in m.sections.values())

    with pytest.raises(ValueError) as excinfo:
        DeleteMaterialCmd(material_id=2).do(m)
    msg = str(excinfo.value)
    assert "element override" in msg.lower()
    # M2 must still be present after the failed delete.
    assert 2 in m.materials

    # Clearing the override should allow the delete to succeed.
    UpdateElementCmd(elem_id=1, section_id=1, kind="frame",
                     material_override_id=None).do(m)
    DeleteMaterialCmd(material_id=2).do(m)
    assert 2 not in m.materials


def test_writer_rejects_dangling_material_override():
    """write_input_file must raise a clear error when an element carries
    a ``material_id_override`` that is no longer in ``model.materials``,
    rather than emitting a file that cannot be reloaded."""
    m = _two_material_model()
    UpdateElementCmd(elem_id=1, section_id=1, kind="frame",
                     material_override_id=2).do(m)
    # Simulate a desync (e.g. caused by a future bug that bypasses
    # DeleteMaterialCmd) — drop M2 directly from the dict.
    del m.materials[2]
    assert m.elements[0].material_id_override == 2

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.txt")
        with pytest.raises(ValueError) as excinfo:
            write_input_file(m, path)
        msg = str(excinfo.value)
        assert "material override" in msg.lower()
        assert "2" in msg
        # Nothing should remain on disk for a half-written file —
        # write_input_file writes via tmp+rename, so the destination
        # must not exist after the failure.
        assert not os.path.exists(path)


def test_parser_rejects_unknown_positional_token_after_release():
    """The ELEMENTS parser silently skipped any positional token past
    the kind/release slots, so a typo like ``material_override_id 2``
    (space instead of ``=``) would have produced an element with no
    override and no warning. Now it must raise ``ValueError``."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bad.txt")
        with open(path, "w") as f:
            f.write(
                "TITLE bad-input\n"
                "NODES 2\n"
                "1  0.0  0.0\n"
                "2  5.0  0.0\n"
                "MATERIALS 1\n"
                "1  2.0e8  1.0e-5  7850.0  Steel\n"
                "SECTIONS 1\n"
                "1  1  0.01  1.0e-4  0.1  S\n"
                # `2` (the would-be override material id) is positional
                # — the user forgot the `=`. Pre-fix this silently
                # parsed as a regular FRAME element with no override.
                "ELEMENTS 1\n"
                "1  1  2  1  FRAME  material_override_id  2\n"
                "SUPPORTS 1\n"
                "1  1  1  1\n"
            )
        with pytest.raises(ValueError) as excinfo:
            read_input_file(path)
        msg = str(excinfo.value)
        assert "Element 1" in msg
        assert "positional" in msg.lower()
