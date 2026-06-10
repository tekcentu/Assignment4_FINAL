"""Rigid end offsets — data model, persistence, commands, validation.

Offsets are OPTIONAL: both default to 0.0 and a zero-offset element
behaves exactly as before (regression pinned in
``test_rigid_offsets_solver.py``).
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.model import (
    Material, Node, PointLoad, Section, StructuralModel, Support,
)
from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.file_io import read_input_file
from structural_analysis.gui_common.file_writer import write_input_file
from structural_analysis.gui_common.commands import (
    AddElementCmd, AddMemberCmd, AssignAutoRigidOffsetsCmd,
    BatchUpdateElementsCmd, ClearRigidOffsetsCmd, SplitElementCmd,
    UpdateElementCmd,
)
from structural_analysis.gui_common.validation import validate_model


BASE_TXT = """TITLE
offsets fixture

NODES 2
1  0.0  0.0
2  6.0  0.0

MATERIALS 1
1  210000000.0  1.2e-05  7850.0  Steel

SECTIONS 1
1  1  0.00285  1.94e-05  0.2  IPE200

ELEMENTS 1
1  1  2  1  FRAME{elem_suffix}

SUPPORTS 2
1  1  1  0
2  0  1  0
"""


def _read_model(elem_suffix: str = ""):
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(BASE_TXT.format(elem_suffix=elem_suffix))
        return read_input_file(tmp)
    finally:
        os.unlink(tmp)


def _roundtrip(model):
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        write_input_file(model, tmp)
        with open(tmp, "r", encoding="utf-8") as f:
            text = f.read()
        return read_input_file(tmp), text
    finally:
        os.unlink(tmp)


# ── defaults ─────────────────────────────────────────────────────────────


def test_offsets_default_to_zero():
    e = FrameElement2D(1, 1, 2, E=1.0, A=1.0, I=1.0)
    assert e.offset_i == 0.0
    assert e.offset_j == 0.0
    assert e.has_offsets is False


def test_flexible_length_with_and_without_offsets():
    nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(1, 1, 2, E=1.0, A=1.0, I=1.0)
    assert e.flexible_length(nodes) == pytest.approx(6.0)
    e2 = FrameElement2D(1, 1, 2, E=1.0, A=1.0, I=1.0,
                        offset_i=0.5, offset_j=0.25)
    assert e2.flexible_length(nodes) == pytest.approx(5.25)
    assert e2.has_offsets is True


def test_flexible_length_rejects_consuming_offsets():
    nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 1.0, 0.0)}
    e = FrameElement2D(1, 1, 2, E=1.0, A=1.0, I=1.0,
                       offset_i=0.6, offset_j=0.6)
    with pytest.raises(ValueError, match="flexible span"):
        e.flexible_length(nodes)


def _command_model():
    m = StructuralModel(title="cmd offsets")
    m.materials = {1: Material(1, E=210000.0, alpha=1.2e-5)}
    m.sections = {1: Section(1, material_id=1, A=0.01, I=0.001)}
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    return m


def test_add_element_cmd_preserves_frame_offsets():
    m = _command_model()
    AddElementCmd(
        node_i=1, node_j=2, section_id=1, kind="frame",
        offset_i=0.4, offset_j=0.3,
    ).do(m)
    e = m.elements[0]
    assert isinstance(e, FrameElement2D)
    assert e.offset_i == pytest.approx(0.4)
    assert e.offset_j == pytest.approx(0.3)


def test_add_member_cmd_forwards_frame_offsets():
    m = _command_model()
    m.nodes = {}
    AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=6.0, y_j=0.0,
        section_id=1, kind="frame", offset_i=0.4, offset_j=0.3,
    ).do(m)
    e = m.elements[0]
    assert e.offset_i == pytest.approx(0.4)
    assert e.offset_j == pytest.approx(0.3)


def test_add_element_cmd_rejects_tiny_flexible_span():
    m = _command_model()
    with pytest.raises(ValueError, match="minimum flexible span"):
        AddElementCmd(
            node_i=1, node_j=2, section_id=1, kind="frame",
            offset_i=6.0 - 5e-13,
        ).do(m)


# ── file parsing ─────────────────────────────────────────────────────────


def test_file_parses_offset_kwargs():
    m = _read_model("  offset_i=0.4  offset_j=0.3")
    e = m.elements[0]
    assert e.offset_i == pytest.approx(0.4)
    assert e.offset_j == pytest.approx(0.3)


def test_file_without_offsets_parses_zero():
    m = _read_model("")
    e = m.elements[0]
    assert e.offset_i == 0.0 and e.offset_j == 0.0


def test_file_rejects_negative_offset():
    with pytest.raises(ValueError, match=">= 0"):
        _read_model("  offset_i=-0.1")


def test_file_rejects_offsets_consuming_member():
    with pytest.raises(ValueError, match="flexible span"):
        _read_model("  offset_i=3.0  offset_j=3.0")


def test_file_rejects_offsets_on_truss():
    txt = BASE_TXT.format(elem_suffix="").replace(
        "1  1  2  1  FRAME", "1  1  2  1  TRUSS  offset_i=0.4",
    )
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(txt)
        with pytest.raises(ValueError, match="FRAME"):
            read_input_file(tmp)
    finally:
        os.unlink(tmp)


# ── round trips ──────────────────────────────────────────────────────────


def test_txt_roundtrip_preserves_offsets():
    m = _read_model("  offset_i=0.4  offset_j=0.3")
    m2, text = _roundtrip(m)
    e = m2.elements[0]
    assert e.offset_i == pytest.approx(0.4)
    assert e.offset_j == pytest.approx(0.3)
    assert "offset_i=0.4" in text
    assert "offset_j=0.3" in text


def test_txt_roundtrip_zero_offsets_emits_no_tokens():
    """Zero-offset files must serialise without any offset kwargs —
    byte-compatibility with pre-0.31 files."""
    m = _read_model("")
    _, text = _roundtrip(m)
    assert "offset_i" not in text
    assert "offset_j" not in text


def test_spa_json_roundtrip_preserves_offsets():
    from structural_analysis.gui_qt.project_io import (
        Project, save_project_json, load_project_json,
    )
    m = _read_model("  offset_i=0.4  offset_j=0.3")
    fd, tmp = tempfile.mkstemp(suffix=".spa.json")
    os.close(fd)
    try:
        save_project_json(Project(model=m, title="offsets"), tmp)
        loaded = load_project_json(tmp)
    finally:
        os.unlink(tmp)
    e = loaded.model.elements[0]
    assert e.offset_i == pytest.approx(0.4)
    assert e.offset_j == pytest.approx(0.3)


# ── commands / undo-redo ────────────────────────────────────────────────


def test_update_element_cmd_sets_and_undoes_offsets():
    m = _read_model("")
    cmd = UpdateElementCmd(
        elem_id=1, section_id=1, kind="frame",
        offset_i=0.5, offset_j=0.2,
    )
    cmd.do(m)
    e = m.elements[0]
    assert e.offset_i == pytest.approx(0.5)
    assert e.offset_j == pytest.approx(0.2)
    cmd.undo(m)
    e = m.elements[0]
    assert e.offset_i == 0.0 and e.offset_j == 0.0
    # redo
    cmd.do(m)
    assert m.elements[0].offset_i == pytest.approx(0.5)


def test_update_element_cmd_rejects_negative_offsets():
    m = _read_model("")
    with pytest.raises(ValueError, match=">= 0"):
        UpdateElementCmd(
            elem_id=1, section_id=1, kind="frame", offset_i=-0.1,
        ).do(m)


def test_update_element_cmd_rejects_offsets_consuming_member():
    m = _read_model("")
    with pytest.raises(ValueError, match="less"):
        UpdateElementCmd(
            elem_id=1, section_id=1, kind="frame",
            offset_i=3.0, offset_j=3.0,
        ).do(m)


def test_update_element_cmd_rejects_offsets_on_truss():
    m = _read_model("")
    with pytest.raises(ValueError, match="frame"):
        UpdateElementCmd(
            elem_id=1, section_id=1, kind="truss", offset_i=0.3,
        ).do(m)


def test_batch_update_preserves_offsets():
    m = _read_model("  offset_i=0.4  offset_j=0.3")
    BatchUpdateElementsCmd(element_ids=[1], section_id=1).do(m)
    e = m.elements[0]
    assert e.offset_i == pytest.approx(0.4)
    assert e.offset_j == pytest.approx(0.3)


def test_split_blocked_for_offset_elements():
    m = _read_model("  offset_i=0.4")
    with pytest.raises(ValueError, match="rigid end offsets"):
        SplitElementCmd(element_id=1, x=3.0, y=0.0).do(m)


# ── pre-solve validation ────────────────────────────────────────────────


def test_validation_catches_offsets_invalidated_by_node_move():
    m = _read_model("  offset_i=2.0  offset_j=2.0")
    # Simulate a node move that shrinks the member to 3 m (< 4 m offsets).
    m.nodes[2] = Node(2, 3.0, 0.0)
    result = validate_model(m)
    assert result.has_errors
    assert any(i.code == "rigid_offsets_exceed_length"
               for i in result.issues)


def test_validation_catches_point_load_in_rigid_zone():
    m = _read_model("  offset_i=1.0")
    m.elements[0].member_loads.append(PointLoad(py=-10.0, a=0.5))
    result = validate_model(m)
    assert result.has_errors
    assert any(i.code == "point_load_in_rigid_zone" for i in result.issues)


def test_validation_clean_for_valid_offsets():
    m = _read_model("  offset_i=0.4  offset_j=0.3")
    m.elements[0].member_loads.append(PointLoad(py=-10.0, a=3.0))
    result = validate_model(m)
    assert not any(
        i.code in ("rigid_offsets_exceed_length",
                   "point_load_in_rigid_zone",
                   "negative_rigid_offset")
        for i in result.issues
    )


# ── ClearRigidOffsetsCmd / AssignAutoRigidOffsetsCmd ─────────────────────


def _two_column_one_beam_model(depth: float = 0.4):
    """L-frame fixture: two columns share a top node with a horizontal beam.

    Used to exercise auto-offsets — the beam's joint overlap at each end
    sits inside the column's body rectangle, giving offsets ≈ depth/2.
    """
    m = StructuralModel(title="auto offsets")
    m.materials = {1: Material(1, E=200_000_000.0, alpha=1.2e-5,
                                 density=7850.0, name="Steel")}
    m.sections = {1: Section(1, material_id=1, A=0.02, I=8e-5,
                              depth=depth)}
    m.nodes = {
        1: Node(1, 0.0, 0.0),     # column-1 base
        2: Node(2, 0.0, 3.0),     # column-1 top / beam-i
        3: Node(3, 6.0, 3.0),     # column-2 top / beam-j
        4: Node(4, 6.0, 0.0),     # column-2 base
    }
    m.elements = [
        FrameElement2D(1, 1, 2, E=200_000_000.0, A=0.02, I=8e-5,
                        depth=depth, section_id=1),  # left column
        FrameElement2D(2, 2, 3, E=200_000_000.0, A=0.02, I=8e-5,
                        depth=depth, section_id=1),  # beam
        FrameElement2D(3, 4, 3, E=200_000_000.0, A=0.02, I=8e-5,
                        depth=depth, section_id=1),  # right column
    ]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=True),
        4: Support(4, ux=True, uy=True, rz=True),
    }
    return m


def test_clear_offsets_cmd_do_undo():
    m = _read_model("  offset_i=0.5  offset_j=0.2")
    cmd = ClearRigidOffsetsCmd(element_ids=[1])
    cmd.do(m)
    e = m.elements[0]
    assert e.offset_i == 0.0 and e.offset_j == 0.0
    assert cmd.n_cleared == 1
    assert cmd.n_skipped_truss == 0
    assert cmd.n_skipped_already_zero == 0
    cmd.undo(m)
    e = m.elements[0]
    assert e.offset_i == pytest.approx(0.5)
    assert e.offset_j == pytest.approx(0.2)
    # redo restores the cleared state
    cmd.do(m)
    assert m.elements[0].offset_i == 0.0
    assert m.elements[0].offset_j == 0.0


def test_clear_offsets_cmd_skips_already_zero_and_truss():
    m = _read_model("")  # offsets default to 0
    # Replace the frame with a truss to exercise the truss-skip branch.
    m.elements[0] = TrussElement2D(
        1, 1, 2, E=200_000_000.0, A=0.02,
    )
    cmd = ClearRigidOffsetsCmd(element_ids=[1])
    cmd.do(m)
    assert cmd.n_cleared == 0
    assert cmd.n_skipped_truss == 1
    cmd.undo(m)  # no-op, must not raise


def test_clear_offsets_partial_undo_preserves_untouched():
    """A clear over {A=offsets, B=zero, C=truss} only snapshots A and
    undo touches only A — B and C are untouched throughout."""
    m = _two_column_one_beam_model()
    m.elements[0].offset_j = 0.3       # column 1 has an offset at j
    m.elements[1].offset_i = 0.0       # beam: zero
    m.elements[2] = TrussElement2D(
        3, 4, 3, E=200_000_000.0, A=0.02,
    )
    cmd = ClearRigidOffsetsCmd(element_ids=[1, 2, 3])
    cmd.do(m)
    assert cmd.n_cleared == 1
    assert cmd.n_skipped_already_zero == 1
    assert cmd.n_skipped_truss == 1
    assert m.elements[0].offset_j == 0.0
    cmd.undo(m)
    assert m.elements[0].offset_j == pytest.approx(0.3)


def test_assign_auto_offsets_cmd_basic():
    m = _two_column_one_beam_model(depth=0.4)
    cmd = AssignAutoRigidOffsetsCmd(element_ids=[2])
    cmd.do(m)
    beam = m.elements[1]
    # Beam runs horizontally; columns are 0.4 m wide so the inward
    # penetration along the beam axis is 0.2 m at each end.
    assert beam.offset_i == pytest.approx(0.2, rel=1e-6)
    assert beam.offset_j == pytest.approx(0.2, rel=1e-6)
    assert cmd.n_assigned == 1
    assert cmd.n_skipped_truss == 0
    assert cmd.n_skipped_too_short == 0


def test_assign_auto_offsets_cmd_undo_restores_prior_offsets():
    m = _two_column_one_beam_model(depth=0.4)
    m.elements[1].offset_i = 0.05  # pre-existing offset
    m.elements[1].offset_j = 0.07
    cmd = AssignAutoRigidOffsetsCmd(element_ids=[2])
    cmd.do(m)
    assert m.elements[1].offset_i != pytest.approx(0.05)
    cmd.undo(m)
    assert m.elements[1].offset_i == pytest.approx(0.05)
    assert m.elements[1].offset_j == pytest.approx(0.07)


def test_assign_auto_offsets_cmd_skips_truss():
    m = _two_column_one_beam_model(depth=0.4)
    # Replace the beam (id 2) with a truss.
    m.elements[1] = TrussElement2D(
        2, 2, 3, E=200_000_000.0, A=0.02,
    )
    cmd = AssignAutoRigidOffsetsCmd(element_ids=[2])
    cmd.do(m)
    assert cmd.n_assigned == 0
    assert cmd.n_skipped_truss == 1


def test_assign_auto_offsets_cmd_isolated_member_stays_zero():
    """A frame with no shared-node neighbor gets no overlap → reported
    as no_overlap, no mutation."""
    m = StructuralModel(title="lonely")
    m.materials = {1: Material(1, E=200_000_000.0, alpha=1.2e-5)}
    m.sections = {1: Section(1, material_id=1, A=0.02, I=8e-5, depth=0.3)}
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 5.0, 0.0)}
    m.elements = [FrameElement2D(
        1, 1, 2, E=200_000_000.0, A=0.02, I=8e-5, depth=0.3, section_id=1,
    )]
    cmd = AssignAutoRigidOffsetsCmd(element_ids=[1])
    cmd.do(m)
    assert cmd.n_assigned == 0
    assert cmd.n_no_overlap == 1
    assert m.elements[0].offset_i == 0.0
    assert m.elements[0].offset_j == 0.0


def test_assign_auto_offsets_cmd_skips_too_short_member():
    """A very short beam between two deep columns would have its whole
    span eaten by the overlap. Must be skipped, not silently clamped."""
    m = _two_column_one_beam_model(depth=2.0)
    # Move beam-j right next to beam-i (0.5 m apart) so the overlap
    # would consume essentially the whole flexible span.
    m.nodes[3] = Node(3, 0.5, 3.0)
    m.nodes[4] = Node(4, 0.5, 0.0)
    cmd = AssignAutoRigidOffsetsCmd(element_ids=[2])
    cmd.do(m)
    assert cmd.n_assigned == 0
    assert cmd.n_skipped_too_short == 1
    assert 2 in cmd.skipped_too_short_ids
    # Offsets stayed at 0 — no silent clamp.
    assert m.elements[1].offset_i == 0.0
    assert m.elements[1].offset_j == 0.0


def test_assign_auto_offsets_cmd_only_assigns_listed_elements():
    """Listing only the beam id must not touch the columns even though
    they also share the joint."""
    m = _two_column_one_beam_model(depth=0.4)
    cmd = AssignAutoRigidOffsetsCmd(element_ids=[2])
    cmd.do(m)
    assert m.elements[0].offset_i == 0.0 and m.elements[0].offset_j == 0.0
    assert m.elements[2].offset_i == 0.0 and m.elements[2].offset_j == 0.0
    assert cmd.n_assigned == 1
