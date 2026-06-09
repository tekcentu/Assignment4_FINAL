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

from structural_analysis.model import Node, PointLoad
from structural_analysis.element import FrameElement2D
from structural_analysis.file_io import read_input_file
from structural_analysis.gui_common.file_writer import write_input_file
from structural_analysis.gui_common.commands import (
    UpdateElementCmd, BatchUpdateElementsCmd, SplitElementCmd,
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
