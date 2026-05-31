"""PR #27 — `load_case` metadata field on all user-created loads.

This is foundation only: the solver still ignores `load_case` and
applies every load unchanged. Tests pin defaults, propagation through
`dataclasses.replace`, file round-trip, split/remap preservation, and
the closed-set key=value parser.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from structural_analysis.file_io import read_input_file
from structural_analysis.gui_common.commands import (
    AddMemberCmd,
    SplitElementCmd,
)
from structural_analysis.gui_common.file_writer import write_input_file
from structural_analysis.model import (
    FrameTemperatureLoad,
    Material,
    NodalLoad,
    Node,
    PointLoad,
    Section,
    StructuralModel,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)


# ── defaults ─────────────────────────────────────────────────────────


def test_udl_default_load_case_is_DEFAULT():
    assert UniformDistributedLoad(wy=-10.0).load_case == "DEFAULT"


def test_pointload_default_load_case_is_DEFAULT():
    assert PointLoad(py=-10.0, a=1.0).load_case == "DEFAULT"


def test_frame_thermal_default_load_case_is_DEFAULT():
    assert FrameTemperatureLoad(t_top=10.0, t_bottom=20.0).load_case == "DEFAULT"


def test_truss_thermal_default_load_case_is_DEFAULT():
    assert TrussTemperatureLoad(delta_T=15.0).load_case == "DEFAULT"


def test_nodal_load_default_load_case_is_DEFAULT():
    assert NodalLoad(node_id=1, fx=10.0).load_case == "DEFAULT"


# ── propagation through dataclasses.replace ──────────────────────────


def test_pointload_replace_preserves_load_case():
    """Split/remap uses ``dataclasses.replace`` to shift PointLoad.a
    onto child elements — load_case must survive that copy."""
    from dataclasses import replace
    p = PointLoad(py=-20.0, a=4.0, load_case="LIVE")
    p2 = replace(p, a=1.5)
    assert p2.load_case == "LIVE"
    assert p2.a == 1.5


# ── file round-trip: existing files (no case token) default to DEFAULT


def _basic_frame_model() -> StructuralModel:
    m = StructuralModel(title="case-test")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=6.0, y_j=0.0,
        kind="frame", section_id=1,
    ).do(m)
    return m


def _round_trip(model: StructuralModel) -> StructuralModel:
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        write_input_file(model, path)
        return read_input_file(path)
    finally:
        os.unlink(path)


def _read_written_text(model: StructuralModel) -> str:
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        write_input_file(model, path)
        with open(path, encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(path)


def test_writer_omits_case_token_when_default_keeps_legacy_format():
    """A model with only DEFAULT-case loads must NOT emit any 'case=' tokens
    — pre-v0.17 files stay byte-identical through a round-trip."""
    m = _basic_frame_model()
    m.nodal_loads.append(NodalLoad(node_id=2, fx=10.0, fy=-5.0, mz=0.0))
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-10.0))
    m.elements[0].member_loads.append(PointLoad(py=-20.0, a=3.0))
    m.elements[0].member_loads.append(
        FrameTemperatureLoad(t_top=5.0, t_bottom=5.0)
    )
    text = _read_written_text(m)
    assert "case=" not in text


def test_round_trip_preserves_nodal_load_case():
    m = _basic_frame_model()
    m.nodal_loads.append(
        NodalLoad(node_id=2, fy=-5.0, load_case="DEAD")
    )
    m2 = _round_trip(m)
    assert m2.nodal_loads[0].load_case == "DEAD"


def test_round_trip_preserves_udl_case():
    m = _basic_frame_model()
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-10.0, load_case="LIVE")
    )
    m2 = _round_trip(m)
    assert m2.elements[0].member_loads[0].load_case == "LIVE"


def test_round_trip_preserves_pointload_case():
    m = _basic_frame_model()
    m.elements[0].member_loads.append(
        PointLoad(py=-20.0, a=2.5, load_case="WIND")
    )
    m2 = _round_trip(m)
    assert m2.elements[0].member_loads[0].load_case == "WIND"


def test_round_trip_preserves_frame_thermal_case():
    m = _basic_frame_model()
    m.elements[0].member_loads.append(
        FrameTemperatureLoad(t_top=10.0, t_bottom=20.0, load_case="THERMAL")
    )
    m2 = _round_trip(m)
    assert m2.elements[0].member_loads[0].load_case == "THERMAL"


def test_round_trip_preserves_truss_thermal_case():
    m = StructuralModel(title="truss-case")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=4.0, y_j=0.0,
        kind="truss", section_id=1,
    ).do(m)
    m.elements[0].member_loads.append(
        TrussTemperatureLoad(delta_T=30.0, load_case="THERMAL")
    )
    m2 = _round_trip(m)
    assert m2.elements[0].member_loads[0].load_case == "THERMAL"


def test_round_trip_preserves_coord_system_and_case_together():
    """Both optional tokens (coord_system positional + case= key=value)
    must round-trip together on a single UDL row."""
    m = _basic_frame_model()
    m.elements[0].member_loads.append(
        UniformDistributedLoad(
            wy=-10.0, wx=4.0,
            coord_system="global", load_case="DEAD",
        )
    )
    m2 = _round_trip(m)
    ld = m2.elements[0].member_loads[0]
    assert ld.coord_system == "global"
    assert ld.load_case == "DEAD"


def test_old_file_without_case_token_defaults_to_DEFAULT():
    """A hand-written pre-v0.17 file (no case= tokens) must load with
    every load tagged 'DEFAULT'. This is the back-compat guarantee."""
    legacy = (
        "TITLE\nlegacy\n\n"
        "NODES 2\n1  0.0  0.0\n2  6.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  1.2e-5  7850.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
        "LOADS 1\n2  10.0  -5.0  0.0\n\n"
        "MEMBER_UDL 1\n1  0.0  -10.0\n\n"
        "MEMBER_POINT_LOADS 1\n1  3.0  0.0  -20.0\n\n"
        "FRAME_TEMPERATURE 1\n1  10.0  20.0\n\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(legacy)
        m = read_input_file(path)
    finally:
        os.unlink(path)
    assert m.nodal_loads[0].load_case == "DEFAULT"
    ml = m.elements[0].member_loads
    assert all(ld.load_case == "DEFAULT" for ld in ml)


def test_unknown_keyvalue_token_raises_clear_error():
    """A trailing token with an unknown key= must fail loudly so a typo
    can't silently degrade. (Strict-set parsing — easier to widen later
    than to tighten.)"""
    bad = (
        "TITLE\nbad\n\n"
        "NODES 2\n1  0.0  0.0\n2  6.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  1.2e-5  7850.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
        "MEMBER_UDL 1\n1  0.0  -10.0  zone=foo\n\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(bad)
        with pytest.raises(ValueError, match=r"unknown key"):
            read_input_file(path)
    finally:
        os.unlink(path)


def test_empty_case_value_raises():
    """Trailing 'case=' with no value is a typo, not 'DEFAULT'."""
    bad = (
        "TITLE\nbad\n\n"
        "NODES 2\n1  0.0  0.0\n2  6.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  1.2e-5  7850.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
        "MEMBER_UDL 1\n1  0.0  -10.0  case=\n\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(bad)
        with pytest.raises(ValueError, match=r"empty case="):
            read_input_file(path)
    finally:
        os.unlink(path)


def test_writer_rejects_load_case_with_whitespace():
    """Whitespace in a case name would corrupt the row on reload (the
    reader splits on whitespace). The writer must refuse to serialise."""
    from structural_analysis.gui_common.file_writer import _case_token
    with pytest.raises(ValueError, match=r"whitespace"):
        _case_token("DEAD LOAD")


def test_writer_rejects_load_case_with_hash():
    """``#`` starts a comment in the input-file format. A case name
    containing ``#`` would be silently truncated on reload, so the
    writer must refuse to serialise it (Gemini PR #27 finding)."""
    from structural_analysis.gui_common.file_writer import _case_token
    with pytest.raises(ValueError, match=r"#"):
        _case_token("DEAD#1")


# ── parser: optional numeric fields omitted but metadata present ────


def test_reader_handles_udl_with_only_case_token_no_numeric_fields():
    """A hand-written ``MEMBER_UDL`` row that omits wx and wy entirely
    but supplies ``case=`` must NOT try to parse ``case=DEAD`` as a
    float. Both wx and wy default to 0 in this shape."""
    legacy = (
        "TITLE\nmeta-only\n\n"
        "NODES 2\n1  0.0  0.0\n2  6.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  1.2e-5  7850.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
        "MEMBER_UDL 1\n1  case=DEAD\n\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(legacy)
        m = read_input_file(path)
    finally:
        os.unlink(path)
    ld = m.elements[0].member_loads[0]
    assert ld.wx == 0.0
    assert ld.wy == 0.0
    assert ld.load_case == "DEAD"


def test_reader_handles_udl_with_coord_system_only_no_numeric_fields():
    """Same as above but with the positional coord_system token
    (``global``) instead of ``case=``."""
    legacy = (
        "TITLE\nmeta-only\n\n"
        "NODES 2\n1  0.0  0.0\n2  6.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  1.2e-5  7850.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
        "MEMBER_UDL 1\n1  global\n\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(legacy)
        m = read_input_file(path)
    finally:
        os.unlink(path)
    ld = m.elements[0].member_loads[0]
    assert ld.wx == 0.0
    assert ld.wy == 0.0
    assert ld.coord_system == "global"


def test_reader_handles_pointload_with_only_position_and_case_no_px_py():
    """``MEMBER_POINT_LOADS`` row that supplies only ``elem_id`` + ``a``
    + ``case=NAME`` (px / py both default to 0)."""
    legacy = (
        "TITLE\nmeta-only\n\n"
        "NODES 2\n1  0.0  0.0\n2  6.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  1.2e-5  7850.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
        "MEMBER_POINT_LOADS 1\n1  3.0  case=LIVE\n\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(legacy)
        m = read_input_file(path)
    finally:
        os.unlink(path)
    ld = m.elements[0].member_loads[0]
    assert ld.a == 3.0
    assert ld.px == 0.0
    assert ld.py == 0.0
    assert ld.load_case == "LIVE"


def _read_text_or_raise(text: str):
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return read_input_file(path)
    finally:
        os.unlink(path)


def test_reader_rejects_udl_with_surplus_numeric_token():
    """``MEMBER_UDL: 1  0  -10  5`` has an extra numeric beyond wx/wy.
    Before PR #27 the fixed coord-system slot would catch this; the
    capped metadata-start scan in v0.17 must still reject it rather
    than silently dropping the ``5`` (Codex P2 finding on PR #27)."""
    body = (
        "TITLE\ntoo-many-numerics\n\n"
        "NODES 2\n1  0.0  0.0\n2  6.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  1.2e-5  7850.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
        "MEMBER_UDL 1\n1  0.0  -10.0  5\n\n"
    )
    with pytest.raises(ValueError):
        _read_text_or_raise(body)


def test_reader_rejects_pointload_with_surplus_numeric_token():
    """``MEMBER_POINT_LOADS: 1  3  0  -20  99  case=LIVE`` has an
    extra numeric ``99`` beyond px/py. Must be rejected."""
    body = (
        "TITLE\ntoo-many-numerics\n\n"
        "NODES 2\n1  0.0  0.0\n2  6.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  1.2e-5  7850.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
        "MEMBER_POINT_LOADS 1\n1  3.0  0.0  -20.0  99  case=LIVE\n\n"
    )
    with pytest.raises(ValueError):
        _read_text_or_raise(body)


# ── split/remap preservation ────────────────────────────────────────


def _model_with_loaded_long_frame() -> StructuralModel:
    """Frame element of length 10 carrying one of each load kind with a
    non-default load_case. Used by the split tests below."""
    m = StructuralModel(title="split-case")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=10.0, y_j=0.0,
        kind="frame", section_id=1,
    ).do(m)
    e = m.elements[0]
    e.member_loads.append(
        UniformDistributedLoad(wy=-10.0, load_case="DEAD")
    )
    e.member_loads.append(
        PointLoad(py=-20.0, a=7.0, load_case="LIVE")
    )
    e.member_loads.append(
        FrameTemperatureLoad(
            t_top=10.0, t_bottom=20.0, load_case="THERMAL",
        )
    )
    return m


def test_split_propagates_load_case_to_both_children():
    """After a SplitElementCmd on a loaded element, every load on the
    two new child elements must carry its parent's load_case."""
    m = _model_with_loaded_long_frame()
    SplitElementCmd(element_id=m.elements[0].id, x=4.0, y=0.0).do(m)
    # After split there are two child elements (parent removed).
    assert len(m.elements) == 2
    cases_seen: set[str] = set()
    for child in m.elements:
        for ld in child.member_loads:
            cases_seen.add(ld.load_case)
    # All three parent cases should have made it onto at least one child.
    assert {"DEAD", "LIVE", "THERMAL"} <= cases_seen


def test_split_pointload_preserves_load_case_after_remap_to_child_b():
    """The PointLoad at a=7 lives on child B after splitting at x=4; the
    dataclasses.replace(a=...) copy must keep load_case='LIVE'."""
    m = _model_with_loaded_long_frame()
    SplitElementCmd(element_id=m.elements[0].id, x=4.0, y=0.0).do(m)
    point_rows = [
        ld for child in m.elements for ld in child.member_loads
        if isinstance(ld, PointLoad)
    ]
    assert len(point_rows) == 1
    assert point_rows[0].load_case == "LIVE"
    # And on the right child (a got remapped from 7.0 → 3.0 on child B).
    assert abs(point_rows[0].a - 3.0) < 1e-9


def test_split_udl_preserves_load_case_on_both_children():
    m = _model_with_loaded_long_frame()
    SplitElementCmd(element_id=m.elements[0].id, x=4.0, y=0.0).do(m)
    udls = [
        ld for child in m.elements for ld in child.member_loads
        if isinstance(ld, UniformDistributedLoad)
    ]
    # UDL is shared with both children — we expect two rows here.
    assert len(udls) == 2
    assert all(ld.load_case == "DEAD" for ld in udls)


def test_split_frame_thermal_preserves_load_case_on_both_children():
    m = _model_with_loaded_long_frame()
    SplitElementCmd(element_id=m.elements[0].id, x=4.0, y=0.0).do(m)
    thermals = [
        ld for child in m.elements for ld in child.member_loads
        if isinstance(ld, FrameTemperatureLoad)
    ]
    assert len(thermals) == 2
    assert all(ld.load_case == "THERMAL" for ld in thermals)
