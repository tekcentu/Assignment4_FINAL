"""Tests for the Material/Section data-class split and the file_io
backwards-compatibility shim."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.file_io import read_input_file
from structural_analysis.gui.file_writer import write_input_file
from structural_analysis.main import run_analysis
from structural_analysis.model import (
    Material,
    Node,
    NodalLoad,
    Section,
    StructuralModel,
    Support,
)


def test_material_has_only_E_and_alpha():
    m = Material(id=1, name="steel", E=200_000.0, alpha=1.2e-5)
    assert m.id == 1 and m.name == "steel"
    assert m.E == 200_000.0 and m.alpha == 1.2e-5
    # No A/I/depth on Material anymore.
    assert not hasattr(m, "A")
    assert not hasattr(m, "I")
    assert not hasattr(m, "depth")


def test_section_carries_geometry():
    s = Section(id=1, name="50x50", material_id=2,
                A=0.25, I=5.208e-3, depth=0.5)
    assert s.material_id == 2
    assert s.A == 0.25
    assert s.I == 5.208e-3
    assert s.depth == 0.5


def test_legacy_input_loads_via_shim():
    """An Assignment 2/3/4 input file (no SECTIONS block) still parses.

    The shim should auto-create a 1:1 Section per legacy MATERIALS row.
    """
    m = read_input_file("inputs/q2a_settlement.txt")
    assert len(m.materials) == 2
    assert len(m.sections) == 2
    # The 1:1 association: section id equals material id.
    for sid, sec in m.sections.items():
        assert sec.material_id == sid
        assert sid in m.materials
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"


def test_round_trip_writer_emits_sections_block():
    m = read_input_file("inputs/q2a_settlement.txt")
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        write_input_file(m, tmp)
        with open(tmp) as f:
            text = f.read()
        assert "SECTIONS" in text
        m2 = read_input_file(tmp)
        # Materials & sections should round-trip
        assert set(m.materials) == set(m2.materials)
        assert set(m.sections) == set(m2.sections)
        # And the analysis matches
        r1 = run_analysis(m, verbose=False)
        r2 = run_analysis(m2, verbose=False)
        for eid in r1.member_results:
            f1 = np.array(r1.member_results[eid]["f_local"], dtype=float)
            f2 = np.array(r2.member_results[eid]["f_local"], dtype=float)
            np.testing.assert_allclose(f1, f2, atol=1e-9)
    finally:
        os.unlink(tmp)


def test_member_udl_rejects_non_zero_wx():
    body = (
        "TITLE\nrejects wx\nNODES 2\n"
        "1 0 0\n2 4 0\n"
        "MATERIALS 1\n1  0.01  1e-4  2e8\n"
        "ELEMENTS 1\n1 1 2 1 FRAME\n"
        "SUPPORTS 1\n1 1 1 1\n"
        "LOADS 0\n"
        "MEMBER_UDL 1\n1  3.0  0.0\n"  # non-zero wx
    )
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            f.write(body)
        with pytest.raises(ValueError, match="wx"):
            read_input_file(tmp)
    finally:
        os.unlink(tmp)
