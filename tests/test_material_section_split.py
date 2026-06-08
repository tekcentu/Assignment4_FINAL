"""Tests for the Material/Section data-class split and the file_io
backwards-compatibility shim."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.file_io import read_input_file
from structural_analysis.gui_common.file_writer import write_input_file
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


def test_identical_section_numerics_distinguished_by_section_id():
    """Two sections with identical (E, A, I, depth) — each element must
    rebind to *its own* section across a save/load round-trip.

    This is the regression test for the property-matching bug: under the
    old writer, both elements would silently bind to the lower-id section
    because the writer matched element.E/A/I against the section table.
    With section_id stored on elements, the assignment survives.
    """
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        # Build a model with two materials sharing E=200e6 and two sections
        # sharing A=0.01, I=1e-4 — i.e. *numerically indistinguishable* —
        # but pointing at different materials. Set α≠0 only on material 2
        # so the analysis result actually depends on which section each
        # element references.
        body = (
            "TITLE\nidentical-numerics regression\n"
            "NODES 3\n"
            "1 0 0\n2 4 0\n3 8 0\n"
            "MATERIALS 2\n"
            "1  200e6  0.0\n"
            "2  200e6  1.2e-5\n"
            "SECTIONS 2\n"
            "1  1  0.01  1e-4  0.0\n"
            "2  2  0.01  1e-4  0.0\n"   # same A, I, depth as section 1
            "ELEMENTS 2\n"
            "1 1 2 1 FRAME\n"
            "2 2 3 2 FRAME\n"
            "SUPPORTS 2\n"
            "1 1 1 1\n3 1 1 1\n"
            "LOADS 0\n"
            "FRAME_TEMPERATURE 1\n"
            "2  30.0  30.0\n"  # uniform heat on element 2 only
        )
        with open(tmp, "w") as f:
            f.write(body)
        m = read_input_file(tmp)

        # Both elements have numerically-identical flat properties.
        assert m.elements[0].E == m.elements[1].E
        assert m.elements[0].A == m.elements[1].A
        # But their section_id assignments differ.
        assert m.elements[0].section_id == 1
        assert m.elements[1].section_id == 2

        # Round-trip via writer.
        write_input_file(m, tmp)
        m2 = read_input_file(tmp)
        # The assignment must survive the writer.
        eids = {e.id: e.section_id for e in m2.elements}
        assert eids == {1: 1, 2: 2}, (
            f"section_id assignment didn't survive round-trip: {eids}"
        )

        # Sanity check: analyses match (the bug would change which element
        # owns the α≠0 material, which would shift the reaction force).
        r1 = run_analysis(m, verbose=False)
        r2 = run_analysis(m2, verbose=False)
        for eid in r1.member_results:
            f1 = np.array(r1.member_results[eid]["f_local"], dtype=float)
            f2 = np.array(r2.member_results[eid]["f_local"], dtype=float)
            np.testing.assert_allclose(f1, f2, atol=1e-9)
    finally:
        os.unlink(tmp)


def test_member_udl_accepts_non_zero_wx():
    """v0.15.0 — axial UDL components (wx) are now valid; they
    contribute to the local axial fixed-end force vector and the N
    diagram. The legacy reader used to reject wx != 0; that guard was
    removed when wx became a real physics field."""
    body = (
        "TITLE\nwx allowed\nNODES 2\n"
        "1 0 0\n2 4 0\n"
        "MATERIALS 1\n1  0.01  1e-4  2e8\n"
        "ELEMENTS 1\n1 1 2 1 FRAME\n"
        "SUPPORTS 1\n1 1 1 1\n"
        "LOADS 0\n"
        "MEMBER_UDL 1\n1  3.0  0.0\n"  # non-zero wx, zero wy
    )
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            f.write(body)
        m = read_input_file(tmp)
        elem = m.elements[0]
        assert len(elem.member_loads) == 1
        ld = elem.member_loads[0]
        assert ld.wx == 3.0
        assert ld.wy == 0.0
        assert ld.coord_system == "local"
    finally:
        os.unlink(tmp)
