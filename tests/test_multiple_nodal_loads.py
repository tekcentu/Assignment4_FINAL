"""PR #30 — multiple nodal loads per node.

Covers the storage / file-IO / solver-case-filter contract:

  - A single node can carry several independent :class:`NodalLoad`
    rows (one per load case, or multiple within a single case — the
    assembler sums via ``+=``).
  - The DEAD case solve sees only DEAD rows; LIVE sees only LIVE; etc.
  - Two DEAD rows on the same node sum into a single equivalent load
    for the DEAD solve.
  - SUM_ALL == DEAD + LIVE (no double-counting).
  - A 1.0*DEAD + 1.0*LIVE combination matches the manual superposition.
  - File round-trip preserves repeated ``LOADS`` rows for the same node.
  - Legacy single-line LOADS files still load unchanged.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np

from structural_analysis.element import FrameElement2D
from structural_analysis.file_io import read_input_file
from structural_analysis.gui_common.file_writer import write_input_file
from structural_analysis.main import run_analysis, run_multi_case_analysis
from structural_analysis.model import (
    LoadCase,
    Material,
    NodalLoad,
    Node,
    Section,
    StructuralModel,
    Support,
)


def _cantilever_with_two_cases() -> StructuralModel:
    """A 4 m fixed-free cantilever with DEAD + LIVE load cases enabled.

    No loads attached — caller adds them per test.
    """
    m = StructuralModel(title="multi-nodal")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, I=8e-5, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.load_cases["DEAD"] = LoadCase(name="DEAD")
    m.load_cases["LIVE"] = LoadCase(name="LIVE")
    # Disable DEFAULT so SUM_ALL == DEAD + LIVE exactly.
    m.load_cases["DEFAULT"].enabled = False
    return m


# ── storage ────────────────────────────────────────────────────────────


def test_node_can_carry_dead_and_live_nodal_loads_simultaneously():
    m = _cantilever_with_two_cases()
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEAD"))
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-20.0, load_case="LIVE"))
    cases = sorted(ld.load_case for ld in m.nodal_loads if ld.node_id == 2)
    assert cases == ["DEAD", "LIVE"]


def test_node_can_carry_two_dead_rows():
    m = _cantilever_with_two_cases()
    m.nodal_loads.append(NodalLoad(node_id=2, fx=3.0, load_case="DEAD"))
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-7.0, load_case="DEAD"))
    dead = [ld for ld in m.nodal_loads if ld.load_case == "DEAD"]
    assert len(dead) == 2


# ── solver / case filter ──────────────────────────────────────────────


def test_case_filter_dead_only_sees_dead_rows():
    """When ``case='DEAD'`` is passed to ``run_analysis``, the LIVE
    nodal-load row contributes nothing — DEAD response equals the
    response of a model with only the DEAD load attached.
    """
    m = _cantilever_with_two_cases()
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEAD"))
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-99.0, load_case="LIVE"))
    res_dead = run_analysis(m, verbose=False, case="DEAD")

    # Reference: a fresh model with ONLY the DEAD row.
    m_ref = _cantilever_with_two_cases()
    m_ref.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEAD"))
    res_ref = run_analysis(m_ref, verbose=False, case="DEAD")
    np.testing.assert_allclose(
        np.asarray(res_dead.D), np.asarray(res_ref.D), atol=1e-12,
    )


def test_two_dead_rows_on_same_node_sum_in_the_dead_solve():
    """Two DEAD rows (fy=-5 and fy=-5) at the same node must produce
    the same DEAD response as a single fy=-10 row.
    """
    m_split = _cantilever_with_two_cases()
    m_split.nodal_loads.append(NodalLoad(node_id=2, fy=-5.0, load_case="DEAD"))
    m_split.nodal_loads.append(NodalLoad(node_id=2, fy=-5.0, load_case="DEAD"))
    res_split = run_analysis(m_split, verbose=False, case="DEAD")

    m_one = _cantilever_with_two_cases()
    m_one.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEAD"))
    res_one = run_analysis(m_one, verbose=False, case="DEAD")

    np.testing.assert_allclose(
        np.asarray(res_split.D), np.asarray(res_one.D), atol=1e-12,
    )


def test_sum_all_equals_dead_plus_live_with_multiple_nodal_rows():
    m = _cantilever_with_two_cases()
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEAD"))
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-20.0, load_case="LIVE"))
    multi = run_multi_case_analysis(m, verbose=False)
    sa = multi.sum_all()
    expected = (
        np.asarray(multi.cases["DEAD"].D)
        + np.asarray(multi.cases["LIVE"].D)
    )
    assert sa is not None
    np.testing.assert_allclose(np.asarray(sa.D), expected, atol=1e-12)


def test_combination_matches_manual_superposition_with_multiple_nodal_rows():
    m = _cantilever_with_two_cases()
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEAD"))
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-20.0, load_case="LIVE"))
    multi = run_multi_case_analysis(m, verbose=False)
    comb = multi.combination({"DEAD": 1.2, "LIVE": 1.6})
    assert comb is not None
    expected = (
        1.2 * np.asarray(multi.cases["DEAD"].D)
        + 1.6 * np.asarray(multi.cases["LIVE"].D)
    )
    np.testing.assert_allclose(np.asarray(comb.D), expected, atol=1e-12)


# ── file I/O round-trip ───────────────────────────────────────────────


def _round_trip(model: StructuralModel) -> StructuralModel:
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        write_input_file(model, path)
        return read_input_file(path)
    finally:
        os.unlink(path)


def test_round_trip_preserves_multiple_nodal_loads_on_same_node():
    m = _cantilever_with_two_cases()
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEAD"))
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-20.0, load_case="LIVE"))
    m.nodal_loads.append(NodalLoad(node_id=2, fx=5.0, load_case="DEAD"))
    m2 = _round_trip(m)
    rows = [ld for ld in m2.nodal_loads if ld.node_id == 2]
    assert len(rows) == 3
    by_case = {(ld.load_case, ld.fx, ld.fy) for ld in rows}
    assert by_case == {
        ("DEAD", 0.0, -10.0),
        ("LIVE", 0.0, -20.0),
        ("DEAD", 5.0, 0.0),
    }


def test_old_single_load_file_still_loads():
    """A pre-v0.20 file with a single LOADS row per node must still
    parse into a 1-element :attr:`StructuralModel.nodal_loads` list."""
    legacy = (
        "TITLE\nlegacy-single\n\n"
        "NODES 2\n1 0 0\n2 4 0\n\n"
        "MATERIALS 1\n1 2.1e8 0 0 Steel\n\n"
        "SECTIONS 1\n1 1 0.02 8e-5 0.3 S\n\n"
        "ELEMENTS 1\n1 1 2 1 FRAME\n\n"
        "SUPPORTS 1\n1 1 1 1\n\n"
        "LOADS 1\n2 0 -10 0\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(legacy)
        m = read_input_file(path)
    finally:
        os.unlink(path)
    assert len(m.nodal_loads) == 1
    ld = m.nodal_loads[0]
    assert ld.node_id == 2 and ld.fy == -10.0


def test_round_trip_of_legacy_multi_line_loads_file():
    """Hand-written file with multiple LOADS rows on the same node
    must read into multiple :class:`NodalLoad` entries."""
    text = (
        "TITLE\nmulti-row\n\n"
        "NODES 2\n1 0 0\n2 4 0\n\n"
        "MATERIALS 1\n1 2.1e8 0 0 Steel\n\n"
        "SECTIONS 1\n1 1 0.02 8e-5 0.3 S\n\n"
        "ELEMENTS 1\n1 1 2 1 FRAME\n\n"
        "SUPPORTS 1\n1 1 1 1\n\n"
        "LOADS 2\n2 0 -10 0  case=DEAD\n2 0 -20 0  case=LIVE\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        m = read_input_file(path)
    finally:
        os.unlink(path)
    rows = [ld for ld in m.nodal_loads if ld.node_id == 2]
    assert {(ld.load_case, ld.fy) for ld in rows} == {
        ("DEAD", -10.0),
        ("LIVE", -20.0),
    }
