"""PR #29 — coefficient-based load combinations.

Covers the model (LoadCombination validation), the combine kernel
(MultiCaseAnalysisResult.combination), file round-trip of the
LOAD_COMBINATIONS block, and the case-rename / case-delete cascade
interactions with combination definitions.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from structural_analysis.element import FrameElement2D
from structural_analysis.file_io import read_input_file
from structural_analysis.gui_common.file_writer import write_input_file
from structural_analysis.main import run_multi_case_analysis
from structural_analysis.model import (
    LoadCase,
    LoadCombination,
    Material,
    NodalLoad,
    Node,
    Section,
    StructuralModel,
    Support,
    UniformDistributedLoad,
)
from structural_analysis.multi_case_result import SUM_ALL_KEY


# ── fixtures ────────────────────────────────────────────────────────


def _two_case_cantilever() -> StructuralModel:
    m = StructuralModel(title="comb-cantilever")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, I=8e-5, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0, load_case="DEAD"))
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-5.0, load_case="LIVE")
    )
    m.load_cases["DEAD"] = LoadCase(name="DEAD")
    m.load_cases["LIVE"] = LoadCase(name="LIVE")
    # Disable DEFAULT (empty) so SUM_ALL == DEAD + LIVE exactly.
    m.load_cases["DEFAULT"].enabled = False
    return m


# ── LoadCombination validation ──────────────────────────────────────


def test_combination_requires_at_least_one_term():
    with pytest.raises(ValueError, match=r"at least one term"):
        LoadCombination(name="EMPTY", terms={})


def test_combination_rejects_sum_all_name():
    with pytest.raises(ValueError, match=r"SUM_ALL"):
        LoadCombination(name="SUM_ALL", terms={"DEAD": 1.0})


def test_combination_rejects_whitespace_name():
    with pytest.raises(ValueError, match=r"whitespace|single token"):
        LoadCombination(name="COMB X", terms={"DEAD": 1.0})


def test_combination_rejects_zero_coefficient():
    with pytest.raises(ValueError, match=r"zero coefficient"):
        LoadCombination(name="C1", terms={"DEAD": 0.0})


def test_combination_rejects_non_finite_coefficient():
    with pytest.raises(ValueError, match=r"finite"):
        LoadCombination(name="C1", terms={"DEAD": math.inf})


def test_combination_allows_negative_coefficient():
    c = LoadCombination(name="C1", terms={"DEAD": 1.0, "WIND_X": -0.7})
    assert c.terms["WIND_X"] == -0.7


# ── combine kernel: SUM_ALL equivalence + scaling + negative ────────


def test_unit_combination_equals_sum_all():
    """1.0 DEAD + 1.0 LIVE must equal SUM_ALL for a two-case model."""
    m = _two_case_cantilever()
    multi = run_multi_case_analysis(m, verbose=False)
    sa = multi.sum_all()
    comb = multi.combination({"DEAD": 1.0, "LIVE": 1.0})
    assert sa is not None and comb is not None
    np.testing.assert_allclose(
        np.asarray(comb.D), np.asarray(sa.D), atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(comb.member_results[1]["f_local"]),
        np.asarray(sa.member_results[1]["f_local"]),
        atol=1e-9,
    )


def test_double_coefficient_doubles_response():
    """2.0 DEAD doubles DEAD displacements, reactions, and member
    forces relative to the 1.0 DEAD case."""
    m = _two_case_cantilever()
    multi = run_multi_case_analysis(m, verbose=False)
    dead = multi.cases["DEAD"]
    comb = multi.combination({"DEAD": 2.0})
    assert comb is not None
    np.testing.assert_allclose(
        np.asarray(comb.D), 2.0 * np.asarray(dead.D), atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(comb.member_results[1]["f_local"]),
        2.0 * np.asarray(dead.member_results[1]["f_local"]),
        atol=1e-9,
    )
    for nid, comp in dead.reactions.items():
        for k, v in comp.items():
            assert abs(comb.reactions[nid][k] - 2.0 * v) < 1e-9


def test_negative_coefficient_subtracts_response():
    """1.0 DEAD + (-1.0) DEAD must produce a zero response."""
    m = _two_case_cantilever()
    multi = run_multi_case_analysis(m, verbose=False)
    # Use a single case referenced twice isn't possible (dict key), so
    # verify -1.0*DEAD negates the DEAD response.
    comb = multi.combination({"DEAD": -1.0})
    dead = multi.cases["DEAD"]
    assert comb is not None
    np.testing.assert_allclose(
        np.asarray(comb.D), -1.0 * np.asarray(dead.D), atol=1e-9,
    )


def test_strength_combination_matches_manual_scaling():
    """1.2 DEAD + 1.6 LIVE == 1.2*DEAD.D + 1.6*LIVE.D."""
    m = _two_case_cantilever()
    multi = run_multi_case_analysis(m, verbose=False)
    comb = multi.combination({"DEAD": 1.2, "LIVE": 1.6})
    expected = (
        1.2 * np.asarray(multi.cases["DEAD"].D)
        + 1.6 * np.asarray(multi.cases["LIVE"].D)
    )
    assert comb is not None
    np.testing.assert_allclose(np.asarray(comb.D), expected, atol=1e-9)


# ── availability / missing cases ────────────────────────────────────


def test_combination_unavailable_when_referenced_case_unsolved():
    """A combination referencing a disabled (hence unsolved) case must
    be unavailable and return None."""
    m = _two_case_cantilever()
    m.load_cases["LIVE"].enabled = False  # LIVE won't be solved
    multi = run_multi_case_analysis(m, verbose=False)
    terms = {"DEAD": 1.0, "LIVE": 1.0}
    assert multi.combination_available(terms) is False
    assert multi.combination(terms) is None
    assert multi.missing_cases_for(terms) == ["LIVE"]


# ── file round-trip ─────────────────────────────────────────────────


def _round_trip(model: StructuralModel) -> StructuralModel:
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        write_input_file(model, path)
        return read_input_file(path)
    finally:
        os.unlink(path)


def test_round_trip_preserves_combinations():
    m = _two_case_cantilever()
    m.load_combinations["COMB_STRENGTH"] = LoadCombination(
        name="COMB_STRENGTH", terms={"DEAD": 1.2, "LIVE": 1.6},
    )
    m.load_combinations["COMB_WIND"] = LoadCombination(
        name="COMB_WIND", terms={"DEAD": 1.0, "LIVE": -0.7},
    )
    m2 = _round_trip(m)
    assert "COMB_STRENGTH" in m2.load_combinations
    assert m2.load_combinations["COMB_STRENGTH"].terms == {
        "DEAD": 1.2, "LIVE": 1.6,
    }
    # Negative coefficient survives.
    assert m2.load_combinations["COMB_WIND"].terms["LIVE"] == -0.7


def test_old_file_without_combinations_still_loads():
    legacy = (
        "TITLE\nlegacy\n\n"
        "NODES 2\n1  0.0  0.0\n2  6.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  1.2e-5  7850.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(legacy)
        m = read_input_file(path)
    finally:
        os.unlink(path)
    assert m.load_combinations == {}


def test_writer_omits_combinations_block_when_none():
    m = _two_case_cantilever()
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        write_input_file(m, path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
    finally:
        os.unlink(path)
    assert "LOAD_COMBINATIONS" not in text


def test_sum_all_is_not_written_as_a_combination():
    """Even if SUM_ALL is forced into the dict, the writer must drop
    it (it's a derived view, never a saved combination)."""
    m = _two_case_cantilever()
    # Bypass __post_init__'s SUM_ALL guard by inserting a normal combo
    # then mutating the dict key. We can't construct LoadCombination
    # named SUM_ALL, so just assert the writer filters the key.
    m.load_combinations["COMB1"] = LoadCombination(
        name="COMB1", terms={"DEAD": 1.0},
    )
    # Simulate a stray SUM_ALL key with a valid object renamed in-dict.
    stray = LoadCombination(name="COMB1", terms={"DEAD": 1.0})
    m.load_combinations["SUM_ALL"] = stray
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        write_input_file(m, path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
    finally:
        os.unlink(path)
    assert "SUM_ALL" not in text


def test_reader_rejects_combination_name_colliding_with_case():
    body = (
        "TITLE\ncollide\n\n"
        "NODES 2\n1  0.0  0.0\n2  6.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  1.2e-5  7850.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
        "LOAD_CASES 1\nDEAD\n\n"
        "LOAD_COMBINATIONS 1\nDEAD  1.0*DEAD\n\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        with pytest.raises(ValueError, match=r"collides with a load-case"):
            read_input_file(path)
    finally:
        os.unlink(path)


def test_reader_combinations_block_skips_indented_comments():
    body = (
        "TITLE\ncomment\n\n"
        "NODES 2\n1  0.0  0.0\n2  6.0  0.0\n\n"
        "MATERIALS 1\n1  2.1e8  1.2e-5  7850.0\n\n"
        "SECTIONS 1\n1  1  0.01  1e-4  0.3\n\n"
        "ELEMENTS 1\n1  1  2  1  FRAME\n\n"
        "LOAD_CASES 2\nDEAD\nLIVE\n\n"
        "LOAD_COMBINATIONS 1\n"
        "    # indented comment\n"
        "COMB1  1.2*DEAD  1.6*LIVE\n\n"
    )
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        m = read_input_file(path)
    finally:
        os.unlink(path)
    assert m.load_combinations["COMB1"].terms == {"DEAD": 1.2, "LIVE": 1.6}
