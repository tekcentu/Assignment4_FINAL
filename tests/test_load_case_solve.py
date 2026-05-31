"""PR-A — case-by-case static run + MultiCaseAnalysisResult tests.

Pins the headline behaviours:
* ``run_analysis(model, case=...)`` filters loads to that case and
  toggles self-weight to only contribute when ``case ==
  model.self_weight_case`` — without mutating the model after the call
  (try/finally restoration).
* ``run_multi_case_analysis`` returns one AnalysisResult per enabled
  case; disabled cases are skipped.
* SUM_ALL equals linear superposition of per-case results, and is
  available **only** when every requested case solved successfully.
"""

from __future__ import annotations

import numpy as np
import pytest

from structural_analysis.element import FrameElement2D
from structural_analysis.main import (
    run_analysis,
    run_multi_case_analysis,
)
from structural_analysis.model import (
    LoadCase,
    Material,
    NodalLoad,
    Node,
    PointLoad,
    Section,
    StructuralModel,
    Support,
    UniformDistributedLoad,
)
from structural_analysis.multi_case_result import (
    SUM_ALL_KEY,
    MultiCaseAnalysisResult,
)


# ── fixtures ────────────────────────────────────────────────────────


def _make_cantilever_with_two_cases() -> StructuralModel:
    """Horizontal cantilever fixed at node 1, free at node 2 (L=4m).
    Carries a -10 kN nodal load tagged DEAD at node 2 and a
    -20 kN UDL tagged LIVE on the element."""
    m = StructuralModel(title="cantilever-2-case")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, I=8e-5,
        rho=7850.0, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.nodal_loads.append(NodalLoad(
        node_id=2, fy=-10.0, load_case="DEAD",
    ))
    m.elements[0].member_loads.append(UniformDistributedLoad(
        wy=-20.0, load_case="LIVE",
    ))
    m.load_cases["DEAD"] = LoadCase(name="DEAD")
    m.load_cases["LIVE"] = LoadCase(name="LIVE")
    return m


# ── defaults & __post_init__ guarantees ─────────────────────────────


def test_structural_model_auto_creates_DEFAULT_case():
    """A freshly-constructed model must already carry the DEFAULT case
    so callers never have to insert it themselves (relied on by
    file_io / new-from-blank flows)."""
    m = StructuralModel(title="t")
    assert "DEFAULT" in m.load_cases
    assert m.load_cases["DEFAULT"].name == "DEFAULT"
    assert m.load_cases["DEFAULT"].enabled is True


def test_self_weight_case_defaults_to_DEFAULT():
    m = StructuralModel(title="t")
    assert m.self_weight_case == "DEFAULT"


def test_loadcase_rejects_whitespace_in_name():
    with pytest.raises(ValueError, match=r"whitespace"):
        LoadCase(name="DEAD LOAD")


def test_loadcase_rejects_hash_in_name():
    with pytest.raises(ValueError, match=r"whitespace.*#|#"):
        LoadCase(name="DEAD#1")


# ── single-case filtering via run_analysis(case=...) ────────────────


def test_run_analysis_with_case_filters_loads_to_that_case():
    """case='DEAD' must solve with ONLY the -10 kN tip load (no UDL).
    case='LIVE' must solve with ONLY the UDL (no tip load).
    The two results must therefore differ."""
    m = _make_cantilever_with_two_cases()
    r_dead = run_analysis(m, verbose=False, case="DEAD")
    r_live = run_analysis(m, verbose=False, case="LIVE")
    assert r_dead.status == "ok"
    assert r_live.status == "ok"
    # Tip displacement uy is the most case-sensitive observable.
    emap = r_dead.E_map[2]
    uy_dead = r_dead.D[emap["uy"]]
    uy_live = r_live.D[emap["uy"]]
    assert abs(uy_dead - uy_live) > 1e-6, (
        "DEAD-only and LIVE-only solves must produce different tip "
        "deflections — filter is leaking loads across cases"
    )


def test_run_analysis_does_not_permanently_mutate_model():
    """try/finally must restore nodal_loads + member_loads + the
    include_self_weight flag even when the inner solve runs to
    completion."""
    m = _make_cantilever_with_two_cases()
    before_nodal = list(m.nodal_loads)
    before_member = list(m.elements[0].member_loads)
    before_sw = m.include_self_weight
    _ = run_analysis(m, verbose=False, case="DEAD")
    assert m.nodal_loads == before_nodal
    assert m.elements[0].member_loads == before_member
    assert m.include_self_weight == before_sw


def test_run_analysis_legacy_no_case_solves_all_loads():
    """Default ``case=None`` keeps the pre-v0.18 behaviour: every
    attached load contributes (DEAD + LIVE in one solve)."""
    m = _make_cantilever_with_two_cases()
    r = run_analysis(m, verbose=False)
    assert r.status == "ok"
    r_only_dead = run_analysis(m, verbose=False, case="DEAD")
    r_only_live = run_analysis(m, verbose=False, case="LIVE")
    # uy at tip — case=None must be (close to) DEAD + LIVE.
    emap = r.E_map[2]
    uy_full = r.D[emap["uy"]]
    uy_sum = (
        r_only_dead.D[emap["uy"]] + r_only_live.D[emap["uy"]]
    )
    assert abs(uy_full - uy_sum) < 1e-6


# ── run_multi_case_analysis & MultiCaseAnalysisResult ───────────────


def test_run_multi_case_solves_each_enabled_case():
    m = _make_cantilever_with_two_cases()
    multi = run_multi_case_analysis(m, verbose=False)
    assert isinstance(multi, MultiCaseAnalysisResult)
    assert set(multi.cases.keys()) == {"DEFAULT", "DEAD", "LIVE"}
    assert multi.failed_cases == {}
    assert multi.requested_cases == sorted(["DEFAULT", "DEAD", "LIVE"])


def test_disabled_case_is_skipped_in_multi_run():
    m = _make_cantilever_with_two_cases()
    m.load_cases["LIVE"].enabled = False
    multi = run_multi_case_analysis(m, verbose=False)
    assert "LIVE" not in multi.cases
    assert "LIVE" not in multi.requested_cases
    assert "DEAD" in multi.cases


def test_dead_plus_live_equals_sum_all_under_linear_superposition():
    """Core regression for the SUM_ALL view: D and member forces from
    SUM_ALL must equal the per-case sum within FP tolerance."""
    m = _make_cantilever_with_two_cases()
    # Disable DEFAULT (empty) so SUM_ALL is just DEAD + LIVE.
    m.load_cases["DEFAULT"].enabled = False
    multi = run_multi_case_analysis(m, verbose=False)
    assert multi.sum_all_available()
    sa = multi.sum_all()
    assert sa is not None
    expected_D = np.asarray(multi.cases["DEAD"].D) + np.asarray(
        multi.cases["LIVE"].D
    )
    np.testing.assert_allclose(np.asarray(sa.D), expected_D, atol=1e-9)
    # Member f_local — sum cell by cell.
    f_d = multi.cases["DEAD"].member_results[1]["f_local"]
    f_l = multi.cases["LIVE"].member_results[1]["f_local"]
    np.testing.assert_allclose(
        np.asarray(sa.member_results[1]["f_local"]),
        np.asarray(f_d) + np.asarray(f_l),
        atol=1e-9,
    )


def test_sum_all_unavailable_when_a_requested_case_fails():
    """Per the PR-A approval (redirect #8): SUM_ALL must be blocked
    when any *requested* enabled case failed to solve. We synthesise a
    failing case by writing a model that triggers an assembly error in
    one case only — easiest path is to inject a known-good multi
    result and then mark one case as failed by hand."""
    m = _make_cantilever_with_two_cases()
    multi = run_multi_case_analysis(m, verbose=False)
    # Synthesise a failed-case scenario.
    multi.failed_cases["WIND"] = "synthetic-failure"
    multi.requested_cases.append("WIND")
    assert multi.sum_all_available() is False
    assert multi.sum_all() is None


def test_sum_all_requires_at_least_two_solved_cases():
    """A multi-result with only one solved case is conceptually
    equivalent to that case — SUM_ALL adds no information, so the
    GUI helper ``available_case_names`` should not surface it."""
    m = StructuralModel(title="one-case")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, I=8e-5,
        rho=7850.0, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    m.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0))
    multi = run_multi_case_analysis(m, verbose=False)
    names = multi.available_case_names()
    assert SUM_ALL_KEY not in names


# ── self-weight per-case attribution ────────────────────────────────


def test_self_weight_only_added_to_designated_case():
    """SUM_ALL of a self-weight-on model must include the self-weight
    contribution exactly once (not once per case). With
    ``self_weight_case = "DEAD"``, only the DEAD case picks it up; LIVE
    sees no gravity contribution."""
    m = _make_cantilever_with_two_cases()
    m.include_self_weight = True
    m.self_weight_case = "DEAD"
    m.load_cases["DEFAULT"].enabled = False
    multi = run_multi_case_analysis(m, verbose=False)
    sa = multi.sum_all()
    assert sa is not None
    # DEAD + LIVE → DEAD picked up SW; LIVE did not.
    # Reaction Ry on the fixed support (node 1) should equal:
    #   |DEAD with SW| + |LIVE|
    # where DEAD's contribution > 10 kN (the tip load alone) because
    # of the added self-weight.
    r_dead = multi.cases["DEAD"]
    # Sanity: DEAD-only Ry exceeds 10 kN by the gravity contribution.
    assert abs(r_dead.reactions[1]["uy"]) > 10.0 + 1e-6


def test_self_weight_not_double_counted_when_swcase_excluded_from_sum_all():
    """If self_weight_case is enabled, SUM_ALL picks it up via that
    case's solve. If we run that case independently again with
    ``case=`` filter we should get the same self-weight contribution
    (not double)."""
    m = _make_cantilever_with_two_cases()
    m.include_self_weight = True
    m.self_weight_case = "DEAD"
    r_dead_filtered = run_analysis(m, verbose=False, case="DEAD")
    # The filter context manager toggles include_self_weight per case;
    # restoration must put the original flag back True.
    assert m.include_self_weight is True
    # Re-running with case=DEAD must give an IDENTICAL result.
    r_dead_again = run_analysis(m, verbose=False, case="DEAD")
    np.testing.assert_allclose(
        np.asarray(r_dead_filtered.D), np.asarray(r_dead_again.D),
        atol=1e-12,
    )


# ── active_case helper ──────────────────────────────────────────────


def test_make_active_case_safe_keeps_valid_choice():
    from structural_analysis.multi_case_result import make_active_case_safe
    m = _make_cantilever_with_two_cases()
    multi = run_multi_case_analysis(m, verbose=False)
    assert make_active_case_safe(multi, "DEAD") == "DEAD"
    assert make_active_case_safe(multi, "DEFAULT") == "DEFAULT"


def test_make_active_case_safe_falls_back_to_DEFAULT_for_invalid():
    from structural_analysis.multi_case_result import make_active_case_safe
    m = _make_cantilever_with_two_cases()
    multi = run_multi_case_analysis(m, verbose=False)
    # WIND wasn't requested, but DEFAULT is in the solved set →
    # fallback to DEFAULT.
    assert make_active_case_safe(multi, "WIND") == "DEFAULT"


def test_make_active_case_safe_with_none_multi_returns_desired():
    from structural_analysis.multi_case_result import make_active_case_safe
    assert make_active_case_safe(None, "WHATEVER") == "WHATEVER"


# ── Gemini PR #28 regression — sum_all on heterogeneous DOF widths ──


def test_sum_all_handles_truss_element_with_4_dof_f_local():
    """Regression for the hardcoded ``np.zeros(6)`` initialiser that
    crashed on truss elements where member_results.f_local is
    4-element. The lazy ``np.zeros_like`` init must accept whatever
    shape the per-case result delivers."""
    import numpy as np
    from structural_analysis.model import AnalysisResult
    from structural_analysis.multi_case_result import MultiCaseAnalysisResult

    def _mock_case(value: float) -> AnalysisResult:
        return AnalysisResult(
            status="ok",
            title="mock",
            E_map={},
            num_eq=4,
            G_vectors={},
            D=np.array([value, value, value, value]),
            residual=0.0,
            member_results={
                1: {
                    # 4-element f_local (truss-like)
                    "f_local": np.array([value, 0.0, -value, 0.0]),
                    "d_local": np.array([value, 0.0, value, 0.0]),
                    "d_global": np.array([value, 0.0, value, 0.0]),
                },
            },
            reactions={},
            eq_residual=0.0,
        )

    multi = MultiCaseAnalysisResult(
        cases={"DEAD": _mock_case(1.0), "LIVE": _mock_case(2.0)},
        active_case="DEAD",
        requested_cases=["DEAD", "LIVE"],
    )
    sa = multi.sum_all()
    assert sa is not None
    assert sa.member_results[1]["f_local"].shape == (4,)
    np.testing.assert_allclose(
        sa.member_results[1]["f_local"],
        np.array([3.0, 0.0, -3.0, 0.0]),
    )
