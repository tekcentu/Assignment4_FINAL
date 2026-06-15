"""Pure-logic tests for the precast handling-stage engine (no Qt).

Covers reactions (always two supports), sling tensions, DAF scaling,
suction (lifting only), input validation, the display-only angle note,
the wL²/8 cross-check that proves the V/M diagrams reuse the shared
element_graphics helpers correctly, and that nothing here mutates the
main model.

Importing ``structural_analysis.gui_qt.precast`` pulls in matplotlib via
``element_graphics`` but no PyQt, so these run headless without a
QApplication.
"""

from __future__ import annotations

import copy

import pytest

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.gui_qt.precast import (
    DISPLAY_ONLY_NOTE,
    STAGE_LIFTING,
    STAGE_STOCK,
    STAGE_TRUCK,
    HandlingResult,
    MemberSpec,
    StageInput,
    auto_even_points,
    compute_handling,
    member_spec_from_element,
    resolve_single_frame,
)
from structural_analysis.model import (
    Material,
    Node,
    Section,
    StructuralModel,
    Support,
)


# ── model factory ─────────────────────────────────────────────────


def _model_with_one_frame(L: float = 8.0, A: float = 0.2,
                          rho: float = 2400.0) -> StructuralModel:
    m = StructuralModel(title="precast")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    m.materials[1] = Material(id=1, name="Concrete", E=3.0e7, density=rho)
    m.sections[1] = Section(id=1, name="PC", material_id=1, A=A, I=0.05,
                            depth=0.4)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, E=3.0e7, A=A, I=0.05, rho=rho,
        depth=0.4, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    return m


def _spec(L: float = 8.0, w: float = 10.0,
         depth: float = 0.0, area: float = 0.0,
         inertia: float = 0.0) -> MemberSpec:
    """A spec with an exact self-weight UDL so reactions are predictable."""
    return MemberSpec(
        elem_id=1, length=L, self_weight=w, section_name="PC",
        depth=depth, area=area, inertia=inertia,
    )


def _stress_spec(L: float = 8.0, w: float = 10.0,
                 depth: float = 0.4, inertia: float = 0.05) -> MemberSpec:
    """A spec with section data populated so the stress check runs."""
    return _spec(L=L, w=w, depth=depth, area=0.2, inertia=inertia)


# ── selection / snapshot ──────────────────────────────────────────


def test_resolve_single_frame_accepts_one_frame():
    m = _model_with_one_frame()
    elem = resolve_single_frame(m, [1])
    assert isinstance(elem, FrameElement2D)


def test_resolve_requires_exactly_one():
    m = _model_with_one_frame()
    with pytest.raises(ValueError, match="one frame element"):
        resolve_single_frame(m, [])
    with pytest.raises(ValueError, match="exactly one"):
        resolve_single_frame(m, [1, 2])


def test_resolve_rejects_truss_clearly():
    m = _model_with_one_frame()
    m.elements.append(TrussElement2D(id=2, node_i=1, node_j=2, E=2e8, A=0.01))
    with pytest.raises(ValueError, match="truss"):
        resolve_single_frame(m, [2])


def test_member_spec_self_weight_formula():
    m = _model_with_one_frame(L=6.0, A=0.25, rho=2400.0)
    spec = member_spec_from_element(m, m.elements[0])
    # ρ·A·g/1000 = 2400·0.25·9.81/1000
    assert spec.self_weight == pytest.approx(2400 * 0.25 * 9.81 / 1000.0)
    assert spec.length == pytest.approx(6.0)


def test_member_spec_rejects_truss():
    t = TrussElement2D(id=9, node_i=1, node_j=2, E=2e8, A=0.01)
    m = _model_with_one_frame()
    with pytest.raises(TypeError):
        member_spec_from_element(m, t)


# ── auto-even spacing ─────────────────────────────────────────────


def test_auto_even_points_uses_per_stage_defaults():
    assert auto_even_points(STAGE_LIFTING, 10.0) == (2.0, 8.0)
    assert auto_even_points(STAGE_STOCK, 10.0) == (2.0, 8.0)
    # Truck stage moves the supports closer to the ends.
    assert auto_even_points(STAGE_TRUCK, 10.0) == (1.0, 9.0)


# ── reactions (always two supports) ───────────────────────────────


def test_two_point_symmetric_reactions_equal():
    spec = _spec(L=8.0, w=10.0)
    stage = StageInput(stage=STAGE_LIFTING, points=(1.6, 6.4),
                       sling_angle_deg=60.0)
    res = compute_handling(spec, stage)
    r = [v for _x, v in res.reactions]
    assert r[0] == pytest.approx(r[1])
    assert r[0] == pytest.approx(40.0)  # half of 80 kN


def test_eccentric_reactions_satisfy_equilibrium():
    spec = _spec(L=10.0, w=12.0)         # W = 120 kN at centroid x=5
    a, b = 1.0, 7.0
    stage = StageInput(stage=STAGE_STOCK, points=(a, b))
    res = compute_handling(spec, stage)
    (xa, ra), (xb, rb) = res.reactions
    # ΣF = W
    assert ra + rb == pytest.approx(res.total_load)
    # ΣM about a = 0  →  rb·(b−a) == W·(centroid − a)
    assert rb * (xb - xa) == pytest.approx(res.total_load * (5.0 - a))


def test_stock_and_truck_support_reactions():
    spec = _spec(L=8.0, w=10.0)
    for stage_kind in (STAGE_STOCK, STAGE_TRUCK):
        stage = StageInput(stage=stage_kind, points=(1.0, 7.0))
        res = compute_handling(spec, stage)
        assert sum(r for _x, r in res.reactions) == pytest.approx(80.0)
        assert len(res.reactions) == 2
        # No slings outside lifting.
        assert res.sling_tensions == ()


# ── sling tensions ────────────────────────────────────────────────


def test_sling_tension_and_horizontal_component():
    import math
    spec = _spec(L=8.0, w=10.0)
    stage = StageInput(stage=STAGE_LIFTING, points=(1.6, 6.4),
                       sling_angle_deg=30.0)
    res = compute_handling(spec, stage)
    r = res.reactions[0][1]                      # 40 kN
    assert res.sling_tensions[0] == pytest.approx(r / math.sin(math.radians(30)))
    assert res.sling_horizontal[0] == pytest.approx(r / math.tan(math.radians(30)))


def test_sling_angle_90_has_zero_horizontal():
    spec = _spec(L=8.0, w=10.0)
    stage = StageInput(stage=STAGE_LIFTING, points=(1.6, 6.4),
                       sling_angle_deg=90.0)
    res = compute_handling(spec, stage)
    assert res.sling_horizontal[0] == pytest.approx(0.0)


def test_invalid_sling_angle_rejected():
    spec = _spec()
    for bad in (0.0, -10.0, 120.0):
        stage = StageInput(stage=STAGE_LIFTING, points=(1.6, 6.4),
                           sling_angle_deg=bad)
        with pytest.raises(ValueError, match="angle"):
            compute_handling(spec, stage)


# ── DAF / suction ─────────────────────────────────────────────────


def test_daf_scales_effects_linearly():
    spec = _spec(L=8.0, w=10.0)
    base = StageInput(stage=STAGE_STOCK, points=(1.0, 7.0), daf=1.0)
    amp = StageInput(stage=STAGE_STOCK, points=(1.0, 7.0), daf=1.5)
    rb = compute_handling(spec, base)
    ra = compute_handling(spec, amp)
    assert ra.total_load == pytest.approx(1.5 * rb.total_load)
    assert ra.v_max == pytest.approx(1.5 * rb.v_max)
    assert ra.m_pos_max == pytest.approx(1.5 * rb.m_pos_max)


def test_suction_adds_load_for_lifting_only():
    spec = _spec(L=8.0, w=10.0)
    lift_no = StageInput(stage=STAGE_LIFTING, points=(1.6, 6.4), suction=0.0)
    lift_yes = StageInput(stage=STAGE_LIFTING, points=(1.6, 6.4), suction=4.0)
    assert (compute_handling(spec, lift_yes).total_load
            > compute_handling(spec, lift_no).total_load)
    # Stock stage ignores suction (and warns).
    stock = StageInput(stage=STAGE_STOCK, points=(1.6, 6.4), suction=4.0)
    res = compute_handling(spec, stock)
    assert res.total_load == pytest.approx(80.0)
    assert any("lifting stage only" in w for w in res.warnings)


def test_extra_udl_adds_load():
    spec = _spec(L=8.0, w=10.0)
    stage = StageInput(stage=STAGE_STOCK, points=(1.0, 7.0), extra_udl=2.0)
    res = compute_handling(spec, stage)
    assert res.total_load == pytest.approx((10.0 + 2.0) * 8.0)


# ── validation ────────────────────────────────────────────────────


def test_invalid_point_positions_rejected():
    spec = _spec(L=8.0, w=10.0)
    with pytest.raises(ValueError, match="outside the member"):
        compute_handling(spec, StageInput(stage=STAGE_STOCK, points=(-1.0, 4.0)))
    with pytest.raises(ValueError, match="outside the member"):
        compute_handling(spec, StageInput(stage=STAGE_STOCK, points=(2.0, 9.0)))


def test_wrong_point_count_rejected():
    spec = _spec()
    with pytest.raises(ValueError, match="exactly 2"):
        compute_handling(
            spec,
            StageInput(stage=STAGE_STOCK, points=(4.0,)),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exactly 2"):
        compute_handling(
            spec,
            StageInput(stage=STAGE_LIFTING,
                       points=(1.0, 4.0, 7.0)),  # type: ignore[arg-type]
        )


def test_coincident_two_points_rejected():
    spec = _spec()
    with pytest.raises(ValueError, match="distinct"):
        compute_handling(spec, StageInput(stage=STAGE_STOCK, points=(4.0, 4.0)))


def test_bad_daf_and_negative_weights_rejected():
    spec = _spec()
    with pytest.raises(ValueError, match="DAF"):
        compute_handling(spec, StageInput(stage=STAGE_STOCK, points=(1.0, 7.0),
                                          daf=0.0))
    with pytest.raises(ValueError, match="negative"):
        compute_handling(spec, StageInput(stage=STAGE_STOCK, points=(1.0, 7.0),
                                          manual_weight=-1.0))


# ── display-only orientation note ─────────────────────────────────


def test_display_only_note_present():
    spec = _spec()
    res = compute_handling(spec, StageInput(stage=STAGE_STOCK, points=(1.0, 7.0)))
    assert res.display_note == DISPLAY_ONLY_NOTE
    assert "display-only" in res.display_note


# ── wL²/8 cross-check via shared V/M helpers ───────────────────────


def test_wl2_over_8_cross_check_uses_shared_helpers():
    """Simply-supported beam (supports at both ends) under a uniform load
    must give midspan M = wL²/8 — proving the reaction-injection model fed
    through element_graphics reproduces the closed form."""
    L, w = 8.0, 10.0
    spec = _spec(L=L, w=w)
    stage = StageInput(stage=STAGE_STOCK, points=(0.0, L))  # supports at ends
    res = compute_handling(spec, stage, n_samples=41)       # midspan sampled
    expected = w * L ** 2 / 8.0
    assert res.m_pos_max == pytest.approx(expected, rel=1e-6)
    # End reactions are each wL/2; peak shear is wL/2.
    assert res.v_max == pytest.approx(w * L / 2.0, rel=1e-6)


# ── no model mutation ─────────────────────────────────────────────


def test_compute_does_not_mutate_model_or_element():
    m = _model_with_one_frame(L=8.0, A=0.2, rho=2400.0)
    before = copy.deepcopy(m)
    spec = member_spec_from_element(m, m.elements[0])
    for stage in (
        StageInput(stage=STAGE_LIFTING, points=(1.6, 6.4)),
        StageInput(stage=STAGE_STOCK, points=(1.0, 7.0)),
        StageInput(stage=STAGE_TRUCK, points=(2.0, 6.0)),
    ):
        compute_handling(spec, stage)
    assert m.elements[0].member_loads == before.elements[0].member_loads
    assert len(m.elements) == len(before.elements)
    assert m.nodes.keys() == before.nodes.keys()
    assert isinstance(
        compute_handling(spec, StageInput(stage=STAGE_STOCK, points=(1.0, 7.0))),
        HandlingResult,
    )


# ── Flexural cracking check (V1) ──────────────────────────────────
#
# Sign convention used by the handling engine (cross-checked by
# ``test_wl2_over_8_cross_check_uses_shared_helpers``):
#
#   Positive M = sagging (concave up).  Sagging puts the **bottom**
#   fiber in tension and the top fiber in compression. Negative M is
#   hogging, so the **top** fiber is in tension. Tensile stress is
#   reported as a positive number; the *_tensile_mpa fields are
#   clamped to 0 when their fiber stays in compression throughout
#   the diagram.


def test_stress_formula_rectangular_known_section():
    """Simply-supported beam, supports at the ends: M_mid = wL²/8.

    With y_top = y_bottom = h/2:
      σ_bot,max = (wL²/8) · (h/2) / I  in kN/m²   → ×0.001 MPa
    For L = 8 m, w = 10 kN/m, h = 0.4 m, I = 0.05 m⁴:
      M = 80 kN·m
      σ = 80 · 0.2 / 0.05 = 320 kN/m² = 0.32 MPa
    """
    spec = _stress_spec(L=8.0, w=10.0, depth=0.4, inertia=0.05)
    stage = StageInput(stage=STAGE_STOCK, points=(0.0, 8.0))
    res = compute_handling(spec, stage, n_samples=41)
    sc = res.stress_check
    assert not sc.skipped
    assert sc.y_top == pytest.approx(0.2)
    assert sc.y_bottom == pytest.approx(0.2)
    assert sc.max_bottom_tensile_mpa == pytest.approx(0.32, rel=1e-3)
    # Top fiber stays in compression for an SS sagging beam.
    assert sc.max_top_tensile_mpa == pytest.approx(0.0)
    assert sc.controlling_fiber == "bottom"
    assert sc.controlling_x == pytest.approx(4.0, abs=0.2)


def test_positive_moment_makes_bottom_fiber_tensile():
    """Convention check: a fully-sagging diagram tensions the bottom."""
    spec = _stress_spec()
    stage = StageInput(stage=STAGE_STOCK, points=(0.0, 8.0))  # SS beam
    sc = compute_handling(spec, stage).stress_check
    # All bottom stresses are tensile (≥ 0) and top stresses non-tensile.
    bottoms = [s for _x, s in sc.bottom_stations]
    tops = [s for _x, s in sc.top_stations]
    assert min(bottoms) >= -1e-9
    assert max(tops) <= 1e-9
    assert sc.controlling_fiber == "bottom"


def test_negative_moment_makes_top_fiber_tensile():
    """Two-point lift with overhangs gives hogging at the supports,
    so the top fiber is the one that cracks."""
    spec = _stress_spec(L=8.0, w=10.0)
    stage = StageInput(stage=STAGE_LIFTING, points=(2.0, 6.0),
                       sling_angle_deg=90.0)
    sc = compute_handling(spec, stage).stress_check
    assert sc.max_top_tensile_mpa > 0.0   # cantilever hogging tensions the top
    # The controlling fiber should be top if it exceeds the bottom peak.
    if sc.max_top_tensile_mpa >= sc.max_bottom_tensile_mpa:
        assert sc.controlling_fiber == "top"


def test_cracking_check_ok_and_warning():
    """Same moment diagram, two different allowable stresses → flip status."""
    spec = _stress_spec(L=8.0, w=10.0, depth=0.4, inertia=0.05)  # σ_bot = 0.32 MPa
    points = (0.0, 8.0)
    sc_ok = compute_handling(
        spec, StageInput(stage=STAGE_STOCK, points=points,
                         allowable_tensile_mpa=2.0),
    ).stress_check
    assert sc_ok.cracking_status == "OK"
    assert sc_ok.cracking_ratio == pytest.approx(0.32 / 2.0, rel=1e-3)

    sc_warn = compute_handling(
        spec, StageInput(stage=STAGE_STOCK, points=points,
                         allowable_tensile_mpa=0.2),
    ).stress_check
    assert sc_warn.cracking_status == "CRACKING WARNING"
    assert sc_warn.cracking_ratio == pytest.approx(0.32 / 0.2, rel=1e-3)
    assert sc_warn.cracking_ratio > 1.0


def test_all_three_stages_report_stress_summary():
    spec = _stress_spec(L=8.0, w=10.0)
    for kind in (STAGE_LIFTING, STAGE_STOCK, STAGE_TRUCK):
        stage = StageInput(stage=kind, points=(1.6, 6.4))
        sc = compute_handling(spec, stage).stress_check
        assert not sc.skipped
        # Each stage reports a controlling station inside the member.
        assert 0.0 <= sc.controlling_x <= spec.length
        assert sc.cracking_status in ("OK", "CRACKING WARNING")


def test_missing_section_data_skips_check_with_warning():
    """No depth and no manual y → clear skip, no silent wrong stress."""
    spec = _spec(L=8.0, w=10.0, inertia=0.05)   # depth defaults to 0
    sc = compute_handling(
        spec, StageInput(stage=STAGE_STOCK, points=(1.0, 7.0)),
    ).stress_check
    assert sc.skipped
    assert sc.cracking_status == "skipped"
    assert "depth" in sc.skip_reason.lower()
    # All numeric peaks are 0; report stays explicit about being skipped.
    assert sc.max_top_tensile_mpa == 0.0
    assert sc.max_bottom_tensile_mpa == 0.0
    assert sc.cracking_ratio == 0.0


def test_missing_inertia_skips_check():
    spec = _spec(L=8.0, w=10.0, depth=0.4)  # I=0
    sc = compute_handling(
        spec, StageInput(stage=STAGE_STOCK, points=(1.0, 7.0)),
    ).stress_check
    assert sc.skipped
    assert "inertia" in sc.skip_reason.lower()


def test_manual_y_override_runs_check_without_depth():
    spec = _spec(L=8.0, w=10.0, inertia=0.05)   # no depth
    sc = compute_handling(
        spec,
        StageInput(stage=STAGE_STOCK, points=(0.0, 8.0),
                   manual_y_top=0.2, manual_y_bottom=0.2),
    ).stress_check
    assert not sc.skipped
    assert sc.max_bottom_tensile_mpa == pytest.approx(0.32, rel=1e-3)


def test_stress_check_disabled_is_skipped_explicitly():
    spec = _stress_spec()
    sc = compute_handling(
        spec,
        StageInput(stage=STAGE_STOCK, points=(0.0, 8.0),
                   stress_check_enabled=False),
    ).stress_check
    assert sc.enabled is False
    assert sc.skipped
    assert sc.cracking_status == "skipped"


def test_report_includes_stress_and_cracking_lines():
    from structural_analysis.gui_qt.precast import format_report
    spec = _stress_spec()
    stage = StageInput(stage=STAGE_STOCK, points=(0.0, 8.0),
                       allowable_tensile_mpa=0.2)  # forces a WARNING
    res = compute_handling(spec, stage)
    txt = format_report(spec, [(stage, res)])
    assert "Flexural cracking check" in txt
    assert "Max bottom tensile" in txt
    assert "Allowable tensile stress" in txt
    assert "CRACKING WARNING" in txt
    assert "ratio" in txt


def test_existing_reactions_and_moments_unchanged_by_stress_check_addition():
    """Regression: the V1 reactions / shears / moments are the same with
    or without the stress check enabled."""
    spec = _stress_spec()
    stage_on = StageInput(stage=STAGE_STOCK, points=(1.0, 7.0),
                          stress_check_enabled=True)
    stage_off = StageInput(stage=STAGE_STOCK, points=(1.0, 7.0),
                           stress_check_enabled=False)
    r_on = compute_handling(spec, stage_on)
    r_off = compute_handling(spec, stage_off)
    assert r_on.reactions == r_off.reactions
    assert r_on.v_max == pytest.approx(r_off.v_max)
    assert r_on.m_pos_max == pytest.approx(r_off.m_pos_max)
    assert r_on.m_neg_max == pytest.approx(r_off.m_neg_max)
    assert r_on.stations == r_off.stations
