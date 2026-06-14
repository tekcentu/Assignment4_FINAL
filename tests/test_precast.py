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


def _spec(L: float = 8.0, w: float = 10.0) -> MemberSpec:
    """A spec with an exact self-weight UDL so reactions are predictable."""
    return MemberSpec(elem_id=1, length=L, self_weight=w, section_name="PC")


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
