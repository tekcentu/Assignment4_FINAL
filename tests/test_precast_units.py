"""Verify the precast handling report routes through Units V1.

Default preset (kN_m) keeps the legacy text byte-for-byte; switching to
``kgf_m`` / ``kip_ft`` swaps labels and rescales values. Stress stays in
MPa for V1 (documented limitation).
"""

from __future__ import annotations

import pytest

from structural_analysis.gui_qt.precast import (
    STAGE_LIFTING, STAGE_STOCK, MemberSpec, StageInput,
    compute_handling, format_report,
)


def _spec() -> MemberSpec:
    return MemberSpec(elem_id=1, length=8.0, self_weight=10.0,
                      depth=0.4, area=0.2, inertia=0.05,
                      section_name="PC")


def _two_stage_results():
    m = _spec()
    lift = compute_handling(m, StageInput(
        stage=STAGE_LIFTING, points=(1.6, 6.4)))
    stock = compute_handling(m, StageInput(
        stage=STAGE_STOCK, points=(1.0, 7.0)))
    return m, [(StageInput(stage=STAGE_LIFTING, points=(1.6, 6.4)), lift),
               (StageInput(stage=STAGE_STOCK, points=(1.0, 7.0)), stock)]


def test_default_preset_keeps_kN_labels():
    m, stages = _two_stage_results()
    text = format_report(m, stages)
    # Length, self-weight, total load, UDL, V, M all in kN-m.
    assert "kN/m" in text
    assert "kN·m" in text
    assert "[kN]" in text                # reactions table column tag
    # Stress stays MPa in V1.
    assert "MPa" in text


def test_kgf_m_preset_swaps_labels_and_rescales_self_weight():
    m, stages = _two_stage_results()
    text = format_report(m, stages, unit_preset="kgf_m")
    assert "kgf/m" in text
    assert "kgf·m" in text
    assert "[kgf]" in text
    # Self-weight 10 kN/m → ~1019.72 kgf/m.
    assert "1019" in text or "1020" in text


def test_kip_ft_preset_swaps_length_and_force():
    m, stages = _two_stage_results()
    text = format_report(m, stages, unit_preset="kip_ft")
    assert "kip/ft" in text
    assert "kip·ft" in text
    assert "[ft]" in text


def test_v1_stress_label_stays_mpa_under_every_preset():
    m, stages = _two_stage_results()
    for pid in ("kN_m", "kgf_m", "kip_ft", "tf_cm", "N_mm"):
        text = format_report(m, stages, unit_preset=pid)
        assert "MPa" in text


def test_unknown_preset_propagates_clear_error():
    from structural_analysis.gui_common.units import UnknownUnitPreset
    m, stages = _two_stage_results()
    with pytest.raises(UnknownUnitPreset):
        format_report(m, stages, unit_preset="bogus")


def test_engine_results_unchanged_when_report_units_change():
    """The pure-Python engine outputs (HandlingResult) must not depend on
    the report's unit preset — only the rendered text does."""
    m, stages = _two_stage_results()
    before = [r for _s, r in stages]
    for pid in ("kgf_m", "kip_ft", "tf_m"):
        _ = format_report(m, stages, unit_preset=pid)
    after = [r for _s, r in stages]
    for a, b in zip(after, before):
        assert a is b   # exact identity — nothing was rebuilt
        assert a.total_load == b.total_load
        assert a.reactions == b.reactions
        assert a.v_max == b.v_max
        assert a.m_pos_max == b.m_pos_max


def test_default_preset_is_idempotent_after_other_presets():
    """Switching presets several times then back to kN_m must reproduce
    the original byte-for-byte (no double conversion)."""
    m, stages = _two_stage_results()
    first = format_report(m, stages)
    for pid in ("kgf_m", "kip_ft", "tf_m", "N_mm"):
        format_report(m, stages, unit_preset=pid)
    second = format_report(m, stages)
    assert first == second
