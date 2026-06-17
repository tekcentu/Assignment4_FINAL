"""Pure-Python tests for the Global Units V1 display helper.

Covers the 21 conversion / invariance checks called out in the V1 spec
plus the V1 stress-stays-MPa rule. No Qt imports — the helper is in
``gui_common`` so we can exercise it headless without QApplication.
"""

from __future__ import annotations

import pytest

from structural_analysis.gui_common import units as U


# ── 1–5: force kN → other ───────────────────────────────────────────


def test_kN_to_N():
    assert U.force_to_display(1.0, "N_m") == pytest.approx(1000.0)
    assert U.force_to_display(10.0, "N_mm") == pytest.approx(10000.0)


def test_kN_to_kgf():
    # 10 kN ≈ 1019.72 kgf  (1 kgf = 9.80665 N)
    assert U.force_to_display(10.0, "kgf_m") == pytest.approx(
        1019.7162129779282, rel=1e-9)


def test_kN_to_tf():
    # 10 kN ≈ 1.01972 tf
    assert U.force_to_display(10.0, "tf_m") == pytest.approx(
        1.0197162129779282, rel=1e-9)


def test_kN_to_lbf():
    # 10 kN ≈ 2248.09 lbf  (1 lbf = 0.0044482216152605 kN)
    assert U.force_to_display(10.0, "lbf_ft") == pytest.approx(
        2248.0894309971047, rel=1e-9)


def test_kN_to_kip():
    # 10 kN ≈ 2.24809 kip
    assert U.force_to_display(10.0, "kip_ft") == pytest.approx(
        2.248089430997105, rel=1e-9)


# ── 6–9: length m → other ───────────────────────────────────────────


def test_m_to_mm():
    assert U.length_to_display(1.0, "kN_mm") == pytest.approx(1000.0)


def test_m_to_cm():
    assert U.length_to_display(1.0, "kN_cm") == pytest.approx(100.0)


def test_m_to_ft():
    # 1 m = 1 / 0.3048 ft ≈ 3.28084 ft
    assert U.length_to_display(1.0, "kip_ft") == pytest.approx(
        3.2808398950131235, rel=1e-9)


def test_m_to_in():
    assert U.length_to_display(1.0, "kip_in") == pytest.approx(
        39.37007874015748, rel=1e-9)


# ── 10–11: moment ───────────────────────────────────────────────────


def test_kNm_to_kgf_m():
    # Same as force-only scaling (length factor = 1).
    assert U.moment_to_display(10.0, "kgf_m") == pytest.approx(
        1019.7162129779282, rel=1e-9)


def test_kNm_to_kip_ft():
    # 10 kN·m → 2.24809 kip × 3.28084 ft ≈ 7.37562 kip·ft
    assert U.moment_to_display(10.0, "kip_ft") == pytest.approx(
        7.375621492772573, rel=1e-9)


# ── 12–13: distributed load (force / length) ────────────────────────


def test_kNperm_to_kgfperm():
    # length factor cancels (m/m), so same as kN→kgf.
    assert U.udl_to_display(10.0, "kgf_m") == pytest.approx(
        1019.7162129779282, rel=1e-9)


def test_kNperm_to_kipperft():
    # 10 kN/m → 2.24809 kip / 3.28084 ft ≈ 0.685218 kip/ft
    assert U.udl_to_display(10.0, "kip_ft") == pytest.approx(
        0.6852176585675065, rel=1e-9)


# ── 14–15: display → internal direction ─────────────────────────────


def test_kgf_input_to_internal_kN():
    # 1000 kgf ≈ 9.80665 kN
    assert U.force_from_display(1000.0, "kgf_m") == pytest.approx(
        9.80665, rel=1e-12)


def test_kip_input_to_internal_kN():
    # 1 kip = 4.4482216152605 kN exactly.
    assert U.force_from_display(1.0, "kip_ft") == pytest.approx(
        4.4482216152605, rel=1e-15)
    # And 2.248089430997105 kip == 10 kN.
    assert U.force_from_display(2.248089430997105, "kip_ft") == pytest.approx(
        10.0, rel=1e-12)


# ── 16: switching units does not mutate model ───────────────────────


def test_module_has_no_per_call_state():
    """The helper holds no hidden state, so calling conversion functions
    cannot mutate any caller's model or repeated calls drift."""
    for pid in U.preset_ids():
        v0 = U.force_to_display(10.0, pid)
        v1 = U.force_to_display(10.0, pid)
        assert v0 == v1


# ── 17–18: covered by results_view / GUI test files ─────────────────
# (kept here as documentation pointers — see test_units_results_view.py
#  and test_units_gui.py for the report/diagram label coverage.)


# ── 19: old project/input values remain unchanged ───────────────────


def test_old_internal_values_pass_through_unchanged_in_default_preset():
    """In the default kN_m preset the display value equals the internal
    value — proves no implicit scaling for a project saved before V1."""
    for v in (0.0, 1.0, -3.14, 1.0e6):
        assert U.force_to_display(v, "kN_m") == pytest.approx(v)
        assert U.length_to_display(v, "kN_m") == pytest.approx(v)
        assert U.moment_to_display(v, "kN_m") == pytest.approx(v)
        assert U.udl_to_display(v, "kN_m") == pytest.approx(v)


# ── 20: no double conversion after switching presets ────────────────


def test_round_trip_for_every_preset_force_length_moment_udl():
    """display(internal(x)) and internal(display(x)) must both equal x —
    rules out double conversion and lost precision when the user flips
    presets repeatedly."""
    values = (0.0, 1.0, -1.0, 12.345, 9.80665e3, -1.0e-6)
    for pid in U.preset_ids():
        for v in values:
            tol = max(1e-12, abs(v) * 1e-12)
            assert U.force_from_display(
                U.force_to_display(v, pid), pid) == pytest.approx(v, abs=tol)
            assert U.length_from_display(
                U.length_to_display(v, pid), pid) == pytest.approx(v, abs=tol)
            assert U.moment_from_display(
                U.moment_to_display(v, pid), pid) == pytest.approx(v, abs=tol)
            assert U.udl_from_display(
                U.udl_to_display(v, pid), pid) == pytest.approx(v, abs=tol)


def test_round_trip_cycle_through_all_presets_returns_original():
    """Convert kN → preset → back to kN for *every* preset chained, the
    final value must equal the original. Catches any preset that quietly
    composes two factors the wrong way."""
    v = 12.345
    cur = v
    for pid in U.preset_ids():
        disp = U.force_to_display(cur, pid)
        cur = U.force_from_display(disp, pid)
    assert cur == pytest.approx(v, abs=1e-12)


# ── 21: unknown preset is rejected ──────────────────────────────────


def test_unknown_preset_is_rejected_with_clear_error():
    with pytest.raises(U.UnknownUnitPreset, match="hectokips"):
        U.force_to_display(1.0, "hectokips")
    with pytest.raises(U.UnknownUnitPreset):
        U.preset_label("nope")
    with pytest.raises(U.UnknownUnitPreset):
        U.moment_label("nope")


# ── Stress: V1 keeps MPa for every preset ───────────────────────────


def test_stress_label_is_mpa_for_every_preset_in_v1():
    for pid in U.preset_ids():
        assert U.stress_label(pid) == "MPa"


# ── Label helpers produce sensible composed labels ──────────────────


def test_composed_labels_for_moment_and_udl():
    assert U.moment_label("kN_m") == "kN·m"
    assert U.moment_label("kip_ft") == "kip·ft"
    assert U.udl_label("kN_m") == "kN/m"
    assert U.udl_label("kgf_cm") == "kgf/cm"
    assert U.displacement_label("kN_mm") == "mm"


# ── Formatting convenience ──────────────────────────────────────────


def test_format_helpers_produce_value_and_label():
    assert U.format_force(10.0, "kN_m") == "10 kN"
    s = U.format_force(10.0, "kgf_m")
    assert s.endswith(" kgf")
    assert "1020" in s or "1019" in s   # ≈ 1019.72 kgf with .4g
    assert U.format_moment(10.0, "kip_ft").endswith(" kip·ft")
    assert U.format_udl(10.0, "kgf_m").endswith(" kgf/m")
    assert U.format_displacement(1.0, "kN_mm").endswith(" mm")


# ── Sanity on the preset table ──────────────────────────────────────


def test_preset_table_has_exactly_the_15_v1_presets():
    expected = {
        "N_mm", "N_m", "kN_mm", "kN_cm", "kN_m", "MN_m",
        "kgf_mm", "kgf_cm", "kgf_m", "tf_cm", "tf_m",
        "lbf_in", "lbf_ft", "kip_in", "kip_ft",
    }
    assert set(U.preset_ids()) == expected
    assert U.DEFAULT_PRESET_ID == "kN_m"
