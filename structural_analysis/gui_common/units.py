"""Display-only unit conversion helper (Global Units V1).

The solver/model layer is fixed at internal units ``kN`` / ``m`` / ``kN·m``
/ ``rad`` / ``MPa``. This module is the single source of truth that the
GUI uses to *display* values in a user-selected unit preset (and, in
later versions, to interpret typed values).

Design notes for V1:
- 15 immutable :class:`UnitPreset` records (force × length pairs).
- Conversion is two-step: a display value is converted to internal kN /
  m via the preset's factors, and back the other way. Composed quantities
  (moment = F·L, distributed load = F/L) reuse the same factors so there
  is exactly one place to fix if a constant is ever wrong.
- **Stress stays in MPa for V1** (`stress_label` returns ``MPa``). Stress
  unit presets are scoped out in V1 — see the project plan.
- Pure Python, no Qt / matplotlib imports, so the helper is reusable from
  ``gui_common`` callers and from CLI report code.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Exact / standard conversion constants (single place) ────────────
# 1 kN = 1000 N (definition).
# 1 kgf = 0.00980665 kN (CIPM standard gravity exact).
# 1 tf  = 9.80665   kN.
# 1 lbf = 0.0044482216152605 kN (NIST exact).
# 1 kip = 1000 lbf = 4.4482216152605 kN.
# 1 m   = 1000 mm; 1 m = 100 cm; 1 m = 1 / 0.3048 ft; 1 m = 1 / 0.0254 in.
_KGF_PER_KN = 1.0 / 0.00980665           # ≈ 101.9716...   (display per kN)
_TF_PER_KN = 1.0 / 9.80665               # ≈ 0.1019716...
_LBF_PER_KN = 1.0 / 0.0044482216152605   # ≈ 224.808943...
_KIP_PER_KN = 1.0 / 4.4482216152605      # ≈ 0.224808943...
_FT_PER_M = 1.0 / 0.3048                 # ≈ 3.28083989...
_IN_PER_M = 1.0 / 0.0254                 # ≈ 39.3700787...


@dataclass(frozen=True)
class UnitPreset:
    """One (force, length) display preset.

    ``force_per_kN`` is how many display force units equal **one kN**
    (so ``display = internal_kN * force_per_kN``).
    ``length_per_m`` is how many display length units equal **one m**.
    The inverse direction (display → internal) divides.

    ``force`` / ``length`` are the bare unit names (``"kN"`` / ``"m"``);
    composed labels (moment, UDL) are derived in the helpers below.
    """

    id: str
    label: str               # human-facing label, e.g. "kN, m"
    force: str               # e.g. "kgf"
    length: str              # e.g. "m"
    force_per_kN: float
    length_per_m: float


# ── 15 V1 presets ───────────────────────────────────────────────────
_PRESETS: dict[str, UnitPreset] = {
    p.id: p for p in (
        # Metric / SI
        UnitPreset("N_mm",  "N, mm",   "N",   "mm", 1000.0, 1000.0),
        UnitPreset("N_m",   "N, m",    "N",   "m",  1000.0, 1.0),
        UnitPreset("kN_mm", "kN, mm",  "kN",  "mm", 1.0,    1000.0),
        UnitPreset("kN_cm", "kN, cm",  "kN",  "cm", 1.0,    100.0),
        UnitPreset("kN_m",  "kN, m",   "kN",  "m",  1.0,    1.0),
        UnitPreset("MN_m",  "MN, m",   "MN",  "m",  0.001,  1.0),
        # Gravity-metric
        UnitPreset("kgf_mm", "kgf, mm", "kgf", "mm", _KGF_PER_KN, 1000.0),
        UnitPreset("kgf_cm", "kgf, cm", "kgf", "cm", _KGF_PER_KN, 100.0),
        UnitPreset("kgf_m",  "kgf, m",  "kgf", "m",  _KGF_PER_KN, 1.0),
        UnitPreset("tf_cm",  "tf, cm",  "tf",  "cm", _TF_PER_KN,  100.0),
        UnitPreset("tf_m",   "tf, m",   "tf",  "m",  _TF_PER_KN,  1.0),
        # Imperial / US
        UnitPreset("lbf_in", "lbf, in", "lbf", "in", _LBF_PER_KN, _IN_PER_M),
        UnitPreset("lbf_ft", "lbf, ft", "lbf", "ft", _LBF_PER_KN, _FT_PER_M),
        UnitPreset("kip_in", "kip, in", "kip", "in", _KIP_PER_KN, _IN_PER_M),
        UnitPreset("kip_ft", "kip, ft", "kip", "ft", _KIP_PER_KN, _FT_PER_M),
    )
}


DEFAULT_PRESET_ID = "kN_m"


class UnknownUnitPreset(ValueError):
    """Raised when an unknown preset id is passed to a conversion helper."""


def preset_ids() -> list[str]:
    """Stable, ordered list of all preset ids (UI populates from this)."""
    return list(_PRESETS.keys())


def preset_label(preset_id: str) -> str:
    """Human-facing label for ``preset_id`` (e.g. ``"kgf, m"``)."""
    return _get(preset_id).label


def all_presets() -> list[UnitPreset]:
    """Ordered list of :class:`UnitPreset` records."""
    return list(_PRESETS.values())


def _get(preset_id: str) -> UnitPreset:
    p = _PRESETS.get(preset_id)
    if p is None:
        raise UnknownUnitPreset(
            f"Unknown unit preset {preset_id!r}; expected one of "
            f"{sorted(_PRESETS)}."
        )
    return p


# ── Conversions: internal kN/m ↔ display ────────────────────────────


def force_to_display(value_kN: float, preset_id: str) -> float:
    return float(value_kN) * _get(preset_id).force_per_kN


def force_from_display(value_display: float, preset_id: str) -> float:
    return float(value_display) / _get(preset_id).force_per_kN


def length_to_display(value_m: float, preset_id: str) -> float:
    return float(value_m) * _get(preset_id).length_per_m


def length_from_display(value_display: float, preset_id: str) -> float:
    return float(value_display) / _get(preset_id).length_per_m


def moment_to_display(value_kNm: float, preset_id: str) -> float:
    p = _get(preset_id)
    return float(value_kNm) * p.force_per_kN * p.length_per_m


def moment_from_display(value_display: float, preset_id: str) -> float:
    p = _get(preset_id)
    return float(value_display) / (p.force_per_kN * p.length_per_m)


def udl_to_display(value_kN_per_m: float, preset_id: str) -> float:
    p = _get(preset_id)
    return float(value_kN_per_m) * p.force_per_kN / p.length_per_m


def udl_from_display(value_display: float, preset_id: str) -> float:
    p = _get(preset_id)
    return float(value_display) * p.length_per_m / p.force_per_kN


def displacement_to_display(value_m: float, preset_id: str) -> float:
    """Displacement is a length — aliased for call-site readability."""
    return length_to_display(value_m, preset_id)


def displacement_from_display(value_display: float, preset_id: str) -> float:
    return length_from_display(value_display, preset_id)


# ── Labels ──────────────────────────────────────────────────────────


def force_label(preset_id: str) -> str:
    return _get(preset_id).force


def length_label(preset_id: str) -> str:
    return _get(preset_id).length


def moment_label(preset_id: str) -> str:
    p = _get(preset_id)
    return f"{p.force}·{p.length}"


def udl_label(preset_id: str) -> str:
    p = _get(preset_id)
    return f"{p.force}/{p.length}"


def displacement_label(preset_id: str) -> str:
    return length_label(preset_id)


def stress_label(_preset_id: str) -> str:
    """V1 keeps stress in MPa regardless of preset (documented limit)."""
    return "MPa"


# ── Formatting convenience ──────────────────────────────────────────


def format_force(value_kN: float, preset_id: str, fmt: str = ".4g") -> str:
    return f"{format(force_to_display(value_kN, preset_id), fmt)} {force_label(preset_id)}"


def format_length(value_m: float, preset_id: str, fmt: str = ".4g") -> str:
    return f"{format(length_to_display(value_m, preset_id), fmt)} {length_label(preset_id)}"


def format_moment(value_kNm: float, preset_id: str, fmt: str = ".4g") -> str:
    return f"{format(moment_to_display(value_kNm, preset_id), fmt)} {moment_label(preset_id)}"


def format_udl(value_kN_per_m: float, preset_id: str, fmt: str = ".4g") -> str:
    return f"{format(udl_to_display(value_kN_per_m, preset_id), fmt)} {udl_label(preset_id)}"


def format_displacement(value_m: float, preset_id: str,
                        fmt: str = ".4g") -> str:
    return f"{format(length_to_display(value_m, preset_id), fmt)} {length_label(preset_id)}"
