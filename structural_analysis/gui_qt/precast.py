"""Precast Handling Stage statics engine (V2) — Qt-free.

Computes support / lifting reactions, sling tensions, and V/M diagrams
for a single precast *frame* member during three temporary handling
stages: lifting, stock / storage support, and truck / transport support.

Design notes
------------
* This is **simple statics on an isolated, horizontal member**. It does
  NOT touch the FEM solver, the assembler, or the live model. The
  selected element is snapshotted into a :class:`MemberSpec` (plain
  floats) so the main model can never be mutated by this tool.

* Every stage uses **exactly two supports / lift points**. Statics
  determinacy keeps the math closed-form; continuous-beam (3+ supports)
  is out of scope for this tool.

* V and M diagrams are NOT computed with a second BMD/SFD formula. The
  handling beam is modelled as a *free–free* member whose member loads
  are the self-weight UDL (plus optional extra UDL / suction) and the
  support reactions injected as upward interior point loads; the shared
  :mod:`structural_analysis.gui_qt.element_graphics` helpers then
  integrate it. This honours the "single source of truth for N/V/M
  math" rule in ``CLAUDE.md``.

* V2 calculates the **horizontal handling position only**. The window
  may *display* the member at the model angle or a custom angle, but
  those are display-only — gravity projection for inclined handling is
  intentionally out of scope (see :data:`DISPLAY_ONLY_NOTE`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..element import FrameElement2D
from ..model import (
    STANDARD_GRAVITY,
    Node,
    PointLoad,
    UniformDistributedLoad,
    effective_material,
)
from .element_graphics import sample_internal_force

# ── Vocabulary ────────────────────────────────────────────────────

STAGE_LIFTING = "lifting"
STAGE_STOCK = "stock"
STAGE_TRUCK = "truck"
STAGES = (STAGE_LIFTING, STAGE_STOCK, STAGE_TRUCK)

STAGE_LABELS = {
    STAGE_LIFTING: "Lifting",
    STAGE_STOCK: "Stock / storage support",
    STAGE_TRUCK: "Truck / transport support",
}

ORIENT_HORIZONTAL = "horizontal"
ORIENT_MODEL = "model_angle"
ORIENT_CUSTOM = "custom"
ORIENTATIONS = (ORIENT_HORIZONTAL, ORIENT_MODEL, ORIENT_CUSTOM)

DISPLAY_ONLY_NOTE = (
    "Diagrams computed for horizontal handling; angle is display-only."
)

# Per-stage default auto-even fractions of L for (point 1, point 2).
DEFAULT_AUTO_POINTS = {
    STAGE_LIFTING: (0.2, 0.8),
    STAGE_STOCK: (0.2, 0.8),
    STAGE_TRUCK: (0.1, 0.9),
}

_POS_TOL = 1e-9


def auto_even_points(stage_key: str, length: float) -> tuple[float, float]:
    """Return the default evenly-spaced two-point layout for a stage."""
    f1, f2 = DEFAULT_AUTO_POINTS.get(stage_key, (0.2, 0.8))
    return (round(f1 * length, 3), round(f2 * length, 3))


# ── Member snapshot ───────────────────────────────────────────────


@dataclass(frozen=True)
class MemberSpec:
    """Immutable snapshot of the selected member's handling-relevant data.

    Built once from the live model via :func:`member_spec_from_element`.
    Holding only plain floats guarantees the tool can never mutate the
    main structural model.
    """

    elem_id: int
    length: float            # m
    self_weight: float       # kN/m  (ρ·A·g / 1000, downward)
    depth: float = 0.0       # m
    area: float = 0.0        # m²   (for σ = N/A; V1 assumes N = 0)
    inertia: float = 0.0     # m⁴   (for σ = M·y/I)
    section_name: str = ""
    model_angle_deg: float = 0.0   # display-only orientation


def member_spec_from_element(model, elem) -> MemberSpec:
    """Snapshot ``elem`` (which must be a frame) into a :class:`MemberSpec`.

    Raises:
        TypeError: if ``elem`` is not a :class:`FrameElement2D`.
        ValueError: if the member has zero / negative length.
    """
    if not isinstance(elem, FrameElement2D):
        raise TypeError(
            f"Precast handling supports frame elements only; element "
            f"{getattr(elem, 'id', '?')} is not a frame."
        )
    L, c, s = elem.length_cos_sin(model.nodes)
    if L <= 0.0:
        raise ValueError(f"Element {elem.id} has non-positive length.")
    # Always re-derive geometry and density from the *live* section + material
    # objects so the precast self-weight tracks edits made in the Section /
    # Material editors. The element's own A / I / rho are kept in sync by the
    # propagation commands, but during mid-edit states they can lag — falling
    # back to them silently has produced "section reads 0 kN/m" reports in
    # the wild (example_09 + a section/material round-trip).
    section = model.sections.get(getattr(elem, "section_id", None))
    try:
        rho = float(effective_material(model, elem).density)
    except (KeyError, AttributeError):
        rho = float(getattr(elem, "rho", 0.0))
    if section is not None:
        A = float(getattr(section, "A", 0.0)) or float(getattr(elem, "A", 0.0))
        inertia = (float(getattr(section, "I", 0.0))
                   or float(getattr(elem, "I", 0.0)))
    else:
        A = float(getattr(elem, "A", 0.0))
        inertia = float(getattr(elem, "I", 0.0))
    w_self = rho * A * STANDARD_GRAVITY / 1000.0  # kN/m
    name = section.name if section and section.name else ""
    # Depth typically lives on the section; the element only carries it when
    # the user opted in for thermal gradient. Fall back to the section so
    # the stress check finds a usable y = depth / 2 by default.
    depth = float(getattr(elem, "depth", 0.0))
    if depth <= 0.0 and section is not None:
        depth = float(getattr(section, "depth", 0.0))
    return MemberSpec(
        elem_id=int(elem.id),
        length=float(L),
        self_weight=w_self,
        depth=depth,
        area=A,
        inertia=inertia,
        section_name=name,
        model_angle_deg=math.degrees(math.atan2(s, c)),
    )


def resolve_single_frame(model, selected_ids):
    """Return the single selected frame element, or raise with a clear
    message the GUI can show verbatim.

    Raises:
        ValueError: if zero / multiple elements are selected, the id is
            missing, or the selected element is not a frame (e.g. a truss).
    """
    ids = list(selected_ids)
    if len(ids) == 0:
        raise ValueError("Select one frame element first.")
    if len(ids) > 1:
        raise ValueError(
            "Select exactly one element — "
            f"{len(ids)} are currently selected."
        )
    eid = ids[0]
    elem = next((e for e in model.elements if e.id == eid), None)
    if elem is None:
        raise ValueError(f"Element {eid} not found in the model.")
    if not isinstance(elem, FrameElement2D):
        raise ValueError(
            f"Element {eid} is a truss; precast handling supports "
            "frame elements only."
        )
    return elem


# ── Stage input / result ──────────────────────────────────────────


@dataclass
class StageInput:
    """User-editable handling-stage parameters (temporary UI state).

    ``points`` is always exactly two positions along the member, from
    end *i* (in metres). The two points must be distinct.

    The flexural-cracking fields (``stress_check_enabled``,
    ``allowable_tensile_mpa``, ``manual_y_top``, ``manual_y_bottom``) are
    physically member-level, but live on each stage's input so the engine
    stays self-contained. The window mirrors a single set of global
    controls into every stage.
    """

    stage: str = STAGE_LIFTING
    points: tuple[float, float] = (0.0, 0.0)   # (x1, x2), 0 ≤ xk ≤ L
    sling_angle_deg: float = 60.0              # lifting only
    daf: float = 1.0                           # per-stage dynamic amp. factor
    enabled: bool = True                       # UI state: skip when False
    manual_weight: float | None = None         # kN/m override of self-weight
    suction: float = 0.0                       # kN/m downward, lifting only
    extra_udl: float = 0.0                     # kN/m downward
    orientation: str = ORIENT_HORIZONTAL
    custom_angle_deg: float = 0.0
    # Flexural cracking check (V1, elastic uncracked, σ = M·y / I).
    stress_check_enabled: bool = True
    allowable_tensile_mpa: float = 2.6
    manual_y_top: float | None = None          # m, overrides depth/2 if set
    manual_y_bottom: float | None = None       # m, overrides depth/2 if set


@dataclass(frozen=True)
class StressCheck:
    """Flexural fiber-stress / cracking summary for one handling stage.

    Sign convention: positive moment from the handling engine is sagging
    (concave-up), so the **bottom fiber goes into tension under positive
    M** and the **top fiber under negative M**. Tension is reported as a
    positive stress (MPa). The ``*_tensile_mpa`` fields are clamped to 0
    when the corresponding fiber stays in compression for the whole
    diagram — so they really are *tensile* peaks, not signed extrema.
    """

    enabled: bool
    skipped: bool
    skip_reason: str
    y_top: float                # m
    y_bottom: float             # m
    top_stations: tuple[tuple[float, float], ...]      # (x, σ_top MPa)
    bottom_stations: tuple[tuple[float, float], ...]   # (x, σ_bot MPa)
    max_top_tensile_mpa: float
    max_top_tensile_x: float
    max_bottom_tensile_mpa: float
    max_bottom_tensile_x: float
    controlling_fiber: str       # "top" / "bottom" / "none"
    controlling_x: float         # x where the peak tensile stress occurs
    allowable_tensile_mpa: float
    cracking_ratio: float        # 0.0 when skipped or no tension
    cracking_status: str         # "OK" / "CRACKING WARNING" / "skipped"


@dataclass(frozen=True)
class HandlingResult:
    """Computed handling-stage outputs (all horizontal-handling statics)."""

    stage: str
    total_load: float                              # kN, downward magnitude
    udl_per_m: float                               # kN/m, incl. DAF
    reactions: tuple[tuple[float, float], ...]     # (x, R) — R>0 is upward
    sling_tensions: tuple[float, ...]
    sling_horizontal: tuple[float, ...]
    v_max: float
    m_pos_max: float
    m_neg_max: float
    stations: tuple[tuple[float, float, float], ...]   # (x, V, M)
    warnings: tuple[str, ...]
    stress_check: StressCheck
    display_note: str = DISPLAY_ONLY_NOTE


# ── Core computation ──────────────────────────────────────────────


def _validate_points(points, length: float) -> list[float]:
    pts = [float(p) for p in points]
    if len(pts) != 2:
        raise ValueError(
            f"Every stage needs exactly 2 support/lift points; got {len(pts)}."
        )
    for p in pts:
        if p < -_POS_TOL or p > length + _POS_TOL:
            raise ValueError(
                f"Support/lift point {p:g} m is outside the member "
                f"[0, {length:g}] m."
            )
    pts = [min(max(p, 0.0), length) for p in pts]
    if abs(pts[0] - pts[1]) < 1e-6:
        raise ValueError(
            "The two support/lift points must be at distinct positions."
        )
    return pts


def _reactions(
    pts: list[float], total_load: float, centroid: float,
    warnings: list[str],
) -> list[tuple[float, float]]:
    """Statically-determinate reactions for two supports under a load
    resultant ``total_load`` acting at ``centroid``."""
    a, b = sorted(pts)
    # ΣM about a: R_b·(b − a) = W·(centroid − a)
    r_b = total_load * (centroid - a) / (b - a)
    r_a = total_load - r_b
    for x, r in ((a, r_a), (b, r_b)):
        if r < -1e-9:
            warnings.append(
                f"Support at {x:g} m develops uplift "
                f"(reaction {r:.3f} kN < 0)."
            )
    return [(a, r_a), (b, r_b)]


def _sample_vm(member: MemberSpec, udl_per_m: float,
               reactions: list[tuple[float, float]], n_samples: int):
    """Build a free–free synthetic member and reuse the shared
    ``element_graphics`` helpers to sample V and M.

    The self-weight UDL is downward (negative local +y); each reaction
    is an upward interior point load (positive local +y). With the
    member-end forces ``f_local`` set to zero, the integrated V and M
    close to zero at the far end exactly when the system is in
    equilibrium — which it is, by construction of the reactions.
    """
    L = member.length
    ni = Node(1, 0.0, 0.0)
    nj = Node(2, L, 0.0)
    loads: list = []
    if udl_per_m != 0.0:
        loads.append(
            UniformDistributedLoad(wy=-udl_per_m, coord_system="local"),
        )
    for x, r in reactions:
        if r != 0.0:
            loads.append(
                PointLoad(py=r, a=min(max(x, 0.0), L), coord_system="local"),
            )
    elem = FrameElement2D(
        id=member.elem_id, node_i=1, node_j=2,
        E=1.0, A=1.0, I=1.0, member_loads=loads,
    )
    f_local = [0.0] * 6
    xs, vs = sample_internal_force(elem, ni, nj, f_local, "shear", n_samples)
    _, ms = sample_internal_force(elem, ni, nj, f_local, "moment", n_samples)
    return xs, vs, ms


def _skipped_stress_check(
    stage: StageInput, reason: str, *, y_top: float = 0.0, y_bot: float = 0.0,
) -> StressCheck:
    return StressCheck(
        enabled=stage.stress_check_enabled,
        skipped=True,
        skip_reason=reason,
        y_top=y_top,
        y_bottom=y_bot,
        top_stations=(),
        bottom_stations=(),
        max_top_tensile_mpa=0.0,
        max_top_tensile_x=0.0,
        max_bottom_tensile_mpa=0.0,
        max_bottom_tensile_x=0.0,
        controlling_fiber="none",
        controlling_x=0.0,
        allowable_tensile_mpa=float(stage.allowable_tensile_mpa),
        cracking_ratio=0.0,
        cracking_status="skipped",
    )


def _stress_check(
    member: MemberSpec,
    stage: StageInput,
    stations: tuple[tuple[float, float, float], ...],
) -> StressCheck:
    """Compute the V1 flexural fiber stress / cracking check.

    σ = M·y / I, evaluated at every station from the moment diagram and
    converted to MPa (×0.001 from kN/m²). Tension is positive. Positive
    moments produce bottom-fiber tension (sagging); negative moments
    produce top-fiber tension (hogging). N = 0 in V1.
    """
    if not stage.stress_check_enabled:
        return _skipped_stress_check(stage, "Stress check disabled by user.")

    inertia = float(member.inertia)
    if inertia <= 0.0:
        return _skipped_stress_check(
            stage,
            f"Section moment of inertia I = {inertia:g} m⁴ is not "
            "positive; cannot compute fiber stresses.",
        )

    # Resolve y_top / y_bottom. Manual override wins; otherwise depth/2 for
    # symmetric sections. Never guess — if neither is available, skip.
    half = float(member.depth) / 2.0 if member.depth > 0.0 else None
    y_top = (float(stage.manual_y_top)
             if stage.manual_y_top is not None else half)
    y_bot = (float(stage.manual_y_bottom)
             if stage.manual_y_bottom is not None else half)
    if y_top is None or y_bot is None:
        return _skipped_stress_check(
            stage,
            "Section depth is not set; enter manual y_top and y_bottom "
            "to run the stress check.",
        )
    if y_top <= 0.0 or y_bot <= 0.0:
        return _skipped_stress_check(
            stage,
            f"Fiber distances y_top = {y_top:g} m, y_bottom = {y_bot:g} m "
            "must both be positive.",
            y_top=y_top, y_bot=y_bot,
        )

    allowable = float(stage.allowable_tensile_mpa)

    # σ = M·y / I  [kN/m²], → MPa via ×0.001.
    # Sagging M > 0 ⇒ bottom tension, top compression. Tension is positive.
    factor_top = -y_top / inertia * 0.001
    factor_bot = +y_bot / inertia * 0.001
    top = tuple((float(x), float(m) * factor_top) for x, _v, m in stations)
    bot = tuple((float(x), float(m) * factor_bot) for x, _v, m in stations)

    max_top_s, max_top_x = 0.0, 0.0
    for x, s in top:
        if s > max_top_s:
            max_top_s, max_top_x = s, x
    max_bot_s, max_bot_x = 0.0, 0.0
    for x, s in bot:
        if s > max_bot_s:
            max_bot_s, max_bot_x = s, x

    if max_top_s >= max_bot_s and max_top_s > 0.0:
        controlling_fiber, controlling_x = "top", max_top_x
        peak = max_top_s
    elif max_bot_s > 0.0:
        controlling_fiber, controlling_x = "bottom", max_bot_x
        peak = max_bot_s
    else:
        controlling_fiber, controlling_x = "none", 0.0
        peak = 0.0

    if allowable <= 0.0:
        return StressCheck(
            enabled=True, skipped=True,
            skip_reason=(f"Allowable tensile stress = {allowable:g} MPa is "
                         "not positive; set a positive limit."),
            y_top=y_top, y_bottom=y_bot,
            top_stations=top, bottom_stations=bot,
            max_top_tensile_mpa=max_top_s, max_top_tensile_x=max_top_x,
            max_bottom_tensile_mpa=max_bot_s, max_bottom_tensile_x=max_bot_x,
            controlling_fiber=controlling_fiber,
            controlling_x=controlling_x,
            allowable_tensile_mpa=allowable,
            cracking_ratio=0.0,
            cracking_status="skipped",
        )

    ratio = peak / allowable
    status = "CRACKING WARNING" if ratio > 1.0 else "OK"
    return StressCheck(
        enabled=True, skipped=False, skip_reason="",
        y_top=y_top, y_bottom=y_bot,
        top_stations=top, bottom_stations=bot,
        max_top_tensile_mpa=max_top_s, max_top_tensile_x=max_top_x,
        max_bottom_tensile_mpa=max_bot_s, max_bottom_tensile_x=max_bot_x,
        controlling_fiber=controlling_fiber,
        controlling_x=controlling_x,
        allowable_tensile_mpa=allowable,
        cracking_ratio=ratio,
        cracking_status=status,
    )


def compute_handling(
    member: MemberSpec, stage: StageInput, *, n_samples: int = 41,
) -> HandlingResult:
    """Compute a handling stage for ``member`` (horizontal handling).

    Raises:
        ValueError: on any invalid input (bad point positions, non-positive
            DAF, negative weights, bad sling angle, …).
    """
    warnings: list[str] = []
    L = float(member.length)
    if L <= 0.0:
        raise ValueError("Member length must be positive.")
    if stage.stage not in STAGES:
        raise ValueError(f"Unknown handling stage {stage.stage!r}.")

    daf = float(stage.daf)
    if daf <= 0.0:
        raise ValueError("DAF must be positive.")
    if daf < 1.0:
        warnings.append(
            f"DAF = {daf:g} is below 1.0 (no dynamic amplification)."
        )
    elif daf > 2.0:
        warnings.append(
            f"DAF = {daf:g} is unusually high (> 2.0) — double-check the "
            "dynamic-amplification assumption for this stage."
        )

    w_self = float(
        member.self_weight if stage.manual_weight is None
        else stage.manual_weight
    )
    if w_self < 0.0:
        raise ValueError("Self-weight / manual weight cannot be negative.")
    w_extra = float(stage.extra_udl)
    if w_extra < 0.0:
        raise ValueError("Extra handling UDL cannot be negative.")

    w_suction = 0.0
    if stage.stage == STAGE_LIFTING:
        w_suction = float(stage.suction)
        if w_suction < 0.0:
            raise ValueError("Suction / adhesion cannot be negative.")
    elif stage.suction:
        warnings.append(
            "Suction / adhesion ignored: it applies to the lifting "
            "stage only."
        )

    udl_per_m = (w_self + w_extra + w_suction) * daf   # kN/m, downward
    total_load = udl_per_m * L                         # kN
    if total_load <= 0.0:
        if stage.manual_weight is None and member.self_weight <= 0.0:
            warnings.append(
                "Section self-weight is zero — the material density and "
                "the section area must both be positive. Set them on the "
                "Material / Section, or enter a manual weight."
            )
        else:
            warnings.append("Total handling load is zero — check the weights.")

    pts = _validate_points(stage.points, L)
    centroid = L / 2.0   # uniform full-length load → midspan
    reactions = _reactions(pts, total_load, centroid, warnings)

    # Sling tensions (lifting only).
    sling_tensions: tuple[float, ...] = ()
    sling_horizontal: tuple[float, ...] = ()
    if stage.stage == STAGE_LIFTING:
        ang = float(stage.sling_angle_deg)
        if not (0.0 < ang <= 90.0):
            raise ValueError(
                "Sling angle must be in (0, 90] degrees from horizontal."
            )
        if ang < 45.0:
            warnings.append(
                f"Sling angle {ang:g}° is below 45° — sling tension T "
                "and horizontal component H are high; verify the rigging."
            )
        theta = math.radians(ang)
        sin_t = math.sin(theta)
        tensions, horiz = [], []
        for _x, r in reactions:
            tensions.append(r / sin_t)
            horiz.append(0.0 if ang == 90.0 else r / math.tan(theta))
        sling_tensions = tuple(tensions)
        sling_horizontal = tuple(horiz)

    # V/M diagrams via the shared single-source helpers.
    xs, vs, ms = _sample_vm(member, udl_per_m, reactions, n_samples)
    if vs:
        v_max = max(abs(v) for v in vs)
        m_pos_max = max([m for m in ms] + [0.0])
        m_neg_max = min([m for m in ms] + [0.0])
        stations = tuple(
            (float(x), float(v), float(m)) for x, v, m in zip(xs, vs, ms)
        )
    else:
        v_max = m_pos_max = m_neg_max = 0.0
        stations = ()

    stress = _stress_check(member, stage, stations)

    return HandlingResult(
        stage=stage.stage,
        total_load=total_load,
        udl_per_m=udl_per_m,
        reactions=tuple(reactions),
        sling_tensions=sling_tensions,
        sling_horizontal=sling_horizontal,
        v_max=v_max,
        m_pos_max=m_pos_max,
        m_neg_max=m_neg_max,
        stations=stations,
        warnings=tuple(warnings),
        stress_check=stress,
        display_note=DISPLAY_ONLY_NOTE,
    )


# ── Report ────────────────────────────────────────────────────────


def format_stage_block(
    stage: StageInput,
    result: HandlingResult,
    *,
    unit_preset: str = "kN_m",
) -> list[str]:
    """Render the lines for one stage in the combined report.

    ``unit_preset`` selects the display units via ``gui_common.units``;
    the default reproduces the legacy report byte-for-byte. Stress stays
    in MPa for V1 regardless of preset (documented limitation).
    """
    from ..gui_common import units as _u
    fL = _u.force_label(unit_preset)
    lL = _u.length_label(unit_preset)
    mL = _u.moment_label(unit_preset)
    uL = _u.udl_label(unit_preset)
    F = _u.force_to_display
    L = _u.length_to_display
    M = _u.moment_to_display
    W = _u.udl_to_display

    lines: list[str] = []
    lines.append(f"── {STAGE_LABELS.get(result.stage, result.stage)} ──")
    if result.stage == STAGE_LIFTING:
        lines.append(f"Sling angle (from horizontal): "
                     f"{stage.sling_angle_deg:g}°")
    lines.append(f"DAF: {stage.daf:g}")
    lines.append(f"Handling UDL (incl. DAF): "
                 f"{W(result.udl_per_m, unit_preset):.4g} {uL}")
    lines.append(f"Total handling load: "
                 f"{F(result.total_load, unit_preset):.4g} {fL}")
    if len(result.reactions) == 2:
        spacing = abs(result.reactions[1][0] - result.reactions[0][0])
        lines.append(f"Support spacing: {L(spacing, unit_preset):.4g} {lL}")
    lines.append(f"Reactions (x [{lL}], R [{fL}], upward +):")
    for i, (x, r) in enumerate(result.reactions):
        extra = ""
        if result.sling_tensions:
            extra = (
                f"   T={F(result.sling_tensions[i], unit_preset):.4g} {fL}"
                f"   H={F(result.sling_horizontal[i], unit_preset):.4g} {fL}"
            )
        lines.append(f"  {L(x, unit_preset):.4g}\t"
                     f"{F(r, unit_preset):.4g}{extra}")
    lines.append(f"Max shear |V|: {F(result.v_max, unit_preset):.4g} {fL}")
    lines.append(f"Max +moment: {M(result.m_pos_max, unit_preset):.4g} {mL}")
    lines.append(f"Max −moment: {M(result.m_neg_max, unit_preset):.4g} {mL}")
    lines.extend(_format_stress_block(result.stress_check))
    if result.warnings:
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")
    return lines


def _format_stress_block(sc: StressCheck) -> list[str]:
    lines = ["Flexural cracking check (elastic, σ = M·y / I):"]
    if sc.skipped:
        lines.append(f"  Skipped: {sc.skip_reason}")
        return lines
    lines.append(f"  y_top = {sc.y_top:g} m   y_bottom = {sc.y_bottom:g} m")
    lines.append(
        f"  Max top tensile = {sc.max_top_tensile_mpa:.3g} MPa at "
        f"x = {sc.max_top_tensile_x:.3g} m"
    )
    lines.append(
        f"  Max bottom tensile = {sc.max_bottom_tensile_mpa:.3g} MPa at "
        f"x = {sc.max_bottom_tensile_x:.3g} m"
    )
    lines.append(
        f"  Allowable tensile stress = {sc.allowable_tensile_mpa:.3g} MPa"
    )
    lines.append(
        f"  Cracking check: {sc.cracking_status}, "
        f"ratio = {sc.cracking_ratio:.3g}"
    )
    return lines


def format_report(
    member: MemberSpec,
    stage_results: list[tuple[StageInput, HandlingResult]],
    *,
    unit_preset: str = "kN_m",
) -> str:
    """Render a plain-text handling-stage report for clipboard / copy.

    ``stage_results`` is a list of ``(StageInput, HandlingResult)`` in the
    order to print (typically lifting, stock, truck). ``unit_preset``
    propagates to every per-stage block; the default keeps the legacy
    text intact.
    """
    from ..gui_common import units as _u
    lL = _u.length_label(unit_preset)
    uL = _u.udl_label(unit_preset)
    L = _u.length_to_display
    W = _u.udl_to_display
    lines: list[str] = []
    lines.append("Precast Handling Stages — V2 (temporary, not saved)")
    lines.append(f"Member: element {member.elem_id}"
                 + (f"  ({member.section_name})" if member.section_name else ""))
    lines.append(f"Length: {L(member.length, unit_preset):.4g} {lL}")
    lines.append(f"Self-weight (section): "
                 f"{W(member.self_weight, unit_preset):.4g} {uL}")
    lines.append("")
    for stage, result in stage_results:
        lines.extend(format_stage_block(stage, result, unit_preset=unit_preset))
        lines.append("")
    lines.append(DISPLAY_ONLY_NOTE)
    return "\n".join(lines)
