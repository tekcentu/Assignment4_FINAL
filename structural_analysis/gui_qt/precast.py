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
    try:
        rho = float(effective_material(model, elem).density)
    except (KeyError, AttributeError):
        rho = float(getattr(elem, "rho", 0.0))
    A = float(getattr(elem, "A", 0.0))
    w_self = rho * A * STANDARD_GRAVITY / 1000.0  # kN/m
    section = model.sections.get(getattr(elem, "section_id", None))
    name = section.name if section and section.name else ""
    return MemberSpec(
        elem_id=int(elem.id),
        length=float(L),
        self_weight=w_self,
        depth=float(getattr(elem, "depth", 0.0)),
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
    """

    stage: str = STAGE_LIFTING
    points: tuple[float, float] = (0.0, 0.0)   # (x1, x2), 0 ≤ xk ≤ L
    sling_angle_deg: float = 60.0              # lifting only
    daf: float = 1.0
    manual_weight: float | None = None         # kN/m override of self-weight
    suction: float = 0.0                       # kN/m downward, lifting only
    extra_udl: float = 0.0                     # kN/m downward
    orientation: str = ORIENT_HORIZONTAL
    custom_angle_deg: float = 0.0


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
        display_note=DISPLAY_ONLY_NOTE,
    )


# ── Report ────────────────────────────────────────────────────────


def format_stage_block(stage: StageInput, result: HandlingResult) -> list[str]:
    """Render the lines for one stage in the combined report."""
    lines: list[str] = []
    lines.append(f"── {STAGE_LABELS.get(result.stage, result.stage)} ──")
    if result.stage == STAGE_LIFTING:
        lines.append(f"Sling angle (from horizontal): "
                     f"{stage.sling_angle_deg:g}°")
    lines.append(f"DAF: {stage.daf:g}")
    lines.append(f"Handling UDL (incl. DAF): {result.udl_per_m:.4g} kN/m")
    lines.append(f"Total handling load: {result.total_load:.4g} kN")
    if len(result.reactions) == 2:
        spacing = abs(result.reactions[1][0] - result.reactions[0][0])
        lines.append(f"Support spacing: {spacing:.4g} m")
    lines.append("Reactions (x [m], R [kN], upward +):")
    for i, (x, r) in enumerate(result.reactions):
        extra = ""
        if result.sling_tensions:
            extra = (f"   T={result.sling_tensions[i]:.4g} kN"
                     f"   H={result.sling_horizontal[i]:.4g} kN")
        lines.append(f"  {x:.4g}\t{r:.4g}{extra}")
    lines.append(f"Max shear |V|: {result.v_max:.4g} kN")
    lines.append(f"Max +moment: {result.m_pos_max:.4g} kN·m")
    lines.append(f"Max −moment: {result.m_neg_max:.4g} kN·m")
    if result.warnings:
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")
    return lines


def format_report(
    member: MemberSpec,
    stage_results: list[tuple[StageInput, HandlingResult]],
) -> str:
    """Render a plain-text handling-stage report for clipboard / copy.

    ``stage_results`` is a list of ``(StageInput, HandlingResult)`` in the
    order to print (typically lifting, stock, truck).
    """
    lines: list[str] = []
    lines.append("Precast Handling Stages — V2 (temporary, not saved)")
    lines.append(f"Member: element {member.elem_id}"
                 + (f"  ({member.section_name})" if member.section_name else ""))
    lines.append(f"Length: {member.length:.4g} m")
    lines.append(f"Self-weight (section): {member.self_weight:.4g} kN/m")
    lines.append("")
    for stage, result in stage_results:
        lines.extend(format_stage_block(stage, result))
        lines.append("")
    lines.append(DISPLAY_ONLY_NOTE)
    return "\n".join(lines)
