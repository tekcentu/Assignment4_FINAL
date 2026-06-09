"""2-D geometry helpers shared by GUI commands and the canvas.

This module intentionally has no Qt / matplotlib dependencies — it
only knows about world-unit floats. Pixel-space variants of these
operations live alongside the snap engine in
``structural_analysis/gui_qt/snap.py``; they exist for visual
targeting (a 10 px cursor radius) and serve a different concern.
"""

from __future__ import annotations

import math


def project_point_on_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> tuple[float, float, float]:
    """Project ``(px, py)`` onto the line through segment ``(ax,ay)-(bx,by)``.

    Returns ``(proj_x, proj_y, t)`` where ``t`` is the *unclamped*
    parametric coordinate: ``t = 0`` at A, ``t = 1`` at B. Callers
    that want strict-interior projection ("clicked point lies on the
    element, not past an endpoint") should test ``ELEMENT_SPLIT_TOL <
    t < 1 - ELEMENT_SPLIT_TOL`` themselves. The returned ``proj_x``,
    ``proj_y`` are computed from the unclamped ``t`` so callers can
    distinguish "exactly at A" (``t == 0.0``) from "past A"
    (``t < 0.0``) without recomputing.

    A degenerate (zero-length) segment yields ``(ax, ay, 0.0)`` —
    no division by zero, and ``t == 0`` is the only sensible value.
    """
    abx = bx - ax
    aby = by - ay
    seg_sq = abx * abx + aby * aby
    if seg_sq < 1e-24:
        # Degenerate — A and B are the same point. Treat the
        # projection as A itself with parametric 0.
        return ax, ay, 0.0
    t = ((px - ax) * abx + (py - ay) * aby) / seg_sq
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return proj_x, proj_y, t


# ── Physical-member overlay geometry ─────────────────────────────────────────
# These constants and functions are used by gui_qt/canvas.py to draw the
# section-aware physical member overlay.  They live here so the canvas can
# import them without dragging the Qt/matplotlib stack into pure-Python tests.

PHYSICAL_DEPTH_FRACTION: float = 0.02   # 2 % of model bbox diagonal
PHYSICAL_DEPTH_MIN: float = 0.05        # metres — lower clamp
PHYSICAL_DEPTH_MAX: float = 1.0         # metres — upper clamp


def resolved_default_depth(model) -> float:
    """Adaptive fallback section depth for elements that carry depth == 0.

    Returns 2 % of the model bounding-box diagonal, clamped to
    [``PHYSICAL_DEPTH_MIN``, ``PHYSICAL_DEPTH_MAX``] metres.  Falls back to
    ``PHYSICAL_DEPTH_MIN`` for degenerate models (< 2 nodes or zero extent).

    ``model`` is a :class:`StructuralModel`; typed as ``Any`` here so this
    module stays importable without importing the full model layer at the
    top level.
    """
    xs = [n.x for n in model.nodes.values()]
    ys = [n.y for n in model.nodes.values()]
    if len(xs) < 2:
        return PHYSICAL_DEPTH_MIN
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    if diag <= 0.0:
        return PHYSICAL_DEPTH_MIN
    raw = PHYSICAL_DEPTH_FRACTION * diag
    return max(PHYSICAL_DEPTH_MIN, min(PHYSICAL_DEPTH_MAX, raw))


def physical_member_polygon(
    xi: float, yi: float,
    xj: float, yj: float,
    depth: float,
) -> list[tuple[float, float]] | None:
    """4-corner rectangle centred on the segment (xi,yi)→(xj,yj).

    The perpendicular half-width is ``depth / 2``.  Returns ``None`` for
    degenerate zero-length segments so callers can skip safely.
    """
    dx, dy = xj - xi, yj - yi
    L = math.hypot(dx, dy)
    if L == 0.0:
        return None
    nx, ny = -dy / L, dx / L   # unit normal 90° CCW from tangent
    h = 0.5 * depth
    return [
        (xi + h * nx, yi + h * ny),
        (xj + h * nx, yj + h * ny),
        (xj - h * nx, yj - h * ny),
        (xi - h * nx, yi - h * ny),
    ]


def physical_display_thickness(elem, sections) -> float:
    """In-plane visual thickness for a frame/truss body rectangle.

    Rules (visual-only, never used by the solver):

    * **I-section** (``shape_type == "i_section"``): outer envelope =
      ``max(section.depth, section.width)`` — the section's overall depth
      and flange width.  We never use the web thickness, since the body
      rectangle is meant to show the outer footprint.  Using ``max(...)``
      is a safe fallback when the section's in-plane orientation is not
      yet recorded — if the user later rotates the flange, the visual
      envelope still encloses the real profile.
    * **Rectangle / square / manual**: ``section.depth`` (which equals
      the user-entered ``h`` for shaped sections, or the manual depth).
      Falls back to ``section.width`` if depth is zero.
    * **No section / no usable dimension**: ``elem.depth`` (already
      copied from ``section.depth`` at file-load time, or 0.0).

    Returns ``0.0`` when every source is missing — the caller then
    applies the adaptive default (``resolved_default_depth``).

    ``sections`` is the mapping ``model.sections`` (or ``None``).
    """
    sid = getattr(elem, "section_id", None)
    section = sections.get(sid) if (sid is not None and sections) else None
    if section is not None:
        shape = getattr(section, "shape_type", "") or ""
        if shape == "i_section":
            envelope = max(
                float(getattr(section, "depth", 0.0) or 0.0),
                float(getattr(section, "width", 0.0) or 0.0),
            )
            if envelope > 0.0:
                return envelope
        d = float(getattr(section, "depth", 0.0) or 0.0)
        if d > 0.0:
            return d
        w = float(getattr(section, "width", 0.0) or 0.0)
        if w > 0.0:
            return w
    return float(getattr(elem, "depth", 0.0) or 0.0)


def _signed_area(poly: list[tuple[float, float]]) -> float:
    """Shoelace signed area; positive ⇒ CCW (in standard y-up axes)."""
    n = len(poly)
    a2 = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a2 += x1 * y2 - x2 * y1
    return 0.5 * a2


def _as_ccw(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return ``poly`` in counter-clockwise order (reverse if CW)."""
    if _signed_area(poly) < 0.0:
        return list(reversed(poly))
    return list(poly)


def polygon_intersection(
    subject: list[tuple[float, float]],
    clip: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sutherland–Hodgman convex polygon clipping.

    Returns the intersection of two convex polygons (both must be
    convex; in this codebase both are body rectangles, which are
    trivially convex).  Both inputs are normalised to CCW order
    internally, so callers can pass either winding.  Returns an empty
    list when the polygons do not overlap.
    """
    if not subject or not clip:
        return []
    subject_ccw = _as_ccw(subject)
    clip_ccw = _as_ccw(clip)

    def inside(p, a, b):
        # Cross product of (b-a) × (p-a) ≥ 0 ⇒ p is on the left of
        # directed edge a→b (CCW: "inside").
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0

    def line_intersect(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0.0:
            return p1  # parallel edges — caller filters degenerate output
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = subject_ccw
    n = len(clip_ccw)
    for k in range(n):
        if not output:
            break
        input_list = output
        output = []
        a = clip_ccw[k]
        b = clip_ccw[(k + 1) % n]
        s = input_list[-1]
        for e in input_list:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output.append(line_intersect(s, e, a, b))
                output.append(e)
            elif inside(s, a, b):
                output.append(line_intersect(s, e, a, b))
            s = e
    return output


def joint_overlap_regions(
    model,
    element_polygons: dict[int, list[tuple[float, float]]],
) -> list[tuple[
    list[tuple[float, float]],   # intersection polygon (≥ 3 vertices)
    tuple[float, float],         # centroid (cx, cy) — mean of vertices
    tuple[float, float],         # axis-aligned bounding (w, h)
    tuple[int, int],             # element id pair (sorted)
]]:
    """Pairwise convex intersection of frame body polygons at shared nodes.

    For every pair of :class:`FrameElement2D` instances that share an
    analytical node, intersect their physical body polygons.  Non-empty
    intersections (≥ 3 vertices) become joint overlap regions.  Pairs are
    deduplicated, so two frames sharing both endpoints produce a single
    intersection (not two).  Trusses are excluded — joint shading is a
    frame-frame diagnostic.

    Returns ``(polygon, (cx, cy), (w, h), (id_lo, id_hi))`` tuples.
    """
    from collections import defaultdict
    from ..element import FrameElement2D  # lazy — keeps module Qt-free

    by_node: dict[int, list[int]] = defaultdict(list)
    for elem in model.elements:
        if not isinstance(elem, FrameElement2D):
            continue
        if elem.id not in element_polygons:
            continue
        by_node[elem.node_i].append(elem.id)
        by_node[elem.node_j].append(elem.id)

    seen: set[tuple[int, int]] = set()
    regions = []
    for ids in by_node.values():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                if pair in seen:
                    continue
                seen.add(pair)
                clipped = polygon_intersection(
                    element_polygons[pair[0]],
                    element_polygons[pair[1]],
                )
                if len(clipped) < 3:
                    continue
                xs = [p[0] for p in clipped]
                ys = [p[1] for p in clipped]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                regions.append((clipped, (cx, cy), (w, h), pair))
    return regions
