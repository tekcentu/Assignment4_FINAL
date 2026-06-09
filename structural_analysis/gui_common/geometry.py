"""2-D geometry helpers shared by GUI commands and the canvas.

This module intentionally has no Qt / matplotlib dependencies — it
only knows about world-unit floats. Pixel-space variants of these
operations live alongside the snap engine in
``structural_analysis/gui_qt/snap.py``; they exist for visual
targeting (a 10 px cursor radius) and serve a different concern.
"""

from __future__ import annotations

import math
from collections import defaultdict


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


def joint_overlap_nodes(
    model,
    default_depth: float,
) -> list[tuple[float, float, float]]:
    """Nodes where ≥ 2 frame elements meet, returned as ``(x, y, side)``.

    ``side`` is the maximum section depth of the meeting members (or
    ``default_depth`` when none carry a non-zero depth).  Used to draw a
    diagnostic hatched square centred on the joint.

    ``model`` is a :class:`StructuralModel`; typed as ``Any`` here to avoid
    a hard import of the model layer at module scope.
    """
    from ..element import FrameElement2D  # lazy — keeps module importable without Qt
    meet: dict[int, list[float]] = defaultdict(list)
    for elem in model.elements:
        if not isinstance(elem, FrameElement2D):
            continue
        d = elem.depth if elem.depth > 0 else default_depth
        meet[elem.node_i].append(d)
        meet[elem.node_j].append(d)
    result = []
    for nid, depths in meet.items():
        if len(depths) >= 2 and nid in model.nodes:
            n = model.nodes[nid]
            result.append((n.x, n.y, max(depths)))
    return result
