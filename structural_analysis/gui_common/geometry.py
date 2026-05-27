"""2-D geometry helpers shared by GUI commands and the canvas.

This module intentionally has no Qt / matplotlib dependencies — it
only knows about world-unit floats. Pixel-space variants of these
operations live alongside the snap engine in
``structural_analysis/gui_qt/snap.py``; they exist for visual
targeting (a 10 px cursor radius) and serve a different concern.
"""

from __future__ import annotations


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
