"""Qt-free spatial math for the 3D viewport (v0.33).

Screen-space projection, cursor picking, and ray/plane construction —
kept free of Qt and OpenGL imports so the geometry that decides *what
a click means* is unit-testable headless. The GL widget only supplies
a 4×4 model-view-projection matrix and the viewport size.

Conventions
-----------
* World points are ``(x, y, z)`` in model coordinates (Y up).
* ``mvp`` is a row-major 4×4 numpy array such that
  ``clip = mvp @ [x, y, z, 1]`` (the standard GL pipeline).
* Screen coordinates are pixels with the origin at the TOP-LEFT and y
  growing downward (Qt mouse-event convention).
"""

from __future__ import annotations

import numpy as np


def project_points(
    points: np.ndarray, mvp: np.ndarray, width: float, height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to Qt screen pixels.

    Args:
        points: (N, 3) world coordinates.
        mvp: 4×4 model-view-projection matrix.
        width / height: viewport size in pixels.

    Returns:
        ``(screen, valid)`` — (N, 2) pixel coordinates and an (N,)
        boolean mask. Points behind the camera / at w ≈ 0 are flagged
        invalid (their screen entries are NaN).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = pts.shape[0]
    hom = np.hstack([pts, np.ones((n, 1))])
    clip = hom @ np.asarray(mvp, dtype=float).T
    w = clip[:, 3]
    valid = np.abs(w) > 1e-12
    ndc = np.full((n, 3), np.nan)
    ndc[valid] = clip[valid, :3] / w[valid, None]
    # Clip-space w < 0 means behind the camera for perspective views.
    valid = valid & (w > 0)
    screen = np.full((n, 2), np.nan)
    screen[valid, 0] = (ndc[valid, 0] + 1.0) * 0.5 * width
    screen[valid, 1] = (1.0 - ndc[valid, 1]) * 0.5 * height
    return screen, valid


def nearest_point_index(
    screen: np.ndarray, valid: np.ndarray,
    cursor_x: float, cursor_y: float, radius_px: float,
) -> int | None:
    """Index of the valid projected point nearest the cursor within
    ``radius_px``, or None."""
    if screen.size == 0:
        return None
    d = np.hypot(screen[:, 0] - cursor_x, screen[:, 1] - cursor_y)
    d = np.where(valid, d, np.inf)
    i = int(np.argmin(d))
    return i if d[i] <= radius_px else None


def nearest_segment_index(
    seg_screen: np.ndarray, seg_valid: np.ndarray,
    cursor_x: float, cursor_y: float, radius_px: float,
) -> int | None:
    """Index of the projected segment nearest the cursor.

    Args:
        seg_screen: (N, 2, 2) — per segment, the two endpoint pixel
            coordinates.
        seg_valid: (N,) — both endpoints projected validly.
    """
    segs = np.asarray(seg_screen, dtype=float)
    if segs.size == 0:
        return None
    best_i: int | None = None
    best_d = float(radius_px)
    c = np.array([cursor_x, cursor_y])
    for i in range(segs.shape[0]):
        if not seg_valid[i]:
            continue
        a, b = segs[i, 0], segs[i, 1]
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-12:
            d = float(np.linalg.norm(c - a))
        else:
            t = float(np.clip((c - a) @ ab / denom, 0.0, 1.0))
            d = float(np.linalg.norm(c - (a + t * ab)))
        if d <= best_d:
            best_d = d
            best_i = i
    return best_i


def ray_from_screen(
    cursor_x: float, cursor_y: float,
    mvp: np.ndarray, width: float, height: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Unproject a cursor position into a world-space ray.

    Returns ``(origin, direction)`` (direction normalised), or None
    when the MVP is singular.
    """
    try:
        inv = np.linalg.inv(np.asarray(mvp, dtype=float))
    except np.linalg.LinAlgError:
        return None
    ndc_x = cursor_x / width * 2.0 - 1.0
    ndc_y = 1.0 - cursor_y / height * 2.0
    p_near = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0])
    p_far = inv @ np.array([ndc_x, ndc_y, +1.0, 1.0])
    if abs(p_near[3]) < 1e-12 or abs(p_far[3]) < 1e-12:
        return None
    a = p_near[:3] / p_near[3]
    b = p_far[:3] / p_far[3]
    d = b - a
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        return None
    return a, d / norm


def ray_axis_plane_intersection(
    origin: np.ndarray, direction: np.ndarray,
    axis: int, value: float,
) -> tuple[float, float, float] | None:
    """Intersect a ray with the axis-aligned plane ``coord[axis] == value``.

    Args:
        axis: 0 = X-const, 1 = Y-const (storey plane), 2 = Z-const.

    Returns the world intersection point, or None when the ray is
    (near-)parallel to the plane or the hit lies behind the origin.
    """
    d = float(direction[axis])
    if abs(d) < 1e-12:
        return None
    t = (float(value) - float(origin[axis])) / d
    if t < 0.0:
        return None
    p = np.asarray(origin, dtype=float) + t * np.asarray(direction, dtype=float)
    return (float(p[0]), float(p[1]), float(p[2]))


def model_fingerprint(model) -> tuple:
    """Cheap structural fingerprint used by the viewport's change poll.

    Captures node geometry, element connectivity/kind, supports and
    load counts — enough that any command, undo, or file open changes
    the value. Deliberately NOT a deep hash (cost stays O(n) tuple
    construction per poll tick).
    """
    nodes = tuple(
        (nid, n.x, n.y, getattr(n, "z", 0.0))
        for nid, n in sorted(model.nodes.items())
    )
    elems = tuple(
        (e.id, e.node_i, e.node_j, getattr(e, "kind", ""))
        for e in model.elements
    )
    sups = tuple(
        (nid, s.ux, s.uy, s.rz, s.uz, s.rx, s.ry)
        for nid, s in sorted(model.supports.items())
    )
    return (nodes, elems, sups, len(model.nodal_loads))
