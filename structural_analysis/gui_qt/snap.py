"""Snap engine — finds the best snap target near the cursor.

Independent from the canvas / GUI framework so it can be unit-tested
without Qt. The canvas calls :func:`SnapEngine.find_snap` on every motion
event and uses the returned :class:`SnapCandidate` to paint a marker and
to anchor click positions.

Snap kinds, in priority order (lower wins):

    EXISTING_NODE       0
    DIAGRAM_EXTREME     0   (only available in post mode, ties with node)
    GRID_INTERSECTION   1
    ELEMENT_ENDPOINT    2
    ELEMENT_MIDPOINT    3
    PROJECTION_ON_ELEM  4

:func:`SnapEngine.find_snap` returns ``None`` when no candidate is
within the configured pixel tolerance — callers should treat that as
"free placement at the raw cursor position".

Pixel distance is computed using a caller-supplied (px_per_dx, px_per_dy)
pair so the engine doesn't depend on matplotlib axes geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

# Priority constants — lower wins when two candidates tie within tolerance.
NODE      = ("node",     0)
DIAGRAM   = ("diagram",  0)   # ties with node; the candidate at smaller
                              # pixel distance wins (so node still wins
                              # when the cursor is exactly on the node).
GRID      = ("grid",     1)
ENDPOINT  = ("endpoint", 2)
MIDPOINT  = ("midpoint", 3)
PROJECT   = ("project",  4)


@dataclass(frozen=True)
class SnapCandidate:
    """A potential snap target near the cursor.

    Attributes:
        x, y: World coordinates of the snap point.
        kind: Snap kind ("node", "grid", "endpoint", "midpoint", "project").
        priority: Lower-wins ordering value.
        screen_distance_px: How far the candidate is from the cursor on screen.
        label: Optional human-readable target ("e3", "A-2", ...).
        object_id: Node id / element id depending on kind (None for grid).
    """

    x: float
    y: float
    kind: str
    priority: int
    screen_distance_px: float
    label: str = ""
    object_id: int | None = None


@dataclass
class SnapEngine:
    """Stateless snap finder. Mutate ``enabled_kinds`` to toggle snap targets."""

    tolerance_px: float = 10.0
    enabled_kinds: set[str] | None = None

    def __post_init__(self) -> None:
        if self.enabled_kinds is None:
            self.enabled_kinds = {
                "node", "diagram", "grid",
                "endpoint", "midpoint", "project",
            }

    def find_snap(self, *, cursor_x: float, cursor_y: float,
                  px_per_dx: float, px_per_dy: float,
                  model, grid=None,
                  diagram_points: list[dict] | None = None,
                  ) -> SnapCandidate | None:
        """Return the best snap candidate or None if nothing is in range.

        ``model`` is the :class:`structural_analysis.model.StructuralModel`.
        ``grid`` is an optional :class:`structural_analysis.gui_qt.grid.GridSystem`.
        ``diagram_points`` is an optional list of per-element diagram
        max/min markers (currently produced by the canvas only when a
        moment / shear / axial diagram is on screen). Each entry is a
        dict with at least ``"x"``, ``"y"``, ``"value"``, ``"unit"``,
        ``"kind"``, and ``"elem_id"``; the engine emits a "diagram"
        snap candidate for each, labelled with the value so the
        status bar can surface it.

        Pixel distance for a candidate at world ``(cx, cy)``:
            d_px = sqrt(((cursor_x - cx) * px_per_dx) ** 2 +
                        ((cursor_y - cy) * px_per_dy) ** 2)
        """
        if px_per_dx <= 0 or px_per_dy <= 0:
            return None

        candidates: list[SnapCandidate] = []

        if "diagram" in self.enabled_kinds and diagram_points:
            for dp in diagram_points:
                dpx = self._dpx(cursor_x, cursor_y,
                                 float(dp["x"]), float(dp["y"]),
                                 px_per_dx, px_per_dy)
                if dpx <= self.tolerance_px:
                    label = (
                        f"{dp['kind']} {float(dp['value']):+.4g} "
                        f"{dp['unit']} at e{dp['elem_id']}"
                    )
                    candidates.append(SnapCandidate(
                        x=float(dp["x"]), y=float(dp["y"]),
                        kind=DIAGRAM[0], priority=DIAGRAM[1],
                        screen_distance_px=dpx,
                        label=label,
                        object_id=int(dp["elem_id"]),
                    ))

        if "node" in self.enabled_kinds:
            for nid, n in model.nodes.items():
                dpx = self._dpx(cursor_x, cursor_y, n.x, n.y,
                                 px_per_dx, px_per_dy)
                if dpx <= self.tolerance_px:
                    candidates.append(SnapCandidate(
                        x=n.x, y=n.y, kind=NODE[0], priority=NODE[1],
                        screen_distance_px=dpx,
                        label=f"node {nid}", object_id=nid,
                    ))

        if "grid" in self.enabled_kinds and grid is not None:
            for xl, yl, xc, yc in grid.intersections():
                dpx = self._dpx(cursor_x, cursor_y, xc, yc,
                                 px_per_dx, px_per_dy)
                if dpx <= self.tolerance_px:
                    candidates.append(SnapCandidate(
                        x=xc, y=yc, kind=GRID[0], priority=GRID[1],
                        screen_distance_px=dpx,
                        label=f"grid {xl.label}-{yl.label}",
                    ))

        if {"endpoint", "midpoint", "project"} & self.enabled_kinds:
            for elem in model.elements:
                ni = model.nodes.get(elem.node_i)
                nj = model.nodes.get(elem.node_j)
                if ni is None or nj is None:
                    continue
                if "endpoint" in self.enabled_kinds:
                    for end_x, end_y, which in (
                        (ni.x, ni.y, "i"), (nj.x, nj.y, "j")
                    ):
                        dpx = self._dpx(cursor_x, cursor_y, end_x, end_y,
                                         px_per_dx, px_per_dy)
                        if dpx <= self.tolerance_px:
                            candidates.append(SnapCandidate(
                                x=end_x, y=end_y,
                                kind=ENDPOINT[0], priority=ENDPOINT[1],
                                screen_distance_px=dpx,
                                label=f"endpoint of e{elem.id}",
                                object_id=elem.id,
                            ))
                if "midpoint" in self.enabled_kinds:
                    mx = 0.5 * (ni.x + nj.x)
                    my = 0.5 * (ni.y + nj.y)
                    dpx = self._dpx(cursor_x, cursor_y, mx, my,
                                     px_per_dx, px_per_dy)
                    if dpx <= self.tolerance_px:
                        candidates.append(SnapCandidate(
                            x=mx, y=my,
                            kind=MIDPOINT[0], priority=MIDPOINT[1],
                            screen_distance_px=dpx,
                            label=f"midpoint of e{elem.id}",
                            object_id=elem.id,
                        ))
                if "project" in self.enabled_kinds:
                    px, py, t, dpx = self._project_px(
                        cursor_x, cursor_y, ni.x, ni.y, nj.x, nj.y,
                        px_per_dx, px_per_dy,
                    )
                    if 0.0 < t < 1.0 and dpx <= self.tolerance_px:
                        candidates.append(SnapCandidate(
                            x=px, y=py,
                            kind=PROJECT[0], priority=PROJECT[1],
                            screen_distance_px=dpx,
                            label=f"on e{elem.id}",
                            object_id=elem.id,
                        ))

        if not candidates:
            return None
        # Lower priority wins; ties broken by smaller pixel distance.
        return min(candidates, key=lambda c: (c.priority, c.screen_distance_px))

    @staticmethod
    def _dpx(cx, cy, x, y, px_per_dx, px_per_dy) -> float:
        dx = (cx - x) * px_per_dx
        dy = (cy - y) * px_per_dy
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _project_px(cx, cy, x1, y1, x2, y2, px_per_dx, px_per_dy):
        """Project (cx,cy) onto segment (x1,y1)-(x2,y2) in pixel space.

        Returns (proj_x, proj_y, t, dpx) where t in [0,1] is the parametric
        position along the segment.
        """
        ax = x1 * px_per_dx
        ay = y1 * px_per_dy
        bx = x2 * px_per_dx
        by = y2 * px_per_dy
        qx = cx * px_per_dx
        qy = cy * px_per_dy
        abx, aby = bx - ax, by - ay
        seg_sq = abx * abx + aby * aby
        if seg_sq < 1e-12:
            return x1, y1, 0.0, ((qx - ax) ** 2 + (qy - ay) ** 2) ** 0.5
        t = ((qx - ax) * abx + (qy - ay) * aby) / seg_sq
        t_clamped = max(0.0, min(1.0, t))
        cx_px = ax + t_clamped * abx
        cy_px = ay + t_clamped * aby
        dpx = ((qx - cx_px) ** 2 + (qy - cy_px) ** 2) ** 0.5
        proj_x = x1 + t_clamped * (x2 - x1)
        proj_y = y1 + t_clamped * (y2 - y1)
        return proj_x, proj_y, t, dpx
