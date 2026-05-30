"""Matplotlib QtAgg canvas for the PyQt6 frontend.

``mpl_connect`` events are backend-agnostic, so the drawing and hit-test
code here stays plain matplotlib; only the embedding widget and the
toolbar are Qt-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib
matplotlib.use("QtAgg")  # noqa: E402  must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as _MplPolygon
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT,
)
from matplotlib.ticker import MultipleLocator

from PyQt6.QtWidgets import QWidget, QVBoxLayout

from ..element import FrameElement2D, TrussElement2D
from ..model import (
    FrameTemperatureLoad,
    NodalLoad,
    PointLoad,
    StructuralModel,
    Support,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)
from .grid import GridSystem
from .snap import SnapCandidate, SnapEngine
from .element_graphics import (
    sample_internal_force as _diagram_ordinates,
    internal_force_at as _diagram_value,
)


@dataclass
class HitResult:
    """What was under the mouse at the time of an event."""
    x: float
    y: float
    node_id: Optional[int] = None
    element_id: Optional[int] = None
    snap_kind: str = ""    # e.g. "node", "grid", "midpoint", "endpoint", "project"
    snap_label: str = ""   # human-readable target description


class ModelCanvas(QWidget):
    """A QWidget containing the matplotlib figure + its navigation toolbar."""

    NODE_PICK_RADIUS_PX = 12
    ELEM_PICK_RADIUS_PX = 8

    def __init__(self, parent: QWidget | None,
                 model_provider: Callable[[], StructuralModel],
                 grid_provider: Callable[[], GridSystem] | None = None) -> None:
        super().__init__(parent)
        self._model = model_provider
        self._grid_provider = grid_provider or (lambda: GridSystem())
        self.grid_spacing: float = 0.5
        self.snap_enabled: bool = True
        self.snap_engine = SnapEngine(tolerance_px=10.0)

        self.show_deformed: bool = True
        self.show_reactions: bool = True
        self.show_diagrams: bool = False
        self.show_section_labels: bool = False
        self.diagram_kind: str = "moment"
        self.deformed_scale: float = 1.0
        self.diagram_scale: float = 1.0
        # Station points are post-processing samples per element, used to
        # draw smooth deformed shapes (cubic-Hermite for frames) and
        # internal-force diagrams. They are NOT model nodes and never
        # affect connectivity, supports, or loads. 21 picks up the
        # midspan exactly and gives smooth UDL / point-load diagrams.
        self.deformed_stations: int = 21
        self.diagram_stations: int = 21
        self._result = None
        self._modal_result = None    # ModalResult or None
        self._modal_mode_idx: int = 0
        self._modal_scale: float = 1.0
        self._snap_marker = None  # current SnapCandidate
        # Either anchored at an existing node id (legacy) or a free
        # start point (v0.10.0 — first click landed on empty space and
        # no node has been created yet). Exactly one is non-None at a
        # time.
        self._element_preview: tuple[int, float, float, str] | None = None
        self._element_preview_free: (
            tuple[float, float, float, float, str] | None
        ) = None
        # Multi-select state (v0.13.0). Each set holds all currently
        # selected node / element ids. Single-object selection is just
        # a one-element set; box-select fills these in bulk.
        self._selected_node_ids: set[int] = set()
        self._selected_element_ids: set[int] = set()
        # Active drag-rectangle for box selection. Tuple:
        # ``(x0, y0, x1, y1, is_crossing)`` in world coords, or None
        # when no drag is in progress. ``is_crossing`` is True for
        # right-to-left drags (Crossing mode) and False for
        # left-to-right drags (Window mode).
        self._drag_rect: (
            tuple[float, float, float, float, bool] | None
        ) = None
        # Per-element max / min markers on the currently-drawn moment /
        # shear / axial diagram. Populated by _draw_diagrams and fed
        # into the snap engine so the cursor snaps to those points in
        # post mode. Empty when no diagram is showing.
        self._diagram_critical_points: list[dict] = []
        # Fallback hover cursor when no snap candidate is active. This
        # is what the user sees when their cursor is over empty space
        # between grid lines — it marks the point a left-click would
        # actually land on (the rectangular-grid-snapped coords).
        self._hover_xy: tuple[float, float] | None = None
        # Whether the view has been initialised at least once. Until
        # this is True, redraw() auto-fits the model + grid extent. Once
        # set, redraw() preserves the user's current xlim/ylim so
        # placing a node or moving the mouse no longer collapses the
        # zoom level.
        self._view_initialised: bool = False
        # Tracks whether the user has manually adjusted the view (scroll
        # zoom, middle-button pan). While False, a window resize is free
        # to re-fit the data limits to match the new widget aspect so
        # the grid fills the canvas. Once True, the user owns the
        # viewport — resize keeps the current limits and only fit_to_view
        # (View → Fit) takes the wheel back.
        self._user_view_dirty: bool = False
        # Guard flipped to True while we (canvas internals) are
        # programmatically setting xlim/ylim — first fit, redraw's
        # save-and-restore, fit_to_view. The xlim/ylim-changed mpl
        # callbacks consult this flag so the dirty bit only flips for
        # *external* mutations (matplotlib navigation toolbar pan/zoom,
        # programmatic test pokes, etc.).
        self._setting_axes_limits: bool = False

        # Middle-mouse-drag pan state (display coordinates at drag start).
        self._pan_origin: tuple[float, float] | None = None
        self._pan_xlim0: tuple[float, float] = (0.0, 1.0)
        self._pan_ylim0: tuple[float, float] = (0.0, 1.0)

        # The pixel-coord parameter is passed alongside the world-coord
        # HitResult so tools (currently SelectTool) can drive direction-
        # aware drag selection independently of axis flips or zoom.
        self.on_click: (
            Callable[[HitResult, str, tuple[float, float], bool], None] | None
        ) = None
        self.on_motion: (
            Callable[[HitResult, tuple[float, float]], None] | None
        ) = None
        self.on_release: (
            Callable[[HitResult, str, tuple[float, float], bool], None] | None
        ) = None
        # Cache of the most recent hit + event pixel coords so
        # _handle_release can route a HitResult without re-running the
        # full hit-test (mpl release events sometimes arrive with no
        # xdata/ydata, e.g. when the cursor leaves the axes mid-drag).
        self._last_hit: HitResult | None = None
        self._last_event_px: tuple[float, float] | None = None

        self.fig = plt.Figure(figsize=(7.5, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        # adjustable="box" lets the user's xlim/ylim be honored exactly
        # (matplotlib resizes the axes rectangle to keep aspect=1).
        # The legacy "datalim" mode would silently rewrite our limits
        # and emit "Ignoring fixed y limits…" on every redraw.
        self.ax.set_aspect("equal", adjustable="box")
        # Mark the view dirty whenever something *outside* canvas
        # internals changes the limits — most importantly the
        # matplotlib navigation toolbar's pan/zoom modes, which
        # otherwise leave _user_view_dirty False and let the next
        # window resize silently discard the user's view.
        self.ax.callbacks.connect("xlim_changed", self._on_limits_changed)
        self.ax.callbacks.connect("ylim_changed", self._on_limits_changed)

        self._mpl_canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self._mpl_canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self._mpl_canvas)
        self.setLayout(layout)

        self._mpl_canvas.mpl_connect("button_press_event", self._handle_click)
        self._mpl_canvas.mpl_connect("button_release_event", self._handle_release)
        self._mpl_canvas.mpl_connect("motion_notify_event", self._handle_motion)
        self._mpl_canvas.mpl_connect("scroll_event", self._handle_scroll)

    # ── public API ──

    def set_grid_spacing(self, spacing: float) -> None:
        if spacing <= 0:
            raise ValueError("Grid spacing must be > 0.")
        self.grid_spacing = float(spacing)
        self.redraw()

    def toggle_snap(self, enabled: bool) -> None:
        self.snap_enabled = bool(enabled)

    def set_element_preview(
        self, start_node_id: int, end_x: float, end_y: float, kind: str
    ) -> None:
        """Show a temporary member preview anchored at an existing node."""
        self._element_preview = (
            int(start_node_id), float(end_x), float(end_y), str(kind)
        )
        self._element_preview_free = None

    def set_element_preview_free(
        self,
        start_x: float, start_y: float, end_x: float, end_y: float,
        kind: str,
    ) -> None:
        """Show a temporary member preview when the start is empty space.

        v0.10.0: the Frame / Truss tools now accept a first click on
        empty space (a node will be auto-created on the second click).
        While only the first click has landed, there is no existing
        node id to anchor the preview to.
        """
        self._element_preview_free = (
            float(start_x), float(start_y),
            float(end_x), float(end_y),
            str(kind),
        )
        self._element_preview = None

    def clear_element_preview(self) -> None:
        self._element_preview = None
        self._element_preview_free = None

    def select_node(self, node_id: int) -> None:
        """Exclusive single-node selection — clears everything else."""
        self._selected_node_ids = {int(node_id)}
        self._selected_element_ids = set()

    def select_element(self, element_id: int) -> None:
        """Exclusive single-element selection — clears everything else."""
        self._selected_element_ids = {int(element_id)}
        self._selected_node_ids = set()

    def add_node_to_selection(self, node_id: int) -> None:
        self._selected_node_ids.add(int(node_id))

    def remove_node_from_selection(self, node_id: int) -> None:
        self._selected_node_ids.discard(int(node_id))

    def add_element_to_selection(self, element_id: int) -> None:
        self._selected_element_ids.add(int(element_id))

    def remove_element_from_selection(self, element_id: int) -> None:
        self._selected_element_ids.discard(int(element_id))

    def get_selected_nodes(self) -> frozenset[int]:
        return frozenset(self._selected_node_ids)

    def get_selected_elements(self) -> frozenset[int]:
        return frozenset(self._selected_element_ids)

    def clear_selection(self) -> None:
        self._selected_node_ids = set()
        self._selected_element_ids = set()

    def set_drag_rect(
        self, x0: float, y0: float, x1: float, y1: float,
        is_crossing: bool,
    ) -> None:
        """Set the active box-select rectangle (world coords + direction)."""
        self._drag_rect = (
            float(x0), float(y0), float(x1), float(y1), bool(is_crossing),
        )

    def clear_drag_rect(self) -> None:
        self._drag_rect = None

    def set_result(self, result) -> None:
        self._result = result
        # Static and modal results are mutually exclusive on screen.
        self._modal_result = None
        self.redraw()

    def clear_result(self) -> None:
        self._result = None
        self.redraw()

    def set_modal_result(self, modal_result, mode_idx: int = 0,
                         scale: float = 1.0) -> None:
        """Display a single mode of a ModalResult on the canvas.

        Static-result overlays are cleared while a modal result is shown
        (the canvas displays one analysis kind at a time).
        """
        self._result = None
        self._modal_result = modal_result
        self._modal_mode_idx = max(0, int(mode_idx))
        self._modal_scale = float(scale)
        self.redraw()

    def update_modal_view(self, mode_idx: int, scale: float) -> None:
        """Update the displayed mode index and/or scale for the current
        :class:`ModalResult` without rebuilding it."""
        if self._modal_result is None:
            return
        self._modal_mode_idx = max(0, int(mode_idx))
        self._modal_scale = float(scale)
        self.redraw()

    def clear_modal_result(self) -> None:
        self._modal_result = None
        self.redraw()

    def fit_to_view(self) -> None:
        """Re-fit the axes to enclose the current model + grid extent.

        ``redraw()`` preserves the user's pan/zoom state by default, so
        callers need to invoke this explicitly to reset the view (e.g.
        after loading a file or via the View → Fit action).
        """
        self._view_initialised = False
        self._user_view_dirty = False
        self.redraw()

    def redraw(self) -> None:
        # Preserve the current view across the clear()/redraw cycle so
        # mouse motion and node-placement events don't collapse the
        # zoom level. On the very first redraw we have nothing to
        # preserve — call _set_axes_limits to fit the (possibly empty)
        # model + grid, and remember that we did.
        if self._view_initialised:
            saved_xlim = self.ax.get_xlim()
            saved_ylim = self.ax.get_ylim()
        else:
            saved_xlim = saved_ylim = None

        # ax.clear() resets xlim/ylim and would fire the limit-changed
        # callbacks; the restore call below would also fire them.
        # Both are programmatic, so suppress the dirty-bit toggle for
        # the duration. (_set_axes_limits handles its own guard.)
        self._setting_axes_limits = True
        try:
            self.ax.clear()
            self.ax.set_aspect("equal", adjustable="box")

            if saved_xlim is not None:
                self.ax.set_xlim(saved_xlim)
                self.ax.set_ylim(saved_ylim)
            else:
                self._set_axes_limits()
                self._view_initialised = True
        finally:
            self._setting_axes_limits = False

        self._draw_grid()
        self._draw_origin_axes()
        self._draw_model()
        self._draw_selection()
        self._draw_element_preview()
        if self._result is not None and self._result.status == "ok":
            if self.show_deformed:
                self._draw_deformed()
            if self.show_reactions:
                self._draw_reactions()
            if self.show_diagrams:
                self._draw_diagrams()
            else:
                # No diagram on screen → no critical-point snaps active.
                self._diagram_critical_points = []
        elif self._modal_result is not None and self._modal_result.status == "ok":
            self._diagram_critical_points = []
            self._draw_mode_shape()
        else:
            self._diagram_critical_points = []
        self._draw_snap_marker()
        self._draw_drag_rect()
        self._mpl_canvas.draw_idle()

    # ── Qt resize → re-fit while the user hasn't taken the wheel ──

    def resizeEvent(self, event) -> None:
        """When the widget is resized and the user hasn't manually
        panned or zoomed yet, re-fit the data limits so the data box's
        aspect matches the new widget aspect — i.e. the grid keeps
        filling the canvas instead of collapsing to a centred square.

        Once the user pans or scroll-zooms, ``_user_view_dirty`` is
        True and we leave the limits alone. ``fit_to_view`` (View →
        Fit) resets the flag and re-engages auto-fit.
        """
        super().resizeEvent(event)
        if self._user_view_dirty:
            return
        if not self._view_initialised:
            # First-ever paint hasn't happened yet; redraw() will set
            # the limits using the new widget size.
            return
        self._set_axes_limits()
        self._mpl_canvas.draw_idle()

    # ── event forwarding ──

    def _handle_click(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        if event.button == 2 and not self.toolbar.mode:
            # Middle-button drag — start pan.
            self._pan_origin = (event.x, event.y)
            self._pan_xlim0 = tuple(self.ax.get_xlim())
            self._pan_ylim0 = tuple(self.ax.get_ylim())
            return
        if self.on_click is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.toolbar.mode:
            # The matplotlib navigation toolbar is in pan or zoom mode,
            # so it absorbs every left-click on the canvas. Surface a
            # status hint via the on_nav_mode_block callback (if the
            # host has wired one up) — otherwise silently swallow.
            cb = getattr(self, "on_nav_mode_block", None)
            if cb is not None:
                try:
                    cb(str(self.toolbar.mode))
                except Exception:
                    pass
            return
        hit = self._hit_test(event)
        button_name = {1: "left", 2: "middle", 3: "right"}.get(event.button, "left")
        event_px = (float(event.x), float(event.y))
        self._last_hit = hit
        self._last_event_px = event_px
        shift = _shift_pressed()
        self.on_click(hit, button_name, event_px, shift)

    def _handle_motion(self, event) -> None:
        if self._pan_origin is not None:
            # Middle-mouse drag in progress — pan without redrawing the model.
            inv = self.ax.transData.inverted()
            x0d, y0d = inv.transform(self._pan_origin)
            xcd, ycd = inv.transform((event.x, event.y))
            dx, dy = xcd - x0d, ycd - y0d
            xl, xr = self._pan_xlim0
            yb, yt = self._pan_ylim0
            self.ax.set_xlim(xl - dx, xr - dx)
            self.ax.set_ylim(yb - dy, yt - dy)
            self._user_view_dirty = True
            self._mpl_canvas.draw_idle()
            return
        if self.on_motion is None or event.inaxes is not self.ax:
            # Cursor left the axes — drop the hover marker.
            if self._hover_xy is not None:
                self._hover_xy = None
                self._mpl_canvas.draw_idle()
            return
        if event.xdata is None or event.ydata is None:
            return
        hit = self._hit_test(event)
        # Record the position a click would land on, so the hover
        # marker can be drawn even when the snap engine has no
        # candidate (empty space between labeled grid lines).
        self._hover_xy = (hit.x, hit.y)
        event_px = (float(event.x), float(event.y))
        self._last_hit = hit
        self._last_event_px = event_px
        try:
            self.on_motion(hit, event_px)
        except Exception:
            pass

    def _handle_release(self, event) -> None:
        if event.button == 2:
            self._pan_origin = None
            return
        if event.button != 1 or self.on_release is None:
            return
        # Release events sometimes lack xdata/ydata (cursor outside
        # axes). Reuse the last recorded hit + pixel position so the
        # tool can still finish a drag cleanly.
        hit = self._last_hit
        event_px = self._last_event_px
        if event.xdata is not None and event.ydata is not None:
            event_px = (float(event.x), float(event.y))
            try:
                hit = self._hit_test(event)
                self._last_hit = hit
                self._last_event_px = event_px
            except Exception:
                pass
        if hit is None or event_px is None:
            return
        shift = _shift_pressed()
        try:
            self.on_release(hit, "left", event_px, shift)
        except Exception:
            pass

    def _handle_scroll(self, event) -> None:
        if event.inaxes is not self.ax or self.toolbar.mode:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Scroll up → zoom in (shrink the visible range); scroll down → zoom out.
        factor = 1.0 / 1.15 if event.button == "up" else 1.15
        xl, xr = self.ax.get_xlim()
        yb, yt = self.ax.get_ylim()
        xd, yd = event.xdata, event.ydata
        self.ax.set_xlim(xd - (xd - xl) * factor, xd + (xr - xd) * factor)
        self.ax.set_ylim(yd - (yd - yb) * factor, yd + (yt - yd) * factor)
        self._user_view_dirty = True
        self._mpl_canvas.draw_idle()

    # ── geometry / hit-test ──

    def _snap(self, x: float, y: float) -> tuple[float, float]:
        if not self.snap_enabled:
            return x, y
        s = self.grid_spacing
        return round(x / s) * s, round(y / s) * s

    def _hit_test(self, event) -> HitResult:
        model = self._model()
        grid = self._grid_provider()

        bbox = self.ax.bbox
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        px_per_dx = bbox.width / max(x1 - x0, 1e-9)
        px_per_dy = bbox.height / max(y1 - y0, 1e-9)

        # Run the snap engine.
        candidate = None
        if self.snap_enabled:
            candidate = self.snap_engine.find_snap(
                cursor_x=event.xdata, cursor_y=event.ydata,
                px_per_dx=px_per_dx, px_per_dy=px_per_dy,
                model=model, grid=grid,
                diagram_points=self._diagram_critical_points or None,
            )
        self._snap_marker = candidate

        if candidate is not None:
            hit = HitResult(x=candidate.x, y=candidate.y,
                            snap_kind=candidate.kind,
                            snap_label=candidate.label)
            if candidate.kind == "node":
                hit.node_id = candidate.object_id
            elif candidate.kind in ("endpoint", "midpoint", "project",
                                     "diagram"):
                hit.element_id = candidate.object_id
            else:
                # Non-element snap (e.g. "grid"): the snap engine
                # prefers grid over project, so a click on an
                # element's interior near a grid intersection arrives
                # here. Still attach the nearest element so the
                # NodeTool / _PairTool split path can engage —
                # otherwise the disconnected-component bug returns
                # whenever a labeled grid is configured (PR #21 review,
                # codex P1).
                hit.element_id = self._pick_nearest_element_px(
                    event.xdata, event.ydata, px_per_dx, px_per_dy, model,
                )
            return hit

        # No snap → fall back to rectangular-grid snapping + element pick.
        sx, sy = self._snap(event.xdata, event.ydata)
        hit = HitResult(x=sx, y=sy)
        hit.element_id = self._pick_nearest_element_px(
            event.xdata, event.ydata, px_per_dx, px_per_dy, model,
        )
        return hit

    def _pick_nearest_element_px(
        self, x: float, y: float,
        px_per_dx: float, px_per_dy: float, model,
    ) -> Optional[int]:
        """Return the id of the element closest to ``(x, y)`` within
        :attr:`ELEM_PICK_RADIUS_PX`, or ``None``. Shared by the
        no-snap fallback and the grid-snap branch — keeps the
        element-pick logic in one place (PR #21 review)."""
        best_eid = None
        best_dpx = self.ELEM_PICK_RADIUS_PX
        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            dpx = _point_segment_distance_px(
                x, y, ni.x, ni.y, nj.x, nj.y, px_per_dx, px_per_dy,
            )
            if dpx < best_dpx:
                best_dpx = dpx
                best_eid = elem.id
        return best_eid

    # ── drawing ──

    def _draw_grid(self) -> None:
        grid = self._grid_provider()
        if grid.is_empty():
            # Fall back to a uniform-spacing grid.
            self.ax.xaxis.set_major_locator(MultipleLocator(self.grid_spacing))
            self.ax.yaxis.set_major_locator(MultipleLocator(self.grid_spacing))
            self.ax.grid(True, which="major", linestyle=":", linewidth=0.5,
                         color="#cccccc")
            return
        # Draw the labeled grid manually. Don't enable matplotlib's auto-grid.
        self.ax.grid(False)
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        # Adjust limits to encompass the grid extent if needed.
        if grid.x_lines:
            x0 = min(x0, min(ln.coord for ln in grid.x_lines))
            x1 = max(x1, max(ln.coord for ln in grid.x_lines))
        if grid.y_lines:
            y0 = min(y0, min(ln.coord for ln in grid.y_lines))
            y1 = max(y1, max(ln.coord for ln in grid.y_lines))
        for ln in grid.x_lines:
            self.ax.axvline(ln.coord, color="#aac8ff", linewidth=0.7,
                            linestyle="-", alpha=0.6, zorder=0)
            self.ax.text(ln.coord, y1, f"  {ln.label}", color="#3060c0",
                         fontsize=8, va="bottom", ha="center", zorder=1)
        for ln in grid.y_lines:
            self.ax.axhline(ln.coord, color="#aac8ff", linewidth=0.7,
                            linestyle="-", alpha=0.6, zorder=0)
            self.ax.text(x1, ln.coord, f"  {ln.label}", color="#3060c0",
                         fontsize=8, va="center", ha="left", zorder=1)

    def _draw_origin_axes(self) -> None:
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        if not (x0 <= 0.0 <= x1 and y0 <= 0.0 <= y1):
            return
        span = max(x1 - x0, y1 - y0, 1.0)
        length = 0.08 * span
        self.ax.plot(0.0, 0.0, marker="o", markersize=4,
                     color="#222222", zorder=8)
        self.ax.annotate(
            "", xy=(length, 0.0), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.4),
            zorder=8,
        )
        self.ax.annotate(
            "", xy=(0.0, length), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.4),
            zorder=8,
        )
        self.ax.annotate("0,0", (0.0, 0.0), xytext=(4, -14),
                         textcoords="offset points", fontsize=8,
                         color="#222222", zorder=9)
        self.ax.annotate("X", (length, 0.0), xytext=(4, -2),
                         textcoords="offset points", fontsize=8,
                         color="#222222", zorder=9)
        self.ax.annotate("Y", (0.0, length), xytext=(4, 2),
                         textcoords="offset points", fontsize=8,
                         color="#222222", zorder=9)

    def _draw_snap_marker(self) -> None:
        c = self._snap_marker
        if c is not None:
            marker_styles = {
                "node":     ("o", "#ff7f0e"),  # filled circle, orange
                "diagram":  ("*", "#d62728"),  # star, red — post mode
                "grid":     ("s", "#1f77b4"),  # square, blue
                "endpoint": ("^", "#9467bd"),  # triangle, purple
                "midpoint": ("D", "#17becf"),  # diamond, cyan
                "project":  ("x", "#2ca02c"),  # x, green
            }
            marker, color = marker_styles.get(c.kind, ("o", "#888"))
            self.ax.plot(c.x, c.y, marker=marker, color=color, markersize=12,
                         markerfacecolor="none", markeredgewidth=2,
                         zorder=10)
            return
        # No real snap candidate → draw a faint "ghost" crosshair at
        # the rectangular-grid-snapped cursor so the user always knows
        # where a left-click would land.
        if self._hover_xy is None:
            return
        x, y = self._hover_xy
        self.ax.plot(x, y, marker="+", color="#888888", markersize=14,
                     markeredgewidth=1.5, alpha=0.7, zorder=10)

    def _draw_element_preview(self) -> None:
        if self._element_preview is not None:
            start_node_id, end_x, end_y, kind = self._element_preview
            start = self._model().nodes.get(start_node_id)
            if start is None:
                return
            start_x, start_y = start.x, start.y
        elif self._element_preview_free is not None:
            start_x, start_y, end_x, end_y, kind = self._element_preview_free
        else:
            return
        is_frame = kind == "frame"
        color = "#1f77b4" if is_frame else "#d62728"
        linestyle = "-" if is_frame else "--"
        self.ax.plot(
            [start_x, end_x], [start_y, end_y],
            color=color, linestyle=linestyle, linewidth=2.4,
            alpha=0.55, zorder=3,
        )
        self.ax.plot(
            end_x, end_y, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor=color,
            alpha=0.85, zorder=9,
        )
        if self._element_preview_free is not None:
            # Mark the free start point too (no real node there yet),
            # so the user has visual confirmation of click 1.
            self.ax.plot(
                start_x, start_y, marker="o", markersize=5,
                markerfacecolor="white", markeredgecolor=color,
                alpha=0.85, zorder=9,
            )

    def _draw_selection(self) -> None:
        model = self._model()
        # Paint element-band highlights behind the element line so the
        # crisp element stroke still reads through.
        for eid in self._selected_element_ids:
            elem = next((e for e in model.elements if e.id == eid), None)
            if elem is None:
                continue
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            self.ax.plot(
                [ni.x, nj.x], [ni.y, nj.y],
                color="#ffbf00", linewidth=6.0, alpha=0.45,
                solid_capstyle="round", zorder=1.5,
            )
        # Paint node highlights on top (orange ring).
        for nid in self._selected_node_ids:
            node = model.nodes.get(nid)
            if node is None:
                continue
            self.ax.plot(
                node.x, node.y, marker="o", markersize=13,
                markerfacecolor="none", markeredgecolor="#ffbf00",
                markeredgewidth=2.4, zorder=11,
            )

    def _draw_drag_rect(self) -> None:
        if self._drag_rect is None:
            return
        from matplotlib.patches import Rectangle
        x0, y0, x1, y1, is_crossing = self._drag_rect
        rx = min(x0, x1)
        ry = min(y0, y1)
        rw = abs(x1 - x0)
        rh = abs(y1 - y0)
        if is_crossing:
            # Right-to-left drag → Crossing mode: dashed green outline,
            # semi-transparent green fill. Selects anything the rect
            # touches.
            edge = "#2da44e"
            face = "#2da44e"
            ls = "--"
        else:
            # Left-to-right drag → Window mode: solid blue outline,
            # semi-transparent blue fill. Selects only fully enclosed
            # objects.
            edge = "#1f6feb"
            face = "#1f6feb"
            ls = "-"
        rect = Rectangle(
            (rx, ry), rw, rh,
            edgecolor=edge, facecolor=face,
            linestyle=ls, linewidth=1.4, alpha=0.10, fill=True,
            zorder=12,
        )
        # Re-apply edge alpha distinctly from face alpha so the outline
        # stays legible even on busy canvases.
        rect.set_edgecolor(edge)
        rect.set_linewidth(1.4)
        self.ax.add_patch(rect)

    def _draw_model(self) -> None:
        model = self._model()
        # Pre-compute the largest force/UDL/point-load magnitude in the
        # model so every drawn arrow length is proportional to the
        # actual load magnitude relative to the rest of the model. We
        # cap it at the model span so a runaway 1e9 load doesn't fill
        # the screen.
        max_force = 0.0
        max_udl = 0.0
        max_point = 0.0
        for ld in model.nodal_loads:
            max_force = max(max_force, (ld.fx ** 2 + ld.fy ** 2) ** 0.5)
        for elem in model.elements:
            for ml in getattr(elem, "member_loads", []):
                if isinstance(ml, UniformDistributedLoad):
                    max_udl = max(
                        max_udl, abs(ml.wy), abs(getattr(ml, "wx", 0.0)),
                    )
                elif isinstance(ml, PointLoad):
                    max_point = max(
                        max_point, abs(ml.py), abs(getattr(ml, "px", 0.0)),
                    )
        span = self._model_span()
        # Map the largest load in each family to ~12% of the model
        # span on screen; smaller loads scale linearly down from there.
        # If no loads of a given family exist, the scale is 0 so the
        # zero-magnitude check inside each draw call falls through.
        target = 0.12 * span
        load_scales = {
            "force": target / max_force if max_force > 0 else 0.0,
            "udl":   target / max_udl   if max_udl   > 0 else 0.0,
            "point": target / max_point if max_point > 0 else 0.0,
        }

        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            color = "#1f77b4" if isinstance(elem, FrameElement2D) else "#d62728"
            ls = "-" if isinstance(elem, FrameElement2D) else "--"
            self.ax.plot([ni.x, nj.x], [ni.y, nj.y], color=color, linestyle=ls,
                         linewidth=2.0, zorder=2)
            mx, my = (ni.x + nj.x) / 2, (ni.y + nj.y) / 2
            self.ax.annotate(f"e{elem.id}", (mx, my), color=color,
                             fontsize=8, ha="center", va="bottom", zorder=4)
            if self.show_section_labels:
                section = model.sections.get(getattr(elem, "section_id", None))
                if section is not None:
                    material = model.materials.get(section.material_id)
                    sec_name = section.name or f"section {section.id}"
                    mat_name = material.name if material and material.name else (
                        f"material {section.material_id}"
                    )
                    self.ax.annotate(
                        f"{sec_name} / {mat_name}", (mx, my),
                        xytext=(0, -14), textcoords="offset points",
                        fontsize=7, ha="center", va="top", color="#555555",
                        bbox=dict(boxstyle="round,pad=0.18",
                                  fc="white", ec="#dddddd", alpha=0.82),
                        zorder=6,
                    )
            if isinstance(elem, FrameElement2D):
                if elem.release_i:
                    self.ax.plot(ni.x + 0.15 * (nj.x - ni.x),
                                 ni.y + 0.15 * (nj.y - ni.y),
                                 marker="o", color="white", markersize=7,
                                 markeredgecolor=color, zorder=5)
                if elem.release_j:
                    self.ax.plot(nj.x - 0.15 * (nj.x - ni.x),
                                 nj.y - 0.15 * (nj.y - ni.y),
                                 marker="o", color="white", markersize=7,
                                 markeredgecolor=color, zorder=5)
            self._draw_member_loads(elem, ni, nj, load_scales)

        for nid, n in model.nodes.items():
            self.ax.plot(n.x, n.y, "o", color="black", markersize=6, zorder=5)
            self.ax.annotate(f"n{nid}", (n.x, n.y), xytext=(5, 5),
                             textcoords="offset points", fontsize=8, zorder=6)
            sup = model.supports.get(nid)
            if sup is not None:
                self._draw_support(sup, n.x, n.y)

        for ld in model.nodal_loads:
            n = model.nodes.get(ld.node_id)
            if n is None:
                continue
            self._draw_nodal_load(ld, n.x, n.y, load_scales["force"])

    def _draw_support(self, sup: Support, x: float, y: float) -> None:
        if sup.ux and sup.uy and sup.rz:
            self.ax.plot(x, y, marker="s", markersize=14, color="black",
                         markerfacecolor="none", zorder=4)
        elif sup.ux and sup.uy:
            self.ax.plot(x, y - 0.15, marker="^", markersize=12, color="black",
                         markerfacecolor="white", zorder=4)
        elif sup.uy and not sup.ux:
            self.ax.plot(x, y - 0.15, marker="o", markersize=10, color="black",
                         markerfacecolor="white", zorder=4)
        elif sup.ux and not sup.uy:
            self.ax.plot(x - 0.15, y, marker="o", markersize=10, color="black",
                         markerfacecolor="white", zorder=4)
        else:
            self.ax.plot(x, y, marker="x", markersize=10, color="black", zorder=4)
        labels = []
        for dof in ("ux", "uy", "rz"):
            v = sup.prescribed(dof)
            if v != 0.0 and getattr(sup, dof):
                labels.append(f"{dof}={v:+.3g}")
        if labels:
            self.ax.annotate(", ".join(labels), (x, y),
                             xytext=(5, -15), textcoords="offset points",
                             fontsize=7, color="#444444", zorder=6)

    def _draw_nodal_load(self, ld: NodalLoad, x: float, y: float,
                          force_scale: float) -> None:
        # ``force_scale`` is "world-units of arrow length per kN" so
        # arrow length is directly proportional to the load magnitude
        # (set by _draw_model from the largest nodal load in the model).
        if force_scale > 0:
            if ld.fx:
                dx = ld.fx * force_scale
                self.ax.annotate(
                    "",
                    xy=(x, y),
                    xytext=(x - dx, y),
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2),
                    zorder=5,
                )
                self.ax.annotate(f"Fx={ld.fx:+.3g}", (x - dx, y),
                                 xytext=(0, 5), textcoords="offset points",
                                 fontsize=7, color="#2ca02c", zorder=6)
            if ld.fy:
                dy = ld.fy * force_scale
                self.ax.annotate(
                    "",
                    xy=(x, y),
                    xytext=(x, y - dy),
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2),
                    zorder=5,
                )
                self.ax.annotate(f"Fy={ld.fy:+.3g}", (x, y - dy),
                                 xytext=(5, 0), textcoords="offset points",
                                 fontsize=7, color="#2ca02c", zorder=6)
        if ld.mz:
            self.ax.annotate(f"M={ld.mz:+.3g}", (x, y), xytext=(8, -8),
                             textcoords="offset points", fontsize=7,
                             color="#2ca02c", zorder=6)

    def _draw_member_loads(self, elem, ni, nj, load_scales: dict) -> None:
        """Draw each member load in its TRUE direction:

        * ``coord_system == "local"`` — axial component along the member
          tangent, transverse component perpendicular to the member.
        * ``coord_system == "global"`` — qX and qY components in the true
          global X / Y directions, regardless of the member's
          orientation.
        * ``coord_system == "gravity"`` — single component straight along
          global -Y (positive magnitude = downward), regardless of
          member orientation.

        Each non-zero (direction, magnitude) pair produces one set of
        arrows (UDL: six along the element; PointLoad: one at ``a``).
        The arrow "tail offset" sign convention matches the legacy
        renderer so visual orientation stays consistent for existing
        local loads."""
        if not elem.member_loads:
            return
        L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
        if L < 1e-12:
            return
        tx, ty = (nj.x - ni.x) / L, (nj.y - ni.y) / L   # member tangent
        nx, ny = -ty, tx                                # member +y_local

        labels: list[str] = []
        udl_scale = load_scales.get("udl", 0.0)
        point_scale = load_scales.get("point", 0.0)

        for ml in elem.member_loads:
            if isinstance(ml, UniformDistributedLoad):
                labels.append(_label_for_udl(ml))
                if udl_scale > 0:
                    for dx, dy, mag in _udl_visual_components(
                        ml, tx, ty, nx, ny,
                    ):
                        if mag == 0.0:
                            continue
                        self._draw_udl_arrow_strip(
                            ni, nj, dx, dy, mag, udl_scale,
                        )
            elif isinstance(ml, PointLoad):
                labels.append(_label_for_pointload(ml))
                if point_scale > 0:
                    a = max(0.0, min(L, float(ml.a)))
                    bx = ni.x + tx * a
                    by = ni.y + ty * a
                    for dx, dy, mag in _pointload_visual_components(
                        ml, tx, ty, nx, ny,
                    ):
                        if mag == 0.0:
                            continue
                        h = mag * point_scale
                        # Tail OPPOSITE to load direction so the
                        # arrowhead at xy=(bx,by) visually points along
                        # (dx, dy) — matches the nodal-load convention.
                        self.ax.annotate(
                            "",
                            xy=(bx, by),
                            xytext=(bx - dx * h, by - dy * h),
                            arrowprops=dict(
                                arrowstyle="->", color="#9467bd", lw=2,
                            ),
                            zorder=5,
                        )
            elif isinstance(ml, TrussTemperatureLoad):
                labels.append(f"ΔT {ml.delta_T:+.3g}°")
            elif isinstance(ml, FrameTemperatureLoad):
                labels.append(f"T {ml.t_top:+.3g}/{ml.t_bottom:+.3g}°")
        if labels:
            mx, my = (ni.x + nj.x) / 2, (ni.y + nj.y) / 2
            self.ax.annotate(", ".join(labels), (mx, my),
                             xytext=(0, -12), textcoords="offset points",
                             fontsize=7, color="#9467bd", ha="center", zorder=6)

    def _draw_udl_arrow_strip(
        self, ni, nj, dx: float, dy: float, magnitude: float,
        udl_scale: float, n_arrows: int = 6,
    ) -> None:
        """Draw ``n_arrows`` evenly spaced arrows along the element in
        the direction ``(dx, dy)`` with length proportional to
        ``magnitude * udl_scale``. The arrowhead lands on the member
        and the tail sits OPPOSITE to the load direction so the visual
        actually points the way the force acts — matching how nodal
        loads are drawn (see ``_draw_nodal_load`` which also offsets
        the tail by ``-`` the force components)."""
        h = magnitude * udl_scale
        if h == 0.0:
            return
        for i in range(n_arrows):
            t = (i + 0.5) / n_arrows
            bx = ni.x + (nj.x - ni.x) * t
            by = ni.y + (nj.y - ni.y) * t
            self.ax.annotate(
                "",
                xy=(bx, by),
                xytext=(bx - dx * h, by - dy * h),
                arrowprops=dict(
                    arrowstyle="->", color="#9467bd", lw=1.0, alpha=0.85,
                ),
                zorder=4,
            )

    def _node_displacement(self, nid: int) -> tuple[float, float]:
        result = self._result
        if result is None or result.D is None:
            return 0.0, 0.0
        emap = result.E_map.get(nid)
        if emap is None:
            return 0.0, 0.0
        D = result.D
        ux = float(D[emap["ux"]]) if emap["ux"] is not None else 0.0
        uy = float(D[emap["uy"]]) if emap["uy"] is not None else 0.0
        return ux, uy

    def _node_rotation(self, nid: int) -> float:
        """Return θ_z at ``nid`` from the global displacement vector, or
        0.0 if the node has no rotational DOF (e.g. truss-only node)."""
        result = self._result
        if result is None or result.D is None:
            return 0.0
        emap = result.E_map.get(nid)
        if emap is None or emap.get("rz") is None:
            return 0.0
        return float(result.D[emap["rz"]])

    @staticmethod
    def _hermite_v(
        r: float, L: float, vi: float, thi: float, vj: float, thj: float,
    ) -> float:
        """Cubic-Hermite transverse interpolation in local frame.

        ``r`` is the dimensionless position 0..1 along the element. Returns
        the transverse displacement v(r) for end DOFs ``[vi, thi, vj, thj]``.
        """
        N1 = 1.0 - 3.0 * r * r + 2.0 * r * r * r
        N2 = L * (r - 2.0 * r * r + r * r * r)
        N3 = 3.0 * r * r - 2.0 * r * r * r
        N4 = L * (-r * r + r * r * r)
        return N1 * vi + N2 * thi + N3 * vj + N4 * thj

    def _frame_deformed_points(
        self, elem: FrameElement2D, scale: float,
    ) -> tuple[list[float], list[float]]:
        """Sample the frame element's deformed centreline.

        Reads ``d_local`` from ``member_results`` — already in the element's
        local frame and, for moment-released ends, back-calculated via static
        condensation — so the Hermite curve is correct even at hinge joints.
        Pure visualization; solver outputs are not modified.
        """
        nodes = self._model().nodes
        ni = nodes[elem.node_i]
        nj = nodes[elem.node_j]
        L, c, s = elem.length_cos_sin(nodes)
        result = self._result
        mr = result.member_results.get(elem.id) if result else None
        if mr is None or "d_local" not in mr:
            return [ni.x, nj.x], [ni.y, nj.y]
        d = mr["d_local"]
        ui_loc, vi_loc, thi = float(d[0]), float(d[1]), float(d[2])
        uj_loc, vj_loc, thj = float(d[3]), float(d[4]), float(d[5])
        n = max(2, int(self.deformed_stations))
        Xs: list[float] = []
        Ys: list[float] = []
        for k in range(n):
            r = k / (n - 1)
            u_loc = (1.0 - r) * ui_loc + r * uj_loc
            v_loc = self._hermite_v(r, L, vi_loc, thi, vj_loc, thj)
            x_def = r * L + scale * u_loc
            y_def = scale * v_loc
            Xs.append(ni.x + c * x_def - s * y_def)
            Ys.append(ni.y + s * x_def + c * y_def)
        return Xs, Ys

    def _draw_deformed(self) -> None:
        result = self._result
        model = self._model()
        if result.D is None:
            return
        span = self._model_span()
        max_disp = max(
            (abs(self._node_displacement(nid)[0]) for nid in model.nodes),
            default=0.0,
        )
        max_disp = max(
            max_disp,
            max((abs(self._node_displacement(nid)[1]) for nid in model.nodes), default=0.0),
        )
        # Extend max_disp to include Hermite transverse amplitudes for frame
        # elements. Without this, a horizontal SS beam (all nodal ux=uy=0)
        # would have max_disp=0 and the deformed shape would be silenced.
        for elem in model.elements:
            if not isinstance(elem, FrameElement2D):
                continue
            mr = result.member_results.get(elem.id)
            if mr is None or "d_local" not in mr:
                continue
            d = mr["d_local"]
            try:
                L, _c, _s = elem.length_cos_sin(model.nodes)
            except (ValueError, ZeroDivisionError):
                continue
            vi, thi, vj, thj = float(d[1]), float(d[2]), float(d[4]), float(d[5])
            for rk in (0.25, 0.5, 0.75):
                max_disp = max(max_disp, abs(self._hermite_v(rk, L, vi, thi, vj, thj)))
        if max_disp <= 0:
            return
        scale = self.deformed_scale * 0.10 * span / max_disp
        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            if isinstance(elem, FrameElement2D):
                try:
                    Xs, Ys = self._frame_deformed_points(elem, scale)
                except (ValueError, ZeroDivisionError):
                    continue
            else:
                # Truss bar stays straight between displaced endpoints —
                # rotations don't contribute to a pin-jointed member.
                uxi, uyi = self._node_displacement(elem.node_i)
                uxj, uyj = self._node_displacement(elem.node_j)
                Xs = [ni.x + scale * uxi, nj.x + scale * uxj]
                Ys = [ni.y + scale * uyi, nj.y + scale * uyj]
            self.ax.plot(
                Xs, Ys,
                color="#ff7f0e", linestyle="-", linewidth=1.5, alpha=0.7,
                zorder=3,
            )
        self.ax.annotate(
            f"deformed × {scale:.2g} (max |u|={max_disp:.3e} m)",
            (0.02, 0.98), xycoords="axes fraction",
            fontsize=8, color="#ff7f0e", va="top",
        )

    def _mode_displacement(self, node_id: int) -> tuple[float, float]:
        """Return (ux, uy) for ``node_id`` in the currently-shown mode."""
        mr = self._modal_result
        if mr is None or mr.dofs is None:
            return 0.0, 0.0
        emap = mr.dofs.active_map.get(node_id)
        if emap is None:
            return 0.0, 0.0
        col = self._modal_mode_idx
        if col < 0 or col >= mr.modes.shape[1]:
            return 0.0, 0.0
        phi = mr.modes[:, col]
        ux = float(phi[emap["ux"]]) if emap["ux"] is not None else 0.0
        uy = float(phi[emap["uy"]]) if emap["uy"] is not None else 0.0
        return ux, uy

    def _draw_mode_shape(self) -> None:
        mr = self._modal_result
        model = self._model()
        k = self._modal_mode_idx
        if mr is None or k < 0 or k >= mr.n_modes:
            return
        span = self._model_span()
        max_disp = 0.0
        for nid in model.nodes:
            ux, uy = self._mode_displacement(nid)
            max_disp = max(max_disp, abs(ux), abs(uy))
        if max_disp <= 0.0:
            return
        # Auto-scale to a tenth of the model span, multiplied by the
        # user-controlled scale slider on the results pane.
        scale = self._modal_scale * 0.10 * span / max_disp
        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            uxi, uyi = self._mode_displacement(elem.node_i)
            uxj, uyj = self._mode_displacement(elem.node_j)
            self.ax.plot(
                [ni.x, nj.x], [ni.y, nj.y],
                color="#888888", linestyle=":", linewidth=1.0, alpha=0.6,
                zorder=2,
            )
            self.ax.plot(
                [ni.x + scale * uxi, nj.x + scale * uxj],
                [ni.y + scale * uyi, nj.y + scale * uyj],
                color="#d62728", linestyle="-", linewidth=1.8, alpha=0.85,
                zorder=4,
            )
        f = float(mr.frequencies[k])
        T = float(mr.periods[k])
        self.ax.annotate(
            f"mode {k + 1} · f = {f:.4g} Hz · T = {T:.4g} s · scale × {scale:.2g}",
            (0.02, 0.98), xycoords="axes fraction",
            fontsize=8, color="#d62728", va="top",
        )

    def _draw_reactions(self) -> None:
        result = self._result
        model = self._model()
        if not result.reactions:
            return
        span = self._model_span()
        max_r = 0.0
        for r in result.reactions.values():
            max_r = max(max_r,
                        ((r.get("ux", 0)) ** 2 + (r.get("uy", 0)) ** 2) ** 0.5)
        if max_r <= 0:
            return
        arrow_len = 0.10 * span / max_r
        for nid, r in result.reactions.items():
            n = model.nodes.get(nid)
            if n is None:
                continue
            rx = r.get("ux", 0.0)
            ry = r.get("uy", 0.0)
            if rx or ry:
                self.ax.annotate(
                    "",
                    xy=(n.x, n.y),
                    xytext=(n.x - rx * arrow_len, n.y - ry * arrow_len),
                    arrowprops=dict(arrowstyle="->", color="#9467bd", lw=2),
                    zorder=5,
                )
                mag = (rx ** 2 + ry ** 2) ** 0.5
                self.ax.annotate(
                    f"R={mag:.3g} kN",
                    (n.x - rx * arrow_len, n.y - ry * arrow_len),
                    fontsize=7, color="#9467bd", zorder=6,
                )
            mz = r.get("rz", 0.0)
            if mz:
                self.ax.annotate(
                    f"M={mz:+.3g}", (n.x, n.y),
                    xytext=(-10, -15), textcoords="offset points",
                    fontsize=7, color="#9467bd", zorder=6,
                )

    def _draw_diagrams(self) -> None:
        # _diagram_critical_points is consumed by _hit_test → the snap
        # engine, so callers can snap the cursor onto the labelled
        # max/min points. Always reset it before drawing so a stale
        # set from a previous result kind doesn't survive.
        self._diagram_critical_points = []
        result = self._result
        model = self._model()
        if not result.member_results:
            return
        span = self._model_span()
        max_ord = 0.0
        per_elem = []
        for elem in model.elements:
            mr = result.member_results.get(elem.id)
            if mr is None:
                continue
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            # Sample density is configurable via View → Diagram stations.
            # Lower counts give a coarser preview; the critical-point
            # search below operates on the same xs / ys, so very low
            # counts (e.g. 5) may miss the true peak between stations —
            # surfaced to the user via the menu tooltip + status hint.
            n = max(2, int(self.diagram_stations))
            xs, ys = _diagram_ordinates(
                elem, ni, nj, mr["f_local"], self.diagram_kind, n_samples=n,
            )
            if xs is None:
                continue
            per_elem.append((elem, ni, nj, xs, ys))
            max_ord = max(max_ord, max(abs(v) for v in ys))
        if max_ord <= 0 or not per_elem:
            return
        scale = self.diagram_scale * 0.12 * span / max_ord
        color = {"moment": "#17becf", "shear": "#bcbd22", "axial": "#8c564b"}[self.diagram_kind]
        unit = _DIAGRAM_UNITS[self.diagram_kind]

        for elem, ni, nj, xs, ys in per_elem:
            L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
            if L < 1e-12:
                continue
            cx, cy = (nj.x - ni.x) / L, (nj.y - ni.y) / L
            nx, ny = -cy, cx
            poly_x = [ni.x]
            poly_y = [ni.y]
            for xx, yy in zip(xs, ys):
                px = ni.x + xx * cx + scale * yy * nx
                py = ni.y + xx * cy + scale * yy * ny
                poly_x.append(px)
                poly_y.append(py)
            poly_x.append(nj.x)
            poly_y.append(nj.y)
            self.ax.fill(poly_x, poly_y, color=color, alpha=0.25, zorder=1)
            self.ax.plot(poly_x, poly_y, color=color, linewidth=1.0, zorder=2)

            # Per-element critical points: argmax(ys) and argmin(ys).
            # If the diagram is constant (axial), max == min — label
            # only once at midspan. Skip near-zero peaks (clutter).
            threshold = 0.05 * max_ord
            i_max = max(range(len(ys)), key=lambda i: ys[i])
            i_min = min(range(len(ys)), key=lambda i: ys[i])
            picks: list[int] = []
            if abs(ys[i_max] - ys[i_min]) < 1e-12 * max(max_ord, 1.0):
                # Constant diagram — label once near the midspan.
                picks = [len(xs) // 2]
            else:
                if abs(ys[i_max]) > threshold:
                    picks.append(i_max)
                if abs(ys[i_min]) > threshold and i_min != i_max:
                    picks.append(i_min)
            for i in picks:
                xx = xs[i]
                yy = ys[i]
                # World-coords on the diagram polyline at this sample.
                world_x = ni.x + xx * cx + scale * yy * nx
                world_y = ni.y + xx * cy + scale * yy * ny
                # World-coords of the matching point on the element
                # axis itself — that's what the snap engine should
                # snap to (so a left-click in modelling tools still
                # lands on the member geometry, not on the offset
                # diagram polyline).
                axis_x = ni.x + xx * cx
                axis_y = ni.y + xx * cy
                # Marker on the diagram outline + annotation.
                self.ax.plot(world_x, world_y, marker="o", color=color,
                             markersize=6, markeredgecolor="#222",
                             markeredgewidth=0.8, zorder=4)
                self.ax.annotate(
                    f"{yy:+.3g} {unit}",
                    (world_x, world_y),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color="#222", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="#ffffffaa", ec=color, lw=0.5),
                )
                # Record a snap target at the element-axis position
                # (not the offset polyline) so cursor snap places a
                # cross/arrow exactly on the member.
                self._diagram_critical_points.append({
                    "x": axis_x,
                    "y": axis_y,
                    "value": yy,
                    "unit": unit,
                    "kind": self.diagram_kind,
                    "elem_id": elem.id,
                    "x_loc": xx,
                    "L": L,
                })

        self.ax.annotate(
            f"{self.diagram_kind} diagram × {scale:.2g}  ·  max |·| = {max_ord:.3g} {unit}",
            (0.02, 0.94), xycoords="axes fraction",
            fontsize=8, color=color, va="top",
        )

    def _model_span(self) -> float:
        model = self._model()
        if not model.nodes:
            return 1.0
        xs = [n.x for n in model.nodes.values()]
        ys = [n.y for n in model.nodes.values()]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        return span if span > 1e-9 else 1.0

    def _set_axes_limits(self) -> None:
        """Fit xlim/ylim to the model + grid, then stretch one axis so
        the data box's aspect matches the canvas widget's pixel aspect.

        Stretching is necessary because ``set_aspect("equal",
        adjustable="box")`` shrinks the axes rectangle to whichever
        dimension is shorter at a 1:1 data scale. If we fed it a
        square data box the user would see the grid as a small square
        floating in the centre of a wide window. By computing xlim/ylim
        whose ratio already matches the widget's pixel ratio, the axes
        rectangle ends up filling the widget with the gridlines and
        origin markers running edge to edge.
        """
        model = self._model()
        grid = self._grid_provider()
        xs: list[float] = [n.x for n in model.nodes.values()]
        ys: list[float] = [n.y for n in model.nodes.values()]
        xs += [ln.coord for ln in grid.x_lines]
        ys += [ln.coord for ln in grid.y_lines]
        if xs and ys:
            x_lo, x_hi = min(xs), max(xs)
            y_lo, y_hi = min(ys), max(ys)
            cx = (x_lo + x_hi) / 2.0
            cy = (y_lo + y_hi) / 2.0
            base = max(x_hi - x_lo, y_hi - y_lo, 1.0) / 2.0 * 1.15
        else:
            cx, cy, base = 5.0, 5.0, 6.0

        size = self._mpl_canvas.size()
        w_px = max(size.width(),  1)
        h_px = max(size.height(), 1)
        if w_px >= h_px:
            x_half = base * (w_px / h_px)
            y_half = base
        else:
            x_half = base
            y_half = base * (h_px / w_px)
        self._setting_axes_limits = True
        try:
            self.ax.set_xlim(cx - x_half, cx + x_half)
            self.ax.set_ylim(cy - y_half, cy + y_half)
        finally:
            self._setting_axes_limits = False

    def _on_limits_changed(self, _ax) -> None:
        """Mark the view as user-owned when xlim/ylim change for any
        reason other than our own programmatic fits. The matplotlib
        navigation toolbar's pan/zoom modes route through here, so
        after the user pans/zooms via the toolbar a subsequent resize
        no longer silently re-fits and throws their view away."""
        if not self._setting_axes_limits:
            self._user_view_dirty = True


def _udl_visual_components(
    ml: UniformDistributedLoad, tx: float, ty: float, nx: float, ny: float,
) -> list[tuple[float, float, float]]:
    """Return ``(direction_x, direction_y, magnitude)`` tuples for each
    drawable component of a UDL, given the element's tangent ``(tx, ty)``
    and +y_local normal ``(nx, ny)``.

    * ``"local"``: two components — axial along the member tangent
      (magnitude ``wx``) and transverse perpendicular to it (magnitude
      ``wy``). Either component may be 0; the caller filters those out.
    * ``"global"``: two components — ``qX`` along true global X
      ``(1, 0)`` and ``qY`` along true global Y ``(0, 1)``. The
      direction vectors are in WORLD axes, independent of the member's
      orientation, so an inclined member draws horizontally / vertically
      not perpendicular to itself.
    * ``"gravity"``: one component straight along global ``-Y`` with
      magnitude ``wy``. Positive magnitude → arrows pointing down."""
    cs = getattr(ml, "coord_system", "local")
    if cs == "local":
        return [(tx, ty, ml.wx), (nx, ny, ml.wy)]
    if cs == "global":
        return [(1.0, 0.0, ml.wx), (0.0, 1.0, ml.wy)]
    if cs == "gravity":
        return [(0.0, -1.0, ml.wy)]
    # Defensive: unknown — fall back to legacy local-y rendering.
    return [(nx, ny, ml.wy)]


def _pointload_visual_components(
    ml: PointLoad, tx: float, ty: float, nx: float, ny: float,
) -> list[tuple[float, float, float]]:
    """Mirror :func:`_udl_visual_components` for a point load."""
    cs = getattr(ml, "coord_system", "local")
    if cs == "local":
        return [(tx, ty, ml.px), (nx, ny, ml.py)]
    if cs == "global":
        return [(1.0, 0.0, ml.px), (0.0, 1.0, ml.py)]
    if cs == "gravity":
        return [(0.0, -1.0, ml.py)]
    return [(nx, ny, ml.py)]


def _label_for_udl(ml: UniformDistributedLoad) -> str:
    """Short magnitude-and-direction tag rendered under the element."""
    cs = getattr(ml, "coord_system", "local")
    if cs == "gravity":
        return f"UDL {ml.wy:+.3g} grav"
    if cs == "global":
        if ml.wx != 0.0 and ml.wy != 0.0:
            return f"UDL ({ml.wx:+.3g},{ml.wy:+.3g}) glob"
        if ml.wx != 0.0:
            return f"UDL qX={ml.wx:+.3g} glob"
        return f"UDL qY={ml.wy:+.3g} glob"
    # local
    if ml.wx != 0.0 and ml.wy != 0.0:
        return f"UDL ({ml.wx:+.3g},{ml.wy:+.3g})"
    if ml.wx != 0.0:
        return f"UDL wx={ml.wx:+.3g}"
    return f"UDL {ml.wy:+.3g}"


def _label_for_pointload(ml: PointLoad) -> str:
    cs = getattr(ml, "coord_system", "local")
    suffix = ""
    if cs == "gravity":
        return f"P {ml.py:+.3g}@{ml.a:.3g} grav"
    if cs == "global":
        suffix = " glob"
    if ml.px != 0.0 and ml.py != 0.0:
        return f"P ({ml.px:+.3g},{ml.py:+.3g})@{ml.a:.3g}{suffix}"
    if ml.px != 0.0:
        return f"P px={ml.px:+.3g}@{ml.a:.3g}{suffix}"
    return f"P {ml.py:+.3g}@{ml.a:.3g}{suffix}"


def _shift_pressed() -> bool:
    """True if the Shift modifier is currently held.

    Read at click/motion/release time so tools see the live keyboard
    state — matplotlib's `event.key` is unreliable for modifier-only
    presses across backends/OSes."""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
    except Exception:
        return False
    app = QApplication.instance()
    if app is None:
        return False
    return bool(app.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)


def _point_in_world_rect(
    px: float, py: float,
    rx0: float, ry0: float, rx1: float, ry1: float,
) -> bool:
    """True iff (px, py) lies inside or on the boundary of the axis-
    aligned rectangle with corners (rx0, ry0)–(rx1, ry1).

    Corner order is not assumed — the rect is normalised here so the
    caller can pass world coords in any order (e.g. press point and
    release point of a right-to-left drag)."""
    lo_x = min(rx0, rx1)
    hi_x = max(rx0, rx1)
    lo_y = min(ry0, ry1)
    hi_y = max(ry0, ry1)
    return lo_x <= px <= hi_x and lo_y <= py <= hi_y


def _segment_intersects_rect(
    x1: float, y1: float, x2: float, y2: float,
    rx0: float, ry0: float, rx1: float, ry1: float,
) -> bool:
    """True iff the segment (x1,y1)→(x2,y2) intersects or lies inside the
    rect. Uses Cohen–Sutherland outcodes — both endpoints inside, any
    endpoint inside, or a clipped segment with positive length all count
    as a hit. Inclusive on the boundary."""
    lo_x = min(rx0, rx1)
    hi_x = max(rx0, rx1)
    lo_y = min(ry0, ry1)
    hi_y = max(ry0, ry1)

    LEFT, RIGHT, BOTTOM, TOP = 1, 2, 4, 8

    def code(x: float, y: float) -> int:
        c = 0
        if x < lo_x:
            c |= LEFT
        elif x > hi_x:
            c |= RIGHT
        if y < lo_y:
            c |= BOTTOM
        elif y > hi_y:
            c |= TOP
        return c

    cx1, cy1 = x1, y1
    cx2, cy2 = x2, y2
    c1 = code(cx1, cy1)
    c2 = code(cx2, cy2)
    while True:
        if c1 == 0 or c2 == 0:
            return True            # at least one endpoint in rect
        if c1 & c2:
            return False           # both share an outside region
        out = c1 or c2
        nx, ny = cx1, cy1
        dx = cx2 - cx1
        dy = cy2 - cy1
        if out & TOP:
            nx = cx1 + dx * (hi_y - cy1) / dy if dy else cx1
            ny = hi_y
        elif out & BOTTOM:
            nx = cx1 + dx * (lo_y - cy1) / dy if dy else cx1
            ny = lo_y
        elif out & RIGHT:
            ny = cy1 + dy * (hi_x - cx1) / dx if dx else cy1
            nx = hi_x
        elif out & LEFT:
            ny = cy1 + dy * (lo_x - cx1) / dx if dx else cy1
            nx = lo_x
        if out == c1:
            cx1, cy1 = nx, ny
            c1 = code(cx1, cy1)
        else:
            cx2, cy2 = nx, ny
            c2 = code(cx2, cy2)


def _point_segment_distance_px(px, py, x1, y1, x2, y2,
                                px_per_dx, px_per_dy) -> float:
    ax = x1 * px_per_dx
    ay = y1 * px_per_dy
    bx = x2 * px_per_dx
    by = y2 * px_per_dy
    qx = px * px_per_dx
    qy = py * px_per_dy
    abx, aby = bx - ax, by - ay
    aqx, aqy = qx - ax, qy - ay
    seg_len_sq = abx * abx + aby * aby
    if seg_len_sq < 1e-9:
        return ((qx - ax) ** 2 + (qy - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, (aqx * abx + aqy * aby) / seg_len_sq))
    cx = ax + t * abx
    cy = ay + t * aby
    return ((qx - cx) ** 2 + (qy - cy) ** 2) ** 0.5


_DIAGRAM_UNITS = {"moment": "kN·m", "shear": "kN", "axial": "kN"}
