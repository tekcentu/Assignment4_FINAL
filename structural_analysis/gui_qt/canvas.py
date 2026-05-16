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
        self.diagram_kind: str = "moment"
        self.deformed_scale: float = 1.0
        self.diagram_scale: float = 1.0
        self._result = None
        self._modal_result = None    # ModalResult or None
        self._modal_mode_idx: int = 0
        self._modal_scale: float = 1.0
        self._snap_marker = None  # current SnapCandidate
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

        self.on_click: Callable[[HitResult, str], None] | None = None
        self.on_motion: Callable[[HitResult], None] | None = None

        self.fig = plt.Figure(figsize=(7.5, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        # adjustable="box" lets the user's xlim/ylim be honored exactly
        # (matplotlib resizes the axes rectangle to keep aspect=1).
        # The legacy "datalim" mode would silently rewrite our limits
        # and emit "Ignoring fixed y limits…" on every redraw.
        self.ax.set_aspect("equal", adjustable="box")

        self._mpl_canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self._mpl_canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self._mpl_canvas)
        self.setLayout(layout)

        self._mpl_canvas.mpl_connect("button_press_event", self._handle_click)
        self._mpl_canvas.mpl_connect("motion_notify_event", self._handle_motion)

    # ── public API ──

    def set_grid_spacing(self, spacing: float) -> None:
        if spacing <= 0:
            raise ValueError("Grid spacing must be > 0.")
        self.grid_spacing = float(spacing)
        self.redraw()

    def toggle_snap(self, enabled: bool) -> None:
        self.snap_enabled = bool(enabled)

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

        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")

        if saved_xlim is not None:
            self.ax.set_xlim(saved_xlim)
            self.ax.set_ylim(saved_ylim)
        else:
            self._set_axes_limits()
            self._view_initialised = True

        self._draw_grid()
        self._draw_model()
        if self._result is not None and self._result.status == "ok":
            if self.show_deformed:
                self._draw_deformed()
            if self.show_reactions:
                self._draw_reactions()
            if self.show_diagrams:
                self._draw_diagrams()
        elif self._modal_result is not None and self._modal_result.status == "ok":
            self._draw_mode_shape()
        self._draw_snap_marker()
        self._mpl_canvas.draw_idle()

    # ── event forwarding ──

    def _handle_click(self, event) -> None:
        if self.on_click is None or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.toolbar.mode:
            return  # pan/zoom mode active
        hit = self._hit_test(event)
        button_name = {1: "left", 2: "middle", 3: "right"}.get(event.button, "left")
        self.on_click(hit, button_name)

    def _handle_motion(self, event) -> None:
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
        try:
            self.on_motion(hit)
        except Exception:
            pass

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
            )
        self._snap_marker = candidate

        if candidate is not None:
            hit = HitResult(x=candidate.x, y=candidate.y,
                            snap_kind=candidate.kind,
                            snap_label=candidate.label)
            if candidate.kind == "node":
                hit.node_id = candidate.object_id
            elif candidate.kind in ("endpoint", "midpoint", "project"):
                hit.element_id = candidate.object_id
            return hit

        # No snap → fall back to rectangular-grid snapping + element pick.
        sx, sy = self._snap(event.xdata, event.ydata)
        hit = HitResult(x=sx, y=sy)
        # Still try to pick a nearby element (for select/right-click on a line).
        best_eid = None
        best_dpx = self.ELEM_PICK_RADIUS_PX
        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            dpx = _point_segment_distance_px(
                event.xdata, event.ydata, ni.x, ni.y, nj.x, nj.y,
                px_per_dx, px_per_dy,
            )
            if dpx < best_dpx:
                best_dpx = dpx
                best_eid = elem.id
        hit.element_id = best_eid
        return hit

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

    def _draw_snap_marker(self) -> None:
        c = self._snap_marker
        if c is not None:
            marker_styles = {
                "node":     ("o", "#ff7f0e"),  # filled circle, orange
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

    def _draw_model(self) -> None:
        model = self._model()
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
            self._draw_member_loads(elem, ni, nj)

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
            self._draw_nodal_load(ld, n.x, n.y)

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

    def _draw_nodal_load(self, ld: NodalLoad, x: float, y: float) -> None:
        if ld.fx or ld.fy:
            mag = (ld.fx ** 2 + ld.fy ** 2) ** 0.5
            if mag > 0:
                scale = 0.5
                dx = ld.fx / mag * scale
                dy = ld.fy / mag * scale
                self.ax.annotate(
                    "",
                    xy=(x, y),
                    xytext=(x - dx, y - dy),
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2),
                    zorder=5,
                )
                self.ax.annotate(f"{mag:.3g} kN", (x - dx, y - dy),
                                 fontsize=7, color="#2ca02c", zorder=6)
        if ld.mz:
            self.ax.annotate(f"M={ld.mz:+.3g}", (x, y), xytext=(8, -8),
                             textcoords="offset points", fontsize=7,
                             color="#2ca02c", zorder=6)

    def _draw_member_loads(self, elem, ni, nj) -> None:
        if not elem.member_loads:
            return
        labels = []
        for ml in elem.member_loads:
            if isinstance(ml, UniformDistributedLoad):
                labels.append(f"UDL {ml.wy:+.3g}")
            elif isinstance(ml, PointLoad):
                labels.append(f"P {ml.py:+.3g}@{ml.a:.3g}")
            elif isinstance(ml, TrussTemperatureLoad):
                labels.append(f"ΔT {ml.delta_T:+.3g}°")
            elif isinstance(ml, FrameTemperatureLoad):
                labels.append(f"T {ml.t_top:+.3g}/{ml.t_bottom:+.3g}°")
        if labels:
            mx, my = (ni.x + nj.x) / 2, (ni.y + nj.y) / 2
            self.ax.annotate(", ".join(labels), (mx, my),
                             xytext=(0, -12), textcoords="offset points",
                             fontsize=7, color="#9467bd", ha="center", zorder=6)

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
        if max_disp <= 0:
            return
        scale = self.deformed_scale * 0.10 * span / max_disp
        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            uxi, uyi = self._node_displacement(elem.node_i)
            uxj, uyj = self._node_displacement(elem.node_j)
            self.ax.plot(
                [ni.x + scale * uxi, nj.x + scale * uxj],
                [ni.y + scale * uyi, nj.y + scale * uyj],
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
            xs, ys = _diagram_ordinates(elem, ni, nj, mr["f_local"], self.diagram_kind)
            if xs is None:
                continue
            per_elem.append((elem, ni, nj, xs, ys))
            max_ord = max(max_ord, max(abs(v) for v in ys))
        if max_ord <= 0 or not per_elem:
            return
        scale = self.diagram_scale * 0.12 * span / max_ord
        color = {"moment": "#17becf", "shear": "#bcbd22", "axial": "#8c564b"}[self.diagram_kind]
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
        self.ax.annotate(
            f"{self.diagram_kind} diagram × {scale:.2g}",
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
        model = self._model()
        grid = self._grid_provider()
        xs: list[float] = [n.x for n in model.nodes.values()]
        ys: list[float] = [n.y for n in model.nodes.values()]
        xs += [ln.coord for ln in grid.x_lines]
        ys += [ln.coord for ln in grid.y_lines]
        if not xs or not ys:
            self.ax.set_xlim(-1, 11)
            self.ax.set_ylim(-1, 11)
            return
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        pad = 0.15 * span
        self.ax.set_xlim(min(xs) - pad, max(xs) + pad)
        self.ax.set_ylim(min(ys) - pad, max(ys) + pad)


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


def _diagram_ordinates(elem, ni, nj, f_local, kind: str,
                        n_samples: int = 21):
    L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
    if L < 1e-12:
        return None, None
    N_i, V_i, M_i, _N_j, _V_j, _M_j = (float(v) for v in f_local)
    udls = []
    points = []
    for ml in getattr(elem, "member_loads", []):
        if isinstance(ml, UniformDistributedLoad):
            udls.append(ml.wy)
        elif isinstance(ml, PointLoad):
            points.append((ml.a, ml.py))
    xs = [i * L / (n_samples - 1) for i in range(n_samples)]
    if kind == "axial":
        # Axial: plot ``-N_i`` so compression on the element reads positive
        # on the page (the local-frame member-end axial force is positive in
        # the +x_local direction, which is tension on the i-end; flipping
        # the sign gives the conventional compression-positive diagram).
        ys = [-N_i for _ in xs]
        return xs, ys
    if isinstance(elem, TrussElement2D):
        return None, None
    w = sum(udls)

    # Shear and moment from the left-of-cut free body. ``V_i`` and ``M_i``
    # are the local member-end shear/moment at the i-end (see
    # FrameElement2D.local_displacement_and_end_forces — they are the
    # entries of ``q_local = K·d − p_local``). ``w`` is the summed UDL
    # intensity in the element's +y_local direction (so ``wy < 0`` is a
    # downward load on a horizontal beam), and ``points = [(a, py)]`` are
    # in-span point loads with ``py`` in +y_local.
    #
    # The point-load contributions in ``shear`` and ``moment`` carry the
    # **same** sign of ``py`` so the differential identity ``dM/dx = V``
    # holds across the in-span discontinuity — see
    # ``tests/test_diagram_signs.py`` for the regression.
    def shear(x):
        v = V_i - w * x
        for a, py in points:
            if x > a:
                v += py
        return v

    def moment(x):
        m = -M_i + V_i * x - 0.5 * w * x * x
        for a, py in points:
            if x > a:
                m += py * (x - a)
        return m

    if kind == "shear":
        ys = [shear(x) for x in xs]
    else:
        ys = [moment(x) for x in xs]
    return xs, ys
