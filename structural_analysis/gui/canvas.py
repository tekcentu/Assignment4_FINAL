"""Matplotlib drawing canvas embedded in Tk.

Renders the StructuralModel and forwards mouse events (with grid-snapped
data coordinates and hit-test results) to the active controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib
matplotlib.use("TkAgg")  # noqa: E402  must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)

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


@dataclass
class HitResult:
    """What was under the mouse at the time of an event."""
    x: float            # data coords, snapped to grid
    y: float            # data coords, snapped to grid
    node_id: Optional[int] = None
    element_id: Optional[int] = None


class ModelCanvas:
    """Matplotlib canvas embedded in a Tk parent.

    The canvas owns its Figure / Axes; the host is responsible for
    repacking and lifecycle. Events are forwarded to ``on_click`` and
    ``on_motion`` callbacks supplied by the controller.
    """

    NODE_PICK_RADIUS_PX = 12
    ELEM_PICK_RADIUS_PX = 8

    def __init__(self, parent, model_provider: Callable[[], StructuralModel]):
        self.parent = parent
        self._model = model_provider
        self.grid_spacing: float = 0.5
        self.snap_enabled: bool = True

        self.show_deformed: bool = True
        self.show_reactions: bool = True
        self.show_diagrams: bool = False
        self.diagram_kind: str = "moment"  # "moment" | "shear" | "axial"
        self.deformed_scale: float = 1.0
        self.diagram_scale: float = 1.0
        self._result = None  # AnalysisResult or None

        self.on_click: Callable[[HitResult, str], None] | None = None
        self.on_motion: Callable[[HitResult], None] | None = None

        self.fig = plt.Figure(figsize=(7.5, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True, which="both", linestyle=":", linewidth=0.5,
                     color="#bbbbbb")

        self.tk_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.widget = self.tk_canvas.get_tk_widget()
        self.toolbar = NavigationToolbar2Tk(self.tk_canvas, parent, pack_toolbar=False)
        self.toolbar.update()

        self.tk_canvas.mpl_connect("button_press_event", self._handle_click)
        self.tk_canvas.mpl_connect("motion_notify_event", self._handle_motion)

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
        self.redraw()

    def clear_result(self) -> None:
        self._result = None
        self.redraw()

    def redraw(self) -> None:
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="datalim")
        self._draw_grid()
        self._draw_model()
        if self._result is not None and self._result.status == "ok":
            if self.show_deformed:
                self._draw_deformed()
            if self.show_reactions:
                self._draw_reactions()
            if self.show_diagrams:
                self._draw_diagrams()
        self._set_axes_limits()
        self.tk_canvas.draw_idle()

    # ── event forwarding ──

    def _handle_click(self, event) -> None:
        if self.on_click is None:
            return
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.toolbar.mode:
            # User is in zoom/pan mode — let mpl handle it.
            return
        hit = self._hit_test(event)
        button_name = {1: "left", 2: "middle", 3: "right"}.get(event.button, "left")
        try:
            self.on_click(hit, button_name)
        except Exception:
            raise  # let app's exception boundary log it

    def _handle_motion(self, event) -> None:
        if self.on_motion is None:
            return
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        hit = self._hit_test(event)
        try:
            self.on_motion(hit)
        except Exception:
            pass  # never let a tooltip raise

    # ── geometry ──

    def _snap(self, x: float, y: float) -> tuple[float, float]:
        if not self.snap_enabled:
            return x, y
        s = self.grid_spacing
        return round(x / s) * s, round(y / s) * s

    def _hit_test(self, event) -> HitResult:
        sx, sy = self._snap(event.xdata, event.ydata)
        hit = HitResult(x=sx, y=sy)
        model = self._model()

        # Convert pixel radius to data coords using axes limits.
        bbox = self.ax.bbox
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        px_per_dx = bbox.width / max(x1 - x0, 1e-9)
        px_per_dy = bbox.height / max(y1 - y0, 1e-9)

        # Nearest node
        best_nid = None
        best_dpx = self.NODE_PICK_RADIUS_PX
        for nid, n in model.nodes.items():
            dpx = ((event.xdata - n.x) * px_per_dx) ** 2 + \
                  ((event.ydata - n.y) * px_per_dy) ** 2
            dpx = dpx ** 0.5
            if dpx < best_dpx:
                best_dpx = dpx
                best_nid = nid
        if best_nid is not None:
            hit.node_id = best_nid
            hit.x = model.nodes[best_nid].x
            hit.y = model.nodes[best_nid].y
            return hit

        # Nearest element midpoint distance
        best_eid = None
        best_dpx = self.ELEM_PICK_RADIUS_PX
        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            dpx = _point_segment_distance_px(
                event.xdata, event.ydata,
                ni.x, ni.y, nj.x, nj.y,
                px_per_dx, px_per_dy,
            )
            if dpx < best_dpx:
                best_dpx = dpx
                best_eid = elem.id
        hit.element_id = best_eid
        return hit

    # ── drawing ──

    def _draw_grid(self) -> None:
        self.ax.grid(True, which="major", linestyle=":", linewidth=0.5,
                     color="#cccccc")

    def _draw_model(self) -> None:
        model = self._model()
        # elements first
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

        # nodes
        for nid, n in model.nodes.items():
            self.ax.plot(n.x, n.y, "o", color="black", markersize=6, zorder=5)
            self.ax.annotate(f"n{nid}", (n.x, n.y), xytext=(5, 5),
                             textcoords="offset points", fontsize=8, zorder=6)
            sup = model.supports.get(nid)
            if sup is not None:
                self._draw_support(sup, n.x, n.y)

        # nodal loads
        for ld in model.nodal_loads:
            n = model.nodes.get(ld.node_id)
            if n is None:
                continue
            self._draw_nodal_load(ld, n.x, n.y)

    def _draw_support(self, sup: Support, x: float, y: float) -> None:
        if sup.ux and sup.uy and sup.rz:
            # fully fixed — cross-hatched square
            self.ax.plot(x, y, marker="s", markersize=14, color="black",
                         markerfacecolor="none", zorder=4)
        elif sup.ux and sup.uy:
            # pin — triangle pointing up below the node
            self.ax.plot(x, y - 0.15, marker="^", markersize=12, color="black",
                         markerfacecolor="white", zorder=4)
        elif sup.uy and not sup.ux:
            # roller (free in x) — circle
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

    # ── results overlays ──

    def _node_displacement(self, nid: int) -> tuple[float, float]:
        """Return (ux, uy) for node ``nid`` from the current AnalysisResult, or (0,0)."""
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

        # Pick a sensible auto-scale: target max displacement = 10 % of model span.
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
        # caption
        self.ax.annotate(
            f"deformed × {scale:.2g} (max |u|={max_disp:.3e} m)",
            (0.02, 0.98), xycoords="axes fraction",
            fontsize=8, color="#ff7f0e", va="top",
        )

    def _draw_reactions(self) -> None:
        result = self._result
        model = self._model()
        if not result.reactions:
            return
        span = self._model_span()
        # auto scale: longest arrow = 10 % of span
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

        # pick max ordinate across all elements
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
            # local axis
            L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
            if L < 1e-12:
                continue
            cx, cy = (nj.x - ni.x) / L, (nj.y - ni.y) / L
            # perpendicular (positive moment plotted on local +y, which is (-s, c))
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
        if not model.nodes:
            self.ax.set_xlim(-1, 11)
            self.ax.set_ylim(-1, 11)
            return
        xs = [n.x for n in model.nodes.values()]
        ys = [n.y for n in model.nodes.values()]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        pad = 0.15 * span
        self.ax.set_xlim(min(xs) - pad, max(xs) + pad)
        self.ax.set_ylim(min(ys) - pad, max(ys) + pad)


def _point_segment_distance_px(px, py, x1, y1, x2, y2,
                                px_per_dx, px_per_dy) -> float:
    # work in pixel space
    ax = (x1) * px_per_dx
    ay = (y1) * px_per_dy
    bx = (x2) * px_per_dx
    by = (y2) * px_per_dy
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
    """Return (xs, ys) along the element for moment, shear, or axial.

    ``f_local`` is the 6-vector [N_i, V_i, M_i, N_j, V_j, M_j] from the
    postprocessor. Combined with any UDL / point loads on the element we
    integrate to obtain the full M(x), V(x), N(x).

    Truss elements only carry axial.
    """
    L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
    if L < 1e-12:
        return None, None

    N_i, V_i, M_i, N_j, V_j, M_j = (float(v) for v in f_local)

    # Aggregate loads
    udls = []
    points = []
    for ml in getattr(elem, "member_loads", []):
        if isinstance(ml, UniformDistributedLoad):
            udls.append(ml.wy)
        elif isinstance(ml, PointLoad):
            points.append((ml.a, ml.py))

    xs = [i * L / (n_samples - 1) for i in range(n_samples)]

    if kind == "axial":
        # Axial is constant on a member with no axial member loads.
        ys = [-N_i for _ in xs]  # plot as N(x); with only end forces = -N_i
        return xs, ys

    if isinstance(elem, TrussElement2D):
        return None, None

    w = sum(udls)

    def shear(x):
        v = V_i - w * x
        for a, py in points:
            if x > a:
                v -= py
        return v

    def moment(x):
        # Integrate shear with M_i at x=0 sign convention: dM/dx = V (using
        # postprocessor sign that already produced V_i and M_i).
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
