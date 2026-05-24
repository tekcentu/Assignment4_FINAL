"""Shared graphics helpers for elements.

The N / V / M station evaluators and samplers live here so the main 2D
canvas (full-model diagrams + hover read-out) and the per-element
detail dialog (mini sketch + FBD + diagrams) use **one** source of
truth for the math. There is no separate BMD / SFD formula inside the
dialog — both call into the same functions.

Public surface:

* :func:`evaluate_internal_force` — build a closure
  ``f(x_loc) -> value`` for ``"axial"``, ``"shear"`` or ``"moment"``
  on one element.
* :func:`sample_internal_force` — discretise that closure at
  ``n_samples`` evenly spaced station points.
* :func:`internal_force_at` — single-point read-out at one arc-length.
* :func:`draw_element_detail` — render the 4-panel detail block
  (member sketch, FBD, N/V/M mini diagrams, cross-section thumbnail)
  into a caller-supplied :class:`matplotlib.figure.Figure`. The
  caller owns the figure / canvas widget; we only draw onto it.
"""

from __future__ import annotations

from typing import Callable, Optional

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ..element import FrameElement2D, TrussElement2D
from ..model import (
    AnalysisResult,
    FrameTemperatureLoad,
    Node,
    PointLoad,
    Section,
    StructuralModel,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)
from ..profiles import section_outline


# ── N / V / M evaluator + sampler ─────────────────────────────────


def evaluate_internal_force(
    elem, ni: Node, nj: Node, f_local, kind: str,
) -> tuple[float, Optional[Callable[[float], float]]]:
    """Build a single-x evaluator ``f(x_loc) -> value`` for ``kind`` on
    this element. Returns ``(L, evaluator)``; ``evaluator`` is ``None``
    when the requested kind doesn't apply (e.g. moment/shear on a truss
    bar). Reused by :func:`sample_internal_force` for drawing and by
    the hover-status path on the main canvas for "value at the cursor's
    projected x_loc".

    Sign convention. ``V_i`` and ``M_i`` are the local member-end
    shear / moment at the i-end (``q_local = K·d − p_local`` entries
    from :meth:`FrameElement2D.local_displacement_and_end_forces`).
    ``w`` is the summed UDL intensity in +y_local. ``points`` are
    in-span point loads with ``py`` in +y_local. The point-load terms
    in ``shear`` and ``moment`` carry the **same** sign of ``py`` so
    ``dM/dx = V`` holds across the discontinuity (regression in
    ``tests/test_diagram_signs.py``).
    """
    L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
    if L < 1e-12:
        return 0.0, None
    N_i, V_i, M_i, _N_j, _V_j, _M_j = (float(v) for v in f_local)

    if kind == "axial":
        n_value = -N_i
        return L, (lambda _x, _v=n_value: _v)

    if isinstance(elem, TrussElement2D):
        return L, None

    udls: list[float] = []
    points: list[tuple[float, float]] = []
    for ml in getattr(elem, "member_loads", []):
        if isinstance(ml, UniformDistributedLoad):
            udls.append(ml.wy)
        elif isinstance(ml, PointLoad):
            points.append((ml.a, ml.py))
    w = sum(udls)

    if kind == "shear":
        def shear(x):
            v = V_i - w * x
            for a, py in points:
                if x > a:
                    v += py
            return v
        return L, shear

    if kind == "moment":
        def moment(x):
            m = -M_i + V_i * x - 0.5 * w * x * x
            for a, py in points:
                if x > a:
                    m += py * (x - a)
            return m
        return L, moment

    return L, None


def sample_internal_force(
    elem, ni: Node, nj: Node, f_local, kind: str, n_samples: int = 21,
):
    """Discretise :func:`evaluate_internal_force` at ``n_samples``
    evenly spaced station points along the element. Returns
    ``(xs, ys)`` or ``(None, None)`` if the kind doesn't apply.
    """
    L, fn = evaluate_internal_force(elem, ni, nj, f_local, kind)
    if fn is None:
        return None, None
    xs = [i * L / (n_samples - 1) for i in range(n_samples)]
    ys = [fn(x) for x in xs]
    return xs, ys


def internal_force_at(
    elem, ni: Node, nj: Node, f_local, kind: str, x_loc: float,
) -> Optional[float]:
    """Return the diagram value at arc-length ``x_loc`` along the
    element, or ``None`` if the kind doesn't apply. Used by the
    canvas hover handler to report the value at the projected cursor.
    """
    L, fn = evaluate_internal_force(elem, ni, nj, f_local, kind)
    if fn is None:
        return None
    x = max(0.0, min(L, float(x_loc)))
    return float(fn(x))


# ── Element-detail figure ─────────────────────────────────────────

_MEMBER_COLOR = "#1f3a5f"
_SELECTED_COLOR = "#d24c4c"
_LOAD_COLOR = "#1a7f37"
_DIAGRAM_COLOR = {
    "axial":  "#5c7aff",
    "shear":  "#0f9d58",
    "moment": "#d24c4c",
}


def _draw_member_sketch(ax, elem, ni: Node, nj: Node) -> None:
    """Top-left panel — member centreline + local axes + node labels
    + release/hinge dots if present."""
    ax.set_title("Member sketch", fontsize=9, pad=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.grid(True, alpha=0.25)

    dx, dy = nj.x - ni.x, nj.y - ni.y
    L = (dx * dx + dy * dy) ** 0.5
    if L < 1e-12:
        ax.text(0.5, 0.5, "zero-length member", transform=ax.transAxes,
                ha="center", va="center", color="#b00", fontsize=9)
        return

    tx, ty = dx / L, dy / L
    nxh, nyh = -ty, tx

    ax.plot([ni.x, nj.x], [ni.y, nj.y],
            color=_SELECTED_COLOR, linewidth=2.4, zorder=3)
    ax.plot([ni.x, nj.x], [ni.y, nj.y],
            marker="o", linestyle="none", color=_MEMBER_COLOR, zorder=4)
    ax.annotate(f"i ({ni.id})", (ni.x, ni.y), textcoords="offset points",
                xytext=(-10, -10), fontsize=8, color=_MEMBER_COLOR)
    ax.annotate(f"j ({nj.id})", (nj.x, nj.y), textcoords="offset points",
                xytext=(6, 6), fontsize=8, color=_MEMBER_COLOR)

    # Local axes anchored at the midpoint of the member. Arrow length
    # ~15 % of L so they read on the small thumbnail.
    cx, cy = (ni.x + nj.x) / 2.0, (ni.y + nj.y) / 2.0
    arrow_len = max(L * 0.18, 1e-6)
    ax.annotate(
        "", xy=(cx + tx * arrow_len, cy + ty * arrow_len),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.2),
    )
    ax.text(cx + tx * arrow_len * 1.08, cy + ty * arrow_len * 1.08,
            "x", fontsize=8, color="#444",
            ha="left", va="bottom")
    ax.annotate(
        "", xy=(cx + nxh * arrow_len, cy + nyh * arrow_len),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.2),
    )
    ax.text(cx + nxh * arrow_len * 1.08, cy + nyh * arrow_len * 1.08,
            "y", fontsize=8, color="#444",
            ha="left", va="bottom")

    # Release / hinge markers — small hollow circles at the released
    # end. Only frame elements carry the release_i / release_j flags.
    if isinstance(elem, FrameElement2D):
        if getattr(elem, "release_i", False):
            ax.plot(ni.x, ni.y, marker="o", markersize=8, mfc="white",
                    mec=_MEMBER_COLOR, mew=1.4, zorder=5)
        if getattr(elem, "release_j", False):
            ax.plot(nj.x, nj.y, marker="o", markersize=8, mfc="white",
                    mec=_MEMBER_COLOR, mew=1.4, zorder=5)


def _draw_fbd(
    ax, elem, ni: Node, nj: Node,
    f_local: Optional[list[float]],
) -> None:
    """Top-right panel — member with member-load glyphs and (when a
    result exists) end-force / end-moment annotations. All loads are
    drawn in the local frame using arrows perpendicular to the member
    axis; thermal loads land as a small tag instead of an arrow.
    """
    ax.set_title("Free body", fontsize=9, pad=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.grid(True, alpha=0.25)

    dx, dy = nj.x - ni.x, nj.y - ni.y
    L = (dx * dx + dy * dy) ** 0.5
    if L < 1e-12:
        ax.text(0.5, 0.5, "zero-length member", transform=ax.transAxes,
                ha="center", va="center", color="#b00", fontsize=9)
        return
    tx, ty = dx / L, dy / L
    nxh, nyh = -ty, tx

    ax.plot([ni.x, nj.x], [ni.y, nj.y],
            color=_MEMBER_COLOR, linewidth=2.0, zorder=3)

    arrow_len = max(L * 0.14, 1e-6)
    udls: list[float] = []
    points: list[tuple[float, float]] = []
    thermals: list[str] = []
    for ml in getattr(elem, "member_loads", []):
        if isinstance(ml, UniformDistributedLoad):
            udls.append(ml.wy)
        elif isinstance(ml, PointLoad):
            points.append((ml.a, ml.py))
        elif isinstance(ml, FrameTemperatureLoad):
            thermals.append(
                f"ΔT_top={ml.t_top:g}, ΔT_bot={ml.t_bottom:g}"
            )
        elif isinstance(ml, TrussTemperatureLoad):
            thermals.append(f"ΔT={ml.delta_T:g}")

    # UDL — six arrows along the span, all pointing the same way (sign
    # of summed w in local +y).
    w_sum = sum(udls)
    if abs(w_sum) > 0:
        sign = -1.0 if w_sum > 0 else 1.0   # arrow points -y_local on +w
        for k in range(6):
            x_loc = (k + 0.5) * L / 6.0
            ax = _arrow_in_local(
                ax, ni, tx, ty, nxh, nyh, x_loc, arrow_len * sign,
            )
        # Use a label that doesn't require unicode escape gymnastics.
        ax.text(ni.x + tx * L / 2.0 + nxh * arrow_len * 1.6,
                ni.y + ty * L / 2.0 + nyh * arrow_len * 1.6,
                f"w = {w_sum:g}", fontsize=8, color=_LOAD_COLOR,
                ha="center")

    for a, py in points:
        sign = -1.0 if py > 0 else 1.0
        ax = _arrow_in_local(
            ax, ni, tx, ty, nxh, nyh, a, arrow_len * 1.5 * sign,
            color=_LOAD_COLOR,
        )
        px = ni.x + tx * a + nxh * arrow_len * 1.7 * sign
        py_label = ni.y + ty * a + nyh * arrow_len * 1.7 * sign
        ax.text(px, py_label, f"P = {py:g}", fontsize=8,
                color=_LOAD_COLOR, ha="center")

    if thermals:
        ax.text(
            (ni.x + nj.x) / 2.0, (ni.y + nj.y) / 2.0 - arrow_len * 0.6,
            "; ".join(thermals), fontsize=8, color="#a06b00",
            ha="center", va="top",
        )

    # End forces only when an analysis result is present. f_local is
    # the 6-vector [N_i, V_i, M_i, N_j, V_j, M_j] in the element's
    # local frame; we render small annotation arrows just outside the
    # nodes so they don't visually overlap the in-span loads.
    if f_local is not None:
        N_i, V_i, M_i, N_j, V_j, M_j = (float(v) for v in f_local)
        _end_force_label(ax, ni, tx, ty, nxh, nyh,
                          N=N_i, V=V_i, M=M_i, side="i")
        _end_force_label(ax, nj, -tx, -ty, -nxh, -nyh,
                          N=N_j, V=V_j, M=M_j, side="j")


def _arrow_in_local(
    ax, anchor: Node, tx: float, ty: float, nxh: float, nyh: float,
    x_loc: float, length: float, *, color: str = _LOAD_COLOR,
):
    """Helper: draw an arrow at ``x_loc`` along the member, length and
    direction given in the *local* y axis (signed). Returns the same
    axes so callers can chain — the trick is so the per-arrow helper
    can be invoked in a list comprehension without exploding the
    drawing block.
    """
    x0 = anchor.x + tx * x_loc + nxh * length
    y0 = anchor.y + ty * x_loc + nyh * length
    x1 = anchor.x + tx * x_loc
    y1 = anchor.y + ty * x_loc
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
    )
    return ax


def _end_force_label(
    ax, anchor: Node, tx: float, ty: float, nxh: float, nyh: float,
    *, N: float, V: float, M: float, side: str,
) -> None:
    """End-force / end-moment annotation at one end of the member.

    Direction conventions match the local frame: ``N`` along +x_local,
    ``V`` along +y_local, ``M`` rendered as a small text tag.
    """
    label_x = anchor.x + nxh * 0.06
    label_y = anchor.y + nyh * 0.06
    ax.text(label_x, label_y,
            f"{side}: N={N:.3g}, V={V:.3g}, M={M:.3g}",
            fontsize=7, color=_SELECTED_COLOR, ha="left", va="bottom")


def _draw_internal_force_diagrams(
    ax, elem, ni: Node, nj: Node,
    f_local: Optional[list[float]],
    n_samples: int = 11,
) -> None:
    """Bottom-left panel — three stacked traces (N, V, M) versus
    arc-length along the member. Uses :func:`sample_internal_force`
    so the dialog and the main canvas share one source of truth.
    """
    ax.set_title("Internal forces", fontsize=9, pad=2)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("force / moment", fontsize=8)
    ax.grid(True, alpha=0.25)

    if f_local is None:
        ax.text(0.5, 0.5, "Run analysis to see diagrams",
                transform=ax.transAxes, ha="center", va="center",
                color="#555", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        return

    is_truss = isinstance(elem, TrussElement2D)
    legend_handles = []
    for kind in ("axial", "shear", "moment"):
        xs, ys = sample_internal_force(
            elem, ni, nj, f_local, kind, n_samples=n_samples,
        )
        if xs is None:
            continue
        (line,) = ax.plot(
            xs, ys, color=_DIAGRAM_COLOR[kind], linewidth=1.4,
            label=kind.upper()[0],
        )
        legend_handles.append(line)
    if is_truss:
        ax.text(
            0.98, 0.02, "shear / moment not applicable (truss)",
            transform=ax.transAxes, ha="right", va="bottom",
            color="#555", fontsize=8, style="italic",
        )
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=7,
                  loc="upper right", frameon=False)


def _draw_section_thumbnail(ax, section: Optional[Section]) -> None:
    """Bottom-right panel — small filled outline of the element's
    section. Uses :func:`profiles.section_outline` for vertices."""
    ax.set_title("Section", fontsize=9, pad=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)
    ax.set_axis_off()

    if section is None:
        ax.text(0.5, 0.5, "no section", transform=ax.transAxes,
                ha="center", va="center", color="#555", fontsize=9)
        return
    try:
        pts = section_outline(section)
    except (ValueError, KeyError):
        ax.text(0.5, 0.5, "(outline unavailable)",
                transform=ax.transAxes, ha="center", va="center",
                color="#b00", fontsize=8)
        return
    zs = [p[1] for p in pts]
    ys = [p[0] for p in pts]
    ax.fill(zs, ys, facecolor="#cfe3f6", edgecolor=_MEMBER_COLOR,
            linewidth=1.2, alpha=0.95)
    label = section.shape_type or "manual"
    ax.text(0.5, -0.02, label, transform=ax.transAxes,
            ha="center", va="top", fontsize=8, color="#555")


def draw_element_detail(
    fig: Figure, elem, model: StructuralModel,
    result: Optional[AnalysisResult] = None,
    *, n_samples: int = 11,
) -> dict:
    """Render the four-panel element detail into ``fig`` and return a
    handle dict keyed by panel name (``"sketch"``, ``"fbd"``,
    ``"diagrams"``, ``"section"``). Callers can introspect the
    returned axes (the tests do, asserting line counts and placeholder
    text).
    """
    fig.clear()
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35,
                   top=0.93, bottom=0.10, left=0.10, right=0.97)
    ax_sketch = fig.add_subplot(gs[0, 0])
    ax_fbd = fig.add_subplot(gs[0, 1])
    ax_diag = fig.add_subplot(gs[1, 0])
    ax_section = fig.add_subplot(gs[1, 1])

    ni = model.nodes.get(getattr(elem, "node_i", None))
    nj = model.nodes.get(getattr(elem, "node_j", None))
    if ni is None or nj is None:
        for a in (ax_sketch, ax_fbd, ax_diag, ax_section):
            a.text(0.5, 0.5, "missing node", transform=a.transAxes,
                   ha="center", va="center", color="#b00", fontsize=9)
        return {"sketch": ax_sketch, "fbd": ax_fbd,
                "diagrams": ax_diag, "section": ax_section}

    section = None
    sid = getattr(elem, "section_id", None)
    if sid is not None:
        section = model.sections.get(sid)

    f_local = None
    if result is not None and getattr(result, "member_results", None):
        mr = result.member_results.get(elem.id)
        if mr is not None and "f_local" in mr:
            f_local = mr["f_local"]

    _draw_member_sketch(ax_sketch, elem, ni, nj)
    _draw_fbd(ax_fbd, elem, ni, nj, f_local)
    _draw_internal_force_diagrams(
        ax_diag, elem, ni, nj, f_local, n_samples=n_samples,
    )
    _draw_section_thumbnail(ax_section, section)
    return {"sketch": ax_sketch, "fbd": ax_fbd,
            "diagrams": ax_diag, "section": ax_section}
