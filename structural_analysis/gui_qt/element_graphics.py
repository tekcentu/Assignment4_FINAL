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
* :func:`draw_element_detail` — render the landscape-stacked detail
  block into a caller-supplied :class:`matplotlib.figure.Figure`.
  Returns an :class:`ElementDetailAxes` dict subclass that carries the
  four standard keys (backward-compat with the smoke-test assertion
  ``set(axes) == {"sketch", "fbd", "diagrams", "section"}``) plus
  ``.ax_n``, ``.ax_v``, ``.ax_m`` attributes for the interactive
  crosshair layer.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

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

    # Project each mechanical load onto the element's local axes so the
    # diagram math sees the same (wx_l, wy_l) / (px_l, py_l) the FEM
    # math used (see element._project_load_to_local). For local loads
    # this is a no-op; for global loads inclined members pick up both
    # axial and transverse contributions.
    from ..element import _project_load_to_local
    c, s = (nj.x - ni.x) / L, (nj.y - ni.y) / L
    udl_wx_total = 0.0
    udl_wy_total = 0.0
    axial_points: list[tuple[float, float]] = []
    transverse_points: list[tuple[float, float]] = []
    for ml in getattr(elem, "member_loads", []):
        if isinstance(ml, UniformDistributedLoad):
            wx_l, wy_l = _project_load_to_local(
                ml.wx, ml.wy, ml.coord_system, c, s,
            )
            udl_wx_total += wx_l
            udl_wy_total += wy_l
        elif isinstance(ml, PointLoad):
            px_l, py_l = _project_load_to_local(
                ml.px, ml.py, ml.coord_system, c, s,
            )
            if px_l != 0.0:
                axial_points.append((ml.a, px_l))
            if py_l != 0.0:
                transverse_points.append((ml.a, py_l))

    if kind == "axial":
        # N(x) tension-positive, derived from left-FBD equilibrium:
        #   N(x) = -N_i - wx_local * x - Σ px_local for a < x
        # When there are no axial member loads this collapses to the
        # constant -N_i used by every existing test.
        def axial(x):
            n = -N_i - udl_wx_total * x
            for a, px in axial_points:
                if x > a:
                    n -= px
            return n
        return L, axial

    if isinstance(elem, TrussElement2D):
        return L, None

    if kind == "shear":
        def shear(x):
            v = V_i - udl_wy_total * x
            for a, py in transverse_points:
                if x > a:
                    v += py
            return v
        return L, shear

    if kind == "moment":
        def moment(x):
            m = -M_i + V_i * x - 0.5 * udl_wy_total * x * x
            for a, py in transverse_points:
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

    Raises :class:`ValueError` if ``n_samples < 2`` — one station is
    not enough to form a polyline, and we'd otherwise divide by zero
    in the step calculation. Fail fast with a clear message instead
    of letting the caller see a ZeroDivisionError.
    """
    if n_samples < 2:
        raise ValueError(
            f"n_samples must be >= 2 to form a polyline, got {n_samples}"
        )
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


# ── Return type ──────────────────────────────────────────────────


class ElementDetailAxes(dict):
    """Backward-compatible dict returned by :func:`draw_element_detail`.

    Carries exactly four standard keys (``"sketch"``, ``"fbd"``,
    ``"diagrams"``, ``"section"``) so existing tests that assert
    ``set(axes) == {"sketch", "fbd", "diagrams", "section"}`` continue
    to pass.  The three N/V/M subplot axes are *also* accessible as
    instance attributes (``.ax_n``, ``.ax_v``, ``.ax_m``) for the
    dialog's interactive crosshair layer without breaking the set check.

    ``"diagrams"`` always maps to the N (axial) subplot so the
    pre-/post-solve line-count assertions in the smoke tests pass:
    pre-solve the N subplot holds the placeholder text and zero lines;
    post-solve it holds the axial trace with ≥ n_samples data points.
    """
    ax_n: object = None
    ax_v: object = None
    ax_m: object = None


# ── Colour palette ───────────────────────────────────────────────

_MEMBER_COLOR = "#1f3a5f"
_SELECTED_COLOR = "#d24c4c"
_LOAD_COLOR = "#1a7f37"
_DIAGRAM_COLOR = {
    "axial":  "#5c7aff",
    "shear":  "#0f9d58",
    "moment": "#d24c4c",
}


# ── Local-frame helpers ──────────────────────────────────────────


def _member_length(ni: Node, nj: Node) -> float:
    dx, dy = nj.x - ni.x, nj.y - ni.y
    return math.sqrt(dx * dx + dy * dy)


def _spine_clean(ax) -> None:
    """Hide top and right spines (clean landscape look)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_member_sketch(ax, elem, ni: Node, nj: Node,
                        section: Optional[Section] = None) -> None:
    """Member sketch panel — local coordinate frame (horizontal).

    x = 0 at node i, x = L at node j.  The member is always drawn as
    a horizontal line so inclined members are still readable.
    """
    L = _member_length(ni, nj)
    ax.set_title("Member sketch — local frame", fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("x (m)", fontsize=7)
    _spine_clean(ax)
    ax.grid(linestyle=":", alpha=0.3)

    if L < 1e-12:
        ax.text(0.5, 0.5, "zero-length member", transform=ax.transAxes,
                ha="center", va="center", color="#b00", fontsize=9)
        return

    h_ref = 0.1 * L
    if section is not None and getattr(section, "depth", 0) > 0:
        h_ref = max(section.depth, 0.05 * L)

    ax.set_xlim(-0.06 * L, 1.06 * L)
    ax.set_ylim(-0.55 * h_ref, 0.85 * h_ref)
    ax.set_yticks([])

    # Member centreline
    ax.plot([0, L], [0, 0], color=_SELECTED_COLOR, linewidth=2.4, zorder=3)
    ax.plot([0, L], [0, 0], marker="o", linestyle="none",
            color=_MEMBER_COLOR, markersize=5, zorder=4)

    # Node labels
    ax.annotate(f"i ({ni.id})", (0, 0), textcoords="offset points",
                xytext=(-4, 6), fontsize=7, color=_MEMBER_COLOR)
    ax.annotate(f"j ({nj.id})", (L, 0), textcoords="offset points",
                xytext=(4, 6), fontsize=7, color=_MEMBER_COLOR)

    # Local axis arrows anchored at 55 % of L
    cx = 0.55 * L
    arr_x = 0.13 * L
    arr_y = 0.3 * h_ref
    ax.annotate("", xy=(cx + arr_x, 0), xytext=(cx, 0),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.1))
    ax.text(cx + arr_x * 1.1, 0.04 * h_ref, "x", fontsize=7, color="#444",
            ha="left", va="bottom")
    ax.annotate("", xy=(cx, arr_y), xytext=(cx, 0),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.1))
    ax.text(cx + 0.02 * L, arr_y * 1.05, "y", fontsize=7, color="#444",
            ha="left", va="bottom")

    # Release / hinge markers
    if isinstance(elem, FrameElement2D):
        if getattr(elem, "release_i", False):
            ax.plot(0, 0, marker="o", markersize=9, mfc="white",
                    mec=_MEMBER_COLOR, mew=1.4, zorder=5)
        if getattr(elem, "release_j", False):
            ax.plot(L, 0, marker="o", markersize=9, mfc="white",
                    mec=_MEMBER_COLOR, mew=1.4, zorder=5)


def _draw_fbd(ax, elem, ni: Node, nj: Node,
              f_local: Optional[list],
              section: Optional[Section] = None) -> None:
    """Free body diagram in LOCAL coordinate frame.

    Member drawn as a horizontal line x ∈ [0, L].  Loads and end-force
    arrows are all projected into local coordinates so inclined members
    remain readable.
    """
    L = _member_length(ni, nj)
    ax.set_title("Free body — local frame", fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("x (m)", fontsize=7)
    _spine_clean(ax)
    ax.grid(linestyle=":", alpha=0.3)

    if L < 1e-12:
        ax.text(0.5, 0.5, "zero-length member", transform=ax.transAxes,
                ha="center", va="center", color="#b00", fontsize=9)
        return

    h_ref = 0.1 * L
    if section is not None and getattr(section, "depth", 0) > 0:
        h_ref = max(section.depth, 0.05 * L)
    arrow_h = 0.45 * h_ref

    ax.set_xlim(-0.12 * L, 1.12 * L)
    ax.set_ylim(-0.75 * h_ref, 0.85 * h_ref)
    ax.set_yticks([])

    # Member baseline
    ax.plot([0, L], [0, 0], color=_MEMBER_COLOR, linewidth=2.0, zorder=3)
    ax.plot([0, L], [0, 0], marker="o", linestyle="none",
            color=_MEMBER_COLOR, markersize=5, zorder=4)

    # FBD shows arrows perpendicular to the (horizontal-drawn) member;
    # that's the local-y projection for both local and global loads.
    # Axial components don't get a perpendicular arrow here — the N
    # subplot below makes axial behavior unambiguous.
    from ..element import _project_load_to_local
    if L > 0:
        c_fbd = (nj.x - ni.x) / L
        s_fbd = (nj.y - ni.y) / L
    else:
        c_fbd, s_fbd = 1.0, 0.0
    udls: list[float] = []
    points_list: list[tuple[float, float]] = []
    thermals: list[str] = []
    for ml in getattr(elem, "member_loads", []):
        if isinstance(ml, UniformDistributedLoad):
            _wx_l, wy_l = _project_load_to_local(
                ml.wx, ml.wy, ml.coord_system, c_fbd, s_fbd,
            )
            udls.append(wy_l)
        elif isinstance(ml, PointLoad):
            _px_l, py_l = _project_load_to_local(
                ml.px, ml.py, ml.coord_system, c_fbd, s_fbd,
            )
            points_list.append((ml.a, py_l))
        elif isinstance(ml, FrameTemperatureLoad):
            thermals.append(f"ΔT_top={ml.t_top:g}, ΔT_bot={ml.t_bottom:g}")
        elif isinstance(ml, TrussTemperatureLoad):
            thermals.append(f"ΔT={ml.delta_T:g}")

    # UDL — six arrows perpendicular to baseline
    w_sum = sum(udls)
    if abs(w_sum) > 0:
        sign = 1.0 if w_sum > 0 else -1.0
        for k in range(6):
            x_loc = (k + 0.5) * L / 6.0
            tail_y = sign * arrow_h
            ax.annotate("", xy=(x_loc, 0), xytext=(x_loc, tail_y),
                        arrowprops=dict(arrowstyle="->", color=_LOAD_COLOR,
                                        lw=1.1))
        ax.text(L / 2.0, sign * (arrow_h * 1.55),
                f"w = {w_sum:g} kN/m", fontsize=7, color=_LOAD_COLOR,
                ha="center", va="center")

    # Point loads
    for a, py in points_list:
        sign = 1.0 if py > 0 else -1.0
        tail_y = sign * arrow_h * 1.2
        ax.annotate("", xy=(a, 0), xytext=(a, tail_y),
                    arrowprops=dict(arrowstyle="->", color=_LOAD_COLOR,
                                    lw=1.2))
        ax.text(a, sign * (arrow_h * 1.85),
                f"P={py:g}", fontsize=7, color=_LOAD_COLOR, ha="center")

    # Thermal load tag
    if thermals:
        ax.text(L / 2.0, 0.7 * h_ref, "; ".join(thermals),
                fontsize=7, color="#a06b00", ha="center", va="bottom",
                style="italic")

    # End-force arrows (only when analysis result present)
    if f_local is not None:
        N_i, V_i, M_i, N_j, V_j, M_j = (float(v) for v in f_local)
        _end_force_arrows(ax, 0, L, h_ref, arrow_h,
                          N_i, V_i, M_i, "i", side_sign=-1)
        _end_force_arrows(ax, L, L, h_ref, arrow_h,
                          N_j, V_j, M_j, "j", side_sign=+1)


def _end_force_arrows(ax, x_node: float, L: float, h_ref: float,
                      arrow_h: float, N: float, V: float, M: float,
                      label: str, side_sign: int) -> None:
    """Draw N, V, M arrows/indicators at one terminal of the member."""
    offset = 0.10 * L * side_sign

    # Axial (N) — horizontal arrow
    n_col = _DIAGRAM_COLOR["axial"]
    ax.annotate(
        "", xy=(x_node, 0), xytext=(x_node - offset, 0),
        arrowprops=dict(arrowstyle="->", color=n_col, lw=1.2),
    )
    ax.text(x_node - offset * 1.3, 0.08 * h_ref,
            f"N={N:.3g}", fontsize=6, color=n_col, ha="center")

    # Shear (V) — vertical arrow
    v_col = _DIAGRAM_COLOR["shear"]
    v_sign = 1.0 if V >= 0 else -1.0
    ax.annotate(
        "", xy=(x_node, 0), xytext=(x_node, -v_sign * arrow_h * 0.8),
        arrowprops=dict(arrowstyle="->", color=v_col, lw=1.2),
    )
    ax.text(x_node + 0.03 * L * side_sign, -v_sign * arrow_h * 1.1,
            f"V={V:.3g}", fontsize=6, color=v_col, ha="center")

    # Moment (M) — arc indicator + label
    m_col = _DIAGRAM_COLOR["moment"]
    arc_char = "↺" if M >= 0 else "↻"
    ax.text(x_node - offset * 0.7, 0.35 * h_ref,
            f"{arc_char} M={M:.3g}", fontsize=6, color=m_col,
            ha="center", va="bottom")


def _draw_single_nvm_diagram(ax, xs, ys, label: str, unit: str,
                             color: str, *, invert: bool = False) -> None:
    """Render one of the N / V / M diagrams onto ``ax``.

    ``xs`` / ``ys`` are the sampled arrays (or ``None`` for inapplicable
    kinds on truss elements).  ``invert=True`` applies the structural
    BMD tension-fibre convention (moment drawn downward).
    """
    _spine_clean(ax)
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel(f"{label} ({unit})", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    if xs is None:
        ax.text(0.5, 0.5, "shear / moment not applicable (truss)",
                transform=ax.transAxes, ha="center", va="center",
                color="#555", fontsize=7, style="italic")
        return

    ax.fill_between(xs, ys, 0, color=color, alpha=0.18)
    ax.plot(xs, ys, color=color, linewidth=1.5)
    ax.axhline(0, color="#888", linewidth=0.5, linestyle="-", zorder=1)

    if invert:
        ax.invert_yaxis()


def _draw_section_thumbnail(ax, section: Optional[Section]) -> None:
    """Section thumbnail panel — section outline in local (z, y) frame.

    z maps to the horizontal plot axis, y to the vertical axis.
    For manual sections the √A equivalent-square fallback is drawn
    with an explanatory italic label.
    """
    ax.set_title("Section", fontsize=8, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

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
            linewidth=1.2, alpha=0.95, zorder=2)
    ax.set_aspect("equal", adjustable="datalim")

    shape = getattr(section, "shape_type", None) or "manual"
    if shape == "manual":
        ax.text(0.5, -0.04,
                "Approximate area-equivalent\nsquare fallback (√A)",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=6, style="italic", color="#666")
    else:
        ax.text(0.5, -0.04, shape, transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color="#555")


# ── Main entry point ─────────────────────────────────────────────


def draw_element_detail(
    fig: Figure, elem, model: StructuralModel,
    result: Optional[AnalysisResult] = None,
    *, n_samples: int = 11,
    section_fig: Optional[Figure] = None,
) -> ElementDetailAxes:
    """Render the landscape-stacked element detail into ``fig``.

    Default layout (5 rows × 6 columns, section inside ``fig``)::

        Row 0, cols 0-1  →  section thumbnail
        Row 0, cols 2-5  →  member sketch   (local frame)
        Row 1, cols 2-5  →  free body       (local frame)
        Row 2, cols 0-5  →  N — axial  (kN)     "diagrams" key
        Row 3, cols 0-5  →  V — shear  (kN)     .ax_v attribute
        Row 4, cols 0-5  →  M — moment (kN·m)   .ax_m attribute

    When ``section_fig`` is provided, the section thumbnail is drawn
    into that *separate* figure instead — letting the host dialog
    place a compact section preview alongside its property form, so
    the main ``fig`` only needs five thin rows for sketch + FBD +
    N + V + M.  This trims the dialog's vertical footprint without
    sacrificing the section drawing.

    Returns an :class:`ElementDetailAxes` dict with exactly four keys
    ``{"sketch", "fbd", "diagrams", "section"}`` for backward compat;
    ``.ax_n / .ax_v / .ax_m`` give the dialog access to all three
    diagram sub-panels.  When ``section_fig`` is given,
    ``axes["section"]`` points at the axis inside that figure.
    """
    fig.clear()

    if section_fig is not None:
        # Compact layout: section lives in its own figure, main figure
        # is a single full-width column with five thin rows.
        section_fig.clear()
        ax_section = section_fig.add_subplot(111)
        gs = GridSpec(
            5, 1, figure=fig,
            hspace=0.70,
            top=0.96, bottom=0.06, left=0.14, right=0.97,
            height_ratios=[1.3, 1.3, 1.0, 1.0, 1.4],
        )
        ax_sketch = fig.add_subplot(gs[0, 0])
        ax_fbd    = fig.add_subplot(gs[1, 0])
        ax_n      = fig.add_subplot(gs[2, 0])
        ax_v      = fig.add_subplot(gs[3, 0], sharex=ax_n)
        ax_m      = fig.add_subplot(gs[4, 0], sharex=ax_n)
    else:
        gs = GridSpec(
            5, 6, figure=fig,
            hspace=0.60, wspace=0.40,
            top=0.96, bottom=0.05, left=0.08, right=0.97,
            height_ratios=[2, 2, 1.2, 1.2, 1.6],
        )
        ax_section = fig.add_subplot(gs[0:2, 0:2])
        ax_sketch  = fig.add_subplot(gs[0, 2:6])
        ax_fbd     = fig.add_subplot(gs[1, 2:6])
        ax_n       = fig.add_subplot(gs[2, 0:6])
        ax_v       = fig.add_subplot(gs[3, 0:6], sharex=ax_n)
        ax_m       = fig.add_subplot(gs[4, 0:6], sharex=ax_n)

    ni = model.nodes.get(getattr(elem, "node_i", None))
    nj = model.nodes.get(getattr(elem, "node_j", None))
    if ni is None or nj is None:
        for a in (ax_sketch, ax_fbd, ax_n, ax_v, ax_m, ax_section):
            a.text(0.5, 0.5, "missing node", transform=a.transAxes,
                   ha="center", va="center", color="#b00", fontsize=9)
        axes = ElementDetailAxes(
            sketch=ax_sketch, fbd=ax_fbd, diagrams=ax_n, section=ax_section,
        )
        axes.ax_n = ax_n
        axes.ax_v = ax_v
        axes.ax_m = ax_m
        return axes

    section = None
    sid = getattr(elem, "section_id", None)
    if sid is not None:
        section = model.sections.get(sid)

    f_local = None
    if result is not None and getattr(result, "member_results", None):
        mr = result.member_results.get(elem.id)
        if mr is not None and "f_local" in mr:
            f_local = mr["f_local"]

    # ── Upper panels (local frame) ────────────────────────────────
    _draw_member_sketch(ax_sketch, elem, ni, nj, section)
    _draw_fbd(ax_fbd, elem, ni, nj, f_local, section)
    _draw_section_thumbnail(ax_section, section)

    L = _member_length(ni, nj)
    if L > 1e-12:
        ax_n.set_xlim(0.0, L)

    # ── N / V / M diagrams ────────────────────────────────────────
    if f_local is None:
        # Pre-solve placeholder — zero lines, placeholder text on ax_n.
        # The "Run analysis" text MUST land on ax_n because the smoke
        # test asserts axes["diagrams"].texts (which is ax_n) contains
        # "Run analysis".
        ax_n.text(0.5, 0.5, "Run analysis to see N/V/M diagrams",
                  transform=ax_n.transAxes, ha="center", va="center",
                  color="#555", fontsize=9)
        ax_n.set_xticks([])
        ax_n.set_yticks([])
        _spine_clean(ax_n)
        for ax, lbl in ((ax_v, "V — shear"), (ax_m, "M — moment")):
            ax.set_xticks([])
            ax.set_yticks([])
            _spine_clean(ax)
            ax.text(0.5, 0.5, lbl, transform=ax.transAxes,
                    ha="center", va="center", color="#bbb", fontsize=8,
                    style="italic")
    else:
        xs_n, ys_n = sample_internal_force(elem, ni, nj, f_local,
                                            "axial",  n_samples)
        xs_v, ys_v = sample_internal_force(elem, ni, nj, f_local,
                                            "shear",  n_samples)
        xs_m, ys_m = sample_internal_force(elem, ni, nj, f_local,
                                            "moment", n_samples)

        _draw_single_nvm_diagram(ax_n, xs_n, ys_n, "N", "kN",
                                 color=_DIAGRAM_COLOR["axial"])
        _draw_single_nvm_diagram(ax_v, xs_v, ys_v, "V", "kN",
                                 color=_DIAGRAM_COLOR["shear"])
        _draw_single_nvm_diagram(ax_m, xs_m, ys_m, "M", "kN·m",
                                 color=_DIAGRAM_COLOR["moment"],
                                 invert=True)

    axes = ElementDetailAxes(
        sketch=ax_sketch, fbd=ax_fbd, diagrams=ax_n, section=ax_section,
    )
    axes.ax_n = ax_n
    axes.ax_v = ax_v
    axes.ax_m = ax_m
    return axes
