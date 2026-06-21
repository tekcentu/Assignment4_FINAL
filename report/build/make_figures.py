"""Generate every figure used in the final report.

Pulls real models through ``structural_analysis.main.run_analysis`` and
plots their results with matplotlib — every figure is reproducible from
this script and represents true program behaviour. Architecture/data-model
diagrams are drawn with matplotlib's primitive shapes (no Mermaid renderer
in this environment).

Output: report/figures/*.png at 200 dpi.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.file_io import read_input_file
from structural_analysis.main import run_analysis
from structural_analysis.gui_qt.element_graphics import sample_internal_force


FIG_DIR = Path("report/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
DPI = 200


# ── Helpers ──────────────────────────────────────────────────────────────

def _save(fig, name: str) -> Path:
    p = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def _draw_box(ax, x, y, w, h, text, fc="#e8eef6", ec="#2c3e50", fs=9,
              ha="center", va="center"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05",
        linewidth=1.2, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, wrap=True)
    return (x, y, w, h)


def _arrow(ax, x1, y1, x2, y2, text=None, ec="#34495e", style="->"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
        color=ec, linewidth=1.2,
    )
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, text,
                ha="center", fontsize=7, color=ec, style="italic")


def _scale_for_plot(values, target_fraction=0.12, span=1.0):
    """Return a scale that maps max |values| to target_fraction*span."""
    m = max((abs(v) for v in values), default=0.0)
    if m <= 0:
        return 0.0
    return target_fraction * span / m


# ── Architecture overview ────────────────────────────────────────────────

def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

    # Layer bands
    for y0, y1, color, label in [
        (4.4, 6.1, "#dfe8f2", "GUI presentation — gui_qt (PyQt6 + matplotlib)"),
        (2.6, 4.2, "#fff2cc", "Shared Qt-free adapter — gui_common"),
        (0.2, 2.4, "#dcf0dc", "Analysis core — pure NumPy / SciPy"),
    ]:
        ax.add_patch(patches.Rectangle((0.2, y0), 9.6, y1 - y0,
                                       facecolor=color, edgecolor="#888",
                                       linewidth=0.7, alpha=0.65))
        ax.text(0.35, y1 - 0.18, label, fontsize=8.5,
                color="#444", style="italic")

    # GUI boxes
    _draw_box(ax, 1.5, 5.4, 2.4, 0.55, "app.MainWindow\n(owns model, undo)")
    _draw_box(ax, 4.5, 5.4, 2.4, 0.55, "canvas.py\nmatplotlib draw + select")
    _draw_box(ax, 7.5, 5.4, 2.4, 0.55, "element_graphics.py\nN/V/M source of truth")
    _draw_box(ax, 1.5, 4.6, 2.4, 0.40, "controllers.py\ntool state machines", fs=8)
    _draw_box(ax, 4.5, 4.6, 2.4, 0.40, "dialogs / grid / snap", fs=8)
    _draw_box(ax, 7.5, 4.6, 2.4, 0.40, "modal_view · *_summary", fs=8)

    # gui_common boxes
    _draw_box(ax, 1.6, 3.4, 2.7, 0.55, "commands.py\nCommand.do / undo")
    _draw_box(ax, 4.6, 3.4, 2.6, 0.55, "validation.py\npre-solve UX checks")
    _draw_box(ax, 7.5, 3.4, 2.4, 0.55, "results_view\nunits · file_writer")

    # Core boxes
    _draw_box(ax, 1.4, 1.5, 2.0, 0.55, "model.py\nStructuralModel")
    _draw_box(ax, 3.7, 1.5, 2.0, 0.55, "assembler.py\nDofManager · K, F")
    _draw_box(ax, 6.0, 1.5, 2.0, 0.55, "solver.py\nKff·Df = Ff − Kfs·Ds")
    _draw_box(ax, 8.3, 1.5, 1.6, 0.55, "postprocessor\nforces · R")
    _draw_box(ax, 1.4, 0.7, 2.0, 0.45, "file_io.py\nread/write *.txt", fs=8)
    _draw_box(ax, 3.7, 0.7, 2.0, 0.45, "main.py\nrun_analysis", fs=8)
    _draw_box(ax, 6.0, 0.7, 2.0, 0.45, "multi_case_result", fs=8)
    _draw_box(ax, 8.3, 0.7, 1.6, 0.45, "modal · mass · profiles", fs=7)

    # Vertical layer flow
    _arrow(ax, 5.0, 4.35, 5.0, 4.0)
    _arrow(ax, 5.0, 2.55, 5.0, 2.2)
    ax.text(5.05, 4.18, "uses", fontsize=7.5, color="#666", style="italic")
    ax.text(5.05, 2.36, "delegates", fontsize=7.5, color="#666", style="italic")

    ax.set_title("Figure 1 — Layered architecture of structural_analysis",
                 fontsize=11, pad=10)
    return _save(fig, "architecture_overview.png")


# ── Analysis pipeline ────────────────────────────────────────────────────

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.0); ax.axis("off")
    boxes = [
        (0.9, 2.0, "input .txt", "#f6e6d4"),
        (2.4, 2.0, "read_input_file\n→ StructuralModel", "#dfe8f2"),
        (4.3, 2.0, "assemble_global_system\nK, F, DofManager", "#dfe8f2"),
        (6.4, 2.0, "solver.solve_system\nKff·Df = Ff − Kfs·Ds", "#dcf0dc"),
        (8.6, 2.7, "postprocessor\ncompute_member_forces", "#dcf0dc"),
        (8.6, 1.3, "postprocessor\ncompute_reactions", "#dcf0dc"),
        (8.6, 0.4, "equilibrium_check", "#dcf0dc"),
        (4.5, 3.4, "run_multi_case_analysis\n(per case)", "#fff2cc"),
        (7.0, 3.4, "MultiCaseAnalysisResult\n+ LoadCombination", "#fff2cc"),
    ]
    for x, y, t, c in boxes:
        _draw_box(ax, x, y, 1.7, 0.55, t, fc=c, fs=8)
    # Forward arrows
    for (a, b) in [(0, 1), (1, 2), (2, 3)]:
        _arrow(ax, boxes[a][0] + 0.85, boxes[a][1], boxes[b][0] - 0.85, boxes[b][1])
    # Solver → postproc + reactions
    _arrow(ax, 6.4 + 0.85, 2.05, 8.6 - 0.85, 2.7)
    _arrow(ax, 6.4 + 0.85, 1.95, 8.6 - 0.85, 1.3)
    _arrow(ax, 8.6, 1.3 - 0.30, 8.6, 0.4 + 0.30)
    # Multi-case branch
    _arrow(ax, 2.4, 2.0 + 0.30, 4.5 - 0.85, 3.4, style="->")
    _arrow(ax, 4.5 + 0.85, 3.4, 7.0 - 0.85, 3.4)
    ax.set_title("Figure 2 — Analysis pipeline (main.run_analysis & multi-case)",
                 fontsize=11, pad=8)
    return _save(fig, "analysis_pipeline.png")


# ── Data model diagram ──────────────────────────────────────────────────

def fig_data_model():
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 6.3); ax.axis("off")

    # Aggregate root
    _draw_box(ax, 5.25, 5.55, 3.2, 0.7,
              "StructuralModel\n— root aggregate —", fc="#f6e6d4")

    # First-row contained entities
    row1 = [
        (1.0, 4.2, "Node\nid · x · y", "#dfe8f2"),
        (2.7, 4.2, "Material\nid · E · α · ρ", "#dfe8f2"),
        (4.4, 4.2, "Section\nid · A · I · depth\n(→ Material)", "#dfe8f2"),
        (6.3, 4.2, "Support\nnode_id · ux/uy/rz\n+ settlements", "#dfe8f2"),
        (8.4, 4.2, "Element2D\n(abstract)", "#dcf0dc"),
    ]
    for x, y, t, c in row1:
        _draw_box(ax, x, y, 1.6, 0.85, t, fc=c, fs=8)

    # Element subclasses
    _draw_box(ax, 7.5, 2.9, 1.5, 0.55, "FrameElement2D\nux uy rz", fc="#dcf0dc", fs=8)
    _draw_box(ax, 9.4, 2.9, 1.5, 0.55, "TrussElement2D\naxial only", fc="#dcf0dc", fs=8)
    _arrow(ax, 8.4, 4.2 - 0.42, 7.5, 2.9 + 0.30, style="-|>")
    _arrow(ax, 8.4, 4.2 - 0.42, 9.4, 2.9 + 0.30, style="-|>")

    # Loads row
    loads = [
        (1.0, 2.6, "NodalLoad\nfx · fy · mz", "#fde9d9"),
        (2.7, 2.6, "UniformDistributedLoad\nwx · wy · coord_system", "#fde9d9"),
        (4.6, 2.6, "PointLoad\npx · py · a", "#fde9d9"),
        (6.6, 2.6, "FrameTemperatureLoad\nt_top · t_bottom", "#fde9d9"),
    ]
    for x, y, t, c in loads:
        _draw_box(ax, x, y, 1.7, 0.85, t, fc=c, fs=8)
    _draw_box(ax, 0.95, 1.4, 1.6, 0.55,
              "TrussTemperatureLoad\nΔT", fc="#fde9d9", fs=8)

    # Case + combination
    _draw_box(ax, 4.0, 1.4, 2.0, 0.55, "LoadCase\nname · enabled", fc="#fff2cc", fs=8)
    _draw_box(ax, 6.7, 1.4, 2.4, 0.55,
              "LoadCombination\nterms = {case: factor}", fc="#fff2cc", fs=8)

    # Result containers (downstream)
    _draw_box(ax, 2.6, 0.4, 2.4, 0.55,
              "AnalysisResult\nD · reactions · member_results", fc="#e8d8f5", fs=8)
    _draw_box(ax, 6.7, 0.4, 2.7, 0.55,
              "MultiCaseAnalysisResult\nper-case + combinations", fc="#e8d8f5", fs=8)

    # Containment arrows from StructuralModel
    for x, y, *_ in row1 + loads + [(0.95, 1.4)]:
        _arrow(ax, 5.25, 5.55 - 0.40, x, y + 0.45, style="-")
    _arrow(ax, 5.25, 5.55 - 0.40, 4.0, 1.4 + 0.30, style="-")
    _arrow(ax, 5.25, 5.55 - 0.40, 6.7, 1.4 + 0.30, style="-")

    # Result derivation
    _arrow(ax, 5.25, 0.95, 2.6, 0.7, style="-|>")
    _arrow(ax, 5.25, 0.95, 6.7, 0.7, style="-|>")
    ax.text(3.0, 0.95, "run_analysis", fontsize=7,
            color="#666", style="italic")
    ax.text(7.0, 0.95, "run_multi_case_analysis", fontsize=7,
            color="#666", style="italic")

    ax.set_title("Figure 3 — Domain model: StructuralModel + result containers",
                 fontsize=11, pad=8)
    return _save(fig, "data_model_diagram.png")


# ── GUI workflow ─────────────────────────────────────────────────────────

def fig_gui_workflow():
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 5.0); ax.axis("off")

    nodes = [
        (1.0, 4.2, "User action\n(click / key / drag)", "#fff2cc"),
        (3.4, 4.2, "controllers.Tool\non_click · on_key", "#dfe8f2"),
        (5.8, 4.2, "app.MainWindow.execute\n(Command)", "#dfe8f2"),
        (8.4, 4.2, "Command.do(model)", "#dcf0dc"),
        (8.4, 2.8, "StructuralModel\n(mutated)", "#f6e6d4"),
        (5.8, 2.8, "Push to undo stack\nclear stale result", "#dfe8f2"),
        (3.4, 2.8, "canvas.redraw\n+ validation overlay", "#dfe8f2"),
        (1.0, 2.8, "User sees update", "#fff2cc"),
        (5.8, 1.2, "Ctrl+Z → pop", "#fde9d9"),
        (8.4, 1.2, "Command.undo(model)", "#dcf0dc"),
    ]
    for x, y, t, c in nodes:
        _draw_box(ax, x, y, 1.9, 0.75, t, fc=c, fs=8)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]:
        x1, y1 = nodes[a][:2]; x2, y2 = nodes[b][:2]
        _arrow(ax, x1 + 0.95, y1, x2 - 0.95, y2) if abs(y1 - y2) < 0.1 else \
            _arrow(ax, x1, y1 - 0.40, x2, y2 + 0.40)
    # Undo path
    _arrow(ax, 5.8 - 0.95, 1.2, 1.0 + 0.95, 2.8 - 0.40, style="<-",
           ec="#a04040")
    _arrow(ax, 5.8 + 0.95, 1.2, 8.4 - 0.95, 1.2)
    _arrow(ax, 8.4, 1.2 + 0.40, 8.4, 2.8 - 0.40, style="->", ec="#a04040")
    ax.text(7.0, 1.45, "undo applies inverse to model",
            fontsize=7, color="#a04040", style="italic")

    ax.set_title("Figure 4 — GUI command / undo flow",
                 fontsize=11, pad=8)
    return _save(fig, "gui_workflow.png")


# ── Real-model figures ──────────────────────────────────────────────────

def _model_canvas(ax, model, *, deformed=False, scale=None, title=""):
    xs = [n.x for n in model.nodes.values()]
    ys = [n.y for n in model.nodes.values()]
    pad = max(0.5, 0.15 * (max(xs) - min(xs) + max(ys) - min(ys)))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.grid(linestyle=":", alpha=0.4)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title, fontsize=10)

    # Undeformed wireframe (always)
    for e in model.elements:
        ni, nj = model.nodes[e.node_i], model.nodes[e.node_j]
        col = "#88a" if deformed else "#1f3a5f"
        ls = "--" if deformed else "-"
        ax.plot([ni.x, nj.x], [ni.y, nj.y], color=col, linestyle=ls,
                linewidth=1.5 if not deformed else 0.9, zorder=2)
    # Nodes
    for nid, n in model.nodes.items():
        ax.plot(n.x, n.y, "o", color="#1f3a5f", ms=5, zorder=4)
        ax.annotate(str(nid), (n.x, n.y), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color="#1f3a5f")
    # Supports — simple triangle markers
    for s in model.supports.values():
        n = model.nodes[s.node_id]
        if s.ux and s.uy:
            ax.plot(n.x, n.y, "v", color="#444", ms=11, zorder=3)


def _draw_loads(ax, model, fscale=None):
    # nodal loads
    fmax = max(
        (abs(l.fx) + abs(l.fy) for l in model.nodal_loads), default=0.0,
    )
    if fmax > 0 and fscale is None:
        xs = [n.x for n in model.nodes.values()]
        ys = [n.y for n in model.nodes.values()]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        fscale = 0.15 * span / fmax
    for l in model.nodal_loads:
        n = model.nodes[l.node_id]
        if l.fx:
            ax.annotate("", xy=(n.x, n.y),
                        xytext=(n.x - l.fx * fscale, n.y),
                        arrowprops=dict(arrowstyle="->", color="#2ca02c",
                                        lw=2))
            ax.text(n.x - l.fx * fscale, n.y + 0.05,
                    f"{l.fx:g} kN", fontsize=7, color="#2ca02c",
                    ha="center")
        if l.fy:
            ax.annotate("", xy=(n.x, n.y),
                        xytext=(n.x, n.y - l.fy * fscale),
                        arrowprops=dict(arrowstyle="->", color="#2ca02c",
                                        lw=2))
            ax.text(n.x + 0.05, n.y - l.fy * fscale,
                    f"{l.fy:g} kN", fontsize=7, color="#2ca02c")
    # UDL strips
    from structural_analysis.model import UniformDistributedLoad, PointLoad
    for e in model.elements:
        for ml in getattr(e, "member_loads", []) or []:
            ni, nj = model.nodes[e.node_i], model.nodes[e.node_j]
            L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
            tx, ty = (nj.x - ni.x) / L, (nj.y - ni.y) / L
            nx, ny = -ty, tx
            if isinstance(ml, UniformDistributedLoad) and ml.wy:
                h = 0.25 * abs(ml.wy) / max(
                    abs(ml.wy), 1e-9,
                ) * 0.4
                sign = 1 if ml.wy > 0 else -1
                for k in range(6):
                    t = (k + 0.5) / 6
                    bx = ni.x + (nj.x - ni.x) * t
                    by = ni.y + (nj.y - ni.y) * t
                    ax.annotate(
                        "", xy=(bx, by),
                        xytext=(bx - nx * sign * h * 0.5,
                                by - ny * sign * h * 0.5),
                        arrowprops=dict(arrowstyle="->",
                                        color="#9467bd", lw=1.0))
                ax.text((ni.x + nj.x) / 2,
                        (ni.y + nj.y) / 2 + sign * 0.35,
                        f"{ml.wy:+g} kN/m", color="#9467bd",
                        fontsize=7, ha="center")


def _deformed_nodes(model, result, scale):
    out = {}
    for nid, n in model.nodes.items():
        ux, uy = 0.0, 0.0
        emap = result.E_map.get(nid)
        if emap is not None and result.D is not None:
            if emap["ux"] is not None:
                ux = float(result.D[emap["ux"]])
            if emap["uy"] is not None:
                uy = float(result.D[emap["uy"]])
        out[nid] = (n.x + ux * scale, n.y + uy * scale)
    return out


def _draw_deformed(ax, model, result, scale):
    if result.D is None:
        return
    pts = _deformed_nodes(model, result, scale)
    for e in model.elements:
        ax_, ay_ = pts[e.node_i]; bx, by = pts[e.node_j]
        ax.plot([ax_, bx], [ay_, by], color="#d24c4c", linewidth=2, zorder=5)
    for nid in pts:
        x, y = pts[nid]
        ax.plot(x, y, "o", color="#d24c4c", ms=4, zorder=6)


def _load_model(path):
    m = read_input_file(path)
    r = run_analysis(m, verbose=False)
    return m, r


def fig_example_model_and_loads():
    """Example model render (Figure 5)."""
    m, _ = _load_model("inputs/example_03_portal_frame_lateral_load.txt")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _model_canvas(ax, m, title=(
        "Example portal frame — example_03_portal_frame_lateral_load.txt\n"
        "Fixed at base, lateral 50 kN at node 4, vertical −20 kN at nodes 3 & 4"
    ))
    _draw_loads(ax, m)
    return _save(fig, "example_model.png")


def fig_example_deformed():
    """Deformed shape after analysis."""
    m, r = _load_model("inputs/example_03_portal_frame_lateral_load.txt")
    # auto-scale displacements so the deflection is visible
    Dmax = max((abs(float(v)) for v in (r.D if r.D is not None else [0])),
               default=0.0)
    span = 8.0
    scale = 0.10 * span / max(Dmax, 1e-9)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _model_canvas(ax, m, deformed=True, title=(
        f"Deformed shape (displacements ×{scale:.0f} for visibility)"
    ))
    _draw_deformed(ax, m, r, scale)
    return _save(fig, "example_deformed_shape.png")


def _diagram_overlay(ax, model, result, kind):
    """Overlay N / V / M diagram polylines on the wireframe."""
    polylines = []
    max_v = 0.0
    for e in model.elements:
        ni, nj = model.nodes[e.node_i], model.nodes[e.node_j]
        mr = result.member_results.get(e.id)
        if mr is None:
            continue
        xs, ys = sample_internal_force(e, ni, nj, list(mr["f_local"]),
                                       kind, n_samples=41)
        if xs is None or ys is None:
            continue
        polylines.append((e, ni, nj, xs, ys))
        max_v = max(max_v, max(abs(v) for v in ys))
    if max_v <= 0:
        return
    xs_n = [n.x for n in model.nodes.values()]
    ys_n = [n.y for n in model.nodes.values()]
    span = max(max(xs_n) - min(xs_n), max(ys_n) - min(ys_n))
    scale = 0.18 * span / max_v
    color = {"axial": "#5c7aff", "shear": "#0f9d58",
             "moment": "#d24c4c"}[kind]
    for e, ni, nj, xs, ys in polylines:
        L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
        if L < 1e-12:
            continue
        tx, ty = (nj.x - ni.x) / L, (nj.y - ni.y) / L
        nx, ny = -ty, tx
        bx = [ni.x + tx * x + nx * y * scale for x, y in zip(xs, ys)]
        by = [ni.y + ty * x + ny * y * scale for x, y in zip(xs, ys)]
        cx = [ni.x + tx * x for x in xs]
        cy = [ni.y + ty * x for x in xs]
        ax.fill_between(bx, by, cy, color=color, alpha=0.20)
        ax.plot(bx, by, color=color, linewidth=1.6)
        # peak label
        pk = max(range(len(ys)), key=lambda i: abs(ys[i]))
        ax.annotate(f"{ys[pk]:+.3g}", (bx[pk], by[pk]),
                    xytext=(0, 8), textcoords="offset points",
                    fontsize=7, color=color, ha="center")


def _example_diagram_figure(kind: str, name: str, caption: str):
    m, r = _load_model("inputs/example_03_portal_frame_lateral_load.txt")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _model_canvas(ax, m, title=caption)
    _diagram_overlay(ax, m, r, kind)
    return _save(fig, name)


def fig_example_axial():
    return _example_diagram_figure(
        "axial", "example_axial_diagram.png",
        "Axial force N(x) overlay — portal frame example",
    )


def fig_example_shear():
    return _example_diagram_figure(
        "shear", "example_shear_diagram.png",
        "Shear force V(x) overlay — portal frame example",
    )


def fig_example_moment():
    return _example_diagram_figure(
        "moment", "example_moment_diagram.png",
        "Bending moment M(x) overlay — portal frame example",
    )


# ── Verification figures ────────────────────────────────────────────────

def fig_verif_cantilever():
    """Verification case 1: cantilever tip load — closed-form comparison."""
    m, r = _load_model("inputs/example_01_cantilever_tip_load.txt")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    _model_canvas(ax, m, title="Case 1 — Cantilever 4 m, tip load 10 kN ↓")
    _draw_loads(ax, m)

    ax = axes[1]
    # M(x) from sample
    e = m.elements[0]
    ni, nj = m.nodes[e.node_i], m.nodes[e.node_j]
    f = r.member_results[e.id]["f_local"]
    xs, ms = sample_internal_force(e, ni, nj, list(f), "moment", n_samples=41)
    P, L = 10.0, 4.0
    M_expected = [P * (L - x) for x in xs]   # textbook M = P(L-x) sagging
    # program convention: ours uses dM/dx=V with f_local end forces.
    ax.plot(xs, ms, "o-", color="#d24c4c", label="program M(x)", markersize=3)
    ax.plot(xs, [-v for v in M_expected], "--",
            color="#444", label="closed-form −P(L−x)")
    ax.axhline(0, color="#999", linewidth=0.6)
    ax.set_xlabel("x along member (m)"); ax.set_ylabel("M (kN·m)")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8); ax.set_title("Bending moment vs. closed form")
    return _save(fig, "verification_case_1.png")


def fig_verif_simply_supported():
    """Verification case 2: simply-supported with central point load."""
    m, r = _load_model("inputs/example_02_simply_supported_point_load.txt")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    _model_canvas(ax, m,
                  title="Case 2 — Simply supported, central point load")
    _draw_loads(ax, m)

    ax = axes[1]
    e = m.elements[0]
    ni, nj = m.nodes[e.node_i], m.nodes[e.node_j]
    f = r.member_results[e.id]["f_local"]
    xs, ms = sample_internal_force(e, ni, nj, list(f), "moment", n_samples=41)
    _, vs = sample_internal_force(e, ni, nj, list(f), "shear", n_samples=41)
    ax.plot(xs, ms, color="#d24c4c", label="M(x) (kN·m)")
    ax.plot(xs, vs, color="#0f9d58", label="V(x) (kN)")
    ax.axhline(0, color="#999", linewidth=0.6)
    ax.set_xlabel("x (m)"); ax.set_ylabel("internal force")
    ax.set_title("V and M from program (textbook expected: PL/4 at mid)")
    ax.grid(linestyle=":", alpha=0.5); ax.legend(fontsize=8)
    return _save(fig, "verification_case_2.png")


def fig_verif_portal():
    """Verification case 3: portal frame, lateral load."""
    m, r = _load_model("inputs/example_03_portal_frame_lateral_load.txt")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    _model_canvas(ax, m,
                  title="Case 3 — Portal frame: lateral 50 kN + 2×(−20 kN)")
    _draw_loads(ax, m)
    _draw_deformed(ax, m, r,
                   scale=0.10 * 8.0 / max(
                       (abs(float(v)) for v in
                        (r.D if r.D is not None else [0])), default=1.0,
                   ))

    ax = axes[1]
    # show base reactions: read from r.reactions
    nodes = sorted(r.reactions.keys())
    cells = []
    headers = ["Node", "Rx (kN)", "Ry (kN)", "Mz (kN·m)"]
    for nid in nodes:
        d = r.reactions[nid]
        cells.append([str(nid),
                      f"{d.get('Rx', 0.0):+.3f}",
                      f"{d.get('Ry', 0.0):+.3f}",
                      f"{d.get('Mz', 0.0):+.3f}"])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    ax.set_title(
        f"Support reactions  (Σ Rx = {sum(r.reactions[n].get('Rx', 0) for n in nodes):+.3f} kN,"
        f"  Σ Ry = {sum(r.reactions[n].get('Ry', 0) for n in nodes):+.3f} kN)",
        fontsize=10,
    )
    return _save(fig, "verification_case_3.png")


# ── Drive all ───────────────────────────────────────────────────────────

def main():
    print(f"writing figures to {FIG_DIR.resolve()}")
    fig_architecture()
    fig_pipeline()
    fig_data_model()
    fig_gui_workflow()
    fig_example_model_and_loads()
    fig_example_deformed()
    fig_example_axial()
    fig_example_shear()
    fig_example_moment()
    fig_verif_cantilever()
    fig_verif_simply_supported()
    fig_verif_portal()
    print("done.")


if __name__ == "__main__":
    main()
