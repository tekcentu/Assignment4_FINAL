"""Render an AnalysisResult as a human-readable text report (no stdout capture).

Also exposes Qt-free helpers used by both the main-window toolbar combo and
the per-element inspector's local result selector to keep their populate /
resolve logic in one place (so labels and SUM_ALL / combination rules can
never drift between them).
"""

from __future__ import annotations

import re

from ..element import FrameElement2D
from ..model import AnalysisResult, StructuralModel
from ..multi_case_result import MultiCaseAnalysisResult, SUM_ALL_KEY
from . import units as _units
from .validation import used_case_names


_COMPONENT_WORD_RE = re.compile(r"\bcomponents?\b", re.IGNORECASE)


def _component_to_structure_word(match: re.Match) -> str:
    """Map one ``component``/``components`` match to ``structure``/``structures``
    while preserving the matched casing (UPPER, Title, or lower)."""
    word = match.group(0)
    plural = word[-1] in ("s", "S")
    if word.isupper():
        return "STRUCTURES" if plural else "STRUCTURE"
    base = "Structure" if word[:1].isupper() else "structure"
    return base + ("s" if plural else "")


def relabel_component_to_structure(text: str) -> str:
    """Convert user-facing 'component' wording to 'structure' at the display
    boundary.

    The solver layer (``modal.py``) speaks of "components" internally; the GUI
    shows disconnected structures as "Structure N" to match structural-
    engineering language. This helper does a whole-word, case-aware swap so
    solver-sourced prose (e.g. ``ModalResult.component_summary``) can be shown
    in user terms WITHOUT touching the solver. Only the surface forms
    ``Component`` / ``component`` / ``COMPONENT`` / ``Components`` /
    ``components`` / ``COMPONENTS`` are mapped; substrings like ``componentry``
    are left untouched by the ``\\b`` guards.
    """
    if not text:
        return text
    return _COMPONENT_WORD_RE.sub(_component_to_structure_word, text)



def case_combo_entries(
    model: StructuralModel,
    multi_result: MultiCaseAnalysisResult | None,
) -> list[tuple[str, str]]:
    """Build the ``(display_label, raw_name)`` pairs for a case/combo combo.

    Order matches the existing toolbar populate logic:

    1. ``DEFAULT`` first (when present), then other real cases sorted by name,
       each shown as ``"<name>"`` or ``"<name>  (disabled)"``.
    2. ``SUM_ALL`` — appended only when every requested case solved and at
       least two cases are present in ``multi_result.cases``.
    3. User-defined combinations sorted by name, labelled ``"<name>  [comb]"``
       or ``"<name>  [comb · needs solve]"`` when any referenced case is
       unsolved.

    The ``raw_name`` is always the bare identifier (case name, ``SUM_ALL``,
    or combination name) and is what the caller stores in ``QComboBox``
    userData and feeds to :func:`resolve_view`.  This keeps the display
    label and the internal key strictly separated.
    """
    entries: list[tuple[str, str]] = []
    ordered = (
        (["DEFAULT"] if "DEFAULT" in model.load_cases else [])
        + sorted(n for n in model.load_cases if n != "DEFAULT")
    )
    # Enabled cases that carry no load source are tagged
    # "(no loads assigned)" so they don't masquerade as ordinary unsolved
    # result cases — Solve All skips them, and the user sees why.
    used = used_case_names(model)
    for name in ordered:
        lc = model.load_cases[name]
        if not lc.enabled:
            label = f"{name}  (disabled)"
        elif name not in used:
            label = f"{name}  (no loads assigned)"
        else:
            label = name
        entries.append((label, name))
    if (
        multi_result is not None
        and multi_result.sum_all_available()
        and len(multi_result.cases) >= 2
    ):
        entries.append((SUM_ALL_KEY, SUM_ALL_KEY))
    for comb_name in sorted(model.load_combinations):
        comb = model.load_combinations[comb_name]
        available = (
            multi_result is not None
            and multi_result.combination_available(comb.terms)
        )
        label = (
            f"{comb_name}  [comb]" if available
            else f"{comb_name}  [comb · needs solve]"
        )
        entries.append((label, comb_name))
    return entries


def resolve_view(
    model: StructuralModel,
    multi_result: MultiCaseAnalysisResult | None,
    name: str,
) -> tuple[AnalysisResult | None, str]:
    """Resolve a raw case / SUM_ALL / combination identifier to a result.

    Returns a ``(result, status_msg)`` tuple where ``status_msg`` is empty
    on success and a short human-readable reason otherwise — ready to drop
    onto a placeholder axis when the diagrams panel has nothing to draw.

    Mirrors :meth:`MainWindow._resolve_active_result` but takes the model
    + multi_result + name as explicit arguments so callers (dialogs, the
    main window, future side-panels) all share the same routing.
    """
    if multi_result is None:
        return None, "No analysis results yet. Run analysis to show N/V/M diagrams."
    if name in model.load_combinations:
        comb = model.load_combinations[name]
        result = multi_result.combination(comb.terms, name=name)
        if result is None:
            missing = multi_result.missing_cases_for(comb.terms)
            return None, (
                f"Combination '{name}' needs solve: "
                f"missing {', '.join(missing)}"
                if missing else
                f"Combination '{name}' needs solve."
            )
        return result, ""
    if name == SUM_ALL_KEY:
        result = multi_result.sum_all()
        if result is None:
            return None, "SUM_ALL needs every requested case solved."
        return result, ""
    if name in multi_result.failed_cases:
        reason = multi_result.failed_cases[name]
        return None, f"Case '{name}' failed: {reason}"
    result = multi_result.get(name)
    if result is None:
        return None, f"Case '{name}' has not been solved yet."
    return result, ""


def format_result(
    model: StructuralModel,
    result: AnalysisResult | None,
    *,
    unit_preset: str = _units.DEFAULT_PRESET_ID,
) -> str:
    """Render an AnalysisResult as a text report.

    ``unit_preset`` selects the display preset (force × length) from the
    Global Units V1 helper. The default is ``"kN_m"``, which reproduces
    the legacy bytes exactly so older callers (CLI, existing tests) see
    no change.
    """
    length_lbl = _units.length_label(unit_preset)
    force_lbl = _units.force_label(unit_preset)
    moment_lbl = _units.moment_label(unit_preset)
    if result is None:
        return "(no analysis run yet)"
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"  2D Structural Analysis — {result.title or model.title}")
    lines.append("=" * 70)
    if result.warnings:
        lines.append("\n  Warnings:")
        for w in result.warnings:
            lines.append(f"    - {w}")
    if result.status != "ok":
        lines.append(f"\n  Status: {result.status}")
        return "\n".join(lines)

    # ── Summary (top-of-report quick read) ──
    # Scan D through E_map for the largest translational displacement
    # magnitude across all nodes so a beginner sees "the answer" before
    # the step-by-step breakdown.
    max_disp = 0.0
    max_disp_node: int | None = None
    has_disp_dofs = False
    if result.D is not None:
        for nid, em in result.E_map.items():
            if em["ux"] is None and em["uy"] is None:
                continue
            has_disp_dofs = True
            ux = float(result.D[em["ux"]]) if em["ux"] is not None else 0.0
            uy = float(result.D[em["uy"]]) if em["uy"] is not None else 0.0
            mag = (ux * ux + uy * uy) ** 0.5
            if mag >= max_disp:
                max_disp = mag
                max_disp_node = nid
    lines.append("\n── Summary ──")
    lines.append(f"  Status:                      {result.status}")
    lines.append(f"  Max equilibrium residual:    {result.eq_residual:.4e}")
    if not has_disp_dofs:
        lines.append("  Max nodal displacement:      (no displacement DOFs)")
    elif max_disp_node is not None:
        disp_disp = _units.length_to_display(max_disp, unit_preset)
        lines.append(
            f"  Max nodal displacement:      |u| = {disp_disp:.4e} {length_lbl} "
            f"at node {max_disp_node}"
        )

    # Step A
    n_frame = sum(1 for e in model.elements if isinstance(e, FrameElement2D))
    n_truss = len(model.elements) - n_frame
    lines.append("\n── Step A: Input ──")
    lines.append(f"  Nodes: {len(model.nodes)}, Elements: {len(model.elements)} "
                 f"(frame: {n_frame}, truss: {n_truss})")
    lines.append(f"  Materials: {len(model.materials)}, Supports: {len(model.supports)}, "
                 f"Nodal loads: {len(model.nodal_loads)}")

    # Step B
    # Note: values printed are 0-based global DOF indices from the active_map.
    # A value of None means the DOF is either restrained at a support ("fix")
    # or inactive (e.g. the rotational DOF of a node connected only to
    # truss elements — labelled "—").
    lines.append("\n── Step B: Global DOF Indices ──")
    lines.append(f"  Active DOFs: {sum(1 for em in result.E_map.values() for v in em.values() if v is not None)}, "
                 f"Free DOFs (NumEq): {result.num_eq}")
    lines.append(f"  {'Node':>6}  {'Tx':>6}  {'Ty':>6}  {'Rz':>6}"
                 f"   (index = global DOF #; fix = restrained; — = inactive)")
    for nid in sorted(result.E_map):
        em = result.E_map[nid]
        sup = model.support_for(nid)
        def f(v, dof):
            if v is not None:
                return str(v)
            return "fix" if getattr(sup, dof) else "—"
        lines.append(f"  {nid:>6}  {f(em['ux'],'ux'):>6}  "
                     f"{f(em['uy'],'uy'):>6}  {f(em['rz'],'rz'):>6}")

    # Step D
    lines.append("\n── Step D: Solve K·D = F ──")
    lines.append(f"  Residual ||K_ff·D_f − F_f|| = {result.residual:.4e}")
    lines.append(f"\n  Nodal displacements:")
    h_ux = f"ux ({length_lbl})"
    h_uy = f"uy ({length_lbl})"
    lines.append(f"  {'Node':>6}  {h_ux:>14}  {h_uy:>14}  {'rz (rad)':>14}")
    D = result.D
    for nid in sorted(result.E_map):
        em = result.E_map[nid]
        ux = float(D[em["ux"]]) if em["ux"] is not None else 0.0
        uy = float(D[em["uy"]]) if em["uy"] is not None else 0.0
        rz = float(D[em["rz"]]) if em["rz"] is not None else 0.0
        ux_d = _units.length_to_display(ux, unit_preset)
        uy_d = _units.length_to_display(uy, unit_preset)
        lines.append(f"  {nid:>6}  {ux_d:>14.6e}  {uy_d:>14.6e}  {rz:>14.6e}")

    # Step E
    lines.append("\n── Step E: Member End Forces ──")
    lines.append(f"  {'Elem':>6} {'Type':>6}  "
                 f"{'N_i ' + force_lbl:>10}  {'V_i ' + force_lbl:>10}  "
                 f"{'M_i ' + moment_lbl:>14}  "
                 f"{'N_j ' + force_lbl:>10}  {'V_j ' + force_lbl:>10}  "
                 f"{'M_j ' + moment_lbl:>14}")
    for elem in model.elements:
        mr = result.member_results.get(elem.id)
        if mr is None:
            continue
        f_local = mr["f_local"]
        kind = "frame" if isinstance(elem, FrameElement2D) else "truss"
        # f_local layout: [N_i, V_i, M_i, N_j, V_j, M_j]
        vals = [
            _units.force_to_display(float(f_local[0]), unit_preset),
            _units.force_to_display(float(f_local[1]), unit_preset),
            _units.moment_to_display(float(f_local[2]), unit_preset),
            _units.force_to_display(float(f_local[3]), unit_preset),
            _units.force_to_display(float(f_local[4]), unit_preset),
            _units.moment_to_display(float(f_local[5]), unit_preset),
        ]
        widths = (10, 10, 14, 10, 10, 14)
        lines.append(
            f"  {elem.id:>6} {kind:>6}  " +
            "  ".join(f"{v:>{w}.4f}" for v, w in zip(vals, widths))
        )

    # Step F
    lines.append("\n── Step F: Support Reactions ──")
    h_rx = f"Rx ({force_lbl})"
    h_ry = f"Ry ({force_lbl})"
    h_mz = f"Mz ({moment_lbl})"
    lines.append(f"  {'Node':>6}  {h_rx:>12}  {h_ry:>12}  {h_mz:>14}")
    for nid in sorted(result.reactions):
        r = result.reactions[nid]
        rx = _units.force_to_display(r.get('ux', 0), unit_preset)
        ry = _units.force_to_display(r.get('uy', 0), unit_preset)
        mz = _units.moment_to_display(r.get('rz', 0), unit_preset)
        lines.append(f"  {nid:>6}  {rx:>12.4f}  {ry:>12.4f}  {mz:>14.4f}")
    lines.append(f"  Max equilibrium residual at free nodes: {result.eq_residual:.4e}")

    lines.append("\n" + "=" * 70)
    lines.append("  Analysis complete.")
    lines.append("=" * 70)
    return "\n".join(lines)
