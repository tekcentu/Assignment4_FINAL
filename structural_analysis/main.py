"""
Main orchestrator: runs the full analysis pipeline (Steps A–G).
"""

from __future__ import annotations

import sys
import numpy as np

from contextlib import contextmanager

from .model import StructuralModel, AnalysisResult
from .multi_case_result import MultiCaseAnalysisResult
from .element import FrameElement2D, TrussElement2D
from .element3d import FrameElement3D, TrussElement3D
from .file_io import read_input_file
from .assembler import assemble_global_system, DofManager
from .solver import solve_system
from .postprocessor import compute_member_forces, compute_reactions, equilibrium_check


@contextmanager
def filter_loads_to_case(model: StructuralModel, case: str):
    """Temporarily restrict ``model``'s nodal + member loads to the
    given case, and toggle ``include_self_weight`` so self-weight is
    applied only for ``model.self_weight_case``. Restored on exit, even
    on exception (PR-A item 9 — never mutate the model permanently).

    Defensive ``getattr`` on ``member_loads``: today's Element2D always
    carries a default-factory ``list`` so the attribute is guaranteed,
    but the context manager guards against future element subclasses
    (or partially-constructed test fixtures) where ``member_loads`` may
    be absent or ``None``."""
    saved_nodal = model.nodal_loads
    saved_member: dict[int, list | None] = {}
    for elem in model.elements:
        saved_member[id(elem)] = getattr(elem, "member_loads", None)
    saved_sw = model.include_self_weight
    try:
        model.nodal_loads = [
            ld for ld in saved_nodal
            if getattr(ld, "load_case", "DEFAULT") == case
        ]
        for elem in model.elements:
            m_loads = saved_member[id(elem)]
            if m_loads is not None:
                elem.member_loads = [
                    ld for ld in m_loads
                    if getattr(ld, "load_case", "DEFAULT") == case
                ]
        # Self-weight is included only for the designated case.
        model.include_self_weight = (
            saved_sw and case == model.self_weight_case
        )
        yield model
    finally:
        model.nodal_loads = saved_nodal
        for elem in model.elements:
            m_loads = saved_member[id(elem)]
            if m_loads is not None:
                elem.member_loads = m_loads
        model.include_self_weight = saved_sw


def run_analysis(
    model: StructuralModel,
    verbose: bool = True,
    *,
    case: str | None = None,
) -> AnalysisResult:
    """Run the complete structural analysis (Steps A–G).

    Args:
        model: The structural model to analyse.
        verbose: If True, print formatted output to stdout.
        case: If given, filter ``model.nodal_loads`` and every
            ``elem.member_loads`` to that ``load_case`` (and toggle
            self-weight to only contribute when ``case ==
            model.self_weight_case``) for the duration of the solve.
            Restored on exit via try/finally — the model is never
            permanently mutated.

    Returns:
        AnalysisResult containing all outputs (displacements, forces,
        reactions) or partial results with error messages if analysis fails.
    """
    if case is not None:
        with filter_loads_to_case(model, case):
            return run_analysis(model, verbose=verbose, case=None)

    lines: list[str] = []

    def log(msg: str = ""):
        lines.append(msg)
        if verbose:
            print(msg)

    from .assembler import model_is_3d
    is_3d = model_is_3d(model)
    dim_tag = "3D" if is_3d else "2D"

    log("=" * 70)
    log(f"  {dim_tag} Structural Analysis — {model.title}")
    log("=" * 70)

    # ── Step A: Input ──
    log("\n── Step A: Input ──")
    n_frame = sum(1 for e in model.elements
                  if isinstance(e, (FrameElement2D, FrameElement3D)))
    n_truss = sum(1 for e in model.elements
                  if isinstance(e, (TrussElement2D, TrussElement3D)))
    n_released = sum(1 for e in model.elements
                     if isinstance(e, (FrameElement2D, FrameElement3D))
                     and (e.release_i or e.release_j))
    n_member_loaded = sum(1 for e in model.elements if e.member_loads)

    log(f"  Nodes: {len(model.nodes)}, Elements: {len(model.elements)} "
        f"(frame: {n_frame}, truss: {n_truss})")
    log(f"  Materials: {len(model.materials)}, Supports: {len(model.supports)}, "
        f"Nodal loads: {len(model.nodal_loads)}")
    if n_released:
        log(f"  Moment releases: {n_released}")
    if n_member_loaded:
        log(f"  Elements with member loads: {n_member_loaded}")

    # ── Step B + C: Assembly ──
    log("\n── Step B: Equation Numbering ──")
    try:
        K, F, dofs, warnings, elem_data = assemble_global_system(model)
    except ValueError as e:
        log(f"  ERROR: {e}")
        return AnalysisResult(status="error", title=model.title,
                              warnings=[str(e)])

    for w in warnings:
        log(f"  {w}")

    NumEq = len(dofs.free_indices)
    log(f"  Active DOFs: {dofs.n_total}, Free DOFs (NumEq): {NumEq}")

    # E matrix
    dof_headers = {"ux": "Tx", "uy": "Ty", "uz": "Tz",
                   "rx": "Rx", "ry": "Ry", "rz": "Rz"}
    E_display = dofs.e_matrix_for_display(model)
    log(f"\n  E matrix (equation numbers):")
    log("  " + f"{'Node':>6}" + "".join(
        f"  {dof_headers[d]:>6}" for d in dofs.dof_names))
    for nid in model.node_ids:
        eq = E_display[nid]
        log(f"  {nid:>6}" + "".join(f"  {v:>6}" for v in eq))

    # G vectors — built from the solve-time elements (in 3D mode the
    # promoted space elements own the 12-entry DOF map).
    log(f"\n  G vectors:")
    solve_elems = [ed["element"] for ed in elem_data.values()]
    for elem in solve_elems:
        G = dofs.g_vector_for_display(elem)
        rel = ""
        if isinstance(elem, (FrameElement2D, FrameElement3D)):
            if elem.release_i and elem.release_j:
                rel = " [both released]"
            elif elem.release_i:
                rel = " [start released]"
            elif elem.release_j:
                rel = " [end released]"
        log(f"  Elem {elem.id} ({elem.node_i}→{elem.node_j}, {elem.kind}{rel}): G = {G}")

    # ── Step C: K and F ──
    log(f"\n── Step C: Assembled Global K ({dofs.n_total}×{dofs.n_total}) ──")
    if dofs.n_total <= 15:
        log("  K =")
        for i in range(dofs.n_total):
            row = "  " + " ".join(f"{K[i,j]:12.2f}" for j in range(dofs.n_total))
            log(row)
    else:
        log(f"  (K is {dofs.n_total}×{dofs.n_total} — too large to display)")

    # K symmetry check
    asym = np.max(np.abs(K - K.T))
    log(f"\n  K symmetry check: max|K − Kᵀ| = {asym:.2e}")

    log(f"\n  Load vector F:")
    for i in range(dofs.n_total):
        log(f"    F[{i}] ({dofs.labels[i]}) = {F[i]:12.4f}")

    # ── Step D: Solve (with optional support settlements) ──
    log("\n── Step D: Solve K·D = F ──")

    # Build prescribed-displacement vector from support settlements
    D_prescribed = np.zeros(dofs.n_total)
    has_settlement = False
    for nid in model.node_ids:
        sup = model.support_for(nid)
        for dof in dofs.dof_names:
            idx = dofs.index(nid, dof)
            if idx is None:
                continue
            val = sup.prescribed(dof)
            if val != 0.0 and getattr(sup, dof):
                D_prescribed[idx] = val
                has_settlement = True
    if has_settlement:
        log("  Support settlements applied:")
        for nid in model.node_ids:
            sup = model.support_for(nid)
            for dof in dofs.dof_names:
                v = sup.prescribed(dof)
                if v != 0.0 and getattr(sup, dof):
                    log(f"    Node {nid} {dof}: {v:+.6e}")

    D, residual, solve_warnings = solve_system(K, F, dofs, D_prescribed)

    for w in solve_warnings:
        log(f"  {w}")

    if np.any(np.isnan(D)):
        log("\n*** ANALYSIS FAILED: Singular stiffness matrix. ***")
        return AnalysisResult(
            status="error", title=model.title,
            warnings=warnings + solve_warnings,
            K=K, F=F, E_map=dofs.active_map, num_eq=NumEq,
        )

    log(f"  Residual ||K_ff·D_f − F_f|| = {residual:.4e}")

    # Displacement table
    disp_units = {"ux": "m", "uy": "m", "uz": "m",
                  "rx": "rad", "ry": "rad", "rz": "rad"}
    log(f"\n  Nodal displacements:")
    log("  " + f"{'Node':>6}" + "".join(
        f"  {f'{d} ({disp_units[d]})':>14}" for d in dofs.dof_names))
    for nid in model.node_ids:
        nm = dofs.active_map[nid]
        vals = [D[nm[d]] if nm.get(d) is not None else 0.0
                for d in dofs.dof_names]
        log(f"  {nid:>6}" + "".join(f"  {v:>14.6e}" for v in vals))

    # ── Step E: Member End Forces ──
    log("\n── Step E: Member End Forces ──")
    member_results = compute_member_forces(model, D, dofs, elem_data)

    if is_3d:
        cols = ["N", "Vy", "Vz", "T", "My", "Mz"]
        log(f"  {'Elem':>6} {'Type':>7} {'End':>4}  " +
            "  ".join(f"{c:>10}" for c in cols))
        for elem in solve_elems:
            f = member_results[elem.id]["f_local"]
            half = len(f) // 2
            for end, sl in (("i", f[:half]), ("j", f[half:])):
                padded = list(sl) + [0.0] * (6 - len(sl))
                log(f"  {elem.id:>6} {elem.kind:>7} {end:>4}  " +
                    "  ".join(f"{v:>10.4f}" for v in padded))
    else:
        log(f"  {'Elem':>6} {'Type':>6}  {'N_i':>10}  {'V_i':>10}  {'M_i':>10}"
            f"  {'N_j':>10}  {'V_j':>10}  {'M_j':>10}")
        for elem in model.elements:
            f = member_results[elem.id]["f_local"]
            log(f"  {elem.id:>6} {elem.kind:>6}  " +
                "  ".join(f"{f[j]:>10.4f}" for j in range(6)))

    # ── Step F: Reactions & Equilibrium ──
    log("\n── Step F: Support Reactions ──")
    reactions = compute_reactions(model, K, D, F, dofs)

    reaction_headers = {
        "ux": "Rx (kN)", "uy": "Ry (kN)", "uz": "Rz (kN)",
        "rx": "Mx (kN·m)", "ry": "My (kN·m)", "rz": "Mz (kN·m)",
    }
    log("  " + f"{'Node':>6}" + "".join(
        f"  {reaction_headers[d]:>12}" for d in dofs.dof_names))
    for nid in sorted(reactions.keys()):
        r = reactions[nid]
        log(f"  {nid:>6}" + "".join(
            f"  {r.get(d, 0):>12.4f}" for d in dofs.dof_names))

    eq_res, eq_msgs = equilibrium_check(model, member_results, dofs,
                                        elem_data)
    for m in eq_msgs:
        log(f"  {m}")
    log(f"  Max equilibrium residual at free nodes: {eq_res:.4e}")

    # Global equilibrium
    total_rx = sum(r.get("ux", 0) for r in reactions.values()) + sum(l.fx for l in model.nodal_loads)
    total_ry = sum(r.get("uy", 0) for r in reactions.values()) + sum(l.fy for l in model.nodal_loads)
    if is_3d:
        total_rz = (sum(r.get("uz", 0) for r in reactions.values())
                    + sum(l.fz for l in model.nodal_loads))
        log(f"  Global: ΣFx = {total_rx:.4f}, ΣFy = {total_ry:.4f}, "
            f"ΣFz = {total_rz:.4f}")
    else:
        log(f"  Global: ΣFx = {total_rx:.4f}, ΣFy = {total_ry:.4f}")

    # ── Step G: Storage Report ──
    log("\n── Step G: Storage Report ──")
    n_full = dofs.n_total
    dense_count = n_full * n_full
    # Compute half-bandwidth from element DOF maps
    hbw = 0
    for eid, ed in elem_data.items():
        active = [g for g in ed["mapping"] if g is not None]
        if active:
            hbw = max(hbw, max(active) - min(active))
    banded_count = n_full * (hbw + 1)
    log(f"  Dense:  {dense_count} elements ({n_full}×{n_full})")
    log(f"  Banded: {banded_count} elements (hbw = {hbw}), "
        f"savings = {100*(1 - banded_count/max(dense_count,1)):.0f}%")
    log(f"  Note: bandwidth depends on node numbering order.")

    log("\n" + "=" * 70)
    log("  Analysis complete.")
    log("=" * 70)

    return AnalysisResult(
        status="ok", title=model.title,
        warnings=warnings + solve_warnings,
        E_map=dofs.active_map, num_eq=NumEq,
        G_vectors={e.id: dofs.g_vector_for_display(e) for e in solve_elems},
        K=K, F=F, D=D, residual=residual,
        member_results=member_results,
        reactions=reactions, eq_residual=eq_res,
        elem_data=elem_data,
    )


def run_multi_case_analysis(
    model: StructuralModel,
    verbose: bool = True,
    *,
    cases: list[str] | None = None,
    active_case: str | None = None,
) -> MultiCaseAnalysisResult:
    """Solve every requested load case independently and bundle the
    results (PR-A — v0.18).

    Args:
        model: The structural model. ``model.load_cases`` defines which
            cases exist and whether each is enabled. Any case the user
            tagged on a load that isn't in ``load_cases`` is treated as
            DEFAULT (defensive — should not happen after file_io's
            auto-create pass).
        verbose: Forwarded to each per-case ``run_analysis`` call —
            note that on a 5-case model with verbose=True you get five
            "Analysis complete" blocks; the GUI calls with False.
        cases: Explicit list of case names to solve. ``None`` →
            "every enabled case in ``model.load_cases``". Disabled
            cases are NEVER auto-included; pass an explicit list to
            override the enabled flag.
        active_case: Initial ``active_case`` on the result wrapper.
            Defaults to DEFAULT if it's in the requested set, otherwise
            the alphabetically-first solved case.

    Returns:
        :class:`MultiCaseAnalysisResult` containing one AnalysisResult
        per case that solved successfully, plus ``failed_cases`` for
        any that errored, plus ``requested_cases`` so SUM_ALL
        availability is decidable.
    """
    if cases is None:
        cases = sorted(
            name for name, lc in model.load_cases.items() if lc.enabled
        )
    # Always keep the requested-list deterministic.
    requested = list(cases)
    solved: dict[str, AnalysisResult] = {}
    failed: dict[str, str] = {}
    for name in requested:
        try:
            r = run_analysis(model, verbose=verbose, case=name)
        except Exception as e:  # noqa: BLE001
            failed[name] = f"{type(e).__name__}: {e}"
            continue
        if r.status != "ok":
            failed[name] = ", ".join(r.warnings) or "(no message)"
            continue
        solved[name] = r
    # Pick a sensible initial active_case for the GUI.
    if active_case is None:
        if "DEFAULT" in solved:
            active_case = "DEFAULT"
        elif solved:
            active_case = sorted(solved.keys())[0]
        else:
            active_case = "DEFAULT"
    return MultiCaseAnalysisResult(
        cases=solved,
        active_case=active_case,
        failed_cases=failed,
        requested_cases=requested,
    )


def run_from_file(filepath: str, verbose: bool = True) -> AnalysisResult:
    """Read an input file and run the full analysis.

    Args:
        filepath: Path to the text input file.
        verbose: If True, print formatted output to stdout.

    Returns:
        AnalysisResult with all outputs.
    """
    model = read_input_file(filepath)
    return run_analysis(model, verbose=verbose)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m structural_analysis.main <input_file>")
        sys.exit(1)
    run_from_file(sys.argv[1])
