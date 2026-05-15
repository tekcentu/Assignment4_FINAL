"""Render an AnalysisResult as a human-readable text report (no stdout capture)."""

from __future__ import annotations

from ..element import FrameElement2D
from ..model import AnalysisResult, StructuralModel


def format_result(model: StructuralModel, result: AnalysisResult | None) -> str:
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
    lines.append(f"  {'Node':>6}  {'ux (m)':>14}  {'uy (m)':>14}  {'rz (rad)':>14}")
    D = result.D
    for nid in sorted(result.E_map):
        em = result.E_map[nid]
        ux = float(D[em["ux"]]) if em["ux"] is not None else 0.0
        uy = float(D[em["uy"]]) if em["uy"] is not None else 0.0
        rz = float(D[em["rz"]]) if em["rz"] is not None else 0.0
        lines.append(f"  {nid:>6}  {ux:>14.6e}  {uy:>14.6e}  {rz:>14.6e}")

    # Step E
    lines.append("\n── Step E: Member End Forces ──")
    lines.append(f"  {'Elem':>6} {'Type':>6}  "
                 f"{'N_i':>10}  {'V_i':>10}  {'M_i':>10}  "
                 f"{'N_j':>10}  {'V_j':>10}  {'M_j':>10}")
    for elem in model.elements:
        mr = result.member_results.get(elem.id)
        if mr is None:
            continue
        f_local = mr["f_local"]
        kind = "frame" if isinstance(elem, FrameElement2D) else "truss"
        lines.append(f"  {elem.id:>6} {kind:>6}  " +
                     "  ".join(f"{float(f_local[j]):>10.4f}" for j in range(6)))

    # Step F
    lines.append("\n── Step F: Support Reactions ──")
    lines.append(f"  {'Node':>6}  {'Rx (kN)':>12}  {'Ry (kN)':>12}  {'Mz (kN·m)':>12}")
    for nid in sorted(result.reactions):
        r = result.reactions[nid]
        lines.append(f"  {nid:>6}  {r.get('ux', 0):>12.4f}  "
                     f"{r.get('uy', 0):>12.4f}  {r.get('rz', 0):>12.4f}")
    lines.append(f"  Max equilibrium residual at free nodes: {result.eq_residual:.4e}")

    lines.append("\n" + "=" * 70)
    lines.append("  Analysis complete.")
    lines.append("=" * 70)
    return "\n".join(lines)
