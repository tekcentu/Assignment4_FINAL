"""Write a StructuralModel back to the text format consumed by file_io.read_input_file.

Emits the **new** MATERIALS + SECTIONS shape (which carries no A/I/depth on
materials and an explicit SECTIONS block referencing materials). Legacy
inputs that lacked a SECTIONS block are still readable by file_io thanks
to its backwards-compat shim, but the writer always produces the new
shape so saved files are explicit about the material/section separation.
"""

from __future__ import annotations

from ..element import FrameElement2D, TrussElement2D
from ..model import (
    FrameTemperatureLoad,
    PointLoad,
    StructuralModel,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)


def _fmt(x: float) -> str:
    if x == 0.0:
        return "0.0"
    # Use repr so that the value round-trips exactly through float(...).
    return repr(float(x))


def _case_token(load_case: str) -> str:
    """Return ``"  case=NAME"`` only when load_case differs from the
    default. Whitespace or ``#`` would break the reader (whitespace
    splits the row; ``#`` starts a comment) so reject those here
    rather than silently emit a corrupt file.
    """
    if not load_case or load_case == "DEFAULT":
        return ""
    if any(ch.isspace() or ch == "#" for ch in load_case):
        raise ValueError(
            f"load_case {load_case!r} contains invalid characters "
            "(whitespace or '#'); the input-file format stores it as "
            "a single token and uses '#' for comments. Rename the "
            "case (underscores or hyphens) before saving."
        )
    return f"  case={load_case}"


def _check_name(kind: str, obj_id: int, name: str) -> None:
    # The parser reads name as a single whitespace-delimited token, so any
    # embedded whitespace would silently truncate on reload.
    if name and any(ch.isspace() for ch in name):
        raise ValueError(
            f"{kind} {obj_id} name {name!r} contains whitespace; the "
            "input-file format stores name as a single token. Rename "
            "the entry (use underscores or hyphens) before saving."
        )


def write_input_file(model: StructuralModel, path: str) -> None:
    """Serialize ``model`` to ``path`` in the text format used by read_input_file.

    The format is round-trip compatible: open the file with read_input_file
    and you get an equivalent model back (modulo internal id identity).

    Raises:
        ValueError: if an element cannot be matched to any Section + Material
            combination in the model (e.g. element constructed with raw
            properties that don't appear in the section table).
    """
    out: list[str] = []

    out.append("TITLE")
    out.append(model.title or "Untitled")
    out.append("")

    node_ids = sorted(model.nodes)
    out.append(f"NODES {len(node_ids)}")
    for nid in node_ids:
        n = model.nodes[nid]
        out.append(f"{nid}  {_fmt(n.x)}  {_fmt(n.y)}")
    out.append("")

    # MATERIALS (new shape): id  E  alpha  density  [name]  [key=value ...]
    mat_ids = sorted(model.materials)
    out.append(f"MATERIALS {len(mat_ids)}")
    for mid in mat_ids:
        m = model.materials[mid]
        _check_name("Material", mid, m.name)
        line = f"{mid}  {_fmt(m.E)}  {_fmt(m.alpha)}  {_fmt(m.density)}"
        if m.name:
            line += f"  {m.name}"
        # Trailing key=value tokens. Only emitted when non-default so
        # legacy round-trips stay byte-identical.
        if m.nu != 0.0:
            line += f"  nu={_fmt(m.nu)}"
        if m.template:
            # Template is stored unquoted as a single whitespace-delimited
            # token; reject embedded whitespace so the reload can't split
            # the value across multiple positional tokens.
            if any(ch.isspace() for ch in m.template):
                raise ValueError(
                    f"Material {mid} template {m.template!r} contains "
                    "whitespace; the input-file format stores it as a "
                    "single token. Rename the template (underscores or "
                    "hyphens) before saving."
                )
            line += f"  template={m.template}"
        out.append(line)
    out.append("")

    # SECTIONS: id  material_id  A  I  depth  [name]  [key=value ...]
    sec_ids = sorted(model.sections)
    out.append(f"SECTIONS {len(sec_ids)}")
    for sid in sec_ids:
        s = model.sections[sid]
        _check_name("Section", sid, s.name)
        line = (f"{sid}  {s.material_id}  {_fmt(s.A)}  {_fmt(s.I)}"
                f"  {_fmt(s.depth)}")
        if s.name:
            line += f"  {s.name}"
        # Non-default shape data. shape_type="manual" is the default and
        # is omitted; the manual default also implies b=h=tf=tw=0.
        if s.width != 0.0:
            line += f"  width={_fmt(s.width)}"
        if s.J != 0.0:
            line += f"  J={_fmt(s.J)}"
        if s.shape_type and s.shape_type != "manual":
            line += f"  shape={s.shape_type}"
        if s.b != 0.0:
            line += f"  b={_fmt(s.b)}"
        if s.h != 0.0:
            line += f"  h={_fmt(s.h)}"
        if s.tf != 0.0:
            line += f"  tf={_fmt(s.tf)}"
        if s.tw != 0.0:
            line += f"  tw={_fmt(s.tw)}"
        out.append(line)
    out.append("")

    out.append(f"ELEMENTS {len(model.elements)}")
    for elem in model.elements:
        kind = "TRUSS" if isinstance(elem, TrussElement2D) else "FRAME"
        if elem.section_id is None:
            raise ValueError(
                f"Element {elem.id} has no section_id assigned — was it "
                "constructed without going through the model layer "
                "(file_io / AddElementCmd)? Cannot serialise."
            )
        if elem.section_id not in model.sections:
            raise ValueError(
                f"Element {elem.id} references section {elem.section_id}, "
                "which is not in the model. Cannot serialise."
            )
        line = f"{elem.id}  {elem.node_i}  {elem.node_j}  {elem.section_id}  {kind}"
        if isinstance(elem, FrameElement2D):
            if elem.release_i and elem.release_j:
                line += "  BOTH"
            elif elem.release_i:
                line += "  START"
            elif elem.release_j:
                line += "  END"
        # Per-element material override — keyword-style trailing token so
        # backward compatibility with files that don't carry it is
        # automatic. Omitted entirely when the override is None. The
        # referenced material id must still exist in MATERIALS, otherwise
        # the file we'd write could not be reloaded.
        override_id = getattr(elem, "material_id_override", None)
        if override_id is not None:
            if override_id not in model.materials:
                raise ValueError(
                    f"Element {elem.id} references material override "
                    f"{override_id}, which is not in the model. "
                    "Cannot serialise."
                )
            line += f"  material_override_id={override_id}"
        out.append(line)
    out.append("")

    if model.supports:
        sup_items = sorted(model.supports.items())
        out.append(f"SUPPORTS {len(sup_items)}")
        for nid, s in sup_items:
            line = f"{nid}  {int(s.ux)}  {int(s.uy)}  {int(s.rz)}"
            if s.settle_ux or s.settle_uy or s.settle_rz:
                line += (f"   {_fmt(s.settle_ux or 0.0)}"
                         f"  {_fmt(s.settle_uy or 0.0)}"
                         f"  {_fmt(s.settle_rz or 0.0)}")
            out.append(line)
        out.append("")

    out.append(f"LOADS {len(model.nodal_loads)}")
    for ld in model.nodal_loads:
        out.append(
            f"{ld.node_id}  {_fmt(ld.fx)}  {_fmt(ld.fy)}  "
            f"{_fmt(ld.mz)}"
            + _case_token(getattr(ld, "load_case", "DEFAULT"))
        )
    out.append("")

    udls: list[tuple[int, UniformDistributedLoad]] = []
    points: list[tuple[int, PointLoad]] = []
    truss_temps: list[tuple[int, TrussTemperatureLoad]] = []
    frame_temps: list[tuple[int, FrameTemperatureLoad]] = []
    for elem in model.elements:
        for ml in elem.member_loads:
            if isinstance(ml, UniformDistributedLoad):
                udls.append((elem.id, ml))
            elif isinstance(ml, PointLoad):
                points.append((elem.id, ml))
            elif isinstance(ml, TrussTemperatureLoad):
                truss_temps.append((elem.id, ml))
            elif isinstance(ml, FrameTemperatureLoad):
                frame_temps.append((elem.id, ml))

    if udls:
        out.append(f"MEMBER_UDL {len(udls)}")
        for eid, u in udls:
            row = f"{eid}  {_fmt(u.wx)}  {_fmt(u.wy)}"
            # Emit the coord-system token only when it differs from the
            # default. Legacy files (no global loads) round-trip
            # byte-identical to pre-v0.15.0 output.
            if u.coord_system != "local":
                row += f"  {u.coord_system}"
            row += _case_token(getattr(u, "load_case", "DEFAULT"))
            out.append(row)
        out.append("")
    if points:
        out.append(f"MEMBER_POINT_LOADS {len(points)}")
        for eid, p in points:
            row = f"{eid}  {_fmt(p.a)}  {_fmt(p.px)}  {_fmt(p.py)}"
            if p.coord_system != "local":
                row += f"  {p.coord_system}"
            row += _case_token(getattr(p, "load_case", "DEFAULT"))
            out.append(row)
        out.append("")
    if truss_temps:
        out.append(f"TRUSS_TEMPERATURE {len(truss_temps)}")
        for eid, t in truss_temps:
            out.append(
                f"{eid}  {_fmt(t.delta_T)}"
                + _case_token(getattr(t, "load_case", "DEFAULT"))
            )
        out.append("")
    if frame_temps:
        out.append(f"FRAME_TEMPERATURE {len(frame_temps)}")
        for eid, t in frame_temps:
            out.append(
                f"{eid}  {_fmt(t.t_top)}  {_fmt(t.t_bottom)}"
                + _case_token(getattr(t, "load_case", "DEFAULT"))
            )
        out.append("")

    # LOAD_CASES — only when the model carries case data that the
    # reader's auto-create pass couldn't reconstruct: extra cases
    # beyond DEFAULT, or any disabled-but-defined case. Skipping the
    # block on plain single-case models keeps every pre-v0.18 fixture's
    # round-trip byte-identical.
    #
    # SUM_ALL is intentionally never serialised: it's a derived view
    # the GUI computes on demand from the solved per-case results, not
    # a stored case. If the user (or a programmatic bug) somehow put
    # it in the dict, drop it here so it doesn't pollute the file.
    case_names_for_disk = {
        n for n in model.load_cases if n != "SUM_ALL"
    }
    needs_load_cases_block = any(
        name != "DEFAULT" or not model.load_cases[name].enabled
        for name in case_names_for_disk
    )
    if needs_load_cases_block:
        # Stable order so round-trips are deterministic; DEFAULT first
        # (when carried), then the rest alphabetically.
        ordered_names = (
            (["DEFAULT"] if "DEFAULT" in case_names_for_disk else [])
            + sorted(n for n in case_names_for_disk if n != "DEFAULT")
        )
        # Drop a trailing DEFAULT row when it's at default state
        # (enabled, no description) — the reader auto-creates it
        # anyway, so emitting it would be noise.
        def _is_default_row(name: str) -> bool:
            lc = model.load_cases[name]
            return name == "DEFAULT" and lc.enabled
        rows = [n for n in ordered_names if not _is_default_row(n)]
        if rows:
            out.append(f"LOAD_CASES {len(rows)}")
            for name in rows:
                lc = model.load_cases[name]
                row = name
                if not lc.enabled:
                    row += "  enabled=false"
                out.append(row)
            out.append("")

    # LOAD_COMBINATIONS (v0.19 — PR #29). Only emitted when the model
    # carries user-defined combinations. SUM_ALL is a derived view and
    # is NEVER written here even if it somehow appears in the dict.
    combos = {
        name: c for name, c in model.load_combinations.items()
        if name != "SUM_ALL"
    }
    if combos:
        out.append(f"LOAD_COMBINATIONS {len(combos)}")
        for name in sorted(combos):
            c = combos[name]
            # Deterministic term order so round-trips are stable.
            term_str = "  ".join(
                f"{coeff:g}*{case_name}"
                for case_name, coeff in sorted(c.terms.items())
            )
            out.append(f"{name}  {term_str}")
        out.append("")

    # ANALYSIS_OPTIONS — only when at least one option differs from
    # the default. Omitting the block on default models keeps every
    # existing fixture's round-trip byte-identical.
    opt_lines: list[str] = []
    if model.include_self_weight:
        opt_lines.append("include_self_weight=true")
    if model.self_weight_case != "DEFAULT":
        opt_lines.append(f"self_weight_case={model.self_weight_case}")
    if opt_lines:
        out.append(f"ANALYSIS_OPTIONS {len(opt_lines)}")
        out.extend(opt_lines)
        out.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out).rstrip() + "\n")
