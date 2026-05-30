"""
File I/O: text-based input parser (compatible with Assignment 2 format).

Extended with optional fields for element type, releases, and member loads.
Creates Element2D subclass instances (FrameElement2D / TrussElement2D).
"""

from __future__ import annotations

import dataclasses

from .model import (
    StructuralModel, Node, Material, Section, Support, NodalLoad,
    UniformDistributedLoad, PointLoad,
    TrussTemperatureLoad, FrameTemperatureLoad,
)
from .element import FrameElement2D, TrussElement2D


# Whitelisted trailing key=value tokens on MATERIALS / SECTIONS rows.
# Anything outside these maps raises ValueError so typos surface.
_MATERIAL_KWARG_TYPES: dict[str, type] = {"nu": float, "template": str}
_SECTION_KWARG_TYPES: dict[str, type] = {
    "width": float, "J": float,
    "shape": str,        # alias → shape_type
    "shape_type": str,
    "b": float, "h": float, "tf": float, "tw": float,
}


def _split_kwargs(parts: list[str]) -> tuple[list[str], dict[str, str]]:
    """Separate positional tokens from trailing ``key=value`` tokens.

    A ``key=value`` token is any token that contains ``=`` and starts
    with an ASCII letter (matches ``str.isalpha`` on the first char).
    Underscore-prefixed keys are intentionally not accepted — every
    whitelisted kwarg starts with a letter. Positional ordering of
    non-kwarg tokens is preserved.
    """
    positional: list[str] = []
    kwargs: dict[str, str] = {}
    for tok in parts:
        if "=" in tok and tok[:1].isalpha():
            key, _, value = tok.partition("=")
            kwargs[key] = value
        else:
            positional.append(tok)
    return positional, kwargs


def _typed_kwargs(
    raw: dict[str, str],
    whitelist: dict[str, type],
    row_kind: str,
    obj_id: int,
) -> dict[str, object]:
    """Coerce raw string kwargs to declared types; reject unknown keys."""
    typed: dict[str, object] = {}
    for k, v in raw.items():
        if k not in whitelist:
            raise ValueError(
                f"Unknown key {k!r} in {row_kind} row for id {obj_id}. "
                f"Allowed: {sorted(whitelist)}."
            )
        caster = whitelist[k]
        try:
            typed[k] = caster(v)
        except ValueError:
            raise ValueError(
                f"{row_kind} row for id {obj_id}: cannot parse "
                f"{k}={v!r} as {caster.__name__}."
            )
    # Alias: "shape" → "shape_type"
    if "shape" in typed:
        typed["shape_type"] = typed.pop("shape")
    # Material.G = E / (2 * (1 + nu)) — reject nu values that would make
    # the formula undefined or unphysical. Match the GUI invariant so a
    # hand-edited file can't slip past validation.
    if row_kind == "MATERIALS" and "nu" in typed:
        nu = typed["nu"]
        if not (0.0 <= nu < 0.5):
            raise ValueError(
                f"MATERIALS row for id {obj_id}: nu={nu!r} is outside "
                "the allowed range [0, 0.5)."
            )
    return typed


def read_input_file(filepath: str) -> StructuralModel:
    """Parse a structural model from a text input file.

    Supports sections: TITLE, NODES, MATERIALS, SECTIONS, ELEMENTS,
    SUPPORTS, LOADS, MEMBER_POINT_LOADS, MEMBER_UDL, TRUSS_TEMPERATURE,
    FRAME_TEMPERATURE. Lines starting with # are comments. Element lines
    accept optional type (FRAME/TRUSS) and release (START/END/BOTH).

    Two MATERIALS shapes are supported:

    - **New shape** (paired with a SECTIONS block):
      ``<id>  <E>  [alpha]  [name]``
    - **Legacy shape** (Assignment 2/3/4 compatibility — no SECTIONS block):
      ``<id>  <A>  <I>  <E>  [alpha]  [depth]``
      Detected automatically when a SECTIONS block is absent. The parser
      synthesises a 1:1 :class:`Section` for each legacy row so existing
      ``inputs/q2*.txt`` files load unchanged.

    Args:
        filepath: Path to the input file.

    Returns:
        A populated StructuralModel with Element2D subclass instances.
    """
    model = StructuralModel()

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]

    # First pass: does this file declare a SECTIONS block? That decides
    # whether MATERIALS rows are the new (id, E, …) or legacy (id, A, I, E, …)
    # shape.
    has_sections_block = any(
        ln.split("#")[0].strip().split()[:1] == ["SECTIONS"]
        for ln in lines if ln and not ln.startswith("#")
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or line.startswith("#"):
            i += 1
            continue

        tokens = line.split()
        keyword = tokens[0].upper()

        if keyword == "TITLE":
            i += 1
            model.title = lines[i] if i < len(lines) else "Untitled"

        elif keyword == "NODES":
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                parts = lines[i].split("#")[0].split()
                nid = int(parts[0])
                model.nodes[nid] = Node(nid, float(parts[1]), float(parts[2]))

        elif keyword == "MATERIALS":
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                parts = lines[i].split("#")[0].split()
                parts, mat_kwargs = _split_kwargs(parts)
                mid = int(parts[0])
                if has_sections_block:
                    # New shape: id  E  <alpha>  <density>  [name]
                    # The four leading numeric columns are positional.
                    # Legacy files written before the density column can
                    # also be loaded: if parts[3] is non-numeric (a name)
                    # density defaults to 0.0 and parts[3] is treated as
                    # the name token.
                    E_val = float(parts[1])
                    if len(parts) > 2:
                        try:
                            alpha = float(parts[2])
                        except ValueError:
                            raise ValueError(
                                f"MATERIALS row for id {mid}: expected a numeric "
                                f"thermal-expansion coefficient (alpha) in column 3, "
                                f"got {parts[2]!r}. The new MATERIALS shape is "
                                f"'id E alpha [density] [name]'; tokens are positional."
                            )
                    else:
                        alpha = 0.0

                    density = 0.0
                    name_start = 3
                    if len(parts) > 3:
                        try:
                            density = float(parts[3])
                            name_start = 4
                        except ValueError:
                            # parts[3] is non-numeric → legacy file: treat
                            # it as the name token and leave density = 0.
                            density = 0.0
                            name_start = 3
                    name = parts[name_start] if len(parts) > name_start else ""
                    mat = Material(id=mid, name=name,
                                    E=E_val, alpha=alpha,
                                    density=density)
                    if mat_kwargs:
                        mat = dataclasses.replace(
                            mat,
                            **_typed_kwargs(mat_kwargs, _MATERIAL_KWARG_TYPES,
                                            "MATERIALS", mid),
                        )
                    model.materials[mid] = mat
                else:
                    # Legacy shape: id  A  I  E  [alpha]  [depth]
                    # Synthesise a 1:1 Material+Section pair so existing
                    # inputs (q2a, q2b, course examples) load unchanged.
                    A_val = float(parts[1])
                    I_val = float(parts[2])
                    E_val = float(parts[3])
                    alpha = float(parts[4]) if len(parts) > 4 else 0.0
                    depth = float(parts[5]) if len(parts) > 5 else 0.0
                    mat = Material(id=mid, E=E_val, alpha=alpha)
                    if mat_kwargs:
                        mat = dataclasses.replace(
                            mat,
                            **_typed_kwargs(mat_kwargs, _MATERIAL_KWARG_TYPES,
                                            "MATERIALS", mid),
                        )
                    model.materials[mid] = mat
                    model.sections[mid] = Section(
                        id=mid, material_id=mid,
                        A=A_val, I=I_val, depth=depth,
                    )

        elif keyword == "SECTIONS":
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                parts = lines[i].split("#")[0].split()
                parts, sec_kwargs = _split_kwargs(parts)
                sid = int(parts[0])
                material_id = int(parts[1])
                A_val = float(parts[2])
                I_val = float(parts[3])
                depth = float(parts[4]) if len(parts) > 4 else 0.0
                name = parts[5] if len(parts) > 5 else ""
                sec = Section(
                    id=sid, name=name, material_id=material_id,
                    A=A_val, I=I_val, depth=depth,
                )
                if sec_kwargs:
                    sec = dataclasses.replace(
                        sec,
                        **_typed_kwargs(sec_kwargs, _SECTION_KWARG_TYPES,
                                        "SECTIONS", sid),
                    )
                model.sections[sid] = sec

        elif keyword == "ELEMENTS":
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                parts = lines[i].split("#")[0].split()
                eid = int(parts[0])
                sn, en = int(parts[1]), int(parts[2])
                # Column 4 historically referenced a Material id (legacy
                # combined storage). With the Material/Section split, it now
                # references the Section id. The 1:1 synthesis in the
                # MATERIALS shim above means legacy files still resolve
                # correctly: section_id == legacy_material_id.
                ref_id = int(parts[3])
                section = model.sections.get(ref_id)
                if section is None:
                    raise ValueError(
                        f"Element {eid} references section/material id "
                        f"{ref_id}, which has no SECTIONS entry."
                    )
                mat = model.materials.get(section.material_id)
                if mat is None:
                    raise ValueError(
                        f"Section {section.id} references material id "
                        f"{section.material_id}, which has no MATERIALS entry."
                    )

                # Optional element type. Only the value at position 5 is
                # treated as a positional kind token; once we see a key=value
                # pair we stop consuming positional tokens.
                etype = "FRAME"
                if len(parts) >= 5 and "=" not in parts[4]:
                    etype = parts[4].upper()

                # Optional release at position 6.
                release_i = False
                release_j = False
                if len(parts) >= 6 and "=" not in parts[5]:
                    r = parts[5].upper()
                    if r == "START":
                        release_i = True
                    elif r == "END":
                        release_j = True
                    elif r == "BOTH":
                        release_i = True
                        release_j = True

                # Trailing key=value kwargs. Currently only
                # ``material_override_id`` is recognised; other unknown
                # keys are rejected so typos surface immediately rather
                # than silently being ignored. Positional tokens are only
                # permitted at idx 4 (kind) and idx 5 (release) — any
                # later non-``key=value`` token is an error so typos like
                # ``material_override_id 2`` don't slip through silently.
                material_override_id: int | None = None
                for idx, tok in enumerate(parts[4:], start=4):
                    if "=" not in tok:
                        positional_slot = (
                            idx == 4
                            or (idx == 5 and "=" not in parts[4])
                        )
                        if positional_slot:
                            continue
                        raise ValueError(
                            f"Element {eid}: unexpected positional token "
                            f"{tok!r}. After the optional kind/release "
                            "tokens, all element options must be key=value "
                            "pairs (e.g. material_override_id=2)."
                        )
                    key, _, value = tok.partition("=")
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "material_override_id":
                        try:
                            mid = int(value)
                        except ValueError:
                            raise ValueError(
                                f"Element {eid}: material_override_id must "
                                f"be an integer, got {value!r}."
                            )
                        if mid not in model.materials:
                            raise ValueError(
                                f"Element {eid}: material_override_id={mid} "
                                "has no MATERIALS entry."
                            )
                        material_override_id = mid
                    else:
                        raise ValueError(
                            f"Element {eid}: unknown element option "
                            f"{tok!r}; expected material_override_id=<id>."
                        )

                # Resolve the *effective* material for E / α / ρ. Geometry
                # (A, I, depth) always comes from the section.
                if material_override_id is not None:
                    eff_mat = model.materials[material_override_id]
                else:
                    eff_mat = mat

                if etype == "TRUSS":
                    elem = TrussElement2D(
                        id=eid, node_i=sn, node_j=en,
                        E=eff_mat.E, A=section.A,
                        alpha=eff_mat.alpha, depth=section.depth,
                        rho=eff_mat.density,
                        section_id=section.id,
                        material_id_override=material_override_id,
                    )
                else:
                    elem = FrameElement2D(
                        id=eid, node_i=sn, node_j=en,
                        E=eff_mat.E, A=section.A, I=section.I,
                        alpha=eff_mat.alpha, depth=section.depth,
                        rho=eff_mat.density,
                        section_id=section.id,
                        material_id_override=material_override_id,
                        release_i=release_i, release_j=release_j,
                    )
                model.elements.append(elem)

        elif keyword == "SUPPORTS":
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                parts = lines[i].split("#")[0].split()
                nid = int(parts[0])
                ux_r = bool(int(parts[1]))
                uy_r = bool(int(parts[2]))
                rz_r = bool(int(parts[3]))
                # Optional settlement fields: settle_ux settle_uy settle_rz
                s_ux = float(parts[4]) if len(parts) > 4 else None
                s_uy = float(parts[5]) if len(parts) > 5 else None
                s_rz = float(parts[6]) if len(parts) > 6 else None
                model.supports[nid] = Support(
                    nid, ux_r, uy_r, rz_r,
                    settle_ux=s_ux if (s_ux and s_ux != 0.0) else None,
                    settle_uy=s_uy if (s_uy and s_uy != 0.0) else None,
                    settle_rz=s_rz if (s_rz and s_rz != 0.0) else None,
                )

        elif keyword == "LOADS":
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                parts = lines[i].split("#")[0].split()
                model.nodal_loads.append(NodalLoad(
                    int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                ))

        elif keyword == "MEMBER_POINT_LOADS":
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                parts = lines[i].split("#")[0].split()
                eid = int(parts[0])
                a = float(parts[1])
                px = float(parts[2]) if len(parts) > 2 else 0.0
                py = float(parts[3]) if len(parts) > 3 else 0.0
                # Optional trailing coord-system token: "local" (default
                # if absent) or "global". Anything else is rejected so a
                # typo doesn't silently parse as the default.
                coord_system = _parse_coord_system_token(
                    parts[4] if len(parts) > 4 else None, "MEMBER_POINT_LOADS",
                )
                for elem in model.elements:
                    if elem.id == eid:
                        elem.member_loads.append(PointLoad(
                            py=py, a=a, px=px, coord_system=coord_system,
                        ))
                        break

        elif keyword == "MEMBER_UDL":
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                parts = lines[i].split("#")[0].split()
                eid = int(parts[0])
                wx = float(parts[1]) if len(parts) > 1 else 0.0
                wy = float(parts[2]) if len(parts) > 2 else 0.0
                # Optional trailing coord-system token; missing → local.
                coord_system = _parse_coord_system_token(
                    parts[3] if len(parts) > 3 else None, "MEMBER_UDL",
                )
                for elem in model.elements:
                    if elem.id == eid:
                        elem.member_loads.append(UniformDistributedLoad(
                            wy=wy, wx=wx, coord_system=coord_system,
                        ))
                        break

        elif keyword == "TRUSS_TEMPERATURE":
            # Format: elem_id  delta_T
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                parts = lines[i].split("#")[0].split()
                eid = int(parts[0])
                dT = float(parts[1])
                for elem in model.elements:
                    if elem.id == eid:
                        elem.member_loads.append(TrussTemperatureLoad(delta_T=dT))
                        break

        elif keyword == "FRAME_TEMPERATURE":
            # Format: elem_id  t_top  t_bottom
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                parts = lines[i].split("#")[0].split()
                eid = int(parts[0])
                t_top = float(parts[1])
                t_bottom = float(parts[2])
                for elem in model.elements:
                    if elem.id == eid:
                        elem.member_loads.append(FrameTemperatureLoad(
                            t_top=t_top, t_bottom=t_bottom,
                        ))
                        break

        elif keyword == "ANALYSIS_OPTIONS":
            # Format: ANALYSIS_OPTIONS <count> followed by count
            # key=value lines. v0.9.0 recognises only
            # ``include_self_weight=<bool>``; unknown keys raise
            # ``ValueError`` so typos surface (mirrors the per-element
            # override-key strictness already in this parser).
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                if i >= len(lines):
                    raise ValueError(
                        "Unexpected end of file inside ANALYSIS_OPTIONS "
                        f"block (expected {count} key=value rows)."
                    )
                opt_line = lines[i].split("#")[0].strip()
                if "=" not in opt_line:
                    raise ValueError(
                        f"ANALYSIS_OPTIONS row {opt_line!r} is not a "
                        "key=value pair."
                    )
                key, _, val = opt_line.partition("=")
                key = key.strip().lower()
                val = val.strip()
                if key == "include_self_weight":
                    model.include_self_weight = _parse_bool(val, key)
                else:
                    raise ValueError(
                        f"Unknown ANALYSIS_OPTIONS key {key!r}. "
                        f"Allowed: ['include_self_weight']."
                    )

        i += 1

    return model


def _parse_bool(s: str, key: str) -> bool:
    low = s.lower()
    if low in ("true", "1", "yes", "on"):
        return True
    if low in ("false", "0", "no", "off"):
        return False
    raise ValueError(
        f"ANALYSIS_OPTIONS {key}={s!r}: expected a boolean "
        "(true/false, 1/0, yes/no, on/off)."
    )


def _parse_coord_system_token(token: str | None, section: str) -> str:
    """Parse the optional trailing coord-system token on a member-load row.

    Missing token → ``"local"`` (preserves byte-identical parsing of
    every pre-v0.15.0 input file). The accepted explicit values are
    ``"local"``, ``"global"``, and ``"gravity"`` (v0.16.0); anything
    else raises so a typo doesn't silently degrade to the default.
    """
    if token is None:
        return "local"
    norm = token.strip().lower()
    if norm in ("local", "global", "gravity"):
        return norm
    raise ValueError(
        f"{section}: unknown coord-system token {token!r}; "
        "expected 'local', 'global', or 'gravity' (or omit for local)."
    )
