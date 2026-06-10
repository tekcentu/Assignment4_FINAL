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
    LoadCase, LoadCombination,
    JointMass, ModalMassSource,
)
from .element import FrameElement2D, MIN_FLEXIBLE_LENGTH, TrussElement2D


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

                # Trailing key=value kwargs. Recognised keys:
                # ``material_override_id``, ``offset_i``, ``offset_j``
                # (rigid end offsets, v0.31.0); other unknown keys are
                # rejected so typos surface immediately rather than
                # silently being ignored. Positional tokens are only
                # permitted at idx 4 (kind) and idx 5 (release) — any
                # later non-``key=value`` token is an error so typos like
                # ``material_override_id 2`` don't slip through silently.
                material_override_id: int | None = None
                offset_i = 0.0
                offset_j = 0.0
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
                    elif key in ("offset_i", "offset_j"):
                        try:
                            off = float(value)
                        except ValueError:
                            raise ValueError(
                                f"Element {eid}: {key} must be a number, "
                                f"got {value!r}."
                            )
                        if off < 0.0:
                            raise ValueError(
                                f"Element {eid}: {key}={off:g} — rigid end "
                                "offsets must be >= 0."
                            )
                        if key == "offset_i":
                            offset_i = off
                        else:
                            offset_j = off
                    else:
                        raise ValueError(
                            f"Element {eid}: unknown element option "
                            f"{tok!r}; expected material_override_id=<id>, "
                            "offset_i=<m> or offset_j=<m>."
                        )

                if (offset_i or offset_j) and etype == "TRUSS":
                    raise ValueError(
                        f"Element {eid}: rigid end offsets are only "
                        "supported on FRAME elements."
                    )
                if offset_i or offset_j:
                    n_i = model.nodes.get(sn)
                    n_j = model.nodes.get(en)
                    if n_i is not None and n_j is not None:
                        L_tot = ((n_j.x - n_i.x) ** 2
                                 + (n_j.y - n_i.y) ** 2) ** 0.5
                        if offset_i + offset_j > L_tot - MIN_FLEXIBLE_LENGTH:
                            raise ValueError(
                                f"Element {eid}: offset_i + offset_j = "
                                f"{offset_i + offset_j:g} m >= member length "
                                f"{L_tot:g} m — the flexible span must be "
                                f"at least {MIN_FLEXIBLE_LENGTH:g} m."
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
                        offset_i=offset_i, offset_j=offset_j,
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
                meta = _parse_load_metadata(
                    parts, start_idx=4, section="LOADS",
                    accept_coord_system=False,
                )
                model.nodal_loads.append(NodalLoad(
                    int(parts[0]), float(parts[1]), float(parts[2]),
                    float(parts[3]), load_case=meta["load_case"],
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
                # Optional trailing tokens: positional coord-system
                # ("local"|"global"|"gravity") followed by any number of
                # key=value tokens (today: ``case=NAME``). Missing →
                # local + DEFAULT, preserving byte-identical parsing of
                # every pre-v0.17.0 input file.
                #
                # The metadata block can start anywhere after the
                # mandatory ``elem_id  a`` pair (px and py are optional
                # — they default to 0). Locate the first metadata token
                # by signature ('=' for key=value, or a known
                # coord_system keyword) so we don't try to parse
                # ``case=DEAD`` as a float.
                start_idx = _find_metadata_start(
                    parts, mandatory_end=2, optional_count=2,
                )
                px = float(parts[2]) if start_idx > 2 else 0.0
                py = float(parts[3]) if start_idx > 3 else 0.0
                meta = _parse_load_metadata(
                    parts, start_idx=start_idx, section="MEMBER_POINT_LOADS",
                    accept_coord_system=True,
                )
                for elem in model.elements:
                    if elem.id == eid:
                        elem.member_loads.append(PointLoad(
                            py=py, a=a, px=px,
                            coord_system=meta["coord_system"],
                            load_case=meta["load_case"],
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
                # Same dynamic-metadata-start logic as
                # MEMBER_POINT_LOADS above — wx and wy are optional but
                # the trailing tokens (coord_system / case=) must not
                # be parsed as floats.
                start_idx = _find_metadata_start(
                    parts, mandatory_end=1, optional_count=2,
                )
                wx = float(parts[1]) if start_idx > 1 else 0.0
                wy = float(parts[2]) if start_idx > 2 else 0.0
                meta = _parse_load_metadata(
                    parts, start_idx=start_idx, section="MEMBER_UDL",
                    accept_coord_system=True,
                )
                for elem in model.elements:
                    if elem.id == eid:
                        elem.member_loads.append(UniformDistributedLoad(
                            wy=wy, wx=wx,
                            coord_system=meta["coord_system"],
                            load_case=meta["load_case"],
                        ))
                        break

        elif keyword == "TRUSS_TEMPERATURE":
            # Format: elem_id  delta_T  [case=NAME]
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                parts = lines[i].split("#")[0].split()
                eid = int(parts[0])
                dT = float(parts[1])
                meta = _parse_load_metadata(
                    parts, start_idx=2, section="TRUSS_TEMPERATURE",
                    accept_coord_system=False,
                )
                for elem in model.elements:
                    if elem.id == eid:
                        elem.member_loads.append(TrussTemperatureLoad(
                            delta_T=dT, load_case=meta["load_case"],
                        ))
                        break

        elif keyword == "FRAME_TEMPERATURE":
            # Format: elem_id  t_top  t_bottom  [case=NAME]
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (not lines[i] or lines[i].startswith("#")):
                    i += 1
                parts = lines[i].split("#")[0].split()
                eid = int(parts[0])
                t_top = float(parts[1])
                t_bottom = float(parts[2])
                meta = _parse_load_metadata(
                    parts, start_idx=3, section="FRAME_TEMPERATURE",
                    accept_coord_system=False,
                )
                for elem in model.elements:
                    if elem.id == eid:
                        elem.member_loads.append(FrameTemperatureLoad(
                            t_top=t_top, t_bottom=t_bottom,
                            load_case=meta["load_case"],
                        ))
                        break

        elif keyword == "ANALYSIS_OPTIONS":
            # Format: ANALYSIS_OPTIONS <count> followed by count
            # key=value lines. v0.18 recognises:
            #   * include_self_weight=<bool>  (v0.9)
            #   * self_weight_case=<NAME>     (v0.18 — PR-A)
            # Unknown keys raise ``ValueError`` so typos surface
            # (mirrors the per-element override-key strictness already
            # in this parser).
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
                elif key == "self_weight_case":
                    if not val:
                        raise ValueError(
                            "ANALYSIS_OPTIONS self_weight_case= requires "
                            "a case name."
                        )
                    model.self_weight_case = val
                else:
                    raise ValueError(
                        f"Unknown ANALYSIS_OPTIONS key {key!r}. "
                        "Allowed: ['include_self_weight', "
                        "'self_weight_case']."
                    )

        elif keyword == "LOAD_CASES":
            # Format: LOAD_CASES <count> followed by count rows of
            #   <name>  [enabled=<bool>]
            # Names normalise to the same single-token rule used by the
            # dialog (no whitespace, no '#'); LoadCase.__post_init__
            # enforces it. Missing ``enabled=`` defaults to True. The
            # DEFAULT case is auto-created by StructuralModel.__post_init__
            # so the block needn't emit it explicitly; a duplicate
            # DEFAULT row simply overrides its ``enabled`` value.
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                # Strip before the leading-'#' check so a comment line
                # with indentation ("  # comment") is still skipped
                # (Gemini PR #28 finding — without the strip the loop
                # would fall through and ``parts`` would be empty).
                while i < len(lines) and (
                    not lines[i].strip()
                    or lines[i].lstrip().startswith("#")
                ):
                    i += 1
                if i >= len(lines):
                    raise ValueError(
                        "Unexpected end of file inside LOAD_CASES "
                        f"block (expected {count} rows)."
                    )
                parts = lines[i].split("#")[0].split()
                if not parts:
                    raise ValueError(
                        "LOAD_CASES: encountered a blank row inside "
                        "the block."
                    )
                name = parts[0]
                enabled = True
                for tok in parts[1:]:
                    if "=" not in tok:
                        raise ValueError(
                            f"LOAD_CASES row {parts!r}: unexpected "
                            f"trailing token {tok!r} (use key=value)."
                        )
                    k, _, v = tok.partition("=")
                    k = k.strip().lower()
                    v = v.strip()
                    if k == "enabled":
                        enabled = _parse_bool(v, k)
                    else:
                        raise ValueError(
                            f"LOAD_CASES row {parts!r}: unknown "
                            f"key={k!r}. Allowed: ['enabled']."
                        )
                if name in model.load_cases:
                    # Override the auto-created DEFAULT (or a duplicate
                    # row) — keep the user's enabled flag.
                    model.load_cases[name] = LoadCase(
                        name=name, enabled=enabled,
                    )
                else:
                    model.load_cases[name] = LoadCase(
                        name=name, enabled=enabled,
                    )

        elif keyword == "LOAD_COMBINATIONS":
            # Format: LOAD_COMBINATIONS <count> followed by count rows of
            #   <name>  <coeff>*<case>  [<coeff>*<case> ...]
            # e.g.  COMB_STRENGTH  1.2*DEAD  1.6*LIVE
            # The term token is ``coefficient*case`` (no spaces around
            # ``*``). At least one term is required; LoadCombination's
            # __post_init__ enforces the finite / non-zero coefficient
            # rules. Combination names must not collide with case names
            # (validated below) and SUM_ALL is rejected by
            # LoadCombination itself.
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (
                    not lines[i].strip()
                    or lines[i].lstrip().startswith("#")
                ):
                    i += 1
                if i >= len(lines):
                    raise ValueError(
                        "Unexpected end of file inside LOAD_COMBINATIONS "
                        f"block (expected {count} rows)."
                    )
                # On a combination row the text after the first ``#`` is
                # the (optional, free-text) description — captured here
                # before it would otherwise be stripped as a comment, so
                # descriptions round-trip through save/reopen.
                before_hash, sep, after_hash = lines[i].partition("#")
                comb_desc = after_hash.strip() if sep else ""
                parts = before_hash.split()
                if len(parts) < 2:
                    raise ValueError(
                        f"LOAD_COMBINATIONS row {parts!r} needs a name "
                        "and at least one coefficient*case term."
                    )
                # Normalise the combination name and term-case names to
                # uppercase — the GUI always stores case names uppercase
                # (``_normalize_load_case``), so a hand-written
                # lower/mixed-case combination row would otherwise refer
                # to a case it can never match and be permanently
                # "unavailable". A post-parse check below validates the
                # (uppercased) references against the case set.
                comb_name = parts[0].upper()
                terms: dict[str, float] = {}
                for tok in parts[1:]:
                    if "*" not in tok:
                        raise ValueError(
                            f"LOAD_COMBINATIONS row {comb_name!r}: term "
                            f"{tok!r} must be 'coefficient*case' "
                            "(e.g. 1.2*DEAD)."
                        )
                    coeff_s, _, case_s = tok.partition("*")
                    case_s = case_s.strip().upper()
                    try:
                        coeff = float(coeff_s.strip())
                    except ValueError:
                        raise ValueError(
                            f"LOAD_COMBINATIONS row {comb_name!r}: "
                            f"coefficient in term {tok!r} is not a number."
                        )
                    if case_s in terms:
                        raise ValueError(
                            f"LOAD_COMBINATIONS row {comb_name!r}: case "
                            f"{case_s!r} appears more than once."
                        )
                    terms[case_s] = coeff
                if comb_name in model.load_cases:
                    raise ValueError(
                        f"LOAD_COMBINATIONS: name {comb_name!r} collides "
                        "with a load-case name; combination names must be "
                        "distinct from case names."
                    )
                if comb_name in model.load_combinations:
                    raise ValueError(
                        "LOAD_COMBINATIONS: duplicate combination name "
                        f"{comb_name!r}."
                    )
                # LoadCombination.__post_init__ enforces the remaining
                # rules (finite / non-zero coeffs, SUM_ALL reject, ≥1
                # term, name shape).
                model.load_combinations[comb_name] = LoadCombination(
                    name=comb_name, terms=terms, description=comb_desc,
                )

        elif keyword == "JOINT_MASSES":
            # Format: JOINT_MASSES <count> followed by count rows of:
            #   <node_id>  [mx=<float>]  [my=<float>]
            # All mass values are optional and default to 0.0.
            # Absent block → empty joint_masses dict (safe default).
            count = int(tokens[1])
            for _ in range(count):
                i += 1
                while i < len(lines) and (
                    not lines[i].strip()
                    or lines[i].lstrip().startswith("#")
                ):
                    i += 1
                if i >= len(lines):
                    raise ValueError(
                        "Unexpected end of file inside JOINT_MASSES block "
                        f"(expected {count} rows)."
                    )
                parts = lines[i].split("#")[0].split()
                if not parts:
                    raise ValueError(
                        "JOINT_MASSES: encountered a blank row inside the block."
                    )
                try:
                    node_id = int(parts[0])
                except ValueError:
                    raise ValueError(
                        f"JOINT_MASSES: expected integer node_id, got {parts[0]!r}."
                    )
                if node_id not in model.nodes:
                    raise ValueError(
                        f"JOINT_MASSES: node_id={node_id} does not exist in the model."
                    )
                jm_kwargs: dict[str, float] = {}
                for tok in parts[1:]:
                    if "=" not in tok:
                        raise ValueError(
                            f"JOINT_MASSES row for node {node_id}: unexpected "
                            f"positional token {tok!r}; use key=value pairs "
                            "(e.g. mx=500.0 my=500.0)."
                        )
                    k, _, v = tok.partition("=")
                    k = k.strip().lower()
                    if k not in ("mx", "my"):
                        raise ValueError(
                            f"JOINT_MASSES row for node {node_id}: unknown "
                            f"key {k!r}. Allowed: ['mx', 'my']."
                        )
                    try:
                        jm_kwargs[k] = float(v)
                    except ValueError:
                        raise ValueError(
                            f"JOINT_MASSES row for node {node_id}: "
                            f"{k}={v!r} is not a valid float."
                        )
                model.joint_masses[node_id] = JointMass(node_id=node_id, **jm_kwargs)

        elif keyword == "MODAL_MASS_SOURCE":
            # Format: MODAL_MASS_SOURCE <count> followed by count key=value rows.
            # Recognised keys:
            #   include_self_mass=<bool>
            #   include_joint_masses=<bool>
            #   include_load_cases=<bool>
            #   case_factor:<NAME>=<float>    (one per entry, multiple allowed)
            # Absent block → default ModalMassSource() (safe default).
            count = int(tokens[1])
            mms_kwargs: dict = {
                "include_self_mass": True,
                "include_joint_masses": True,
                "include_load_cases": False,
            }
            case_factors: dict[str, float] = {}
            for _ in range(count):
                i += 1
                while i < len(lines) and (
                    not lines[i].strip()
                    or lines[i].lstrip().startswith("#")
                ):
                    i += 1
                if i >= len(lines):
                    raise ValueError(
                        "Unexpected end of file inside MODAL_MASS_SOURCE "
                        f"block (expected {count} key=value rows)."
                    )
                opt_line = lines[i].split("#")[0].strip()
                if "=" not in opt_line:
                    raise ValueError(
                        f"MODAL_MASS_SOURCE row {opt_line!r} is not a "
                        "key=value pair."
                    )
                key, _, val = opt_line.partition("=")
                key = key.strip()
                val = val.strip()
                if key == "include_self_mass":
                    mms_kwargs["include_self_mass"] = _parse_bool(val, key)
                elif key == "include_joint_masses":
                    mms_kwargs["include_joint_masses"] = _parse_bool(val, key)
                elif key == "include_load_cases":
                    mms_kwargs["include_load_cases"] = _parse_bool(val, key)
                elif key.startswith("case_factor:"):
                    case_name = key[len("case_factor:"):]
                    if not case_name:
                        raise ValueError(
                            "MODAL_MASS_SOURCE: case_factor: requires a "
                            "case name (e.g. case_factor:DEAD=1.0)."
                        )
                    try:
                        factor = float(val)
                    except ValueError:
                        raise ValueError(
                            f"MODAL_MASS_SOURCE: case_factor:{case_name}="
                            f"{val!r} is not a valid float."
                        )
                    case_factors[case_name] = factor
                else:
                    raise ValueError(
                        f"Unknown MODAL_MASS_SOURCE key {key!r}. Allowed: "
                        "['include_self_mass', 'include_joint_masses', "
                        "'include_load_cases', 'case_factor:<NAME>']."
                    )
            mms_kwargs["load_case_factors"] = case_factors
            model.modal_mass_source = ModalMassSource(**mms_kwargs)

        i += 1

    # Final sweep: auto-create any case referenced by a load tag that
    # the LOAD_CASES block (or its absence) didn't define. Keeps
    # legacy files that carry per-load ``case=NAME`` tokens — without
    # a LOAD_CASES block — fully self-describing.
    referenced: set[str] = set()
    for ld in model.nodal_loads:
        referenced.add(getattr(ld, "load_case", "DEFAULT"))
    for elem in model.elements:
        for ld in getattr(elem, "member_loads", []) or []:
            referenced.add(getattr(ld, "load_case", "DEFAULT"))
    for name in referenced:
        if name not in model.load_cases:
            model.load_cases[name] = LoadCase(name=name)

    # Validate combinations AFTER the auto-create sweep — the case set
    # is only complete here (a case may exist solely via a per-load
    # ``case=`` tag, or because LOAD_COMBINATIONS appeared before
    # LOAD_CASES in a hand-written file):
    #   1. A combination name must not collide with ANY case name. The
    #      per-row check during parsing only saw cases defined so far;
    #      a collision with a later-created case would leave the model
    #      with a case and a combination sharing a name, which the GUI's
    #      ``_resolve_active_result`` mis-resolves (Codex PR #29 P2).
    #   2. Every referenced case must exist — a dangling reference is a
    #      malformed file, not a silently-unavailable combination.
    for comb in model.load_combinations.values():
        if comb.name in model.load_cases:
            raise ValueError(
                f"LOAD_COMBINATIONS: combination {comb.name!r} collides "
                "with a load-case name; combination names must be "
                "distinct from case names."
            )
        missing = sorted(c for c in comb.terms if c not in model.load_cases)
        if missing:
            raise ValueError(
                f"LOAD_COMBINATIONS: combination {comb.name!r} references "
                f"load case(s) that do not exist: {', '.join(missing)}."
            )

    # Validate MODAL_MASS_SOURCE case_factor references after the full case
    # set is known (same pattern as LOAD_COMBINATIONS validation above).
    for case_name in model.modal_mass_source.load_case_factors:
        if case_name not in model.load_cases:
            raise ValueError(
                f"MODAL_MASS_SOURCE: case_factor:{case_name} references a "
                "load case that does not exist in the model."
            )

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


def _find_metadata_start(
    parts: list[str], *, mandatory_end: int, optional_count: int,
) -> int:
    """Locate the first trailing-metadata token.

    Each load section has a fixed-width body: a known number of
    mandatory positional fields followed by ``optional_count`` optional
    positional numeric fields (``wx`` / ``wy`` on MEMBER_UDL;
    ``px`` / ``py`` on MEMBER_POINT_LOADS; zero on the thermal /
    nodal-load sections).

    Scan rule: walk parts[mandatory_end : mandatory_end + optional_count]
    and stop at the first non-floatable token — that's where the
    metadata block starts. If every optional slot is floatable, the
    cap (``mandatory_end + optional_count``) is the metadata-start;
    any further tokens MUST be metadata or are rejected by
    :func:`_parse_load_metadata`.

    The cap stops a stray numeric in the surplus slot
    (e.g. ``MEMBER_UDL  1  0  -10  5``) from being silently dropped:
    before the cap, ``5`` would be scanned past as "another numeric"
    and the metadata parser would see no tokens; with the cap, ``5``
    falls into the metadata range and is rejected as an unknown
    coord-system token.
    """
    cap = mandatory_end + optional_count
    for idx in range(mandatory_end, min(len(parts), cap)):
        tok = parts[idx]
        try:
            float(tok)
        except ValueError:
            return idx
    return cap


def _parse_load_metadata(
    parts: list[str], *, start_idx: int, section: str,
    accept_coord_system: bool,
) -> dict:
    """Parse optional trailing tokens after the mandatory numeric fields.

    Two flavours of trailing token (in this order if both present):

    * ``coord_system`` — bare positional token ``local`` / ``global`` /
      ``gravity`` (only for mechanical loads where
      ``accept_coord_system=True``). Missing → ``"local"``.
    * ``key=value`` — recognised today: ``case=<name>``. Missing →
      ``"DEFAULT"``. Unknown keys raise so typos can't degrade silently.

    Returns ``{"coord_system": str, "load_case": str}``. ``coord_system``
    is always returned (set to ``"local"`` even on rows that don't carry
    one, so callers can pass it through unconditionally).
    """
    coord_system = "local"
    load_case = "DEFAULT"
    seen_coord = False
    for tok in parts[start_idx:]:
        if "=" in tok:
            key, _, val = tok.partition("=")
            key_norm = key.strip().lower()
            val_norm = val.strip()
            if key_norm == "case":
                if not val_norm:
                    raise ValueError(
                        f"{section}: empty case= value in row {parts!r}."
                    )
                load_case = val_norm
            else:
                raise ValueError(
                    f"{section}: unknown key={key!r} in trailing token "
                    f"{tok!r}. Allowed keys: ['case']."
                )
        else:
            if not accept_coord_system:
                raise ValueError(
                    f"{section}: unexpected trailing token {tok!r} "
                    "(no positional coord_system token allowed here; "
                    "use key=value for metadata)."
                )
            if seen_coord:
                raise ValueError(
                    f"{section}: more than one positional "
                    f"coord_system token in row {parts!r}."
                )
            coord_system = _parse_coord_system_token(tok, section)
            seen_coord = True
    return {"coord_system": coord_system, "load_case": load_case}
