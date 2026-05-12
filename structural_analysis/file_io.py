"""
File I/O: text-based input parser (compatible with Assignment 2 format).

Extended with optional fields for element type, releases, and member loads.
Creates Element2D subclass instances (FrameElement2D / TrussElement2D).
"""

from __future__ import annotations

from .model import (
    StructuralModel, Node, Material, Section, Support, NodalLoad,
    UniformDistributedLoad, PointLoad,
    TrussTemperatureLoad, FrameTemperatureLoad,
)
from .element import FrameElement2D, TrussElement2D


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

    with open(filepath, "r") as f:
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
                mid = int(parts[0])
                if has_sections_block:
                    # New shape: id  E  [alpha]  [name]
                    E_val = float(parts[1])
                    alpha = float(parts[2]) if len(parts) > 2 else 0.0
                    name = parts[3] if len(parts) > 3 else ""
                    model.materials[mid] = Material(id=mid, name=name,
                                                    E=E_val, alpha=alpha)
                else:
                    # Legacy shape: id  A  I  E  [alpha]  [depth]
                    # Synthesise a 1:1 Material+Section pair so existing
                    # inputs (q2a, q2b, course examples) load unchanged.
                    A_val = float(parts[1])
                    I_val = float(parts[2])
                    E_val = float(parts[3])
                    alpha = float(parts[4]) if len(parts) > 4 else 0.0
                    depth = float(parts[5]) if len(parts) > 5 else 0.0
                    model.materials[mid] = Material(id=mid, E=E_val, alpha=alpha)
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
                sid = int(parts[0])
                material_id = int(parts[1])
                A_val = float(parts[2])
                I_val = float(parts[3])
                depth = float(parts[4]) if len(parts) > 4 else 0.0
                name = parts[5] if len(parts) > 5 else ""
                model.sections[sid] = Section(
                    id=sid, name=name, material_id=material_id,
                    A=A_val, I=I_val, depth=depth,
                )

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

                # Optional element type
                etype = "FRAME"
                if len(parts) >= 5:
                    etype = parts[4].upper()

                # Optional release
                release_i = False
                release_j = False
                if len(parts) >= 6:
                    r = parts[5].upper()
                    if r == "START":
                        release_i = True
                    elif r == "END":
                        release_j = True
                    elif r == "BOTH":
                        release_i = True
                        release_j = True

                if etype == "TRUSS":
                    elem = TrussElement2D(
                        id=eid, node_i=sn, node_j=en,
                        E=mat.E, A=section.A,
                        alpha=mat.alpha, depth=section.depth,
                        section_id=section.id,
                    )
                else:
                    elem = FrameElement2D(
                        id=eid, node_i=sn, node_j=en,
                        E=mat.E, A=section.A, I=section.I,
                        alpha=mat.alpha, depth=section.depth,
                        section_id=section.id,
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
                for elem in model.elements:
                    if elem.id == eid:
                        elem.member_loads.append(PointLoad(py=py, a=a))
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
                if wx != 0.0:
                    # Axial distributed loads are not implemented. Fail loudly
                    # rather than silently dropping the value.
                    raise ValueError(
                        f"MEMBER_UDL for element {eid}: non-zero wx={wx} is "
                        "not supported (only transverse wy is implemented). "
                        "Set wx to 0.0 or remove the column."
                    )
                for elem in model.elements:
                    if elem.id == eid:
                        elem.member_loads.append(UniformDistributedLoad(wy=wy))
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

        i += 1

    return model
