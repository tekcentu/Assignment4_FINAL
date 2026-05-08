"""Write a StructuralModel back to the text format consumed by file_io.read_input_file."""

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


def write_input_file(model: StructuralModel, path: str) -> None:
    """Serialize ``model`` to ``path`` in the text format used by read_input_file.

    The format is round-trip compatible: open the file with read_input_file
    and you get an equivalent model back (modulo internal id identity).
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

    mat_ids = sorted(model.materials)
    out.append(f"MATERIALS {len(mat_ids)}")
    for mid in mat_ids:
        m = model.materials[mid]
        line = f"{mid}  {_fmt(m.A)}  {_fmt(m.I)}  {_fmt(m.E)}"
        if m.alpha or m.depth:
            line += f"  {_fmt(m.alpha)}  {_fmt(m.depth)}"
        out.append(line)
    out.append("")

    out.append(f"ELEMENTS {len(model.elements)}")
    elem_to_mat: dict[int, int] = _element_material_lookup(model)
    for elem in model.elements:
        kind = "TRUSS" if isinstance(elem, TrussElement2D) else "FRAME"
        mat_id = elem_to_mat.get(elem.id, _first_or_zero(mat_ids))
        line = f"{elem.id}  {elem.node_i}  {elem.node_j}  {mat_id}  {kind}"
        if isinstance(elem, FrameElement2D):
            if elem.release_i and elem.release_j:
                line += "  BOTH"
            elif elem.release_i:
                line += "  START"
            elif elem.release_j:
                line += "  END"
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
        out.append(f"{ld.node_id}  {_fmt(ld.fx)}  {_fmt(ld.fy)}  {_fmt(ld.mz)}")
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
            out.append(f"{eid}  0.0  {_fmt(u.wy)}")
        out.append("")
    if points:
        out.append(f"MEMBER_POINT_LOADS {len(points)}")
        for eid, p in points:
            out.append(f"{eid}  {_fmt(p.a)}  0.0  {_fmt(p.py)}")
        out.append("")
    if truss_temps:
        out.append(f"TRUSS_TEMPERATURE {len(truss_temps)}")
        for eid, t in truss_temps:
            out.append(f"{eid}  {_fmt(t.delta_T)}")
        out.append("")
    if frame_temps:
        out.append(f"FRAME_TEMPERATURE {len(frame_temps)}")
        for eid, t in frame_temps:
            out.append(f"{eid}  {_fmt(t.t_top)}  {_fmt(t.t_bottom)}")
        out.append("")

    with open(path, "w") as f:
        f.write("\n".join(out).rstrip() + "\n")


def _element_material_lookup(model: StructuralModel) -> dict[int, int]:
    """Recover (elem_id → material_id) by matching E/A/I against the material table.

    The element classes only store E, A, (I), alpha, depth — not the material id.
    For round-tripping we match those numbers back to the material list.
    """
    lookup: dict[int, int] = {}
    for elem in model.elements:
        for mid, m in model.materials.items():
            if m.E == elem.E and m.A == elem.A:
                if isinstance(elem, FrameElement2D) and m.I != elem.I:
                    continue
                lookup[elem.id] = mid
                break
    return lookup


def _first_or_zero(ids: list[int]) -> int:
    return ids[0] if ids else 0
