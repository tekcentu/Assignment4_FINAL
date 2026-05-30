"""Element-load formatting helpers for the inspector load table.

Pure / Qt-free: this module only consumes the model objects defined in
``structural_analysis/model.py`` so the formatters can be unit-tested
without a QApplication. The Qt UI layer (``ElementPropertiesDialog``)
calls :func:`format_element_loads` to populate its load table; the
selection status bar calls :func:`summarize_selection_loads` for the
grouped multi-select counts.

Self-weight is intentionally NOT considered here — the solver never
appends self-weight to ``elem.member_loads`` (see
``assembler._apply_self_weight``), so iterating ``member_loads`` will
not surface generated loads as if they were user-attached.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..element import FrameElement2D, TrussElement2D
from ..model import (
    FrameTemperatureLoad,
    PointLoad,
    StructuralModel,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)

# Floating-point tolerance for "is this point load at an endpoint?". Same
# scale as the one used in commands._remap_member_loads (decoupled from
# ELEMENT_SPLIT_TOL — see commands.py docstring).
_ENDPOINT_TOL: float = 1e-9


@dataclass(frozen=True)
class ElementLoadRow:
    """One row of the element-loads table.

    ``index`` is the position in ``elem.member_loads`` and is what
    :class:`DeleteMemberLoadCmd` uses to identify the row. The remaining
    fields are formatted text for the inspector table.
    """

    index: int
    kind: str          # short tag: "UDL" | "PointLoad" | "Thermal"
    type_label: str    # display in the Type column
    magnitude: str     # display in the Magnitude column
    position: str      # display in the Position / Notes column
    meaning: str       # short physical interpretation


def _element_length(model: StructuralModel, elem) -> float:
    ni = model.nodes.get(elem.node_i)
    nj = model.nodes.get(elem.node_j)
    if ni is None or nj is None:
        return 0.0
    return ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5


def _format_point_load_position(a: float, L: float) -> str:
    if L > 0 and abs(a) <= _ENDPOINT_TOL:
        return "a = 0  (at i-end)"
    if L > 0 and abs(a - L) <= _ENDPOINT_TOL:
        return f"a = L = {L:g} m  (at j-end / split-node endpoint)"
    return f"a = {a:g} m"


def format_element_loads(
    model: StructuralModel, elem,
) -> list[ElementLoadRow]:
    """Return one :class:`ElementLoadRow` per entry in ``elem.member_loads``.

    Duplicates (e.g. two thermal loads in a row) produce two distinct
    rows — the table never hides repeated entries.
    """
    rows: list[ElementLoadRow] = []
    loads = list(getattr(elem, "member_loads", []) or [])
    L = _element_length(model, elem)
    for idx, ld in enumerate(loads):
        if isinstance(ld, UniformDistributedLoad):
            rows.append(ElementLoadRow(
                index=idx,
                kind="UDL",
                type_label="UDL",
                magnitude=f"wy = {ld.wy:g} kN/m",
                position="Full length, local +y",
                meaning="Transverse line load (local axes)",
            ))
        elif isinstance(ld, PointLoad):
            rows.append(ElementLoadRow(
                index=idx,
                kind="PointLoad",
                type_label="PointLoad",
                magnitude=f"py = {ld.py:g} kN",
                position=_format_point_load_position(ld.a, L),
                meaning="Transverse point load (local +y)",
            ))
        elif isinstance(ld, FrameTemperatureLoad):
            mean = 0.5 * (ld.t_top + ld.t_bottom)
            grad = ld.t_bottom - ld.t_top
            if abs(grad) <= _ENDPOINT_TOL:
                meaning = "Uniform → axial thermal strain"
            elif abs(mean) <= _ENDPOINT_TOL:
                meaning = "Gradient only → bending (no axial)"
            else:
                meaning = "Uniform + gradient → axial + bending"
            rows.append(ElementLoadRow(
                index=idx,
                kind="Thermal",
                type_label="Thermal (frame)",
                magnitude=(
                    f"t_top = {ld.t_top:g} °C, t_bottom = {ld.t_bottom:g} °C"
                ),
                position=f"ΔT̄ = {mean:g} °C, Δ(b−t) = {grad:g} °C",
                meaning=meaning,
            ))
        elif isinstance(ld, TrussTemperatureLoad):
            rows.append(ElementLoadRow(
                index=idx,
                kind="Thermal",
                type_label="Thermal (truss)",
                magnitude=f"ΔT = {ld.delta_T:g} °C",
                position="Uniform along member",
                meaning="Uniform → axial thermal strain (truss)",
            ))
        else:
            rows.append(ElementLoadRow(
                index=idx,
                kind="Other",
                type_label=type(ld).__name__,
                magnitude=repr(ld),
                position="",
                meaning="",
            ))
    return rows


def summarize_selection_loads(
    model: StructuralModel, element_ids,
) -> dict[str, int]:
    """Return a count of loads by kind across the given element ids.

    Used by the selection status bar when ≥ 2 elements are selected.
    Unknown ids are skipped silently. Self-weight does not appear.
    """
    counts: dict[str, int] = {
        "UDL": 0, "PointLoad": 0, "Thermal": 0,
    }
    id_set = set(element_ids)
    for elem in model.elements:
        if elem.id not in id_set:
            continue
        for ld in getattr(elem, "member_loads", []) or []:
            if isinstance(ld, UniformDistributedLoad):
                counts["UDL"] += 1
            elif isinstance(ld, PointLoad):
                counts["PointLoad"] += 1
            elif isinstance(
                ld, (FrameTemperatureLoad, TrussTemperatureLoad),
            ):
                counts["Thermal"] += 1
    return counts


def format_selection_load_counts(counts: dict[str, int]) -> str:
    """Format the dict returned by :func:`summarize_selection_loads`.

    Returns "" when every count is zero so the caller can decide
    whether to append it to the status text or skip it entirely.
    """
    parts: list[str] = []
    for tag, label in (
        ("UDL", "UDL"),
        ("PointLoad", "PointLoad"),
        ("Thermal", "Thermal"),
    ):
        n = counts.get(tag, 0)
        if n:
            parts.append(f"{n} {label}")
    return " · ".join(parts)


__all__ = [
    "ElementLoadRow",
    "format_element_loads",
    "summarize_selection_loads",
    "format_selection_load_counts",
]
