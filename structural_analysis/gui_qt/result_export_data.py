"""Pure row builders for read-only result export tables."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..element import FrameElement2D
from ..model import AnalysisResult, StructuralModel
from .element_graphics import diagram_domain, internal_force_at, sample_internal_force


MEMBER_STATION_HEADERS = [
    "case_or_combination",
    "element_id",
    "station_index",
    "x_global_member_m",
    "s_flexible_m",
    "x_over_L_total",
    "s_over_L_flex",
    "N_kN",
    "V_kN",
    "M_kN_m",
]

NODE_RESULT_HEADERS = [
    "case_or_combination",
    "node_id",
    "ux_m",
    "uy_m",
    "rz_rad",
    "Rx_kN",
    "Ry_kN",
    "Mz_kN_m",
]

_FORBIDDEN_COMPARISON_HEADER_PARTS = ("SAP", "Diff", "Pct", "Percent", "%")


@dataclass(frozen=True)
class MemberStationMetadata:
    element_id: int
    L_total: float
    offset_i: float
    offset_j: float
    L_flex: float
    x_start: float
    x_end: float

    @property
    def has_offsets(self) -> bool:
        return self.offset_i != 0.0 or self.offset_j != 0.0


def format_export_value(value: object) -> str:
    """Excel-friendly display/export formatting for scalar result values."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def assert_clean_headers(headers: Sequence[str]) -> None:
    """Fail fast if a result export grows comparison-only columns."""
    bad = [
        h for h in headers
        if any(part.lower() in h.lower() for part in _FORBIDDEN_COMPARISON_HEADER_PARTS)
    ]
    if bad:
        raise ValueError(
            "Result export headers must contain only program result data; "
            f"comparison columns are out of scope: {bad}"
        )


def write_csv(path: str | Path, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Write one header row plus data rows only."""
    assert_clean_headers(headers)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(headers))
        for row in rows:
            writer.writerow([format_export_value(v) for v in row])


def member_station_metadata(model: StructuralModel, elem: FrameElement2D) -> MemberStationMetadata:
    ni = model.nodes[elem.node_i]
    nj = model.nodes[elem.node_j]
    L_total = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
    offset_i = float(getattr(elem, "offset_i", 0.0) or 0.0)
    offset_j = float(getattr(elem, "offset_j", 0.0) or 0.0)
    x_start, x_end = diagram_domain(elem, ni, nj)
    return MemberStationMetadata(
        element_id=elem.id,
        L_total=L_total,
        offset_i=offset_i,
        offset_j=offset_j,
        L_flex=max(0.0, L_total - offset_i - offset_j),
        x_start=x_start,
        x_end=x_end,
    )


def member_station_rows(
    model: StructuralModel,
    result: AnalysisResult,
    element_ids: Sequence[int],
    *,
    case_or_combination: str,
    n_stations: int = 21,
) -> list[list[object]]:
    """Build clean member station export rows using existing force helpers only."""
    if n_stations < 2:
        raise ValueError(f"n_stations must be >= 2, got {n_stations}")
    rows: list[list[object]] = []
    by_id = {e.id: e for e in model.elements}
    for eid in element_ids:
        elem = by_id.get(eid)
        if not isinstance(elem, FrameElement2D):
            raise ValueError("Station Force Table is available for frame elements only.")
        ni = model.nodes.get(elem.node_i)
        nj = model.nodes.get(elem.node_j)
        if ni is None or nj is None:
            raise ValueError(f"Element {eid} references missing node(s).")
        mr = result.member_results.get(eid)
        if mr is None or "f_local" not in mr:
            raise ValueError(f"No member-force result is available for element {eid}.")
        f_local = mr["f_local"]
        xs, _moment_values = sample_internal_force(
            elem, ni, nj, f_local, "moment", n_samples=n_stations,
        )
        if xs is None:
            raise ValueError(f"Element {eid} has no frame moment station domain.")
        meta = member_station_metadata(model, elem)
        for idx, x in enumerate(xs):
            s_flex = float(x) - meta.offset_i
            rows.append([
                case_or_combination,
                eid,
                idx,
                float(x),
                s_flex,
                (float(x) / meta.L_total) if meta.L_total else None,
                (s_flex / meta.L_flex) if meta.L_flex else None,
                internal_force_at(elem, ni, nj, f_local, "axial", float(x)),
                internal_force_at(elem, ni, nj, f_local, "shear", float(x)),
                internal_force_at(elem, ni, nj, f_local, "moment", float(x)),
            ])
    return rows


def node_result_rows(
    model: StructuralModel,
    result: AnalysisResult,
    *,
    case_or_combination: str,
) -> list[list[object]]:
    """Build clean nodal displacement/reaction export rows."""
    rows: list[list[object]] = []
    D = result.D
    reactions = result.reactions or {}
    for nid in sorted(model.nodes):
        emap = result.E_map.get(nid, {})

        def disp(dof: str) -> float | None:
            idx = emap.get(dof)
            return float(D[idx]) if idx is not None else None

        r = reactions.get(nid, {})
        rows.append([
            case_or_combination,
            nid,
            disp("ux"),
            disp("uy"),
            disp("rz"),
            r.get("ux"),
            r.get("uy"),
            r.get("rz"),
        ])
    return rows
