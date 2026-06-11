"""JSON project I/O — saves both the structural model (as the canonical
``.txt`` text emitted by :func:`gui_common.file_writer.write_input_file`) and
GUI-only state (labeled grid system, view limits, snap settings, named groups)
in a single ``.spa.json`` file.

The embedded ``model_txt`` is exactly what the solver consumes, so a
project written from the GUI can be exported back to a plain ``.txt``
and run via the CLI with identical results.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from ..file_io import read_input_file
from ..model import StructuralModel
from ..gui_common.file_writer import write_input_file

from .grid import GridSystem

SCHEMA_VERSION = 1
# Legacy alias kept while transitioning — readers accept either key.
PROJECT_VERSION = SCHEMA_VERSION


@dataclass
class ViewState:
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    snap_kinds: list[str] = field(default_factory=lambda: [
        "node", "grid", "endpoint", "midpoint", "project"
    ])
    # v0.33 — named z-levels (storey manager). Each entry is
    # ``(name, z)``; GUI metadata only, never passed to the solver.
    storeys: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "xlim": list(self.xlim) if self.xlim is not None else None,
            "ylim": list(self.ylim) if self.ylim is not None else None,
            "snap_kinds": list(self.snap_kinds),
            "storeys": [[str(n), float(z)] for n, z in self.storeys],
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "ViewState":
        if data is None:
            return cls()
        xlim = tuple(data["xlim"]) if data.get("xlim") else None
        ylim = tuple(data["ylim"]) if data.get("ylim") else None
        kinds = list(data.get("snap_kinds") or [
            "node", "grid", "endpoint", "midpoint", "project"
        ])
        storeys = [
            (str(row[0]), float(row[1]))
            for row in (data.get("storeys") or [])
            if isinstance(row, (list, tuple)) and len(row) == 2
        ]
        return cls(xlim=xlim, ylim=ylim, snap_kinds=kinds,
                   storeys=storeys)


@dataclass
class SelectionGroup:
    """Named set of node IDs and element IDs — GUI/project metadata only.

    Groups are never passed to the solver.  They are persisted in the
    ``.spa.json`` project wrapper under the top-level ``"groups"`` key.
    """

    name: str
    node_ids: list[int] = field(default_factory=list)
    element_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "node_ids": sorted(self.node_ids),
            "element_ids": sorted(self.element_ids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SelectionGroup":
        return cls(
            name=str(data.get("name", "")),
            node_ids=[int(i) for i in (data.get("node_ids") or [])],
            element_ids=[int(i) for i in (data.get("element_ids") or [])],
        )


@dataclass
class Project:
    model: StructuralModel
    grid: GridSystem = field(default_factory=GridSystem)
    view: ViewState = field(default_factory=ViewState)
    groups: list[SelectionGroup] = field(default_factory=list)
    title: str = "Untitled"


def save_project_json(project: Project, path: str) -> None:
    """Write the project to ``path`` as ``.spa.json``.

    The structural model is serialised to its canonical .txt form (via
    :func:`gui_common.file_writer.write_input_file`) and embedded as a string,
    so the solver remains the single source of truth for model parsing.
    """
    # write_input_file writes to a file; capture via a temp file then read back.
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        write_input_file(project.model, tmp)
        with open(tmp, "r", encoding="utf-8") as f:
            model_txt = f.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "title": project.title or project.model.title,
        "units": "kN_m",
        "grid": project.grid.to_dict(),
        "view": project.view.to_dict(),
        "groups": [g.to_dict() for g in project.groups],
        "model_txt": model_txt,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_project_json(path: str) -> Project:
    """Read a ``.spa.json`` file. The embedded model_txt is parsed by
    :func:`structural_analysis.file_io.read_input_file`."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON is not an object.")
    # Accept the new key (schema_version) and the legacy one (version) so
    # files written during the transitional release still load. New files
    # are always written with schema_version (see save_project_json).
    version = payload.get("schema_version", payload.get("version"))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema_version {version}; "
            f"expected {SCHEMA_VERSION}."
        )
    model_txt = payload.get("model_txt")
    if not isinstance(model_txt, str):
        raise ValueError(f"{path}: missing or invalid model_txt.")

    # Write to a temp file so read_input_file can parse it.
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(model_txt)
        model = read_input_file(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    grid = GridSystem.from_dict(payload.get("grid") or {})
    view = ViewState.from_dict(payload.get("view"))
    title = payload.get("title", model.title)
    # "groups" key absent in old files → empty list (backward compat).
    raw_groups = payload.get("groups") or []
    groups = [SelectionGroup.from_dict(g) for g in raw_groups if isinstance(g, dict)]
    return Project(model=model, grid=grid, view=view, groups=groups, title=title)
