"""JSON project I/O — saves both the structural model (as the canonical
``.txt`` text emitted by :func:`gui.file_writer.write_input_file`) and
GUI-only state (labeled grid system, view limits, snap settings) in a
single ``.spa.json`` file.

The embedded ``model_txt`` is exactly what the solver consumes, so a
project written from the GUI can be exported back to a plain ``.txt``
and run via the CLI with identical results.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from ..file_io import read_input_file
from ..model import StructuralModel
from ..gui.file_writer import write_input_file

from .grid import GridSystem

PROJECT_VERSION = 1


@dataclass
class ViewState:
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    snap_kinds: list[str] = field(default_factory=lambda: [
        "node", "grid", "endpoint", "midpoint", "project"
    ])

    def to_dict(self) -> dict:
        return {
            "xlim": list(self.xlim) if self.xlim is not None else None,
            "ylim": list(self.ylim) if self.ylim is not None else None,
            "snap_kinds": list(self.snap_kinds),
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
        return cls(xlim=xlim, ylim=ylim, snap_kinds=kinds)


@dataclass
class Project:
    model: StructuralModel
    grid: GridSystem = field(default_factory=GridSystem)
    view: ViewState = field(default_factory=ViewState)
    title: str = "Untitled"


def save_project_json(project: Project, path: str) -> None:
    """Write the project to ``path`` as ``.spa.json``.

    The structural model is serialised to its canonical .txt form (via
    :func:`gui.file_writer.write_input_file`) and embedded as a string,
    so the solver remains the single source of truth for model parsing.
    """
    # write_input_file writes to a file; capture via a temp file then read back.
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        write_input_file(project.model, tmp)
        with open(tmp, "r") as f:
            model_txt = f.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    payload: dict[str, Any] = {
        "version": PROJECT_VERSION,
        "title": project.title or project.model.title,
        "units": "kN_m",
        "grid": project.grid.to_dict(),
        "view": project.view.to_dict(),
        "model_txt": model_txt,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_project_json(path: str) -> Project:
    """Read a ``.spa.json`` file. The embedded model_txt is parsed by
    :func:`structural_analysis.file_io.read_input_file`."""
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON is not an object.")
    version = payload.get("version")
    if version != PROJECT_VERSION:
        raise ValueError(
            f"{path}: unsupported project version {version}; "
            f"expected {PROJECT_VERSION}."
        )
    model_txt = payload.get("model_txt")
    if not isinstance(model_txt, str):
        raise ValueError(f"{path}: missing or invalid model_txt.")

    # Write to a temp file so read_input_file can parse it.
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
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
    return Project(model=model, grid=grid, view=view, title=title)
