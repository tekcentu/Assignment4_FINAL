"""Command pattern for model mutations — supports undo/redo.

Every user-visible mutation flows through a Command. ``do(model)`` validates
its inputs **before** mutating; if validation fails it raises ``ValueError``
with a human-readable message and the model is left untouched.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..element import FrameElement2D, TrussElement2D
from ..model import (
    Material,
    Node,
    NodalLoad,
    Section,
    StructuralModel,
    Support,
)

if TYPE_CHECKING:
    from ..model import MemberLoad


class Command:
    """Abstract base. Subclasses implement ``do`` and ``undo``."""

    description: str = "command"

    def do(self, model: StructuralModel) -> None:
        raise NotImplementedError

    def undo(self, model: StructuralModel) -> None:
        raise NotImplementedError


def _next_id(existing: list[int] | dict[int, object]) -> int:
    keys = list(existing.keys()) if isinstance(existing, dict) else list(existing)
    return (max(keys) + 1) if keys else 1


# ── nodes ────────────────────────────────────────────────────────────────


@dataclass
class AddNodeCmd(Command):
    x: float
    y: float
    node_id: int | None = None
    description: str = "add node"

    def do(self, model: StructuralModel) -> None:
        if self.node_id is None:
            self.node_id = _next_id(model.nodes)
        if self.node_id in model.nodes:
            raise ValueError(f"Node id {self.node_id} already exists.")
        for n in model.nodes.values():
            if abs(n.x - self.x) < 1e-9 and abs(n.y - self.y) < 1e-9:
                raise ValueError(
                    f"A node already exists at ({self.x}, {self.y}) "
                    f"(id {n.id})."
                )
        model.nodes[self.node_id] = Node(self.node_id, float(self.x), float(self.y))

    def undo(self, model: StructuralModel) -> None:
        model.nodes.pop(self.node_id, None)


@dataclass
class MoveNodeCmd(Command):
    node_id: int
    new_x: float
    new_y: float
    _old: tuple[float, float] | None = None
    description: str = "move node"

    def do(self, model: StructuralModel) -> None:
        if self.node_id not in model.nodes:
            raise ValueError(f"Node {self.node_id} does not exist.")
        old = model.nodes[self.node_id]
        self._old = (old.x, old.y)
        model.nodes[self.node_id] = Node(self.node_id, float(self.new_x), float(self.new_y))

    def undo(self, model: StructuralModel) -> None:
        if self._old is None:
            return
        model.nodes[self.node_id] = Node(self.node_id, *self._old)


@dataclass
class DeleteNodeCmd(Command):
    node_id: int
    _saved_node: Node | None = None
    _saved_support: Support | None = None
    _saved_loads: list[NodalLoad] = field(default_factory=list)
    _saved_elements: list[object] = field(default_factory=list)
    description: str = "delete node"

    def do(self, model: StructuralModel) -> None:
        if self.node_id not in model.nodes:
            raise ValueError(f"Node {self.node_id} does not exist.")
        # Reset saved state so redo (after a prior undo) doesn't accumulate.
        self._saved_node = None
        self._saved_support = None
        self._saved_loads = []
        self._saved_elements = []
        self._saved_node = model.nodes.pop(self.node_id)
        self._saved_support = model.supports.pop(self.node_id, None)
        self._saved_loads = [ld for ld in model.nodal_loads if ld.node_id == self.node_id]
        model.nodal_loads = [ld for ld in model.nodal_loads if ld.node_id != self.node_id]
        kept = []
        for elem in model.elements:
            if elem.node_i == self.node_id or elem.node_j == self.node_id:
                self._saved_elements.append(elem)
            else:
                kept.append(elem)
        model.elements = kept

    def undo(self, model: StructuralModel) -> None:
        if self._saved_node is not None:
            model.nodes[self.node_id] = self._saved_node
        if self._saved_support is not None:
            model.supports[self.node_id] = self._saved_support
        model.nodal_loads.extend(self._saved_loads)
        model.elements.extend(self._saved_elements)
        model.elements.sort(key=lambda e: e.id)


# ── elements ─────────────────────────────────────────────────────────────


@dataclass
class AddElementCmd(Command):
    node_i: int
    node_j: int
    section_id: int
    kind: str  # "frame" or "truss"
    release_i: bool = False
    release_j: bool = False
    elem_id: int | None = None
    description: str = "add element"

    def do(self, model: StructuralModel) -> None:
        if self.node_i not in model.nodes:
            raise ValueError(f"Start node {self.node_i} does not exist.")
        if self.node_j not in model.nodes:
            raise ValueError(f"End node {self.node_j} does not exist.")
        if self.node_i == self.node_j:
            raise ValueError("Element start and end node cannot be the same.")
        if self.section_id not in model.sections:
            raise ValueError(f"Section {self.section_id} does not exist.")
        section = model.sections[self.section_id]
        if section.material_id not in model.materials:
            raise ValueError(
                f"Section {self.section_id} references material "
                f"{section.material_id}, which does not exist."
            )
        mat = model.materials[section.material_id]
        ni = model.nodes[self.node_i]
        nj = model.nodes[self.node_j]
        if abs(ni.x - nj.x) < 1e-12 and abs(ni.y - nj.y) < 1e-12:
            raise ValueError("Element has zero length (coincident nodes).")
        for elem in model.elements:
            if {elem.node_i, elem.node_j} == {self.node_i, self.node_j}:
                raise ValueError(
                    f"An element already connects nodes {self.node_i} and "
                    f"{self.node_j} (id {elem.id})."
                )
        kind = self.kind.lower()
        if kind not in ("frame", "truss"):
            raise ValueError(f"Element kind must be 'frame' or 'truss', got {self.kind!r}.")

        if self.elem_id is None:
            ids = [e.id for e in model.elements]
            self.elem_id = (max(ids) + 1) if ids else 1
        elif any(e.id == self.elem_id for e in model.elements):
            raise ValueError(f"Element id {self.elem_id} already exists.")

        if kind == "truss":
            elem = TrussElement2D(
                id=self.elem_id, node_i=self.node_i, node_j=self.node_j,
                E=mat.E, A=section.A, alpha=mat.alpha, depth=section.depth,
                section_id=section.id,
            )
        else:
            elem = FrameElement2D(
                id=self.elem_id, node_i=self.node_i, node_j=self.node_j,
                E=mat.E, A=section.A, I=section.I,
                alpha=mat.alpha, depth=section.depth,
                section_id=section.id,
                release_i=self.release_i, release_j=self.release_j,
            )
        model.elements.append(elem)

    def undo(self, model: StructuralModel) -> None:
        model.elements = [e for e in model.elements if e.id != self.elem_id]


@dataclass
class DeleteElementCmd(Command):
    elem_id: int
    _saved: object | None = None
    _saved_index: int = -1
    description: str = "delete element"

    def do(self, model: StructuralModel) -> None:
        for idx, elem in enumerate(model.elements):
            if elem.id == self.elem_id:
                self._saved = elem
                self._saved_index = idx
                model.elements.pop(idx)
                return
        raise ValueError(f"Element {self.elem_id} does not exist.")

    def undo(self, model: StructuralModel) -> None:
        if self._saved is None:
            return
        model.elements.insert(self._saved_index, self._saved)


# ── materials ────────────────────────────────────────────────────────────


@dataclass
class AddOrUpdateMaterialCmd(Command):
    material: Material
    _previous: Material | None = None
    description: str = "edit material"

    def do(self, model: StructuralModel) -> None:
        if self.material.E <= 0:
            raise ValueError("Material E must be positive.")
        self._previous = model.materials.get(self.material.id)
        model.materials[self.material.id] = self.material
        # Propagate updated E/α to elements that belong to any section
        # whose material_id is this material.
        if self._previous is not None:
            owned_sections = {s.id for s in model.sections.values()
                              if s.material_id == self.material.id}
            for elem in model.elements:
                if elem.section_id in owned_sections:
                    elem.E = self.material.E
                    elem.alpha = self.material.alpha

    def undo(self, model: StructuralModel) -> None:
        if self._previous is None:
            model.materials.pop(self.material.id, None)
        else:
            model.materials[self.material.id] = self._previous
            owned_sections = {s.id for s in model.sections.values()
                              if s.material_id == self.material.id}
            for elem in model.elements:
                if elem.section_id in owned_sections:
                    elem.E = self._previous.E
                    elem.alpha = self._previous.alpha


@dataclass
class DeleteMaterialCmd(Command):
    material_id: int
    _saved: Material | None = None
    description: str = "delete material"

    def do(self, model: StructuralModel) -> None:
        if self.material_id not in model.materials:
            raise ValueError(f"Material {self.material_id} does not exist.")
        used_by_sec = [s.id for s in model.sections.values()
                       if s.material_id == self.material_id]
        if used_by_sec:
            raise ValueError(
                f"Material {self.material_id} is in use by section(s) "
                f"{used_by_sec}; delete those sections first."
            )
        self._saved = model.materials.pop(self.material_id)

    def undo(self, model: StructuralModel) -> None:
        if self._saved is not None:
            model.materials[self.material_id] = self._saved


# ── sections ────────────────────────────────────────────────────────────


@dataclass
class AddOrUpdateSectionCmd(Command):
    section: Section
    _previous: Section | None = None
    description: str = "edit section"

    def do(self, model: StructuralModel) -> None:
        if self.section.A <= 0:
            raise ValueError("Section A must be positive.")
        if self.section.I < 0:
            raise ValueError("Section I cannot be negative.")
        if self.section.material_id not in model.materials:
            raise ValueError(
                f"Section references material {self.section.material_id}, "
                "which does not exist."
            )
        self._previous = model.sections.get(self.section.id)
        model.sections[self.section.id] = self.section
        # Propagate A/I/depth to elements that point at this section.
        # If the section's material_id changed, also re-pull E/α from the
        # new material so the element matches the new combination.
        new_mat = model.materials.get(self.section.material_id)
        for elem in model.elements:
            if elem.section_id == self.section.id:
                elem.A = self.section.A
                elem.depth = self.section.depth
                if isinstance(elem, FrameElement2D):
                    elem.I = self.section.I
                if new_mat is not None:
                    elem.E = new_mat.E
                    elem.alpha = new_mat.alpha

    def undo(self, model: StructuralModel) -> None:
        if self._previous is None:
            model.sections.pop(self.section.id, None)
        else:
            model.sections[self.section.id] = self._previous
            prev = self._previous
            prev_mat = model.materials.get(prev.material_id)
            for elem in model.elements:
                if elem.section_id == self.section.id:
                    elem.A = prev.A
                    elem.depth = prev.depth
                    if isinstance(elem, FrameElement2D):
                        elem.I = prev.I
                    if prev_mat is not None:
                        elem.E = prev_mat.E
                        elem.alpha = prev_mat.alpha


@dataclass
class DeleteSectionCmd(Command):
    section_id: int
    _saved: Section | None = None
    description: str = "delete section"

    def do(self, model: StructuralModel) -> None:
        if self.section_id not in model.sections:
            raise ValueError(f"Section {self.section_id} does not exist.")
        used_by = [e.id for e in model.elements
                   if e.section_id == self.section_id]
        if used_by:
            raise ValueError(
                f"Section {self.section_id} is in use by element(s) "
                f"{used_by}; delete those elements first."
            )
        self._saved = model.sections.pop(self.section_id)

    def undo(self, model: StructuralModel) -> None:
        if self._saved is not None:
            model.sections[self.section_id] = self._saved


# ── supports ─────────────────────────────────────────────────────────────


@dataclass
class SetSupportCmd(Command):
    support: Support | None  # None to remove
    node_id: int = 0
    _previous: Support | None = None
    _had: bool = False
    description: str = "edit support"

    def __post_init__(self) -> None:
        if self.support is not None:
            self.node_id = self.support.node_id

    def do(self, model: StructuralModel) -> None:
        if self.node_id not in model.nodes:
            raise ValueError(f"Node {self.node_id} does not exist.")
        self._had = self.node_id in model.supports
        self._previous = model.supports.get(self.node_id)
        if self.support is None:
            model.supports.pop(self.node_id, None)
        else:
            model.supports[self.node_id] = self.support

    def undo(self, model: StructuralModel) -> None:
        if self._had and self._previous is not None:
            model.supports[self.node_id] = self._previous
        else:
            model.supports.pop(self.node_id, None)


# ── loads ────────────────────────────────────────────────────────────────


@dataclass
class SetNodalLoadCmd(Command):
    """Replaces (or removes) the consolidated nodal load at ``node_id``."""
    node_id: int
    fx: float = 0.0
    fy: float = 0.0
    mz: float = 0.0
    _saved: list[NodalLoad] = field(default_factory=list)
    description: str = "edit nodal load"

    def do(self, model: StructuralModel) -> None:
        if self.node_id not in model.nodes:
            raise ValueError(f"Node {self.node_id} does not exist.")
        self._saved = [ld for ld in model.nodal_loads if ld.node_id == self.node_id]
        model.nodal_loads = [ld for ld in model.nodal_loads if ld.node_id != self.node_id]
        if self.fx or self.fy or self.mz:
            model.nodal_loads.append(NodalLoad(self.node_id, float(self.fx),
                                               float(self.fy), float(self.mz)))

    def undo(self, model: StructuralModel) -> None:
        model.nodal_loads = [ld for ld in model.nodal_loads if ld.node_id != self.node_id]
        model.nodal_loads.extend(self._saved)


@dataclass
class AddMemberLoadCmd(Command):
    elem_id: int
    load: object  # MemberLoad
    description: str = "add member load"

    def do(self, model: StructuralModel) -> None:
        from ..model import (
            FrameTemperatureLoad,
            TrussTemperatureLoad,
        )
        for elem in model.elements:
            if elem.id == self.elem_id:
                if isinstance(elem, TrussElement2D) and isinstance(self.load, FrameTemperatureLoad):
                    raise ValueError(
                        "FrameTemperatureLoad is only valid on frame elements."
                    )
                if isinstance(elem, FrameElement2D) and isinstance(self.load, TrussTemperatureLoad):
                    raise ValueError(
                        "TrussTemperatureLoad is only valid on truss elements."
                    )
                elem.member_loads.append(self.load)
                return
        raise ValueError(f"Element {self.elem_id} does not exist.")

    def undo(self, model: StructuralModel) -> None:
        for elem in model.elements:
            if elem.id == self.elem_id:
                if elem.member_loads and elem.member_loads[-1] is self.load:
                    elem.member_loads.pop()
                else:
                    try:
                        elem.member_loads.remove(self.load)
                    except ValueError:
                        pass
                return


@dataclass
class ClearMemberLoadsCmd(Command):
    elem_id: int
    _saved: list = field(default_factory=list)
    description: str = "clear member loads"

    def do(self, model: StructuralModel) -> None:
        for elem in model.elements:
            if elem.id == self.elem_id:
                self._saved = list(elem.member_loads)
                elem.member_loads.clear()
                return
        raise ValueError(f"Element {self.elem_id} does not exist.")

    def undo(self, model: StructuralModel) -> None:
        for elem in model.elements:
            if elem.id == self.elem_id:
                elem.member_loads.extend(self._saved)
                return


# ── replace whole model (for File→Open) ─────────────────────────────────


@dataclass
class ReplaceModelCmd(Command):
    new_model: StructuralModel
    _saved: StructuralModel | None = None
    description: str = "replace model"

    def do(self, model: StructuralModel) -> None:
        self._saved = copy.deepcopy(_snapshot(model))
        _restore(model, self.new_model)

    def undo(self, model: StructuralModel) -> None:
        if self._saved is not None:
            _restore(model, self._saved)


def _snapshot(model: StructuralModel) -> StructuralModel:
    return StructuralModel(
        title=model.title,
        nodes=dict(model.nodes),
        materials=dict(model.materials),
        sections=dict(model.sections),
        elements=list(model.elements),
        supports=dict(model.supports),
        nodal_loads=list(model.nodal_loads),
    )


def _restore(model: StructuralModel, source: StructuralModel) -> None:
    model.title = source.title
    model.nodes = dict(source.nodes)
    model.materials = dict(source.materials)
    model.sections = dict(source.sections)
    model.elements = list(source.elements)
    model.supports = dict(source.supports)
    model.nodal_loads = list(source.nodal_loads)


# ── grid system (GUI-only metadata, not part of StructuralModel) ────────


@dataclass
class SetGridSystemCmd(Command):
    """Replace the host's GridSystem.

    The grid lives on the GUI side (not on StructuralModel). The host
    must expose a mutable container via the ``host_grid_setter`` callback
    pair so the command can do/undo without touching the structural model.
    """

    new_grid: object   # GridSystem
    getter: object     # callable() -> GridSystem
    setter: object     # callable(GridSystem) -> None
    _previous: object | None = None
    description: str = "edit grid"

    def do(self, model: StructuralModel) -> None:
        # ``model`` is unused; grid is GUI state. We still go through the
        # command pipeline so undo/redo behaves uniformly.
        self._previous = self.getter()
        self.setter(self.new_grid)

    def undo(self, model: StructuralModel) -> None:
        self.setter(self._previous)
