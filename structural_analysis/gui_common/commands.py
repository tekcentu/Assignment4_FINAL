"""Command pattern for model mutations — supports undo/redo.

Every user-visible mutation flows through a Command. ``do(model)`` validates
its inputs **before** mutating; if validation fails it raises ``ValueError``
with a human-readable message and the model is left untouched.

Invariant: each command captures the inverse-state it needs (via ``_previous``,
``_saved``, or ``_snapshot``) **before** it mutates the model, and performs all
the reads it needs from the model (e.g. which sections own a material, which
elements point at a section) before any writes. ``do()`` and ``undo()`` are
inverses when ``do()`` completes successfully.

Most commands are single-mutation: if their ``do()`` raises mid-mutation, the
controller must not push the command on the undo stack — the model layer trusts
internal callers to pass well-formed data, so no transactional rollback is
wired in at the framework level.

Composite commands are the exception. :class:`AddMemberCmd` (v0.10.0) auto-
creates up to two nodes before delegating element creation to
:class:`AddElementCmd`; if the inner element step raises, ``AddMemberCmd.do()``
rolls back any nodes it created in the same call so the model is left in its
pre-``do`` state. New composite commands should follow that pattern explicitly
in their docstring + implementation rather than relying on framework support.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..element import FrameElement2D, TrussElement2D
from ..model import (
    NODE_COINCIDENCE_TOL,
    Material,
    Node,
    NodalLoad,
    Section,
    StructuralModel,
    Support,
)

if TYPE_CHECKING:
    from ..model import MemberLoad

# Re-exported here so callers can keep importing
# `NODE_COINCIDENCE_TOL` from `commands` — the constant moved to
# `model.py` in v0.11.0 so the analytic core
# (`assembler.validate_model`) can also consume it without importing
# the GUI layer. Single source of truth is now `model.py`.
__all__ = (
    "NODE_COINCIDENCE_TOL",
)

# Parametric-`t` tolerance for "is this click on an element interior
# vs. close enough to an endpoint to be that endpoint?" Used by
# SplitElementCmd to reject splits within ELEMENT_SPLIT_TOL of either
# end. Deliberately separate from NODE_COINCIDENCE_TOL — the two
# answer different questions (parametric-along-element vs. world-unit
# point equality) and may need to evolve independently.
ELEMENT_SPLIT_TOL: float = 1e-6


def _find_or_create_node(
    model: StructuralModel,
    x: float,
    y: float,
    hinted_id: int | None,
) -> tuple[int, bool]:
    """Return ``(node_id, was_created)`` for the given coordinate.

    Priority: explicit ``hinted_id`` (e.g. from the snap engine) → an
    existing node within :data:`NODE_COINCIDENCE_TOL` of ``(x, y)`` →
    allocate a new id and add the node. Composite commands
    (``AddMemberCmd``, ``SplitElementCmd``) share this so the two
    paths cannot drift.
    """
    if hinted_id is not None and hinted_id in model.nodes:
        return hinted_id, False
    for n in model.nodes.values():
        if (abs(n.x - x) < NODE_COINCIDENCE_TOL
                and abs(n.y - y) < NODE_COINCIDENCE_TOL):
            return n.id, False
    new_id = _next_id(model.nodes)
    model.nodes[new_id] = Node(new_id, float(x), float(y))
    return new_id, True


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
            if (abs(n.x - self.x) < NODE_COINCIDENCE_TOL
                    and abs(n.y - self.y) < NODE_COINCIDENCE_TOL):
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
    material_override_id: int | None = None
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
        # Resolve the effective material: override if given, else section default.
        if self.material_override_id is not None:
            if self.material_override_id not in model.materials:
                raise ValueError(
                    f"Material override id {self.material_override_id} "
                    "does not exist."
                )
            mat = model.materials[self.material_override_id]
        else:
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
                rho=mat.density,
                section_id=section.id,
                material_id_override=self.material_override_id,
            )
        else:
            elem = FrameElement2D(
                id=self.elem_id, node_i=self.node_i, node_j=self.node_j,
                E=mat.E, A=section.A, I=section.I,
                alpha=mat.alpha, depth=section.depth,
                rho=mat.density,
                section_id=section.id,
                material_id_override=self.material_override_id,
                release_i=self.release_i, release_j=self.release_j,
            )
        model.elements.append(elem)

    def undo(self, model: StructuralModel) -> None:
        model.elements = [e for e in model.elements if e.id != self.elem_id]


@dataclass
class AddMemberCmd(Command):
    """Composite: snap-or-create node A, snap-or-create node B, add element.

    Used by the Frame / Truss tools when the user draws a member by
    clicking two points (v0.10.0). Each click is either over an
    existing node (``node_i`` / ``node_j`` set to that node's id) or
    over empty space (``None``). For each ``None`` end, ``do`` first
    looks for an existing node within ``1e-9`` of the requested
    coordinate and reuses it (matches the add-time coincidence block in
    :class:`AddNodeCmd`); only if no match exists does it allocate a
    new node id and add the node.

    Element creation is then delegated to :class:`AddElementCmd` so the
    section / material / release / zero-length / duplicate-element
    rules stay defined in exactly one place.

    do / undo are atomic: a failure mid-``do`` rolls back any nodes
    this command auto-created; ``undo`` removes the element and then
    any auto-created node whose id is no longer referenced by some
    other element / support / nodal load (defensive against a user
    interleaving an unrelated edit between this draw and its undo).
    """

    x_i: float
    y_i: float
    x_j: float
    y_j: float
    kind: str  # "frame" or "truss"
    section_id: int
    release_i: bool = False
    release_j: bool = False
    material_override_id: int | None = None
    node_i: int | None = None  # if set, reuse this node id
    node_j: int | None = None
    elem_id: int | None = None

    _created_node_i: int | None = field(default=None, init=False)
    _created_node_j: int | None = field(default=None, init=False)
    _inner: "AddElementCmd | None" = field(default=None, init=False)
    description: str = "add member"

    # _find_or_create lifted to module-level _find_or_create_node so
    # SplitElementCmd shares the same reuse-or-allocate rules. Kept
    # this stub for any external caller that imported the static
    # method; delegates to the shared helper.
    @staticmethod
    def _find_or_create(
        model: StructuralModel,
        x: float,
        y: float,
        hinted_id: int | None,
    ) -> tuple[int, bool]:
        return _find_or_create_node(model, x, y, hinted_id)

    def do(self, model: StructuralModel) -> None:
        # Reset bookkeeping so a redo (do → undo → do) doesn't
        # accumulate stale state from the previous do().
        self._created_node_i = None
        self._created_node_j = None
        self._inner = None
        try:
            resolved_i, created_i = _find_or_create_node(
                model, self.x_i, self.y_i, self.node_i,
            )
            if created_i:
                self._created_node_i = resolved_i
            resolved_j, created_j = _find_or_create_node(
                model, self.x_j, self.y_j, self.node_j,
            )
            if created_j:
                self._created_node_j = resolved_j
            inner = AddElementCmd(
                node_i=resolved_i,
                node_j=resolved_j,
                section_id=self.section_id,
                kind=self.kind,
                release_i=self.release_i,
                release_j=self.release_j,
                material_override_id=self.material_override_id,
                elem_id=self.elem_id,
            )
            inner.do(model)
            self.elem_id = inner.elem_id
            self._inner = inner
        except Exception:
            # Roll back any nodes we auto-created in this do() call,
            # newest first. AddElementCmd raised before mutating
            # model.elements, so element rollback is a no-op.
            for nid in (self._created_node_j, self._created_node_i):
                if nid is not None:
                    model.nodes.pop(nid, None)
            self._created_node_i = None
            self._created_node_j = None
            self._inner = None
            raise

    def undo(self, model: StructuralModel) -> None:
        if self._inner is not None:
            self._inner.undo(model)
        # Remove auto-created nodes, but only if nothing else now
        # depends on them. A node we created may have been attached to
        # by a later element / support / nodal load between this draw
        # and its undo; in that case leaving it preserves the user's
        # later work.
        for nid in (self._created_node_j, self._created_node_i):
            if nid is None or nid not in model.nodes:
                continue
            if any(
                e.node_i == nid or e.node_j == nid for e in model.elements
            ):
                continue
            if nid in model.supports:
                continue
            if any(ld.node_id == nid for ld in model.nodal_loads):
                continue
            del model.nodes[nid]


_SPLIT_BLOCKED_LOADS_MSG: str = (
    "This element has member loads. Splitting loaded elements requires "
    "load remapping and is not implemented yet."
)


@dataclass
class SplitElementCmd(Command):
    """Replace element A→B with A→C and C→B, where C is the projected click.

    Used when the user clicks an existing element's interior with the
    Node tool or as a member-draw endpoint (v0.11.0). The split point
    is computed by projecting ``(x, y)`` onto the parent element's
    segment in world space; the parametric position must satisfy
    ``ELEMENT_SPLIT_TOL < t < 1 - ELEMENT_SPLIT_TOL`` or the command
    raises (the controller is expected to have routed near-endpoint
    clicks to a node-reuse path instead).

    Member-load policy (v1, per PR #21 design): if the parent element
    has any entries in ``member_loads``, the command raises with a
    clear message and the model is left untouched. Splitting with a
    proper load remap is a follow-up.

    Composite-rollback contract (mirrors :class:`AddMemberCmd`): any
    mutation a failing ``do()`` may have performed is undone before
    the exception propagates, so the controller never sees a
    half-split model. ``undo()`` removes the two child elements,
    re-inserts the parent at its original index, then conditionally
    removes the auto-created node C (only if nothing else has come to
    depend on it in the meantime).
    """

    element_id: int
    x: float
    y: float
    node_id: int | None = None  # if set, reuse this node id as C

    _saved_parent: object | None = field(default=None, init=False)
    _saved_parent_index: int = field(default=-1, init=False)
    _created_node_c: int | None = field(default=None, init=False)
    _resolved_node_c: int | None = field(default=None, init=False)
    _child_a_id: int | None = field(default=None, init=False)
    _child_b_id: int | None = field(default=None, init=False)
    description: str = "split element"

    def do(self, model: StructuralModel) -> None:
        # Reset bookkeeping so redo (do → undo → do) doesn't carry
        # stale state from the previous do().
        self._saved_parent = None
        self._saved_parent_index = -1
        self._created_node_c = None
        self._resolved_node_c = None
        self._child_a_id = None
        self._child_b_id = None

        # Local import keeps the model→commands dependency direction
        # unchanged (commands.py never imports gui_qt).
        from .geometry import project_point_on_segment

        parent_idx = next(
            (i for i, e in enumerate(model.elements)
             if e.id == self.element_id),
            -1,
        )
        if parent_idx < 0:
            raise ValueError(
                f"Element {self.element_id} does not exist; cannot split."
            )
        parent = model.elements[parent_idx]

        if getattr(parent, "member_loads", None):
            raise ValueError(_SPLIT_BLOCKED_LOADS_MSG)

        ni = model.nodes.get(parent.node_i)
        nj = model.nodes.get(parent.node_j)
        if ni is None or nj is None:
            raise ValueError(
                f"Element {self.element_id} references a missing node; "
                "cannot split."
            )

        proj_x, proj_y, t = project_point_on_segment(
            self.x, self.y, ni.x, ni.y, nj.x, nj.y,
        )
        if t <= ELEMENT_SPLIT_TOL or t >= 1.0 - ELEMENT_SPLIT_TOL:
            raise ValueError(
                f"Split point is too close to an endpoint of element "
                f"{self.element_id} (t={t:.6g}); reuse the endpoint "
                "node instead."
            )

        # Resolve / create node C. Auto-create is the common path; an
        # explicit node_id is supported so the controller can pass a
        # snapped node when the click hits a node-on-element.
        resolved_c, created_c = _find_or_create_node(
            model, proj_x, proj_y, self.node_id,
        )
        if created_c:
            self._created_node_c = resolved_c
        self._resolved_node_c = resolved_c

        if resolved_c == parent.node_i or resolved_c == parent.node_j:
            # Tolerance race: ELEMENT_SPLIT_TOL passed but the
            # projected point coincided with an endpoint within
            # NODE_COINCIDENCE_TOL. Roll back the auto-created node
            # and bail.
            if self._created_node_c is not None:
                model.nodes.pop(self._created_node_c, None)
                self._created_node_c = None
            self._resolved_node_c = None
            raise ValueError(
                "Split point coincides with an existing endpoint "
                f"node of element {self.element_id}; reuse that node "
                "instead."
            )

        try:
            child_a, child_b = self._build_children(parent, resolved_c, model)
        except Exception:
            # _build_children allocates two new element ids but does
            # not append to model.elements until both succeed; the
            # only side effect on failure is the auto-created node.
            if self._created_node_c is not None:
                model.nodes.pop(self._created_node_c, None)
                self._created_node_c = None
            self._resolved_node_c = None
            raise

        # Atomically swap parent for the two children at parent's
        # original slot. Append-then-pop would also work but leaves a
        # bigger gap in the iteration order.
        self._saved_parent = parent
        self._saved_parent_index = parent_idx
        model.elements[parent_idx:parent_idx + 1] = [child_a, child_b]
        self._child_a_id = child_a.id
        self._child_b_id = child_b.id

    def _build_children(
        self,
        parent,
        node_c: int,
        model: StructuralModel,
    ) -> tuple[object, object]:
        """Construct A→C and C→B elements with parent's properties.

        Per the inner-end-loses-release rule: ``release_i`` of A→C
        inherits from the parent's ``release_i``; ``release_j`` of A→C
        is False (the inner end of A→C is at C). Similarly C→B's
        ``release_i`` at C is False; its ``release_j`` is the parent's
        ``release_j``. Truss children carry no release fields.
        """
        # Reserve two distinct element ids. We INCLUDE the parent in
        # the max() so children never reuse the parent's freed id —
        # keeping the parent's id strictly historical avoids the
        # "wait, did this element split or did it just get renamed?"
        # confusion at undo time.
        existing_ids = [e.id for e in model.elements]
        id_a = (max(existing_ids) + 1) if existing_ids else 1
        id_b = id_a + 1

        is_frame = isinstance(parent, FrameElement2D)
        if is_frame:
            child_a = FrameElement2D(
                id=id_a, node_i=parent.node_i, node_j=node_c,
                E=parent.E, A=parent.A, I=parent.I,
                alpha=parent.alpha, depth=parent.depth, rho=parent.rho,
                section_id=parent.section_id,
                material_id_override=parent.material_id_override,
                release_i=parent.release_i, release_j=False,
            )
            child_b = FrameElement2D(
                id=id_b, node_i=node_c, node_j=parent.node_j,
                E=parent.E, A=parent.A, I=parent.I,
                alpha=parent.alpha, depth=parent.depth, rho=parent.rho,
                section_id=parent.section_id,
                material_id_override=parent.material_id_override,
                release_i=False, release_j=parent.release_j,
            )
        else:
            child_a = TrussElement2D(
                id=id_a, node_i=parent.node_i, node_j=node_c,
                E=parent.E, A=parent.A,
                alpha=parent.alpha, depth=parent.depth, rho=parent.rho,
                section_id=parent.section_id,
                material_id_override=parent.material_id_override,
            )
            child_b = TrussElement2D(
                id=id_b, node_i=node_c, node_j=parent.node_j,
                E=parent.E, A=parent.A,
                alpha=parent.alpha, depth=parent.depth, rho=parent.rho,
                section_id=parent.section_id,
                material_id_override=parent.material_id_override,
            )
        return child_a, child_b

    def undo(self, model: StructuralModel) -> None:
        if self._saved_parent is None:
            return
        # Remove the children.
        model.elements = [
            e for e in model.elements
            if e.id not in (self._child_a_id, self._child_b_id)
        ]
        # Re-insert the parent at its original index. If the list
        # shrank below that index because of unrelated edits, append.
        idx = self._saved_parent_index
        if 0 <= idx <= len(model.elements):
            model.elements.insert(idx, self._saved_parent)
        else:
            model.elements.append(self._saved_parent)
        # Remove the auto-created node C only if nothing else has
        # come to depend on it.
        nid = self._created_node_c
        if nid is None or nid not in model.nodes:
            return
        if any(e.node_i == nid or e.node_j == nid for e in model.elements):
            return
        if nid in model.supports:
            return
        if any(ld.node_id == nid for ld in model.nodal_loads):
            return
        del model.nodes[nid]


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


@dataclass
class UpdateElementCmd(Command):
    elem_id: int
    section_id: int
    kind: str
    release_i: bool = False
    release_j: bool = False
    material_override_id: int | None = None
    _saved: object | None = None
    _saved_index: int = -1
    description: str = "edit element"

    def do(self, model: StructuralModel) -> None:
        for idx, old in enumerate(model.elements):
            if old.id == self.elem_id:
                self._saved = old
                self._saved_index = idx
                break
        else:
            raise ValueError(f"Element {self.elem_id} does not exist.")
        if self.section_id not in model.sections:
            raise ValueError(f"Section {self.section_id} does not exist.")
        section = model.sections[self.section_id]
        if section.material_id not in model.materials:
            raise ValueError(
                f"Section {self.section_id} references material "
                f"{section.material_id}, which does not exist."
            )
        kind = self.kind.lower()
        if kind not in ("frame", "truss"):
            raise ValueError(f"Element kind must be 'frame' or 'truss', got {self.kind!r}.")
        old = self._saved
        old_kind = getattr(old, "kind", "")
        if old_kind and old_kind != kind and getattr(old, "member_loads", []):
            raise ValueError(
                "Clear member loads before changing an element between frame and truss."
            )
        # Resolve the effective material: override if given, else section default.
        if self.material_override_id is not None:
            if self.material_override_id not in model.materials:
                raise ValueError(
                    f"Material override id {self.material_override_id} "
                    "does not exist."
                )
            mat = model.materials[self.material_override_id]
        else:
            mat = model.materials[section.material_id]
        if kind == "truss":
            elem = TrussElement2D(
                id=self.elem_id, node_i=old.node_i, node_j=old.node_j,
                E=mat.E, A=section.A, alpha=mat.alpha, depth=section.depth,
                rho=mat.density,
                section_id=section.id,
                material_id_override=self.material_override_id,
            )
        else:
            elem = FrameElement2D(
                id=self.elem_id, node_i=old.node_i, node_j=old.node_j,
                E=mat.E, A=section.A, I=section.I,
                alpha=mat.alpha, depth=section.depth,
                rho=mat.density,
                section_id=section.id,
                material_id_override=self.material_override_id,
                release_i=self.release_i, release_j=self.release_j,
            )
        elem.member_loads = list(getattr(old, "member_loads", []))
        model.elements[self._saved_index] = elem

    def undo(self, model: StructuralModel) -> None:
        if self._saved is not None and self._saved_index >= 0:
            model.elements[self._saved_index] = self._saved


# ── materials ────────────────────────────────────────────────────────────


@dataclass
class AddOrUpdateMaterialCmd(Command):
    material: Material
    _previous: Material | None = None
    description: str = "edit material"

    def do(self, model: StructuralModel) -> None:
        if self.material.E <= 0:
            raise ValueError("Material E must be positive.")
        if self.material.density < 0:
            raise ValueError("Material density cannot be negative.")
        previous = model.materials.get(self.material.id)
        self._previous = previous
        model.materials[self.material.id] = self.material
        # Propagate updated E / α / ρ to every element whose *effective*
        # material is this one. An element's effective material is its
        # override if set, else its section's default material — so an
        # overridden element only updates when its override matches this
        # material id, regardless of section.
        if previous is not None:
            mat_id = self.material.id
            for elem in model.elements:
                if _effective_material_id(model, elem) == mat_id:
                    elem.E = self.material.E
                    elem.alpha = self.material.alpha
                    elem.rho = self.material.density

    def undo(self, model: StructuralModel) -> None:
        if self._previous is None:
            model.materials.pop(self.material.id, None)
        else:
            model.materials[self.material.id] = self._previous
            mat_id = self._previous.id
            for elem in model.elements:
                if _effective_material_id(model, elem) == mat_id:
                    elem.E = self._previous.E
                    elem.alpha = self._previous.alpha
                    elem.rho = self._previous.density


def _effective_material_id(model: StructuralModel, elem) -> int | None:
    """Return the id of the material driving ``elem``'s E/α/ρ.

    Used only inside the command-propagation paths in this module —
    higher-level code (GUI display, future self-weight) should call
    :func:`structural_analysis.model.effective_material`, which returns
    the full Material object.
    """
    override = getattr(elem, "material_id_override", None)
    if override is not None:
        return override
    sec = model.sections.get(elem.section_id)
    return sec.material_id if sec is not None else None


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
        used_by_override = [
            e.id for e in model.elements
            if getattr(e, "material_id_override", None) == self.material_id
        ]
        if used_by_sec or used_by_override:
            parts = []
            if used_by_sec:
                parts.append(f"section(s) {used_by_sec}")
            if used_by_override:
                parts.append(
                    f"element override(s) {used_by_override}"
                )
            raise ValueError(
                f"Material {self.material_id} is in use by "
                + " and ".join(parts)
                + "; clear those references first."
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
        previous = model.sections.get(self.section.id)
        new_mat = model.materials.get(self.section.material_id)
        affected_elements = [e for e in model.elements
                             if e.section_id == self.section.id]
        self._previous = previous
        model.sections[self.section.id] = self.section
        # Geometry (A, I, depth) always propagates — it follows the section.
        # Material-derived properties (E, α, ρ) propagate only to elements
        # that don't carry an override. Overridden elements keep their
        # override material's properties even if the section's default
        # material changed.
        for elem in affected_elements:
            elem.A = self.section.A
            elem.depth = self.section.depth
            if isinstance(elem, FrameElement2D):
                elem.I = self.section.I
            if (new_mat is not None
                    and getattr(elem, "material_id_override", None) is None):
                elem.E = new_mat.E
                elem.alpha = new_mat.alpha
                elem.rho = new_mat.density

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
                    if (prev_mat is not None
                            and getattr(elem, "material_id_override", None)
                            is None):
                        elem.E = prev_mat.E
                        elem.alpha = prev_mat.alpha
                        elem.rho = prev_mat.density


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
