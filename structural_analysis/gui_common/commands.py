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
    FrameTemperatureLoad,
    JointMass,
    LoadCase,
    LoadCombination,
    Material,
    Node,
    NodalLoad,
    PointLoad,
    Section,
    StructuralModel,
    Support,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)

if TYPE_CHECKING:
    from ..model import MemberLoad

# Re-exported here so callers can keep importing
# `NODE_COINCIDENCE_TOL` from `commands` — the constant moved to
# `model.py` in v0.11.0 so the analytic core
# (`assembler.validate_model`) can also consume it without importing
# the GUI layer. Single source of truth is now `model.py`. Plain
# module-level binding (no `__all__`) so `from … import *` keeps
# exporting the public command classes as before.

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
    _saved_joint_mass: JointMass | None = None
    description: str = "delete node"

    def do(self, model: StructuralModel) -> None:
        if self.node_id not in model.nodes:
            raise ValueError(f"Node {self.node_id} does not exist.")
        # Reset saved state so redo (after a prior undo) doesn't accumulate.
        self._saved_node = None
        self._saved_support = None
        self._saved_loads = []
        self._saved_elements = []
        self._saved_joint_mass = None
        self._saved_node = model.nodes.pop(self.node_id)
        self._saved_support = model.supports.pop(self.node_id, None)
        self._saved_loads = [ld for ld in model.nodal_loads if ld.node_id == self.node_id]
        model.nodal_loads = [ld for ld in model.nodal_loads if ld.node_id != self.node_id]
        self._saved_joint_mass = model.joint_masses.pop(self.node_id, None)
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
        if self._saved_joint_mass is not None:
            model.joint_masses[self.node_id] = self._saved_joint_mass
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


def _remap_member_loads(
    parent_loads: list,
    L1: float, L_parent: float,
    L_child_a: float, L_child_b: float,
) -> tuple[list, list]:
    """Split parent member_loads into (child_A_loads, child_B_loads).

    Rules (v0.12.0 — Feature B):

    * ``UniformDistributedLoad``: append the same (frozen) instance to
      BOTH children. Resultant w·L = w·L1 + w·L2; FEMs at the shared
      node C cancel.
    * ``PointLoad(py, a)``: FP-roundoff tolerance band ``tol_a =
      1e-12 * max(L_parent, 1.0)`` — tight enough that a load with
      any real offset from the split routes to the correct child
      with its true offset, no matter the parent's length. If ``a <
      L1 - tol_a`` → child A (unchanged). If ``a > L1 + tol_a`` →
      child B with ``a -= L1`` (clamped to ``L_child_b`` to survive
      element.py's ``a <= L + 1e-10`` bounds guard). Otherwise (load
      at split point in FP) → child A with ``a = L_child_a`` (use
      child A's ACTUAL length so we don't lose the same FP race on
      the left).
    * ``FrameTemperatureLoad`` / ``TrussTemperatureLoad``: append
      unchanged to BOTH children — thermal fixed-end forces are
      length-independent and contributions at node C cancel exactly.
    * Anything else: raise ``ValueError`` BEFORE the caller mutates the
      model so atomicity is preserved.
    """
    from dataclasses import replace
    # Pure FP roundoff tolerance: "is `a` the same number as `L1` in
    # floating point?", NOT "is `a` within some user-meaningful
    # distance of L1?". Deliberately decoupled from ELEMENT_SPLIT_TOL
    # (whose 1e-6 value is correct for the parametric-`t` reject at
    # SplitElementCmd.do() but six orders too loose here — a long
    # member would have snapped real loads into the at-split branch).
    # Fall back to 1.0 for sub-metre elements so a 0.1 m bar still
    # uses a 1e-12 tolerance, not 1e-13.
    tol_a = 1e-12 * max(L_parent, 1.0)
    a_loads: list = []
    b_loads: list = []
    for ml in parent_loads:
        if isinstance(ml, UniformDistributedLoad):
            # Frozen dataclass — safe to share the same instance.
            a_loads.append(ml)
            b_loads.append(ml)
        elif isinstance(ml, PointLoad):
            if ml.a < L1 - tol_a:
                a_loads.append(ml)  # unchanged: a stays the same
            elif ml.a > L1 + tol_a:
                b_loads.append(replace(ml, a=min(ml.a - L1, L_child_b)))
            else:
                # At split point: deterministic assign to child A's
                # right end. Use child A's ACTUAL length for round-off
                # robustness against element.py's `a <= L + 1e-10`
                # bounds guard.
                #
                # Known edge case (near-unreachable): if node C ends up
                # REUSING a pre-existing stray node within
                # NODE_COINCIDENCE_TOL (1e-9) of the projected point
                # rather than being freshly created, child A's
                # solve-time length can differ from L_child_a by up to
                # ~sqrt(2)*1e-9, which could nudge `a` past that guard.
                # No live caller passes a node_id hint and a stray node
                # sitting ~1e-9 from an interior split point is a
                # degenerate model, so this is left as-is rather than
                # clamped (a clamp would break the exact-length
                # assertion in the unit test).
                a_loads.append(replace(ml, a=L_child_a))
        elif isinstance(ml, FrameTemperatureLoad):
            a_loads.append(ml)
            b_loads.append(ml)
        elif isinstance(ml, TrussTemperatureLoad):
            a_loads.append(ml)
            b_loads.append(ml)
        else:
            raise ValueError(
                f"Splitting an element with "
                f"{type(ml).__name__} loads is not yet supported."
            )
    return a_loads, b_loads


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

    Member-load policy (v0.12.0 — Feature B): parent ``member_loads``
    are remapped onto the children via :func:`_remap_member_loads`
    BEFORE any model mutation occurs. Supported load types:

      * ``UniformDistributedLoad`` — copied unchanged to BOTH children.
      * ``PointLoad`` — routed left/right of the split, with the local
        coordinate ``a`` shifted on the right child; a load exactly at
        the split point is assigned to child A at its full length.
      * ``FrameTemperatureLoad`` / ``TrussTemperatureLoad`` — copied
        unchanged to BOTH children (length-independent).
      * any other type raises ``ValueError`` and the model is untouched
        (atomic: the remap is performed BEFORE the parent slot is
        swapped for the two children).

    Composite-rollback contract (mirrors :class:`AddMemberCmd`): any
    mutation a failing ``do()`` may have performed is undone before
    the exception propagates, so the controller never sees a
    half-split model. ``undo()`` removes the two child elements,
    re-inserts the parent at its original index, then conditionally
    removes the auto-created node C (only if nothing else has come to
    depend on it in the meantime). The parent's original
    ``member_loads`` list comes back intact because the parent
    instance is held by reference in ``_saved_parent``.
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

        # Remap member_loads BEFORE any model mutation. If any load
        # type is unsupported this raises and the model is untouched.
        import math as _math
        L_parent = _math.hypot(nj.x - ni.x, nj.y - ni.y)
        L1 = t * L_parent
        L_child_a = _math.hypot(proj_x - ni.x, proj_y - ni.y)
        L_child_b = _math.hypot(nj.x - proj_x, nj.y - proj_y)
        parent_member_loads = list(getattr(parent, "member_loads", None) or [])
        a_loads, b_loads = _remap_member_loads(
            parent_member_loads, L1, L_parent, L_child_a, L_child_b,
        )

        # Resolve / create node C. Auto-create is the common path; an
        # explicit node_id is supported so the controller can pass a
        # snapped node when the click hits a node-on-element.
        #
        # Defensive guard (PR #21 review): _find_or_create_node trusts
        # a non-None hint purely by existence, so a mis-wired caller
        # passing an off-element node id would silently produce
        # geometrically incoherent children (A→far → far→B). Reject
        # any hint whose coordinates don't match the projected split
        # point within NODE_COINCIDENCE_TOL. Current callers always
        # leave self.node_id=None; this just prevents a future
        # foot-gun.
        if self.node_id is not None and self.node_id in model.nodes:
            hinted = model.nodes[self.node_id]
            if (abs(hinted.x - proj_x) >= NODE_COINCIDENCE_TOL
                    or abs(hinted.y - proj_y) >= NODE_COINCIDENCE_TOL):
                raise ValueError(
                    f"Hinted node_id={self.node_id} at "
                    f"({hinted.x}, {hinted.y}) does not lie on element "
                    f"{self.element_id}'s split point "
                    f"({proj_x}, {proj_y}); refusing to split."
                )
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

        # Attach the remapped loads to the children BEFORE the swap so
        # that the only mutation of model state is a single atomic
        # slice assignment below. Children are local objects up to
        # this point, so this can't leave the model half-mutated.
        if a_loads:
            child_a.member_loads.extend(a_loads)
        if b_loads:
            child_b.member_loads.extend(b_loads)

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
class DrawMemberWithSplitsCmd(Command):
    """Composite: split 0–2 parent elements, then add a new member.

    Used by the Frame / Truss tools (v0.11.0 follow-up) so that
    drawing a member whose endpoint(s) land on an existing element's
    interior collapses to **one undo step** — one ``Ctrl+Z`` reverses
    every side effect of that single user gesture, instead of the
    1–3 separate `Ctrl+Z`s the previous wiring required.

    Composition rules:

    - ``split_target_i`` / ``split_target_j`` are
      ``(parent_element_id, projected_x, projected_y)`` tuples or
      ``None``. A ``None`` slot means that endpoint came from a node-
      snap or free-space click, not from an element interior — the
      composite skips the split for that slot and uses the supplied
      ``node_*_hint`` / ``x_*, y_*`` like a plain ``AddMemberCmd``.
    - When *both* targets are ``None`` the composite degenerates to a
      thin wrapper around :class:`AddMemberCmd`. The dialog dispatch
      in :mod:`gui_qt.app` short-circuits this case (plain
      ``AddMemberCmd`` push), but the degenerate path is supported
      here so the command class is usable without that branching.

    Atomic-rollback contract (mirrors :class:`AddMemberCmd`): if any
    step raises after earlier steps succeeded, the composite reverses
    them in LIFO order and re-raises. So:

    - split-1 fails → model untouched.
    - split-2 fails (e.g. unsupported member-load type on the parent)
      → split-1's parent is restored, no member is created, model
      untouched.
    - inner ``AddMemberCmd`` fails (e.g. zero-length, duplicate)
      → both splits are reversed, model untouched.
    """

    # Pending split targets — None for endpoints that come from a
    # node-snap / free-space click and don't need a split.
    split_target_i: tuple[int, float, float] | None = None
    split_target_j: tuple[int, float, float] | None = None

    # AddMemberCmd parameters. When a corresponding ``split_target_*``
    # is set, the matching ``node_*_hint`` is ignored (the composite
    # uses the split-result node id instead). Otherwise the existing
    # AddMemberCmd reuse-or-create rules apply.
    x_i: float = 0.0
    y_i: float = 0.0
    node_i_hint: int | None = None
    x_j: float = 0.0
    y_j: float = 0.0
    node_j_hint: int | None = None
    kind: str = "frame"
    section_id: int = 0
    release_i: bool = False
    release_j: bool = False
    material_override_id: int | None = None

    _splits: list[SplitElementCmd] = field(default_factory=list, init=False)
    _inner: "AddMemberCmd | None" = field(default=None, init=False)
    description: str = "draw member"

    def do(self, model: StructuralModel) -> None:
        # Reset bookkeeping so a redo (do → undo → do) starts clean.
        self._splits = []
        self._inner = None
        try:
            resolved_i = self.node_i_hint
            if self.split_target_i is not None:
                eid, px, py = self.split_target_i
                s = SplitElementCmd(element_id=eid, x=px, y=py)
                s.do(model)
                self._splits.append(s)
                resolved_i = s._resolved_node_c

            resolved_j = self.node_j_hint
            if self.split_target_j is not None:
                eid, px, py = self.split_target_j
                s = SplitElementCmd(element_id=eid, x=px, y=py)
                s.do(model)
                self._splits.append(s)
                resolved_j = s._resolved_node_c

            inner = AddMemberCmd(
                x_i=self.x_i, y_i=self.y_i, node_i=resolved_i,
                x_j=self.x_j, y_j=self.y_j, node_j=resolved_j,
                kind=self.kind, section_id=self.section_id,
                release_i=self.release_i, release_j=self.release_j,
                material_override_id=self.material_override_id,
            )
            inner.do(model)
            self._inner = inner
        except Exception:
            # LIFO rollback. AddMemberCmd / SplitElementCmd each
            # guarantee atomicity in their own do(); composing them
            # in reverse undo order leaves the model in its pre-do
            # state. _inner is None if AddMemberCmd never started.
            if self._inner is not None:
                self._inner.undo(model)
                self._inner = None
            for s in reversed(self._splits):
                s.undo(model)
            self._splits = []
            raise

    def undo(self, model: StructuralModel) -> None:
        if self._inner is not None:
            self._inner.undo(model)
        for s in reversed(self._splits):
            s.undo(model)


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


# ── batch element ops (v0.13.0) ──────────────────────────────────────────


# Sentinel sentinel for "clear override" in BatchUpdateElementsCmd's
# material_override_id field. ``None`` already means "don't touch the
# override" so we need a distinct value to mean "set override to None".
CLEAR_MATERIAL_OVERRIDE: int = -1


@dataclass
class BatchUpdateElementsCmd(Command):
    """Apply section and/or material-override changes to many elements.

    Two-field design (both optional):

    * ``section_id``: if ``None``, each element keeps its current
      section. If set, every selected element switches to this section.
    * ``material_override_id``: if ``None``, the override is untouched.
      If ``CLEAR_MATERIAL_OVERRIDE`` (-1), the override is cleared
      (falls back to the section's default material). If any positive
      int, that material id is set as the per-element override.

    The "leave unchanged" semantics protect mixed selections — a user
    can update only the override across elements of varying sections
    without overwriting their distinct sections with a single default.

    Implementation: delegates to ``UpdateElementCmd`` per element so
    propagation of E/A/I/depth/alpha/rho stays in one place. Atomic
    rollback: if any per-element update raises mid-batch, every
    successful sub-update is undone in reverse and the model is left
    untouched.
    """

    element_ids: list[int]
    section_id: int | None = None
    material_override_id: int | None = None
    description: str = "batch assign properties"
    _sub_cmds: list[Command] = field(default_factory=list, init=False)

    def do(self, model: StructuralModel) -> None:
        if not self.element_ids:
            return
        if self.section_id is None and self.material_override_id is None:
            return
        if (
            self.section_id is not None
            and self.section_id not in model.sections
        ):
            raise ValueError(
                f"Section {self.section_id} does not exist."
            )
        if (
            self.material_override_id is not None
            and self.material_override_id != CLEAR_MATERIAL_OVERRIDE
            and self.material_override_id not in model.materials
        ):
            raise ValueError(
                f"Material {self.material_override_id} does not exist."
            )
        self._sub_cmds = []
        try:
            for eid in self.element_ids:
                elem = next(
                    (e for e in model.elements if e.id == eid), None,
                )
                if elem is None:
                    # Skip silently — the selection set may have stale
                    # ids from a deletion that happened in between.
                    continue
                kind = getattr(elem, "kind", "").lower() or (
                    "truss"
                    if isinstance(elem, TrussElement2D)
                    else "frame"
                )
                new_section = (
                    self.section_id
                    if self.section_id is not None
                    else elem.section_id
                )
                if self.material_override_id is None:
                    new_override = getattr(elem, "material_id_override", None)
                elif self.material_override_id == CLEAR_MATERIAL_OVERRIDE:
                    new_override = None
                else:
                    new_override = self.material_override_id
                sub = UpdateElementCmd(
                    elem_id=eid,
                    section_id=new_section,
                    kind=kind,
                    release_i=bool(getattr(elem, "release_i", False)),
                    release_j=bool(getattr(elem, "release_j", False)),
                    material_override_id=new_override,
                )
                sub.do(model)
                self._sub_cmds.append(sub)
        except Exception:
            for cmd in reversed(self._sub_cmds):
                cmd.undo(model)
            self._sub_cmds = []
            raise

    def undo(self, model: StructuralModel) -> None:
        for cmd in reversed(self._sub_cmds):
            cmd.undo(model)


@dataclass
class BatchDeleteCmd(Command):
    """Delete every selected node and element in one undo step.

    Elements are deleted before nodes so an element already explicitly
    marked for deletion never double-fires through the node-cascade
    path. Nodes use ``DeleteNodeCmd``, which already cascades supports,
    nodal loads, and any remaining connected elements (mirroring the
    single-object delete behaviour). A node id that was already
    cascade-deleted by a prior step is silently skipped — its undo
    chain is restored by the original cascading command.
    """

    node_ids: list[int]
    element_ids: list[int]
    description: str = "delete selected"
    _sub_cmds: list[Command] = field(default_factory=list, init=False)

    def do(self, model: StructuralModel) -> None:
        self._sub_cmds = []
        try:
            # Elements first.
            for eid in list(self.element_ids):
                if not any(e.id == eid for e in model.elements):
                    continue
                sub: Command = DeleteElementCmd(elem_id=eid)
                sub.do(model)
                self._sub_cmds.append(sub)
            # Then nodes (cascades remaining connected elements + supports
            # + nodal loads). Skip ids that no longer exist.
            for nid in list(self.node_ids):
                if nid not in model.nodes:
                    continue
                sub = DeleteNodeCmd(node_id=nid)
                sub.do(model)
                self._sub_cmds.append(sub)
        except Exception:
            for cmd in reversed(self._sub_cmds):
                cmd.undo(model)
            self._sub_cmds = []
            raise

    def undo(self, model: StructuralModel) -> None:
        for cmd in reversed(self._sub_cmds):
            cmd.undo(model)


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
    """Replaces (or removes) the consolidated nodal load at ``node_id``.

    Retained for backward compatibility with pre-v0.20 callers / fixtures.
    The PR #30 GUI uses :class:`AddNodalLoadCmd` / :class:`EditNodalLoadRowCmd`
    / :class:`DeleteNodalLoadRowCmd` instead so that a node can carry
    multiple independent load rows (one per case, or several per case).
    """
    node_id: int
    fx: float = 0.0
    fy: float = 0.0
    mz: float = 0.0
    load_case: str = "DEFAULT"
    _saved: list[NodalLoad] = field(default_factory=list)
    description: str = "edit nodal load"

    def do(self, model: StructuralModel) -> None:
        if self.node_id not in model.nodes:
            raise ValueError(f"Node {self.node_id} does not exist.")
        self._saved = [ld for ld in model.nodal_loads if ld.node_id == self.node_id]
        model.nodal_loads = [ld for ld in model.nodal_loads if ld.node_id != self.node_id]
        if self.fx or self.fy or self.mz:
            model.nodal_loads.append(NodalLoad(
                self.node_id, float(self.fx), float(self.fy),
                float(self.mz), load_case=self.load_case,
            ))

    def undo(self, model: StructuralModel) -> None:
        model.nodal_loads = [ld for ld in model.nodal_loads if ld.node_id != self.node_id]
        model.nodal_loads.extend(self._saved)


@dataclass
class AddNodalLoadCmd(Command):
    """Append a new nodal-load row to ``model.nodal_loads`` (v0.20 — PR #30).

    Unlike :class:`SetNodalLoadCmd`, this never overwrites existing rows
    on the same node. A node can carry several rows — one per case, or
    multiple per case (the assembler sums them naturally via ``+=``).

    All-zero rows (fx == fy == mz == 0) are rejected; an empty load has
    no solver effect and would only clutter the inspector.

    The appended row is tracked by **object identity**, not by index,
    so undo correctly removes only this command's row even when
    unrelated commands have inserted / deleted other rows between the
    do() and undo() calls (the LIFO controller never triggers that,
    but the command-layer tests do, and the same identity discipline
    is used by :class:`AddMemberLoadCmd`).
    """
    node_id: int
    fx: float = 0.0
    fy: float = 0.0
    mz: float = 0.0
    load_case: str = "DEFAULT"
    _appended_load: NodalLoad | None = field(default=None, init=False)
    description: str = "add nodal load"

    def do(self, model: StructuralModel) -> None:
        if self.node_id not in model.nodes:
            raise ValueError(f"Node {self.node_id} does not exist.")
        if self.fx == 0.0 and self.fy == 0.0 and self.mz == 0.0:
            raise ValueError(
                "Nodal load has Fx = Fy = Mz = 0 — nothing to add."
            )
        self._appended_load = NodalLoad(
            self.node_id, float(self.fx), float(self.fy),
            float(self.mz), load_case=self.load_case,
        )
        model.nodal_loads.append(self._appended_load)

    def undo(self, model: StructuralModel) -> None:
        if self._appended_load is None:
            return
        for i, ld in enumerate(model.nodal_loads):
            if ld is self._appended_load:
                del model.nodal_loads[i]
                return


@dataclass
class EditNodalLoadRowCmd(Command):
    """Replace one row in ``model.nodal_loads`` by index (v0.20 — PR #30).

    ``row_index`` is the row's position in the flat ``model.nodal_loads``
    list at the moment the GUI captures it. The captured index is only
    used by ``do()``; ``undo()`` finds the replacement row by object
    identity and swaps the saved row back in, so an unrelated insert /
    delete between do() and undo() (non-LIFO usage) can't corrupt a
    different load.

    Editing the row's ``node_id`` is intentionally not supported: a
    user who wants to move a load to a different node should delete
    and re-add it (different command intent).
    """
    row_index: int
    fx: float
    fy: float
    mz: float
    load_case: str = "DEFAULT"
    _saved: NodalLoad | None = field(default=None, init=False)
    _new_load: NodalLoad | None = field(default=None, init=False)
    description: str = "edit nodal load row"

    def do(self, model: StructuralModel) -> None:
        if not (0 <= self.row_index < len(model.nodal_loads)):
            raise ValueError(
                f"Nodal-load row {self.row_index} out of range "
                f"(have {len(model.nodal_loads)} row"
                f"{'s' if len(model.nodal_loads) != 1 else ''})."
            )
        if self.fx == 0.0 and self.fy == 0.0 and self.mz == 0.0:
            raise ValueError(
                "Nodal load has Fx = Fy = Mz = 0 — use Delete to remove it."
            )
        self._saved = model.nodal_loads[self.row_index]
        self._new_load = NodalLoad(
            self._saved.node_id, float(self.fx), float(self.fy),
            float(self.mz), load_case=self.load_case,
        )
        model.nodal_loads[self.row_index] = self._new_load

    def undo(self, model: StructuralModel) -> None:
        if self._saved is None or self._new_load is None:
            return
        for i, ld in enumerate(model.nodal_loads):
            if ld is self._new_load:
                model.nodal_loads[i] = self._saved
                return


@dataclass
class DeleteNodalLoadRowCmd(Command):
    """Remove one row from ``model.nodal_loads`` by index (v0.20 — PR #30).

    Mirrors :class:`DeleteMemberLoadCmd`: undo re-inserts the saved row
    at the same index, so under LIFO undo the row goes back to its
    original position and any later Edit commands referencing later
    rows still address the right load.

    Identity-based undo is not used here because the saved row must
    re-occupy a *position* in the list, not be looked up — and after
    a successful delete the saved row is no longer in the list to
    locate by identity. Callers exercising commands outside LIFO order
    on this list must restore the surrounding state themselves before
    invoking undo (the same constraint applies to DeleteMemberLoadCmd).
    """
    row_index: int
    _saved: NodalLoad | None = field(default=None, init=False)
    description: str = "delete nodal load row"

    def do(self, model: StructuralModel) -> None:
        if not (0 <= self.row_index < len(model.nodal_loads)):
            raise ValueError(
                f"Nodal-load row {self.row_index} out of range "
                f"(have {len(model.nodal_loads)} row"
                f"{'s' if len(model.nodal_loads) != 1 else ''})."
            )
        self._saved = model.nodal_loads[self.row_index]
        del model.nodal_loads[self.row_index]

    def undo(self, model: StructuralModel) -> None:
        if self._saved is None:
            return
        model.nodal_loads.insert(self.row_index, self._saved)


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


@dataclass
class DeleteMemberLoadCmd(Command):
    """Remove a single member-load row from one element.

    Identified by ``elem_id`` + ``load_index`` (position in
    ``elem.member_loads``). Undo re-inserts the saved load at the same
    index so undo/redo preserves order relative to the element's other
    loads. The command is atomic: if the index is out of range the
    model is left untouched and a ``ValueError`` is raised.

    Note: index identification is safe because undo is LIFO — the
    immediate inverse re-inserts at the exact position vacated by
    ``do``. A subsequent unrelated command (e.g. AddMemberLoadCmd on
    the same element) does not interfere because that command's own
    undo restores by identity, not by index.
    """

    elem_id: int
    load_index: int
    _saved_load: object = field(default=None, init=False)
    description: str = "delete member load"

    def do(self, model: StructuralModel) -> None:
        for elem in model.elements:
            if elem.id == self.elem_id:
                if not (0 <= self.load_index < len(elem.member_loads)):
                    raise ValueError(
                        f"Load index {self.load_index} out of range "
                        f"for element {self.elem_id} "
                        f"(has {len(elem.member_loads)} load"
                        f"{'s' if len(elem.member_loads) != 1 else ''})."
                    )
                self._saved_load = elem.member_loads[self.load_index]
                del elem.member_loads[self.load_index]
                return
        raise ValueError(f"Element {self.elem_id} does not exist.")

    def undo(self, model: StructuralModel) -> None:
        for elem in model.elements:
            if elem.id == self.elem_id:
                elem.member_loads.insert(self.load_index, self._saved_load)
                return


@dataclass
class UpdateMemberLoadCmd(Command):
    """Replace one member-load row on an element atomically.

    Identified by ``elem_id`` + ``load_index`` (position in
    ``elem.member_loads``).  Stores both the saved (old) instance and the
    new instance so undo restores the *exact* original by identity, not
    just by value.  Same index identity guarantees as
    :class:`DeleteMemberLoadCmd` — safe because undo is LIFO.

    Atomic: if the index is out of range or the new load is incompatible
    with the element type (frame thermal on a truss, truss thermal on a
    frame), a ``ValueError`` is raised before any mutation.
    """

    elem_id: int
    load_index: int
    new_load: object  # MemberLoad
    _saved_load: object = field(default=None, init=False)
    description: str = "edit member load"

    def do(self, model: StructuralModel) -> None:
        from ..model import (
            FrameTemperatureLoad,
            TrussTemperatureLoad,
        )
        for elem in model.elements:
            if elem.id == self.elem_id:
                if not (0 <= self.load_index < len(elem.member_loads)):
                    raise ValueError(
                        f"Load index {self.load_index} out of range "
                        f"for element {self.elem_id} "
                        f"(has {len(elem.member_loads)} load"
                        f"{'s' if len(elem.member_loads) != 1 else ''})."
                    )
                if isinstance(elem, TrussElement2D) and isinstance(
                    self.new_load, FrameTemperatureLoad
                ):
                    raise ValueError(
                        "FrameTemperatureLoad is only valid on frame elements."
                    )
                if isinstance(elem, FrameElement2D) and isinstance(
                    self.new_load, TrussTemperatureLoad
                ):
                    raise ValueError(
                        "TrussTemperatureLoad is only valid on truss elements."
                    )
                self._saved_load = elem.member_loads[self.load_index]
                elem.member_loads[self.load_index] = self.new_load
                return
        raise ValueError(f"Element {self.elem_id} does not exist.")

    def undo(self, model: StructuralModel) -> None:
        for elem in model.elements:
            if elem.id == self.elem_id:
                elem.member_loads[self.load_index] = self._saved_load
                return


# ── load cases (v0.18 — PR-A) ───────────────────────────────────────────
#
# All five commands invalidate any cached multi-case result the host
# holds (the host's ``_invalidate_result`` runs after every ``execute``
# call). Rename cascades through every attached load's ``load_case``
# field via ``dataclasses.replace`` since the loads are frozen.


def _replace_load_case_on_loads(
    model: StructuralModel, old: str, new: str,
) -> tuple[list, dict[int, list | None]]:
    """Replace ``load_case == old`` with ``new`` on every attached load.

    Returns the **previous** (nodal_loads_snapshot, member_loads_snapshot)
    so the caller can restore on ``undo``. Member-load snapshot is keyed
    by ``element.id``; the value is ``None`` for elements whose
    ``member_loads`` attribute is missing (defensive — today's
    Element2D always has the list, but a future subclass might not)."""
    from dataclasses import replace
    prev_nodal = list(model.nodal_loads)
    prev_member: dict[int, list | None] = {}
    for elem in model.elements:
        m_loads = getattr(elem, "member_loads", None)
        prev_member[elem.id] = list(m_loads) if m_loads is not None else None
    model.nodal_loads = [
        replace(ld, load_case=new) if ld.load_case == old else ld
        for ld in model.nodal_loads
    ]
    for elem in model.elements:
        m_loads = getattr(elem, "member_loads", None)
        if m_loads is None:
            continue
        elem.member_loads[:] = [
            replace(ld, load_case=new) if ld.load_case == old else ld
            for ld in m_loads
        ]
    return prev_nodal, prev_member


def _restore_load_attachments(
    model: StructuralModel,
    prev_nodal: list,
    prev_member: dict[int, list | None],
) -> None:
    model.nodal_loads = list(prev_nodal)
    for elem in model.elements:
        if elem.id not in prev_member:
            continue
        snapshot = prev_member[elem.id]
        if snapshot is None:
            continue
        if getattr(elem, "member_loads", None) is None:
            continue
        elem.member_loads[:] = list(snapshot)


@dataclass
class AddLoadCaseCmd(Command):
    """Create a new load case. ``name`` is normalised by the caller
    (uppercase, no whitespace, no '#')."""
    name: str
    enabled: bool = True
    description: str = "add load case"

    def do(self, model: StructuralModel) -> None:
        if self.name in model.load_cases:
            raise ValueError(
                f"Load case {self.name!r} already exists."
            )
        # LoadCase.__post_init__ validates the name shape.
        model.load_cases[self.name] = LoadCase(
            name=self.name, enabled=self.enabled,
        )

    def undo(self, model: StructuralModel) -> None:
        model.load_cases.pop(self.name, None)


@dataclass
class DeleteLoadCaseCmd(Command):
    """Delete a load case. Either reassigns its loads to ``reassign_to``
    (default DEFAULT) or refuses to run if any load references it AND
    ``reassign_to`` is None.

    Deletion of ``DEFAULT`` is always blocked. If ``model.self_weight_case``
    points at this case it is reset to DEFAULT on do() and restored on
    undo()."""
    name: str
    reassign_to: str | None = "DEFAULT"
    description: str = "delete load case"

    _saved_case: LoadCase | None = field(default=None, init=False)
    _saved_nodal: list = field(default_factory=list, init=False)
    _saved_member: dict[int, list] = field(default_factory=dict, init=False)
    _saved_self_weight_case: str | None = field(default=None, init=False)

    def do(self, model: StructuralModel) -> None:
        if self.name == "DEFAULT":
            raise ValueError(
                "The DEFAULT load case cannot be deleted; it is the "
                "fallback case for every load."
            )
        if self.name not in model.load_cases:
            raise ValueError(f"Load case {self.name!r} does not exist.")
        # PR #29: a case referenced by any load combination cannot be
        # deleted — the user must edit/delete the combination first
        # (otherwise the combination would silently lose a term).
        combos_referencing = sorted(
            c.name for c in model.load_combinations.values()
            if self.name in c.terms
        )
        if combos_referencing:
            raise ValueError(
                f"Load case {self.name!r} is referenced by load "
                f"combination(s) {', '.join(combos_referencing)}. Edit "
                "or delete the combination(s) before deleting the case."
            )
        referenced = any(
            ld.load_case == self.name for ld in model.nodal_loads
        ) or any(
            ld.load_case == self.name
            for elem in model.elements
            for ld in (getattr(elem, "member_loads", None) or [])
        )
        if referenced and self.reassign_to is None:
            raise ValueError(
                f"Load case {self.name!r} is referenced by attached "
                "loads. Pass reassign_to= to reassign them, or remove "
                "the loads first."
            )
        if (
            self.reassign_to is not None
            and self.reassign_to not in model.load_cases
        ):
            raise ValueError(
                f"reassign_to={self.reassign_to!r} is not an existing "
                "load case."
            )
        # Snapshot for undo BEFORE any mutation.
        self._saved_case = model.load_cases[self.name]
        self._saved_self_weight_case = model.self_weight_case
        if referenced and self.reassign_to is not None:
            self._saved_nodal, self._saved_member = (
                _replace_load_case_on_loads(
                    model, self.name, self.reassign_to,
                )
            )
        else:
            self._saved_nodal = list(model.nodal_loads)
            self._saved_member = {
                elem.id: list(elem.member_loads)
                for elem in model.elements
            }
        if model.self_weight_case == self.name:
            model.self_weight_case = "DEFAULT"
        del model.load_cases[self.name]

    def undo(self, model: StructuralModel) -> None:
        if self._saved_case is not None:
            model.load_cases[self.name] = self._saved_case
        _restore_load_attachments(
            model, self._saved_nodal, self._saved_member,
        )
        if self._saved_self_weight_case is not None:
            model.self_weight_case = self._saved_self_weight_case


@dataclass
class RenameLoadCaseCmd(Command):
    """Rename a load case and cascade the new name to every attached
    load's ``load_case`` field. Renaming DEFAULT is blocked (it's a
    sentinel relied on by the file reader's auto-create pass)."""
    old_name: str
    new_name: str
    description: str = "rename load case"

    _saved_case: LoadCase | None = field(default=None, init=False)
    _saved_nodal: list = field(default_factory=list, init=False)
    _saved_member: dict[int, list] = field(default_factory=dict, init=False)
    _saved_self_weight_case: str | None = field(default=None, init=False)
    # Snapshot of every combination whose terms referenced the old
    # case name (keyed by combination name → original LoadCombination),
    # so undo restores the pre-rename term dict exactly.
    _saved_combos: dict[str, LoadCombination] = field(
        default_factory=dict, init=False,
    )

    def do(self, model: StructuralModel) -> None:
        if self.old_name == "DEFAULT":
            raise ValueError(
                "The DEFAULT load case cannot be renamed."
            )
        if self.old_name not in model.load_cases:
            raise ValueError(
                f"Load case {self.old_name!r} does not exist."
            )
        if self.new_name in model.load_cases:
            raise ValueError(
                f"Load case {self.new_name!r} already exists."
            )
        self._saved_case = model.load_cases[self.old_name]
        self._saved_self_weight_case = model.self_weight_case
        # Construct via dataclass to validate the new name's shape.
        renamed = LoadCase(
            name=self.new_name, enabled=self._saved_case.enabled,
            description=self._saved_case.description,
        )
        self._saved_nodal, self._saved_member = (
            _replace_load_case_on_loads(model, self.old_name, self.new_name)
        )
        del model.load_cases[self.old_name]
        model.load_cases[self.new_name] = renamed
        if model.self_weight_case == self.old_name:
            model.self_weight_case = self.new_name
        # PR #29: cascade the rename into every combination term that
        # referenced the old case name.
        self._saved_combos = {}
        for comb_name, comb in list(model.load_combinations.items()):
            if self.old_name in comb.terms:
                self._saved_combos[comb_name] = comb
                new_terms = {
                    (self.new_name if k == self.old_name else k): v
                    for k, v in comb.terms.items()
                }
                model.load_combinations[comb_name] = LoadCombination(
                    name=comb.name, terms=new_terms,
                    description=comb.description,
                )

    def undo(self, model: StructuralModel) -> None:
        model.load_cases.pop(self.new_name, None)
        if self._saved_case is not None:
            model.load_cases[self.old_name] = self._saved_case
        _restore_load_attachments(
            model, self._saved_nodal, self._saved_member,
        )
        if self._saved_self_weight_case is not None:
            model.self_weight_case = self._saved_self_weight_case
        for comb_name, comb in self._saved_combos.items():
            model.load_combinations[comb_name] = comb


@dataclass
class SetLoadCaseEnabledCmd(Command):
    """Toggle the ``enabled`` flag on a load case."""
    name: str
    enabled: bool
    description: str = "toggle load case"

    _saved_enabled: bool | None = field(default=None, init=False)

    def do(self, model: StructuralModel) -> None:
        if self.name not in model.load_cases:
            raise ValueError(f"Load case {self.name!r} does not exist.")
        old = model.load_cases[self.name]
        self._saved_enabled = old.enabled
        model.load_cases[self.name] = LoadCase(
            name=old.name, enabled=self.enabled,
            description=old.description,
        )

    def undo(self, model: StructuralModel) -> None:
        if self._saved_enabled is None:
            return
        old = model.load_cases[self.name]
        model.load_cases[self.name] = LoadCase(
            name=old.name, enabled=self._saved_enabled,
            description=old.description,
        )


@dataclass
class SetSelfWeightCaseCmd(Command):
    """Change ``model.self_weight_case`` (which case absorbs the
    self-weight contribution when ``include_self_weight=True``)."""
    case_name: str
    description: str = "set self-weight case"

    _saved: str | None = field(default=None, init=False)

    def do(self, model: StructuralModel) -> None:
        if self.case_name not in model.load_cases:
            raise ValueError(
                f"Load case {self.case_name!r} does not exist."
            )
        self._saved = model.self_weight_case
        model.self_weight_case = self.case_name

    def undo(self, model: StructuralModel) -> None:
        if self._saved is not None:
            model.self_weight_case = self._saved


# ── load combinations (v0.19 — PR #29) ──────────────────────────────────
#
# Combinations are coefficient-weighted derived views over solved load
# cases. The model holds the definitions in ``load_combinations``; the
# actual combined response is computed on the result wrapper. These
# CRUD commands edit definitions only (and invalidate any stale derived
# result via the host's post-execute invalidation).


def _validate_combination_terms(
    model: StructuralModel, terms: dict[str, float],
) -> None:
    """Shared validation: every referenced case must exist in the
    model. (Finite / non-zero coefficient + ≥1-term rules are enforced
    by ``LoadCombination.__post_init__``.)"""
    missing = sorted(name for name in terms if name not in model.load_cases)
    if missing:
        raise ValueError(
            "Combination references load case(s) that do not exist: "
            + ", ".join(missing)
        )


@dataclass
class AddLoadCombinationCmd(Command):
    """Create a coefficient combination. ``name`` is normalised by the
    caller (uppercase, single token). Validates name uniqueness across
    BOTH combinations and cases, plus the referenced-case existence."""
    name: str
    terms: dict[str, float] = field(default_factory=dict)
    combo_description: str = ""
    description: str = "add load combination"

    def do(self, model: StructuralModel) -> None:
        if self.name in model.load_combinations:
            raise ValueError(
                f"Load combination {self.name!r} already exists."
            )
        if self.name in model.load_cases:
            raise ValueError(
                f"{self.name!r} is already a load-case name; combination "
                "names must be distinct from case names."
            )
        _validate_combination_terms(model, self.terms)
        # LoadCombination.__post_init__ enforces the remaining rules
        # (SUM_ALL reject, ≥1 term, finite / non-zero coeffs, name shape).
        model.load_combinations[self.name] = LoadCombination(
            name=self.name, terms=dict(self.terms),
            description=self.combo_description,
        )

    def undo(self, model: StructuralModel) -> None:
        model.load_combinations.pop(self.name, None)


@dataclass
class DeleteLoadCombinationCmd(Command):
    """Delete a combination. Always allowed (combinations are leaf
    derived views — nothing references them)."""
    name: str
    description: str = "delete load combination"

    _saved: LoadCombination | None = field(default=None, init=False)

    def do(self, model: StructuralModel) -> None:
        if self.name not in model.load_combinations:
            raise ValueError(
                f"Load combination {self.name!r} does not exist."
            )
        self._saved = model.load_combinations.pop(self.name)

    def undo(self, model: StructuralModel) -> None:
        if self._saved is not None:
            model.load_combinations[self.name] = self._saved


@dataclass
class RenameLoadCombinationCmd(Command):
    """Rename a combination (preserving its terms + description)."""
    old_name: str
    new_name: str
    description: str = "rename load combination"

    _saved: LoadCombination | None = field(default=None, init=False)

    def do(self, model: StructuralModel) -> None:
        if self.old_name not in model.load_combinations:
            raise ValueError(
                f"Load combination {self.old_name!r} does not exist."
            )
        if self.new_name in model.load_combinations:
            raise ValueError(
                f"Load combination {self.new_name!r} already exists."
            )
        if self.new_name in model.load_cases:
            raise ValueError(
                f"{self.new_name!r} is already a load-case name; "
                "combination names must be distinct from case names."
            )
        self._saved = model.load_combinations[self.old_name]
        renamed = LoadCombination(
            name=self.new_name, terms=dict(self._saved.terms),
            description=self._saved.description,
        )
        del model.load_combinations[self.old_name]
        model.load_combinations[self.new_name] = renamed

    def undo(self, model: StructuralModel) -> None:
        model.load_combinations.pop(self.new_name, None)
        if self._saved is not None:
            model.load_combinations[self.old_name] = self._saved


@dataclass
class SetLoadCombinationTermsCmd(Command):
    """Replace a combination's term dict (edit coefficients, add/remove
    terms) and optionally its description in one undoable step."""
    name: str
    terms: dict[str, float] = field(default_factory=dict)
    combo_description: str | None = None
    description: str = "edit load combination"

    _saved: LoadCombination | None = field(default=None, init=False)

    def do(self, model: StructuralModel) -> None:
        if self.name not in model.load_combinations:
            raise ValueError(
                f"Load combination {self.name!r} does not exist."
            )
        _validate_combination_terms(model, self.terms)
        self._saved = model.load_combinations[self.name]
        desc = (
            self.combo_description if self.combo_description is not None
            else self._saved.description
        )
        # __post_init__ re-validates the new term set.
        model.load_combinations[self.name] = LoadCombination(
            name=self.name, terms=dict(self.terms), description=desc,
        )

    def undo(self, model: StructuralModel) -> None:
        if self._saved is not None:
            model.load_combinations[self.name] = self._saved


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


# v0.24.0 — Element-orientation & manual-cleanup commands.


@dataclass
class RenumberElementsCmd(Command):
    """Reassign element ids across the model via a bijective mapping.

    The mapping must cover *exactly* the current set of element ids;
    values must be a permutation of positive ints. After do(), the
    element list is sorted ascending by new id so manager tables and
    the inspector iterate in their new order. Member loads are stored
    on the element object (not keyed by id) so they automatically
    travel with the renumbered element.

    Undo restores both ids and the original list order.
    """

    mapping: dict[int, int]
    _saved_order: list[int] = field(default_factory=list, init=False)
    description: str = "renumber elements"

    def do(self, model: StructuralModel) -> None:
        current_ids = [e.id for e in model.elements]
        if set(self.mapping) != set(current_ids):
            raise ValueError(
                "Renumber mapping must cover exactly the current element "
                "ids (got "
                f"{sorted(self.mapping)}, model has {sorted(current_ids)})."
            )
        new_ids = list(self.mapping.values())
        if len(set(new_ids)) != len(new_ids):
            raise ValueError("Renumber mapping must be bijective (no duplicates).")
        if any(v < 1 for v in new_ids):
            raise ValueError("Renumber mapping must assign positive ids.")
        # Save before mutating so undo can restore the original order.
        self._saved_order = current_ids[:]
        for e in model.elements:
            e.id = self.mapping[e.id]
        model.elements.sort(key=lambda e: e.id)

    def undo(self, model: StructuralModel) -> None:
        inv = {new: old for old, new in self.mapping.items()}
        for e in model.elements:
            e.id = inv[e.id]
        # Restore the pre-renumber list order.
        by_id = {e.id: e for e in model.elements}
        model.elements[:] = [by_id[oid] for oid in self._saved_order]


def check_merge_preconditions(
    model: "StructuralModel", middle_node_id: int
) -> "tuple[bool, str | None]":
    """Pre-flight check for :class:`MergeAdjacentElementsCmd`.

    Returns ``(True, None)`` when the merge is allowed, or
    ``(False, reason)`` with a short, user-facing explanation when it
    is not.  The verdict is consistent with what the command's ``do()``
    would decide — if this returns ``True``, the command will succeed;
    if it returns ``False``, the command will raise.  The phrasing may
    differ because the command's ``ValueError`` text is richer (shown
    in a ``QMessageBox``), while the reason here is designed for a
    menu action label or tooltip.
    """
    from ..element import FrameElement2D

    node_m = model.nodes.get(middle_node_id)
    if node_m is None:
        return False, f"node {middle_node_id} not found"

    incident = [
        e for e in model.elements
        if e.node_i == middle_node_id or e.node_j == middle_node_id
    ]
    n = len(incident)
    if n != 2:
        return False, f"not exactly 2 incident elements (found {n})"
    e1, e2 = incident

    if type(e1) is not type(e2):
        return False, "cannot merge frame and truss elements"

    def _outer(e: object) -> int:
        return e.node_j if e.node_i == middle_node_id else e.node_i  # type: ignore[union-attr]

    na = model.nodes.get(_outer(e1))
    nb = model.nodes.get(_outer(e2))
    if na is None or nb is None:
        return False, "outer endpoint node missing from model"

    dxa, dya = na.x - node_m.x, na.y - node_m.y
    dxb, dyb = nb.x - node_m.x, nb.y - node_m.y
    cross = dxa * dyb - dya * dxb
    dot = dxa * dxb + dya * dyb
    L_a = (dxa * dxa + dya * dya) ** 0.5
    L_b = (dxb * dxb + dyb * dyb) ** 0.5
    tol = 1e-9 * max(L_a, L_b, 1.0)
    if abs(cross) > tol * max(L_a, L_b, 1.0):
        return False, "elements are not collinear"
    if dot >= -tol:
        return False, "elements are not on opposite sides of the middle node"

    if e1.section_id != e2.section_id:
        return False, "elements have different sections"
    if e1.material_id_override != e2.material_id_override:
        return False, "elements have different material overrides"

    if isinstance(e1, FrameElement2D):
        def _inner(e: object) -> bool:
            return e.release_j if e.node_i != middle_node_id else e.release_i  # type: ignore[union-attr]

        if _inner(e1) or _inner(e2):
            return False, "release/hinge at the middle node"

    if middle_node_id in model.supports:
        sup = model.supports[middle_node_id]
        if any(
            getattr(sup, f"settle_{dof}", None) is not None
            for dof in ("ux", "uy", "rz")
        ):
            return False, "middle node has a support settlement"
        return False, "middle node has a support"

    if any(nl.node_id == middle_node_id for nl in model.nodal_loads):
        return False, "middle node has a nodal load"

    if e1.member_loads or e2.member_loads:
        return False, "member loads present — remapping not implemented"

    joint_masses = getattr(model, "joint_masses", None)
    if joint_masses is not None:
        jm = (
            joint_masses.get(middle_node_id)
            if hasattr(joint_masses, "get") else None
        )
        if jm is not None and any(
            getattr(jm, k, 0.0) != 0.0 for k in ("mx", "my", "mrz")
        ):
            return False, "middle node has a joint mass"

    return True, None


@dataclass
class MergeAdjacentElementsCmd(Command):
    """Merge two collinear, compatible, unloaded elements that share
    the given middle node into a single element. V1 is conservative:

    Pre-conditions (each raises ValueError on failure — surfaced by
    the host as a QMessageBox.warning so the user sees exactly why):

      1. middle node exists,
      2. exactly two incident elements,
      3. same element subtype (frame-frame or truss-truss),
      4. collinear within tol,
      5. same section_id AND same material_id_override,
      6. (frame) inner releases at the middle node are both False,
      7. no support at the middle node,
      8. no nodal load at the middle node in any load case,
      9. neither incident element carries any member load,
     10. no joint mass on the middle node (when the model supports masses).

    Orientation rule:
      - The merged element keeps the *lower* of the two ids (the
        "surviving" element).
      - It also preserves the surviving element's i→j direction:
        if surviving was outer_S → middle, the merge is outer_S → outer_O;
        if surviving was middle → outer_S, the merge is outer_O → outer_S.
      This avoids surprise flips of the local-x axis on the kept id.

    Undo restores both incident elements at their original list
    positions and resurrects the middle node.
    """

    middle_node_id: int
    _saved_node: object | None = None
    _saved_elements: list[tuple[int, object]] = field(
        default_factory=list, init=False,
    )
    description: str = "merge adjacent elements"

    def do(self, model: StructuralModel) -> None:
        from ..element import FrameElement2D, TrussElement2D

        m = self.middle_node_id
        # 1. node exists
        node_m = model.nodes.get(m)
        if node_m is None:
            raise ValueError(f"Node {m} not found.")

        # 2. exactly two incident elements
        incident = [
            (idx, e)
            for idx, e in enumerate(model.elements)
            if e.node_i == m or e.node_j == m
        ]
        if len(incident) != 2:
            raise ValueError(
                "Merge requires exactly 2 elements at the middle node "
                f"(found {len(incident)})."
            )
        (i1, e1), (i2, e2) = incident

        # 3. same subtype
        if type(e1) is not type(e2):
            raise ValueError("Cannot merge a frame and a truss element.")

        # 4. collinear — direction-cosine cross product within tol of L.
        def _outer_node(e):
            return e.node_j if e.node_i == m else e.node_i

        a_id = _outer_node(e1)
        b_id = _outer_node(e2)
        na = model.nodes.get(a_id)
        nb = model.nodes.get(b_id)
        if na is None or nb is None:
            raise ValueError(
                "Outer endpoint nodes of the incident elements are not "
                f"in the model (a={a_id}, b={b_id})."
            )
        dxa, dya = na.x - node_m.x, na.y - node_m.y
        dxb, dyb = nb.x - node_m.x, nb.y - node_m.y
        # Outer nodes must lie on opposite sides of the middle node
        # along a single straight line, so the vectors (a − m) and
        # (b − m) point in opposite directions. Cross = 0 ⇒ collinear;
        # dot < 0 ⇒ opposite sides.
        cross = dxa * dyb - dya * dxb
        dot = dxa * dxb + dya * dyb
        L_a = (dxa * dxa + dya * dya) ** 0.5
        L_b = (dxb * dxb + dyb * dyb) ** 0.5
        tol = 1e-9 * max(L_a, L_b, 1.0)
        if abs(cross) > tol * max(L_a, L_b, 1.0):
            raise ValueError(
                f"Elements are not collinear (cross = {cross:.3e})."
            )
        if dot >= -tol:
            raise ValueError(
                "Elements do not extend on opposite sides of the middle node."
            )

        # 5. same section / material override
        if e1.section_id != e2.section_id:
            raise ValueError(
                "Elements have different sections "
                f"(section_id {e1.section_id} vs {e2.section_id})."
            )
        if e1.material_id_override != e2.material_id_override:
            raise ValueError(
                "Elements have different material overrides."
            )

        # 6. inner releases at middle node must be False (frame only)
        if isinstance(e1, FrameElement2D):
            def _inner_release(e):
                return e.release_j if e.node_i != m else e.release_i

            if _inner_release(e1) or _inner_release(e2):
                raise ValueError(
                    "Merging at a released end would change structural "
                    "behaviour (inner release detected at the middle node)."
                )

        # 7. no support at middle node (covers settlement / imposed disp
        #    since those live on Support too).
        if m in model.supports:
            sup = model.supports[m]
            if any(getattr(sup, f"settle_{dof}", None) is not None
                   for dof in ("ux", "uy", "rz")):
                raise ValueError(
                    "Middle node has a prescribed support settlement; "
                    "cannot drop it during merge."
                )
            raise ValueError(
                "Middle node has a support; remove it before merging."
            )

        # 8. no nodal load at middle node in any load case
        if any(nl.node_id == m for nl in model.nodal_loads):
            raise ValueError(
                "Middle node carries a nodal load; remove it before merging."
            )

        # 9. neither incident element carries member loads (V1 strict)
        if e1.member_loads or e2.member_loads:
            raise ValueError(
                "Merging loaded elements requires load remapping and is "
                "not implemented yet."
            )

        # 10. joint mass at middle node
        joint_masses = getattr(model, "joint_masses", None)
        if joint_masses is not None:
            jm = joint_masses.get(m) if hasattr(joint_masses, "get") else None
            if jm is not None and any(getattr(jm, k, 0.0) != 0.0
                                       for k in ("mx", "my", "mrz")):
                raise ValueError(
                    "Middle node carries a joint mass; cannot drop it "
                    "during merge."
                )

        # ── Build merged element ─────────────────────────────────────
        # Surviving element = lower id; merged keeps surviving's id and
        # the surviving element's i→j orientation.
        if e1.id <= e2.id:
            surviving, other = e1, e2
            surv_idx, other_idx = i1, i2
        else:
            surviving, other = e2, e1
            surv_idx, other_idx = i2, i1
        surv_outer = _outer_node(surviving)
        other_outer = _outer_node(other)
        if surviving.node_i == surv_outer:
            # Surviving was outer_S → middle; merged = outer_S → outer_O.
            new_i, new_j = surv_outer, other_outer
            release_i_src, release_j_src = "surv_outer_i", "other_outer_j"
        else:
            # Surviving was middle → outer_S; merged = outer_O → outer_S.
            new_i, new_j = other_outer, surv_outer
            release_i_src, release_j_src = "other_outer_i", "surv_outer_j"

        # Outer release mapping (frame only).
        if isinstance(surviving, FrameElement2D):
            def _outer_release(e, outer):
                return e.release_i if e.node_i == outer else e.release_j

            if release_i_src == "surv_outer_i":
                rel_i = _outer_release(surviving, surv_outer)
                rel_j = _outer_release(other, other_outer)
            else:
                rel_i = _outer_release(other, other_outer)
                rel_j = _outer_release(surviving, surv_outer)

            merged = FrameElement2D(
                id=surviving.id,
                node_i=new_i, node_j=new_j,
                E=surviving.E, A=surviving.A,
                alpha=surviving.alpha, depth=surviving.depth,
                rho=surviving.rho,
                section_id=surviving.section_id,
                material_id_override=surviving.material_id_override,
                I=surviving.I,
                release_i=rel_i, release_j=rel_j,
            )
        else:
            merged = TrussElement2D(
                id=surviving.id,
                node_i=new_i, node_j=new_j,
                E=surviving.E, A=surviving.A,
                alpha=surviving.alpha, depth=surviving.depth,
                rho=surviving.rho,
                section_id=surviving.section_id,
                material_id_override=surviving.material_id_override,
            )

        # ── Apply mutation atomically ────────────────────────────────
        self._saved_node = node_m
        self._saved_elements = sorted(
            [(i1, e1), (i2, e2)], key=lambda t: t[0],
        )
        lo_idx, hi_idx = self._saved_elements[0][0], self._saved_elements[1][0]
        # Drop the higher-index element first so the lower index stays
        # valid, then replace the lower with the merged element.
        del model.elements[hi_idx]
        model.elements[lo_idx] = merged
        # Finally drop the middle node.
        del model.nodes[m]

    def undo(self, model: StructuralModel) -> None:
        if self._saved_node is None or not self._saved_elements:
            return
        m = self.middle_node_id
        model.nodes[m] = self._saved_node
        # Restore by index in ascending order so insertions are stable.
        lo_idx, lo_elem = self._saved_elements[0]
        hi_idx, hi_elem = self._saved_elements[1]
        # The merged element currently occupies lo_idx — replace it
        # with the original lo element, then insert hi at hi_idx.
        model.elements[lo_idx] = lo_elem
        model.elements.insert(hi_idx, hi_elem)


# ── modal mass source (v0.25 — PR #40) ──────────────────────────────────


@dataclass
class UpdateModalMassSourceCmd(Command):
    """Replace ``model.modal_mass_source`` with a new :class:`ModalMassSource`.

    Captured on both do and undo so the mass-source dialog's OK path can
    be wired through ``host.execute(...)`` and receive the same undo/redo
    treatment as all other model mutations.  The command never calls GUI
    methods directly — result invalidation happens in ``MainWindow.execute``
    via ``_invalidate_result``.
    """

    from ..model import ModalMassSource as _ModalMassSource  # type: ignore[misc]
    new_source: object   # ModalMassSource
    _previous: object | None = None
    description: str = "update modal mass source"

    def do(self, model: StructuralModel) -> None:
        self._previous = getattr(model, "modal_mass_source", None)
        model.modal_mass_source = self.new_source

    def undo(self, model: StructuralModel) -> None:
        if self._previous is not None:
            model.modal_mass_source = self._previous
