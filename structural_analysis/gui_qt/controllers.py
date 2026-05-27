"""Tool-mode controllers — same pattern as the Tk backend.

Translates canvas events into model commands. Each tool implements the same
minimal interface (``on_click(hit, button)``, ``on_motion(hit)``,
``description``). The MainWindow switches the active tool when the user
clicks a toolbar action or presses a keyboard shortcut.
"""

from __future__ import annotations

from typing import Optional, Protocol

from .canvas import HitResult
from ..gui_common.commands import (
    ELEMENT_SPLIT_TOL,
    NODE_COINCIDENCE_TOL,
    AddElementCmd,
    AddNodeCmd,
    DeleteElementCmd,
    DeleteNodeCmd,
    SplitElementCmd,
)
from ..gui_common.geometry import project_point_on_segment


class _Host(Protocol):
    def execute(self, cmd) -> None: ...
    def model(self): ...
    def show_node_menu(self, node_id: int, action: Optional[str] = None) -> None: ...
    def show_element_menu(self, elem_id: int, action: Optional[str] = None) -> None: ...
    def show_node_details(self, node_id: int) -> None: ...
    def show_element_details(self, elem_id: int) -> None: ...
    def set_status(self, text: str) -> None: ...
    def open_element_dialog_for_pair(
        self, n_i: int, n_j: int, kind: str | None = None
    ) -> None: ...
    def open_element_dialog_for_member(
        self,
        *,
        first_x: float, first_y: float, first_node_id: int | None,
        second_x: float, second_y: float, second_node_id: int | None,
        kind: str | None = None,
    ) -> None: ...
    def set_element_preview(self, start_node_id: int, end_x: float,
                            end_y: float, kind: str) -> None: ...
    def set_element_preview_free(
        self, start_x: float, start_y: float, end_x: float, end_y: float,
        kind: str,
    ) -> None: ...
    def clear_element_preview(self) -> None: ...
    def select_node(self, node_id: int) -> None: ...
    def select_element(self, element_id: int) -> None: ...
    def clear_selection(self) -> None: ...


class Tool:
    name: str = ""
    description: str = ""

    def __init__(self, host: _Host) -> None:
        self.host = host

    def activate(self) -> None:
        self.host.set_status(self.description)

    def deactivate(self) -> None:
        pass

    def on_click(self, hit: HitResult, button: str) -> None:
        pass

    def on_motion(self, hit: HitResult) -> None:
        pass


class SelectTool(Tool):
    name = "select"
    description = (
        "Select: left-click a node or element to highlight it. "
        "Right-click an element for its detail inspector (FBD + "
        "internal-force diagrams)."
    )

    def on_click(self, hit: HitResult, button: str) -> None:
        if button == "right":
            # Right-click routing is owned by MainWindow._on_canvas_click
            # so the inspector / node-menu paths run even when a
            # non-select tool is active. Nothing for the tool to do here.
            return
        if button != "left":
            return
        if hit.node_id is not None:
            self.host.select_node(hit.node_id)
        elif hit.element_id is not None:
            self.host.select_element(hit.element_id)
        else:
            self.host.clear_selection()


def _split_target_for(
    hit: HitResult, host: "_Host",
) -> tuple[int, float, float] | None:
    """Resolve a click to a split target ``(element_id, x_world, y_world)``.

    Returns ``None`` when the click is not on an element interior —
    i.e. either ``hit.element_id`` is missing entirely (click on
    empty space / grid intersection / a node), or the click's world
    coordinates project to a parametric position outside the strict
    interior ``(ELEMENT_SPLIT_TOL, 1 - ELEMENT_SPLIT_TOL)`` of the
    referenced element.

    The geometric check is necessary because the snap engine
    (``gui_qt/snap.py``) gives GRID priority 1 and PROJECT priority
    4 (snap.py:30-37), so most clicks near a grid line win a "grid"
    snap and never advertise `snap_kind == "project"` even when
    visually on top of an element. The canvas fallback path
    (canvas.py:465-484) also sets ``hit.element_id`` without setting
    ``snap_kind``. Trusting ``hit.element_id`` + a world-space
    projection covers both paths.
    """
    if hit.node_id is not None or hit.element_id is None:
        return None
    model = host.model()
    elem = next(
        (e for e in model.elements if e.id == hit.element_id),
        None,
    )
    if elem is None:
        return None
    ni = model.nodes.get(elem.node_i)
    nj = model.nodes.get(elem.node_j)
    if ni is None or nj is None:
        return None
    proj_x, proj_y, t = project_point_on_segment(
        hit.x, hit.y, ni.x, ni.y, nj.x, nj.y,
    )
    if t <= ELEMENT_SPLIT_TOL or t >= 1.0 - ELEMENT_SPLIT_TOL:
        return None
    return (hit.element_id, proj_x, proj_y)


class NodeTool(Tool):
    name = "node"
    description = "Node tool: click on the grid to place a node."

    def on_click(self, hit: HitResult, button: str) -> None:
        if button != "left":
            return
        if hit.node_id is not None:
            self.host.set_status(f"Node {hit.node_id} already at this location.")
            return
        # v0.11.0: clicking an existing element's interior splits the
        # element at the projected point — otherwise the user gets a
        # node that looks "on" the element but isn't connected to it
        # (the disconnected-component bug PR #21 fixes). We project
        # in world space rather than trusting hit.snap_kind because
        # the snap engine's GRID priority (1) beats PROJECT (4) for
        # most clicks near a grid line, leaving snap_kind != "project"
        # even when the cursor is visually on the element.
        target = _split_target_for(hit, self.host)
        if target is not None:
            elem_id, x_world, y_world = target
            self.host.execute(SplitElementCmd(
                element_id=elem_id, x=x_world, y=y_world,
            ))
            return
        self.host.execute(AddNodeCmd(x=hit.x, y=hit.y))


class _PairTool(Tool):
    def __init__(self, host: _Host, kind: str) -> None:
        super().__init__(host)
        self.kind = kind
        # v0.10.0: clicks can land on empty space, so we remember both
        # the coordinates and the snapped node id (which may be None).
        self._first: Optional[tuple[float, float, int | None]] = None

    @property
    def description(self) -> str:
        if self._first is None:
            return (
                f"{self.kind.capitalize()} tool: click the start point. "
                "Snaps to nodes; a new node is created if you click empty space."
            )
        ref = f"node {self._first[2]}" if self._first[2] is not None else "point"
        return (
            f"{self.kind.capitalize()} tool: click the end point "
            f"(start = {ref})."
        )

    def deactivate(self) -> None:
        self._first = None
        self.host.clear_element_preview()

    def _resolve_endpoint(
        self, hit: HitResult,
    ) -> tuple[float, float, int | None] | None:
        """Resolve a member-draw click to ``(x, y, node_id_or_None)``.

        If the click lands on an existing element's interior, fires a
        :class:`SplitElementCmd` first. On split success, the
        projected point and the freshly-created mid node id become
        the endpoint. On split failure (e.g. parent has member
        loads), returns ``None`` so the caller can cancel the draw.

        Each split is its own undo step — per the PR #21 design call,
        we deliberately avoid the atomic-composite path here so
        SplitElementCmd stays cheap to reason about. A user who drew
        a member that bisected two elements will see three entries on
        the undo stack (two splits + one member); each is
        individually undoable.
        """
        target = _split_target_for(hit, self.host)
        if target is None:
            return (hit.x, hit.y, hit.node_id)
        elem_id, x_world, y_world = target
        cmd = SplitElementCmd(
            element_id=elem_id, x=x_world, y=y_world,
        )
        self.host.execute(cmd)
        if cmd._resolved_node_c is None:
            # Split was rejected (member-loaded element, tolerance
            # race, etc.). host.execute already surfaced the error
            # message via QMessageBox.warning, so we just signal
            # cancel to the caller.
            return None
        return (x_world, y_world, cmd._resolved_node_c)

    def on_click(self, hit: HitResult, button: str) -> None:
        if button != "left":
            return
        if self._first is None:
            resolved = self._resolve_endpoint(hit)
            if resolved is None:
                # Split blocked — don't even stash the first click.
                # The user can adjust and try again.
                return
            first_x, first_y, first_id = resolved
            self._first = (first_x, first_y, first_id)
            if first_id is not None:
                self.host.set_element_preview(
                    first_id, first_x, first_y, self.kind,
                )
            else:
                self.host.set_element_preview_free(
                    first_x, first_y, first_x, first_y, self.kind,
                )
            self.host.set_status(self.description)
            return
        first_x, first_y, first_id = self._first
        # Guard against "click in the same spot twice" *before* opening
        # the element-properties dialog — otherwise the user fills the
        # dialog in, only to get an "element has zero length" error on
        # accept. Cover both flavours:
        #  - both clicks snapped to the same existing node (id == id)
        #  - both clicks landed on empty space at coincident coords
        #    (covers the case where neither end has a hinted node id
        #    but the world coords are within NODE_COINCIDENCE_TOL).
        same_node = (
            first_id is not None
            and hit.node_id is not None
            and first_id == hit.node_id
        )
        same_point = (
            abs(first_x - hit.x) < NODE_COINCIDENCE_TOL
            and abs(first_y - hit.y) < NODE_COINCIDENCE_TOL
        )
        if same_node or same_point:
            self.host.set_status("Start and end must be different points.")
            return
        # Second click can also land on an element interior. Resolve
        # via the same helper — on split failure, abandon the draw
        # (any first-click split that succeeded stays as a standalone
        # undo step; the user can revert it themselves).
        resolved = self._resolve_endpoint(hit)
        if resolved is None:
            self.host.clear_element_preview()
            self._first = None
            self.host.set_status(
                "Member draw cancelled — second click could not be "
                "resolved (split blocked or invalid). Use Ctrl+Z to "
                "revert any pending split."
            )
            return
        second_x, second_y, second_id = resolved
        self.host.clear_element_preview()
        self.host.open_element_dialog_for_member(
            first_x=first_x, first_y=first_y, first_node_id=first_id,
            second_x=second_x, second_y=second_y, second_node_id=second_id,
            kind=self.kind,
        )
        self._first = None

    def on_motion(self, hit: HitResult) -> None:
        if self._first is None:
            return
        first_x, first_y, first_id = self._first
        if first_id is not None:
            self.host.set_element_preview(first_id, hit.x, hit.y, self.kind)
        else:
            self.host.set_element_preview_free(
                first_x, first_y, hit.x, hit.y, self.kind,
            )


class FrameTool(_PairTool):
    name = "frame"

    def __init__(self, host: _Host) -> None:
        super().__init__(host, "frame")


class TrussTool(_PairTool):
    name = "truss"

    def __init__(self, host: _Host) -> None:
        super().__init__(host, "truss")


class SupportTool(Tool):
    name = "support"
    description = "Support tool: click a node to edit its support."

    def on_click(self, hit: HitResult, button: str) -> None:
        if button != "left":
            return
        if hit.node_id is None:
            self.host.set_status("Support tool: click an existing node, not empty space.")
            return
        self.host.show_node_menu(hit.node_id, action="support")


class NodalLoadTool(Tool):
    name = "nodal_load"
    description = "Nodal load tool: click a node to add/edit its load."

    def on_click(self, hit: HitResult, button: str) -> None:
        if button != "left":
            return
        if hit.node_id is None:
            self.host.set_status("Nodal-load tool: click an existing node, not empty space.")
            return
        self.host.show_node_menu(hit.node_id, action="nodal_load")


class MemberLoadTool(Tool):
    name = "member_load"
    description = "Member load tool: click an element to add a load."

    def on_click(self, hit: HitResult, button: str) -> None:
        if button != "left":
            return
        if hit.element_id is None:
            self.host.set_status("Member-load tool: click an element line, not empty space or a node.")
            return
        self.host.show_element_menu(hit.element_id, action="member_load")


class DeleteTool(Tool):
    name = "delete"
    description = "Delete tool: click a node or element to delete it."

    def on_click(self, hit: HitResult, button: str) -> None:
        if button != "left":
            return
        if hit.node_id is not None:
            self.host.execute(DeleteNodeCmd(node_id=hit.node_id))
        elif hit.element_id is not None:
            self.host.execute(DeleteElementCmd(elem_id=hit.element_id))
        else:
            self.host.set_status("Delete tool: click a node or element to remove it.")
