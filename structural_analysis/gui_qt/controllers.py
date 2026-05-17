"""Tool-mode controllers — same pattern as the Tk backend.

Translates canvas events into model commands. Each tool implements the same
minimal interface (``on_click(hit, button)``, ``on_motion(hit)``,
``description``). The MainWindow switches the active tool when the user
clicks a toolbar action or presses a keyboard shortcut.
"""

from __future__ import annotations

from typing import Optional, Protocol

from .canvas import HitResult
from ..gui_common.commands import AddElementCmd, AddNodeCmd, DeleteElementCmd, DeleteNodeCmd


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
    def set_element_preview(self, start_node_id: int, end_x: float,
                            end_y: float, kind: str) -> None: ...
    def clear_element_preview(self) -> None: ...


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
        "Select: left-click a node or element to view its details. "
        "Right-click for edit actions."
    )

    def on_click(self, hit: HitResult, button: str) -> None:
        if button == "right":
            if hit.node_id is not None:
                self.host.show_node_menu(hit.node_id)
            elif hit.element_id is not None:
                self.host.show_element_menu(hit.element_id)
        elif button == "left":
            if hit.node_id is not None:
                self.host.show_node_details(hit.node_id)
            elif hit.element_id is not None:
                self.host.show_element_details(hit.element_id)


class NodeTool(Tool):
    name = "node"
    description = "Node tool: click on the grid to place a node."

    def on_click(self, hit: HitResult, button: str) -> None:
        if button != "left":
            return
        if hit.node_id is not None:
            self.host.set_status(f"Node {hit.node_id} already at this location.")
            return
        self.host.execute(AddNodeCmd(x=hit.x, y=hit.y))


class _PairTool(Tool):
    def __init__(self, host: _Host, kind: str) -> None:
        super().__init__(host)
        self.kind = kind
        self._first: Optional[int] = None

    @property
    def description(self) -> str:
        if self._first is None:
            return f"{self.kind.capitalize()} tool: click the start node."
        return (f"{self.kind.capitalize()} tool: click the end node "
                f"(first = node {self._first}).")

    def deactivate(self) -> None:
        self._first = None
        self.host.clear_element_preview()

    def on_click(self, hit: HitResult, button: str) -> None:
        if button != "left":
            return
        if hit.node_id is None:
            self.host.set_status(
                f"Click an existing node to "
                f"{'start' if self._first is None else 'finish'} the element."
            )
            return
        if self._first is None:
            self._first = hit.node_id
            self.host.set_element_preview(hit.node_id, hit.x, hit.y, self.kind)
            self.host.set_status(self.description)
            return
        if hit.node_id == self._first:
            self.host.set_status("Start and end nodes must differ.")
            return
        self.host.clear_element_preview()
        self.host.open_element_dialog_for_pair(self._first, hit.node_id, self.kind)
        self._first = None

    def on_motion(self, hit: HitResult) -> None:
        if self._first is None:
            return
        self.host.set_element_preview(self._first, hit.x, hit.y, self.kind)


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
