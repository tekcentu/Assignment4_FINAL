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
        first_split_target: tuple[int, float, float] | None = None,
        second_split_target: tuple[int, float, float] | None = None,
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
    # v0.13.0 — multi-select / box-select
    def toggle_node_in_selection(self, node_id: int) -> None: ...
    def toggle_element_in_selection(self, element_id: int) -> None: ...
    def set_drag_rect(
        self, x0: float, y0: float, x1: float, y1: float,
        is_crossing: bool,
    ) -> None: ...
    def clear_drag_rect(self) -> None: ...
    def apply_box_select(
        self,
        rect: tuple[float, float, float, float],
        shift: bool,
        is_crossing: bool,
    ) -> None: ...
    def select_to_neutral_mode(self) -> None: ...
    # 3D work plane (v0.32) — duck-typed on the host; helpers below
    # fall back to legacy behaviour when the host doesn't provide them.
    def working_z(self) -> float: ...
    def can_edit_geometry(self) -> bool: ...


def _host_working_z(host) -> float:
    """Working depth for new geometry (0.0 on legacy hosts)."""
    getter = getattr(host, "working_z", None)
    return float(getter()) if callable(getter) else 0.0


def _host_can_edit(host) -> bool:
    """False when the canvas is in a display-only view (isometric)."""
    getter = getattr(host, "can_edit_geometry", None)
    return bool(getter()) if callable(getter) else True


_VIEW_BLOCKED_MSG = (
    "Geometry editing works on the XY work plane only — switch back "
    "via View → Work plane → XY. Use View → Working depth to build "
    "at other z levels, and Model → Connect selected nodes for "
    "out-of-plane members."
)


class Tool:
    name: str = ""
    description: str = ""

    def __init__(self, host: _Host) -> None:
        self.host = host

    def activate(self) -> None:
        self.host.set_status(self.description)

    def deactivate(self) -> None:
        pass

    def on_click(
        self, hit: HitResult, button: str,
        *,
        press_px: tuple[float, float] = (0.0, 0.0),
        shift: bool = False,
    ) -> None:
        pass

    def on_motion(
        self, hit: HitResult,
        *,
        cursor_px: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        pass

    def on_release(
        self, hit: HitResult, button: str,
        *,
        release_px: tuple[float, float] = (0.0, 0.0),
        shift: bool = False,
    ) -> None:
        pass

    def on_key(self, key: str) -> None:
        """Handle named key events (currently only ``"escape"``).
        Tools that have cancellable state should override."""
        pass


_DRAG_THRESHOLD_PX = 4.0


class SelectTool(Tool):
    """Press / drag / release state machine for the Select tool.

    A short click selects a single object exclusively. Shift-click
    toggles an object in/out of the selection without disturbing the
    rest. Dragging past the pixel threshold opens a CAD-style box
    selection — left-to-right is Window (only fully enclosed objects),
    right-to-left is Crossing (anything the rect touches). ESC clears
    selection or cancels an active drag.
    """

    name = "select"
    description = (
        "Select: click to pick one object, Shift-click to add/remove, "
        "drag a box (left→right Window, right→left Crossing). "
        "Right-click an element for the inspector."
    )

    def __init__(self, host: _Host) -> None:
        super().__init__(host)
        self._press_world: tuple[float, float] | None = None
        self._press_px: tuple[float, float] | None = None
        self._press_hit: HitResult | None = None
        self._press_shift: bool = False
        self._dragging: bool = False

    def deactivate(self) -> None:
        self._reset_drag_state()
        self.host.clear_drag_rect()

    def _reset_drag_state(self) -> None:
        self._press_world = None
        self._press_px = None
        self._press_hit = None
        self._press_shift = False
        self._dragging = False

    def on_click(
        self, hit: HitResult, button: str,
        *,
        press_px: tuple[float, float] = (0.0, 0.0),
        shift: bool = False,
    ) -> None:
        if button == "right":
            # Right-click is routed by MainWindow._on_canvas_click to
            # the inspector / context menus regardless of active tool.
            return
        if button != "left":
            return
        # Record press; the actual select/box-select decision happens
        # in on_release so a drag isn't double-counted as click+drag.
        self._press_world = (hit.x, hit.y)
        self._press_px = press_px
        self._press_hit = hit
        self._press_shift = shift
        self._dragging = False

    def on_motion(
        self, hit: HitResult,
        *,
        cursor_px: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if self._press_world is None or self._press_px is None:
            return
        dx = cursor_px[0] - self._press_px[0]
        dy = cursor_px[1] - self._press_px[1]
        if not self._dragging and (dx * dx + dy * dy) < (
            _DRAG_THRESHOLD_PX * _DRAG_THRESHOLD_PX
        ):
            return
        self._dragging = True
        # Direction in pixel space — robust against axis flips / zoom.
        is_crossing = cursor_px[0] < self._press_px[0]
        x0, y0 = self._press_world
        self.host.set_drag_rect(x0, y0, hit.x, hit.y, is_crossing)

    def on_release(
        self, hit: HitResult, button: str,
        *,
        release_px: tuple[float, float] = (0.0, 0.0),
        shift: bool = False,
    ) -> None:
        if button != "left":
            return
        if self._press_world is None or self._press_px is None:
            return
        was_dragging = self._dragging
        press_shift = self._press_shift
        if was_dragging:
            is_crossing = release_px[0] < self._press_px[0]
            x0, y0 = self._press_world
            self.host.clear_drag_rect()
            self.host.apply_box_select(
                (x0, y0, hit.x, hit.y), press_shift, is_crossing,
            )
            self._reset_drag_state()
            return
        # Single click — use the modifier captured at press time so
        # later modifier changes during the click don't affect the
        # decision.
        target_hit = self._press_hit or hit
        self._reset_drag_state()
        self._apply_click_select(target_hit, press_shift)

    def _apply_click_select(self, hit: HitResult, shift: bool) -> None:
        if shift:
            if hit.node_id is not None:
                self.host.toggle_node_in_selection(hit.node_id)
            elif hit.element_id is not None:
                self.host.toggle_element_in_selection(hit.element_id)
            # Shift on empty space: keep current selection.
            return
        if hit.node_id is not None:
            self.host.select_node(hit.node_id)
        elif hit.element_id is not None:
            self.host.select_element(hit.element_id)
        else:
            self.host.clear_selection()

    def on_key(self, key: str) -> None:
        if key != "escape":
            return
        if self._dragging:
            # Cancel an in-progress drag without disturbing the
            # selection that was live before the press.
            self.host.clear_drag_rect()
            self._reset_drag_state()
            return
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

    def on_click(
        self, hit: HitResult, button: str,
        *,
        press_px: tuple[float, float] = (0.0, 0.0),
        shift: bool = False,
    ) -> None:
        if button != "left":
            return
        if not _host_can_edit(self.host):
            self.host.set_status(_VIEW_BLOCKED_MSG)
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
        self.host.execute(AddNodeCmd(
            x=hit.x, y=hit.y, z=_host_working_z(self.host),
        ))


class _PairTool(Tool):
    def __init__(self, host: _Host, kind: str) -> None:
        super().__init__(host)
        self.kind = kind
        # v0.11.0 (post-PR21): the per-endpoint stash carries the
        # world coords, a node-id hint (None if the click came from a
        # split target or free space), and an optional split target
        # `(parent_element_id, projected_x, projected_y)` which gets
        # plumbed through the dialog so the composite can run the
        # split on accept rather than eagerly.
        #
        # Fields: (x, y, node_id_hint, split_target_or_None)
        self._first: Optional[
            tuple[float, float, int | None, tuple[int, float, float] | None]
        ] = None

    @property
    def description(self) -> str:
        if self._first is None:
            return (
                f"{self.kind.capitalize()} tool: click the start point. "
                "Snaps to nodes; a new node is created if you click empty space."
            )
        first_id = self._first[2]
        first_split = self._first[3]
        if first_id is not None:
            ref = f"node {first_id}"
        elif first_split is not None:
            ref = f"on element {first_split[0]}"
        else:
            ref = "point"
        return (
            f"{self.kind.capitalize()} tool: click the end point "
            f"(start = {ref})."
        )

    def deactivate(self) -> None:
        self._first = None
        self.host.clear_element_preview()

    def _resolve_endpoint(
        self, hit: HitResult,
    ) -> tuple[float, float, int | None, tuple[int, float, float] | None]:
        """Resolve a member-draw click into ``(x, y, node_id, split_target)``.

        Deferred-split semantics (v0.11.0 follow-up): this helper no
        longer calls ``host.execute``. It only *classifies* the click
        and returns the data the dialog flow needs. The actual
        :class:`SplitElementCmd` and :class:`AddMemberCmd` are bundled
        into a single :class:`DrawMemberWithSplitsCmd` on dialog
        accept, so the whole draw collapses to one undo step and
        cancelling the dialog leaves the model untouched.

        Returns one of:

        - ``(hit.x, hit.y, hit.node_id, None)`` — endpoint is at a
          snapped node or in free space. No split needed.
        - ``(proj_x, proj_y, None, (parent_element_id, proj_x, proj_y))``
          — endpoint will split the named element on dialog accept.

        The previous v0.11.0 cut had a third "None ⇒ cancel" return
        path for split failures; that no longer applies because we
        don't execute anything here. Member-load blocks now surface
        from :class:`DrawMemberWithSplitsCmd.do` via the host's
        existing ValueError handler.
        """
        target = _split_target_for(hit, self.host)
        if target is None:
            return (hit.x, hit.y, hit.node_id, None)
        elem_id, x_world, y_world = target
        return (x_world, y_world, None, (elem_id, x_world, y_world))

    def on_key(self, key: str) -> None:
        if key != "escape":
            return
        # Atomic cancel: drop pending-first state + preview WITHOUT
        # touching the model or the undo stack. The MainWindow ESC
        # handler will then switch back to the Select tool.
        self._first = None
        self.host.clear_element_preview()

    def on_click(
        self, hit: HitResult, button: str,
        *,
        press_px: tuple[float, float] = (0.0, 0.0),
        shift: bool = False,
    ) -> None:
        if button != "left":
            return
        if not _host_can_edit(self.host):
            self.host.set_status(_VIEW_BLOCKED_MSG)
            return
        if self._first is None:
            self._first = self._resolve_endpoint(hit)
            first_x, first_y, first_id, first_split = self._first
            if first_id is not None:
                self.host.set_element_preview(
                    first_id, first_x, first_y, self.kind,
                )
            else:
                # Free space OR a deferred split target both anchor
                # the rubber-band preview from the projected world
                # coordinate — no node id is needed for the preview.
                # Visible diff from the eager-split version: the
                # parent element stays whole during the preview phase
                # (split happens on dialog accept, not on click 1).
                self.host.set_element_preview_free(
                    first_x, first_y, first_x, first_y, self.kind,
                )
            self.host.set_status(self.description)
            return
        first_x, first_y, first_id, first_split = self._first
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
        second_x, second_y, second_id, second_split = self._resolve_endpoint(hit)
        self.host.clear_element_preview()
        self.host.open_element_dialog_for_member(
            first_x=first_x, first_y=first_y, first_node_id=first_id,
            second_x=second_x, second_y=second_y, second_node_id=second_id,
            kind=self.kind,
            first_split_target=first_split,
            second_split_target=second_split,
        )
        self._first = None

    def on_motion(
        self, hit: HitResult,
        *,
        cursor_px: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if self._first is None:
            return
        first_x, first_y, first_id, _first_split = self._first
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

    def on_click(
        self, hit: HitResult, button: str,
        *,
        press_px: tuple[float, float] = (0.0, 0.0),
        shift: bool = False,
    ) -> None:
        if button != "left":
            return
        if hit.node_id is None:
            self.host.set_status("Support tool: click an existing node, not empty space.")
            return
        self.host.show_node_menu(hit.node_id, action="support")


class NodalLoadTool(Tool):
    name = "nodal_load"
    description = "Nodal load tool: click a node to add/edit its load."

    def on_click(
        self, hit: HitResult, button: str,
        *,
        press_px: tuple[float, float] = (0.0, 0.0),
        shift: bool = False,
    ) -> None:
        if button != "left":
            return
        if hit.node_id is None:
            self.host.set_status("Nodal-load tool: click an existing node, not empty space.")
            return
        self.host.show_node_menu(hit.node_id, action="nodal_load")


class MemberLoadTool(Tool):
    name = "member_load"
    description = "Member load tool: click an element to add a load."

    def on_click(
        self, hit: HitResult, button: str,
        *,
        press_px: tuple[float, float] = (0.0, 0.0),
        shift: bool = False,
    ) -> None:
        if button != "left":
            return
        if hit.element_id is None:
            self.host.set_status("Member-load tool: click an element line, not empty space or a node.")
            return
        self.host.show_element_menu(hit.element_id, action="member_load")


class DeleteTool(Tool):
    name = "delete"
    description = "Delete tool: click a node or element to delete it."

    def on_click(
        self, hit: HitResult, button: str,
        *,
        press_px: tuple[float, float] = (0.0, 0.0),
        shift: bool = False,
    ) -> None:
        if button != "left":
            return
        if hit.node_id is not None:
            self.host.execute(DeleteNodeCmd(node_id=hit.node_id))
        elif hit.element_id is not None:
            self.host.execute(DeleteElementCmd(elem_id=hit.element_id))
        else:
            self.host.set_status("Delete tool: click a node or element to remove it.")
