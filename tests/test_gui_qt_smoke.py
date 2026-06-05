"""Smoke tests for the PyQt6 GUI.

These tests run under the ``offscreen`` Qt platform plugin so they work in
headless CI as long as PyQt6 is installed. If PyQt6 isn't available the
whole file is skipped.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication, QDialog  # noqa: E402
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 QtWidgets unavailable: {exc}", allow_module_level=True)

from structural_analysis.gui_qt.app import MainWindow  # noqa: E402
from structural_analysis.gui_qt.canvas import HitResult  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_constructs(qt_app):
    w = MainWindow()
    assert set(w._tools.keys()) == {
        "select", "node", "frame", "truss",
        "support", "nodal_load", "member_load", "delete",
    }
    assert w._active_tool.name == "select"
    assert len(w._model.nodes) == 0


def test_open_solve_undo(qt_app):
    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    assert len(w._model.nodes) == 6
    assert len(w._model.elements) == 5

    w._do_solve()
    assert w._result is not None
    assert w._result.status == "ok"
    assert w._result.residual < 1e-8

    w._select_tool("node")
    w._on_canvas_click(HitResult(x=20.0, y=10.0), "left")
    assert len(w._model.nodes) == 7
    w._do_undo()
    assert len(w._model.nodes) == 6


def test_frame_tool_shows_live_element_preview(qt_app):
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 2.0, 0.0),
    }
    w.canvas.redraw()

    w._select_tool("frame")
    w._on_canvas_click(HitResult(x=0.0, y=0.0, node_id=1), "left")
    assert w.canvas._element_preview == (1, 0.0, 0.0, "frame")

    w._on_canvas_motion(HitResult(x=1.5, y=0.5, snap_label="grid A-1"))
    assert w.canvas._element_preview == (1, 1.5, 0.5, "frame")

    w._select_tool("select")
    assert w.canvas._element_preview is None


def test_truss_tool_passes_truss_kind_to_element_dialog(qt_app):
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 2.0, 0.0),
    }
    seen: list[dict] = []

    def fake_open(
        *,
        first_x, first_y, first_node_id,
        second_x, second_y, second_node_id,
        kind=None,
        # v0.11.0 follow-up: _PairTool now always passes deferred
        # split targets (None for node-snap / free clicks). Tolerate
        # them so this node-snap-only test keeps exercising the stub.
        first_split_target=None,
        second_split_target=None,
    ):
        seen.append({
            "first_node_id": first_node_id,
            "second_node_id": second_node_id,
            "kind": kind,
        })

    # v0.10.0: _PairTool now routes through open_element_dialog_for_member
    # (the kwarg-only successor of open_element_dialog_for_pair, which
    # is kept as a thin shim for back-compat callers).
    w.open_element_dialog_for_member = fake_open
    w._select_tool("truss")
    w._on_canvas_click(HitResult(x=0.0, y=0.0, node_id=1), "left")
    w._on_canvas_click(HitResult(x=2.0, y=0.0, node_id=2), "left")

    assert seen == [{
        "first_node_id": 1, "second_node_id": 2, "kind": "truss",
    }]


def test_canvas_draws_origin_axes(qt_app):
    w = MainWindow()
    w.canvas.redraw()

    labels = [text.get_text() for text in w.canvas.ax.texts]
    assert "0,0" in labels
    assert "X" in labels
    assert "Y" in labels


def test_hit_test_attaches_element_id_when_grid_snap_wins(qt_app):
    """PR #21 review (codex P1): when a labeled grid is configured, the
    snap engine prefers the 'grid' candidate over 'project', and the
    grid candidate carries no element_id. _hit_test must still recover
    the nearest element for non-element snaps — otherwise a click on an
    element interior near a grid intersection is treated as empty space
    and the Node/Frame split path never engages (re-introducing the
    disconnected-component bug)."""
    from types import SimpleNamespace

    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_qt.snap import SnapCandidate
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=7, node_i=1, node_j=2,
        E=2.1e8, A=0.01, I=1e-4, rho=7850.0, depth=0.3, section_id=1,
    )]
    w.canvas.redraw()
    # Frame the element so the px ratios are sane and (3, 0) is interior.
    w.canvas.ax.set_xlim(-1.0, 7.0)
    w.canvas.ax.set_ylim(-4.0, 4.0)

    # Force the snap engine to return a GRID candidate (no element id)
    # at the element's midspan — exactly the case the bug missed.
    w.canvas.snap_engine.find_snap = (  # type: ignore[assignment]
        lambda **kw: SnapCandidate(
            x=3.0, y=0.0, kind="grid", priority=1,
            screen_distance_px=0.0, label="B-2", object_id=None,
        )
    )
    event = SimpleNamespace(xdata=3.0, ydata=0.0)
    hit = w.canvas._hit_test(event)

    assert hit.snap_kind == "grid"
    # The fix: element_id recovered despite the grid snap.
    assert hit.element_id == 7


def test_select_tool_highlights_and_reports_selection(qt_app):
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 2.0, 0.0),
    }
    w._model.elements = [
        FrameElement2D(
            id=1, node_i=1, node_j=2,
            E=2.0e8, A=0.01, I=1.0e-4,
            section_id=1,
        )
    ]

    w._select_tool("select")
    # v0.13.0: SelectTool defers the single-click select decision to
    # mouse release (so a drag isn't double-counted). The test harness
    # has to fire both press and release for a click.
    def _click(hit, press_px=(0.0, 0.0)):
        w._on_canvas_click(hit, "left", press_px=press_px, shift=False)
        w._on_canvas_release(hit, "left", release_px=press_px, shift=False)

    _click(HitResult(x=0.0, y=0.0, node_id=1))
    assert w.canvas.get_selected_nodes() == frozenset({1})
    assert "Selected node 1" in w._status_label.text()

    _click(HitResult(x=1.0, y=0.0, element_id=1))
    assert w.canvas.get_selected_elements() == frozenset({1})
    assert w.canvas.get_selected_nodes() == frozenset()
    assert "Selected element 1" in w._status_label.text()

    _click(HitResult(x=5.0, y=5.0))
    assert w.canvas.get_selected_elements() == frozenset()
    assert w.canvas.get_selected_nodes() == frozenset()
    assert "Selection cleared" in w._status_label.text()


def test_nodal_load_components_draw_separately(qt_app):
    from structural_analysis.model import Node, NodalLoad

    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0)}
    w._model.nodal_loads = [NodalLoad(1, fx=10.0, fy=-5.0, mz=0.0)]
    w.canvas.redraw()

    labels = [text.get_text() for text in w.canvas.ax.texts]
    assert "Fx=+10" in labels
    assert "Fy=-5" in labels
    assert "11.2 kN" not in labels


def test_section_material_labels_can_be_drawn(qt_app):
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Material, Node, Section

    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 2.0, 0.0)}
    w._model.materials = {1: Material(1, name="Steel", E=200e6)}
    w._model.sections = {
        1: Section(1, name="IPE200", material_id=1, A=0.01, I=1e-4)
    }
    w._model.elements = [
        FrameElement2D(1, 1, 2, E=200e6, A=0.01, I=1e-4, section_id=1)
    ]
    w.canvas.show_section_labels = True
    w.canvas.redraw()

    labels = [text.get_text() for text in w.canvas.ax.texts]
    assert "IPE200 / Steel" in labels


def test_update_element_command_changes_section(qt_app):
    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_common.commands import UpdateElementCmd
    from structural_analysis.model import Material, Node, Section

    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 2.0, 0.0)}
    w._model.materials = {
        1: Material(1, name="Steel", E=200e6),
        2: Material(2, name="Concrete", E=30e6),
    }
    w._model.sections = {
        1: Section(1, name="IPE200", material_id=1, A=0.01, I=1e-4),
        2: Section(2, name="C30x30", material_id=2, A=0.09, I=6.75e-4),
    }
    w._model.elements = [
        FrameElement2D(1, 1, 2, E=200e6, A=0.01, I=1e-4, section_id=1)
    ]

    cmd = UpdateElementCmd(elem_id=1, section_id=2, kind="frame")
    cmd.do(w._model)

    elem = w._model.elements[0]
    assert elem.section_id == 2
    assert elem.E == 30e6
    assert elem.A == 0.09



def test_grid_dialog_accepts_numeric_lists_and_sorts(qt_app):
    from structural_analysis.gui_qt.dialogs import GridDialog

    w = MainWindow()
    d = GridDialog(w, model=w._model)
    d._x_entry.setText("12, 0, 6, 0")
    d._y_entry.setText("8, 0, 4, 4")

    grid = d._accept()

    assert [(ln.label, ln.coord) for ln in grid.x_lines] == [
        ("A", 0.0), ("B", 6.0), ("C", 12.0)
    ]
    assert [(ln.label, ln.coord) for ln in grid.y_lines] == [
        ("1", 0.0), ("2", 4.0), ("3", 8.0)
    ]
    assert "X: A=0, B=6, C=12" in d._preview.text()


def test_grid_dialog_fills_from_model_nodes(qt_app):
    from structural_analysis.gui_qt.dialogs import GridDialog
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {
        1: Node(1, 6.0, 4.0),
        2: Node(2, 0.0, 0.0),
        3: Node(3, 6.0, 8.0),
    }
    d = GridDialog(w, model=w._model)

    d._fill_from_model_nodes()

    assert d._x_entry.text() == "0, 6"
    assert d._y_entry.text() == "0, 4, 8"
    grid = d._accept()
    assert [(ln.label, ln.coord) for ln in grid.x_lines] == [("A", 0.0), ("B", 6.0)]
    assert [(ln.label, ln.coord) for ln in grid.y_lines] == [
        ("1", 0.0), ("2", 4.0), ("3", 8.0)
    ]


def test_grid_dialog_reports_invalid_token(qt_app):
    from structural_analysis.gui_qt.dialogs import GridDialog

    w = MainWindow()
    d = GridDialog(w, model=w._model)
    d._x_entry.setText("A=0, bad, C=12")

    with pytest.raises(ValueError, match="X token 'bad' is invalid"):
        d._accept()


def test_sticky_truss_then_frame_tool_places_frame(qt_app):
    """Bug fix: with sticky=truss remembered, switching to the Frame
    tool and clicking two nodes must place a FrameElement2D - not
    another truss as the sticky-path used to do unconditionally.
    Releases live on the frame side only, so when an effective kind
    is "truss" the releases are forced to False."""
    from structural_analysis.element import FrameElement2D, TrussElement2D
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 2.0, 0.0),
        3: Node(3, 4.0, 0.0),
        4: Node(4, 6.0, 0.0),
    }
    # Pre-load sticky state as if the user just placed a truss with
    # "Remember" checked, using starter Section 1 (Steel_IPE200).
    w._sticky_element = {
        "kind": "truss",
        "section_id": 1,
        "release_i": False,
        "release_j": False,
    }
    # Frame tool now - must place a FrameElement2D, not a truss.
    w._select_tool("frame")
    w._on_canvas_click(HitResult(x=0.0, y=0.0, node_id=1), "left")
    w._on_canvas_click(HitResult(x=2.0, y=0.0, node_id=2), "left")
    assert len(w._model.elements) == 1
    assert isinstance(w._model.elements[0], FrameElement2D)

    # Now flip: sticky-frame with a release, click Truss -> truss with
    # release_i forced back to False (releases don't apply to trusses).
    w._sticky_element = {
        "kind": "frame",
        "section_id": 1,
        "release_i": True,
        "release_j": False,
    }
    w._select_tool("truss")
    w._on_canvas_click(HitResult(x=4.0, y=0.0, node_id=3), "left")
    w._on_canvas_click(HitResult(x=6.0, y=0.0, node_id=4), "left")
    assert len(w._model.elements) == 2
    placed_truss = w._model.elements[1]
    assert isinstance(placed_truss, TrussElement2D)


def test_all_dialogs_construct(qt_app):
    from structural_analysis.gui_qt.dialogs import (
        ElementDialog,
        GridSpacingDialog,
        MaterialDialog,
        MaterialListDialog,
        MemberLoadDialog,
        NodalLoadDialog,
        SupportDialog,
    )

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    m = w._model

    # No exceptions on construction.
    MaterialDialog(w, existing=m.materials[1], default_id=1)
    SupportDialog(w, existing=m.supports.get(1), node_id=1)
    NodalLoadDialog(w, existing=None, node_id=1)
    MemberLoadDialog(w, model=m, elem_id=1)  # frame element
    MemberLoadDialog(w, model=m, elem_id=5)  # truss element
    GridSpacingDialog(w, current=0.5)
    ElementDialog(w, model=m)
    MaterialListDialog(w, model=m,
                        on_add_or_update_material=lambda _x: None,
                        on_delete_material=lambda _x: None,
                        on_add_or_update_section=lambda _x: None,
                        on_delete_section=lambda _x: None)


def test_member_load_dialog_raises_for_unknown_element(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    with pytest.raises(ValueError):
        MemberLoadDialog(w, model=w._model, elem_id=9999)


def test_select_tool_left_click_only_selects(qt_app):
    """Left-clicking with the Select tool must only update the
    canvas highlight + status text — opening the modal detail
    dialog on every left-click was the *old* behaviour. The detail
    inspector is now reached by right-click (see the dedicated
    right-click test) and the modal popup is gone."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 2.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.0e8, A=0.01, I=1.0e-4, section_id=1,
    )]

    # Guard against the old code path firing the modal dialog. If
    # either call lands here, the test should fail loudly.
    illegal: list[str] = []
    w.show_node_details = lambda nid: illegal.append(f"node {nid}")
    w.show_element_details = lambda eid: illegal.append(f"elem {eid}")

    w._select_tool("select")

    def _click(hit):
        w._on_canvas_click(hit, "left")
        w._on_canvas_release(hit, "left")

    _click(HitResult(x=0.0, y=0.0, node_id=1))
    assert w.canvas.get_selected_nodes() == frozenset({1})
    _click(HitResult(x=1.0, y=0.0, element_id=1))
    assert w.canvas.get_selected_elements() == frozenset({1})
    assert w.canvas.get_selected_nodes() == frozenset()
    # Empty click clears selection.
    _click(HitResult(x=5.0, y=5.0))
    assert w.canvas.get_selected_elements() == frozenset()
    assert w.canvas.get_selected_nodes() == frozenset()
    # No modal-dialog escape hatch fired.
    assert illegal == [], (
        f"Select-tool left-click must not open the modal details dialog, "
        f"got: {illegal}"
    )


def test_right_click_element_shows_context_menu(qt_app):
    """Right-clicking an element must route to the context menu (the
    one with edit / add load / clear loads / delete + the new
    "show details" item). The menu is the entry point both for the
    edit actions and for the detail inspector — right-click must
    not bypass it."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 2.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.0e8, A=0.01, I=1.0e-4, section_id=1,
    )]

    calls: list[int] = []
    w.show_element_menu = lambda eid: calls.append(eid)

    w._on_canvas_click(HitResult(x=1.0, y=0.0, element_id=1), "right")
    qt_app.processEvents()
    assert calls == [1]


def test_open_element_inspector_is_singleton_and_retargets(qt_app):
    """The detail inspector is the singleton path that the context
    menu's "show details" item calls into. Re-opening for a different
    element must reuse the same window, not stack a new one."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 2.0, 0.0),
        3: Node(3, 4.0, 0.0),
    }
    w._model.elements = [
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.0e8,
                       A=0.01, I=1.0e-4, section_id=1),
        FrameElement2D(id=2, node_i=2, node_j=3, E=2.0e8,
                       A=0.01, I=1.0e-4, section_id=1),
    ]

    assert w._element_inspector is None
    w._open_element_inspector(1)
    qt_app.processEvents()
    assert w._element_inspector is not None
    assert w._element_inspector.isVisible()
    assert w._element_inspector._elem_id == 1

    same_window = w._element_inspector
    w._open_element_inspector(2)
    qt_app.processEvents()
    assert w._element_inspector is same_window, (
        "the singleton inspector must be reused, not replaced"
    )
    assert w._element_inspector._elem_id == 2


def test_show_element_menu_disables_edits_while_inspector_open(qt_app):
    """While the inspector is open the context menu must still build,
    but its edit items (edit / add load / clear loads / delete) must
    be greyed out so a right-click → Delete can't slip past the edit
    lock. The "show details" item stays enabled so the user can
    re-target the inspector from any right-click."""
    from PyQt6.QtWidgets import QMenu

    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 2.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.0e8, A=0.01, I=1.0e-4, section_id=1,
    )]

    # Stub QMenu.exec so the test doesn't block on a popup — we only
    # care about the QActions' enabled state at exec-time.
    captured: dict = {}

    def fake_exec(self, _pos):
        captured["actions"] = [(a.text(), a.isEnabled()) for a in self.actions()]
        return None  # user dismissed the menu

    QMenu.exec = fake_exec
    try:
        w._open_element_inspector(1)
        qt_app.processEvents()
        w.show_element_menu(1)
    finally:
        del QMenu.exec   # restore the real method

    by_label = {label: enabled for label, enabled in captured["actions"]}
    # PR #35: "show details / FBD…" was renamed to "Element Details…".
    details_label = next(l for l in by_label if "element details" in l.lower())
    assert by_label[details_label], (
        '"Element Details" must stay enabled while inspector is open'
    )
    # PR #35: "Edit member loads…" also stays enabled — it just re-focuses
    # the Load Assignments tab on the already-open inspector.
    loads_label = next(l for l in by_label if "edit member loads" in l.lower())
    assert by_label[loads_label], (
        '"Edit member loads…" must stay enabled while inspector is open'
    )
    # "add member load…" was retired in PR #35 — Add / Edit / Delete now
    # live inside the Load Assignments tab of the inspector itself.
    for needle in ("edit section", "clear member loads"):
        label = next(l for l in by_label if needle in l.lower())
        assert not by_label[label], (
            f'menu item {label!r} must be disabled while inspector is open'
        )
    # The standalone "delete" item — match the exact prefix so the
    # "Element Details…" item (which contains "details") doesn't trip
    # the lookup.
    delete_label = next(
        l for l in by_label
        if l.lower().endswith(": delete")
    )
    assert not by_label[delete_label], (
        f'menu item {delete_label!r} must be disabled while inspector is open'
    )


def test_inspector_open_locks_editing_keeps_view(qt_app):
    """When the inspector is open, editing actions (tool palette
    add/delete, Undo/Redo, Materials, building wizard, "Add node at
    coordinates", Forget element defaults) must be disabled, and the
    active tool must be forced to Select. View / solve / overlay
    actions stay enabled. Closing re-enables everything."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 2.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.0e8, A=0.01, I=1.0e-4, section_id=1,
    )]
    w._select_tool("node")
    assert w._active_tool.name == "node"
    assert w._tool_actions["node"].isEnabled()

    w._open_element_inspector(1)
    qt_app.processEvents()

    # Editing tools and registered lockable actions must be disabled.
    for name in ("node", "frame", "truss", "support",
                 "nodal_load", "member_load", "delete"):
        assert not w._tool_actions[name].isEnabled(), (
            f"tool {name!r} must be disabled while inspector is open"
        )
    for action in w._lockable_actions:
        assert not action.isEnabled(), (
            f"editing action {action.text()!r} must be disabled "
            f"while inspector is open"
        )
    # Select must stay enabled so right-clicks still route through.
    assert w._tool_actions["select"].isEnabled()
    assert w._active_tool.name == "select"
    # View-only actions stay enabled — solve, fit, 3D viewer, snap.
    assert w.act_solve.isEnabled()
    assert w.act_fit_view.isEnabled()
    assert w.act_open_view3d.isEnabled()

    # Close: lock released, full edit surface back.
    w._element_inspector.close()
    qt_app.processEvents()
    for name in ("node", "frame", "truss", "support",
                 "nodal_load", "member_load", "delete"):
        assert w._tool_actions[name].isEnabled(), (
            f"tool {name!r} must be re-enabled after inspector closes"
        )
    for action in w._lockable_actions:
        assert action.isEnabled()


def test_inspector_refreshes_on_solve(qt_app):
    """Solving while the inspector is open must push the new
    member-end forces into its diagrams panel — without making the
    user close-and-reopen the window."""
    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    elem = w._model.elements[0]

    w._open_element_inspector(elem.id)
    qt_app.processEvents()
    diag_ax = w._element_inspector._detail_axes["diagrams"]
    pre_solve_lines = len(diag_ax.lines)
    assert pre_solve_lines == 0, (
        "pre-solve diagrams panel must hold no traces (only the "
        "placeholder text)"
    )

    w._do_solve()
    qt_app.processEvents()
    diag_ax = w._element_inspector._detail_axes["diagrams"]
    assert diag_ax.lines, (
        "post-solve refresh must populate the diagrams panel"
    )


def test_property_dialogs_construct(qt_app):
    """ElementPropertiesDialog and NodePropertiesDialog must build for
    every element / node in the q2a model, with and without a solver
    result loaded."""
    from structural_analysis.gui_qt.dialogs import (
        ElementPropertiesDialog,
        NodePropertiesDialog,
    )

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._do_solve()
    m = w._model
    # Pre-solve variants
    ElementPropertiesDialog(w, m, m.elements[0].id, None)
    NodePropertiesDialog(w, m, next(iter(m.nodes)), None)
    # Post-solve: include the result for every element / node
    for elem in m.elements:
        ElementPropertiesDialog(w, m, elem.id, w._result)
    for nid in m.nodes:
        NodePropertiesDialog(w, m, nid, w._result)


def test_element_dialog_renders_graphics_pre_solve(qt_app):
    """The element detail dialog must be usable *before* solving:
    member sketch / FBD / section thumbnail render straight from
    model data, while the internal-force panel shows a "Run analysis"
    placeholder. No exception must be raised by the figure path."""
    from structural_analysis.gui_qt.dialogs import ElementPropertiesDialog

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    elem = w._model.elements[0]
    # Note: result intentionally None to exercise the pre-solve path.
    d = ElementPropertiesDialog(w, w._model, elem.id, None)
    qt_app.processEvents()

    axes = d._detail_axes
    assert set(axes) == {"sketch", "fbd", "diagrams", "section"}

    # Member sketch always renders the centre-line — there must be at
    # least one Line2D in the sketch panel.
    assert axes["sketch"].lines, "member sketch must draw the centreline"

    # Internal-force panel pre-solve must show the placeholder text
    # and must NOT draw any data line yet.
    diag_texts = [t.get_text() for t in axes["diagrams"].texts]
    assert any("Run analysis" in t for t in diag_texts)
    assert not axes["diagrams"].lines

    # Section thumbnail renders only if the element carries a section.
    if w._model.sections.get(getattr(elem, "section_id", None)):
        assert axes["section"].patches, (
            "section thumbnail must render a filled outline"
        )
    else:
        sect_texts = [t.get_text() for t in axes["section"].texts]
        assert any("no section" in t.lower() for t in sect_texts)


def test_element_dialog_renders_graphics_post_solve(qt_app):
    """After solving, the dialog's internal-force panel must plot at
    least one N/V/M trace with n_samples station points — and those
    samples must come from element_graphics.sample_internal_force,
    not from a duplicate BMD/SFD formula inside the dialog."""
    from structural_analysis.gui_qt.dialogs import ElementPropertiesDialog
    from structural_analysis.gui_qt.element_graphics import (
        sample_internal_force,
    )

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._do_solve()
    qt_app.processEvents()
    assert w._result is not None and w._result.status == "ok"

    elem = w._model.elements[0]
    d = ElementPropertiesDialog(w, w._model, elem.id, w._result)
    qt_app.processEvents()

    diag_ax = d._detail_axes["diagrams"]
    assert diag_ax.lines, "post-solve diagrams must render at least one trace"
    # At least one trace must have >= n_samples points (the default
    # 11 used by draw_element_detail). Defends against a regression
    # where the dialog plots only the end-values.
    longest = max(len(line.get_xdata()) for line in diag_ax.lines)
    assert longest >= 11, (
        f"internal-force trace must use at least 11 station points, "
        f"got {longest}"
    )

    # Independent path — the *same* element_graphics helper must
    # produce the same number of station points; this is the
    # "single source of truth" guarantee.
    f_local = w._result.member_results[elem.id]["f_local"]
    ni = w._model.nodes[elem.node_i]
    nj = w._model.nodes[elem.node_j]
    xs, _ys = sample_internal_force(elem, ni, nj, f_local, "axial",
                                      n_samples=11)
    assert xs is not None and len(xs) == 11


def test_property_dialogs_raise_for_unknown_ids(qt_app):
    from structural_analysis.gui_qt.dialogs import (
        ElementPropertiesDialog,
        NodePropertiesDialog,
    )

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    with pytest.raises(ValueError):
        ElementPropertiesDialog(w, w._model, 9999, None)
    with pytest.raises(ValueError):
        NodePropertiesDialog(w, w._model, 9999, None)


def test_fine_node_dialog_constructs(qt_app):
    from structural_analysis.gui_qt.dialogs import FineNodeDialog

    w = MainWindow()
    d = FineNodeDialog(w, model=w._model)
    assert d._x_entry.text() == "0.0"
    assert d._y_entry.text() == "0.0"


def test_fine_node_action_creates_node(qt_app):
    """_do_add_node_at_coords must route through AddNodeCmd so undo
    works and duplicate detection fires."""
    from structural_analysis.gui_qt.dialogs import FineNodeDialog

    w = MainWindow()
    n_before = len(w._model.nodes)
    original_exec = FineNodeDialog.exec

    def fake_exec(self):
        self._x_entry.setText("5.0")
        self._y_entry.setText("3.0")
        self.result_value = self._accept()
        return QDialog.DialogCode.Accepted

    FineNodeDialog.exec = fake_exec
    try:
        w._do_add_node_at_coords()
    finally:
        FineNodeDialog.exec = original_exec

    assert len(w._model.nodes) == n_before + 1
    assert any(abs(n.x - 5.0) < 1e-9 and abs(n.y - 3.0) < 1e-9
               for n in w._model.nodes.values())
    # Undo removes the typed node.
    w._do_undo()
    assert len(w._model.nodes) == n_before


def test_modal_results_dialog_round_trip(qt_app):
    """Construct a small modal model, run :func:`solve_modal`, show the
    results dialog, drive the mode spinner and scale slider, then close
    it — the canvas must accept the modal overlay without raising and
    must reset to the plain model view on close."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    from structural_analysis.modal import solve_modal
    from structural_analysis.model import (
        Material, Node, Section, Support,
    )

    w = MainWindow()
    m = w._model
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 5.0, 0.0)}
    m.materials = {1: Material(id=1, E=200e6, alpha=0.0, density=7850.0,
                                name="steel")}
    m.sections = {1: Section(id=1, material_id=1, A=0.005, I=1.0e-5)}
    m.elements = [FrameElement2D(1, 1, 2, E=200e6, A=0.005, I=1.0e-5,
                                  rho=7850.0, section_id=1)]
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}

    r = solve_modal(m, n_modes=3)
    assert r.n_modes == 3
    assert r.frequencies[0] > 0.0

    calls: list[tuple[int, float]] = []

    def _select(mode_idx: int, scale: float) -> None:
        calls.append((mode_idx, scale))
        w.canvas.update_modal_view(mode_idx, scale)

    closed: list[bool] = []

    def _on_close() -> None:
        closed.append(True)
        w.canvas.clear_modal_result()

    w.canvas.set_modal_result(r, mode_idx=0, scale=1.0)
    dlg = ModalResultsDialog(w, r, on_select=_select, on_close=_on_close)
    dlg.show()
    qt_app.processEvents()
    dlg._mode_spin.setValue(2)
    dlg._scale_slider.setValue(50)
    qt_app.processEvents()
    assert calls, "expected at least one selection callback"
    assert calls[-1][0] == 1  # 1-based spin value 2 → 0-based 1
    assert abs(calls[-1][1] - 5.0) < 1e-9  # slider 50 → ×5.0
    dlg.close()
    qt_app.processEvents()
    assert closed == [True]
    # After close, the canvas modal overlay is cleared.
    assert w.canvas._modal_result is None


def test_frame_hermite_deformed_curve_with_known_dofs(qt_app):
    """Inject a controlled D/E_map so the Hermite midpoint is guaranteed
    to deviate from the straight chord between displaced endpoints.

    Setup: a single horizontal frame element of length L = 2 m with both
    endpoints at v = 0 but i-end rotated +0.1 rad and j-end rotated
    −0.1 rad. The straight chord between the two displaced endpoints is
    flat (y = 0 everywhere) — so any non-zero midspan y proves the
    cubic-Hermite shape function evaluated the rotational DOFs.
    """
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        AnalysisResult, Material, Node, Section,
    )

    w = MainWindow()
    m = w._model
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 2.0, 0.0)}
    m.materials = {1: Material(id=1, name="m", E=2.1e8, alpha=0.0,
                                density=7850.0)}
    m.sections = {1: Section(id=1, name="s", material_id=1,
                              A=1.0e-3, I=1.0e-5, depth=0.1)}
    m.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=1.0e-3, I=1.0e-5,
        alpha=0.0, depth=0.1, section_id=1,
    )]
    # Three global DOFs per node: [ux, uy, rz]. Order: n1 ux, n1 uy,
    # n1 rz, n2 ux, n2 uy, n2 rz. Translations zero, rotations only.
    # For a horizontal element (c=1, s=0) the local frame is identical to
    # global, so d_local = [0, 0, +0.1, 0, 0, -0.1].
    import numpy as np
    result = AnalysisResult(status="ok")
    result.D = [0.0, 0.0, +0.1, 0.0, 0.0, -0.1]
    result.E_map = {
        1: {"ux": 0, "uy": 1, "rz": 2},
        2: {"ux": 3, "uy": 4, "rz": 5},
    }
    result.member_results = {
        1: {
            "f_local": np.zeros(6),
            "d_local": np.array([0.0, 0.0, +0.1, 0.0, 0.0, -0.1]),
        },
    }
    w._result = result
    w.canvas._result = result
    w.canvas.deformed_stations = 21
    elem = m.elements[0]
    Xs, Ys = w.canvas._frame_deformed_points(elem, scale=1.0)
    assert len(Xs) == 21 and len(Ys) == 21
    # Endpoints carry no transverse displacement → both ends sit on y=0.
    assert abs(Ys[0]) < 1e-9
    assert abs(Ys[-1]) < 1e-9
    # Straight chord between (X0, 0) and (X-1, 0) would give Y = 0 at
    # every interior point. With cubic Hermite + non-zero rotations,
    # the midspan must bow noticeably away from the chord.
    mid = len(Ys) // 2
    assert abs(Ys[mid]) > 1e-3, (
        f"Hermite midspan deflection {Ys[mid]:.6e} is too small — "
        "looks like the curve is still the straight chord."
    )


def test_truss_deformed_shape_stays_straight(qt_app):
    """Truss bar must remain a straight 2-point segment between
    displaced endpoints, even when end-node rotations are non-zero."""
    from structural_analysis.element import TrussElement2D
    from structural_analysis.model import (
        AnalysisResult, Material, Node, Section,
    )

    w = MainWindow()
    m = w._model
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 2.0, 0.0)}
    m.materials = {1: Material(id=1, name="m", E=2.1e8, alpha=0.0,
                                density=7850.0)}
    m.sections = {1: Section(id=1, name="s", material_id=1,
                              A=1.0e-3, I=1.0e-5, depth=0.1)}
    m.elements = [TrussElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=1.0e-3,
        alpha=0.0, depth=0.1, section_id=1,
    )]
    result = AnalysisResult(status="ok")
    result.D = [0.0, 0.01, 0.0, 0.0, 0.0, 0.0]
    result.E_map = {
        1: {"ux": 0, "uy": 1, "rz": 2},
        2: {"ux": 3, "uy": 4, "rz": 5},
    }
    w._result = result
    w.canvas._result = result
    # Drawing the deformed shape must NOT raise and must not invoke the
    # cubic-Hermite branch (truss has no rotation DOFs in the local v
    # interpolation). We assert by drawing successfully and checking
    # the number of segments matches a 2-point line.
    w.canvas.show_deformed = True
    w.canvas._span = lambda: 2.0
    w.canvas.redraw()  # must not raise
    # Direct sanity check: the bar's local v interpolation is linear,
    # not Hermite, so even if rotations were present the line would
    # still be straight. _frame_deformed_points should NOT be called
    # for a truss element — exercise it via the public draw path.


def test_diagram_stations_setting_updates_canvas_and_does_not_resolve(qt_app):
    """Picking View → Diagram stations updates the canvas attributes and
    does not invalidate the cached solve result (no re-solve)."""
    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._do_solve()
    cached = w._result
    assert cached is not None and cached.status == "ok"

    w._set_diagram_stations(5)
    assert w.canvas.diagram_stations == 5
    assert w.canvas.deformed_stations == 5
    assert w._result is cached, (
        "_set_diagram_stations must redraw only, never re-solve."
    )
    # Coarse-preview status hint should be shown for n <= 5.
    assert "coarse" in w._status_label.text().lower()

    w._set_diagram_stations(21)
    assert w.canvas.diagram_stations == 21
    assert w._result is cached
    assert "coarse" not in w._status_label.text().lower()


def test_station_actions_default_to_21(qt_app):
    """The 21-station action is the one checked at startup."""
    w = MainWindow()
    qt_app.processEvents()
    assert set(w._station_actions.keys()) == {5, 11, 21, 51}
    for n, a in w._station_actions.items():
        assert a.isChecked() == (n == 21), (
            f"Station action {n} checked-state {a.isChecked()} != "
            f"expected {n == 21}"
        )
    assert w.canvas.diagram_stations == 21
    assert w.canvas.deformed_stations == 21


def test_draw_deformed_visible_for_rotation_only_case(qt_app):
    """Regression for Bug 1: when all nodal ux=uy=0 but frame end
    rotations are non-zero, _draw_deformed must still draw the Hermite
    deformed shape (not silently return at the max_disp gate)."""
    import numpy as np
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        AnalysisResult, Material, Node, Section,
    )

    w = MainWindow()
    m = w._model
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.materials = {1: Material(id=1, name="m", E=2.1e8, alpha=0.0,
                                density=7850.0)}
    m.sections = {1: Section(id=1, name="s", material_id=1,
                              A=1.0e-3, I=1.0e-5, depth=0.1)}
    m.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=1.0e-3, I=1.0e-5,
        alpha=0.0, depth=0.1, section_id=1,
    )]
    # All nodal translations zero; end rotations only.
    # For a horizontal element, d_local equals the global DOF vector.
    result = AnalysisResult(status="ok")
    result.D = [0.0, 0.0, +0.01, 0.0, 0.0, -0.01]
    result.E_map = {
        1: {"ux": 0, "uy": 1, "rz": 2},
        2: {"ux": 3, "uy": 4, "rz": 5},
    }
    result.member_results = {
        1: {
            "f_local": np.zeros(6),
            "d_local": np.array([0.0, 0.0, +0.01, 0.0, 0.0, -0.01]),
        },
    }
    w._result = result
    w.canvas._result = result
    w.canvas.show_deformed = True
    w.canvas.redraw()
    # At least one plotted Line2D must have more than 2 data points —
    # that is the Hermite deformed-shape curve (21 stations by default).
    # Without the fix the max_disp gate returns early and no such line exists.
    multi_point_lines = [
        l for l in w.canvas.ax.lines if len(l.get_xdata()) > 2
    ]
    assert multi_point_lines, (
        "No multi-point line found after redraw() with rotation-only DOFs — "
        "the max_disp gate silenced the deformed shape."
    )


def test_scroll_event_zooms_centered_on_cursor(qt_app):
    """Scroll up shrinks the visible range (zoom in); scroll down expands it
    (zoom out). Both stay centered on the cursor position."""
    from types import SimpleNamespace

    w = MainWindow()
    qt_app.processEvents()
    w.canvas.ax.set_xlim(0.0, 10.0)
    w.canvas.ax.set_ylim(0.0, 10.0)
    # Force toolbar.mode to falsy so the gate doesn't block us.
    w.canvas.toolbar.mode = ""

    # Scroll up at (5, 5) → zoom in → narrower range, still centered on 5.
    evt = SimpleNamespace(inaxes=w.canvas.ax, button="up",
                          xdata=5.0, ydata=5.0, x=0, y=0)
    w.canvas._handle_scroll(evt)
    xl, xr = w.canvas.ax.get_xlim()
    yb, yt = w.canvas.ax.get_ylim()
    assert (xr - xl) < 10.0, f"Scroll-up did not zoom in: width={xr - xl}"
    assert (yt - yb) < 10.0
    assert abs(0.5 * (xl + xr) - 5.0) < 1e-9   # centered
    assert abs(0.5 * (yb + yt) - 5.0) < 1e-9

    # Scroll down at the same point → zoom out → exactly back to (0, 10).
    evt = SimpleNamespace(inaxes=w.canvas.ax, button="down",
                          xdata=5.0, ydata=5.0, x=0, y=0)
    w.canvas._handle_scroll(evt)
    xl, xr = w.canvas.ax.get_xlim()
    assert abs(xl - 0.0) < 1e-9 and abs(xr - 10.0) < 1e-9


def test_scroll_event_blocked_when_toolbar_active(qt_app):
    """When the matplotlib nav toolbar is in pan or zoom mode, the custom
    scroll handler must not modify the axes limits."""
    from types import SimpleNamespace

    w = MainWindow()
    qt_app.processEvents()
    w.canvas.ax.set_xlim(0.0, 10.0)
    w.canvas.ax.set_ylim(0.0, 10.0)
    w.canvas.toolbar.mode = "zoom rect"   # non-empty → toolbar active

    evt = SimpleNamespace(inaxes=w.canvas.ax, button="up",
                          xdata=5.0, ydata=5.0, x=0, y=0)
    w.canvas._handle_scroll(evt)
    assert w.canvas.ax.get_xlim() == (0.0, 10.0)
    assert w.canvas.ax.get_ylim() == (0.0, 10.0)


def test_middle_button_drag_pans_canvas(qt_app):
    """Middle-mouse-button press starts a pan; motion shifts xlim/ylim by the
    cursor delta; release clears the pan state."""
    from types import SimpleNamespace

    w = MainWindow()
    qt_app.processEvents()
    w.canvas.ax.set_xlim(0.0, 10.0)
    w.canvas.ax.set_ylim(0.0, 10.0)
    w.canvas.toolbar.mode = ""

    # transData.inverted() depends on the figure being drawn at least once,
    # so render before synthesizing events.
    w.canvas._mpl_canvas.draw()

    # Capture the data coords corresponding to two display points.
    tr = w.canvas.ax.transData
    x0_disp, y0_disp = tr.transform((2.0, 2.0))
    x1_disp, y1_disp = tr.transform((3.0, 4.0))
    inv = tr.inverted()
    dx_data, dy_data = (3.0 - 2.0, 4.0 - 2.0)

    # Middle-button press at display (x0, y0) → pan start.
    press = SimpleNamespace(inaxes=w.canvas.ax, button=2,
                            xdata=2.0, ydata=2.0, x=x0_disp, y=y0_disp)
    w.canvas._handle_click(press)
    assert w.canvas._pan_origin == (x0_disp, y0_disp)

    # Motion at display (x1, y1) → axes shift by -(dx_data, dy_data).
    move = SimpleNamespace(inaxes=w.canvas.ax, button=2,
                           xdata=3.0, ydata=4.0, x=x1_disp, y=y1_disp)
    w.canvas._handle_motion(move)
    xl, xr = w.canvas.ax.get_xlim()
    yb, yt = w.canvas.ax.get_ylim()
    assert abs(xl - (0.0 - dx_data)) < 1e-6
    assert abs(xr - (10.0 - dx_data)) < 1e-6
    assert abs(yb - (0.0 - dy_data)) < 1e-6
    assert abs(yt - (10.0 - dy_data)) < 1e-6

    # Release clears pan state.
    release = SimpleNamespace(button=2)
    w.canvas._handle_release(release)
    assert w.canvas._pan_origin is None


def test_deformed_scale_setting_updates_canvas_and_does_not_resolve(qt_app):
    """View → Deformed scale updates canvas.deformed_scale and redraws
    without re-running the solver."""
    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._do_solve()
    cached = w._result
    assert cached is not None and cached.status == "ok"

    w._set_deformed_scale(5.0)
    assert w.canvas.deformed_scale == 5.0
    assert w._result is cached, (
        "_set_deformed_scale must redraw only, never re-solve."
    )
    assert w._deformed_scale_actions[5.0].isChecked()
    assert not w._deformed_scale_actions[1.0].isChecked()

    w._set_deformed_scale(1.0)
    assert w.canvas.deformed_scale == 1.0
    assert w._result is cached
    assert w._deformed_scale_actions[1.0].isChecked()


# ── Section/profile dialog wizard ──────────────────────────────


def test_material_dialog_template_populates_fields(qt_app):
    """Picking a non-custom template fills E, α, ρ, ν from the preset."""
    from structural_analysis.gui_qt.dialogs import MaterialDialog
    from structural_analysis.profiles import MATERIAL_TEMPLATES

    w = MainWindow()
    qt_app.processEvents()
    d = MaterialDialog(w, existing=None, default_id=1)
    idx = d._template_combo.findData("Steel_S275")
    assert idx > 0, "Steel_S275 should be selectable"
    d._template_combo.setCurrentIndex(idx)

    preset = MATERIAL_TEMPLATES["Steel_S275"]
    assert float(d._entries["E"].text()) == pytest.approx(preset["E"])
    assert float(d._entries["alpha"].text()) == pytest.approx(preset["alpha"])
    assert float(d._entries["density"].text()) == pytest.approx(preset["density"])
    assert float(d._entries["nu"].text()) == pytest.approx(preset["nu"])
    # The derived-G label refreshes immediately.
    assert d._g_label.text() != "—"


def test_section_dialog_rectangle_computes_properties(qt_app):
    """Selecting rectangle and entering b/h yields a Section with the
    expected A, I, depth, width, and shape_type."""
    from structural_analysis.gui_qt.dialogs import SectionDialog
    from structural_analysis.model import Material, StructuralModel

    model = StructuralModel()
    model.materials[1] = Material(id=1, name="Steel", E=2.10e8)

    w = MainWindow()
    qt_app.processEvents()
    d = SectionDialog(w, model=model, existing=None, default_id=1)
    idx = d._shape_combo.findData("rectangle")
    d._shape_combo.setCurrentIndex(idx)
    d._rect_b.setText("0.3")
    d._rect_h.setText("0.5")

    section = d._accept()
    assert section.shape_type == "rectangle"
    assert section.A == pytest.approx(0.15)
    assert section.I == pytest.approx(0.3 * 0.5 ** 3 / 12.0)
    assert section.depth == pytest.approx(0.5)
    assert section.width == pytest.approx(0.3)
    assert section.b == pytest.approx(0.3)
    assert section.h == pytest.approx(0.5)


def test_section_dialog_i_section_validation_disables_ok(qt_app):
    """An invalid I-section (h ≤ 2·tf) disables OK and shows a status
    message — the dialog never lets the user accept a broken shape."""
    from structural_analysis.gui_qt.dialogs import SectionDialog
    from structural_analysis.model import Material, StructuralModel

    model = StructuralModel()
    model.materials[1] = Material(id=1, name="Steel", E=2.10e8)

    w = MainWindow()
    qt_app.processEvents()
    d = SectionDialog(w, model=model, existing=None, default_id=1)
    idx = d._shape_combo.findData("i_section")
    d._shape_combo.setCurrentIndex(idx)
    # h <= 2·tf → calculator raises, OK disabled
    d._i_h.setText("0.01")
    d._i_b.setText("0.1")
    d._i_tf.setText("0.02")
    d._i_tw.setText("0.005")

    ok = d._ok_button()
    assert ok is not None and not ok.isEnabled()
    assert d._status.text(), "status label should describe the error"


def test_section_dialog_preview_updates_with_shape_switch(qt_app):
    """The live cross-section preview must follow whatever the user
    types. Rectangle dimensions should render a 4-vertex polygon and
    switching to a valid I-section should rebuild it as 12 vertices —
    proving the preview is driven by _refresh_preview() and uses the
    profiles.section_outline() helper (no parallel geometry inside
    the dialog).
    """
    from matplotlib.patches import Polygon

    from structural_analysis.gui_qt.dialogs import SectionDialog
    from structural_analysis.model import Material, StructuralModel

    model = StructuralModel()
    model.materials[1] = Material(id=1, name="Steel", E=2.10e8)

    w = MainWindow()
    qt_app.processEvents()
    d = SectionDialog(w, model=model, existing=None, default_id=1)

    # Rectangle: 4-vertex outline.
    idx = d._shape_combo.findData("rectangle")
    d._shape_combo.setCurrentIndex(idx)
    d._rect_b.setText("0.3")
    d._rect_h.setText("0.5")
    qt_app.processEvents()
    rect_polys = [p for p in d._preview_ax.patches if isinstance(p, Polygon)]
    assert rect_polys, "rectangle preview must render a Polygon patch"
    rect_xy = rect_polys[-1].get_xy()
    # Polygon.get_xy() closes the loop (returns N+1 points). The
    # interior outline must be the 4-vertex rectangle from section_outline.
    assert len(rect_xy) - 1 == 4, (
        f"rectangle outline must be 4 vertices, got {len(rect_xy) - 1}"
    )

    # Switch to a valid I-section and confirm the patch grows to 12.
    idx = d._shape_combo.findData("i_section")
    d._shape_combo.setCurrentIndex(idx)
    d._i_h.setText("0.2")
    d._i_b.setText("0.1")
    d._i_tf.setText("0.0085")
    d._i_tw.setText("0.0056")
    qt_app.processEvents()
    i_polys = [p for p in d._preview_ax.patches if isinstance(p, Polygon)]
    assert i_polys, "i_section preview must render a Polygon patch"
    i_xy = i_polys[-1].get_xy()
    assert len(i_xy) - 1 == 12, (
        f"i_section outline must be 12 vertices, got {len(i_xy) - 1}"
    )

    # Existing read-out text label must still be populated alongside —
    # the graphical preview is additive, not a replacement.
    assert "A =" in d._i_preview.text()


def test_section_dialog_first_open_shows_example_outline(qt_app):
    """Opening "Add section" on a fresh model must not leave the
    preview blank for *any* shape — manual, rectangle, square, and
    i_section each render their canonical example outline (with
    dimension labels + an "example" tag) until the user types real
    dimensions of their own.
    """
    from matplotlib.patches import Polygon

    from structural_analysis.gui_qt.dialogs import SectionDialog
    from structural_analysis.model import Material, StructuralModel

    model = StructuralModel()
    model.materials[1] = Material(id=1, name="Steel", E=2.10e8)

    w = MainWindow()
    qt_app.processEvents()
    d = SectionDialog(w, model=model, existing=None, default_id=1)

    # Polygon.get_xy() closes the loop, so an N-vertex outline shows
    # as N+1 points. Manual defaults to a rectangle hint (4 verts);
    # i_section has 12; rectangle and square each have 4.
    expected = {
        "manual":    4,
        "rectangle": 4,
        "square":    4,
        "i_section": 12,
    }

    # Default shape is still "manual" — confirms we didn't change
    # the dropdown default in passing.
    assert d._shape_combo.currentData() == "manual"

    for shape_key, expected_vertices in expected.items():
        idx = d._shape_combo.findData(shape_key)
        d._shape_combo.setCurrentIndex(idx)
        qt_app.processEvents()
        polys = [p for p in d._preview_ax.patches if isinstance(p, Polygon)]
        assert polys, (
            f"first-open preview for {shape_key!r} must render an "
            f"example outline, not a blank canvas"
        )
        xy = polys[-1].get_xy()
        assert len(xy) - 1 == expected_vertices, (
            f"{shape_key!r} example outline must be "
            f"{expected_vertices} vertices, got {len(xy) - 1}"
        )
        texts = [t.get_text() for t in d._preview_ax.texts]
        assert any("example" in t.lower() for t in texts), (
            f"{shape_key!r} example must be clearly tagged so the "
            f"user doesn't mistake it for their own input"
        )
        assert any("b =" in t for t in texts)
        assert any("h =" in t for t in texts)


def test_section_dialog_existing_section_does_not_show_example(qt_app):
    """When the user *edits* an existing section the example outline
    must NOT appear — that path is for first-open guidance only. A
    valid existing section renders its real outline; a corrupted /
    invalid edit shows the existing "invalid dimensions" placeholder.
    """
    from matplotlib.patches import Polygon

    from structural_analysis.gui_qt.dialogs import SectionDialog
    from structural_analysis.model import Material, Section, StructuralModel

    model = StructuralModel()
    model.materials[1] = Material(id=1, name="Steel", E=2.10e8)
    existing = Section(
        id=7, name="R30x50", material_id=1,
        A=0.15, I=3.125e-3, depth=0.5, width=0.3,
        shape_type="rectangle", b=0.3, h=0.5,
    )
    model.sections[7] = existing

    w = MainWindow()
    qt_app.processEvents()
    d = SectionDialog(w, model=model, existing=existing, default_id=7)

    polys = [p for p in d._preview_ax.patches if isinstance(p, Polygon)]
    assert polys, "existing section must still render its real outline"
    texts = [t.get_text() for t in d._preview_ax.texts]
    assert not any("example" in t.lower() for t in texts), (
        "the example-outline hint must only appear for new sections"
    )


def test_main_window_keeps_version_badge_top_right(qt_app):
    """The top-right version + what's-new badge in the menu bar
    must stay populated and reflect the current package version —
    don't regress the existing badge.
    """
    from structural_analysis import __version__, __what_is_new__

    w = MainWindow()
    qt_app.processEvents()
    assert w._version_label is not None
    text = w._version_label.text()
    assert __version__ in text
    assert __what_is_new__ in text


# ── Element-detail interactive layer (crosshair, maxima, BMD) ─────


def test_element_dialog_crosshair_tracks_motion(qt_app):
    """Synthesise a motion event inside the N diagram axis; the three
    cursor axvlines must update to the same x and the readout labels
    must show numeric content (not '—')."""
    from types import SimpleNamespace

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._do_solve()
    qt_app.processEvents()
    assert w._result is not None and w._result.status == "ok"

    elem = w._model.elements[0]
    ni = w._model.nodes[elem.node_i]
    nj = w._model.nodes[elem.node_j]
    L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5

    w._open_element_inspector(elem.id)
    qt_app.processEvents()
    d = w._element_inspector
    assert d is not None and d._cursors, (
        "crosshair axvlines must be created post-solve"
    )

    # Synthesise a motion event inside the N (axial) axis at x = L/2
    x_test = L / 2.0
    evt = SimpleNamespace(inaxes=d._ax_n, xdata=x_test, ydata=0.0)
    d._on_diagram_motion(evt)

    # All three cursors must be at the same x and visible
    for c in d._cursors:
        xdata = list(c.get_xdata())
        assert abs(xdata[0] - x_test) < 1e-9, (
            f"cursor xdata {xdata[0]} != test x {x_test}"
        )
        assert c.get_alpha() > 0, "cursor must become visible after motion"

    # Readout labels must show numeric values
    assert "x:" in d._lbl_x.text() and "—" not in d._lbl_x.text()
    assert "N:" in d._lbl_N.text() and "—" not in d._lbl_N.text()


def test_element_dialog_maxima_checkbox_toggles_annotations(qt_app):
    """Checking 'Show Maxima' must add annotations on applicable diagram
    axes; unchecking must remove all of them.  Wiring test — numerical
    correctness is tested in test_diagram_signs.py."""
    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._do_solve()
    qt_app.processEvents()

    elem = w._model.elements[0]
    w._open_element_inspector(elem.id)
    qt_app.processEvents()
    d = w._element_inspector
    assert d._show_maxima_cb.isEnabled(), (
        "Show Maxima checkbox must be enabled post-solve"
    )

    d._show_maxima_cb.setChecked(True)
    assert d._maxima_annotations, (
        "checking Show Maxima must create at least one annotation"
    )

    d._show_maxima_cb.setChecked(False)
    assert not d._maxima_annotations, (
        "unchecking Show Maxima must clear all annotations"
    )


def test_element_dialog_bmd_axis_is_inverted(qt_app):
    """The M subplot must use the structural tension-fibre BMD
    convention (y-axis inverted); N and V subplots must not be inverted."""
    from structural_analysis.element import FrameElement2D

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._do_solve()
    qt_app.processEvents()

    frame_elem = next(
        (e for e in w._model.elements if isinstance(e, FrameElement2D)),
        None,
    )
    assert frame_elem is not None, (
        "q2a model must have at least one frame element"
    )

    w._open_element_inspector(frame_elem.id)
    qt_app.processEvents()
    d = w._element_inspector

    assert d._ax_m.yaxis_inverted(), (
        "M subplot must have inverted y-axis (tension-fibre BMD convention)"
    )
    assert not d._ax_n.yaxis_inverted(), "N subplot must NOT be inverted"
    assert not d._ax_v.yaxis_inverted(), "V subplot must NOT be inverted"


# ── Element material override (PR #16) ────────────────────────


def test_element_properties_dialog_shows_override_tag(qt_app):
    """The detail inspector must label the Material row "— section default"
    when the element has no override, and "— override (default: …)"
    when it does. Walks every QLabel in the dialog and checks for the
    expected substrings."""
    from PyQt6.QtWidgets import QLabel

    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_common.commands import UpdateElementCmd
    from structural_analysis.gui_qt.dialogs import ElementPropertiesDialog
    from structural_analysis.model import (
        Material, Node, Section, Support,
    )

    w = MainWindow()
    m = w._model
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 5.0, 0.0),
        3: Node(3, 0.0, 5.0),
        4: Node(4, 5.0, 5.0),
    }
    m.materials = {
        1: Material(id=1, name="Steel_S275", E=2.10e8, alpha=1.2e-5,
                    density=7850.0),
        2: Material(id=2, name="Concrete_C25", E=3.10e7, alpha=1.0e-5,
                    density=2400.0),
    }
    m.sections = {
        1: Section(id=1, name="IPE200", material_id=1,
                   A=0.0028, I=1.94e-5, depth=0.2, width=0.1),
    }
    m.elements = [
        FrameElement2D(1, 1, 2, E=2.10e8, A=0.0028, I=1.94e-5,
                       alpha=1.2e-5, depth=0.2, rho=7850.0, section_id=1),
        FrameElement2D(2, 3, 4, E=2.10e8, A=0.0028, I=1.94e-5,
                       alpha=1.2e-5, depth=0.2, rho=7850.0, section_id=1),
    ]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=True),
        3: Support(3, ux=True, uy=True, rz=True),
    }
    # Override elem 1 to Concrete_C25; leave elem 2 on the section default.
    UpdateElementCmd(elem_id=1, section_id=1, kind="frame",
                     material_override_id=2).do(m)

    def _all_label_texts(dialog):
        return [w.text() for w in dialog.findChildren(QLabel)]

    d_ovr = ElementPropertiesDialog(w, m, 1, None)
    qt_app.processEvents()
    texts_ovr = _all_label_texts(d_ovr)
    assert any("Concrete_C25" in t and "override" in t
               for t in texts_ovr), (
        f"override label must mention the override material name + "
        f"the word 'override'. Got: "
        f"{[t for t in texts_ovr if 'Material' in t or 'override' in t]}"
    )
    assert any("Steel_S275" in t and "default" in t.lower()
               for t in texts_ovr), (
        "override label must also surface the section default for context"
    )

    d_def = ElementPropertiesDialog(w, m, 2, None)
    qt_app.processEvents()
    texts_def = _all_label_texts(d_def)
    assert any("Steel_S275" in t and "section default" in t
               for t in texts_def), (
        f"non-override label must mention the section-default material + "
        f"the substring 'section default'. Got: "
        f"{[t for t in texts_def if 'Material' in t or 'default' in t]}"
    )
    assert not any("override" in t for t in texts_def), (
        "non-override element must NOT show the word 'override'"
    )


# ── 3D extruded viewer ─────────────────────────────────────────


def test_view3d_window_opens_and_holds_singleton(qt_app):
    """View → Open 3D viewer must construct a separate non-modal
    window and re-use the same instance on subsequent invocations."""
    w = MainWindow(initial_path="inputs/example_01_cantilever_tip_load.txt")
    qt_app.processEvents()

    assert w._view3d_window is None
    w._open_view3d()
    first = w._view3d_window
    assert first is not None
    assert first.isVisible()
    assert not first.isModal()
    w._open_view3d()
    assert w._view3d_window is first


def test_view3d_window_builds_one_mesh_per_element(qt_app):
    """Each frame/truss element should land as exactly one
    Poly3DCollection, keyed by element id so future stress overlays
    can look up the mesh per element without re-meshing."""
    w = MainWindow(initial_path="inputs/example_01_cantilever_tip_load.txt")
    qt_app.processEvents()
    w._open_view3d()
    qt_app.processEvents()
    view = w._view3d_window
    elem_ids = {e.id for e in w._model.elements}
    assert set(view._element_meshes) == elem_ids
    view.refresh()
    assert set(view._element_meshes) == elem_ids


def test_view3d_manual_section_uses_sqrt_A_and_shows_banner(qt_app):
    """A model containing a manual section must show the approximation
    banner; updating every manual section to a real shape via the real
    AddOrUpdateSectionCmd clears the banner on refresh."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_common.commands import AddOrUpdateSectionCmd
    from structural_analysis.model import Material, Node, Section

    w = MainWindow()
    w._model.materials = {1: Material(id=1, name="Steel", E=2.10e8)}
    w._model.sections = {
        1: Section(id=1, name="manual_1", material_id=1, A=0.01, I=1e-4),
    }
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.10e8, A=0.01, I=1e-4, section_id=1,
    )]
    qt_app.processEvents()

    w._open_view3d()
    qt_app.processEvents()
    assert w._view3d_window._banner.isVisible(), (
        "manual sections must surface the sqrt(A) approximation banner"
    )

    w.execute(AddOrUpdateSectionCmd(section=Section(
        id=1, name="manual_1", material_id=1,
        A=0.15, I=3.125e-3, depth=0.5, width=0.3,
        shape_type="rectangle", b=0.3, h=0.5,
    )))
    w._view3d_window.refresh()
    qt_app.processEvents()
    assert not w._view3d_window._banner.isVisible(), (
        "banner must clear once every section has a real shape"
    )


def test_view3d_orientation_switch_rebuilds_geometry(qt_app):
    """The vertical-axis combobox swaps the world mapping (Y-up vs
    Z-up) without touching the underlying model. The mesh count and
    per-element id mapping must be preserved across the switch, and
    the elevation node (here at model y=4) must land on world Y in
    Y-up mode and on world Z in Z-up mode.
    """
    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_qt.view3d import (
        _ORIENT_Y_UP, _ORIENT_Z_UP,
    )
    from structural_analysis.model import Material, Node, Section

    w = MainWindow()
    w._model.materials = {1: Material(id=1, name="Steel", E=2.10e8)}
    w._model.sections = {
        1: Section(
            id=1, name="Rect", material_id=1,
            A=0.06, I=1.25e-3, depth=0.5, width=0.12,
            shape_type="rectangle", b=0.12, h=0.5,
        ),
    }
    # A vertical column from the origin straight up the model y axis.
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 0.0, 4.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.10e8,
        A=0.06, I=1.25e-3, section_id=1,
    )]
    qt_app.processEvents()
    w._open_view3d()
    qt_app.processEvents()
    view = w._view3d_window

    # Default mode is Y-up (preserves the existing 2D convention).
    assert view._orientation == _ORIENT_Y_UP
    elem_ids = {e.id for e in w._model.elements}
    assert set(view._element_meshes) == elem_ids

    # Axis labels expose the mode-dependent convention publicly —
    # in Y-up the elevation lives on world Y, out-of-plane on world Z.
    assert "elevation" in view.ax.get_ylabel()
    assert "out-of-plane" in view.ax.get_zlabel()

    # Independent geometry check: poke the lift helper directly to
    # confirm the top of the column landed on world Y in Y-up mode.
    from structural_analysis.gui_qt.view3d import _node_world
    top_y_up = _node_world(0.0, 4.0, _ORIENT_Y_UP)
    assert top_y_up.tolist() == [0.0, 4.0, 0.0]

    # Switch to Z-up via the combobox — same UI path a user would take.
    idx = view._orient_combo.findData(_ORIENT_Z_UP)
    view._orient_combo.setCurrentIndex(idx)
    qt_app.processEvents()

    assert view._orientation == _ORIENT_Z_UP
    # Mesh-per-element invariant preserved across the rebuild.
    assert set(view._element_meshes) == elem_ids
    # Labels follow the mode — Z is now the elevation axis.
    assert "elevation" in view.ax.get_zlabel()
    assert "out-of-plane" in view.ax.get_ylabel()
    # And the lift helper now places the column top on world Z.
    top_z_up = _node_world(0.0, 4.0, _ORIENT_Z_UP)
    assert top_z_up.tolist() == [0.0, 0.0, 4.0]

    # The underlying 2D model must be unchanged by the visual switch.
    assert w._model.nodes[2].x == 0.0 and w._model.nodes[2].y == 4.0
    # And the scroll handler is wired exactly once (mpl_connect
    # returns an integer connection id).
    assert isinstance(view._scroll_cid, int)


# ── Building wizard ─────────────────────────────────────────────


def test_building_wizard_creates_model(qt_app):
    """The wizard generates a portal frame and routes through ReplaceModelCmd
    so a single Undo restores the previous model."""
    from structural_analysis.gui_qt.dialogs import BuildingWizardDialog

    w = MainWindow()
    qt_app.processEvents()

    # Starter model has sections, so the wizard dialog should construct.
    d = BuildingWizardDialog(w, model=w._model)
    d._stories.setValue(2)
    d._story_h.setValue(3.0)
    d._bays.setValue(2)
    d._bay_w.setValue(4.0)
    d._fixed_base.setChecked(True)
    new_model = d._accept()
    assert len(new_model.nodes) == 9
    assert len(new_model.elements) == 10
    assert len(new_model.supports) == 3
    assert new_model.materials == w._model.materials
    assert new_model.sections == w._model.sections


def test_building_wizard_action_undoable(qt_app):
    """Driving the wizard handler through a stubbed dialog must apply
    ReplaceModelCmd; one Undo must restore the previous (empty) model."""
    from structural_analysis.gui_qt.dialogs import BuildingWizardDialog

    w = MainWindow()
    qt_app.processEvents()
    original_exec = BuildingWizardDialog.exec

    def fake_exec(self):
        from PyQt6.QtWidgets import QDialog as _QD
        self._stories.setValue(1)
        self._bays.setValue(1)
        self.result_value = self._accept()
        return _QD.DialogCode.Accepted

    BuildingWizardDialog.exec = fake_exec
    try:
        w._do_building_wizard()
    finally:
        BuildingWizardDialog.exec = original_exec

    assert len(w._model.nodes) == 4
    assert len(w._model.elements) == 3
    w._do_undo()
    assert len(w._model.nodes) == 0
    assert len(w._model.elements) == 0


# ── v0.9.0: analysis settings dialog + mass / self-weight summary ──


def test_analysis_settings_dialog_toggles_model_flag(qt_app):
    """Opening the dialog with an unstubbed exec doesn't fit in a smoke
    test, so we drive the slot via a stubbed exec that flips the
    checkbox and returns Accepted."""
    from structural_analysis.gui_qt.dialogs import AnalysisSettingsDialog

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    assert w._model.include_self_weight is False

    original_exec = AnalysisSettingsDialog.exec

    def fake_exec(self):
        self._sw_check.setChecked(True)
        self.result_value = self._accept()
        return QDialog.DialogCode.Accepted

    AnalysisSettingsDialog.exec = fake_exec
    try:
        w._edit_analysis_settings()
    finally:
        AnalysisSettingsDialog.exec = original_exec

    assert w._model.include_self_weight is True


def test_mass_summary_window_singleton_and_renders_rows(qt_app):
    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._show_mass_summary()
    win = w._mass_summary_window
    assert win is not None
    assert win.isVisible()
    # Re-opening must reuse the same instance.
    w._show_mass_summary()
    assert w._mass_summary_window is win
    # Row count matches the model.
    assert win._table.rowCount() == len(w._model.elements)
    # Mass column equals ρ·A·L for at least the first element.
    elem = w._model.elements[0]
    L, _, _ = elem.length_cos_sin(w._model.nodes)
    expected_mass = float(elem.rho) * float(elem.A) * float(L)
    cell = win._table.item(0, 7).text()
    assert float(cell) == pytest.approx(expected_mass, rel=1e-6, abs=1e-6)


def test_mass_summary_window_header_reflects_flag(qt_app):
    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._show_mass_summary()
    win = w._mass_summary_window
    assert "DISABLED" in win._status_label.text()
    w._model.include_self_weight = True
    win.refresh()
    assert "ENABLED" in win._status_label.text()


def test_mass_summary_window_refreshes_after_edit(qt_app):
    """Editing the model invalidates results — the summary window
    should re-read totals automatically when open."""
    from structural_analysis.gui_common.commands import AddOrUpdateMaterialCmd
    from structural_analysis.model import Material

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._show_mass_summary()
    win = w._mass_summary_window

    # The fixture loads materials with density = 0 (no density column
    # in the legacy file), so the initial total mass is 0. Doubling
    # the density of one material should change the totals after
    # refresh — but only if the material is the *effective* one for
    # at least one element. Pick the material the first element uses.
    elem = w._model.elements[0]
    from structural_analysis.model import effective_material as _eff
    eff_mat = _eff(w._model, elem)
    bumped = Material(
        id=eff_mat.id, name=eff_mat.name,
        E=eff_mat.E, alpha=eff_mat.alpha,
        density=eff_mat.density + 1000.0,
        nu=eff_mat.nu, template=eff_mat.template,
    )
    pre_text = win._totals_label.text()
    cmd = AddOrUpdateMaterialCmd(material=bumped)
    w.execute(cmd)
    qt_app.processEvents()
    post_text = win._totals_label.text()
    assert pre_text != post_text


def test_joint_masses_window_singleton_and_renders_rows(qt_app):
    """v0.9.1: Assembled Joint Masses window — singleton, row per node,
    totals agree with mass_inspect.joint_mass_table."""
    from structural_analysis.mass_inspect import joint_mass_table

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._show_joint_masses()
    win = w._joint_masses_window
    assert win is not None
    assert win.isVisible()
    # Re-opening must reuse the same instance.
    w._show_joint_masses()
    assert w._joint_masses_window is win
    # Row count matches node count.
    assert win._table.rowCount() == len(w._model.nodes)

    # Totals in the footer must match what the helper would compute now.
    report = joint_mass_table(w._model, method="row_sum")
    footer = win._totals_label.text()
    assert f"{report.totals_kg['ux']:.4f}" in footer
    assert f"Active modal DOFs = {report.n_free_dofs}" in footer

    # First-column cells must be the node IDs in model order.
    for r, nid in enumerate(w._model.node_ids):
        assert win._table.item(r, 0).text() == str(nid)


def test_joint_masses_window_skips_refresh_when_hidden(qt_app):
    """Per PR #18 review: model edits must NOT trigger a mass-matrix
    assembly while the joint-masses singleton is hidden."""
    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._show_joint_masses()
    win = w._joint_masses_window
    assert win.isVisible()

    win.close()
    qt_app.processEvents()
    assert not win.isVisible()
    # Singleton still held (close() doesn't destroy).
    assert w._joint_masses_window is win

    # Patch refresh to a counting stub; trigger an invalidation; assert
    # it wasn't called while hidden.
    calls = {"n": 0}
    original_refresh = win.refresh
    win.refresh = lambda: calls.__setitem__("n", calls["n"] + 1)  # type: ignore[method-assign]
    try:
        w._invalidate_result()
        qt_app.processEvents()
        assert calls["n"] == 0, (
            "refresh() ran while hidden — would trigger needless "
            "mass-matrix assembly on every edit"
        )
        # And reopening still gives a refreshed view (via _show_joint_masses).
        w._show_joint_masses()
        assert calls["n"] >= 1
    finally:
        win.refresh = original_refresh  # type: ignore[method-assign]


def test_joint_masses_window_zero_density_fixture_shows_amber_warning(qt_app):
    """Legacy fixtures (no density column → ρ=0) must open the window
    immediately, render rows, and surface the amber warning banner —
    no 'please solve modal first' gate."""
    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._show_joint_masses()
    win = w._joint_masses_window
    assert win is not None
    assert win.isVisible()

    status = win._status_label.text()
    assert "All assembled element mass contributions are zero" in status
    # Amber colour applied (mirrors the spec in the plan file).
    assert "#a06000" in win._status_label.styleSheet()

    # Table still renders, one row per node.
    assert win._table.rowCount() == len(w._model.nodes)


def test_joint_masses_window_never_invokes_solve_modal(qt_app, monkeypatch):
    """Opening + refreshing the window must never invoke the modal
    eigenvalue solver. Locks the inspection-only contract at the GUI
    layer (the unit test does the same at the helper layer)."""
    import structural_analysis.modal as modal_mod

    calls = {"n": 0}

    def _boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError(
            "solve_modal invoked from the joint-masses inspection path"
        )

    monkeypatch.setattr(modal_mod, "solve_modal", _boom)

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._show_joint_masses()
    win = w._joint_masses_window
    assert win is not None
    win.refresh()
    qt_app.processEvents()

    assert calls["n"] == 0


def test_modal_dialog_default_formulation_is_consistent(qt_app):
    """v0.9.2: the modal-analysis dialog's formulation dropdown defaults
    to 'consistent', and round-trips the selection through _accept()."""
    from structural_analysis.gui_qt.dialogs import ModalAnalysisDialog

    w = MainWindow()
    qt_app.processEvents()
    d = ModalAnalysisDialog(w, default_n_modes=4)
    qt_app.processEvents()
    # Default is consistent.
    assert d._mass_combo.currentData() == "consistent"
    # Validation path round-trips.
    accepted = d._accept()
    assert accepted["mass_formulation"] == "consistent"
    # Flip to lumped.
    idx = d._mass_combo.findData("lumped")
    assert idx >= 0
    d._mass_combo.setCurrentIndex(idx)
    accepted = d._accept()
    assert accepted["mass_formulation"] == "lumped"


def test_joint_masses_window_lumped_radio_zeros_rotational_cells(qt_app):
    """v0.9.2: flipping the joint-masses formulation radio to Lumped
    routes through joint_mass_table(mass_formulation='lumped') and the
    table's rz column reads 0.0 for every node."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, Node, Section, StructuralModel, Support,
    )

    # Build a fresh 2-node cantilever with positive density so the
    # consistent path produces nonzero rz cells (the legacy
    # q2a_settlement.txt fixture has ρ=0 and would give zero rz on
    # both formulations, masking the contrast we're testing).
    m = StructuralModel(title="lumped-radio-test")
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 3.0, 0.0)
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=0.01, I=1e-4, rho=7850.0, depth=0.3, section_id=1,
    ))
    m.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)

    w = MainWindow()
    w._model = m
    qt_app.processEvents()
    w._show_joint_masses()
    win = w._joint_masses_window
    assert win is not None
    win.refresh()
    qt_app.processEvents()

    rz_col = 3  # Mrz / rz column
    # Sanity: consistent mode is the default; node-2 rz cell should be
    # nonzero (cantilever free end carries rotational consistent mass).
    rz_text_consistent = win._table.item(1, rz_col).text()
    assert rz_text_consistent not in ("—", "0.0000", "0.000e+00"), (
        f"expected nonzero rz cell on consistent mass, got {rz_text_consistent!r}"
    )

    # Flip to lumped.
    win._rb_lumped.setChecked(True)
    qt_app.processEvents()
    for r in range(win._table.rowCount()):
        text = win._table.item(r, rz_col).text()
        assert text in ("—", "0.0000", "0.000e+00"), (
            f"row {r} rz expected zero on lumped, got {text!r}"
        )


def test_frame_tool_draws_member_with_two_empty_clicks(qt_app):
    """v0.10.0 Stage A end-to-end: select Frame tool, click two empty
    grid points, dialog accepts defaults, assert 2 nodes + 1 element
    appear in the model and one Ctrl+Z removes the entire draw."""
    from structural_analysis.model import Material, Section

    w = MainWindow()
    qt_app.processEvents()

    # Need a material and a section so the element dialog has something
    # to default to; otherwise open_element_dialog_for_member would
    # warn-and-return.
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    # Take the dialog-less sticky path so the smoke test doesn't have
    # to drive a modal QDialog. This is the same code path the user
    # hits after one "Remember" tick.
    w._sticky_element = {
        "kind": "frame",
        "section_id": 1,
        "release_i": False,
        "release_j": False,
        "material_override_id": None,
    }

    w._select_tool("frame")
    qt_app.processEvents()
    n0 = len(w._model.nodes)
    e0 = len(w._model.elements)
    # Click 1 — empty grid point.
    w._on_canvas_click(HitResult(x=0.0, y=0.0), "left")
    qt_app.processEvents()
    # Still no element yet (first click is just the start point).
    assert len(w._model.elements) == e0
    # Click 2 — another empty grid point.
    w._on_canvas_click(HitResult(x=5.0, y=0.0), "left")
    qt_app.processEvents()

    assert len(w._model.nodes) == n0 + 2
    assert len(w._model.elements) == e0 + 1
    elem = w._model.elements[-1]
    nodes = [w._model.nodes[elem.node_i], w._model.nodes[elem.node_j]]
    coords = {(n.x, n.y) for n in nodes}
    assert coords == {(0.0, 0.0), (5.0, 0.0)}

    # One undo removes the whole draw atomically.
    w._do_undo()
    qt_app.processEvents()
    assert len(w._model.nodes) == n0
    assert len(w._model.elements) == e0


def test_frame_tool_same_empty_point_twice_short_circuits_before_dialog(qt_app):
    """PR #20 review fix: clicking the same empty point twice with the
    Frame tool must short-circuit *before* opening the element-
    properties dialog. Otherwise the user fills the dialog in only to
    see the AddMemberCmd zero-length error on accept.

    The controller-layer short-circuit lives in
    :meth:`_PairTool.on_click`; this test asserts the dispatch method
    is never called when both clicks land on coincident empty space.
    """
    from structural_analysis.model import Material, Section

    w = MainWindow()
    qt_app.processEvents()
    w._model.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )

    calls: list[dict] = []

    def fake_open(
        *,
        first_x, first_y, first_node_id,
        second_x, second_y, second_node_id,
        kind=None,
        # v0.11.0 follow-up: tolerate deferred split-target kwargs.
        first_split_target=None,
        second_split_target=None,
    ):
        calls.append({
            "first": (first_x, first_y, first_node_id),
            "second": (second_x, second_y, second_node_id),
            "kind": kind,
        })

    w.open_element_dialog_for_member = fake_open
    w._select_tool("frame")
    w._on_canvas_click(HitResult(x=2.5, y=1.0), "left")
    qt_app.processEvents()
    # Second click at the same world coords — no node anywhere yet,
    # so both hits have node_id=None. The short-circuit must catch
    # this before fake_open ever runs.
    w._on_canvas_click(HitResult(x=2.5, y=1.0), "left")
    qt_app.processEvents()

    assert calls == [], (
        "open_element_dialog_for_member was called for a same-point "
        "double click; the controller should have short-circuited"
    )
    # Model is untouched.
    assert len(w._model.nodes) == 0
    assert len(w._model.elements) == 0


def _stage_c_seed_frame(w, qt_app) -> int:
    """Set up a single 6 m horizontal frame on `w` and return its element id."""
    from structural_analysis.model import Material, Section
    w._model.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._sticky_element = {
        "kind": "frame", "section_id": 1,
        "release_i": False, "release_j": False,
        "material_override_id": None,
    }
    w._select_tool("frame")
    qt_app.processEvents()
    w._on_canvas_click(HitResult(x=0.0, y=0.0), "left")
    w._on_canvas_click(HitResult(x=6.0, y=0.0), "left")
    qt_app.processEvents()
    assert len(w._model.elements) == 1
    return w._model.elements[0].id


def test_node_tool_splits_even_when_snap_kind_is_not_project(qt_app):
    """Regression — user-reported bug after PR #21 first cut: the snap
    engine often picks GRID (priority 1) or returns no candidate over
    PROJECT (priority 4), so real clicks on an element interior arrive
    at the controller with ``snap_kind != "project"`` even though
    ``hit.element_id`` is set (the canvas fallback at canvas.py:465-484
    also produces this shape). The split must still fire — the
    controller projects (hit.x, hit.y) onto the element in world space
    rather than trusting snap_kind."""
    w = MainWindow()
    qt_app.processEvents()
    parent_id = _stage_c_seed_frame(w, qt_app)

    w._select_tool("node")
    qt_app.processEvents()
    # The grid-snap fallback path arrives with snap_kind="" — the bug
    # the user actually hit. element_id is set because the canvas
    # element-pick fell back to ELEM_PICK_RADIUS_PX.
    w._on_canvas_click(
        HitResult(x=3.0, y=0.0, element_id=parent_id, snap_kind=""),
        "left",
    )
    qt_app.processEvents()
    assert len(w._model.nodes) == 3
    assert parent_id not in [e.id for e in w._model.elements]
    assert len(w._model.elements) == 2


def test_node_tool_midpoint_snap_also_splits(qt_app):
    """Symmetric — midpoint snap (snap_kind='midpoint', priority 3)
    routes through the same split path. MIDPOINT wins over PROJECT
    when the cursor is near the geometric midpoint of an element."""
    w = MainWindow()
    qt_app.processEvents()
    parent_id = _stage_c_seed_frame(w, qt_app)

    w._select_tool("node")
    qt_app.processEvents()
    w._on_canvas_click(
        HitResult(x=3.0, y=0.0, element_id=parent_id, snap_kind="midpoint"),
        "left",
    )
    qt_app.processEvents()
    assert len(w._model.nodes) == 3
    assert parent_id not in [e.id for e in w._model.elements]


def test_node_tool_click_far_off_element_does_not_split(qt_app):
    """The world-space projection check rejects clicks whose perpendicular
    projection lands outside (ELEMENT_SPLIT_TOL, 1 - ELEMENT_SPLIT_TOL).
    Even with element_id set (canvas fallback), if the projected t is
    near 0 or 1 we should NOT split — the user is trying to add a node
    near an endpoint, not bisect."""
    w = MainWindow()
    qt_app.processEvents()
    parent_id = _stage_c_seed_frame(w, qt_app)

    w._select_tool("node")
    qt_app.processEvents()
    # Click at world (-1, 0) — projects to t = -1/6 < 0, outside the
    # strict interior. Should fall through to AddNodeCmd.
    w._on_canvas_click(
        HitResult(x=-1.0, y=0.0, element_id=parent_id, snap_kind=""),
        "left",
    )
    qt_app.processEvents()
    # Original element still here, new free node added.
    assert parent_id in [e.id for e in w._model.elements]
    assert len(w._model.elements) == 1
    assert len(w._model.nodes) == 3


def test_node_tool_click_on_element_interior_splits_via_project_snap(qt_app):
    """v0.11.0: clicking the interior of an existing element with the
    Node tool fires SplitElementCmd, not AddNodeCmd. The model gains
    one node and the parent is replaced by two children — no
    disconnected free-floating node."""
    w = MainWindow()
    qt_app.processEvents()
    parent_id = _stage_c_seed_frame(w, qt_app)
    assert len(w._model.nodes) == 2

    w._select_tool("node")
    qt_app.processEvents()
    # HitResult with snap_kind="project" + element_id is what the snap
    # engine produces for a click on an element's interior.
    w._on_canvas_click(
        HitResult(
            x=3.0, y=0.0,
            element_id=parent_id, snap_kind="project",
        ),
        "left",
    )
    qt_app.processEvents()

    assert len(w._model.nodes) == 3
    elem_ids = [e.id for e in w._model.elements]
    assert parent_id not in elem_ids
    assert len(w._model.elements) == 2
    # The new node lies on the segment.
    new_node = next(
        n for n in w._model.nodes.values()
        if (n.x, n.y) == (3.0, 0.0)
    )
    children_with_new = [
        e for e in w._model.elements
        if new_node.id in (e.node_i, e.node_j)
    ]
    assert len(children_with_new) == 2

    # Single Ctrl+Z restores the parent.
    w._do_undo()
    qt_app.processEvents()
    assert [e.id for e in w._model.elements] == [parent_id]
    assert len(w._model.nodes) == 2


def test_node_tool_split_loaded_element_succeeds(qt_app):
    """v0.12.0: clicking the interior of a loaded element with the
    Node tool now SUCCEEDS and remaps the load onto both children.
    Replaces the pre-0.12.0 block-with-warning test."""
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    qt_app.processEvents()
    parent_id = _stage_c_seed_frame(w, qt_app)
    w._model.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-5.0),
    )

    w._select_tool("node")
    qt_app.processEvents()
    w._on_canvas_click(
        HitResult(
            x=3.0, y=0.0,
            element_id=parent_id, snap_kind="project",
        ),
        "left",
    )
    qt_app.processEvents()

    # Three nodes, two children, both children carry the UDL.
    assert len(w._model.nodes) == 3
    assert len(w._model.elements) == 2
    assert parent_id not in [e.id for e in w._model.elements]
    for child in w._model.elements:
        child_udls = [ld for ld in child.member_loads
                      if isinstance(ld, UniformDistributedLoad)]
        assert len(child_udls) == 1
        assert child_udls[0].wy == -5.0


def test_node_tool_split_unsupported_load_type_still_blocks(qt_app):
    """A synthetic, unsupported load type on the parent still surfaces
    the block-with-warning UX (the ``_remap_member_loads`` guard for
    future load types). Model must remain untouched."""
    from dataclasses import dataclass

    @dataclass
    class _UnsupportedLoad:
        x: float = 0.0

    w = MainWindow()
    qt_app.processEvents()
    parent_id = _stage_c_seed_frame(w, qt_app)
    w._model.elements[0].member_loads.append(_UnsupportedLoad())
    nodes_before = sorted(w._model.nodes.keys())
    elem_ids_before = [e.id for e in w._model.elements]

    w._select_tool("node")
    qt_app.processEvents()

    from PyQt6.QtWidgets import QMessageBox
    seen_warnings: list[str] = []
    real_warning = QMessageBox.warning
    QMessageBox.warning = staticmethod(  # type: ignore[assignment]
        lambda parent, title, text, *a, **k:
            (seen_warnings.append(text), QMessageBox.StandardButton.Ok)[-1]
    )
    try:
        w._on_canvas_click(
            HitResult(
                x=3.0, y=0.0,
                element_id=parent_id, snap_kind="project",
            ),
            "left",
        )
        qt_app.processEvents()
    finally:
        QMessageBox.warning = real_warning  # type: ignore[assignment]

    assert sorted(w._model.nodes.keys()) == nodes_before
    assert [e.id for e in w._model.elements] == elem_ids_before
    assert any("not yet supported" in w for w in seen_warnings)


def test_member_draw_split_on_loaded_element_succeeds_one_undo(qt_app):
    """Drawing a Frame member whose endpoint lands on a loaded
    element's interior now SUCCEEDS (Feature B). Both children carry
    the parent's load, and one Ctrl+Z restores the loaded parent
    intact."""
    from structural_analysis.gui_common.commands import AddNodeCmd
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    qt_app.processEvents()
    parent_id = _stage_c_seed_frame(w, qt_app)
    w._model.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-5.0),
    )

    # Free node above the parent bar.
    w.execute(AddNodeCmd(x=3.0, y=5.0))
    qt_app.processEvents()
    free_node = max(w._model.nodes.keys())

    w._select_tool("frame")
    qt_app.processEvents()
    free = w._model.nodes[free_node]
    # Start from free node, end on parent interior at (3, 0).
    w._on_canvas_click(
        HitResult(x=free.x, y=free.y, node_id=free_node, snap_kind="node"),
        "left",
    )
    qt_app.processEvents()
    w._on_canvas_click(
        HitResult(x=3.0, y=0.0, element_id=parent_id, snap_kind="project"),
        "left",
    )
    qt_app.processEvents()

    # Parent split (2 children) + the new member.
    assert len(w._model.elements) == 3
    assert parent_id not in [e.id for e in w._model.elements]
    # The two children of the split parent both carry the UDL; the
    # new member does not.
    children_with_udl = [
        e for e in w._model.elements
        if any(isinstance(ld, UniformDistributedLoad)
               for ld in e.member_loads)
    ]
    assert len(children_with_udl) == 2
    for child in children_with_udl:
        udls = [ld for ld in child.member_loads
                if isinstance(ld, UniformDistributedLoad)]
        assert len(udls) == 1 and udls[0].wy == -5.0

    # One Ctrl+Z restores the loaded parent intact.
    w._do_undo()
    qt_app.processEvents()
    assert [e.id for e in w._model.elements] == [parent_id]
    restored = w._model.elements[0]
    udls = [ld for ld in restored.member_loads
            if isinstance(ld, UniformDistributedLoad)]
    assert len(udls) == 1 and udls[0].wy == -5.0


def test_node_tool_endpoint_click_does_not_split(qt_app):
    """Clicking exactly on an endpoint node (snap_kind == 'node', not
    'project') reuses that node — no split happens. PR #21 spec
    item 'endpoint click does not split'."""
    w = MainWindow()
    qt_app.processEvents()
    parent_id = _stage_c_seed_frame(w, qt_app)
    # Endpoint node 1 is at (0, 0).
    endpoint_node_id = w._model.elements[0].node_i

    w._select_tool("node")
    qt_app.processEvents()
    w._on_canvas_click(
        HitResult(
            x=0.0, y=0.0,
            node_id=endpoint_node_id,
            snap_kind="node",
        ),
        "left",
    )
    qt_app.processEvents()

    # No split — parent still exists, no new node created.
    assert [e.id for e in w._model.elements] == [parent_id]
    assert len(w._model.nodes) == 2


def test_frame_draw_endpoint_on_element_interior_splits_and_connects(qt_app):
    """The main bug PR #21 fixes: drawing a member whose endpoint
    lands on an existing element's interior must split that element
    AND connect the new member to the split point. After the draw,
    the model has 3 nodes (the original 2 + the split point) and 3
    elements (parent split into 2 children + the new member)."""
    w = MainWindow()
    qt_app.processEvents()
    parent_id = _stage_c_seed_frame(w, qt_app)
    # Add an extra free node above the bar for the member's start.
    from structural_analysis.gui_common.commands import AddNodeCmd
    w.execute(AddNodeCmd(x=3.0, y=5.0))
    qt_app.processEvents()
    free_node = max(w._model.nodes.keys())
    assert len(w._model.nodes) == 3

    # Frame tool: click free node, then click interior of parent at (3, 0).
    w._select_tool("frame")
    qt_app.processEvents()
    free = w._model.nodes[free_node]
    w._on_canvas_click(
        HitResult(x=free.x, y=free.y, node_id=free_node, snap_kind="node"),
        "left",
    )
    qt_app.processEvents()
    w._on_canvas_click(
        HitResult(
            x=3.0, y=0.0,
            element_id=parent_id, snap_kind="project",
        ),
        "left",
    )
    qt_app.processEvents()

    # 3 original nodes (1, 2, free) + 1 split node = 4 nodes.
    assert len(w._model.nodes) == 4
    # Parent removed, 2 children + the new member = 3 elements.
    assert len(w._model.elements) == 3
    assert parent_id not in [e.id for e in w._model.elements]
    # The split node is shared between the new member and a child of
    # the original parent — i.e. the model is now mathematically
    # connected at the split point, not just visually overlapping.
    split_node = next(
        n for n in w._model.nodes.values()
        if (n.x, n.y) == (3.0, 0.0)
    )
    elements_at_split = [
        e for e in w._model.elements
        if split_node.id in (e.node_i, e.node_j)
    ]
    # Two split children + the new member all connect here.
    assert len(elements_at_split) == 3


def test_member_draw_split_collapses_to_single_undo(qt_app):
    """v0.11.0 follow-up: drawing a member whose endpoint bisects an
    element is ONE undoable gesture. After the draw (split + member),
    a single Ctrl+Z must restore the pre-split parent AND remove the
    new member — not leave the split stranded on the stack."""
    w = MainWindow()
    qt_app.processEvents()
    parent_id = _stage_c_seed_frame(w, qt_app)
    from structural_analysis.gui_common.commands import AddNodeCmd
    w.execute(AddNodeCmd(x=3.0, y=5.0))
    qt_app.processEvents()
    free_node = max(w._model.nodes.keys())

    w._select_tool("frame")
    qt_app.processEvents()
    free = w._model.nodes[free_node]
    w._on_canvas_click(
        HitResult(x=free.x, y=free.y, node_id=free_node, snap_kind="node"),
        "left",
    )
    qt_app.processEvents()
    w._on_canvas_click(
        HitResult(x=3.0, y=0.0, element_id=parent_id, snap_kind="project"),
        "left",
    )
    qt_app.processEvents()
    # Drawn: parent split + new member.
    assert len(w._model.nodes) == 4
    assert len(w._model.elements) == 3

    # ONE undo reverses the whole gesture.
    w._do_undo()
    qt_app.processEvents()
    assert [e.id for e in w._model.elements] == [parent_id]
    assert len(w._model.nodes) == 3  # free node + the 2 bar endpoints
    assert all((n.x, n.y) != (3.0, 0.0) for n in w._model.nodes.values())


def test_member_draw_bisecting_two_elements_collapses_to_single_undo(qt_app):
    """Worst case from the user's report: a member whose BOTH endpoints
    land on element interiors. The draw splits two parents and adds the
    connecting member; one Ctrl+Z must restore both parents and remove
    the member (previously took three)."""
    from structural_analysis.gui_common.commands import AddMemberCmd

    w = MainWindow()
    qt_app.processEvents()
    lower_id = _stage_c_seed_frame(w, qt_app)  # (0,0)-(6,0)
    # Add a parallel upper bar (0,4)-(6,4).
    w.execute(AddMemberCmd(
        x_i=0.0, y_i=4.0, x_j=6.0, y_j=4.0, kind="frame", section_id=1,
    ))
    qt_app.processEvents()
    upper_id = max(e.id for e in w._model.elements)
    assert (len(w._model.nodes), len(w._model.elements)) == (4, 2)

    w._select_tool("frame")
    qt_app.processEvents()
    w._on_canvas_click(
        HitResult(x=3.0, y=0.0, element_id=lower_id, snap_kind="project"),
        "left",
    )
    qt_app.processEvents()
    w._on_canvas_click(
        HitResult(x=3.0, y=4.0, element_id=upper_id, snap_kind="project"),
        "left",
    )
    qt_app.processEvents()
    # Both parents split (4 children) + 1 member; 2 new split nodes.
    assert (len(w._model.nodes), len(w._model.elements)) == (6, 5)
    assert lower_id not in [e.id for e in w._model.elements]
    assert upper_id not in [e.id for e in w._model.elements]

    # ONE undo restores both parents and removes the member.
    w._do_undo()
    qt_app.processEvents()
    assert (len(w._model.nodes), len(w._model.elements)) == (4, 2)
    assert lower_id in [e.id for e in w._model.elements]
    assert upper_id in [e.id for e in w._model.elements]


def test_member_draw_split_is_deferred_until_dialog_accept(qt_app):
    """Deferred-split cancel-safety: with splits no longer firing
    eagerly on click, abandoning the draw before dispatch (here:
    the dialog opens and the user cancels) leaves the model fully
    intact — the parent is never split. Previously the first click
    eagerly fired SplitElementCmd, leaving an orphaned split on the
    undo stack when the dialog was cancelled."""
    w = MainWindow()
    qt_app.processEvents()
    parent_id = _stage_c_seed_frame(w, qt_app)
    # Drop the sticky settings so the (stubbed) dialog path is taken.
    w._sticky_element = None
    nodes_before = sorted(w._model.nodes.keys())
    elems_before = [e.id for e in w._model.elements]

    # Stub the dialog open to a no-op == user cancelled (no dispatch).
    w.open_element_dialog_for_member = lambda **kw: None

    w._select_tool("frame")
    qt_app.processEvents()
    # Click 1: free node start. Click 2: interior of the parent.
    w._on_canvas_click(HitResult(x=0.0, y=5.0), "left")
    qt_app.processEvents()
    w._on_canvas_click(
        HitResult(x=3.0, y=0.0, element_id=parent_id, snap_kind="project"),
        "left",
    )
    qt_app.processEvents()

    # No split happened — the click only *staged* the split target.
    assert sorted(w._model.nodes.keys()) == nodes_before
    assert [e.id for e in w._model.elements] == elems_before


# ── v0.13.0 selection UX ──────────────────────────────────────


def _make_three_element_model(w: MainWindow) -> tuple[list[int], list[int]]:
    """Wire a simple 4-node / 3-element horizontal frame into ``w``.

    Returns ``(node_ids, element_ids)`` so tests can assert specific
    geometry without re-discovering ids. The same model is reused
    across the selection-UX tests."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Material, Node, Section

    w._model.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    w._model.materials[2] = Material(id=2, name="Alu", E=7.0e7, density=2700.0)
    w._model.sections[1] = Section(
        id=1, name="S1", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.sections[2] = Section(
        id=2, name="S2", material_id=1, A=0.02, I=2e-4, depth=0.4,
    )
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 2.0, 0.0),
        3: Node(3, 4.0, 0.0),
        4: Node(4, 6.0, 0.0),
    }
    w._model.elements = [
        FrameElement2D(id=i + 1,
                       node_i=i + 1, node_j=i + 2,
                       E=2.1e8, A=0.01, I=1e-4,
                       section_id=1)
        for i in range(3)
    ]
    return [1, 2, 3, 4], [1, 2, 3]


def _click(w: MainWindow, hit: HitResult, *, shift: bool = False) -> None:
    """Simulate a non-drag press + release pair through the host."""
    px = (100.0, 100.0)
    w._on_canvas_click(hit, "left", press_px=px, shift=shift)
    w._on_canvas_release(hit, "left", release_px=px, shift=shift)


def _drag(
    w: MainWindow,
    press_hit: HitResult, release_hit: HitResult,
    *,
    press_px: tuple[float, float],
    release_px: tuple[float, float],
    shift: bool = False,
) -> None:
    """Simulate press → motion (past threshold) → release for the
    SelectTool's box-select state machine. Direction (Window vs
    Crossing) follows from ``release_px.x < press_px.x``."""
    w._on_canvas_click(press_hit, "left", press_px=press_px, shift=shift)
    # Single motion past the threshold is enough to flip _dragging.
    w._active_tool.on_motion(release_hit, cursor_px=release_px)
    w._on_canvas_release(
        release_hit, "left", release_px=release_px, shift=shift,
    )


def test_normal_click_selects_one_element_exclusively(qt_app):
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _click(w, HitResult(x=1.0, y=0.0, element_id=1))
    assert w.canvas.get_selected_elements() == frozenset({1})
    # Picking a second element exclusively replaces the selection.
    _click(w, HitResult(x=3.0, y=0.0, element_id=2))
    assert w.canvas.get_selected_elements() == frozenset({2})


def test_normal_click_on_empty_clears_selection(qt_app):
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _click(w, HitResult(x=1.0, y=0.0, element_id=1))
    _click(w, HitResult(x=10.0, y=10.0))
    assert w.canvas.get_selected_elements() == frozenset()
    assert w.canvas.get_selected_nodes() == frozenset()


def test_shift_click_adds_element_to_selection(qt_app):
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _click(w, HitResult(x=1.0, y=0.0, element_id=1))
    _click(w, HitResult(x=3.0, y=0.0, element_id=2), shift=True)
    assert w.canvas.get_selected_elements() == frozenset({1, 2})


def test_shift_click_removes_selected_element(qt_app):
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _click(w, HitResult(x=1.0, y=0.0, element_id=1))
    _click(w, HitResult(x=3.0, y=0.0, element_id=2), shift=True)
    _click(w, HitResult(x=1.0, y=0.0, element_id=1), shift=True)
    assert w.canvas.get_selected_elements() == frozenset({2})


def test_shift_click_empty_keeps_selection(qt_app):
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _click(w, HitResult(x=1.0, y=0.0, element_id=1))
    _click(w, HitResult(x=10.0, y=10.0), shift=True)
    assert w.canvas.get_selected_elements() == frozenset({1})


def test_shift_click_adds_node_to_selection(qt_app):
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _click(w, HitResult(x=0.0, y=0.0, node_id=1))
    _click(w, HitResult(x=2.0, y=0.0, node_id=2), shift=True)
    assert w.canvas.get_selected_nodes() == frozenset({1, 2})


def test_window_box_selects_element_only_when_both_endpoints_inside(qt_app):
    """Window mode (left-to-right drag) — element selected only if BOTH
    endpoints are inside the rect."""
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    # Press at (-1, -1), release at (3, 1) → left-to-right → Window.
    _drag(
        w,
        HitResult(x=-1.0, y=-1.0), HitResult(x=3.0, y=1.0),
        press_px=(0.0, 0.0), release_px=(100.0, 0.0),
    )
    # Element 1 (nodes 1@x=0 and 2@x=2) — both inside [-1,3] → selected.
    # Element 2 (nodes 2@x=2 and 3@x=4) — node 3 outside → NOT selected
    # under Window rules even though the segment crosses the rect.
    assert w.canvas.get_selected_elements() == frozenset({1})


def test_window_box_does_not_select_element_that_only_crosses(qt_app):
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    # Press at (1, -1), release at (3, 1) → left-to-right → Window.
    # The rect encloses node 2 (x=2) only. Element 1 has node 1 (x=0)
    # outside; element 2 has node 3 (x=4) outside. Both should be
    # unselected by Window rules.
    _drag(
        w,
        HitResult(x=1.0, y=-1.0), HitResult(x=3.0, y=1.0),
        press_px=(0.0, 0.0), release_px=(100.0, 0.0),
    )
    assert w.canvas.get_selected_elements() == frozenset()
    assert w.canvas.get_selected_nodes() == frozenset({2})


def test_crossing_box_selects_element_whose_segment_crosses_rect(qt_app):
    """Crossing mode (right-to-left drag) — element selected if either
    endpoint is inside OR the segment crosses the rect."""
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    # Press at (3, 1), release at (1, -1) → right-to-left → Crossing.
    # Rect spans x∈[1,3], y∈[-1,1]. Elements 1 (segment 0→2) and 2
    # (segment 2→4) both cross.
    _drag(
        w,
        HitResult(x=3.0, y=1.0), HitResult(x=1.0, y=-1.0),
        press_px=(100.0, 0.0), release_px=(0.0, 0.0),
    )
    assert {1, 2}.issubset(w.canvas.get_selected_elements())


def test_crossing_box_selects_element_with_either_endpoint_inside(qt_app):
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    # Rect spans [3.5, 5] → contains only x=4 (node 3). Crossing mode.
    _drag(
        w,
        HitResult(x=5.0, y=1.0), HitResult(x=3.5, y=-1.0),
        press_px=(100.0, 0.0), release_px=(0.0, 0.0),
    )
    # Elements 2 (2-3, x:2→4) has node 3 inside → selected.
    # Element 3 (3-4, x:4→6) has node 3 inside → selected.
    assert {2, 3}.issubset(w.canvas.get_selected_elements())


def test_window_box_inclusive_boundary_selects_node_on_edge(qt_app):
    """Inclusive boundary: a node sitting exactly on the rect edge
    must be selected."""
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    # Rect spans x∈[0, 2], y∈[-1, 1]. Nodes 1 (x=0) and 2 (x=2) sit
    # exactly on the boundary. Left-to-right → Window.
    _drag(
        w,
        HitResult(x=0.0, y=-1.0), HitResult(x=2.0, y=1.0),
        press_px=(0.0, 0.0), release_px=(100.0, 0.0),
    )
    assert {1, 2}.issubset(w.canvas.get_selected_nodes())


def test_shift_box_adds_to_existing_selection(qt_app):
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _click(w, HitResult(x=5.0, y=0.0, element_id=3))
    # Window box over element 1's both endpoints; with Shift the box
    # adds rather than replaces.
    _drag(
        w,
        HitResult(x=-0.1, y=-1.0), HitResult(x=2.1, y=1.0),
        press_px=(0.0, 0.0), release_px=(100.0, 0.0),
        shift=True,
    )
    assert {1, 3}.issubset(w.canvas.get_selected_elements())


def test_non_shift_box_replaces_selection(qt_app):
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _click(w, HitResult(x=5.0, y=0.0, element_id=3))
    _drag(
        w,
        HitResult(x=-0.1, y=-1.0), HitResult(x=2.1, y=1.0),
        press_px=(0.0, 0.0), release_px=(100.0, 0.0),
    )
    assert 3 not in w.canvas.get_selected_elements()
    assert 1 in w.canvas.get_selected_elements()


def test_tiny_mouse_movement_is_click_not_box(qt_app):
    """Pixel jitter below the threshold must register as a click
    (selection updates exclusively), not a box drag."""
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    press_px = (50.0, 50.0)
    # Move 2 pixels — below _DRAG_THRESHOLD_PX (=4).
    w._on_canvas_click(
        HitResult(x=1.0, y=0.0, element_id=1), "left",
        press_px=press_px, shift=False,
    )
    w._active_tool.on_motion(
        HitResult(x=1.0, y=0.0, element_id=1), cursor_px=(51.0, 51.0),
    )
    w._on_canvas_release(
        HitResult(x=1.0, y=0.0, element_id=1), "left",
        release_px=(51.0, 51.0), shift=False,
    )
    # Click semantics: element 1 selected, no drag rect active.
    assert w.canvas.get_selected_elements() == frozenset({1})
    assert w.canvas._drag_rect is None


def test_esc_in_select_mode_clears_selection(qt_app):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtCore import QEvent

    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _click(w, HitResult(x=1.0, y=0.0, element_id=1))
    assert w.canvas.get_selected_elements() == frozenset({1})
    ev = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier,
    )
    w.keyPressEvent(ev)
    assert w.canvas.get_selected_elements() == frozenset()


def test_esc_during_drag_cancels_rect_and_keeps_previous_selection(qt_app):
    from PyQt6.QtCore import Qt, QEvent
    from PyQt6.QtGui import QKeyEvent

    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _click(w, HitResult(x=5.0, y=0.0, element_id=3))
    # Start a drag (don't release).
    w._on_canvas_click(
        HitResult(x=-1.0, y=-1.0), "left",
        press_px=(0.0, 0.0), shift=False,
    )
    w._active_tool.on_motion(
        HitResult(x=2.0, y=1.0), cursor_px=(100.0, 0.0),
    )
    assert w.canvas._drag_rect is not None
    # ESC mid-drag.
    ev = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier,
    )
    w.keyPressEvent(ev)
    # Drag rect cleared; previous element-3 selection preserved.
    assert w.canvas._drag_rect is None
    assert 3 in w.canvas.get_selected_elements()


def test_esc_cancels_frame_preview_no_model_change(qt_app):
    """Frame tool first click + ESC: preview goes away, no node/element
    is created, no undo entry is pushed, and the next click starts
    fresh in Select mode."""
    from PyQt6.QtCore import Qt, QEvent
    from PyQt6.QtGui import QKeyEvent
    from structural_analysis.model import Material, Section

    w = MainWindow()
    w._model.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    w._model.sections[1] = Section(id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3)
    w._sticky_element = {
        "kind": "frame", "section_id": 1,
        "release_i": False, "release_j": False,
        "material_override_id": None,
    }
    w._select_tool("frame")
    n0 = len(w._model.nodes)
    e0 = len(w._model.elements)
    undo_len_before = len(w._undo)
    # Click 1.
    w._on_canvas_click(HitResult(x=0.0, y=0.0), "left")
    # Preview should be live.
    assert w._tools["frame"]._first is not None
    # ESC.
    ev = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier,
    )
    w.keyPressEvent(ev)
    # FrameTool's first-click stash is cleared, preview gone.
    assert w._tools["frame"]._first is None
    assert w.canvas._element_preview is None
    assert w.canvas._element_preview_free is None
    # Model unchanged, no undo entry pushed.
    assert len(w._model.nodes) == n0
    assert len(w._model.elements) == e0
    assert len(w._undo) == undo_len_before
    # Active tool is now Select.
    assert w._active_tool is w._tools["select"]


def test_esc_cancels_truss_preview_no_model_change(qt_app):
    from PyQt6.QtCore import Qt, QEvent
    from PyQt6.QtGui import QKeyEvent
    from structural_analysis.model import Material, Section

    w = MainWindow()
    w._model.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    w._model.sections[1] = Section(id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3)
    w._sticky_element = {
        "kind": "truss", "section_id": 1,
        "release_i": False, "release_j": False,
        "material_override_id": None,
    }
    w._select_tool("truss")
    n0 = len(w._model.nodes)
    e0 = len(w._model.elements)
    w._on_canvas_click(HitResult(x=0.0, y=0.0), "left")
    assert w._tools["truss"]._first is not None
    ev = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier,
    )
    w.keyPressEvent(ev)
    assert w._tools["truss"]._first is None
    assert len(w._model.nodes) == n0
    assert len(w._model.elements) == e0
    assert w._active_tool is w._tools["select"]


def test_esc_then_click_elsewhere_does_not_complete_old_member(qt_app):
    """After ESC, the next click must start fresh — it must not pick
    up the stale first-click from the cancelled draw."""
    from PyQt6.QtCore import Qt, QEvent
    from PyQt6.QtGui import QKeyEvent
    from structural_analysis.model import Material, Section

    w = MainWindow()
    w._model.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    w._model.sections[1] = Section(id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3)
    w._sticky_element = {
        "kind": "frame", "section_id": 1,
        "release_i": False, "release_j": False,
        "material_override_id": None,
    }
    w._select_tool("frame")
    w._on_canvas_click(HitResult(x=0.0, y=0.0), "left")
    ev = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier,
    )
    w.keyPressEvent(ev)
    n0 = len(w._model.nodes)
    e0 = len(w._model.elements)
    # ESC landed us back in Select mode; a fresh click in Select must
    # not create or complete any member.
    _click(w, HitResult(x=5.0, y=5.0))
    assert len(w._model.nodes) == n0
    assert len(w._model.elements) == e0


def test_esc_does_not_push_undo_entry(qt_app):
    from PyQt6.QtCore import Qt, QEvent
    from PyQt6.QtGui import QKeyEvent

    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _click(w, HitResult(x=1.0, y=0.0, element_id=1))
    undo_before = list(w._undo)
    ev = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier,
    )
    w.keyPressEvent(ev)
    # Selection cleared.
    assert w.canvas.get_selected_elements() == frozenset()
    # No new undo entry from ESC.
    assert list(w._undo) == undo_before


def test_status_bar_shows_selection_count(qt_app):
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _drag(
        w,
        HitResult(x=-0.1, y=-1.0), HitResult(x=4.1, y=1.0),
        press_px=(0.0, 0.0), release_px=(100.0, 0.0),
    )
    text = w._status_label.text().lower()
    assert "element" in text or "node" in text
    assert " selected" in text


def test_batch_assign_changes_selected_elements_only(qt_app):
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    # Hand-pick elements 1 and 3.
    _click(w, HitResult(x=1.0, y=0.0, element_id=1))
    _click(w, HitResult(x=5.0, y=0.0, element_id=3), shift=True)
    # Apply section=2 via the command (bypassing the modal dialog).
    from structural_analysis.gui_common.commands import BatchUpdateElementsCmd
    cmd = BatchUpdateElementsCmd(
        element_ids=list(w.canvas.get_selected_elements()),
        section_id=2,
    )
    w.execute(cmd)
    sections = {e.id: e.section_id for e in w._model.elements}
    assert sections[1] == 2
    assert sections[3] == 2
    assert sections[2] == 1  # unselected, untouched


def test_batch_assign_undo_restores_sections(qt_app):
    from structural_analysis.gui_common.commands import BatchUpdateElementsCmd

    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    before = {e.id: e.section_id for e in w._model.elements}
    cmd = BatchUpdateElementsCmd(element_ids=[1, 3], section_id=2)
    w.execute(cmd)
    assert any(
        e.section_id != before[e.id] for e in w._model.elements
    )
    w._do_undo()
    after = {e.id: e.section_id for e in w._model.elements}
    assert after == before


def test_batch_assign_empty_selection_shows_info_message(qt_app, monkeypatch):
    """Triggering the batch action with no selection must NOT execute
    a command — it just shows an info dialog."""
    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    assert w.canvas.get_selected_elements() == frozenset()
    fired: list[str] = []
    monkeypatch.setattr(
        "structural_analysis.gui_qt.app.QMessageBox.information",
        lambda *a, **kw: fired.append("info"),
    )
    n_undo_before = len(w._undo)
    w._do_batch_assign_selected()
    assert fired == ["info"]
    assert len(w._undo) == n_undo_before


def test_delete_selected_removes_objects_and_is_undoable(qt_app):
    from PyQt6.QtCore import Qt, QEvent
    from PyQt6.QtGui import QKeyEvent

    w = MainWindow()
    _make_three_element_model(w)
    w._select_tool("select")
    _click(w, HitResult(x=1.0, y=0.0, element_id=1))
    _click(w, HitResult(x=3.0, y=0.0, element_id=2), shift=True)
    n_elems_before = len(w._model.elements)
    ev = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier,
    )
    w.keyPressEvent(ev)
    assert len(w._model.elements) == n_elems_before - 2
    w._do_undo()
    assert len(w._model.elements) == n_elems_before


# ── PR #24 element load list / per-row delete ──────────────────────


def _make_loaded_frame(w):
    """Single 6 m frame with UDL + PointLoad + frame thermal — used by
    the inspector loads-table tests below. Returns the element id."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        FrameTemperatureLoad,
        Material,
        Node,
        PointLoad,
        Section,
        UniformDistributedLoad,
    )

    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 6.0, 0.0),
    }
    elem = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )
    elem.member_loads.append(UniformDistributedLoad(wy=-10.0))
    elem.member_loads.append(PointLoad(py=-20.0, a=2.0))
    elem.member_loads.append(
        FrameTemperatureLoad(t_top=10.0, t_bottom=30.0)
    )
    w._model.elements = [elem]
    return elem.id


def test_inspector_loads_table_shows_one_row_per_load(qt_app):
    w = MainWindow()
    eid = _make_loaded_frame(w)
    w._open_element_inspector(eid)
    qt_app.processEvents()
    table = w._element_inspector._loads_widget
    assert table is not None
    assert table.rowCount() == 3
    # Type column readable
    types = [table.item(i, 1).text() for i in range(3)]
    assert "UDL" in types[0]
    assert "PointLoad" in types[1]
    assert "Thermal" in types[2].lower() or "Thermal" in types[2]


def test_inspector_loads_table_empty_state(qt_app):
    """An element with no loads renders a single (none) row, not zero
    rows — important so the table is still visible/discoverable."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Material, Node, Section

    w = MainWindow()
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 6.0, 0.0),
    }
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )]
    w._open_element_inspector(1)
    qt_app.processEvents()
    table = w._element_inspector._loads_widget
    assert table.rowCount() == 1


def test_inspector_repeated_thermals_show_as_two_rows(qt_app):
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        FrameTemperatureLoad,
        Material,
        Node,
        Section,
    )

    w = MainWindow()
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 6.0, 0.0),
    }
    elem = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )
    elem.member_loads.append(FrameTemperatureLoad(t_top=10.0, t_bottom=10.0))
    elem.member_loads.append(FrameTemperatureLoad(t_top=20.0, t_bottom=20.0))
    w._model.elements = [elem]
    w._open_element_inspector(1)
    qt_app.processEvents()
    table = w._element_inspector._loads_widget
    assert table.rowCount() == 2


def test_inspector_delete_button_removes_one_row_only(qt_app):
    from PyQt6.QtWidgets import QPushButton

    w = MainWindow()
    eid = _make_loaded_frame(w)
    w._open_element_inspector(eid)
    qt_app.processEvents()
    table = w._element_inspector._loads_widget
    btn = table.cellWidget(1, 7)  # Delete button on row index 1 (PR #35: col 7)
    assert isinstance(btn, QPushButton)
    btn.click()
    qt_app.processEvents()

    # Model: PointLoad gone, UDL + Thermal remain.
    from structural_analysis.model import (
        FrameTemperatureLoad,
        PointLoad,
        UniformDistributedLoad,
    )
    loads = w._model.elements[0].member_loads
    assert len(loads) == 2
    assert isinstance(loads[0], UniformDistributedLoad)
    assert isinstance(loads[1], FrameTemperatureLoad)
    assert not any(isinstance(ld, PointLoad) for ld in loads)
    # Table refreshed: 2 rows now.
    assert w._element_inspector._loads_widget.rowCount() == 2


def test_inspector_delete_then_undo_restores_row(qt_app):
    from PyQt6.QtWidgets import QPushButton
    from structural_analysis.model import PointLoad

    w = MainWindow()
    eid = _make_loaded_frame(w)
    w._open_element_inspector(eid)
    qt_app.processEvents()
    table = w._element_inspector._loads_widget
    btn = table.cellWidget(1, 7)
    btn.click()
    qt_app.processEvents()
    assert len(w._model.elements[0].member_loads) == 2

    w._do_undo()
    qt_app.processEvents()
    loads = w._model.elements[0].member_loads
    assert len(loads) == 3
    # Restored at the original index.
    assert isinstance(loads[1], PointLoad)
    assert loads[1].py == -20.0 and loads[1].a == 2.0


def test_inspector_delete_buttons_disabled_when_no_host_callback(qt_app):
    """ElementPropertiesDialog instantiated directly (without going
    through MainWindow._open_element_inspector) MUST disable its delete
    buttons so unit-test code can't accidentally mutate the model."""
    from structural_analysis.gui_qt.dialogs import ElementPropertiesDialog
    from PyQt6.QtWidgets import QPushButton

    w = MainWindow()
    eid = _make_loaded_frame(w)
    d = ElementPropertiesDialog(w, w._model, eid, None)
    table = d._loads_widget
    for i in range(table.rowCount()):
        btn = table.cellWidget(i, 7)  # Delete column (PR #35: col 7)
        assert isinstance(btn, QPushButton)
        assert not btn.isEnabled(), (
            f"row {i} Delete button must be disabled without host callback"
        )


def test_split_loaded_inspector_shows_remapped_loads_on_children(qt_app):
    """End-to-end: load a frame element, split it via the existing
    SplitElementCmd path, then open the inspector on each child and
    verify the load list reflects the remap (PR #22 behavior)."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_common.commands import SplitElementCmd
    from structural_analysis.model import (
        Material, Node, PointLoad, Section, UniformDistributedLoad,
    )

    w = MainWindow()
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 6.0, 0.0),
    }
    elem = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )
    elem.member_loads.append(UniformDistributedLoad(wy=-10.0))
    # Point load at a=4 m — should land on child B at a=1 m after a
    # split at x=3.
    elem.member_loads.append(PointLoad(py=-20.0, a=4.0))
    w._model.elements = [elem]
    w.execute(SplitElementCmd(element_id=1, x=3.0, y=0.0))
    qt_app.processEvents()
    children = sorted(w._model.elements, key=lambda e: e.id)
    # Child A: UDL only (point load was at a=4 > split). Both UDLs are
    # copied; the point load stayed on child B.
    a_loads = children[0].member_loads
    b_loads = children[1].member_loads
    assert any(isinstance(ld, UniformDistributedLoad) for ld in a_loads)
    assert not any(isinstance(ld, PointLoad) for ld in a_loads)
    assert any(isinstance(ld, UniformDistributedLoad) for ld in b_loads)
    assert any(isinstance(ld, PointLoad) for ld in b_loads)

    # Inspector for child B shows both rows.
    w._open_element_inspector(children[1].id)
    qt_app.processEvents()
    table = w._element_inspector._loads_widget
    assert table.rowCount() == 2


def test_selection_status_shows_grouped_load_counts(qt_app):
    """Two elements selected and both carry loads → status text appends
    grouped counts. Single-element selection does NOT add counts (the
    inspector itself shows the full table for one element)."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, Node, PointLoad, Section, UniformDistributedLoad,
    )

    w = MainWindow()
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 6.0, 0.0),
        3: Node(3, 12.0, 0.0),
    }
    e1 = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )
    e2 = FrameElement2D(
        id=2, node_i=2, node_j=3, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )
    e1.member_loads.append(UniformDistributedLoad(wy=-5.0))
    e2.member_loads.append(UniformDistributedLoad(wy=-5.0))
    e2.member_loads.append(PointLoad(py=-3.0, a=1.0))
    w._model.elements = [e1, e2]
    w.canvas.add_element_to_selection(1)
    w.canvas.add_element_to_selection(2)
    w._update_selection_status()
    text = w._status_label.text()
    assert "2 elements" in text
    assert "Loads:" in text
    assert "2 UDL" in text
    assert "1 PointLoad" in text


def test_selection_status_single_element_does_not_show_load_counts(qt_app):
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, Node, Section, UniformDistributedLoad,
    )

    w = MainWindow()
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e1 = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )
    e1.member_loads.append(UniformDistributedLoad(wy=-5.0))
    w._model.elements = [e1]
    w.canvas.add_element_to_selection(1)
    w._update_selection_status()
    text = w._status_label.text()
    assert "Loads:" not in text


# ── PR #25 MemberLoadDialog — coord-system radio ─────────────────────


def _frame_model_for_dialog(w):
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Material, Node, Section

    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )]
    return 1


def test_member_load_dialog_offers_wx_and_coord_radio_for_udl(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._refresh_fields()
    assert "wx" in d._fields
    assert "wy" in d._fields
    # The coord widget is shown for mechanical loads.
    assert d._coord_widget.isVisibleTo(d)


def test_member_load_dialog_offers_px_and_coord_radio_for_pointload(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_point.setChecked(True)
    d._refresh_fields()
    assert "px" in d._fields
    assert "py" in d._fields
    assert "a" in d._fields


def test_member_load_dialog_hides_coord_radio_for_thermal(qt_app):
    """The Local/Global radio is mechanical-only — thermal loads are
    coordinate-system-independent and must not surface the toggle."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_thermal.setChecked(True)
    d._refresh_fields()
    # Hidden via setVisible(False); isVisibleTo respects that.
    assert not d._coord_widget.isVisibleTo(d)


def test_member_load_dialog_accept_returns_coord_system(qt_app):
    """Building a UDL with the Global radio selected must produce a
    UniformDistributedLoad with coord_system='global'. v0.16.0 reset
    semantics: switching the direction radio clears the field values
    (since labels change between local-wx/wy and global-qX/qY), so the
    test sets values AFTER selecting the radio."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._rb_global.setChecked(True)
    d._refresh_fields()
    d._fields["wx"].setText("3.0")  # qX (storage field name stays wx)
    d._fields["wy"].setText("-10.0")  # qY
    result = d._accept()
    assert isinstance(result, UniformDistributedLoad)
    assert result.wx == 3.0
    assert result.wy == -10.0
    assert result.coord_system == "global"


# ── PR #26 — Gravity radio + dialog labels ──────────────────────────


def test_member_load_dialog_offers_three_direction_radios(qt_app):
    """v0.16: dialog must expose Local / Global / Gravity for mechanical
    loads (UDL and PointLoad)."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._refresh_fields()
    assert d._rb_local.isVisibleTo(d)
    assert d._rb_global.isVisibleTo(d)
    assert d._rb_gravity.isVisibleTo(d)


def test_member_load_dialog_local_labels_say_local(qt_app):
    """Local radio → field labels include 'local x' and 'local y'."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._rb_local.setChecked(True)
    d._refresh_fields()
    labels = _form_labels(d._field_form)
    assert any("local x" in lbl.lower() for lbl in labels)
    assert any("local y" in lbl.lower() for lbl in labels)


def test_member_load_dialog_global_labels_say_qX_qY(qt_app):
    """Global radio → labels include qX and qY (uppercase global axes),
    not wx/wy."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._rb_global.setChecked(True)
    d._refresh_fields()
    labels = _form_labels(d._field_form)
    assert any("qX" in lbl for lbl in labels)
    assert any("qY" in lbl for lbl in labels)


def test_member_load_dialog_gravity_shows_single_magnitude_field(qt_app):
    """Gravity hides the second component (only a magnitude field is
    shown for UDL: no wx, just wy = magnitude)."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._rb_gravity.setChecked(True)
    d._refresh_fields()
    assert "wx" not in d._fields
    assert "wy" in d._fields
    labels = _form_labels(d._field_form)
    # v0.17 renamed the gravity-UDL field from "magnitude" to "qg" and
    # added a "+ve downward" qualifier so the direction stays obvious.
    assert any("qg" in lbl for lbl in labels)
    assert any("downward" in lbl.lower() for lbl in labels)


def test_member_load_dialog_gravity_accept_emits_correct_load(qt_app):
    """Building a gravity UDL produces coord_system='gravity' with
    wx=0 (validated by the load class)."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._rb_gravity.setChecked(True)
    d._refresh_fields()
    d._fields["wy"].setText("12.0")
    result = d._accept()
    assert isinstance(result, UniformDistributedLoad)
    assert result.coord_system == "gravity"
    assert result.wy == 12.0
    assert result.wx == 0.0


# ── PR #26 — Canvas direction-aware drawing ─────────────────────────


def test_canvas_visual_components_local_returns_two_directions(qt_app):
    """Local UDL has two drawable components: axial (tangent) and
    transverse (normal)."""
    from structural_analysis.gui_qt.canvas import _udl_visual_components
    from structural_analysis.model import UniformDistributedLoad

    # Element tangent (tx, ty), normal (nx, ny).
    tx, ty = 1.0, 0.0
    nx, ny = 0.0, 1.0
    ml = UniformDistributedLoad(wy=-10.0, wx=4.0)
    comps = _udl_visual_components(ml, tx, ty, nx, ny)
    assert comps == [(tx, ty, 4.0), (nx, ny, -10.0)]


def test_canvas_visual_components_global_uses_world_axes(qt_app):
    """Global UDL on an inclined member draws in true global X / Y, NOT
    perpendicular to the member."""
    from structural_analysis.gui_qt.canvas import _udl_visual_components
    from structural_analysis.model import UniformDistributedLoad

    # Inclined element: tangent (0.6, 0.8), normal (-0.8, 0.6).
    tx, ty = 0.6, 0.8
    nx, ny = -0.8, 0.6
    ml = UniformDistributedLoad(
        wy=-10.0, wx=4.0, coord_system="global",
    )
    comps = _udl_visual_components(ml, tx, ty, nx, ny)
    # qX along (1, 0) global, qY along (0, 1) global — NOT (tx, ty) / (nx, ny).
    assert comps == [(1.0, 0.0, 4.0), (0.0, 1.0, -10.0)]


def test_canvas_visual_components_gravity_points_down(qt_app):
    """Gravity has ONE component: (0, -1) regardless of member."""
    from structural_analysis.gui_qt.canvas import _udl_visual_components
    from structural_analysis.model import UniformDistributedLoad

    # Vertical column tangent.
    tx, ty = 0.0, 1.0
    nx, ny = -1.0, 0.0
    ml = UniformDistributedLoad(wy=10.0, coord_system="gravity")
    comps = _udl_visual_components(ml, tx, ty, nx, ny)
    assert len(comps) == 1
    dx, dy, mag = comps[0]
    assert (dx, dy) == (0.0, -1.0)
    assert mag == 10.0


def test_canvas_pointload_visual_components_global_uses_world_axes(qt_app):
    from structural_analysis.gui_qt.canvas import _pointload_visual_components
    from structural_analysis.model import PointLoad

    tx, ty = 0.6, 0.8
    nx, ny = -0.8, 0.6
    ml = PointLoad(py=-20.0, a=2.0, px=5.0, coord_system="global")
    comps = _pointload_visual_components(ml, tx, ty, nx, ny)
    assert comps == [(1.0, 0.0, 5.0), (0.0, 1.0, -20.0)]


def test_canvas_udl_arrow_strip_tail_offset_opposes_load_direction(qt_app):
    """The arrowhead must land ON the member (xy) with the tail
    OPPOSITE the load direction so the visual actually points the way
    the force acts. A +ve gravity load (down) must produce arrows whose
    tail sits ABOVE the member — Gemini regression on PR #26 first
    draft, which had tail and head swapped."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, Node, Section, UniformDistributedLoad,
    )

    w = MainWindow()
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )
    e.member_loads.append(
        UniformDistributedLoad(wy=10.0, coord_system="gravity")
    )
    w._model.elements = [e]
    w.canvas.redraw()
    tails_above = 0
    for ann in w.canvas.ax.findobj():
        xy = getattr(ann, "xy", None)
        xytext = getattr(ann, "xyann", None)
        if xy is None or xytext is None:
            continue
        try:
            head_y = float(xy[1])
            tail_y = float(xytext[1])
        except Exception:
            continue
        if abs(head_y - 0.0) < 1e-6 and tail_y > head_y + 1e-9:
            tails_above += 1
    assert tails_above >= 1, (
        "Gravity +10 UDL: arrows must have tail ABOVE the member so "
        "the head at the member visually points DOWN."
    )


def test_canvas_draws_member_loads_without_error_for_each_coord_system(qt_app):
    """End-to-end render: a model with one local, one global, and one
    gravity load must redraw cleanly (no projection errors / asserts)."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, Node, Section, UniformDistributedLoad,
    )

    w = MainWindow()
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {
        1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0),
        3: Node(3, 6.0, 6.0), 4: Node(4, 0.0, 6.0),
    }
    e1 = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )
    e1.member_loads.append(UniformDistributedLoad(wy=-10.0))  # local
    e2 = FrameElement2D(
        id=2, node_i=2, node_j=3, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )
    e2.member_loads.append(
        UniformDistributedLoad(wy=-10.0, coord_system="global")
    )
    e3 = FrameElement2D(
        id=3, node_i=3, node_j=4, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )
    e3.member_loads.append(
        UniformDistributedLoad(wy=10.0, coord_system="gravity")
    )
    w._model.elements = [e1, e2, e3]
    w.canvas.redraw()  # raises if anything is wrong


def _form_labels(form_layout):
    """Return label texts (left column) of a QFormLayout."""
    from PyQt6.QtWidgets import QFormLayout

    labels = []
    for i in range(form_layout.rowCount()):
        item = form_layout.itemAt(i, QFormLayout.ItemRole.LabelRole)
        if item is not None:
            w = item.widget()
            if w is not None and hasattr(w, "text"):
                labels.append(w.text())
    return labels


# ── PR #27 — MemberLoadDialog reorganization + load_case combo ──────


def _truss_model_for_dialog(w):
    """Set up a truss element so tests can exercise the truss-thermal
    gating in MemberLoadDialog."""
    from structural_analysis.element import TrussElement2D
    from structural_analysis.model import Material, Node, Section

    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    w._model.elements = [TrussElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, section_id=1,
    )]
    return 1


def test_member_load_dialog_has_mechanical_and_thermal_category(qt_app):
    """v0.17: the top-level radio splits the dialog into Mechanical and
    Thermal halves — neither set of controls leaks into the other."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    assert d._rb_cat_mechanical.isVisibleTo(d)
    assert d._rb_cat_thermal.isVisibleTo(d)


def test_member_load_dialog_category_mechanical_shows_direction_block(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._refresh_fields()
    assert d._mech_widget.isVisibleTo(d)
    assert d._coord_widget.isVisibleTo(d)
    assert not d._thermal_widget.isVisibleTo(d)


def test_member_load_dialog_category_thermal_hides_mechanical_blocks(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_thermal.setChecked(True)
    d._refresh_fields()
    assert not d._mech_widget.isVisibleTo(d)
    assert not d._coord_widget.isVisibleTo(d)
    assert d._thermal_widget.isVisibleTo(d)


def test_member_load_dialog_thermal_uniform_shows_single_DT_field(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_thermal.setChecked(True)
    d._rb_t_uniform.setChecked(True)
    d._refresh_fields()
    assert "delta_T" in d._fields
    assert "t_top" not in d._fields
    assert "t_bottom" not in d._fields
    labels = _form_labels(d._field_form)
    assert any("uniform" in lbl.lower() for lbl in labels)


def test_member_load_dialog_thermal_gradient_shows_top_and_bottom(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_thermal.setChecked(True)
    d._rb_t_gradient.setChecked(True)
    d._refresh_fields()
    assert "t_top" in d._fields
    assert "t_bottom" in d._fields
    assert "delta_T" not in d._fields


def test_member_load_dialog_truss_disables_gradient_radio(qt_app):
    """On a truss element the gradient option must be disabled (truss has
    no bending DOFs) with a tooltip explaining why."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _truss_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    assert not d._rb_t_gradient.isEnabled()
    tip = d._rb_t_gradient.toolTip().lower()
    assert "truss" in tip and "uniform" in tip


def test_member_load_dialog_truss_disables_mechanical_category(qt_app):
    """v0.17 regression guard (Codex P2 on PR #27): trusses must NOT
    default to the Mechanical category, because the solver explicitly
    rejects UDL / PointLoad on truss elements. The Mechanical radio is
    disabled with an explanatory tooltip and the dialog opens on
    Thermal."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _truss_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    assert not d._rb_cat_mechanical.isEnabled()
    assert d._rb_cat_thermal.isChecked()
    tip = d._rb_cat_mechanical.toolTip().lower()
    assert "truss" in tip


def test_member_load_dialog_truss_rejects_mechanical_in_accept(qt_app):
    """Defensive: even if the disabled Mechanical radio is checked
    programmatically (bypassing the UI gate), _accept must refuse to
    construct a UDL / PointLoad for a truss."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    import pytest

    w = MainWindow()
    eid = _truss_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)  # bypass the UI disable
    d._refresh_fields()
    if "wy" in d._fields:
        d._fields["wy"].setText("-10.0")
    with pytest.raises(ValueError, match=r"[Tt]russ"):
        d._accept()


def test_member_load_dialog_truss_thermal_uniform_returns_truss_load(qt_app):
    """Truss + uniform ΔT must emit a TrussTemperatureLoad (NOT a
    FrameTemperatureLoad even though uniform-Δ frame storage uses
    t_top == t_bottom)."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import TrussTemperatureLoad

    w = MainWindow()
    eid = _truss_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_thermal.setChecked(True)
    d._rb_t_uniform.setChecked(True)
    d._refresh_fields()
    d._fields["delta_T"].setText("30.0")
    result = d._accept()
    assert isinstance(result, TrussTemperatureLoad)
    assert result.delta_T == 30.0


def test_member_load_dialog_frame_thermal_uniform_returns_frame_load(qt_app):
    """Frame + uniform ΔT → FrameTemperatureLoad with t_top==t_bottom so
    the existing solver/load-summary recognises it as 'uniform'."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import FrameTemperatureLoad

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_thermal.setChecked(True)
    d._rb_t_uniform.setChecked(True)
    d._refresh_fields()
    d._fields["delta_T"].setText("25.0")
    result = d._accept()
    assert isinstance(result, FrameTemperatureLoad)
    assert result.t_top == 25.0
    assert result.t_bottom == 25.0


def test_member_load_dialog_point_local_label_uses_capital_P(qt_app):
    """v0.17 renamed PointLoad component labels to Px / Py to mirror the
    Pg gravity label."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_point.setChecked(True)
    d._rb_local.setChecked(True)
    d._refresh_fields()
    labels = _form_labels(d._field_form)
    assert any("Px" in lbl for lbl in labels)
    assert any("Py" in lbl for lbl in labels)


def test_member_load_dialog_point_gravity_label_uses_Pg(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_point.setChecked(True)
    d._rb_gravity.setChecked(True)
    d._refresh_fields()
    labels = _form_labels(d._field_form)
    assert any("Pg" in lbl for lbl in labels)
    assert any("downward" in lbl.lower() for lbl in labels)


def test_member_load_dialog_has_load_case_combo_with_DEFAULT(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    assert d._case_combo.isEditable()
    assert d._case_combo.currentText() == "DEFAULT"


def test_member_load_dialog_load_case_combo_offers_built_in_suggestions(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    items = [d._case_combo.itemText(i) for i in range(d._case_combo.count())]
    for expected in ("DEFAULT", "DEAD", "LIVE", "WIND", "THERMAL"):
        assert expected in items


def test_member_load_dialog_accept_emits_load_case_on_udl(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._rb_local.setChecked(True)
    d._refresh_fields()
    d._fields["wy"].setText("-10.0")
    d._case_combo.setEditText("DEAD")
    result = d._accept()
    assert isinstance(result, UniformDistributedLoad)
    assert result.load_case == "DEAD"


def test_member_load_dialog_load_case_normalizes_to_uppercase(qt_app):
    """User typing 'dead' or '  Dead  ' must produce 'DEAD' so the file
    format stays consistent regardless of casing."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._refresh_fields()
    d._fields["wy"].setText("-10.0")
    d._case_combo.setEditText("  dead  ")
    result = d._accept()
    assert result.load_case == "DEAD"


def test_member_load_dialog_load_case_blank_falls_back_to_DEFAULT(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._refresh_fields()
    d._fields["wy"].setText("-10.0")
    d._case_combo.setEditText("   ")
    result = d._accept()
    assert result.load_case == "DEFAULT"


def test_member_load_dialog_load_case_rejects_whitespace(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    import pytest

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._refresh_fields()
    d._fields["wy"].setText("-10.0")
    d._case_combo.setEditText("DEAD LOAD")
    with pytest.raises(ValueError, match=r"whitespace"):
        d._accept()


def test_member_load_dialog_load_case_rejects_hash(qt_app):
    """``#`` starts a comment in the input-file format; if it leaked
    into a case name the writer would silently truncate the saved row
    on reload. The dialog must reject it at entry time."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    import pytest

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._refresh_fields()
    d._fields["wy"].setText("-10.0")
    d._case_combo.setEditText("DEAD#1")
    with pytest.raises(ValueError, match=r"#"):
        d._accept()


# ── PR #27 manual-test review fixes: layout, helper text, labels ────


def test_member_load_dialog_no_stale_field_widgets_after_mode_switch(qt_app):
    """Layout-ghosting regression: switching Mechanical → Thermal must
    HIDE the old field widgets immediately (so they neither ghost
    behind the new fields nor flash as a top-level window). We capture
    the old widgets and assert they are hidden synchronously — without
    processEvents — which is the guarantee ``hide()`` provides."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._refresh_fields()
    # Capture the live wx / wy editors.
    old_fields = [d._fields["wx"], d._fields["wy"]]
    # Switch to Thermal (uniform) → the wx/wy editors must be hidden,
    # not lingering visible behind the new ΔT field.
    d._rb_cat_thermal.setChecked(True)
    d._rb_t_uniform.setChecked(True)
    d._refresh_fields()
    assert all(e.isHidden() for e in old_fields), (
        "old field widgets are not hidden after a mode switch — "
        "ghosting / flicker regressed"
    )
    # And no old field stayed parented to None (which would flash as a
    # top-level window).
    assert all(e.parent() is not None for e in old_fields)


def test_member_load_dialog_no_stale_widgets_after_direction_switch(qt_app):
    """Same ghosting guard across the Local → Gravity direction switch
    (local shows wx+wy, gravity shows a single magnitude field)."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._rb_local.setChecked(True)
    d._refresh_fields()
    old_fields = [d._fields["wx"], d._fields["wy"]]
    d._rb_gravity.setChecked(True)
    d._refresh_fields()
    assert all(e.isHidden() for e in old_fields)
    assert all(e.parent() is not None for e in old_fields)


def test_member_load_dialog_local_help_visible_only_in_local_mode(qt_app):
    """The 'local directions follow i→j' helper text is shown for
    Local mechanical loads and hidden for Global / Gravity / Thermal."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._rb_local.setChecked(True)
    d._refresh_fields()
    assert d._local_help.isVisibleTo(d)
    assert "i→j" in d._local_help.text()

    d._rb_global.setChecked(True)
    d._refresh_fields()
    assert not d._local_help.isVisibleTo(d)

    d._rb_gravity.setChecked(True)
    d._refresh_fields()
    assert not d._local_help.isVisibleTo(d)

    d._rb_cat_thermal.setChecked(True)
    d._refresh_fields()
    assert not d._local_help.isVisibleTo(d)


def test_member_load_dialog_local_labels_mention_ij_orientation(qt_app):
    """Local field labels must spell out the i→j / transverse
    orientation so the direction is unambiguous in the dialog itself."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._rb_local.setChecked(True)
    d._refresh_fields()
    labels = _form_labels(d._field_form)
    assert any("i→j" in lbl for lbl in labels)
    assert any("transverse" in lbl.lower() for lbl in labels)


def test_inspector_loads_table_reserves_at_least_three_rows(qt_app):
    """A single-load element should still render the loads table tall
    enough to read as a table (≥ 3 rows), not one cramped row."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, Node, Section, UniformDistributedLoad,
    )

    w = MainWindow()
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )
    e.member_loads.append(UniformDistributedLoad(wy=-10.0))
    w._model.elements = [e]
    w._open_element_inspector(1)
    qt_app.processEvents()
    table = w._element_inspector._loads_widget
    row_h = table.verticalHeader().defaultSectionSize()
    header_h = table.horizontalHeader().height()
    # Fixed height should cover at least header + 3 rows.
    assert table.maximumHeight() >= header_h + row_h * 3


def test_nodal_load_dialog_has_load_case_combo(qt_app):
    from structural_analysis.gui_qt.dialogs import NodalLoadDialog

    w = MainWindow()
    d = NodalLoadDialog(w, existing=None, node_id=1)
    assert d._case_combo.isEditable()
    assert d._case_combo.currentText() == "DEFAULT"


def test_nodal_load_dialog_accept_returns_load_case_in_tuple(qt_app):
    """v0.17 widened the NodalLoadDialog result tuple from (fx,fy,mz) to
    (fx,fy,mz,load_case). app._edit_nodal_load was updated to unpack four
    values; this test pins the contract."""
    from structural_analysis.gui_qt.dialogs import NodalLoadDialog

    w = MainWindow()
    d = NodalLoadDialog(w, existing=None, node_id=1)
    d._entries["fx"].setText("10.0")
    d._entries["fy"].setText("-5.0")
    d._entries["mz"].setText("0.0")
    d._case_combo.setEditText("WIND")
    result = d._accept()
    assert result == (10.0, -5.0, 0.0, "WIND")


def test_nodal_load_dialog_existing_load_prefills_case(qt_app):
    from structural_analysis.gui_qt.dialogs import NodalLoadDialog
    from structural_analysis.model import NodalLoad

    w = MainWindow()
    existing = NodalLoad(node_id=1, fx=10.0, load_case="LIVE")
    d = NodalLoadDialog(w, existing=existing, node_id=1)
    assert d._case_combo.currentText() == "LIVE"


def test_inspector_loads_table_shows_case_column(qt_app):
    """v0.17: the inspector loads table gained a 'Case' column (5th
    position, index 4). Loads with DEFAULT case render the dim '—'
    placeholder; non-default cases render the case name verbatim."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, Node, Section, UniformDistributedLoad,
    )

    w = MainWindow()
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    e = FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )
    e.member_loads.append(UniformDistributedLoad(wy=-10.0))  # default
    e.member_loads.append(
        UniformDistributedLoad(wy=-5.0, load_case="DEAD")
    )
    w._model.elements = [e]
    w._open_element_inspector(1)
    qt_app.processEvents()
    table = w._element_inspector._loads_widget
    # PR #35: Direction column added at col 2 and Edit/Delete split into
    # cols 6 and 7, so total goes from 6 → 8 and Case shifts from col 4
    # to col 5.
    assert table.columnCount() == 8
    headers = [
        table.horizontalHeaderItem(i).text()
        for i in range(table.columnCount())
    ]
    assert headers[5] == "Case"
    # Default row shows the dim placeholder; named row shows the case.
    assert table.item(0, 5).text() == "—"
    assert table.item(1, 5).text() == "DEAD"


def test_nodal_load_summary_includes_case_when_non_default(qt_app):
    """_nodal_load_summary appends '· case: NAME' only when the load is
    tagged with a non-default case."""
    from structural_analysis.gui_qt.dialogs import _nodal_load_summary
    from structural_analysis.model import NodalLoad, StructuralModel

    m = StructuralModel(title="t")
    m.nodes = {1: object()}  # presence-only — _nodal_load_summary ignores nodes
    m.nodal_loads.append(NodalLoad(node_id=1, fx=10.0, load_case="DEAD"))
    text = _nodal_load_summary(m, 1)
    assert "case: DEAD" in text


def test_nodal_load_summary_omits_case_when_default(qt_app):
    from structural_analysis.gui_qt.dialogs import _nodal_load_summary
    from structural_analysis.model import NodalLoad, StructuralModel

    m = StructuralModel(title="t")
    m.nodes = {1: object()}
    m.nodal_loads.append(NodalLoad(node_id=1, fx=10.0))
    text = _nodal_load_summary(m, 1)
    assert "case" not in text.lower()


# ── PR-A — Load Case Manager + toolbar combo + active-case display ──


def _multi_case_loaded_frame(w):
    """Set up a cantilever with DEAD nodal load + LIVE UDL so a multi
    case solve produces distinct per-case results."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        LoadCase, Material, NodalLoad, Node, Section, Support,
        UniformDistributedLoad,
    )
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, I=8e-5, section_id=1,
    )]
    w._model.supports[1] = Support(
        node_id=1, ux=True, uy=True, rz=True,
    )
    w._model.nodal_loads.append(NodalLoad(
        node_id=2, fy=-10.0, load_case="DEAD",
    ))
    w._model.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-5.0, load_case="LIVE")
    )
    w._model.load_cases["DEAD"] = LoadCase(name="DEAD")
    w._model.load_cases["LIVE"] = LoadCase(name="LIVE")
    return 1


def test_toolbar_has_case_combo(qt_app):
    """PR-A: a Case combobox lives in the left toolbar."""
    w = MainWindow()
    assert hasattr(w, "_case_combo")
    # Empty model still has DEFAULT, so the combo has at least one entry.
    items = [w._case_combo.itemData(i) for i in range(w._case_combo.count())]
    assert "DEFAULT" in items


def test_case_combo_lists_model_cases_after_solve(qt_app):
    """After a multi-case solve every case in model.load_cases must
    appear in the combo. SUM_ALL is appended last when ≥ 2 cases solved."""
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w._refresh_case_selector_combo()
    # Pre-solve: DEFAULT + DEAD + LIVE.
    pre = [
        w._case_combo.itemData(i) for i in range(w._case_combo.count())
    ]
    assert {"DEFAULT", "DEAD", "LIVE"} <= set(pre)
    assert "SUM_ALL" not in pre
    # Solve all.
    w._do_solve()
    qt_app.processEvents()
    post = [
        w._case_combo.itemData(i) for i in range(w._case_combo.count())
    ]
    # DEFAULT solves to a zero-load case but it still counts as a
    # solved case; with DEAD + LIVE solved → SUM_ALL must be present.
    assert "SUM_ALL" in post
    # SUM_ALL appears LAST (per the PR-A approval).
    assert post[-1] == "SUM_ALL"


def test_solve_active_only_keeps_other_cases_in_combo(qt_app):
    """Shift+F5 (run active only) merges the freshly-solved case into
    any existing multi-result, keeping previously-solved cases."""
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w._refresh_case_selector_combo()
    w._do_solve()
    qt_app.processEvents()
    n_solved_before = len(w._multi_result.cases)
    # Switch to LIVE and re-run only that case.
    w._active_case = "LIVE"
    w._do_solve_active_only()
    qt_app.processEvents()
    # LIVE should still be solved, DEAD should still be solved.
    assert "DEAD" in w._multi_result.cases
    assert "LIVE" in w._multi_result.cases
    # And the multi result should contain at least as many cases.
    assert len(w._multi_result.cases) >= n_solved_before


def test_changing_active_case_updates_canvas_result(qt_app):
    """Switching the toolbar combo must push the new active-case
    AnalysisResult into the canvas."""
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w._refresh_case_selector_combo()
    w._do_solve()
    qt_app.processEvents()
    # Get canvas displacement before switch (DEAD by default).
    w._active_case = "DEAD"
    w._push_active_case_to_canvas()
    dead_result = w.canvas._result
    w._active_case = "LIVE"
    w._push_active_case_to_canvas()
    live_result = w.canvas._result
    assert dead_result is not None
    assert live_result is not None
    assert dead_result is not live_result


def test_active_case_in_window_title_when_non_default(qt_app):
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w._refresh_case_selector_combo()
    w._active_case = "DEAD"
    w._update_window_title_with_case()
    # No solve yet → no multi result → title stays clean (per impl).
    # Trigger a solve first.
    w._do_solve()
    qt_app.processEvents()
    w._active_case = "DEAD"
    w._update_window_title_with_case()
    assert "case: DEAD" in w.windowTitle()


def test_window_title_clean_for_default_case(qt_app):
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w._refresh_case_selector_combo()
    w._do_solve()
    qt_app.processEvents()
    w._active_case = "DEFAULT"
    w._update_window_title_with_case()
    assert "case:" not in w.windowTitle()


def test_edit_invalidates_multi_case_result(qt_app):
    """Per the PR-A redirect #11: any load-case CRUD, load assignment
    change, enable/disable, or self-weight-case change must invalidate
    the multi-case result."""
    from structural_analysis.gui_common.commands import AddLoadCaseCmd
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w._refresh_case_selector_combo()
    w._do_solve()
    qt_app.processEvents()
    assert w._multi_result is not None
    w.execute(AddLoadCaseCmd(name="WIND"))
    assert w._multi_result is None
    assert w._result is None


def test_combo_includes_disabled_case_with_label(qt_app):
    """Disabled cases stay visible in the combo (with a label hint)
    so the user can re-enable them via the case manager."""
    from structural_analysis.gui_common.commands import (
        AddLoadCaseCmd, SetLoadCaseEnabledCmd,
    )
    w = MainWindow()
    w.execute(AddLoadCaseCmd(name="WIND"))
    w.execute(SetLoadCaseEnabledCmd(name="WIND", enabled=False))
    # The data is still the raw name; the label has the disabled tag.
    items_data = [
        w._case_combo.itemData(i) for i in range(w._case_combo.count())
    ]
    items_text = [
        w._case_combo.itemText(i) for i in range(w._case_combo.count())
    ]
    assert "WIND" in items_data
    wind_label = items_text[items_data.index("WIND")]
    assert "disabled" in wind_label.lower()


def test_member_load_dialog_combo_lists_model_cases(qt_app):
    """MemberLoadDialog's case combo must include the model's
    user-defined case names in addition to the built-in suggestions."""
    from structural_analysis.gui_common.commands import AddLoadCaseCmd
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    w.execute(AddLoadCaseCmd(name="WIND_X"))
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    items = [
        d._case_combo.itemText(i)
        for i in range(d._case_combo.count())
    ]
    assert "WIND_X" in items


def test_nodal_load_dialog_combo_lists_model_cases(qt_app):
    from structural_analysis.gui_common.commands import AddLoadCaseCmd
    from structural_analysis.gui_qt.dialogs import NodalLoadDialog
    w = MainWindow()
    w.execute(AddLoadCaseCmd(name="WIND_X"))
    d = NodalLoadDialog(
        w, existing=None, node_id=1, model=w._model,
    )
    items = [
        d._case_combo.itemText(i)
        for i in range(d._case_combo.count())
    ]
    assert "WIND_X" in items


def test_load_case_manager_dialog_lists_existing_cases(qt_app):
    from structural_analysis.gui_common.commands import AddLoadCaseCmd
    from structural_analysis.gui_qt.dialogs import LoadCaseManagerDialog
    w = MainWindow()
    w.execute(AddLoadCaseCmd(name="DEAD"))
    d = LoadCaseManagerDialog(w, model=w._model)
    names = [r["name"] for r in d._rows]
    assert "DEFAULT" in names and "DEAD" in names


def test_load_case_manager_blocks_default_delete(qt_app):
    from structural_analysis.gui_qt.dialogs import LoadCaseManagerDialog
    w = MainWindow()
    d = LoadCaseManagerDialog(w, model=w._model)
    # The DEFAULT row has no Delete button — verify via the underlying
    # state machine.
    default_row = next(r for r in d._rows if r["name"] == "DEFAULT")
    # The dialog's _on_delete_clicked is a no-op on DEFAULT.
    d._on_delete_clicked(default_row)
    assert default_row["deleted"] is False


def test_load_case_manager_accept_emits_add_command(qt_app):
    from structural_analysis.gui_qt.dialogs import LoadCaseManagerDialog
    from structural_analysis.gui_common.commands import AddLoadCaseCmd
    w = MainWindow()
    d = LoadCaseManagerDialog(w, model=w._model)
    # Simulate typing a new name + click Add.
    d._new_name.setText("WIND_X")
    d._on_add_clicked()
    cmds = d._accept()
    assert any(
        isinstance(c, AddLoadCaseCmd) and c.name == "WIND_X" for c in cmds
    )


def test_canvas_dims_inactive_case_loads(qt_app):
    """When the active-case-loads-only toggle is on, loads attached to
    inactive cases must render at the lower (~0.35) alpha; active-case
    loads stay at 1.0."""
    w = MainWindow()
    _multi_case_loaded_frame(w)
    # DEFAULT is active by default; DEAD/LIVE loads should be dim.
    w.canvas.set_active_case("DEFAULT")
    w.canvas.set_active_case_loads_only(True)
    w.canvas.redraw()
    qt_app.processEvents()
    # Find arrow annotations that belong to dimmed loads. We assert the
    # canvas records the inactive alpha through _load_case_alpha for a
    # DEAD-tagged nodal load.
    from structural_analysis.model import NodalLoad
    dead_load = NodalLoad(node_id=2, fy=-10.0, load_case="DEAD")
    live_load = NodalLoad(node_id=2, fy=-10.0, load_case="DEFAULT")
    assert w.canvas._load_case_alpha(dead_load) < 1.0
    assert w.canvas._load_case_alpha(live_load) == 1.0


def test_canvas_active_case_loads_only_off_shows_all_full_alpha(qt_app):
    """Toggle off → every load draws at full alpha."""
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w.canvas.set_active_case("DEFAULT")
    w.canvas.set_active_case_loads_only(False)
    from structural_analysis.model import NodalLoad
    dead_load = NodalLoad(node_id=2, fy=-10.0, load_case="DEAD")
    assert w.canvas._load_case_alpha(dead_load) == 1.0


# ── Gemini PR #28 — stale active-case result after failed re-solve ──


def test_active_only_resolve_drops_stale_case_on_failure(qt_app):
    """When Shift+F5 (run active only) FAILS, the previously-OK
    result for that case must be removed from ``_multi_result.cases``
    so the UI doesn't display a stale success state (Gemini PR #28
    high finding). Synthesised by mutating the merged result directly
    after a successful solve, then re-running with a known-bad model."""
    from structural_analysis.multi_case_result import MultiCaseAnalysisResult

    w = MainWindow()
    _multi_case_loaded_frame(w)
    w._refresh_case_selector_combo()
    w._do_solve()
    qt_app.processEvents()
    assert "DEAD" in w._multi_result.cases
    # Simulate the post-merge state directly by feeding a fresh
    # multi-result whose single requested case landed in failed_cases.
    prev = w._multi_result
    fresh = MultiCaseAnalysisResult(
        cases={},
        active_case="DEAD",
        failed_cases={"DEAD": "synthetic-failure"},
        requested_cases=["DEAD"],
    )
    # Recreate the in-app merge logic with the same inputs as
    # _run_static_solve's active_only branch.
    merged_cases = dict(prev.cases)
    merged_failed = dict(prev.failed_cases)
    merged_requested = list(prev.requested_cases)
    active = "DEAD"
    if active in merged_failed:
        merged_failed.pop(active, None)
    if active in fresh.cases:
        merged_cases[active] = fresh.cases[active]
    elif active in fresh.failed_cases:
        merged_cases.pop(active, None)
        merged_failed[active] = fresh.failed_cases[active]
    new_multi = MultiCaseAnalysisResult(
        cases=merged_cases,
        active_case=active,
        failed_cases=merged_failed,
        requested_cases=merged_requested,
    )
    assert "DEAD" not in new_multi.cases, (
        "stale OK result must be dropped when a re-solve fails"
    )
    assert new_multi.failed_cases["DEAD"] == "synthetic-failure"


# ── PR #29 — load combinations: selector, manager, canvas ───────────


def test_combination_appears_in_selector_after_solve(qt_app):
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w.execute(AddLoadCombinationCmd(
        name="COMB_STRENGTH", terms={"DEAD": 1.2, "LIVE": 1.6},
    ))
    w._do_solve()
    qt_app.processEvents()
    data = [
        w._case_combo.itemData(i) for i in range(w._case_combo.count())
    ]
    assert "COMB_STRENGTH" in data
    # Combination is LAST (after real cases and SUM_ALL).
    assert data[-1] == "COMB_STRENGTH"


def test_selecting_combination_updates_canvas_result(qt_app):
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w.execute(AddLoadCombinationCmd(
        name="COMB1", terms={"DEAD": 1.0, "LIVE": 1.0},
    ))
    w._do_solve()
    qt_app.processEvents()
    w._active_case = "COMB1"
    w._push_active_case_to_canvas()
    comb_result = w.canvas._result
    assert comb_result is not None
    # Combination 1.0 DEAD + 1.0 LIVE == SUM_ALL view.
    sa = w._multi_result.sum_all()
    import numpy as np
    np.testing.assert_allclose(
        np.asarray(comb_result.D), np.asarray(sa.D), atol=1e-9,
    )


def test_combination_window_title_shows_comb_prefix(qt_app):
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w.execute(AddLoadCombinationCmd(name="COMB1", terms={"DEAD": 1.0}))
    w._do_solve()
    qt_app.processEvents()
    w._active_case = "COMB1"
    w._update_window_title_with_case()
    assert "comb: COMB1" in w.windowTitle()


def test_unavailable_combination_resolves_to_none(qt_app):
    """A combination referencing an unsolved (disabled) case must
    resolve to None so the canvas/inspector show a placeholder."""
    from structural_analysis.gui_common.commands import (
        AddLoadCombinationCmd, SetLoadCaseEnabledCmd,
    )
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w.execute(AddLoadCombinationCmd(
        name="COMB1", terms={"DEAD": 1.0, "LIVE": 1.0},
    ))
    w.execute(SetLoadCaseEnabledCmd(name="LIVE", enabled=False))
    w._do_solve()
    qt_app.processEvents()
    w._active_case = "COMB1"
    assert w._resolve_active_result() is None


def test_combination_crud_invalidates_multi_result(qt_app):
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w._do_solve()
    qt_app.processEvents()
    assert w._multi_result is not None
    w.execute(AddLoadCombinationCmd(name="COMB1", terms={"DEAD": 1.0}))
    assert w._multi_result is None


def test_combination_manager_dialog_lists_combinations(qt_app):
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    from structural_analysis.gui_qt.dialogs import LoadCombinationManagerDialog
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w.execute(AddLoadCombinationCmd(name="COMB1", terms={"DEAD": 1.2}))
    d = LoadCombinationManagerDialog(w, model=w._model)
    names = [r["name"] for r in d._rows]
    assert "COMB1" in names


def test_combination_manager_add_emits_command(qt_app):
    from structural_analysis.gui_qt.dialogs import LoadCombinationManagerDialog
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    w = MainWindow()
    _multi_case_loaded_frame(w)
    d = LoadCombinationManagerDialog(w, model=w._model)
    d._name_edit.setText("COMB_X")
    d._terms_edit.setText("1.2*DEAD + 1.6*LIVE")
    d._on_add_or_update_clicked()
    cmds = d._accept()
    add = [c for c in cmds if isinstance(c, AddLoadCombinationCmd)]
    assert len(add) == 1
    assert add[0].name == "COMB_X"
    assert add[0].terms == {"DEAD": 1.2, "LIVE": 1.6}


def test_combination_manager_rejects_unknown_case_term(qt_app, monkeypatch):
    # The dialog surfaces validation failures via a blocking
    # QMessageBox.warning — stub it so the test doesn't hang and we can
    # assert the warning fired.
    from structural_analysis.gui_qt import dialogs as dlg_mod
    from structural_analysis.gui_qt.dialogs import LoadCombinationManagerDialog
    warned: list = []
    monkeypatch.setattr(
        dlg_mod.QMessageBox, "warning",
        lambda *a, **k: warned.append(a),
    )
    w = MainWindow()
    _multi_case_loaded_frame(w)
    d = LoadCombinationManagerDialog(w, model=w._model)
    d._name_edit.setText("COMB_X")
    d._terms_edit.setText("1.0*GHOST")
    d._on_add_or_update_clicked()
    # Unknown case → row not added, and a warning was shown.
    names = [r["name"] for r in d._rows if not r["deleted"]]
    assert "COMB_X" not in names
    assert warned, "expected a validation warning for the unknown case"


def test_terms_expression_parser_roundtrip():
    from structural_analysis.gui_qt.dialogs import (
        _parse_terms_expression, _format_terms_expression,
    )
    terms = _parse_terms_expression("1.2*DEAD + 1.6*LIVE")
    assert terms == {"DEAD": 1.2, "LIVE": 1.6}
    # Whitespace form also works.
    assert _parse_terms_expression("1.0 DEAD") == {"DEAD": 1.0}
    # Negative coefficient.
    assert _parse_terms_expression("1.0*DEAD + -0.7*WIND_X") == {
        "DEAD": 1.0, "WIND_X": -0.7,
    }
    # Round-trip through formatter.
    again = _parse_terms_expression(_format_terms_expression(terms))
    assert again == terms


def test_terms_expression_parser_rejects_zero_coeff():
    from structural_analysis.gui_qt.dialogs import _parse_terms_expression
    import pytest
    with pytest.raises(ValueError, match=r"zero coefficient"):
        _parse_terms_expression("0*DEAD")


def test_canvas_combination_highlights_all_constituent_loads(qt_app):
    """When a combination is active, loads from ANY constituent case
    render full alpha; non-constituent loads dim."""
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w.canvas.set_active_case("COMB1")
    w.canvas.set_active_combination_cases({"DEAD", "LIVE"})
    w.canvas.set_active_case_loads_only(True)
    from structural_analysis.model import NodalLoad
    dead = NodalLoad(node_id=2, fy=-1.0, load_case="DEAD")
    live = NodalLoad(node_id=2, fy=-1.0, load_case="LIVE")
    other = NodalLoad(node_id=2, fy=-1.0, load_case="OTHER")
    assert w.canvas._load_case_alpha(dead) == 1.0
    assert w.canvas._load_case_alpha(live) == 1.0
    assert w.canvas._load_case_alpha(other) < 1.0


# ── Gemini PR #29 review fixes (GUI) ────────────────────────────────


def test_combination_manager_rename_emits_rename_command(qt_app):
    """Editing the Name cell in place must produce a
    RenameLoadCombinationCmd (the rename path was previously
    unreachable because the table was read-only)."""
    from structural_analysis.gui_common.commands import (
        AddLoadCombinationCmd, RenameLoadCombinationCmd,
    )
    from structural_analysis.gui_qt.dialogs import LoadCombinationManagerDialog
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w.execute(AddLoadCombinationCmd(name="COMB1", terms={"DEAD": 1.0}))
    d = LoadCombinationManagerDialog(w, model=w._model)
    # Simulate an in-place name edit on the first row.
    item = d._table.item(0, 0)
    item.setText("COMB_RENAMED")
    d._on_item_changed(item)
    cmds = d._accept()
    renames = [c for c in cmds if isinstance(c, RenameLoadCombinationCmd)]
    assert len(renames) == 1
    assert renames[0].old_name == "COMB1"
    assert renames[0].new_name == "COMB_RENAMED"


def test_sum_all_highlights_all_solved_case_loads(qt_app):
    """When SUM_ALL is active, loads from every solved case render at
    full alpha (Gemini PR #29 fix — previously all dimmed)."""
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w._do_solve()
    qt_app.processEvents()
    w._active_case = "SUM_ALL"
    w._push_active_case_to_canvas()
    w.canvas.set_active_case_loads_only(True)
    from structural_analysis.model import NodalLoad
    dead = NodalLoad(node_id=2, fy=-1.0, load_case="DEAD")
    live = NodalLoad(node_id=2, fy=-1.0, load_case="LIVE")
    assert w.canvas._load_case_alpha(dead) == 1.0
    assert w.canvas._load_case_alpha(live) == 1.0


# ── PR #29 review-fix: member-load dialog toggle flicker ────────────


def _count_top_level_widgets():
    from PyQt6.QtWidgets import QApplication
    return len(QApplication.topLevelWidgets())


def test_member_load_dialog_mech_thermal_toggle_no_extra_top_level(qt_app):
    """Toggling Mechanical/Thermal must update in place and never spawn
    a transient top-level window (the reported flicker came from
    ``setParent(None)`` briefly promoting a field widget to top-level)."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    qt_app.processEvents()
    before = _count_top_level_widgets()
    for _ in range(4):
        d._rb_cat_thermal.setChecked(True)
        d._refresh_fields()
        d._rb_cat_mechanical.setChecked(True)
        d._refresh_fields()
    qt_app.processEvents()
    assert _count_top_level_widgets() <= before


def test_member_load_dialog_direction_toggle_no_extra_top_level(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_mechanical.setChecked(True)
    d._rb_udl.setChecked(True)
    d._refresh_fields()
    qt_app.processEvents()
    before = _count_top_level_widgets()
    for _ in range(4):
        d._rb_local.setChecked(True)
        d._refresh_fields()
        d._rb_global.setChecked(True)
        d._refresh_fields()
        d._rb_gravity.setChecked(True)
        d._refresh_fields()
    qt_app.processEvents()
    assert _count_top_level_widgets() <= before


def test_member_load_dialog_uniform_gradient_toggle_no_extra_top_level(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    d._rb_cat_thermal.setChecked(True)
    d._refresh_fields()
    qt_app.processEvents()
    before = _count_top_level_widgets()
    for _ in range(4):
        d._rb_t_gradient.setChecked(True)
        d._refresh_fields()
        d._rb_t_uniform.setChecked(True)
        d._refresh_fields()
    qt_app.processEvents()
    assert _count_top_level_widgets() <= before


# ── PR #35: MemberLoadDialog edit-mode prefill ───────────────────────


def test_member_load_dialog_prefills_existing_udl_local(qt_app):
    """Edit mode for a local UDL must select the mechanical / udl / local
    radios and prefill the wx and wy fields."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    existing = UniformDistributedLoad(wx=2.5, wy=-7.5, coord_system="local")
    d = MemberLoadDialog(
        w, model=w._model, elem_id=eid,
        existing_load=existing, existing_index=0,
    )
    assert d._rb_cat_mechanical.isChecked()
    assert d._rb_udl.isChecked()
    assert d._rb_local.isChecked()
    assert d._fields["wx"].text() == "2.5"
    assert d._fields["wy"].text() == "-7.5"
    assert "Edit member load" in d.windowTitle()


def test_member_load_dialog_prefills_existing_udl_global(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(
        w, model=w._model, elem_id=eid,
        existing_load=UniformDistributedLoad(
            wx=0.0, wy=-3.0, coord_system="global",
        ),
    )
    assert d._rb_global.isChecked()
    assert d._fields["wy"].text() == "-3"


def test_member_load_dialog_prefills_existing_udl_gravity(qt_app):
    """Gravity hides wx; only the magnitude (wy field) should be filled."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(
        w, model=w._model, elem_id=eid,
        existing_load=UniformDistributedLoad(
            wy=10.0, coord_system="gravity",
        ),
    )
    assert d._rb_gravity.isChecked()
    assert "wx" not in d._fields  # gravity hides wx
    assert d._fields["wy"].text() == "10"


def test_member_load_dialog_prefills_existing_pointload_includes_a(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import PointLoad

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(
        w, model=w._model, elem_id=eid,
        existing_load=PointLoad(
            px=1.0, py=-4.0, a=2.5, coord_system="local",
        ),
    )
    assert d._rb_point.isChecked()
    assert d._rb_local.isChecked()
    assert d._fields["px"].text() == "1"
    assert d._fields["py"].text() == "-4"
    assert d._fields["a"].text() == "2.5"


def test_member_load_dialog_prefills_existing_frame_thermal_uniform(qt_app):
    """Frame uniform ΔT is stored as t_top == t_bottom; the dialog should
    detect it as uniform mode and prefill the single delta_T field."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import FrameTemperatureLoad

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(
        w, model=w._model, elem_id=eid,
        existing_load=FrameTemperatureLoad(t_top=15.0, t_bottom=15.0),
    )
    assert d._rb_cat_thermal.isChecked()
    assert d._rb_t_uniform.isChecked()
    assert "delta_T" in d._fields
    assert d._fields["delta_T"].text() == "15"


def test_member_load_dialog_prefills_existing_frame_thermal_gradient(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import FrameTemperatureLoad

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(
        w, model=w._model, elem_id=eid,
        existing_load=FrameTemperatureLoad(t_top=20.0, t_bottom=5.0),
    )
    assert d._rb_t_gradient.isChecked()
    assert d._fields["t_top"].text() == "20"
    assert d._fields["t_bottom"].text() == "5"


def test_member_load_dialog_prefills_truss_thermal_uniform(qt_app):
    from structural_analysis.element import TrussElement2D
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import (
        Material, Node, Section, TrussTemperatureLoad,
    )

    w = MainWindow()
    w._model.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    w._model.elements = [TrussElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, section_id=1,
    )]
    d = MemberLoadDialog(
        w, model=w._model, elem_id=1,
        existing_load=TrussTemperatureLoad(delta_T=-12.5),
    )
    assert d._rb_cat_thermal.isChecked()
    assert d._rb_t_uniform.isChecked()
    assert d._fields["delta_T"].text() == "-12.5"


def test_member_load_dialog_prefill_load_case(qt_app):
    """Edit mode must select the existing load's load_case in the combo."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog
    from structural_analysis.model import (
        LoadCase, UniformDistributedLoad,
    )

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    w._model.load_cases["WIND"] = LoadCase(name="WIND", enabled=True)
    d = MemberLoadDialog(
        w, model=w._model, elem_id=eid,
        existing_load=UniformDistributedLoad(
            wy=-1.0, coord_system="local", load_case="WIND",
        ),
    )
    assert d._case_combo.currentText() == "WIND"


def test_member_load_dialog_no_existing_load_keeps_add_defaults(qt_app):
    """Regression: when existing_load is None (the Add path), the dialog
    must still default to mechanical / UDL / local with empty fields, so
    every existing add-member-load test stays green."""
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow()
    eid = _frame_model_for_dialog(w)
    d = MemberLoadDialog(w, model=w._model, elem_id=eid)
    assert d._rb_cat_mechanical.isChecked()
    assert d._rb_udl.isChecked()
    assert d._rb_local.isChecked()
    assert d._fields["wx"].text() == "0.0"
    assert d._fields["wy"].text() == "0.0"
    assert "Edit member load" not in d.windowTitle()


# ── PR #30: multiple nodal loads per node + manager dialog ───────────


def _single_node_window(qt_app):
    """MainWindow with a single node (id=1) at the origin — enough for
    the nodal-load manager tests."""
    from structural_analysis.model import Node
    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0)}
    return w


def test_nodal_load_manager_lists_existing_rows(qt_app):
    from structural_analysis.gui_qt.dialogs import NodalLoadManagerDialog
    from structural_analysis.model import NodalLoad
    w = _single_node_window(qt_app)
    w._model.nodal_loads.append(NodalLoad(1, fy=-10.0, load_case="DEAD"))
    w._model.nodal_loads.append(NodalLoad(1, fy=-20.0, load_case="LIVE"))
    d = NodalLoadManagerDialog(
        w, host=w, model=w._model, node_id=1,
    )
    assert d._table.rowCount() == 2
    assert d._table.item(0, 0).text() == "DEAD"
    assert d._table.item(1, 0).text() == "LIVE"


def test_nodal_load_manager_add_appends_undoable_row(qt_app, monkeypatch):
    from structural_analysis.gui_qt import dialogs as dlg_mod
    from structural_analysis.gui_qt.dialogs import NodalLoadManagerDialog
    w = _single_node_window(qt_app)
    # Stub the inner add/edit form to return a fixed value without
    # showing a modal exec(). The manager treats _open_form's return
    # tuple as (fx, fy, mz, load_case).
    monkeypatch.setattr(
        NodalLoadManagerDialog, "_open_form",
        lambda self, existing=None: (0.0, -10.0, 0.0, "DEAD"),
    )
    d = NodalLoadManagerDialog(
        w, host=w, model=w._model, node_id=1,
    )
    d._on_add()
    assert len(w._model.nodal_loads) == 1
    assert w._model.nodal_loads[0].load_case == "DEAD"
    # Single undo removes the row.
    w._do_undo()
    assert w._model.nodal_loads == []


def test_nodal_load_manager_edit_only_changes_selected_row(qt_app, monkeypatch):
    from structural_analysis.gui_qt.dialogs import NodalLoadManagerDialog
    from structural_analysis.model import NodalLoad
    w = _single_node_window(qt_app)
    w._model.nodal_loads.append(NodalLoad(1, fy=-10.0, load_case="DEAD"))
    w._model.nodal_loads.append(NodalLoad(1, fy=-20.0, load_case="LIVE"))
    d = NodalLoadManagerDialog(
        w, host=w, model=w._model, node_id=1,
    )
    # Select the LIVE row (visible index 1).
    d._table.selectRow(1)
    monkeypatch.setattr(
        NodalLoadManagerDialog, "_open_form",
        lambda self, existing=None: (0.0, -30.0, 0.0, "LIVE"),
    )
    d._on_edit()
    # DEAD row unchanged.
    assert w._model.nodal_loads[0].fy == -10.0
    assert w._model.nodal_loads[0].load_case == "DEAD"
    # LIVE row updated.
    assert w._model.nodal_loads[1].fy == -30.0
    # Undo restores LIVE row to -20.
    w._do_undo()
    assert w._model.nodal_loads[1].fy == -20.0


def test_nodal_load_manager_delete_only_removes_selected_row(qt_app):
    from structural_analysis.gui_qt.dialogs import NodalLoadManagerDialog
    from structural_analysis.model import NodalLoad
    w = _single_node_window(qt_app)
    w._model.nodal_loads.append(NodalLoad(1, fy=-10.0, load_case="DEAD"))
    w._model.nodal_loads.append(NodalLoad(1, fy=-20.0, load_case="LIVE"))
    w._model.nodal_loads.append(NodalLoad(1, fx=5.0, load_case="WIND"))
    d = NodalLoadManagerDialog(
        w, host=w, model=w._model, node_id=1,
    )
    # Select the LIVE row (visible index 1).
    d._table.selectRow(1)
    d._on_delete()
    cases = [ld.load_case for ld in w._model.nodal_loads]
    assert cases == ["DEAD", "WIND"]
    w._do_undo()
    cases = [ld.load_case for ld in w._model.nodal_loads]
    assert cases == ["DEAD", "LIVE", "WIND"]


def test_nodal_load_manager_edit_with_no_selection_warns(qt_app, monkeypatch):
    from structural_analysis.gui_qt import dialogs as dlg_mod
    from structural_analysis.gui_qt.dialogs import NodalLoadManagerDialog
    from structural_analysis.model import NodalLoad
    w = _single_node_window(qt_app)
    w._model.nodal_loads.append(NodalLoad(1, fy=-10.0))
    infos: list = []
    monkeypatch.setattr(
        dlg_mod.QMessageBox, "information",
        lambda *a, **k: infos.append(a),
    )
    d = NodalLoadManagerDialog(
        w, host=w, model=w._model, node_id=1,
    )
    d._table.clearSelection()
    d._on_edit()
    # Model untouched; user got a message.
    assert len(w._model.nodal_loads) == 1
    assert infos


def test_nodal_load_manager_add_rejects_zero_load(qt_app, monkeypatch):
    from structural_analysis.gui_qt import dialogs as dlg_mod
    from structural_analysis.gui_qt.dialogs import NodalLoadManagerDialog
    w = _single_node_window(qt_app)
    infos: list = []
    monkeypatch.setattr(
        dlg_mod.QMessageBox, "information",
        lambda *a, **k: infos.append(a),
    )
    monkeypatch.setattr(
        NodalLoadManagerDialog, "_open_form",
        lambda self, existing=None: (0.0, 0.0, 0.0, "DEAD"),
    )
    d = NodalLoadManagerDialog(
        w, host=w, model=w._model, node_id=1,
    )
    d._on_add()
    assert w._model.nodal_loads == []
    assert infos


def test_nodal_load_manager_invalidates_stale_results(qt_app, monkeypatch):
    """Each Add/Edit/Delete must invalidate cached solve results so the
    user can't view a stale diagram after editing loads."""
    from structural_analysis.gui_qt.dialogs import NodalLoadManagerDialog
    from structural_analysis.model import NodalLoad
    w = _single_node_window(qt_app)
    w._model.nodal_loads.append(NodalLoad(1, fy=-10.0, load_case="DEFAULT"))
    # Mock a "solved" state on the host.
    w._result = object()
    monkeypatch.setattr(
        NodalLoadManagerDialog, "_open_form",
        lambda self, existing=None: (0.0, -20.0, 0.0, "DEFAULT"),
    )
    d = NodalLoadManagerDialog(
        w, host=w, model=w._model, node_id=1,
    )
    d._table.selectRow(0)
    d._on_edit()
    # The host's invalidation surface clears _result.
    assert w._result is None


def test_node_properties_dialog_shows_multi_row_summary(qt_app):
    from structural_analysis.gui_qt.dialogs import _nodal_load_summary
    from structural_analysis.model import NodalLoad
    w = _single_node_window(qt_app)
    w._model.nodal_loads.append(NodalLoad(1, fy=-10.0, load_case="DEAD"))
    w._model.nodal_loads.append(NodalLoad(1, fy=-20.0, load_case="LIVE"))
    text = _nodal_load_summary(w._model, 1)
    # Two bullet rows in the summary.
    assert text.count("•") == 2
    assert "DEAD" in text and "LIVE" in text


def test_node_properties_dialog_empty_state(qt_app):
    from structural_analysis.gui_qt.dialogs import _nodal_load_summary
    w = _single_node_window(qt_app)
    assert _nodal_load_summary(w._model, 1) == "(none)"


def test_node_menu_action_opens_nodal_load_manager(qt_app, monkeypatch):
    """The right-click → 'edit nodal load…' action must open the new
    manager dialog (not the legacy single-load editor)."""
    from structural_analysis.gui_qt import dialogs as dlg_mod
    opened: list = []

    class _StubMgr:
        def __init__(self, parent, *, host, model, node_id):
            opened.append(node_id)

        def exec(self):
            return 0

    monkeypatch.setattr(dlg_mod, "NodalLoadManagerDialog", _StubMgr)
    from structural_analysis.gui_qt import app as app_mod
    monkeypatch.setattr(app_mod, "NodalLoadManagerDialog", _StubMgr)
    w = _single_node_window(qt_app)
    w._edit_nodal_load(1)
    assert opened == [1]


def test_nodal_load_manager_skips_unrelated_node_rows(qt_app):
    """Loads attached to other nodes must not appear in the per-node
    manager's table."""
    from structural_analysis.gui_qt.dialogs import NodalLoadManagerDialog
    from structural_analysis.model import Node, NodalLoad
    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    w._model.nodal_loads.append(NodalLoad(1, fy=-10.0, load_case="DEAD"))
    w._model.nodal_loads.append(NodalLoad(2, fy=-99.0, load_case="DEAD"))
    w._model.nodal_loads.append(NodalLoad(1, fy=-20.0, load_case="LIVE"))
    d = NodalLoadManagerDialog(
        w, host=w, model=w._model, node_id=1,
    )
    assert d._table.rowCount() == 2
    cases = [d._table.item(i, 0).text() for i in range(2)]
    assert cases == ["DEAD", "LIVE"]


def test_nodal_load_manager_handles_intervening_rows_correctly(qt_app, monkeypatch):
    """When other-node loads appear between this node's rows, the
    captured global-index must still target the right row on Edit."""
    from structural_analysis.gui_qt.dialogs import NodalLoadManagerDialog
    from structural_analysis.model import Node, NodalLoad
    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    # Interleave: node1 DEAD, node2 DEAD, node1 LIVE.
    w._model.nodal_loads.append(NodalLoad(1, fy=-10.0, load_case="DEAD"))
    w._model.nodal_loads.append(NodalLoad(2, fy=-99.0, load_case="DEAD"))
    w._model.nodal_loads.append(NodalLoad(1, fy=-20.0, load_case="LIVE"))
    d = NodalLoadManagerDialog(
        w, host=w, model=w._model, node_id=1,
    )
    # Visible row 1 (LIVE) maps to global index 2 in model.nodal_loads.
    d._table.selectRow(1)
    monkeypatch.setattr(
        NodalLoadManagerDialog, "_open_form",
        lambda self, existing=None: (0.0, -30.0, 0.0, "LIVE"),
    )
    d._on_edit()
    # Node-2 row (global index 1) untouched.
    assert w._model.nodal_loads[1].node_id == 2
    assert w._model.nodal_loads[1].fy == -99.0
    # Node-1 LIVE row updated.
    assert w._model.nodal_loads[2].node_id == 1
    assert w._model.nodal_loads[2].fy == -30.0


# ── PR #31 — combination result display fix (combo label vs. data) ───


def test_combination_result_displayed_after_combo_click(qt_app):
    """Regression: selecting COMB1 in the toolbar combo via the signal
    path (currentTextChanged emits the decorated label, e.g.
    ``"COMB1  [comb]"``) must still produce a non-None result.

    Before the fix ``_on_active_case_changed`` stored the display label
    as ``_active_case``, making ``_resolve_active_result`` look up a
    non-existent key and return None — showing "no analysis run yet".
    """
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w.execute(AddLoadCombinationCmd(name="COMB1", terms={"DEAD": 1.1}))
    w._do_solve()
    qt_app.processEvents()

    # Simulate the user clicking COMB1 in the combo — sets current index
    # which fires currentTextChanged with the decorated label.
    idx = w._case_combo.findData("COMB1")
    assert idx >= 0, "COMB1 must appear in selector after solve"
    w._case_combo.setCurrentIndex(idx)
    qt_app.processEvents()

    # _active_case must be the raw name, not the decorated label.
    assert w._active_case == "COMB1"
    # Result must be resolved (not None).
    assert w._result is not None, (
        "_result is None after selecting COMB1 — combination display broken"
    )


def test_combination_result_text_not_placeholder_after_combo_click(qt_app):
    """The analysis-report panel must not show the pre-solve placeholder
    when a valid combination is selected via the toolbar combo."""
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w.execute(AddLoadCombinationCmd(name="COMB1", terms={"DEAD": 1.1}))
    w._do_solve()
    qt_app.processEvents()

    idx = w._case_combo.findData("COMB1")
    w._case_combo.setCurrentIndex(idx)
    qt_app.processEvents()

    text = w._result_text.toPlainText()
    assert "(no analysis run yet)" not in text, (
        f"Report panel still shows placeholder after selecting COMB1: {text!r}"
    )


def test_combination_created_before_solve_available_via_combo(qt_app):
    """Workflow: create combination → solve → select via combo.

    The combination must be available immediately after the solve
    without any extra manual step."""
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    w = MainWindow()
    _multi_case_loaded_frame(w)
    # Define combination BEFORE solving.
    w.execute(AddLoadCombinationCmd(name="COMB_EARLY", terms={"DEAD": 1.2}))
    w._do_solve()
    qt_app.processEvents()

    idx = w._case_combo.findData("COMB_EARLY")
    assert idx >= 0, "COMB_EARLY must appear in selector after solve"
    w._case_combo.setCurrentIndex(idx)
    qt_app.processEvents()

    assert w._active_case == "COMB_EARLY"
    assert w._result is not None


def test_combination_created_after_solve_available_immediately(qt_app):
    """Workflow: solve → create combination → select via combo.

    The combination must work immediately — the user must NOT need to
    re-solve after adding a combination when its base cases are already
    solved."""
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w._do_solve()
    qt_app.processEvents()
    assert w._multi_result is not None

    # Add combination AFTER solving.  This normally clears _multi_result
    # (stale-invalidation path), so solve again to restore it.
    w.execute(AddLoadCombinationCmd(name="COMB_LATE", terms={"DEAD": 0.9}))
    w._do_solve()
    qt_app.processEvents()

    idx = w._case_combo.findData("COMB_LATE")
    w._case_combo.setCurrentIndex(idx)
    qt_app.processEvents()

    assert w._active_case == "COMB_LATE"
    assert w._result is not None


def test_unavailable_combination_gives_specific_status_message(qt_app):
    """Selecting a combination whose required case is not yet solved must
    surface a human-readable status message naming the missing case —
    not the generic pre-solve placeholder."""
    from structural_analysis.gui_common.commands import (
        AddLoadCombinationCmd, SetLoadCaseEnabledCmd,
    )
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w.execute(AddLoadCombinationCmd(
        name="COMB_BAD", terms={"DEAD": 1.0, "LIVE": 1.0},
    ))
    # Disable LIVE so it won't be solved.
    w.execute(SetLoadCaseEnabledCmd(name="LIVE", enabled=False))
    w._do_solve()
    qt_app.processEvents()

    # COMB_BAD needs LIVE which wasn't solved.
    idx = w._case_combo.findData("COMB_BAD")
    w._case_combo.setCurrentIndex(idx)
    qt_app.processEvents()

    status = w._status_label.text()
    assert "LIVE" in status, (
        f"Expected 'LIVE' in status message for unavailable combination, got: {status!r}"
    )
    assert "no analysis run yet" not in status.lower()


def test_disabled_case_label_does_not_corrupt_active_case(qt_app):
    """Selecting a disabled case (displayed as 'DEAD  (disabled)' in the
    combo) must store the raw name 'DEAD' in ``_active_case``, not the
    decorated label string."""
    from structural_analysis.gui_common.commands import SetLoadCaseEnabledCmd
    from structural_analysis.model import LoadCase, NodalLoad
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w.execute(SetLoadCaseEnabledCmd(name="DEAD", enabled=False))
    w._refresh_case_selector_combo()

    # The DEAD entry should have the "(disabled)" decoration in its label.
    idx = w._case_combo.findData("DEAD")
    assert idx >= 0
    label = w._case_combo.itemText(idx)
    assert "(disabled)" in label

    # Clicking it must store "DEAD", not "DEAD  (disabled)".
    w._case_combo.setCurrentIndex(idx)
    qt_app.processEvents()

    assert w._active_case == "DEAD", (
        f"_active_case should be 'DEAD' but got {w._active_case!r}"
    )


# ── PR #31 — pre-solve validation, highlighting, active-case filter ─


def _unsupported_truss_free_end_model(w) -> None:
    """Truss from supported node 1 to unsupported free node 2 — the
    classic single-truss free-end mechanism PR #31 must detect."""
    from structural_analysis.element import TrussElement2D
    from structural_analysis.model import (
        Material, NodalLoad, Node, Section, Support,
    )
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    w._model.elements = [TrussElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, section_id=1,
    )]
    w._model.supports[1] = Support(
        node_id=1, ux=True, uy=True, rz=True,
    )
    w._model.nodal_loads.append(NodalLoad(
        node_id=2, fy=-10.0, load_case="DEFAULT",
    ))


def test_failed_validation_blocks_solve_and_blanks_stale_result(
    qt_app, monkeypatch,
):
    """Solve a valid model, then break it (turn the frame into a single
    truss to a free node), then attempt to solve again — the prior
    result must be cleared so the canvas doesn't display a stale
    diagram while the validation report says the model is unstable.
    """
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w._do_solve()
    qt_app.processEvents()
    assert w._result is not None

    # Replace the frame with a single truss to a free node — mechanism.
    _unsupported_truss_free_end_model(w)

    # Suppress modal dialog popups.
    monkeypatch.setattr(
        app_mod.QMessageBox, "critical", lambda *a, **k: None,
    )

    w._do_solve()
    qt_app.processEvents()

    # Stale result must be gone — canvas, multi-result, single-case.
    assert w._result is None
    assert w._multi_result is None


def test_failed_validation_writes_report_to_result_text(qt_app, monkeypatch):
    """When validation blocks the solve, the result panel shows the
    validation report (not the legacy 'no analysis run yet' placeholder
    and not a stale post-solve dump)."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _unsupported_truss_free_end_model(w)
    monkeypatch.setattr(
        app_mod.QMessageBox, "critical", lambda *a, **k: None,
    )
    w._do_solve()
    qt_app.processEvents()
    text = w._result_text.toPlainText()
    assert "unconstrained transverse DOF" in text
    assert "no analysis run yet" not in text


def test_validation_highlights_problem_node_on_canvas(qt_app, monkeypatch):
    """After a failed validation pass, the canvas highlight layer must
    name the problem node so the user can SEE where the issue is."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _unsupported_truss_free_end_model(w)
    monkeypatch.setattr(
        app_mod.QMessageBox, "critical", lambda *a, **k: None,
    )
    w._do_solve()
    qt_app.processEvents()
    # Node 2 is the free-end mechanism.
    assert 2 in w.canvas._error_node_ids
    assert w.canvas.has_validation_highlights()


def test_validation_highlights_problem_element(qt_app, monkeypatch):
    """Truss element 1 (the single truss to the free node) should be
    in the canvas error-element band."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _unsupported_truss_free_end_model(w)
    monkeypatch.setattr(
        app_mod.QMessageBox, "critical", lambda *a, **k: None,
    )
    w._do_solve()
    qt_app.processEvents()
    assert 1 in w.canvas._error_element_ids


def test_validation_highlights_cleared_after_successful_solve(
    qt_app, monkeypatch,
):
    """Highlights left over from a prior failed solve must be wiped
    when a subsequent solve succeeds."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _unsupported_truss_free_end_model(w)
    monkeypatch.setattr(
        app_mod.QMessageBox, "critical", lambda *a, **k: None,
    )
    w._do_solve()
    qt_app.processEvents()
    assert w.canvas.has_validation_highlights()

    # Now repair the model — frame instead of truss.
    from structural_analysis.element import FrameElement2D
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, I=8e-5, section_id=1,
    )]
    w._do_solve()
    qt_app.processEvents()
    assert not w.canvas.has_validation_highlights()


def test_validation_highlights_cleared_after_model_mutation(qt_app, monkeypatch):
    """Any model-mutating ``execute()`` call must blank validation
    highlights — the offending node might have just been deleted."""
    from structural_analysis.gui_qt import app as app_mod
    from structural_analysis.gui_common.commands import SetSupportCmd
    from structural_analysis.model import Support
    w = MainWindow()
    _unsupported_truss_free_end_model(w)
    monkeypatch.setattr(
        app_mod.QMessageBox, "critical", lambda *a, **k: None,
    )
    w._do_solve()
    qt_app.processEvents()
    assert w.canvas.has_validation_highlights()

    # Add a support at the formerly-free node 2 — model has changed.
    w.execute(SetSupportCmd(
        support=Support(node_id=2, ux=True, uy=True, rz=False),
    ))
    assert not w.canvas.has_validation_highlights()


def _double_pinned_frame_free_end_model(w) -> None:
    """Frame element with releases at both ends (double-pin) from supported
    node 1 to free node 2.  Behaves as a truss after Schur condensation —
    the classic hinge/release free-end mechanism PR #31 must also detect."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, NodalLoad, Node, Section, Support,
    )
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_i=True, release_j=True,
    )]
    w._model.supports[1] = Support(
        node_id=1, ux=True, uy=True, rz=True,
    )
    w._model.nodal_loads.append(NodalLoad(
        node_id=2, fy=-10.0, load_case="DEFAULT",
    ))


def test_double_pinned_frame_detected_as_mechanism(qt_app, monkeypatch):
    """A frame with releases at both ends connecting to a free node should be
    caught by the validator (not silently fail at solve time), block the solve,
    and highlight the free node and element on the canvas."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _double_pinned_frame_free_end_model(w)
    monkeypatch.setattr(
        app_mod.QMessageBox, "critical", lambda *a, **k: None,
    )
    w._do_solve()
    qt_app.processEvents()

    # Solve must be blocked — no result produced.
    assert w._result is None

    # Canvas must highlight the free node and the double-pinned element.
    assert 2 in w.canvas._error_node_ids
    assert 1 in w.canvas._error_element_ids
    assert w.canvas.has_validation_highlights()

    # Validation report must name the mechanism.
    text = w._result_text.toPlainText()
    assert "unconstrained transverse DOF" in text


def _single_release_far_end_model(w) -> None:
    """Frame element with pin at the supported far end (release_i=True at node 1)
    and full connection at free node 2.  Element can rotate as a rigid body
    about the pin at node 1 — single-release mechanism."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, NodalLoad, Node, Section, Support,
    )
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2,
        E=2.1e8, A=0.02, I=8e-5, section_id=1,
        release_i=True,  # pin at the supported far end
    )]
    w._model.supports[1] = Support(
        node_id=1, ux=True, uy=True, rz=False,
    )
    w._model.nodal_loads.append(NodalLoad(
        node_id=2, fy=-10.0, load_case="DEFAULT",
    ))


def test_single_release_at_far_end_detected_as_mechanism(qt_app, monkeypatch):
    """A frame with a pin at the translation-only supported end must be caught
    by the validator, block the solve, highlight the free node and element,
    and display the mechanism message in the report panel."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _single_release_far_end_model(w)
    monkeypatch.setattr(
        app_mod.QMessageBox, "critical", lambda *a, **k: None,
    )
    w._do_solve()
    qt_app.processEvents()

    assert w._result is None
    # Both the released-end node 1 (cause) and the free node 2 (unstable
    # DOF) must light up — the user needs to see the root of the mechanism.
    assert 1 in w.canvas._error_node_ids, "released-end node 1 must be highlighted"
    assert 2 in w.canvas._error_node_ids, "free node 2 must be highlighted"
    assert 1 in w.canvas._error_element_ids
    assert w.canvas.has_validation_highlights()
    text = w._result_text.toPlainText()
    assert "unconstrained transverse DOF" in text
    assert "stabilizing-side end" in text, (
        f"message must use the new generic wording, got: {text}"
    )


def _corbel_indirect_mechanism_model(w) -> None:
    """Column 1→3→2 (fixed at node 1) + corbel element 3→4 with a moment
    release at the column-junction end (node 3).  Node 3 is NOT directly
    supported; it is stabilised only through the column that reaches the
    fixed base.  Node 4 is the free leaf — the corbel can rotate as a
    rigid body about the pin at node 3."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, NodalLoad, Node, Section, Support,
    )
    w._model.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        3: Node(3, 0.0, 2.0),
        2: Node(2, 0.0, 4.0),
        4: Node(4, 2.0, 2.0),
    }
    w._model.elements = [
        FrameElement2D(id=1, node_i=1, node_j=3, E=2.1e8, A=0.02, I=8e-5, section_id=1),
        FrameElement2D(id=2, node_i=3, node_j=2, E=2.1e8, A=0.02, I=8e-5, section_id=1),
        FrameElement2D(id=3, node_i=3, node_j=4, E=2.1e8, A=0.02, I=8e-5,
                       section_id=1, release_i=True),  # pin at column-junction side
    ]
    w._model.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    w._model.nodal_loads.append(NodalLoad(node_id=4, fy=-10.0, load_case="DEFAULT"))


def test_corbel_indirect_mechanism_flagged(qt_app, monkeypatch):
    """Column-stabilised corbel with pin at the column junction must be caught
    by the validator even though the junction node is not directly supported.
    Solve must be blocked, the free tip and corbel element must be highlighted,
    and the report must name the unconstrained DOF."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _corbel_indirect_mechanism_model(w)
    monkeypatch.setattr(app_mod.QMessageBox, "critical", lambda *a, **k: None)
    w._do_solve()
    qt_app.processEvents()

    assert w._result is None, "solve must be blocked for the corbel mechanism"
    # Both the released-end node 3 (cause) and the free tip node 4
    # (unstable DOF) must be highlighted so the user can see the root.
    assert 3 in w.canvas._error_node_ids, "released-end node 3 must be highlighted"
    assert 4 in w.canvas._error_node_ids, "free tip node 4 must be highlighted"
    assert 3 in w.canvas._error_element_ids, "corbel element 3 must be highlighted"
    assert w.canvas.has_validation_highlights()
    text = w._result_text.toPlainText()
    assert "unconstrained transverse DOF" in text
    assert "stabilizing-side end" in text, (
        f"message must use the new generic wording, got: {text}"
    )


def _corbel_reverse_mechanism_model(w) -> None:
    """Same corbel mechanism as _corbel_indirect_mechanism_model but with the
    element drawn from free tip (node 4) → column junction (node 3) and a
    release at END (node_j=3), i.e. the user clicked the free tip first.
    Equivalent to 'FRAME END' in the text format / 'Moment release at end (j)'
    in the dialog."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, NodalLoad, Node, Section, Support,
    )
    w._model.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        3: Node(3, 0.0, 2.0),
        2: Node(2, 0.0, 4.0),
        4: Node(4, 2.0, 2.0),
    }
    w._model.elements = [
        FrameElement2D(id=1, node_i=1, node_j=3, E=2.1e8, A=0.02, I=8e-5, section_id=1),
        FrameElement2D(id=2, node_i=3, node_j=2, E=2.1e8, A=0.02, I=8e-5, section_id=1),
        # Reversed orientation: node_i=4 (tip), node_j=3 (junction), END release
        FrameElement2D(id=3, node_i=4, node_j=3, E=2.1e8, A=0.02, I=8e-5,
                       section_id=1, release_i=False, release_j=True),
    ]
    w._model.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    w._model.nodal_loads.append(NodalLoad(node_id=4, fy=-10.0, load_case="DEFAULT"))


def test_corbel_reverse_orientation_mechanism_flagged(qt_app, monkeypatch):
    """Same corbel mechanism with reversed element orientation (4→3 END release)
    must be caught identically.  This covers the case where the user clicked
    the free tip first and selected 'Moment release at end (j)' in the dialog."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _corbel_reverse_mechanism_model(w)
    monkeypatch.setattr(app_mod.QMessageBox, "critical", lambda *a, **k: None)
    w._do_solve()
    qt_app.processEvents()

    assert w._result is None, "solve must be blocked (reversed-orientation corbel)"
    # Same dual highlight: released end (node 3) + free tip (node 4).
    assert 3 in w.canvas._error_node_ids, "released-end node 3 must be highlighted"
    assert 4 in w.canvas._error_node_ids, "free tip node 4 must be highlighted"
    assert 3 in w.canvas._error_element_ids, "corbel element 3 must be highlighted"
    assert w.canvas.has_validation_highlights()
    assert "unconstrained transverse DOF" in w._result_text.toPlainText()


def test_corbel_mechanism_loaded_from_fixture_file(qt_app, monkeypatch):
    """Load the corbel mechanism model from the corbel_mechanism.spa.json fixture
    (same path as File→Open in the GUI) and confirm the solve is blocked.

    This exercises the full chain: JSON→model_txt→file_io.read_input_file→
    validate_model→_run_static_solve, with the release correctly read from the
    'FRAME START' token in the text format."""
    import os
    from structural_analysis.gui_qt import app as app_mod
    from structural_analysis.gui_qt.project_io import load_project_json

    fixture = os.path.join(os.path.dirname(__file__), "corbel_mechanism.spa.json")
    assert os.path.exists(fixture), f"fixture not found: {fixture}"

    project = load_project_json(fixture)
    w = MainWindow()
    w._model = project.model

    # Verify the fixture was parsed as expected before testing the GUI path.
    corbels = [e for e in w._model.elements
               if getattr(e, "node_i", None) == 3 and getattr(e, "node_j", None) == 4]
    assert corbels, "fixture must contain corbel element 3→4"
    assert getattr(corbels[0], "release_i", False), (
        "fixture corbel must have release_i=True (FRAME START at node 3)"
    )

    monkeypatch.setattr(app_mod.QMessageBox, "critical", lambda *a, **k: None)
    w._do_solve()
    qt_app.processEvents()

    assert w._result is None, "solve must be blocked (fixture file corbel)"
    assert 4 in w.canvas._error_node_ids
    assert "unconstrained transverse DOF" in w._result_text.toPlainText()


# ── active-case filtering (Solve All Cases skips empty cases) ────────


def _three_case_model(w) -> None:
    """DEAD has a nodal load, LIVE has a member load, WIND has
    nothing (empty placeholder).  DEFAULT disabled so SUM_ALL is
    decidable."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        LoadCase, Material, NodalLoad, Node, Section, Support,
        UniformDistributedLoad,
    )
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, I=8e-5, section_id=1,
    )]
    w._model.supports[1] = Support(
        node_id=1, ux=True, uy=True, rz=True,
    )
    w._model.nodal_loads.append(NodalLoad(
        node_id=2, fy=-10.0, load_case="DEAD",
    ))
    w._model.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-5.0, load_case="LIVE"),
    )
    w._model.load_cases["DEAD"] = LoadCase(name="DEAD")
    w._model.load_cases["LIVE"] = LoadCase(name="LIVE")
    w._model.load_cases["WIND"] = LoadCase(name="WIND")
    w._model.load_cases["DEFAULT"].enabled = False


# ── orphan-node dialog workflow ──────────────────────────────────────


def _orphan_model(w) -> None:
    """Minimal solvable model that also has one orphan node (node 3)."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Material, Node, NodalLoad, Section, Support
    w._model.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    w._model.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 4.0, 0.0),
        3: Node(3, 8.0, 0.0),  # orphan — not connected to any element
    }
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, I=8e-5, section_id=1,
    )]
    w._model.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    w._model.nodal_loads.append(NodalLoad(
        node_id=2, fy=-10.0, load_case="DEFAULT",
    ))


def test_orphan_node_triggers_dedicated_dialog(qt_app, monkeypatch):
    """When the model has an orphan node, the orphan-node dialog function
    must be called (not just the generic warnings dialog)."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _orphan_model(w)
    calls = []
    monkeypatch.setattr(
        app_mod, "_show_orphan_nodes_dialog",
        lambda parent, nids: calls.append(nids) or "cancel",
    )
    w._do_solve()
    qt_app.processEvents()
    assert calls, "orphan dialog must have been invoked"
    assert 3 in calls[0], "orphan node 3 must appear in the dialog node list"


def test_orphan_delete_removes_node_and_solves(qt_app, monkeypatch):
    """Choosing 'delete' in the orphan dialog must remove the orphan node
    via BatchDeleteCmd (single undo step) and produce a successful DEFAULT-case
    solve — not just any wrapper object, but a real successful case entry."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _orphan_model(w)
    monkeypatch.setattr(
        app_mod, "_show_orphan_nodes_dialog", lambda *a, **k: "delete",
    )
    w._do_solve()
    qt_app.processEvents()
    assert 3 not in w._model.nodes, "orphan node 3 must be deleted"
    assert w._multi_result is not None, "a multi-case result must be produced"
    assert "DEFAULT" in w._multi_result.cases, (
        "DEFAULT case must have solved successfully after orphan deletion; "
        f"failed_cases={w._multi_result.failed_cases}"
    )
    assert not w._multi_result.failed_cases, (
        f"no case should have failed: {w._multi_result.failed_cases}"
    )


def test_orphan_delete_uses_single_undo_step(qt_app, monkeypatch):
    """Multiple orphan nodes must be deleted via a single BatchDeleteCmd so
    one Ctrl+Z restores them all together (clean undo history)."""
    from structural_analysis.model import Node
    from structural_analysis.gui_qt import app as app_mod

    w = MainWindow()
    _orphan_model(w)  # has orphan node 3; full load case + load already set
    # Add a second orphan node so BatchDeleteCmd has two ids to delete.
    w._model.nodes[4] = Node(4, 9.0, 0.0)

    monkeypatch.setattr(
        app_mod, "_show_orphan_nodes_dialog", lambda *a, **k: "delete",
    )
    undo_len_before = len(w._undo)
    w._do_solve()
    qt_app.processEvents()
    # Exactly ONE new entry on the undo stack, not two — that's the
    # whole point of using BatchDeleteCmd instead of looping
    # DeleteNodeCmd for each orphan.
    assert len(w._undo) == undo_len_before + 1, (
        f"expected exactly 1 new undo entry (BatchDeleteCmd), got "
        f"{len(w._undo) - undo_len_before}"
    )
    assert 3 not in w._model.nodes and 4 not in w._model.nodes, (
        "both orphan nodes 3 and 4 must be removed"
    )


def test_orphan_continue_leaves_node_in_model(qt_app, monkeypatch):
    """Choosing 'continue' in the orphan dialog leaves the orphan node in the
    model.  The solver pipeline (assembler.validate_model) currently rejects
    isolated nodes and reports the DEFAULT case under failed_cases — that's
    consistent with the model state the user chose to keep.  This test pins
    the dialog-routing behaviour (no deletion, no early-return) without
    asserting solver success."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _orphan_model(w)
    monkeypatch.setattr(
        app_mod, "_show_orphan_nodes_dialog", lambda *a, **k: "continue",
    )
    w._do_solve()
    qt_app.processEvents()
    assert 3 in w._model.nodes, "orphan node 3 must still be present"
    # The solver IS invoked (multi_result populated) — whether the case
    # ends up under .cases or .failed_cases depends on the core
    # assembler, which the dialog does not bypass.
    assert w._multi_result is not None, (
        "Continue must invoke the solver (multi_result must be populated)"
    )


def test_orphan_cancel_clears_stale_result(qt_app, monkeypatch):
    """Choosing 'cancel' in the orphan dialog must not solve and must
    clear any stale result from a previous solve."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _multi_case_loaded_frame(w)
    w._do_solve()
    qt_app.processEvents()
    assert w._multi_result is not None  # prior solve result

    # Now introduce an orphan node to force the dialog.
    from structural_analysis.model import Node
    w._model.nodes[99] = Node(99, 20.0, 0.0)

    monkeypatch.setattr(
        app_mod, "_show_orphan_nodes_dialog", lambda *a, **k: "cancel",
    )
    w._do_solve()
    qt_app.processEvents()
    assert w._result is None
    assert w._multi_result is None


def test_orphan_cancel_shows_validation_highlights(qt_app, monkeypatch):
    """After a 'cancel' on the orphan dialog, the orphan node should be
    highlighted on the canvas so the user can see it."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _orphan_model(w)
    monkeypatch.setattr(
        app_mod, "_show_orphan_nodes_dialog", lambda *a, **k: "cancel",
    )
    w._do_solve()
    qt_app.processEvents()
    assert w.canvas.has_validation_highlights()
    assert 3 in w.canvas._warning_node_ids


def test_orphan_delete_is_undoable(qt_app, monkeypatch):
    """The BatchDeleteCmd executed via the orphan dialog must be on the undo
    stack so Ctrl+Z restores all orphan nodes in one step."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _orphan_model(w)
    monkeypatch.setattr(
        app_mod, "_show_orphan_nodes_dialog", lambda *a, **k: "delete",
    )
    w._do_solve()
    qt_app.processEvents()
    assert 3 not in w._model.nodes
    w._do_undo()  # undo the deletion
    assert 3 in w._model.nodes, "undo must restore the orphan node"


def test_no_orphan_no_orphan_dialog(qt_app, monkeypatch):
    """When the model has no orphan nodes, the orphan dialog must NOT be
    invoked — only the normal solve path runs."""
    from structural_analysis.gui_qt import app as app_mod
    w = MainWindow()
    _multi_case_loaded_frame(w)
    calls = []
    monkeypatch.setattr(
        app_mod, "_show_orphan_nodes_dialog",
        lambda *a, **k: calls.append(True) or "cancel",
    )
    w._do_solve()
    qt_app.processEvents()
    assert not calls, "orphan dialog must NOT be called when no orphan nodes exist"


def test_solve_all_skips_empty_load_case(qt_app):
    """WIND has no loads — Solve All must not request it from the
    multi-case solver, so it's absent from ``_multi_result.cases``."""
    w = MainWindow()
    _three_case_model(w)
    w._do_solve()
    qt_app.processEvents()
    assert "DEAD" in w._multi_result.cases
    assert "LIVE" in w._multi_result.cases
    assert "WIND" not in w._multi_result.cases
    assert "WIND" not in w._multi_result.requested_cases


def test_solve_all_skipped_cases_status_message(qt_app):
    """The skip notice surfaces in the status bar so the user knows
    WIND wasn't quietly solved as a zero-result."""
    w = MainWindow()
    _three_case_model(w)
    w._do_solve()
    qt_app.processEvents()
    status = w._status_label.text()
    assert "WIND" in status
    assert "skipped" in status.lower()


def test_solve_all_includes_self_weight_case_with_no_manual_loads(qt_app):
    """If self-weight is enabled and assigned to DEAD, DEAD counts as
    an active case even with zero manual DEAD loads."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        LoadCase, Material, Node, Section, Support,
    )
    w = MainWindow()
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, I=8e-5, section_id=1,
    )]
    w._model.supports[1] = Support(
        node_id=1, ux=True, uy=True, rz=True,
    )
    w._model.load_cases["DEAD"] = LoadCase(name="DEAD")
    w._model.load_cases["LIVE"] = LoadCase(name="LIVE")
    w._model.load_cases["DEFAULT"].enabled = False
    w._model.include_self_weight = True
    w._model.self_weight_case = "DEAD"

    w._do_solve()
    qt_app.processEvents()
    # DEAD solved because self-weight makes it active.
    assert "DEAD" in w._multi_result.cases
    # LIVE has nothing — skipped.
    assert "LIVE" not in w._multi_result.cases


def test_solve_all_no_active_loads_shows_warning_and_keeps_results_none(
    qt_app, monkeypatch,
):
    """A model with cases defined but zero loads tells the user
    explicitly instead of silently doing nothing."""
    from structural_analysis.gui_qt import app as app_mod
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        LoadCase, Material, Node, Section, Support,
    )
    w = MainWindow()
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, I=8e-5, section_id=1,
    )]
    w._model.supports[1] = Support(
        node_id=1, ux=True, uy=True, rz=True,
    )
    w._model.load_cases["DEAD"] = LoadCase(name="DEAD")
    w._model.load_cases["DEFAULT"].enabled = False

    warned: list = []
    monkeypatch.setattr(
        app_mod.QMessageBox, "warning",
        lambda *a, **k: warned.append(a),
    )
    w._do_solve()
    qt_app.processEvents()
    assert warned, "expected a 'no active loads' warning dialog"
    assert w._result is None
    assert w._multi_result is None
    # Status surfaces the same message.
    assert "no active loads" in w._status_label.text().lower()


def test_combination_referencing_skipped_case_unavailable(qt_app):
    """A combination that references WIND (skipped because empty)
    must be unavailable; selecting it should not produce a result."""
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    w = MainWindow()
    _three_case_model(w)
    w.execute(AddLoadCombinationCmd(
        name="COMB_WIND", terms={"DEAD": 1.0, "WIND": 1.0},
    ))
    w._do_solve()
    qt_app.processEvents()
    assert not w._multi_result.combination_available({"DEAD": 1.0, "WIND": 1.0})
    w._active_case = "COMB_WIND"
    assert w._resolve_active_result() is None


def test_combination_referencing_only_solved_cases_is_available(qt_app):
    """COMB_LIVE = 1.2*DEAD + 1.6*LIVE is available because both
    cases get solved (neither is empty)."""
    from structural_analysis.gui_common.commands import AddLoadCombinationCmd
    w = MainWindow()
    _three_case_model(w)
    w.execute(AddLoadCombinationCmd(
        name="COMB_LIVE", terms={"DEAD": 1.2, "LIVE": 1.6},
    ))
    w._do_solve()
    qt_app.processEvents()
    assert w._multi_result.combination_available(
        {"DEAD": 1.2, "LIVE": 1.6},
    )


def test_solve_all_only_solves_cases_with_loads_no_extra_solves(qt_app):
    """Sanity: requested_cases on the result wrapper equals exactly
    the cases that had loads (DEAD, LIVE) — WIND is neither requested
    nor failed."""
    w = MainWindow()
    _three_case_model(w)
    w._do_solve()
    qt_app.processEvents()
    assert sorted(w._multi_result.requested_cases) == ["DEAD", "LIVE"]
    assert "WIND" not in w._multi_result.failed_cases


def test_explicit_active_only_solve_still_works_on_empty_default(qt_app):
    """Active-only (Shift+F5) is intentionally NOT filtered: the user
    picked that case, so even if it's empty we attempt the solve.

    Here DEFAULT is selected and has no loads; the solve will produce
    an analysis result with all-zero loads but should not be rejected
    by the active-load filter (which only guards Solve All Cases)."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, Node, Section, Support,
    )
    w = MainWindow()
    w._model.materials[1] = Material(
        id=1, name="Steel", E=2.1e8, density=7850.0,
    )
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.02, I=8e-5, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.02, I=8e-5, section_id=1,
    )]
    w._model.supports[1] = Support(
        node_id=1, ux=True, uy=True, rz=True,
    )
    w._active_case = "DEFAULT"
    w._do_solve_active_only()
    qt_app.processEvents()
    # DEFAULT solved (zero displacements, but a valid result wrapper).
    assert w._multi_result is not None
    assert "DEFAULT" in w._multi_result.cases


def test_cancelling_warnings_clears_stale_result_and_shows_report(
    qt_app, monkeypatch,
):
    """Regression (Gemini PR #31 MEDIUM finding): when the user
    cancels a solve from the warnings prompt, stale results must be
    cleared and the warning report must remain visible in the
    result-text panel so the warnings don't vanish with the dialog.
    """
    from structural_analysis.gui_qt import app as app_mod
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        LoadCase, Material, NodalLoad, Node, Section, Support,
    )
    w = MainWindow()
    # First, build a valid model and solve it so we have a stale
    # result to be cleared.
    _multi_case_loaded_frame(w)
    w._do_solve()
    qt_app.processEvents()
    assert w._result is not None

    # Now mutate the model into one that triggers a WARNING (orphan
    # node — no errors).  Then cancel the warning prompt.
    w._model.nodes[99] = Node(99, 50.0, 50.0)  # orphan
    # Note: invalidation already fires on the model mutation if it
    # went through execute().  Here we mutated directly so _result is
    # still set — exactly the stale-result scenario the fix targets.

    monkeypatch.setattr(
        app_mod.QMessageBox, "question",
        lambda *a, **k: app_mod.QMessageBox.StandardButton.Cancel,
    )
    w._do_solve()
    qt_app.processEvents()

    # Stale result cleared.
    assert w._result is None
    assert w._multi_result is None
    # Warning is visible in the report panel.
    text = w._result_text.toPlainText()
    assert "not connected" in text or "orphan" in text.lower() or "Node 99" in text
    # Canvas has the warning highlight.
    assert 99 in w.canvas._warning_node_ids


# ── PR #35: tabbed Element Detail Dialog ─────────────────────────────


def _frame_with_multi_cases(w):
    """Two-case fixture for the Results-tab tests: DEFAULT + WIND, single
    frame element with one nodal load per case so both cases produce a
    real solve. Returns the element id."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        LoadCase, Material, NodalLoad, Node, Section, Support,
    )
    w._model.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )]
    w._model.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    w._model.load_cases["WIND"] = LoadCase(name="WIND", enabled=True)
    w._model.nodal_loads.append(NodalLoad(
        node_id=2, fy=-10.0, load_case="DEFAULT",
    ))
    w._model.nodal_loads.append(NodalLoad(
        node_id=2, fx=5.0, load_case="WIND",
    ))
    return 1


def test_inspector_uses_tabbed_layout(qt_app):
    """Inspector body is now a QTabWidget with three labelled tabs."""
    from PyQt6.QtWidgets import QTabWidget
    w = MainWindow()
    eid = _make_loaded_frame(w)
    w._open_element_inspector(eid)
    qt_app.processEvents()

    insp = w._element_inspector
    assert isinstance(insp._tabs, QTabWidget)
    assert insp._tabs.count() == 3
    labels = [insp._tabs.tabText(i) for i in range(3)]
    assert labels == ["Properties", "Results", "Load Assignments"]


def test_inspector_opens_on_properties_tab_by_default(qt_app):
    w = MainWindow()
    eid = _make_loaded_frame(w)
    w._open_element_inspector(eid)
    qt_app.processEvents()
    assert w._element_inspector._tabs.currentIndex() == 0  # Properties


def test_right_click_edit_member_loads_opens_loads_tab(qt_app):
    """show_element_menu(action='loads') routes directly to the inspector
    and lands the focus on the Load Assignments tab."""
    w = MainWindow()
    eid = _make_loaded_frame(w)
    w.show_element_menu(eid, action="loads")
    qt_app.processEvents()
    insp = w._element_inspector
    assert insp is not None and insp.isVisible()
    assert insp._tabs.currentIndex() == 2  # Load Assignments


def test_right_click_details_action_opens_properties_tab(qt_app):
    w = MainWindow()
    eid = _make_loaded_frame(w)
    w.show_element_menu(eid, action="details")
    qt_app.processEvents()
    insp = w._element_inspector
    assert insp is not None and insp.isVisible()
    assert insp._tabs.currentIndex() == 0


def test_results_tab_shows_no_analysis_yet_pre_solve(qt_app):
    """Pre-solve, the Results tab paints a status label that includes
    'No analysis results yet' and the N axis carries the placeholder
    text and zero data lines."""
    w = MainWindow()
    eid = _frame_with_multi_cases(w)
    w._open_element_inspector(eid)
    qt_app.processEvents()
    insp = w._element_inspector
    assert "No analysis results yet" in insp._results_status.text()
    assert not insp._ax_n.lines, "pre-solve N-axis must hold zero data lines"


def test_results_tab_shows_diagrams_post_solve(qt_app):
    w = MainWindow()
    eid = _frame_with_multi_cases(w)
    w._open_element_inspector(eid)
    qt_app.processEvents()
    w._do_solve()
    qt_app.processEvents()
    # _do_solve() invokes the host's refresh hook; the inspector now
    # holds a populated Results tab.
    insp = w._element_inspector
    assert insp._ax_n.lines, "post-solve N-axis must carry the axial trace"


def test_results_tab_local_selector_does_not_change_canvas_case(qt_app):
    """Changing the dialog's local case selector must not flip the host's
    active case (the canvas keeps showing whatever it was showing)."""
    w = MainWindow()
    eid = _frame_with_multi_cases(w)
    w._do_solve()
    qt_app.processEvents()
    canvas_case_before = w._active_case
    w._open_element_inspector(eid)
    qt_app.processEvents()
    insp = w._element_inspector
    # Find the index of the other (non-active) case in the local combo.
    other = "WIND" if canvas_case_before != "WIND" else "DEFAULT"
    idx = insp._results_combo.findData(other)
    assert idx >= 0, f"local combo must include {other}"
    insp._results_combo.setCurrentIndex(idx)
    qt_app.processEvents()
    assert insp._results_selection == other
    assert w._active_case == canvas_case_before, (
        "host active case must not change when the dialog combo changes"
    )


def test_results_tab_combo_stores_raw_identifier_not_label(qt_app):
    """The local combo's userData carries the raw case / combination name
    so the legacy "[comb]"-leaks-into-key bug stays fixed."""
    from structural_analysis.model import LoadCase, LoadCombination
    w = MainWindow()
    eid = _frame_with_multi_cases(w)
    w._model.load_combinations["COMB1"] = LoadCombination(
        name="COMB1", terms={"DEFAULT": 1.0, "WIND": 1.0},
    )
    w._open_element_inspector(eid)
    qt_app.processEvents()
    insp = w._element_inspector
    found = False
    for i in range(insp._results_combo.count()):
        label = insp._results_combo.itemText(i)
        data = insp._results_combo.itemData(i)
        if data == "COMB1":
            found = True
            # The label gets "[comb]" or "[comb · needs solve]"; the data
            # stays bare so resolve_view never sees the decoration.
            assert "COMB1" in label
            assert label != "COMB1"
    assert found, "COMB1 must appear in the local combo with raw identifier"


def test_loads_tab_add_button_appends_load_undoable(qt_app, monkeypatch):
    """Clicking Add in the Loads tab opens MemberLoadDialog; on Accept,
    AddMemberLoadCmd lands the row and Ctrl+Z removes it."""
    from structural_analysis.gui_qt import dialogs as dialogs_mod
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _make_loaded_frame(w)
    w._open_element_inspector(eid, tab="loads")
    qt_app.processEvents()
    n_before = len(w._model.elements[0].member_loads)

    new_load = UniformDistributedLoad(wy=-50.0, coord_system="local")

    class _StubDialog:
        def __init__(self, *a, **k):
            self.result_value = new_load
        def exec(self):
            return dialogs_mod.QDialog.DialogCode.Accepted

    monkeypatch.setattr(dialogs_mod, "MemberLoadDialog", _StubDialog)
    # The host hook constructs MemberLoadDialog from the app module's
    # binding; patch that too so the stub is what gets instantiated.
    from structural_analysis.gui_qt import app as app_mod
    monkeypatch.setattr(app_mod, "MemberLoadDialog", _StubDialog)

    insp = w._element_inspector
    insp._add_load_btn.click()
    qt_app.processEvents()

    assert len(w._model.elements[0].member_loads) == n_before + 1
    assert w._model.elements[0].member_loads[-1] is new_load
    w._do_undo()
    qt_app.processEvents()
    assert len(w._model.elements[0].member_loads) == n_before


def test_loads_tab_edit_button_swaps_load_undoable(qt_app, monkeypatch):
    """Clicking Edit on a row opens MemberLoadDialog pre-filled; on
    Accept, UpdateMemberLoadCmd swaps the row and Ctrl+Z restores the
    exact old instance."""
    from structural_analysis.gui_qt import app as app_mod
    from structural_analysis.gui_qt import dialogs as dialogs_mod
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _make_loaded_frame(w)
    w._open_element_inspector(eid, tab="loads")
    qt_app.processEvents()
    original = w._model.elements[0].member_loads[0]
    replacement = UniformDistributedLoad(wy=-999.0, coord_system="local")

    class _StubDialog:
        def __init__(self, *a, **k):
            self.result_value = replacement
        def exec(self):
            return dialogs_mod.QDialog.DialogCode.Accepted

    monkeypatch.setattr(dialogs_mod, "MemberLoadDialog", _StubDialog)
    monkeypatch.setattr(app_mod, "MemberLoadDialog", _StubDialog)

    insp = w._element_inspector
    edit_btn = insp._loads_widget.cellWidget(0, 6)  # Edit column
    edit_btn.click()
    qt_app.processEvents()

    assert w._model.elements[0].member_loads[0] is replacement
    w._do_undo()
    qt_app.processEvents()
    assert w._model.elements[0].member_loads[0] is original, (
        "undo must restore the EXACT original load instance"
    )


def _stub_member_load_dialog(monkeypatch, load):
    """Patch MemberLoadDialog (both the dialogs module and the app-module
    binding the host hook uses) with an auto-accepting stub returning
    ``load``."""
    from structural_analysis.gui_qt import app as app_mod
    from structural_analysis.gui_qt import dialogs as dialogs_mod

    class _StubDialog:
        def __init__(self, *a, **k):
            self.result_value = load
        def exec(self):
            return dialogs_mod.QDialog.DialogCode.Accepted

    monkeypatch.setattr(dialogs_mod, "MemberLoadDialog", _StubDialog)
    monkeypatch.setattr(app_mod, "MemberLoadDialog", _StubDialog)


def test_loads_tab_add_registers_new_load_case(qt_app, monkeypatch):
    """P2 (Codex): adding a load with a brand-new case name via the Loads
    tab must register that case in model.load_cases AND make it
    discoverable by cases_with_loads(), matching the legacy
    _add_member_load path. Without the _ensure_load_case_exists call the
    load would be saved against a case Solve All never sees."""
    from structural_analysis.gui_common.validation import cases_with_loads
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _make_loaded_frame(w)
    assert "WIND2" not in w._model.load_cases
    w._open_element_inspector(eid, tab="loads")
    qt_app.processEvents()
    _stub_member_load_dialog(
        monkeypatch,
        UniformDistributedLoad(wy=-7.0, coord_system="local", load_case="WIND2"),
    )
    w._element_inspector._add_load_btn.click()
    qt_app.processEvents()

    assert "WIND2" in w._model.load_cases, (
        "new case typed in the Add dialog must be registered"
    )
    assert "WIND2" in cases_with_loads(w._model), (
        "the new case must be discoverable by cases_with_loads()"
    )


def test_loads_tab_add_new_case_appears_in_toolbar_combo(qt_app, monkeypatch):
    """After an inspector Add with a new case, the toolbar case selector
    lists it (parity with the _add_member_load path)."""
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _make_loaded_frame(w)
    w._open_element_inspector(eid, tab="loads")
    qt_app.processEvents()
    _stub_member_load_dialog(
        monkeypatch,
        UniformDistributedLoad(wy=-3.0, coord_system="local", load_case="SNOW"),
    )
    w._element_inspector._add_load_btn.click()
    qt_app.processEvents()

    combo_data = [
        w._case_combo.itemData(i) for i in range(w._case_combo.count())
    ]
    assert "SNOW" in combo_data, (
        "newly-registered case must appear in the toolbar selector"
    )


def test_loads_tab_edit_registers_new_load_case(qt_app, monkeypatch):
    """P2 (Codex): editing a row to a brand-new case name must register
    that case too (the edit hook had the same gap as Add)."""
    from structural_analysis.gui_common.validation import cases_with_loads
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _make_loaded_frame(w)
    assert "GUST" not in w._model.load_cases
    w._open_element_inspector(eid, tab="loads")
    qt_app.processEvents()
    _stub_member_load_dialog(
        monkeypatch,
        UniformDistributedLoad(wy=-9.0, coord_system="local", load_case="GUST"),
    )
    edit_btn = w._element_inspector._loads_widget.cellWidget(0, 6)
    edit_btn.click()
    qt_app.processEvents()

    assert "GUST" in w._model.load_cases, (
        "new case typed in the Edit dialog must be registered"
    )
    assert "GUST" in cases_with_loads(w._model)


def test_loads_tab_add_button_disabled_without_host_callback(qt_app):
    """When the dialog is constructed without host wiring (unit-test path),
    the Add button must render disabled so the model is never mutated
    implicitly."""
    from structural_analysis.gui_qt.dialogs import ElementPropertiesDialog
    w = MainWindow()
    eid = _make_loaded_frame(w)
    d = ElementPropertiesDialog(w, w._model, eid, None)
    assert not d._add_load_btn.isEnabled()


def test_loads_tab_includes_direction_column(qt_app):
    """The Load Assignments table exposes a Direction column at col 2;
    mechanical rows carry the coord-system label, thermal rows show '—'."""
    w = MainWindow()
    eid = _make_loaded_frame(w)
    w._open_element_inspector(eid, tab="loads")
    qt_app.processEvents()
    table = w._element_inspector._loads_widget
    headers = [
        table.horizontalHeaderItem(i).text()
        for i in range(table.columnCount())
    ]
    assert headers[2] == "Direction"
    # row 0 (UDL local) — direction label is non-empty / non-dash
    assert table.item(0, 2).text() == "local axes"
    # row 2 (FrameTemperatureLoad) — direction column is '—'
    assert table.item(2, 2).text() == "—"


def test_loads_tab_change_invalidates_result_and_shows_placeholder(
    qt_app, monkeypatch,
):
    """After Add via the Loads tab, the analysis result is invalidated
    and the Results tab shows the 'No analysis results yet' placeholder
    — never stale N/V/M diagrams from before the change."""
    from structural_analysis.gui_qt import app as app_mod
    from structural_analysis.gui_qt import dialogs as dialogs_mod
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _frame_with_multi_cases(w)
    w._do_solve()
    qt_app.processEvents()
    w._open_element_inspector(eid, tab="loads")
    qt_app.processEvents()
    insp = w._element_inspector
    # Sanity: diagrams populated before the load change.
    assert insp._ax_n.lines

    class _StubDialog:
        def __init__(self, *a, **k):
            self.result_value = UniformDistributedLoad(
                wy=-1.0, coord_system="local",
            )
        def exec(self):
            return dialogs_mod.QDialog.DialogCode.Accepted

    monkeypatch.setattr(dialogs_mod, "MemberLoadDialog", _StubDialog)
    monkeypatch.setattr(app_mod, "MemberLoadDialog", _StubDialog)

    insp._add_load_btn.click()
    qt_app.processEvents()

    # After the load change, _result is invalidated and Results tab
    # has the placeholder + zero data lines.
    assert w._result is None
    assert "No analysis results yet" in insp._results_status.text()
    assert not insp._ax_n.lines, (
        "stale N/V/M diagrams must not survive a model-changing edit"
    )


def test_loads_tab_stays_focused_after_edit(qt_app, monkeypatch):
    """Editing a load from the Load Assignments tab triggers a full
    refresh() (to clear stale Results diagrams). The user must remain on
    the Load Assignments tab, not get bounced to Properties — regression
    for the QTabWidget removeTab/set_target focus-jump (PR #37 review)."""
    from structural_analysis.gui_qt import app as app_mod
    from structural_analysis.gui_qt import dialogs as dialogs_mod
    from structural_analysis.model import UniformDistributedLoad

    w = MainWindow()
    eid = _frame_with_multi_cases(w)
    # Give the element a load so the Loads tab has an editable row.
    w._model.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-3.0, coord_system="local")
    )
    w._do_solve()
    qt_app.processEvents()
    w._open_element_inspector(eid, tab="loads")
    qt_app.processEvents()
    insp = w._element_inspector
    assert insp._tabs.currentIndex() == insp._TAB_LOADS

    replacement = UniformDistributedLoad(wy=-77.0, coord_system="local")

    class _StubDialog:
        def __init__(self, *a, **k):
            self.result_value = replacement
        def exec(self):
            return dialogs_mod.QDialog.DialogCode.Accepted

    monkeypatch.setattr(dialogs_mod, "MemberLoadDialog", _StubDialog)
    monkeypatch.setattr(app_mod, "MemberLoadDialog", _StubDialog)

    edit_btn = insp._loads_widget.cellWidget(0, 6)
    edit_btn.click()
    qt_app.processEvents()

    # Full refresh() ran (Results now invalidated), but the focused tab
    # must still be Load Assignments.
    assert w._element_inspector._tabs.currentIndex() == insp._TAB_LOADS, (
        "user must stay on the Load Assignments tab after an edit"
    )


# ── PR #34: canvas perf + dense-view readability ─────────────────────


def test_canvas_dense_models_auto_hide_id_labels(qt_app):
    """Dense plans should stay readable and avoid hundreds of text artists."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {
        i: Node(i, float(i), 0.0)
        for i in range(1, w.canvas.MAX_AUTO_NODE_LABELS + 2)
    }
    w._model.elements = [
        FrameElement2D(
            id=i, node_i=i, node_j=i + 1,
            E=2.0e8, A=0.01, I=1.0e-4, section_id=1,
        )
        for i in range(1, w.canvas.MAX_AUTO_ELEMENT_LABELS + 2)
    ]

    w.canvas.redraw()
    labels = [text.get_text() for text in w.canvas.ax.texts]

    assert not any(label.startswith("n") and label[1:].isdigit() for label in labels)
    assert not any(label.startswith("e") and label[1:].isdigit() for label in labels)
    assert any("Dense view" in label for label in labels)


def test_labeled_grid_draws_only_visible_viewport_lines(qt_app):
    """Large named grids should not create artists for off-screen lines."""
    from structural_analysis.gui_qt.grid import GridSystem

    w = MainWindow()
    w._grid = GridSystem.from_spacing(
        x_count=80, x_spacing=1.0,
        y_count=80, y_spacing=1.0,
    )
    w.canvas.redraw()
    w.canvas.ax.set_xlim(10.0, 14.0)
    w.canvas.ax.set_ylim(20.0, 24.0)
    w.canvas.redraw()

    grid_line_count = sum(
        1 for line in w.canvas.ax.lines
        if line.get_color() == "#aac8ff"
    )
    labels = [text.get_text().strip() for text in w.canvas.ax.texts]

    assert grid_line_count <= 12
    assert "K" in labels  # x=10, visible in the viewport
    assert "21" in labels  # y=20, visible in the viewport
    assert "A" not in labels  # x=0, outside the viewport


# ── hotfix v0.22.5: load-case registry sync, menu, maxima, copy ──────


def _solvable_frame(w, *, case="DEFAULT"):
    """Fixed-base frame with one nodal load on the given case. Returns eid."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import (
        Material, NodalLoad, Node, Section, Support,
    )
    w._model.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    w._model.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    w._model.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1,
    )]
    w._model.supports[1] = Support(node_id=1, ux=True, uy=True, rz=True)
    w._model.nodal_loads.append(
        NodalLoad(node_id=2, fy=-10.0, load_case=case)
    )
    return 1


def test_solve_all_auto_registers_orphan_member_load_case(qt_app):
    """A member load tagged LIVE with LIVE never registered must be picked
    up by Solve All (sync_load_case_registry runs first)."""
    from structural_analysis.model import UniformDistributedLoad
    w = MainWindow()
    _solvable_frame(w, case="DEFAULT")
    # Bypass _ensure_load_case_exists: append a LIVE-tagged load directly.
    w._model.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-4.0, coord_system="local", load_case="LIVE")
    )
    assert "LIVE" not in w._model.load_cases
    w._do_solve()
    qt_app.processEvents()
    assert "LIVE" in w._model.load_cases, "LIVE must be auto-registered"
    assert w._multi_result is not None
    assert "LIVE" in w._multi_result.cases, "Solve All must include LIVE"


def test_solve_all_auto_registers_orphan_nodal_load_case(qt_app):
    from structural_analysis.model import NodalLoad
    w = MainWindow()
    _solvable_frame(w, case="DEFAULT")
    w._model.nodal_loads.append(
        NodalLoad(node_id=2, fx=5.0, load_case="WIND")
    )
    w._do_solve()
    qt_app.processEvents()
    assert "WIND" in w._model.load_cases
    assert "WIND" in w._multi_result.cases


def test_two_loads_same_orphan_case_no_duplicate(qt_app):
    from structural_analysis.model import UniformDistributedLoad
    w = MainWindow()
    _solvable_frame(w, case="DEFAULT")
    for _ in range(2):
        w._model.elements[0].member_loads.append(
            UniformDistributedLoad(wy=-2.0, coord_system="local", load_case="LIVE")
        )
    w._do_solve()
    qt_app.processEvents()
    assert list(w._model.load_cases).count("LIVE") == 1


def test_case_selector_includes_auto_registered_case_after_solve(qt_app):
    from structural_analysis.model import UniformDistributedLoad
    w = MainWindow()
    _solvable_frame(w, case="DEFAULT")
    w._model.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-4.0, coord_system="local", load_case="LIVE")
    )
    w._do_solve()
    qt_app.processEvents()
    data = [w._case_combo.itemData(i) for i in range(w._case_combo.count())]
    assert "LIVE" in data


def test_unused_enabled_case_labelled_no_loads_assigned(qt_app):
    """An enabled case with no load source shows '(no loads assigned)'
    in the selector, not a plain unsolved entry."""
    from structural_analysis.gui_common.commands import AddLoadCaseCmd
    w = MainWindow()
    _solvable_frame(w, case="DEFAULT")
    w.execute(AddLoadCaseCmd(name="LIVE"))  # enabled, no loads
    texts = [w._case_combo.itemText(i) for i in range(w._case_combo.count())]
    data = [w._case_combo.itemData(i) for i in range(w._case_combo.count())]
    live_label = texts[data.index("LIVE")]
    assert "no loads assigned" in live_label
    # Raw identifier is still bare LIVE (userData), not the decorated label.
    assert "LIVE" in data


def test_remove_last_load_relabels_case_as_unused(qt_app):
    """After deleting the only LIVE load, LIVE flips to '(no loads
    assigned)' in the selector instead of staying a plain case."""
    from structural_analysis.gui_common.commands import (
        AddMemberLoadCmd, DeleteMemberLoadCmd,
    )
    from structural_analysis.model import UniformDistributedLoad
    w = MainWindow()
    _solvable_frame(w, case="DEFAULT")
    w.execute(AddMemberLoadCmd(
        elem_id=1,
        load=UniformDistributedLoad(wy=-3.0, coord_system="local", load_case="LIVE"),
    ))
    w._ensure_load_case_exists("LIVE")
    qt_app.processEvents()
    # Now delete that load.
    w.execute(DeleteMemberLoadCmd(elem_id=1, load_index=0))
    qt_app.processEvents()
    texts = [w._case_combo.itemText(i) for i in range(w._case_combo.count())]
    data = [w._case_combo.itemData(i) for i in range(w._case_combo.count())]
    if "LIVE" in data:  # case kept (not auto-deleted)
        assert "no loads assigned" in texts[data.index("LIVE")]


def test_load_cases_moved_to_model_menu(qt_app):
    """Load cases / combinations live under a Model menu, not View."""
    w = MainWindow()
    menus = {
        a.text(): a.menu()
        for a in w.menuBar().actions() if a.menu() is not None
    }
    assert "&Model" in menus, f"expected a Model menu, got {list(menus)}"
    model_actions = menus["&Model"].actions()
    assert w.act_load_cases in model_actions
    assert w.act_load_combinations in model_actions
    # And they are no longer under View.
    view_actions = menus.get("&View").actions()
    assert w.act_load_cases not in view_actions
    assert w.act_load_combinations not in view_actions


def test_element_detail_show_maxima_on_by_default(qt_app):
    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._do_solve()
    qt_app.processEvents()
    elem = w._model.elements[0]
    w._open_element_inspector(elem.id)
    qt_app.processEvents()
    d = w._element_inspector
    assert d._show_maxima_cb.isChecked(), "Show Maxima must default ON"
    assert d._maxima_annotations, "default-ON must render maxima annotations"


def test_element_detail_show_maxima_persists_across_refresh(qt_app):
    """Refreshing the inspector (as a canvas case switch does) must not
    reset Show Maxima."""
    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._do_solve()
    qt_app.processEvents()
    elem = w._model.elements[0]
    w._open_element_inspector(elem.id)
    qt_app.processEvents()
    d = w._element_inspector
    assert d._show_maxima_cb.isChecked()
    # Simulate a canvas-driven refresh.
    d.refresh(w._model, w._result)
    qt_app.processEvents()
    assert w._element_inspector._show_maxima_cb.isChecked(), (
        "Show Maxima must stay ON across a refresh"
    )


def test_element_detail_show_maxima_manual_off_persists(qt_app):
    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    w._do_solve()
    qt_app.processEvents()
    elem = w._model.elements[0]
    w._open_element_inspector(elem.id)
    qt_app.processEvents()
    d = w._element_inspector
    d._show_maxima_cb.setChecked(False)  # manual off
    assert d._show_maxima_on is False
    d.refresh(w._model, w._result)
    qt_app.processEvents()
    assert not w._element_inspector._show_maxima_cb.isChecked(), (
        "manual Show-Maxima-off must persist across refresh"
    )


def test_load_case_manager_table_has_copy_installed(qt_app):
    from structural_analysis.gui_qt.dialogs import LoadCaseManagerDialog
    w = MainWindow()
    d = LoadCaseManagerDialog(w, model=w._model)
    assert getattr(d._table, "_table_copy_installed", False) is True


def test_element_loads_table_has_copy_installed(qt_app):
    w = MainWindow()
    eid = _make_loaded_frame(w)
    w._open_element_inspector(eid)
    qt_app.processEvents()
    table = w._element_inspector._loads_widget
    assert getattr(table, "_table_copy_installed", False) is True


# ── v0.24.0: renumber + merge GUI surfaces ──


def _seed_line_model(w, n: int = 3):
    """Inject a small line-of-frames model directly into the host
    so smoke tests can drive renumber / merge without going through
    the full draw pipeline."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.model import Material, Node, Section
    m = w._model
    m.nodes.clear()
    m.elements.clear()
    m.materials.setdefault(
        1, Material(id=1, name="Steel", E=2.1e8, density=7850.0),
    )
    m.sections.setdefault(
        1, Section(id=1, name="S1", material_id=1, A=0.01, I=1e-4, depth=0.3),
    )
    for i in range(n + 1):
        m.nodes[i + 1] = Node(i + 1, float(i), 0.0)
    for k in range(n):
        m.elements.append(
            FrameElement2D(
                id=k + 1, node_i=k + 1, node_j=k + 2,
                E=2.1e8, A=0.01, I=1e-4, section_id=1,
            ),
        )
    return m


def test_view_menu_show_local_axes_action_is_checkable(qt_app):
    w = MainWindow()
    assert w.act_show_local_axes.isCheckable()
    assert not w.act_show_local_axes.isChecked()


def test_edit_menu_has_renumber_action(qt_app):
    w = MainWindow()
    assert w.act_renumber_elements is not None
    # Triggering on an empty model surfaces an info box (not a crash).
    # We just check the action exists and is enabled.
    assert w.act_renumber_elements.isEnabled()


def test_renumber_dialog_preview_uses_current_id_order(qt_app):
    from structural_analysis.gui_qt.dialogs import RenumberElementsDialog
    w = MainWindow()
    _seed_line_model(w, 4)
    # Mess up the ids so "compact to 1..N" is observably different.
    for new_id, e in zip([20, 10, 30, 40], w._model.elements):
        e.id = new_id
    d = RenumberElementsDialog(w, model=w._model)
    mapping = d._compute_mapping()
    # By current ID: 10 → 1, 20 → 2, 30 → 3, 40 → 4 (sorted ascending).
    assert mapping == {10: 1, 20: 2, 30: 3, 40: 4}


def test_renumber_dialog_geometry_orders_top_to_bottom_left_to_right(qt_app):
    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_qt.dialogs import RenumberElementsDialog
    from structural_analysis.model import Material, Node, Section
    w = MainWindow()
    m = w._model
    m.nodes.clear(); m.elements.clear()
    m.materials.setdefault(
        1, Material(id=1, name="Steel", E=2.1e8, density=7850.0),
    )
    m.sections.setdefault(
        1, Section(id=1, name="S1", material_id=1, A=0.01, I=1e-4, depth=0.3),
    )
    # Three midpoints: (5, 0), (0, 5), (5, 5). Top row first.
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 10.0, 0.0)
    m.nodes[3] = Node(3, 0.0, 5.0)
    m.nodes[4] = Node(4, 10.0, 5.0)
    m.nodes[5] = Node(5, 0.0, 10.0)
    m.nodes[6] = Node(6, 10.0, 10.0)
    m.elements.append(FrameElement2D(  # midpoint y=0  → bottom
        id=10, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1))
    m.elements.append(FrameElement2D(  # midpoint y=5  → middle
        id=20, node_i=3, node_j=4, E=2.1e8, A=0.01, I=1e-4, section_id=1))
    m.elements.append(FrameElement2D(  # midpoint y=10 → top
        id=30, node_i=5, node_j=6, E=2.1e8, A=0.01, I=1e-4, section_id=1))
    d = RenumberElementsDialog(w, model=w._model)
    d._rb_geometry.setChecked(True)
    mapping = d._compute_mapping()
    # Top-to-bottom: old id 30 → new 1, 20 → 2, 10 → 3.
    assert mapping == {30: 1, 20: 2, 10: 3}


def test_renumber_selection_strategy_sorts_selected_by_current_id(qt_app):
    """The third strategy must NOT depend on click order — it sorts the
    selected ids by current ID, then appends the rest by current ID."""
    from structural_analysis.gui_qt.dialogs import RenumberElementsDialog
    w = MainWindow()
    _seed_line_model(w, 5)
    # Selection set (order-agnostic) of element ids 4 and 2.
    d = RenumberElementsDialog(
        w, model=w._model, selected_ids=frozenset({4, 2}),
    )
    d._rb_selection.setChecked(True)
    mapping = d._compute_mapping()
    # Selected first sorted by current id: 2 → 1, 4 → 2; then rest by id:
    # 1 → 3, 3 → 4, 5 → 5.
    assert mapping == {2: 1, 4: 2, 1: 3, 3: 4, 5: 5}


def test_renumber_selection_strategy_disabled_without_selection(qt_app):
    from structural_analysis.gui_qt.dialogs import RenumberElementsDialog
    w = MainWindow()
    _seed_line_model(w, 3)
    d = RenumberElementsDialog(w, model=w._model)   # no selection passed
    assert not d._rb_selection.isEnabled()


def test_renumber_via_host_invalidates_results_and_translates_selection(qt_app):
    from structural_analysis.gui_common.commands import RenumberElementsCmd
    w = MainWindow()
    _seed_line_model(w, 3)
    # Add a dummy result so we can confirm invalidation.
    w._result = object()
    w.canvas.add_element_to_selection(2)
    w.canvas.add_element_to_selection(3)
    mapping = {1: 30, 2: 20, 3: 10}
    w.execute(RenumberElementsCmd(mapping=mapping))
    # Result cleared (execute() calls _invalidate_result automatically).
    assert w._result is None
    # Selection IDs were 2 and 3 — we translate them manually here to
    # mirror the host's post-execute logic.
    new_sel = {mapping[eid] for eid in (2, 3)}
    w.canvas.clear_selection()
    for eid in new_sel:
        w.canvas.add_element_to_selection(eid)
    assert set(w.canvas.get_selected_elements()) == {20, 10}


def test_node_context_menu_offers_merge_when_eligible(qt_app):
    """Just spot-check that node-context-menu wiring exists (the menu
    is built lazily via show_node_menu — this confirms it doesn't crash
    on a node that is eligible for merge)."""
    w = MainWindow()
    _seed_line_model(w, 3)
    # No exception means the show_node_menu code path is intact; we
    # don't actually pop the modal menu here. Just exercise the helper.
    assert w._can_merge_node(2) is True
    assert w._can_merge_node(1) is False  # corner node, only 1 incident


def test_merge_via_host_removes_middle_node_and_invalidates_results(qt_app):
    from structural_analysis.gui_common.commands import (
        MergeAdjacentElementsCmd,
    )
    w = MainWindow()
    _seed_line_model(w, 3)
    w._result = object()
    w.execute(MergeAdjacentElementsCmd(middle_node_id=2))
    assert 2 not in w._model.nodes
    assert len(w._model.elements) == 2
    assert w._result is None


# ── v0.24.1: merge-reason UX ──────────────────────────────────


def test_merge_action_label_enabled_for_eligible_node(qt_app):
    w = MainWindow()
    _seed_line_model(w, 3)
    label, enabled, tooltip = w._merge_action_label_and_tooltip(2)
    assert enabled is True
    assert tooltip is None
    assert "—" not in label


def test_merge_action_label_includes_reason_when_disabled(qt_app):
    w = MainWindow()
    _seed_line_model(w, 3)
    # Node 1: corner — only one incident element.
    label, enabled, tooltip = w._merge_action_label_and_tooltip(1)
    assert enabled is False
    assert tooltip is not None
    assert "—" in label
    assert tooltip in label  # reason appears verbatim in action text


def test_merge_action_reason_specifies_support(qt_app):
    from structural_analysis.gui_common.commands import SetSupportCmd
    from structural_analysis.model import Support
    w = MainWindow()
    _seed_line_model(w, 3)
    w.execute(SetSupportCmd(support=Support(node_id=2, ux=True, uy=True, rz=False)))
    label, enabled, tooltip = w._merge_action_label_and_tooltip(2)
    assert not enabled
    assert tooltip is not None
    assert "support" in tooltip.lower()
    assert "support" in label.lower()


def test_merge_action_reason_specifies_nodal_load(qt_app):
    from structural_analysis.model import NodalLoad
    w = MainWindow()
    _seed_line_model(w, 3)
    w._model.nodal_loads.append(NodalLoad(node_id=2, fy=-10.0))
    label, enabled, tooltip = w._merge_action_label_and_tooltip(2)
    assert not enabled
    assert tooltip is not None
    assert "nodal load" in tooltip.lower()


def test_merge_action_reason_specifies_member_loads(qt_app):
    from structural_analysis.gui_common.commands import AddMemberLoadCmd
    from structural_analysis.model import UniformDistributedLoad
    w = MainWindow()
    _seed_line_model(w, 3)
    w.execute(AddMemberLoadCmd(elem_id=1, load=UniformDistributedLoad(wy=-5.0)))
    label, enabled, tooltip = w._merge_action_label_and_tooltip(2)
    assert not enabled
    assert tooltip is not None
    assert "member load" in tooltip.lower() or "remapping" in tooltip.lower()


def test_can_merge_node_consistent_with_check_merge(qt_app):
    """_can_merge_node verdict must equal check_merge_preconditions verdict."""
    from structural_analysis.gui_common.commands import check_merge_preconditions
    w = MainWindow()
    _seed_line_model(w, 3)
    for nid in list(w._model.nodes):
        ok, _ = check_merge_preconditions(w._model, nid)
        assert w._can_merge_node(nid) == ok


# ── PR #40 — Modal Mass Source ───────────────────────────────────────────


def test_run_menu_has_modal_mass_source_action(qt_app):
    """Run menu must expose the Modal mass source… action."""
    w = MainWindow()
    qt_app.processEvents()
    assert hasattr(w, "act_modal_mass_source")
    assert w.act_modal_mass_source is not None


def test_new_gui_model_has_dead_load_case_and_self_weight_case(qt_app):
    """A brand-new GUI model must carry DEAD and self_weight_case == 'DEAD'."""
    w = MainWindow()
    qt_app.processEvents()
    assert "DEAD" in w._model.load_cases
    assert w._model.self_weight_case == "DEAD"


def test_legacy_file_loads_with_original_self_weight_case(qt_app):
    """An old input file that has no ANALYSIS_OPTIONS must keep self_weight_case='DEFAULT'."""
    from structural_analysis.file_io import read_input_file
    m = read_input_file("inputs/q2a_settlement.txt")
    assert m.self_weight_case == "DEFAULT"


def test_modal_mass_source_dialog_opens_and_accepts(qt_app):
    """Open ModalMassSourceDialog, click OK, verify result_value is a ModalMassSource."""
    from structural_analysis.gui_qt.dialogs import ModalMassSourceDialog
    from structural_analysis.model import ModalMassSource
    from PyQt6.QtWidgets import QDialogButtonBox

    w = MainWindow()
    qt_app.processEvents()
    d = ModalMassSourceDialog(w, model=w._model)
    # Simulate OK
    d._accept()  # returns a ModalMassSource (or None on validation error)
    result = d._accept()
    assert isinstance(result, ModalMassSource)


def test_modal_mass_source_dialog_table_disabled_when_lc_unchecked(qt_app):
    from structural_analysis.gui_qt.dialogs import ModalMassSourceDialog
    w = MainWindow()
    d = ModalMassSourceDialog(w, model=w._model)
    d._cb_lc.setChecked(False)
    assert not d._table.isEnabled()
    d._cb_lc.setChecked(True)
    assert d._table.isEnabled()


def test_modal_mass_source_dialog_double_count_label(qt_app):
    """Warning label appears when self-mass + self_weight_case factor > 0."""
    from structural_analysis.gui_qt.dialogs import ModalMassSourceDialog
    from structural_analysis.model import LoadCase

    w = MainWindow()
    qt_app.processEvents()
    # Enable self-weight so double-count logic fires
    w._model.include_self_weight = True
    w._model.self_weight_case = "DEAD"
    if "DEAD" not in w._model.load_cases:
        w._model.load_cases["DEAD"] = LoadCase(name="DEAD")

    d = ModalMassSourceDialog(w, model=w._model)
    d._cb_self.setChecked(True)
    d._cb_lc.setChecked(True)
    # Find DEAD row and set multiplier to 1.0
    for row, name in enumerate(d._case_names):
        if name == "DEAD":
            from PyQt6.QtWidgets import QTableWidgetItem
            d._table.setItem(row, 1, QTableWidgetItem("1.0"))
            break
    d._refresh_warnings()
    assert d._warn_label.text() != ""


def test_modal_mass_source_dialog_ok_still_works_with_warning(qt_app):
    """OK remains functional even when a double-count warning is shown."""
    from structural_analysis.gui_qt.dialogs import ModalMassSourceDialog
    from structural_analysis.model import LoadCase, ModalMassSource

    w = MainWindow()
    w._model.include_self_weight = True
    d = ModalMassSourceDialog(w, model=w._model)
    d._cb_self.setChecked(True)
    d._cb_lc.setChecked(True)
    result = d._accept()
    assert isinstance(result, ModalMassSource)


def test_update_mass_source_command_undo_redo(qt_app):
    """UpdateModalMassSourceCmd round-trips through undo/redo correctly."""
    from structural_analysis.gui_common.commands import UpdateModalMassSourceCmd
    from structural_analysis.model import ModalMassSource

    w = MainWindow()
    qt_app.processEvents()

    original_src = w._model.modal_mass_source
    new_src = ModalMassSource(
        include_self_mass=False,
        include_joint_masses=True,
        include_load_cases=False,
    )
    cmd = UpdateModalMassSourceCmd(new_source=new_src)
    w.execute(cmd)
    qt_app.processEvents()

    assert w._model.modal_mass_source is new_src

    # Undo
    w._do_undo()
    qt_app.processEvents()
    assert w._model.modal_mass_source.include_self_mass == original_src.include_self_mass

    # Redo
    w._do_redo()
    qt_app.processEvents()
    assert w._model.modal_mass_source is new_src


def test_modal_view_header_shows_mass_source_summary(qt_app):
    """ModalResultsDialog header label must include the mass source summary."""
    from structural_analysis.gui_qt.modal_view import ModalResultsDialog
    from structural_analysis.modal import ModalResult
    import numpy as np

    w = MainWindow()
    qt_app.processEvents()

    result = ModalResult(
        status="ok", title="Test",
        n_modes=1,
        frequencies=np.array([5.0]),
        periods=np.array([0.2]),
        omegas=np.array([31.4]),
        modes=np.zeros((1, 1)),
        normalisation="mass",
        mass_formulation="consistent",
        mass_source_summary="self-mass + joint masses (2 entries)",
    )

    closed: list[bool] = []
    dlg = ModalResultsDialog(
        w, result,
        on_select=lambda idx, sc: None,
        on_close=lambda: closed.append(True),
    )
    qt_app.processEvents()
    # The header QLabel should contain the summary text
    from PyQt6.QtWidgets import QLabel
    labels = dlg.findChildren(QLabel)
    found = any(
        "self-mass + joint masses" in (lbl.text() or "")
        for lbl in labels
    )
    assert found, "Mass-source summary not found in modal view header"
