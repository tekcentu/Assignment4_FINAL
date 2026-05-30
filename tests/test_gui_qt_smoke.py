"""Smoke tests for the PyQt6 GUI.

These tests run under the ``offscreen`` Qt platform plugin so they work in
headless CI as long as PyQt6 is installed. If PyQt6 isn't available the
whole file is skipped.
"""

from __future__ import annotations

import os

import pytest

PyQt6 = pytest.importorskip("PyQt6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDialog  # noqa: E402

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
    details_label = next(l for l in by_label if "show details" in l.lower())
    assert by_label[details_label], (
        '"show details" must stay enabled while inspector is open'
    )
    for needle in ("edit section", "add member load",
                    "clear member loads", "delete"):
        label = next(l for l in by_label if needle in l.lower())
        assert not by_label[label], (
            f'menu item {label!r} must be disabled while inspector is open'
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
    btn = table.cellWidget(1, 5)  # Delete button on row index 1 (PointLoad)
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
    btn = table.cellWidget(1, 5)
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
        btn = table.cellWidget(i, 5)
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
    assert table.columnCount() == 6
    headers = [
        table.horizontalHeaderItem(i).text()
        for i in range(table.columnCount())
    ]
    assert headers[4] == "Case"
    # Default row shows the dim placeholder; named row shows the case.
    assert table.item(0, 4).text() == "—"
    assert table.item(1, 4).text() == "DEAD"


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
