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
    seen: list[tuple[int, int, str | None]] = []

    def fake_open(n_i, n_j, kind=None):
        seen.append((n_i, n_j, kind))

    w.open_element_dialog_for_pair = fake_open
    w._select_tool("truss")
    w._on_canvas_click(HitResult(x=0.0, y=0.0, node_id=1), "left")
    w._on_canvas_click(HitResult(x=2.0, y=0.0, node_id=2), "left")

    assert seen == [(1, 2, "truss")]


def test_canvas_draws_origin_axes(qt_app):
    w = MainWindow()
    w.canvas.redraw()

    labels = [text.get_text() for text in w.canvas.ax.texts]
    assert "0,0" in labels
    assert "X" in labels
    assert "Y" in labels


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
    w._on_canvas_click(HitResult(x=0.0, y=0.0, node_id=1), "left")
    assert w.canvas._selected_node_id == 1
    assert "Selected node 1" in w._status_label.text()

    w._on_canvas_click(HitResult(x=1.0, y=0.0, element_id=1), "left")
    assert w.canvas._selected_element_id == 1
    assert w.canvas._selected_node_id is None
    assert "Selected element 1" in w._status_label.text()

    w._on_canvas_click(HitResult(x=5.0, y=5.0), "left")
    assert w.canvas._selected_element_id is None
    assert w.canvas._selected_node_id is None
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


def test_select_tool_left_click_shows_details(qt_app):
    """Left-clicking a node or element with the Select tool must open
    the read-only details dialog directly — no right-click menu needed."""
    from structural_analysis.model import Node

    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 2.0, 0.0)}

    calls: list[tuple[str, int]] = []
    w.show_node_details = lambda nid: calls.append(("node", nid))
    w.show_element_details = lambda eid: calls.append(("elem", eid))

    w._select_tool("select")
    w._on_canvas_click(HitResult(x=0.0, y=0.0, node_id=1), "left")
    assert calls == [("node", 1)]
    w._on_canvas_click(HitResult(x=1.0, y=0.0, element_id=2), "left")
    assert calls == [("node", 1), ("elem", 2)]
    # Empty click — no detail dialog opens.
    w._on_canvas_click(HitResult(x=5.0, y=5.0), "left")
    assert calls == [("node", 1), ("elem", 2)]


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


def test_version_label_visible_in_menubar(qt_app):
    """The window menu bar must carry a version + what's-new label."""
    from structural_analysis import __version__, __what_is_new__

    w = MainWindow()
    qt_app.processEvents()
    assert hasattr(w, "_version_label")
    text = w._version_label.text()
    assert __version__ in text
    assert __what_is_new__ in text
    # Corner widget is wired into the menu bar.
    from PyQt6.QtCore import Qt as _Qt
    assert (
        w.menuBar().cornerWidget(_Qt.Corner.TopRightCorner)
        is w._version_label
    )


def test_building_wizard_creates_model(qt_app):
    """The wizard generates a portal frame and routes through ReplaceModelCmd
    so a single Undo restores the previous model."""
    from structural_analysis.gui_qt.dialogs import BuildingWizardDialog

    w = MainWindow()
    qt_app.processEvents()

    # The starter model is empty of nodes/elements but has sections,
    # so the wizard dialog should construct.
    d = BuildingWizardDialog(w, model=w._model)
    d._stories.setValue(2)
    d._story_h.setValue(3.0)
    d._bays.setValue(2)
    d._bay_w.setValue(4.0)
    d._fixed_base.setChecked(True)
    new_model = d._accept()
    # 2 stories × 2 bays → (2+1)*(2+1) = 9 nodes,
    # columns: 3 lines × 2 stories = 6; beams: 2 floors × 2 bays = 4
    assert len(new_model.nodes) == 9
    assert len(new_model.elements) == 10
    # All ground nodes get fixed supports.
    assert len(new_model.supports) == 3
    # Materials / sections preserved from source.
    assert new_model.materials == w._model.materials
    assert new_model.sections == w._model.sections


def test_building_wizard_action_undoable(qt_app):
    """Driving the wizard handler through a stubbed dialog must apply
    ReplaceModelCmd; one Undo must restore the previous (empty) model."""
    from structural_analysis.gui_qt import app as app_mod
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

    # Bypass the QMessageBox.question confirmation when the model already
    # has content. Starter model is empty so the confirm path isn't taken.
    BuildingWizardDialog.exec = fake_exec
    try:
        w._do_building_wizard()
    finally:
        BuildingWizardDialog.exec = original_exec

    # 1 story × 1 bay → 4 nodes, 2 columns + 1 beam = 3 elements
    assert len(w._model.nodes) == 4
    assert len(w._model.elements) == 3
    # Undo restores the empty starter model.
    w._do_undo()
    assert len(w._model.nodes) == 0
    assert len(w._model.elements) == 0


def test_canvas_scroll_zoom_changes_xlim(qt_app):
    """Scrolling up over the canvas zooms in (xlim/ylim shrink around the
    cursor)."""
    import types
    w = MainWindow()
    qt_app.processEvents()
    w.canvas.ax.set_xlim(0.0, 10.0)
    w.canvas.ax.set_ylim(0.0, 10.0)
    w.canvas._view_initialised = True

    ev = types.SimpleNamespace(
        inaxes=w.canvas.ax, xdata=5.0, ydata=5.0, button="up",
    )
    w.canvas._handle_scroll(ev)
    x0, x1 = w.canvas.ax.get_xlim()
    # Range must have shrunk and stayed centered on (5, 5).
    assert (x1 - x0) < 10.0 - 1e-9
    assert abs((x0 + x1) / 2 - 5.0) < 1e-6


def test_canvas_middle_button_pan(qt_app):
    """Pressing the middle button, dragging, and releasing pans the
    axes by the data-space delta."""
    import types
    w = MainWindow()
    qt_app.processEvents()
    w.canvas.ax.set_xlim(0.0, 10.0)
    w.canvas.ax.set_ylim(0.0, 10.0)
    w.canvas._view_initialised = True

    press = types.SimpleNamespace(
        button=2, inaxes=w.canvas.ax, xdata=5.0, ydata=5.0,
    )
    w.canvas._handle_pan_press(press)
    assert w.canvas._pan_state is not None

    move = types.SimpleNamespace(xdata=6.0, ydata=5.5)
    w.canvas._handle_pan_motion(move)
    # 1.0 unit drag right and 0.5 up means the world shifts left/down.
    x0, x1 = w.canvas.ax.get_xlim()
    y0, y1 = w.canvas.ax.get_ylim()
    assert abs(x0 - (-1.0)) < 1e-6
    assert abs(x1 - 9.0) < 1e-6
    assert abs(y0 - (-0.5)) < 1e-6
    assert abs(y1 - 9.5) < 1e-6

    release = types.SimpleNamespace(button=2)
    w.canvas._handle_pan_release(release)
    assert w.canvas._pan_state is None


def test_canvas_middle_click_does_not_trigger_tools(qt_app):
    """Middle-button presses must be reserved for panning and never reach
    the active tool's on_click."""
    import types
    w = MainWindow()
    qt_app.processEvents()
    received: list[tuple] = []
    w.canvas.on_click = lambda hit, button: received.append((hit, button))
    ev = types.SimpleNamespace(
        button=2, inaxes=w.canvas.ax, xdata=1.0, ydata=2.0,
    )
    w.canvas._handle_click(ev)
    assert received == []


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
    # Non-modal: parent window must remain interactive.
    assert not first.isModal()
    # Re-opening reuses the same instance.
    w._open_view3d()
    assert w._view3d_window is first


def test_view3d_window_builds_one_mesh_per_element(qt_app):
    """Each frame/truss element should land as exactly one
    Poly3DCollection so future stress overlays can recolour faces
    per-element without re-meshing."""
    w = MainWindow(initial_path="inputs/example_01_cantilever_tip_load.txt")
    qt_app.processEvents()
    w._open_view3d()
    qt_app.processEvents()
    view = w._view3d_window
    n_elems = len(w._model.elements)
    assert len(view._element_meshes) == n_elems
    # Refresh re-builds in place; count must still match.
    view.refresh()
    assert len(view._element_meshes) == n_elems


def test_view3d_manual_section_uses_sqrt_A_and_shows_banner(qt_app):
    """A model containing a manual section must show the approximation
    banner; updating *every* manual section to a real shape (via the
    real AddOrUpdateSectionCmd) clears the banner on refresh."""
    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_common.commands import AddOrUpdateSectionCmd
    from structural_analysis.model import Material, Node, Section

    w = MainWindow()
    w._model.materials = {1: Material(id=1, name="Steel", E=2.10e8)}
    w._model.sections = {
        # shape_type defaults to "manual" — exactly what we want here.
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
        "manual sections must surface the √A approximation banner"
    )

    # Promote through AddOrUpdateSectionCmd — the real command path.
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
