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

from PyQt6.QtWidgets import QApplication  # noqa: E402

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
    d._x_entry.setText("12, 0, 6")
    d._y_entry.setText("8, 0, 4")

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
