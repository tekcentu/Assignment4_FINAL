"""GUI tests for the 3D upgrade (v0.32): work planes, projected model
view, working depth, 6-DOF support dialog, and end-to-end 3D solving
through the MainWindow.

Runs under the ``offscreen`` Qt platform like the other GUI smoke
tests; skipped entirely when PyQt6 is unavailable.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication  # noqa: E402
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 QtWidgets unavailable: {exc}",
                allow_module_level=True)

from structural_analysis.gui_qt.app import MainWindow  # noqa: E402
from structural_analysis.gui_qt.canvas import HitResult  # noqa: E402
from structural_analysis.model import Node, Support  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_canvas_view_plane_default_and_switching(qt_app):
    w = MainWindow()
    assert w.canvas.view_plane == "XY"
    assert w.canvas.is_editable_view

    w.canvas.set_view_plane("XZ")
    assert w.canvas.view_plane == "XZ"
    assert not w.canvas.is_editable_view

    w.canvas.set_view_plane("ISO")
    assert not w.canvas.is_editable_view

    with pytest.raises(ValueError):
        w.canvas.set_view_plane("QQ")


def test_projection_math(qt_app):
    w = MainWindow()
    c = w.canvas
    assert c.project_xyz(1.0, 2.0, 3.0) == (1.0, 2.0)  # XY identity
    c.set_view_plane("XZ")
    assert c.project_xyz(1.0, 2.0, 3.0) == (1.0, 3.0)
    assert c.plane_to_world(1.0, 3.0, depth=2.0) == (1.0, 2.0, 3.0)
    c.set_view_plane("ZY")
    assert c.project_xyz(1.0, 2.0, 3.0) == (3.0, 2.0)
    assert c.plane_to_world(3.0, 2.0, depth=1.0) == (1.0, 2.0, 3.0)
    c.set_view_plane("ISO")
    with pytest.raises(ValueError):
        c.plane_to_world(0.0, 0.0)


def test_projected_model_identity_in_xy_and_shim_in_xz(qt_app):
    w = MainWindow()
    w._model.nodes = {1: Node(1, 1.0, 2.0, 3.0)}
    assert w.canvas.projected_model() is w._model  # identity in XY

    w.canvas.set_view_plane("XZ")
    proj = w.canvas.projected_model()
    assert proj is not w._model
    n = proj.nodes[1]
    assert (n.x, n.y) == (1.0, 3.0)
    # Non-node attributes delegate to the real model.
    assert proj.elements is w._model.elements


def test_node_tool_uses_working_depth(qt_app):
    w = MainWindow()
    w._working_depth = 2.5
    w._select_tool("node")
    w._on_canvas_click(HitResult(x=1.0, y=4.0), "left")
    assert len(w._model.nodes) == 1
    node = next(iter(w._model.nodes.values()))
    assert (node.x, node.y, node.z) == (1.0, 4.0, 2.5)


def test_creation_tools_blocked_in_iso_view(qt_app):
    w = MainWindow()
    w.canvas.set_view_plane("ISO")
    w._select_tool("node")
    w._on_canvas_click(HitResult(x=1.0, y=1.0), "left")
    assert len(w._model.nodes) == 0  # blocked, no node created

    w._select_tool("frame")
    w._on_canvas_click(HitResult(x=0.0, y=0.0), "left")
    assert w._tools["frame"]._first is None  # click was rejected


def test_support_dialog_six_dof_roundtrip(qt_app):
    from structural_analysis.gui_qt.dialogs import SupportDialog

    existing = Support(1, True, True, False, uz=True, ry=True,
                       settle_uz=-0.01)
    d = SupportDialog(None, existing=existing, node_id=1)
    assert d._cb_uz.isChecked()
    assert d._cb_ry.isChecked()
    assert not d._cb_rx.isChecked()
    assert d._settle["settle_uz"].text() == repr(-0.01)

    d._cb_rx.setChecked(True)
    action, sup = d._accept()
    assert action == "set"
    assert sup.uz and sup.rx and sup.ry
    assert sup.settle_uz == -0.01


def test_support_dialog_rejects_settlement_on_free_3d_dof(qt_app):
    from structural_analysis.gui_qt.dialogs import SupportDialog

    d = SupportDialog(None, existing=None, node_id=1)
    d._cb_ux.setChecked(True)
    d._settle["settle_uz"].setText("0.01")
    with pytest.raises(ValueError, match="uz"):
        d._accept()


def test_fine_node_dialog_returns_z(qt_app):
    from structural_analysis.gui_qt.dialogs import FineNodeDialog

    w = MainWindow()
    d = FineNodeDialog(w, model=w._model)
    d._x_entry.setText("1.0")
    d._y_entry.setText("2.0")
    d._z_entry.setText("3.0")
    assert d._accept() == (1.0, 2.0, 3.0)


def test_open_and_solve_3d_example_through_gui(qt_app):
    w = MainWindow(initial_path="inputs/example_3d_grillage.txt")
    qt_app.processEvents()
    assert len(w._model.nodes) == 3
    assert w._model.nodes[3].z == 3.0

    w._do_solve()
    assert w._result is not None
    assert w._result.status == "ok"
    # 3D result: a 12-component member force vector with the in-plane
    # slice the canvas diagram overlays consume.
    mr = w._result.member_results[1]
    assert len(mr["f_local"]) == 12
    assert len(mr["f_local_inplane"]) == 6

    # Redraw in every view plane with the result showing — no crashes,
    # and the deformed overlay path runs through the 3D fallbacks.
    for plane in ("XY", "XZ", "ZY", "ISO"):
        w.canvas.set_view_plane(plane)
        w.canvas.redraw()
    w.canvas.set_view_plane("XY")


def test_box_select_in_xz_plane(qt_app):
    w = MainWindow()
    w._model.nodes = {
        1: Node(1, 0.0, 0.0, 0.0),
        2: Node(2, 2.0, 0.0, 2.0),
    }
    w.canvas.set_view_plane("XZ")
    # In the XZ plan view node 2 projects to (2, 2).
    w.apply_box_select((1.5, 1.5, 2.5, 2.5), shift=False,
                       is_crossing=False)
    assert w.canvas.get_selected_nodes() == frozenset({2})


def test_connect_selected_nodes_across_depths(qt_app):
    w = MainWindow(initial_path="inputs/example_3d_grillage.txt")
    qt_app.processEvents()
    seen: list[tuple[int, int]] = []
    w.open_element_dialog_for_pair = (
        lambda n_i, n_j, kind=None: seen.append((n_i, n_j))
    )
    w.canvas.add_node_to_selection(1)
    w.canvas.add_node_to_selection(3)
    w._connect_selected_nodes()
    assert seen == [(1, 3)]


def test_origin_axes_show_z_in_xz_view(qt_app):
    w = MainWindow()
    w.canvas.set_view_plane("XZ")
    w.canvas.redraw()
    labels = [t.get_text() for t in w.canvas.ax.texts]
    assert "X" in labels
    assert "Z" in labels
    assert "Y" not in labels  # Y looks at the camera in plan view


def test_view3d_window_uses_native_z(qt_app):
    from structural_analysis.gui_qt.view3d import _node_world

    p = _node_world(1.0, 2.0, "y_up", 3.0)
    assert tuple(p) == (1.0, 2.0, 3.0)
    p = _node_world(1.0, 2.0, "z_up", 3.0)
    assert tuple(p) == (1.0, 3.0, 2.0)
