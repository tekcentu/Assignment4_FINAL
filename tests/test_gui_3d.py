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


# ── v0.33: 3D load dialogs + projected arrows ──────────────────


def test_mechanical_world_components_triad():
    import numpy as np
    from structural_analysis.element3d import local_axes
    from structural_analysis.gui_qt.canvas import (
        _mechanical_world_components,
    )

    ni, nj = Node(1, 0.0, 0.0, 0.0), Node(2, 4.0, 0.0, 0.0)
    _, lam = local_axes(ni, nj)
    comps = _mechanical_world_components(lam, 1.0, -2.0, 3.0, "local")
    np.testing.assert_allclose(comps[0][0], (1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(comps[1][0], (0.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(comps[2][0], (0.0, 0.0, 1.0), atol=1e-12)
    assert [m for _, m in comps] == [1.0, -2.0, 3.0]

    grav = _mechanical_world_components(lam, 0.0, 5.0, 0.0, "gravity")
    assert grav == [((0.0, -1.0, 0.0), 5.0)]


def test_nodal_load_dialog_3d_components_reach_model(qt_app):
    from structural_analysis.gui_common.commands import AddNodalLoadCmd
    from structural_analysis.assembler import model_is_3d

    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0)}
    w.execute(AddNodalLoadCmd(node_id=1, fz=-4.0, mx=1.0))
    ld = w._model.nodal_loads[0]
    assert (ld.fz, ld.mx, ld.my) == (-4.0, 1.0, 0.0)
    assert model_is_3d(w._model)
    # All-zero rows are still rejected.
    import pytest as _pytest
    cmd = AddNodalLoadCmd(node_id=1)
    with _pytest.raises(ValueError, match="six components"):
        cmd.do(w._model)


def test_fz_arrow_drawn_in_plan_view(qt_app):
    from structural_analysis.model import NodalLoad

    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0, 0.0), 2: Node(2, 4.0, 0.0, 0.0)}
    w._model.nodal_loads.append(NodalLoad(2, fz=-9.0))

    # Front view: Fz looks at the camera -> text tag only.
    w.canvas.set_view_plane("XY")
    w.canvas.redraw()
    labels_xy = [t.get_text() for t in w.canvas.ax.texts]
    assert any("Fz=-9" in s for s in labels_xy)

    # Plan view: Fz becomes a real projected arrow with its label.
    w.canvas.set_view_plane("XZ")
    w.canvas.redraw()
    labels_xz = [t.get_text() for t in w.canvas.ax.texts]
    assert any(s.startswith("Fz=") for s in labels_xz)


def test_member_udl_wz_draws_in_plan_view(qt_app):
    from structural_analysis.model import (
        Material, Section, UniformDistributedLoad,
    )
    from structural_analysis.element import FrameElement2D

    w = MainWindow()
    w._model.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    e = FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01,
                       I=1e-4, section_id=1)
    e.member_loads.append(UniformDistributedLoad(wy=0.0, wz=-3.0))
    w._model.elements.append(e)
    for plane in ("XY", "XZ", "ISO"):
        w.canvas.set_view_plane(plane)
        w.canvas.redraw()  # no crash; wz arrows render where projectable
    labels = [t.get_text() for t in w.canvas.ax.texts]
    assert any("UDL" in s and "-3" in s for s in labels)


# ── v0.33: depth-aware snapping (stacked-node Tab cycling) ─────


def _stacked_window(qt_app):
    w = MainWindow()
    w._model.nodes = {
        1: Node(1, 2.0, 1.0, 0.0),
        2: Node(2, 2.0, 1.0, 4.0),   # same XY projection, z = 4
        3: Node(3, 5.0, 1.0, 0.0),   # unrelated
    }
    return w


def test_depth_stack_resolution_and_cycle(qt_app):
    w = _stacked_window(qt_app)
    c = w.canvas
    stack = c._node_stack_for(c.projected_model(), 1)
    assert stack == (1, 2)  # depth-sorted
    assert c._node_stack_for(c.projected_model(), 3) == (3,)

    # Working depth 4 biases the pick to node 2.
    w._working_depth = 4.0
    c._stack_ids = stack
    assert c._stack_order() == [2, 1]
    w._working_depth = 0.0
    assert c._stack_order() == [1, 2]


def test_depth_stack_applied_to_node_hits(qt_app):
    from structural_analysis.gui_qt.canvas import HitResult

    w = _stacked_window(qt_app)
    c = w.canvas
    hit = HitResult(x=2.0, y=1.0, node_id=1, snap_kind="node",
                    snap_label="node 1")
    c._apply_depth_stack(hit, c.projected_model())
    assert hit.node_id == 1
    assert "1/2 stacked" in hit.snap_label and "Tab" in hit.snap_label

    # Cycle: next Tab resolves to the deeper node.
    c._last_hit = hit
    c._last_event_px = (10.0, 10.0)
    seen: list[int] = []
    c.on_motion = lambda h, px: seen.append(h.node_id)
    assert c._cycle_depth_stack()
    assert seen == [2]
    assert c._last_hit.node_id == 2
    assert "2/2 stacked" in c._last_hit.snap_label

    # Cycling wraps around.
    assert c._cycle_depth_stack()
    assert c._last_hit.node_id == 1


def test_depth_stack_working_depth_wins_initial_pick(qt_app):
    from structural_analysis.gui_qt.canvas import HitResult

    w = _stacked_window(qt_app)
    w._working_depth = 4.0
    hit = HitResult(x=2.0, y=1.0, node_id=1, snap_kind="node",
                    snap_label="node 1")
    w.canvas._apply_depth_stack(hit, w.canvas.projected_model())
    assert hit.node_id == 2  # the z = 4 twin matches the working depth
    assert "depth 4" in hit.snap_label


# ── v0.33: storey manager ──────────────────────────────────────


def test_storey_helpers():
    from structural_analysis.gui_qt.storeys import (
        normalized_storeys, storey_name_for_depth,
    )

    rows = [("Roof", 6.0), ("Level 1", 0.0), ("Level 2", 3.0)]
    out = normalized_storeys(rows)
    assert out == [("Level 1", 0.0), ("Level 2", 3.0), ("Roof", 6.0)]
    assert storey_name_for_depth(out, 3.0) == "Level 2"
    assert storey_name_for_depth(out, 1.5) is None

    import pytest as _pytest
    with _pytest.raises(ValueError, match="Duplicate"):
        normalized_storeys([("A", 0.0), ("A", 1.0)])
    with _pytest.raises(ValueError, match="share the level"):
        normalized_storeys([("A", 0.0), ("B", 0.0)])


def test_storey_dialog_roundtrip_and_activation(qt_app):
    from structural_analysis.gui_qt.storeys import StoreyManagerDialog

    w = MainWindow()
    d = StoreyManagerDialog(w, storeys=[("Level 1", 0.0)],
                            current_depth=0.0)
    d._name_entry.setText("Level 2")
    d._z_spin.setValue(3.0)
    d._on_add()
    d._table.setCurrentCell(1, 0)
    d._on_activate()  # accepts + requests the depth jump
    assert d.result_storeys == [("Level 1", 0.0), ("Level 2", 3.0)]
    assert d.activated_depth == 3.0

    # The host applies the jump and labels the status with the storey.
    w._storeys = d.result_storeys
    w._apply_working_depth(3.0)
    assert w._working_depth == 3.0


def test_storeys_persist_in_project_viewstate(qt_app, tmp_path):
    from structural_analysis.gui_qt.project_io import ViewState

    vs = ViewState(storeys=[("Level 1", 0.0), ("Roof", 6.0)])
    data = vs.to_dict()
    back = ViewState.from_dict(data)
    assert back.storeys == [("Level 1", 0.0), ("Roof", 6.0)]
    # Legacy dicts (no storeys key) stay loadable.
    legacy = {k: v for k, v in data.items() if k != "storeys"}
    assert ViewState.from_dict(legacy).storeys == []


# ── v0.33.1: blitted hover overlay (canvas performance) ────────


def test_mouse_motion_never_triggers_full_redraw(qt_app, monkeypatch):
    """Per-move full scene rebuilds were the canvas lag — pin the fix:
    the motion handler must use the blit overlay, not canvas.redraw."""
    w = MainWindow(initial_path="inputs/example_3d_table_frame.txt")
    qt_app.processEvents()
    w.canvas._mpl_canvas.draw()  # establish the blit background

    full_redraws: list[int] = []
    overlay_updates: list[int] = []
    monkeypatch.setattr(w.canvas, "redraw",
                        lambda: full_redraws.append(1))
    monkeypatch.setattr(w.canvas, "update_hover_overlay",
                        lambda: overlay_updates.append(1))
    w._on_canvas_motion(HitResult(x=1.0, y=1.0), (100.0, 100.0))
    assert full_redraws == []
    assert overlay_updates == [1]


def test_update_hover_overlay_paths(qt_app):
    w = MainWindow(initial_path="inputs/example_3d_table_frame.txt")
    qt_app.processEvents()
    w.canvas._mpl_canvas.draw()
    assert w.canvas._overlay_bg is not None  # captured via draw_event

    # Hover ghost, member rubber-band, and box-select rect all repaint
    # through the blit path without touching the scene artists.
    w.canvas._hover_xy = (1.0, 1.0)
    w.canvas.update_hover_overlay()
    w.canvas.set_element_preview_free(0.0, 0.0, 2.0, 1.0, "frame")
    w.canvas.update_hover_overlay()
    assert list(w.canvas._overlay_preview_line.get_xdata()) == [0.0, 2.0]
    w.canvas.set_drag_rect(0.0, 0.0, 1.0, 1.0, True)
    w.canvas.update_hover_overlay()
    assert w.canvas._overlay_rect.get_visible()
    w.canvas.clear_drag_rect()
    w.canvas.clear_element_preview()
    w.canvas.update_hover_overlay()
    assert not w.canvas._overlay_rect.get_visible()
