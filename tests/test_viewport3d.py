"""Tests for the 3D viewport (v0.33): the Qt-free spatial math in
``gui_common/spatial.py`` plus an offscreen shell smoke test of
``gui_qt/viewport3d.py`` (skipped when the optional GL stack is
missing).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from structural_analysis.gui_common import spatial
from structural_analysis.model import Node, StructuralModel, Support

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _ortho_mvp(half: float = 10.0) -> np.ndarray:
    """Simple orthographic MVP looking down -Z, world XY → screen."""
    m = np.eye(4)
    m[0, 0] = 1.0 / half
    m[1, 1] = 1.0 / half
    m[2, 2] = -1.0 / half
    return m


# ── projection ─────────────────────────────────────────────────


def test_project_points_orthographic_center_and_corners():
    mvp = _ortho_mvp(10.0)
    pts = np.array([
        [0.0, 0.0, 0.0],    # center
        [10.0, 10.0, 0.0],  # top-right of the view volume
        [-10.0, -10.0, 0.0],
    ])
    screen, valid = spatial.project_points(pts, mvp, 200, 100)
    assert valid.all()
    np.testing.assert_allclose(screen[0], [100.0, 50.0])
    np.testing.assert_allclose(screen[1], [200.0, 0.0])   # y flips (Qt)
    np.testing.assert_allclose(screen[2], [0.0, 100.0])


def test_project_points_flags_degenerate_w():
    mvp = np.zeros((4, 4))  # w == 0 for everything
    screen, valid = spatial.project_points(
        np.array([[1.0, 2.0, 3.0]]), mvp, 100, 100,
    )
    assert not valid.any()
    assert np.isnan(screen).all()


# ── picking ────────────────────────────────────────────────────


def test_nearest_point_index_radius_and_validity():
    screen = np.array([[10.0, 10.0], [50.0, 50.0]])
    valid = np.array([True, True])
    assert spatial.nearest_point_index(screen, valid, 12, 11, 5) == 0
    assert spatial.nearest_point_index(screen, valid, 30, 30, 5) is None
    valid = np.array([False, True])
    assert spatial.nearest_point_index(screen, valid, 12, 11, 5) is None


def test_nearest_segment_index_interior_hit():
    segs = np.array([
        [[0.0, 0.0], [100.0, 0.0]],
        [[0.0, 50.0], [100.0, 50.0]],
    ])
    valid = np.array([True, True])
    assert spatial.nearest_segment_index(segs, valid, 50, 4, 8) == 0
    assert spatial.nearest_segment_index(segs, valid, 50, 46, 8) == 1
    assert spatial.nearest_segment_index(segs, valid, 50, 25, 8) is None


# ── rays / construction planes ─────────────────────────────────


def test_ray_from_screen_and_storey_plane_intersection():
    mvp = _ortho_mvp(10.0)
    ray = spatial.ray_from_screen(100, 50, mvp, 200, 100)
    assert ray is not None
    origin, direction = ray
    # Center cursor under this MVP looks along ±Z through the origin.
    np.testing.assert_allclose(direction[:2], [0.0, 0.0], atol=1e-12)
    hit = spatial.ray_axis_plane_intersection(
        np.array([0.0, 5.0, -10.0]), np.array([0.0, -1.0, 0.0]),
        axis=1, value=2.0,
    )
    assert hit == pytest.approx((0.0, 2.0, -10.0))
    # Parallel ray → no hit.
    assert spatial.ray_axis_plane_intersection(
        np.array([0.0, 5.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        axis=1, value=2.0,
    ) is None
    # Hit behind the origin → no hit.
    assert spatial.ray_axis_plane_intersection(
        np.array([0.0, 5.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        axis=1, value=2.0,
    ) is None


# ── fingerprint ────────────────────────────────────────────────


def test_model_fingerprint_tracks_changes():
    m = StructuralModel()
    m.nodes[1] = Node(1, 0.0, 0.0)
    fp0 = spatial.model_fingerprint(m)
    assert spatial.model_fingerprint(m) == fp0
    m.nodes[2] = Node(2, 1.0, 0.0, 2.0)
    fp1 = spatial.model_fingerprint(m)
    assert fp1 != fp0
    m.supports[1] = Support(1, True, True, False, uz=True)
    assert spatial.model_fingerprint(m) != fp1


# ── viewport shell (needs PyQt6 + optional GL stack) ───────────


@pytest.fixture(scope="module")
def qt_app():
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_viewport_window_constructs_and_tracks_model(qt_app):
    pytest.importorskip("pyqtgraph.opengl")
    from structural_analysis.gui_qt.app import MainWindow
    from structural_analysis.gui_qt.viewport3d import Viewport3DWindow

    w = MainWindow(initial_path="inputs/example_3d_table_frame.txt")
    qt_app.processEvents()
    vp = Viewport3DWindow(w)
    try:
        assert vp._fingerprint == spatial.model_fingerprint(w._model)
        n_items = len(vp._scene_items)
        assert n_items > 0  # element lines + node scatter at minimum

        # Model edits show up on refresh (the poll calls the same path).
        from structural_analysis.gui_common.commands import AddNodeCmd
        w.execute(AddNodeCmd(x=9.0, y=0.0, z=9.0))
        vp._poll_model()
        assert vp._fingerprint == spatial.model_fingerprint(w._model)

        # Construction-plane state drives the click → world resolution.
        vp._plane_combo.setCurrentIndex(0)  # Y = const storey plane
        vp._depth_spin.setValue(3.0)
        assert vp._plane() == (1, 3.0)
    finally:
        vp.close()


def test_viewport_member_tool_uses_command_stack(qt_app):
    pytest.importorskip("pyqtgraph.opengl")
    from structural_analysis.gui_qt.app import MainWindow
    from structural_analysis.gui_qt.viewport3d import Viewport3DWindow

    w = MainWindow()  # starter model carries two sections
    vp = Viewport3DWindow(w)
    try:
        first = (None, (0.0, 0.0, 0.0))
        second = (None, (3.0, 0.0, 3.0))
        vp._create_member("frame", first, second)
        assert len(w._model.elements) == 1
        elem = w._model.elements[0]
        ni = w._model.nodes[elem.node_i]
        nj = w._model.nodes[elem.node_j]
        assert (ni.x, ni.y, ni.z) == (0.0, 0.0, 0.0)
        assert (nj.x, nj.y, nj.z) == (3.0, 0.0, 3.0)
        # One undo reverses the whole gesture.
        w._do_undo()
        assert len(w._model.elements) == 0
    finally:
        vp.close()


def test_pyqtgraph_version_advisory(monkeypatch):
    pytest.importorskip("pyqtgraph")
    import pyqtgraph
    from structural_analysis.gui_qt import viewport3d

    monkeypatch.setattr(pyqtgraph, "__version__", "0.13.7")
    msg = viewport3d._pyqtgraph_version_advisory()
    assert msg is not None and "pip install -U pyqtgraph" in msg

    monkeypatch.setattr(pyqtgraph, "__version__", "0.14.0")
    assert viewport3d._pyqtgraph_version_advisory() is None


def test_scene_view_requests_compatibility_profile(qt_app):
    pytest.importorskip("pyqtgraph.opengl")
    from PyQt6.QtGui import QSurfaceFormat
    from structural_analysis.gui_qt.viewport3d import _SceneView

    view = _SceneView(on_click=lambda x, y: None)
    fmt = view.format()
    assert (fmt.profile()
            == QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
