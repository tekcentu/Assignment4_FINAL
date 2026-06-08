"""Tests for PR #47: toolbar cleanup and ESC focus restoration."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 unavailable: {exc}", allow_module_level=True)

from structural_analysis.gui_qt.app import MainWindow
from structural_analysis.gui_qt.canvas import AppNavigationToolbar, HitResult


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


# ── Toolbar action filtering ───────────────────────────────────────────────


def _toolbar_action_texts(w: MainWindow) -> list[str]:
    return [a.text() for a in w.canvas.toolbar.actions()]


def test_toolbar_does_not_expose_subplots(qt_app):
    w = MainWindow()
    assert "Subplots" not in _toolbar_action_texts(w)


def test_toolbar_does_not_expose_customize(qt_app):
    w = MainWindow()
    texts = _toolbar_action_texts(w)
    assert "Customize" not in texts
    assert "Edit axis, curve and image parameters" not in texts


def test_toolbar_retains_home(qt_app):
    w = MainWindow()
    assert "Home" in _toolbar_action_texts(w)


def test_toolbar_retains_pan(qt_app):
    w = MainWindow()
    assert "Pan" in _toolbar_action_texts(w)


def test_toolbar_retains_zoom(qt_app):
    w = MainWindow()
    assert "Zoom" in _toolbar_action_texts(w)


def test_toolbar_retains_save(qt_app):
    w = MainWindow()
    assert "Save" in _toolbar_action_texts(w)


def test_toolbar_has_fit_action(qt_app):
    w = MainWindow()
    assert "Fit" in _toolbar_action_texts(w)


def test_toolbar_is_app_subclass(qt_app):
    w = MainWindow()
    assert isinstance(w.canvas.toolbar, AppNavigationToolbar)


# ── Fit action calls fit_to_view ───────────────────────────────────────────


def test_fit_action_resets_view(qt_app):
    w = MainWindow()
    # Manually dirty the view, then trigger the Fit action.
    w.canvas._user_view_dirty = True
    w.canvas._view_initialised = True

    fit_act = w.canvas.toolbar._fit_action
    assert fit_act is not None
    fit_act.trigger()

    # fit_to_view clears the dirty flag; redraw() re-initialises the view.
    assert not w.canvas._user_view_dirty


# ── Focus policy ───────────────────────────────────────────────────────────


def test_canvas_widget_has_strong_focus(qt_app):
    from PyQt6.QtCore import Qt
    w = MainWindow()
    assert w.canvas.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_mpl_canvas_has_strong_focus(qt_app):
    from PyQt6.QtCore import Qt
    w = MainWindow()
    assert w.canvas._mpl_canvas.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_main_window_has_strong_focus(qt_app):
    from PyQt6.QtCore import Qt
    w = MainWindow()
    assert w.focusPolicy() == Qt.FocusPolicy.StrongFocus


# ── ESC routing via eventFilter ────────────────────────────────────────────


def _make_esc_key_event():
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    return QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                     Qt.KeyboardModifier.NoModifier)


def test_esc_via_event_filter_cancels_frame_tool(qt_app):
    """ESC dispatched to the mpl canvas must cancel an in-progress frame draw."""
    w = MainWindow()
    w._select_tool("frame")
    tool = w._tools["frame"]

    # Simulate a first click landing on empty space.
    tool._first = (1.0, 1.0, None, None)
    w.canvas.set_element_preview_free(1.0, 1.0, 2.0, 2.0, "frame")

    # Fire ESC via eventFilter (mimics focus being on the mpl canvas).
    evt = _make_esc_key_event()
    w.canvas.eventFilter(w.canvas._mpl_canvas, evt)

    assert tool._first is None
    assert w.canvas._element_preview_free is None
    assert w._active_tool is w._tools["select"]


def test_esc_via_event_filter_cancels_truss_tool(qt_app):
    """Same as above for the Truss tool."""
    w = MainWindow()
    w._select_tool("truss")
    tool = w._tools["truss"]
    tool._first = (0.0, 0.0, None, None)
    w.canvas.set_element_preview_free(0.0, 0.0, 1.0, 1.0, "truss")

    evt = _make_esc_key_event()
    w.canvas.eventFilter(w.canvas._mpl_canvas, evt)

    assert tool._first is None
    assert w.canvas._element_preview_free is None
    assert w._active_tool is w._tools["select"]


def test_esc_noop_when_no_tool_active(qt_app):
    """ESC in the select tool with nothing in progress must not raise."""
    w = MainWindow()
    assert w._active_tool is w._tools["select"]
    undo_before = len(w._undo)

    evt = _make_esc_key_event()
    w.canvas.eventFilter(w.canvas._mpl_canvas, evt)

    assert len(w._undo) == undo_before  # no side-effect
    assert w._active_tool is w._tools["select"]


# ── ESC pan/zoom priority ──────────────────────────────────────────────────


def test_esc_in_pan_mode_does_not_switch_tool(qt_app):
    """When pan mode is active, ESC exits it but keeps the active drawing tool."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeyEvent, QKeySequence

    w = MainWindow()
    w._select_tool("frame")

    # Simulate toolbar being in pan mode.
    w.canvas.toolbar.mode = "pan/zoom"

    class _FakeEsc:
        def key(self):
            from PyQt6.QtCore import Qt
            return Qt.Key.Key_Escape
        def accept(self):
            pass

    # Patch pan so we can count calls without actually toggling Qt state.
    called = []
    orig_pan = w.canvas.toolbar.__class__.pan

    def _mock_pan(self_tb):
        called.append("pan")
        w.canvas.toolbar.mode = ""

    w.canvas.toolbar.__class__.pan = _mock_pan
    try:
        w.keyPressEvent(_FakeEsc())
    finally:
        w.canvas.toolbar.__class__.pan = orig_pan

    assert called == ["pan"], "pan() should have been called once"
    # Active tool must remain "frame", not be switched to "select".
    assert w._active_tool is w._tools["frame"]


def test_esc_in_zoom_mode_exits_zoom(qt_app):
    """When zoom mode is active, ESC exits it without switching tools."""
    w = MainWindow()
    w._select_tool("truss")
    w.canvas.toolbar.mode = "zoom rect"

    called = []
    orig_zoom = w.canvas.toolbar.__class__.zoom

    def _mock_zoom(self_tb):
        called.append("zoom")
        w.canvas.toolbar.mode = ""

    w.canvas.toolbar.__class__.zoom = _mock_zoom
    try:
        class _FakeEsc:
            def key(self):
                from PyQt6.QtCore import Qt
                return Qt.Key.Key_Escape
            def accept(self):
                pass

        w.keyPressEvent(_FakeEsc())
    finally:
        w.canvas.toolbar.__class__.zoom = orig_zoom

    assert called == ["zoom"]
    assert w._active_tool is w._tools["truss"]


def test_esc_after_pan_exit_switches_tool(qt_app):
    """Second ESC (after pan mode is already cleared) must switch to Select."""
    from PyQt6.QtCore import Qt

    w = MainWindow()
    w._select_tool("frame")
    # Pan mode already off.
    w.canvas.toolbar.mode = ""

    class _FakeEsc:
        def key(self):
            return Qt.Key.Key_Escape
        def accept(self):
            pass

    w.keyPressEvent(_FakeEsc())
    assert w._active_tool is w._tools["select"]
