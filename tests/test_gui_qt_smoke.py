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
                        on_add_or_update=lambda _x: None,
                        on_delete=lambda _x: None)


def test_member_load_dialog_raises_for_unknown_element(qt_app):
    from structural_analysis.gui_qt.dialogs import MemberLoadDialog

    w = MainWindow(initial_path="inputs/q2a_settlement.txt")
    qt_app.processEvents()
    with pytest.raises(ValueError):
        MemberLoadDialog(w, model=w._model, elem_id=9999)
