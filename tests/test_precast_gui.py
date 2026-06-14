"""PyQt6 smoke tests for the precast handling-stage tool window.

Run under the offscreen Qt platform plugin so they work headless. The
pure statics are exercised in ``test_precast.py``; here we only confirm
the window constructs, drives the engine, rejects a truss selection
clearly, and never mutates the main model.
"""

from __future__ import annotations

import copy
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication  # noqa: E402
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 QtWidgets unavailable: {exc}", allow_module_level=True)

from structural_analysis.element import FrameElement2D, TrussElement2D  # noqa: E402
from structural_analysis.gui_qt.app import MainWindow  # noqa: E402
from structural_analysis.gui_qt.precast import (  # noqa: E402
    SCHEME_ONE_POINT,
    STAGE_TRUCK,
)
from structural_analysis.gui_qt.precast_window import (  # noqa: E402
    PrecastHandlingWindow,
)
from structural_analysis.model import (  # noqa: E402
    Material,
    Node,
    Section,
    Support,
)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _seed_frame(w: MainWindow, L: float = 8.0) -> int:
    m = w._model
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, L, 0.0)}
    m.materials = {1: Material(id=1, name="C", E=3e7, density=2400.0)}
    m.sections = {1: Section(id=1, name="PC", material_id=1, A=0.2, I=0.05,
                             depth=0.4)}
    m.elements = [FrameElement2D(id=1, node_i=1, node_j=2, E=3e7, A=0.2,
                                 I=0.05, rho=2400.0, section_id=1)]
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    return 1


def test_window_constructs_and_targets_frame(qt_app):
    w = MainWindow()
    eid = _seed_frame(w)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    # Default stage produces two reactions in the results table.
    assert win._results.rowCount() == 2
    assert "Element 1" in win._member_label.text()


def test_window_recomputes_on_stage_and_scheme_change(qt_app):
    w = MainWindow()
    eid = _seed_frame(w)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)

    # Switch to one-point lifting → exactly one reaction row.
    win._stage_combo.setCurrentIndex(0)  # lifting
    idx = win._scheme_combo.findData(SCHEME_ONE_POINT)
    win._scheme_combo.setCurrentIndex(idx)
    qt_app.processEvents()
    assert win._results.rowCount() == 1

    # Truck stage → two supports, no sling columns.
    tidx = win._stage_combo.findData(STAGE_TRUCK)
    win._stage_combo.setCurrentIndex(tidx)
    qt_app.processEvents()
    assert win._results.rowCount() == 2


def test_copy_report_populates_clipboard(qt_app):
    w = MainWindow()
    eid = _seed_frame(w)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    win._copy_report()
    cb = QApplication.clipboard()
    assert "Precast Handling Stage" in cb.text()
    assert "display-only" in cb.text()


def test_menu_handler_rejects_truss(qt_app, monkeypatch):
    w = MainWindow()
    m = w._model
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 5.0, 0.0)}
    m.materials = {1: Material(id=1, name="S", E=2e8, density=7850.0)}
    m.sections = {1: Section(id=1, name="bar", material_id=1, A=0.01)}
    m.elements = [TrussElement2D(id=7, node_i=1, node_j=2, E=2e8, A=0.01,
                                 section_id=1)]

    warned = {}

    def fake_warning(parent, title, text):
        warned["text"] = text

    monkeypatch.setattr(
        "structural_analysis.gui_qt.app.QMessageBox.warning", fake_warning,
    )
    w._show_precast_stages(elem_id=7)
    assert "truss" in warned.get("text", "").lower()
    assert w._precast_window is None  # no window opened for a truss


def test_menu_trigger_bool_is_treated_as_no_target(qt_app):
    """Regression: QAction.triggered passes checked=False, which PyQt
    feeds into elem_id. The handler must treat a bool as 'use selection',
    not look up element id False ('Element False not found')."""
    w = MainWindow()
    eid = _seed_frame(w)
    w.canvas.select_element(eid)
    # Simulate the menu action firing with the checked bool.
    w._show_precast_stages(False)
    assert w._precast_window is not None
    assert w._precast_window._member is not None
    assert w._precast_window._member.elem_id == eid


def test_window_does_not_mutate_main_model(qt_app):
    w = MainWindow()
    eid = _seed_frame(w)
    before = copy.deepcopy(w._model)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    # Exercise several stages / inputs.
    win._daf.setValue(1.4)
    win._suction.setValue(2.0)
    qt_app.processEvents()
    win._stage_combo.setCurrentIndex(1)
    qt_app.processEvents()

    assert len(w._model.elements) == len(before.elements)
    assert w._model.elements[0].member_loads == before.elements[0].member_loads
    assert w._model.nodes.keys() == before.nodes.keys()
