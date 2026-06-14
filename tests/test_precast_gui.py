"""PyQt6 smoke tests for the precast handling-stage tool window.

Run under the offscreen Qt platform plugin so they work headless. The
pure statics are exercised in ``test_precast.py``; here we only confirm
the window constructs, drives the engine across all three stages on a
single sheet, rejects a truss selection clearly, and never mutates the
main model.
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
    STAGE_LIFTING,
    STAGE_STOCK,
    STAGE_TRUCK,
    STAGES,
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


def test_window_constructs_with_all_three_stages_on_one_sheet(qt_app):
    w = MainWindow()
    eid = _seed_frame(w)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    # All three stages computed and rendered together — no stage selector.
    assert set(win._rows.keys()) == set(STAGES)
    assert set(win._last_results.keys()) == set(STAGES)
    for key in STAGES:
        assert len(win._last_results[key].reactions) == 2
    # Only the lifting row owns the sling and suction controls.
    assert win._rows[STAGE_LIFTING].sling_angle is not None
    assert win._rows[STAGE_STOCK].sling_angle is None
    assert win._rows[STAGE_TRUCK].sling_angle is None
    assert "Element 1" in win._member_label.text()


def test_auto_space_seeds_per_stage_defaults(qt_app):
    w = MainWindow()
    eid = _seed_frame(w, L=10.0)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    # Truck stage: 0.1L / 0.9L; stock / lifting: 0.2L / 0.8L.
    assert win._rows[STAGE_LIFTING].p1.value() == pytest.approx(2.0)
    assert win._rows[STAGE_LIFTING].p2.value() == pytest.approx(8.0)
    assert win._rows[STAGE_TRUCK].p1.value() == pytest.approx(1.0)
    assert win._rows[STAGE_TRUCK].p2.value() == pytest.approx(9.0)

    # Edit truck positions, then click Auto-space → defaults restored.
    win._rows[STAGE_TRUCK].p1.setValue(3.0)
    win._rows[STAGE_TRUCK].p2.setValue(6.0)
    qt_app.processEvents()
    win._rows[STAGE_TRUCK].auto_btn.click()
    qt_app.processEvents()
    assert win._rows[STAGE_TRUCK].p1.value() == pytest.approx(1.0)
    assert win._rows[STAGE_TRUCK].p2.value() == pytest.approx(9.0)


def test_global_daf_change_rescales_all_stages(qt_app):
    w = MainWindow()
    eid = _seed_frame(w)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    base = {k: win._last_results[k].total_load for k in STAGES}
    win._daf.setValue(1.5)
    qt_app.processEvents()
    for k in STAGES:
        assert win._last_results[k].total_load == pytest.approx(1.5 * base[k])


def test_row_position_edit_triggers_recompute(qt_app):
    w = MainWindow()
    eid = _seed_frame(w, L=8.0)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    before = win._last_results[STAGE_STOCK].reactions
    win._rows[STAGE_STOCK].p1.setValue(0.5)
    win._rows[STAGE_STOCK].p2.setValue(7.5)
    qt_app.processEvents()
    after = win._last_results[STAGE_STOCK].reactions
    assert after != before
    # Other stages unaffected by editing only stock points.
    assert (win._last_results[STAGE_LIFTING].reactions[0][0]
            == pytest.approx(0.2 * 8.0))


def test_copy_report_includes_all_three_stages(qt_app):
    w = MainWindow()
    eid = _seed_frame(w)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    win._copy_report()
    cb = QApplication.clipboard()
    txt = cb.text()
    assert "Precast Handling Stages" in txt
    assert "Lifting" in txt
    assert "Stock" in txt
    assert "Truck" in txt
    assert "display-only" in txt


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
    # Exercise several inputs across rows.
    win._daf.setValue(1.4)
    win._rows[STAGE_LIFTING].suction.setValue(2.0)
    win._rows[STAGE_STOCK].p1.setValue(1.0)
    qt_app.processEvents()

    assert len(w._model.elements) == len(before.elements)
    assert w._model.elements[0].member_loads == before.elements[0].member_loads
    assert w._model.nodes.keys() == before.nodes.keys()
