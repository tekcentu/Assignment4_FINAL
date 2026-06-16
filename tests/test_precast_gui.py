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


def test_per_stage_daf_rescales_only_that_stage(qt_app):
    w = MainWindow()
    eid = _seed_frame(w)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    base = {k: win._last_results[k].total_load for k in STAGES}
    # DAF now lives per stage row, not as a global control.
    assert not hasattr(win, "_daf")
    win._rows[STAGE_STOCK].daf.setValue(1.5)
    qt_app.processEvents()
    assert win._last_results[STAGE_STOCK].total_load == pytest.approx(
        1.5 * base[STAGE_STOCK])
    # Other stages keep their own DAF (unchanged).
    assert win._last_results[STAGE_LIFTING].total_load == pytest.approx(
        base[STAGE_LIFTING])
    assert win._last_results[STAGE_TRUCK].total_load == pytest.approx(
        base[STAGE_TRUCK])


def test_high_daf_emits_soft_warning_and_warning_chip(qt_app):
    w = MainWindow()
    eid = _seed_frame(w)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    win._rows[STAGE_STOCK].daf.setValue(2.5)
    qt_app.processEvents()
    res = win._last_results[STAGE_STOCK]
    assert any("unusually high" in m for m in res.warnings)
    assert "WARNING" in win._rows[STAGE_STOCK].status_chip.text()


def test_disabled_stage_greys_out_and_is_skipped(qt_app):
    w = MainWindow()
    eid = _seed_frame(w)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    assert STAGE_STOCK in win._last_results
    win._rows[STAGE_STOCK].enabled_cb.setChecked(False)
    qt_app.processEvents()
    # Dropped from results, controls greyed, chip reads DISABLED.
    assert STAGE_STOCK not in win._last_results
    assert not win._rows[STAGE_STOCK].p1.isEnabled()
    assert not win._rows[STAGE_STOCK].daf.isEnabled()
    assert win._rows[STAGE_STOCK].status_chip.text() == "DISABLED"
    # Enabled stages still computed.
    assert STAGE_LIFTING in win._last_results
    # The copied report omits the disabled stage.
    win._copy_report()
    txt = QApplication.clipboard().text()
    assert "Stock" not in txt
    assert "Lifting" in txt
    # Re-enabling restores it.
    win._rows[STAGE_STOCK].enabled_cb.setChecked(True)
    qt_app.processEvents()
    assert STAGE_STOCK in win._last_results
    assert win._rows[STAGE_STOCK].p1.isEnabled()


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
    assert "Support spacing" in txt


def test_cracking_check_appears_in_summary_and_report(qt_app):
    """End-to-end: stress check runs for the seeded frame and the result
    shows up in each stage's stress label and in the copied report."""
    w = MainWindow()
    eid = _seed_frame(w, L=8.0)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    # Force a cracking warning by lowering the allowable.
    win._allowable_tensile.setValue(0.05)
    qt_app.processEvents()
    for key in STAGES:
        text = win._rows[key].stress.text()
        assert "σ_bot_max" in text or "σ_top_max" in text
    # Some stage hits "CRACKING WARNING" with this low allowable.
    assert any(
        "CRACKING WARNING" in win._rows[k].stress.text() for k in STAGES
    )
    win._copy_report()
    cb_text = QApplication.clipboard().text()
    assert "Flexural cracking check" in cb_text
    assert "Allowable tensile stress" in cb_text
    assert "CRACKING WARNING" in cb_text


def test_stage_sketch_draws_member_udl_reactions_and_slings(qt_app):
    """The in-dialog 2D sketch must show the member line, UDL load band,
    upward reaction arrows + values, and (lifting only) sling lines with
    T / H labels, plus populated V and M diagrams."""
    w = MainWindow()
    eid = _seed_frame(w, L=8.0)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)

    lift = win._rows[STAGE_LIFTING]
    member_texts = [t.get_text() for t in lift._ax_member.texts]
    # UDL band label and reaction value labels.
    assert any(s.startswith("w =") for s in member_texts)
    assert any(s.startswith("R=") for s in member_texts)
    # Sling tension / horizontal labels on the lifting row.
    assert any("T=" in s and "H=" in s for s in member_texts)
    # The member line itself is drawn.
    assert len(lift._ax_member.lines) >= 1
    # V and M diagrams are populated.
    assert len(lift._ax_v.lines) >= 1
    assert len(lift._ax_m.lines) >= 1
    # The OK / WARNING chip is populated for an enabled, valid stage.
    assert lift.status_chip.text() in ("OK", "WARNING")

    # A non-lifting stage shows reactions but no sling T / H labels.
    stock_texts = [t.get_text() for t in win._rows[STAGE_STOCK]._ax_member.texts]
    assert any(s.startswith("R=") for s in stock_texts)
    assert not any("T=" in s for s in stock_texts)


def test_vm_xaxis_tracks_member_length_after_target_switch(qt_app):
    """Regression: switching from a long member to a short one used to
    leave the V / M x-axis stuck at the previous member's length (so a
    4 m member showed a 0–10 m axis with empty space on the right)."""
    w = MainWindow()
    m = w._model
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 10.0, 0.0),
               3: Node(3, 0.0, 0.0), 4: Node(4, 4.0, 0.0)}
    m.materials = {1: Material(id=1, name="C", E=3e7, density=2400.0)}
    m.sections = {1: Section(id=1, name="S", material_id=1, A=0.2, I=0.05,
                             depth=0.4)}
    m.elements = [
        FrameElement2D(id=1, node_i=1, node_j=2, E=3e7, A=0.2, I=0.05,
                       rho=2400.0, section_id=1),
        FrameElement2D(id=2, node_i=3, node_j=4, E=3e7, A=0.2, I=0.05,
                       rho=2400.0, section_id=1),
    ]
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(1)              # L = 10 m
    win.set_target(2)              # L = 4 m
    for key in STAGES:
        x0, x1 = win._rows[key]._ax_v.get_xlim()
        assert x1 <= 4.2, f"V axis on {key} extends to {x1}, expected ≤ 4.2"
        assert x0 >= -0.2
        x0, x1 = win._rows[key]._ax_m.get_xlim()
        assert x1 <= 4.2, f"M axis on {key} extends to {x1}, expected ≤ 4.2"


def test_manual_y_toggle_enables_y_spinboxes(qt_app):
    w = MainWindow()
    eid = _seed_frame(w, L=8.0)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    assert not win._y_top.isEnabled()
    win._manual_y.setChecked(True)
    qt_app.processEvents()
    assert win._y_top.isEnabled()
    assert win._y_bottom.isEnabled()
    # Disabling the check disables the y boxes regardless of the manual toggle.
    win._stress_enabled.setChecked(False)
    qt_app.processEvents()
    assert not win._y_top.isEnabled()
    assert not win._allowable_tensile.isEnabled()


def test_summary_shows_support_spacing(qt_app):
    w = MainWindow()
    eid = _seed_frame(w, L=8.0)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    # Stock auto-spaces to 0.2L / 0.8L → spacing 0.6·8 = 4.8 m.
    assert "Support spacing = 4.8 m" in win._rows[STAGE_STOCK].summary.text()


def test_spin_boxes_ignore_mouse_wheel(qt_app):
    """Scrolling the sheet must not nudge spin-box values — the wheel
    event is ignored so it bubbles up to the scroll area."""
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent

    w = MainWindow()
    eid = _seed_frame(w, L=8.0)
    win = PrecastHandlingWindow(w, lambda: w._model)
    win.set_target(eid)
    sp = win._rows[STAGE_STOCK].p1
    before = sp.value()
    ev = QWheelEvent(
        QPointF(5, 5), QPointF(5, 5),
        QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    sp.wheelEvent(ev)
    qt_app.processEvents()
    assert sp.value() == pytest.approx(before)
    assert not ev.isAccepted()


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
    win._rows[STAGE_LIFTING].daf.setValue(1.4)
    win._rows[STAGE_LIFTING].suction.setValue(2.0)
    win._rows[STAGE_STOCK].p1.setValue(1.0)
    qt_app.processEvents()

    assert len(w._model.elements) == len(before.elements)
    assert w._model.elements[0].member_loads == before.elements[0].member_loads
    assert w._model.nodes.keys() == before.nodes.keys()
