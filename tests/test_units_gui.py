"""PyQt6 smoke tests for the Global Units V1 selector.

Verifies:
- the View → Units submenu is wired and synced with a status-bar combo;
- switching the preset re-renders the result text panel and the diagram
  label without mutating the internal model or AnalysisResult;
- switching back to the default restores the original byte-for-byte
  output (no double conversion);
- QSettings persists the chosen preset across MainWindow instances.
"""

from __future__ import annotations

import copy
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QSettings  # noqa: E402
    from PyQt6.QtWidgets import QApplication  # noqa: E402
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 unavailable: {exc}", allow_module_level=True)

from structural_analysis.element import FrameElement2D  # noqa: E402
from structural_analysis.gui_common import units as U  # noqa: E402
from structural_analysis.gui_qt.app import MainWindow  # noqa: E402
from structural_analysis.model import (  # noqa: E402
    Material, Node, Section, Support, NodalLoad,
)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _seed_solved(w: MainWindow) -> None:
    m = w._model
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 5.0, 0.0)}
    m.materials = {1: Material(id=1, name="C", E=3e7, density=0.0)}
    m.sections = {1: Section(id=1, name="S", material_id=1,
                             A=0.05, I=1.0e-3, depth=0.3)}
    m.elements = [FrameElement2D(id=1, node_i=1, node_j=2, E=3e7,
                                 A=0.05, I=1.0e-3, section_id=1)]
    m.supports = {1: Support(node_id=1, ux=True, uy=True, rz=True),
                  2: Support(node_id=2, ux=False, uy=True, rz=False)}
    m.nodal_loads = [NodalLoad(node_id=2, fx=10.0)]
    w._run_static_solve(active_only=False)


@pytest.fixture(autouse=True)
def _clean_qsettings():
    """Reset the units-preset QSetting before & after each test so the
    persistence test can rely on a known starting state."""
    s = QSettings("CE4011", "StructuralAnalysis")
    s.remove("units_preset")
    yield
    s.remove("units_preset")


def test_units_menu_and_status_combo_exist_with_default_kN_m(qt_app):
    w = MainWindow()
    assert w._units_preset == "kN_m"
    # Menu: all 15 presets present, the current one checked.
    assert len(w._units_actions) == 15
    assert w._units_actions["kN_m"].isChecked()
    # Status combo: same population + currentData matches.
    assert w._units_combo.count() == 15
    assert w._units_combo.currentData() == "kN_m"


def test_switching_preset_updates_result_text_and_diagram_label(qt_app):
    w = MainWindow()
    _seed_solved(w)
    default_text = w._result_text.toPlainText()
    assert "Rx (kN)" in default_text

    w._set_units_preset("kip_ft")
    text_kip = w._result_text.toPlainText()
    assert "Rx (kip)" in text_kip
    assert "Mz (kip·ft)" in text_kip
    # Canvas got the new preset too.
    assert w.canvas._units_preset == "kip_ft"
    # Status bar reports the switch with the "internal kN-m" reminder.
    assert "Internal model remains kN-m" in w._status_label.text()
    assert "kip, ft" in w._status_label.text()


def test_switching_back_restores_byte_identical_output(qt_app):
    """No double-conversion: cycling through several presets and back to
    the default reproduces the original result text exactly."""
    w = MainWindow()
    _seed_solved(w)
    original = w._result_text.toPlainText()
    for pid in ("kgf_m", "N_mm", "kip_ft", "tf_cm", "lbf_ft"):
        w._set_units_preset(pid)
    w._set_units_preset("kN_m")
    assert w._result_text.toPlainText() == original


def test_switching_preset_does_not_mutate_model_or_result(qt_app):
    w = MainWindow()
    _seed_solved(w)
    model_before = copy.deepcopy(w._model)
    reactions_before = {nid: dict(d) for nid, d in w._result.reactions.items()}
    D_before = None if w._result.D is None else list(w._result.D)

    for pid in ("kgf_m", "kip_ft", "MN_m", "kN_m"):
        w._set_units_preset(pid)

    assert w._model.nodal_loads == model_before.nodal_loads
    assert w._model.nodes.keys() == model_before.nodes.keys()
    for nid, nd in model_before.nodes.items():
        assert w._model.nodes[nid].x == nd.x
        assert w._model.nodes[nid].y == nd.y
    assert {nid: dict(d) for nid, d in w._result.reactions.items()} == \
        reactions_before
    D_after = None if w._result.D is None else list(w._result.D)
    assert D_after == D_before


def test_menu_action_and_combo_stay_in_sync(qt_app):
    w = MainWindow()
    # User picks via combo → menu radio reflects it.
    idx = w._units_combo.findData("tf_m")
    w._units_combo.setCurrentIndex(idx)
    qt_app.processEvents()
    assert w._units_preset == "tf_m"
    assert w._units_actions["tf_m"].isChecked()
    # User picks via menu → combo reflects it.
    w._units_actions["lbf_in"].trigger()
    qt_app.processEvents()
    assert w._units_preset == "lbf_in"
    assert w._units_combo.currentData() == "lbf_in"


def test_qsettings_persists_preset_across_window_instances(qt_app):
    w1 = MainWindow()
    w1._set_units_preset("kgf_m")
    # Fresh MainWindow should pick the saved preference up.
    w2 = MainWindow()
    assert w2._units_preset == "kgf_m"
    assert w2._units_combo.currentData() == "kgf_m"
    assert w2._units_actions["kgf_m"].isChecked()


def test_unknown_qsettings_value_falls_back_to_default(qt_app):
    QSettings("CE4011", "StructuralAnalysis").setValue(
        "units_preset", "junk_unit")
    w = MainWindow()
    assert w._units_preset == U.DEFAULT_PRESET_ID


def test_member_end_force_readout_converts_with_preset(qt_app, monkeypatch):
    """The element free-body / local-end-force readout (a result surface)
    converts to the active preset."""
    w = MainWindow()
    _seed_solved(w)

    captured = {}

    class _FakeBox:
        def __init__(self, *a, **k):
            pass

        def setWindowTitle(self, *_a):
            pass

        def setText(self, *_a):
            pass

        def setInformativeText(self, *_a):
            pass

        def setDetailedText(self, text):
            captured["detail"] = text

        def exec(self):
            return 0

    import structural_analysis.gui_qt.app as appmod
    monkeypatch.setattr(appmod, "QMessageBox", _FakeBox)

    w._set_units_preset("kip_ft")
    w._show_element_results(1)
    detail = captured.get("detail", "")
    assert "kip" in detail
    assert "kN" not in detail   # fully converted, no stray internal unit


def test_load_render_labels_stay_kN_not_misleading(qt_app):
    """V1 must not relabel un-converted load annotations. Even in kip
    mode the canvas load tags must read kN / kN·m / kN/m so the user is
    never misled into thinking loads were converted."""
    from structural_analysis.gui_qt.canvas import (
        _label_for_udl, _label_for_pointload,
    )
    from structural_analysis.model import UniformDistributedLoad, PointLoad
    udl = UniformDistributedLoad(wy=10.0, coord_system="gravity")
    pl = PointLoad(py=15.0, a=2.0, coord_system="gravity")
    assert "kN/m" in _label_for_udl(udl)
    assert "kN" in _label_for_pointload(pl)
    # Switching the global preset doesn't touch these labels (they are
    # not preset-aware — they are always internal kN).
    w = MainWindow()
    w._set_units_preset("kip_ft")
    assert "kN/m" in _label_for_udl(udl)
    assert "kN" in _label_for_pointload(pl)


def test_diagram_hover_readout_converts_value_but_keeps_x_in_m(qt_app):
    w = MainWindow()
    _seed_solved(w)
    w.canvas.diagram_kind = "axial"
    w._set_units_preset("kip_ft")
    # Build a synthetic hit on element 1 near its midpoint.
    from structural_analysis.gui_qt.canvas import HitResult
    hit = HitResult(x=2.5, y=0.0, element_id=1)
    txt = w._diagram_value_text_for_hit(hit)
    assert txt is not None
    assert "kip" in txt          # value converted
    assert "@ x=" in txt and " m " in txt   # coordinate stays metres

