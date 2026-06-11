"""Rigid end offsets — visualization and dialog round trips (Qt).

Pure-Python assertions live in test_rigid_offsets_{data,solver,loads};
this file covers the canvas rigid-zone strokes, the physical-view
hatching, the detail-sketch note and the properties dialog.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.model import StructuralModel, Node, Support, NodalLoad
from structural_analysis.element import FrameElement2D

E, A, I = 200_000.0, 0.02, 0.08
_RIGID_COLOR = "#4d4d4d"


@pytest.fixture
def qt_app():
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PyQt6 unavailable: {exc}")
    app = QApplication.instance() or QApplication([])
    yield app


def _offset_model(ei=1.0, ej=0.5):
    m = StructuralModel(title="offsets gui")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 6.0, 0.0)}
    m.elements = [FrameElement2D(
        1, 1, 2, E=E, A=A, I=I, offset_i=ei, offset_j=ej, depth=0.3,
    )]
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    m.nodal_loads = [NodalLoad(node_id=2, fy=-10.0)]
    return m


def _canvas_for(qt_app, model):
    from structural_analysis.gui_qt.canvas import ModelCanvas
    canvas = ModelCanvas(None, model_provider=lambda: model)
    return canvas


# ── canvas centerline marking ────────────────────────────────────────────


def test_rigid_zones_drawn_thick_and_distinct_on_centerline(qt_app):
    m = _offset_model()
    canvas = _canvas_for(qt_app, m)
    canvas.redraw()
    rigid_lines = [
        ln for ln in canvas.ax.lines
        if ln.get_color() == _RIGID_COLOR and ln.get_linewidth() > 4.0
    ]
    assert rigid_lines, (
        "expected a thick dark rigid-zone stroke on the centerline view"
    )
    # The stubs cover [0, e_i] and [L−e_j, L]: collect drawn x extents.
    xs = []
    for ln in rigid_lines:
        xs.extend(x for x in ln.get_xdata() if x is not None)
    finite = [float(x) for x in xs if x == x]  # drop None/NaN separators
    assert min(finite) == pytest.approx(0.0)
    assert max(finite) == pytest.approx(6.0)


def test_no_rigid_stroke_without_offsets(qt_app):
    m = _offset_model(ei=0.0, ej=0.0)
    canvas = _canvas_for(qt_app, m)
    canvas.redraw()
    rigid_lines = [
        ln for ln in canvas.ax.lines
        if ln.get_color() == _RIGID_COLOR and ln.get_linewidth() > 4.0
    ]
    assert not rigid_lines


def test_rigid_zone_hatched_in_physical_view(qt_app):
    m = _offset_model()
    canvas = _canvas_for(qt_app, m)
    canvas.show_physical_members = True
    canvas.redraw()
    hatched = [
        c for c in canvas.ax.collections
        if getattr(c, "get_hatch", None) and c.get_hatch() == "xx"
    ]
    assert hatched, "expected hatched rigid-zone bodies in physical view"


# ── detail sketch note ───────────────────────────────────────────────────


def test_detail_sketch_shows_offset_note(qt_app):
    from matplotlib.figure import Figure
    from structural_analysis.gui_qt.element_graphics import (
        draw_element_detail,
    )
    m = _offset_model()
    fig = Figure()
    sec = Figure()
    axes = draw_element_detail(fig, m.elements[0], m, None, section_fig=sec)
    texts = " ".join(t.get_text() for t in axes["sketch"].texts)
    assert "rigid offsets" in texts
    assert "flexible span" in texts


def test_detail_sketch_no_note_without_offsets(qt_app):
    from matplotlib.figure import Figure
    from structural_analysis.gui_qt.element_graphics import (
        draw_element_detail,
    )
    m = _offset_model(ei=0.0, ej=0.0)
    fig = Figure()
    sec = Figure()
    axes = draw_element_detail(fig, m.elements[0], m, None, section_fig=sec)
    texts = " ".join(t.get_text() for t in axes["sketch"].texts)
    assert "rigid offsets" not in texts


# ── properties dialog ────────────────────────────────────────────────────


def test_element_dialog_round_trips_offsets(qt_app):
    from structural_analysis.gui_qt.dialogs import ElementDialog
    from structural_analysis.model import Material, Section

    m = StructuralModel(title="dlg")
    m.materials = {1: Material(1, E=E, alpha=1e-5)}
    m.sections = {1: Section(1, material_id=1, A=A, I=I, depth=0.3)}
    d = ElementDialog(
        None, model=m,
        existing_kind="frame", existing_section_id=1,
        existing_offset_i=0.4, existing_offset_j=0.3,
        member_length=6.0,
    )
    assert d._sb_off_i.value() == pytest.approx(0.4)
    assert d._sb_off_j.value() == pytest.approx(0.3)
    rv = d._accept()
    assert rv["offset_i"] == pytest.approx(0.4)
    assert rv["offset_j"] == pytest.approx(0.3)


def test_element_dialog_rejects_consuming_offsets(qt_app):
    from structural_analysis.gui_qt.dialogs import ElementDialog
    from structural_analysis.model import Material, Section

    m = StructuralModel(title="dlg")
    m.materials = {1: Material(1, E=E, alpha=1e-5)}
    m.sections = {1: Section(1, material_id=1, A=A, I=I, depth=0.3)}
    d = ElementDialog(
        None, model=m,
        existing_kind="frame", existing_section_id=1,
        existing_offset_i=3.5, existing_offset_j=3.5,
        member_length=6.0,
    )
    with pytest.raises(ValueError, match="less than the member length"):
        d._accept()


def test_element_dialog_truss_returns_zero_offsets(qt_app):
    from structural_analysis.gui_qt.dialogs import ElementDialog
    from structural_analysis.model import Material, Section

    m = StructuralModel(title="dlg")
    m.materials = {1: Material(1, E=E, alpha=1e-5)}
    m.sections = {1: Section(1, material_id=1, A=A, I=I, depth=0.3)}
    d = ElementDialog(
        None, model=m,
        existing_kind="truss", existing_section_id=1,
        existing_offset_i=0.4, member_length=6.0,
    )
    rv = d._accept()
    assert rv["offset_i"] == 0.0
    assert rv["offset_j"] == 0.0


# ── creation-mode dialog (show_offsets=False) + L_total / L_flex ─────────


def _dlg_model():
    from structural_analysis.model import Material, Section
    m = StructuralModel(title="dlg")
    m.materials = {1: Material(1, E=E, alpha=1e-5)}
    m.sections = {1: Section(1, material_id=1, A=A, I=I, depth=0.3)}
    return m


def test_element_dialog_hides_offset_fields_in_creation_mode(qt_app):
    from structural_analysis.gui_qt.dialogs import ElementDialog
    m = _dlg_model()
    d = ElementDialog(
        None, model=m, existing_kind="frame", existing_section_id=1,
        show_offsets=False,
    )
    assert d._sb_off_i is None
    assert d._sb_off_j is None
    assert d._lbl_l_total is None
    assert d._lbl_l_flex is None
    rv = d._accept()
    # No offset spinboxes ⇒ creation result is always zero offsets.
    assert rv["offset_i"] == 0.0
    assert rv["offset_j"] == 0.0


def test_element_dialog_shows_l_total_and_l_flex_labels(qt_app):
    from structural_analysis.gui_qt.dialogs import ElementDialog
    m = _dlg_model()
    d = ElementDialog(
        None, model=m, existing_kind="frame", existing_section_id=1,
        existing_offset_i=0.5, existing_offset_j=0.25, member_length=6.0,
    )
    assert d._lbl_l_total is not None
    assert d._lbl_l_flex is not None
    assert "6" in d._lbl_l_total.text()
    # 6.0 - 0.5 - 0.25 = 5.25
    assert "5.25" in d._lbl_l_flex.text()


def test_element_dialog_l_flex_updates_live(qt_app):
    from structural_analysis.gui_qt.dialogs import ElementDialog
    m = _dlg_model()
    d = ElementDialog(
        None, model=m, existing_kind="frame", existing_section_id=1,
        existing_offset_i=0.0, existing_offset_j=0.0, member_length=6.0,
    )
    assert "6" in d._lbl_l_flex.text()
    d._sb_off_i.setValue(1.0)
    d._sb_off_j.setValue(0.5)
    assert "4.5" in d._lbl_l_flex.text()


# ── batch offset commands wired through MainWindow ───────────────────────


def _multi_frame_model_for_app():
    """L-frame: columns 1 & 3, beam id 2 between top nodes."""
    from structural_analysis.model import Material, Section
    m = StructuralModel(title="gui auto offsets")
    m.materials = {1: Material(1, E=E, alpha=1e-5, density=7850.0)}
    m.sections = {1: Section(1, material_id=1, A=A, I=I, depth=0.4)}
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 0.0, 3.0),
        3: Node(3, 6.0, 3.0),
        4: Node(4, 6.0, 0.0),
    }
    m.elements = [
        FrameElement2D(1, 1, 2, E=E, A=A, I=I, depth=0.4, section_id=1),
        FrameElement2D(2, 2, 3, E=E, A=A, I=I, depth=0.4, section_id=1),
        FrameElement2D(3, 4, 3, E=E, A=A, I=I, depth=0.4, section_id=1),
    ]
    m.supports = {
        1: Support(1, ux=True, uy=True, rz=True),
        4: Support(4, ux=True, uy=True, rz=True),
    }
    return m


def _make_window(qt_app, model):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    w._model = model
    return w


def test_app_assign_auto_offsets_applies_to_selection(qt_app):
    m = _multi_frame_model_for_app()
    w = _make_window(qt_app, m)
    w.canvas.add_element_to_selection(2)
    w._do_assign_auto_rigid_offsets(prompt_on_empty=False)
    beam = w._model.elements[1]
    assert beam.offset_i == pytest.approx(0.2, rel=1e-6)
    assert beam.offset_j == pytest.approx(0.2, rel=1e-6)


def test_app_assign_auto_offsets_no_selection_no_prompt_does_nothing(qt_app):
    m = _multi_frame_model_for_app()
    w = _make_window(qt_app, m)
    w.canvas.clear_selection()
    w._do_assign_auto_rigid_offsets(prompt_on_empty=False)
    assert all(
        e.offset_i == 0.0 and e.offset_j == 0.0
        for e in w._model.elements
    )
    # Status bar tells the user what to do.
    assert "select" in w._status_label.text().lower()


def test_app_clear_rigid_offsets_zeros_selection(qt_app):
    m = _multi_frame_model_for_app()
    m.elements[1].offset_i = 0.2
    m.elements[1].offset_j = 0.2
    w = _make_window(qt_app, m)
    w.canvas.add_element_to_selection(2)
    w._do_clear_rigid_offsets(prompt_on_empty=False)
    beam = w._model.elements[1]
    assert beam.offset_i == 0.0 and beam.offset_j == 0.0
    assert "1 element(s) cleared" in w._status_label.text()


def test_app_assign_auto_offsets_undo_redo_via_main_window(qt_app):
    m = _multi_frame_model_for_app()
    w = _make_window(qt_app, m)
    w.canvas.add_element_to_selection(2)
    w._do_assign_auto_rigid_offsets(prompt_on_empty=False)
    assert w._model.elements[1].offset_i == pytest.approx(0.2, rel=1e-6)
    w.act_undo.trigger()
    assert w._model.elements[1].offset_i == 0.0
    w.act_redo.trigger()
    assert w._model.elements[1].offset_i == pytest.approx(0.2, rel=1e-6)


def test_app_status_reports_assigned_and_skipped_counts(qt_app):
    """Status bar message reflects assigned + truss-skipped counts."""
    from structural_analysis.element import TrussElement2D
    m = _multi_frame_model_for_app()
    # Replace column 1 with a truss so it lands in the skipped bucket.
    m.elements[0] = TrussElement2D(1, 1, 2, E=E, A=A)
    w = _make_window(qt_app, m)
    w.canvas.add_element_to_selection(1)
    w.canvas.add_element_to_selection(2)
    w._do_assign_auto_rigid_offsets(prompt_on_empty=False)
    status = w._status_label.text()
    assert "1 element(s) assigned" in status
    assert "truss" in status
