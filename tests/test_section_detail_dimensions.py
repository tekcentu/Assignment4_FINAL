"""Element Details — section preview shows measured dimensions.

Before this change the detail-view section thumbnail drew only the
outline plus a shape-name caption, so the user couldn't read off the
section sizes (and manual sections showed an unlabelled fallback shape).
The thumbnail now annotates b / h (and tf / tw for I-sections), and the
manual √A fallback labels its own side — matching the Add-Section live
preview.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from structural_analysis.model import (
    StructuralModel, Node, Material, Section, Support, NodalLoad,
)
from structural_analysis.element import FrameElement2D


@pytest.fixture
def qt_app():
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PyQt6 unavailable: {exc}")
    return QApplication.instance() or QApplication([])


def _model_with_section(section: Section) -> StructuralModel:
    m = StructuralModel(title="section detail")
    m.materials = {1: Material(1, E=200_000.0, alpha=1e-5)}
    m.sections = {1: section}
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 4.0, 0.0)}
    m.elements = [FrameElement2D(
        1, 1, 2, E=200_000.0, A=section.A or 0.02, I=section.I or 0.08,
        section_id=1,
    )]
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    m.nodal_loads = [NodalLoad(node_id=2, fy=-10.0)]
    return m


def _section_texts(qt_app, section: Section) -> str:
    from matplotlib.figure import Figure
    from structural_analysis.gui_qt.element_graphics import draw_element_detail
    m = _model_with_section(section)
    fig, sec = Figure(), Figure()
    axes = draw_element_detail(fig, m.elements[0], m, None, section_fig=sec)
    return " ".join(t.get_text() for t in axes["section"].texts)


# ── known shapes ─────────────────────────────────────────────────────────


def test_rectangle_section_shows_b_and_h(qt_app):
    s = Section(id=1, material_id=1, shape_type="rectangle",
                b=0.30, h=0.50, A=0.15, I=3.125e-3, depth=0.50, width=0.30)
    texts = _section_texts(qt_app, s)
    assert "b = 0.3 m" in texts
    assert "h = 0.5 m" in texts


def test_square_section_shows_dimensions(qt_app):
    s = Section(id=1, material_id=1, shape_type="square",
                b=0.40, h=0.40, A=0.16, I=2.133e-3, depth=0.40, width=0.40)
    texts = _section_texts(qt_app, s)
    assert "b = 0.4 m" in texts
    assert "h = 0.4 m" in texts


def test_i_section_shows_flange_and_web_thickness(qt_app):
    s = Section(id=1, material_id=1, shape_type="i_section",
                b=0.20, h=0.40, tf=0.02, tw=0.012,
                A=0.0136, I=3.6e-4, depth=0.40, width=0.20)
    texts = _section_texts(qt_app, s)
    assert "b = 0.2 m" in texts
    assert "h = 0.4 m" in texts
    assert "tf = 0.02" in texts
    assert "tw = 0.012" in texts


# ── manual fallback ───────────────────────────────────────────────────────


def test_manual_section_labels_equivalent_square_side(qt_app):
    """The user's complaint: manual sections used to draw a fallback
    shape with no measures. Now the √A side is annotated."""
    # A = 0.25 → √A = 0.5 m.
    s = Section(id=1, material_id=1, shape_type="manual",
                A=0.25, I=5.2e-3, depth=0.5, width=0.5)
    texts = _section_texts(qt_app, s)
    assert "0.5 m" in texts          # the √A side length is shown
    assert "√A" in texts             # honest note it's area-equivalent


def test_manual_section_zero_area_does_not_crash(qt_app):
    s = Section(id=1, material_id=1, shape_type="manual", A=0.0)
    # Should render the fallback note without raising.
    texts = _section_texts(qt_app, s)
    assert isinstance(texts, str)


# ── no section ────────────────────────────────────────────────────────────


def test_element_without_section_still_shows_placeholder(qt_app):
    from matplotlib.figure import Figure
    from structural_analysis.gui_qt.element_graphics import draw_element_detail
    m = StructuralModel(title="no section")
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 3.0, 0.0)}
    m.elements = [FrameElement2D(1, 1, 2, E=200_000.0, A=0.02, I=0.08)]
    m.supports = {1: Support(1, ux=True, uy=True, rz=True)}
    fig, sec = Figure(), Figure()
    axes = draw_element_detail(fig, m.elements[0], m, None, section_fig=sec)
    texts = " ".join(t.get_text() for t in axes["section"].texts)
    assert "no section" in texts.lower()
    # No spurious dimension labels when there's nothing to measure.
    assert "b =" not in texts and "h =" not in texts


# ── dimensions match the drawn outline bounding box ───────────────────────


def test_annotated_width_matches_drawn_outline(qt_app):
    """The b/h labels are sourced so they always equal the drawn
    outline extents (regression against label/drawing drift)."""
    from matplotlib.figure import Figure
    from structural_analysis.gui_qt.element_graphics import draw_element_detail
    s = Section(id=1, material_id=1, shape_type="rectangle",
                b=0.30, h=0.50, A=0.15, I=3.125e-3, depth=0.50, width=0.30)
    m = _model_with_section(s)
    fig, sec = Figure(), Figure()
    axes = draw_element_detail(fig, m.elements[0], m, None, section_fig=sec)
    # The filled outline patch spans b along z (x) and h along y.
    poly = [p for p in axes["section"].patches]
    assert poly, "expected a filled section outline"
    verts = poly[0].get_xy()
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    assert max(xs) - min(xs) == pytest.approx(0.30, abs=1e-9)
    assert max(ys) - min(ys) == pytest.approx(0.50, abs=1e-9)


# ── shared helper: detail view and dialog preview don't drift ─────────────


def test_detail_and_dialog_share_one_dimension_helper(qt_app):
    """Both the Element-Details thumbnail and the Add-Section dialog
    preview must route their b/h labels through the SAME helper
    (element_graphics.annotate_section_dimensions), so the two panels
    can't drift. Verify the helper exists and that, given the same
    section geometry, both render matching 'b =' / 'h =' numbers."""
    from matplotlib.figure import Figure
    from structural_analysis.gui_qt import element_graphics

    # The single source of truth is a public callable.
    assert callable(getattr(element_graphics, "annotate_section_dimensions"))

    # Drive the helper directly the way each call site does and confirm
    # the placement geometry + number formatting is identical (only the
    # cosmetic units suffix differs by design).
    fig_detail, fig_dialog = Figure(), Figure()
    ax_detail = fig_detail.add_subplot(111)
    ax_dialog = fig_dialog.add_subplot(111)
    element_graphics.annotate_section_dimensions(
        ax_detail, b=0.30, h=0.50, units=" m")        # detail view
    element_graphics.annotate_section_dimensions(
        ax_dialog, b=0.30, h=0.50, units="")           # dialog preview
    detail_txt = sorted(t.get_text() for t in ax_detail.texts)
    dialog_txt = sorted(t.get_text() for t in ax_dialog.texts)
    assert detail_txt == ["b = 0.3 m", "h = 0.5 m"]
    assert dialog_txt == ["b = 0.3", "h = 0.5"]


def _all_labels_fit(fig, ax) -> bool:
    fig.canvas.draw()
    ax_bb = ax.get_window_extent()
    for t in ax.texts:
        bb = t.get_window_extent()
        if not (bb.x0 >= ax_bb.x0 - 1 and bb.x1 <= ax_bb.x1 + 1
                and bb.y0 >= ax_bb.y0 - 1 and bb.y1 <= ax_bb.y1 + 1):
            return False
    return True


@pytest.mark.parametrize("b,h,tf,tw,figsize", [
    (0.4, 0.4, 0.0, 0.0, (1.6, 1.6)),    # square — the reported overflow case
    (0.6, 0.3, 0.0, 0.0, (1.6, 1.6)),    # wide
    (0.2, 0.5, 0.0, 0.0, (1.3, 2.4)),    # tall, narrow panel
    (0.3, 0.4, 0.02, 0.012, (1.6, 1.6)),  # I-section with tf/tw
    (900.0, 500.0, 0.0, 0.0, (2.4, 1.3)),  # large values, wide panel
])
def test_section_dimension_labels_fit_in_panel(qt_app, b, h, tf, tw, figsize):
    """Every dimension label must stay inside the panel — the square case
    used to overflow the right edge (the 'h = …' label was unreadable)."""
    from matplotlib.figure import Figure
    from structural_analysis.gui_qt.element_graphics import (
        annotate_section_dimensions,
    )
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    ax.fill([-b / 2, b / 2, b / 2, -b / 2],
            [-h / 2, -h / 2, h / 2, h / 2])
    annotate_section_dimensions(ax, b=b, h=h, tf=tf, tw=tw, units=" m")
    assert _all_labels_fit(fig, ax), (
        f"labels overflow for b={b} h={h} tf={tf} tw={tw} fig={figsize}"
    )


def test_dimension_helper_fallback_and_i_section_formatting(qt_app):
    """The shared helper covers the manual √A fallback (≈ prefix) and the
    I-section tf/tw labels — the cases each call site relies on."""
    from matplotlib.figure import Figure
    from structural_analysis.gui_qt.element_graphics import (
        annotate_section_dimensions,
    )
    fig = Figure()
    ax = fig.add_subplot(111)
    annotate_section_dimensions(ax, b=0.5, h=0.5, fallback=True, units=" m")
    txt = sorted(t.get_text() for t in ax.texts)
    assert txt == ["≈ 0.5 m", "≈ 0.5 m"]

    fig2 = Figure()
    ax2 = fig2.add_subplot(111)
    annotate_section_dimensions(ax2, b=0.2, h=0.4, tf=0.02, tw=0.012, units="")
    txt2 = sorted(t.get_text() for t in ax2.texts)
    assert txt2 == ["b = 0.2", "h = 0.4", "tf = 0.02", "tw = 0.012"]
