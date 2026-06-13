"""Two independent grid layers (default + generated) and their toggles.

Before PR #56 follow-up, populating a structural GridSystem silently
hid the dotted default reference grid — users lost their drawing
scaffold the moment they ran "Generate grid from nodes". The two layers
are now independent visual concerns; snap behavior is unchanged.

Pure/Qt-light assertions live up top; MainWindow wiring at the bottom.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def qt_app():
    try:
        from PyQt6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PyQt6 unavailable: {exc}")
    return QApplication.instance() or QApplication([])


def _model_with_nodes():
    from structural_analysis.model import (
        StructuralModel, Node, Support, NodalLoad,
    )
    from structural_analysis.element import FrameElement2D
    m = StructuralModel(title="grid layers")
    m.nodes = {
        1: Node(1, 0.0, 0.0),
        2: Node(2, 6.0, 0.0),
        3: Node(3, 6.0, 3.0),
        4: Node(4, 0.0, 3.0),
    }
    m.elements = [
        FrameElement2D(1, 1, 2, E=200e3, A=0.02, I=0.08),
        FrameElement2D(2, 2, 3, E=200e3, A=0.02, I=0.08),
        FrameElement2D(3, 4, 3, E=200e3, A=0.02, I=0.08),
    ]
    m.supports = {1: __import__("structural_analysis.model", fromlist=["Support"]).Support(
        1, ux=True, uy=True, rz=True,
    )}
    m.nodal_loads = [NodalLoad(node_id=3, fy=-1.0)]
    return m


def _canvas(qt_app, model, grid_provider=None):
    from structural_analysis.gui_qt.canvas import ModelCanvas
    from structural_analysis.gui_qt.grid import GridSystem
    return ModelCanvas(
        None, model_provider=lambda: model,
        grid_provider=grid_provider or (lambda: GridSystem()),
    )


# ── canvas layer behavior ────────────────────────────────────────────────


def test_default_grid_visible_when_generated_is_empty(qt_app):
    canvas = _canvas(qt_app, _model_with_nodes())
    canvas.redraw()
    # ax.get_xgridlines() are matplotlib's built-in major gridlines —
    # these are what the default reference grid uses.
    assert any(ln.get_visible() for ln in canvas.ax.get_xgridlines())


def test_default_grid_remains_visible_when_generated_is_populated(qt_app):
    """The original bug: populating a GridSystem used to hide the
    default reference grid completely. It must stay drawn."""
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 6.0)],
        y_lines=[GridLine("1", 0.0), GridLine("2", 3.0)],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.redraw()
    assert any(ln.get_visible() for ln in canvas.ax.get_xgridlines())
    assert any(ln.get_visible() for ln in canvas.ax.get_ygridlines())


def test_generated_grid_can_be_hidden_independently(qt_app):
    """Toggling show_generated_grid off must remove the labeled lines
    (the colored axvline/axhline strokes) while leaving the default
    dotted grid alone."""
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 6.0)],
        y_lines=[GridLine("1", 0.0)],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.show_generated_grid = False
    canvas.redraw()
    # The generated grid is drawn as Line2D strokes coloured "#aac8ff".
    generated = [
        ln for ln in canvas.ax.lines
        if ln.get_color() == "#aac8ff"
    ]
    assert generated == []
    # The default dotted grid is still on.
    assert any(ln.get_visible() for ln in canvas.ax.get_xgridlines())


def test_default_grid_can_be_hidden_independently(qt_app):
    """Hiding the default grid leaves the labeled lines drawn."""
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 6.0)],
        y_lines=[GridLine("1", 0.0)],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.show_default_grid = False
    canvas.redraw()
    assert not any(ln.get_visible() for ln in canvas.ax.get_xgridlines())
    generated = [
        ln for ln in canvas.ax.lines
        if ln.get_color() == "#aac8ff"
    ]
    assert generated, "generated grid lines must still be drawn"


def test_both_layers_shown_default_grid_is_faded(qt_app):
    """When both are on, the default grid is drawn lighter so the
    generated grid stays visually dominant."""
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0)], y_lines=[GridLine("1", 0.0)],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.redraw()
    # When generated grid is drawn, the default's alpha must be < 1.
    default_alpha = next(
        ln.get_alpha() for ln in canvas.ax.get_xgridlines()
        if ln.get_visible()
    )
    assert default_alpha is not None and default_alpha < 1.0


def test_both_layers_hidden_draws_no_grid(qt_app):
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(x_lines=[GridLine("A", 0.0)], y_lines=[])
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.show_default_grid = False
    canvas.show_generated_grid = False
    canvas.redraw()
    assert not any(ln.get_visible() for ln in canvas.ax.get_xgridlines())
    assert [ln for ln in canvas.ax.lines
            if ln.get_color() == "#aac8ff"] == []


# ── generated grid coordinates appear on the spine (constant values) ─────


def _locator_locs(locator):
    from matplotlib.ticker import FixedLocator
    assert isinstance(locator, FixedLocator)
    return sorted(float(v) for v in locator.locs)


def test_visible_generated_grid_puts_its_coords_on_the_axes(qt_app):
    """When the generated grid is shown, the spine ticks land on the
    structural grid-line coordinates (e.g. 0, 3, 6 / 0, 3.2) — their
    constant values — instead of the default adaptive spacing."""
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 3.0), GridLine("C", 6.0)],
        y_lines=[GridLine("1", 0.0), GridLine("2", 3.2)],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.redraw()
    assert _locator_locs(canvas.ax.xaxis.get_major_locator()) == [0.0, 3.0, 6.0]
    assert _locator_locs(canvas.ax.yaxis.get_major_locator()) == [0.0, 3.2]


def test_generated_coords_shown_even_when_default_grid_hidden(qt_app):
    """The reported case: hiding the default grid must not revert the
    spine to the default 'every 2.5 m' spacing — it still shows the
    generated coordinates."""
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 3.0)],
        y_lines=[GridLine("1", 0.0)],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.show_default_grid = False
    canvas.redraw()
    assert _locator_locs(canvas.ax.xaxis.get_major_locator()) == [0.0, 3.0]


def test_hidden_generated_grid_reverts_to_adaptive_ticks(qt_app):
    from structural_analysis.gui_qt.canvas import AdaptiveGridLocator
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 3.0)], y_lines=[],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.show_generated_grid = False
    canvas.redraw()
    assert isinstance(
        canvas.ax.xaxis.get_major_locator(), AdaptiveGridLocator
    )


def test_axis_without_generated_lines_stays_adaptive(qt_app):
    """A grid that only defines X lines must leave the Y spine on the
    adaptive locator, not collapse it to a single tick."""
    from structural_analysis.gui_qt.canvas import AdaptiveGridLocator
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    from matplotlib.ticker import FixedLocator
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 3.0)], y_lines=[],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.redraw()
    assert isinstance(
        canvas.ax.xaxis.get_major_locator(), FixedLocator
    )
    assert isinstance(
        canvas.ax.yaxis.get_major_locator(), AdaptiveGridLocator
    )


def test_empty_grid_uses_adaptive_on_both_axes(qt_app):
    from structural_analysis.gui_qt.canvas import AdaptiveGridLocator
    canvas = _canvas(qt_app, _model_with_nodes())
    canvas.redraw()
    assert isinstance(
        canvas.ax.xaxis.get_major_locator(), AdaptiveGridLocator
    )
    assert isinstance(
        canvas.ax.yaxis.get_major_locator(), AdaptiveGridLocator
    )


# ── generated-grid letter labels stick to the top/right spine ────────────


def _generated_letter_texts(canvas):
    """Letter labels for the X+Y generated grid (color "#3060c0")."""
    return [t for t in canvas.ax.texts if t.get_color() == "#3060c0"]


def test_x_line_letter_labels_use_xaxis_transform(qt_app):
    """The X-line letters must use the get_xaxis_transform() mixed
    transform (x: data, y: axes) so they always sit on the top spine
    regardless of the current view interval. This is the deterministic
    pure-Python assertion that the labels can't drift on scroll-zoom."""
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 3.0)],
        y_lines=[],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.redraw()
    letters = _generated_letter_texts(canvas)
    assert letters, "expected generated X-line letter labels"
    expected = canvas.ax.get_xaxis_transform()
    for t in letters:
        assert t.get_transform() is expected


def test_y_line_letter_labels_use_yaxis_transform(qt_app):
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[],
        y_lines=[GridLine("1", 0.0), GridLine("2", 3.2)],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.redraw()
    letters = _generated_letter_texts(canvas)
    assert letters, "expected generated Y-line letter labels"
    expected = canvas.ax.get_yaxis_transform()
    for t in letters:
        assert t.get_transform() is expected


def test_x_line_letter_label_sits_on_top_spine_after_zoom(qt_app):
    """End-to-end: after a hard scroll-zoom-style xlim/ylim change
    (without a redraw), the on-screen pixel Y of an X-line letter label
    equals the pixel Y of the top spine. Catches the original bug
    (anchored at stale data Y → drifts off the canvas)."""
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0)], y_lines=[],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.redraw()
    letters = _generated_letter_texts(canvas)
    assert letters
    label = letters[0]
    # Simulate scroll-zoom: change limits without going through redraw().
    canvas.ax.set_xlim(-100.0, 100.0)
    canvas.ax.set_ylim(-100.0, 100.0)
    canvas.fig.canvas.draw()
    label_y_px = label.get_window_extent().y0
    top_spine_y_px = canvas.ax.transAxes.transform((0.0, 1.0))[1]
    # Mixed-transform anchor at y=1.0 (axes) sits exactly on the spine;
    # 2 px tolerance covers anti-aliasing + the va="bottom" offset.
    assert abs(label_y_px - top_spine_y_px) <= 2.0, (
        f"X-line letter drifted off the top spine "
        f"({label_y_px} vs {top_spine_y_px})"
    )


# ── optional "show grid letter next to coord" toggle ─────────────────────


def _xaxis_label_texts_at_draw(canvas):
    canvas.fig.canvas.draw()
    return [t.get_text() for t in canvas.ax.xaxis.get_majorticklabels()
            if t.get_visible()]


def test_letters_on_ticks_off_by_default(qt_app):
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 3.0)], y_lines=[],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    assert canvas.show_generated_grid_labels_on_ticks is False
    canvas.redraw()
    labels = _xaxis_label_texts_at_draw(canvas)
    assert labels, "expected some x-axis tick labels"
    for lbl in labels:
        assert "(" not in lbl, f"unexpected letter on tick: {lbl!r}"


def test_letters_on_ticks_when_enabled(qt_app):
    """When the toggle is on, FixedLocator ticks at generated-grid
    coordinates render as '<num> (<letter>)'."""
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 3.0)], y_lines=[],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.show_generated_grid_labels_on_ticks = True
    canvas.redraw()
    labels = _xaxis_label_texts_at_draw(canvas)
    assert "0 (A)" in labels
    assert "3 (B)" in labels


def test_letters_on_ticks_only_on_generated_axes(qt_app):
    """Axes that aren't covered by the generated grid (AdaptiveGridLocator)
    must NOT pick up parentheses — the formatter is per-axis."""
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0)], y_lines=[],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.show_generated_grid_labels_on_ticks = True
    canvas.redraw()
    canvas.fig.canvas.draw()
    y_labels = [t.get_text() for t in canvas.ax.yaxis.get_majorticklabels()
                if t.get_visible()]
    for lbl in y_labels:
        assert "(" not in lbl


def test_letters_on_ticks_ignores_non_grid_ticks(qt_app):
    """Even with the toggle on, the FixedLocator only ticks ON the
    generated coords, so every visible tick on a covered axis carries
    its matching letter (no orphan '5 ()' labels)."""
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 0.0), GridLine("B", 3.0)], y_lines=[],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.show_generated_grid_labels_on_ticks = True
    canvas.redraw()
    labels = _xaxis_label_texts_at_draw(canvas)
    # Every visible label should have either no parenthesis (no match,
    # shouldn't happen on a FixedLocator) or a non-empty letter.
    for lbl in labels:
        if "(" in lbl:
            assert lbl.endswith(")") and "()" not in lbl


# ── snap behavior must not change with display toggles ───────────────────


def test_snap_to_spacing_works_when_default_grid_hidden(qt_app):
    """Hiding the dotted default grid is a DISPLAY toggle only —
    click positions still snap to grid_spacing multiples."""
    canvas = _canvas(qt_app, _model_with_nodes())
    canvas.show_default_grid = False
    canvas.grid_spacing = 0.5
    x, y = canvas._snap(0.74, 1.26)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(1.5)


def test_snap_engine_still_emits_grid_candidates_when_generated_hidden(qt_app):
    """The labeled GridSystem keeps acting as a snap source even when
    its visual layer is hidden — this is the documented V1 behavior."""
    from structural_analysis.gui_qt.grid import GridLine, GridSystem
    sys_grid = GridSystem(
        x_lines=[GridLine("A", 1.0)], y_lines=[GridLine("1", 2.0)],
    )
    canvas = _canvas(qt_app, _model_with_nodes(), lambda: sys_grid)
    canvas.show_generated_grid = False
    cand = canvas.snap_engine.find_snap(
        cursor_x=1.001, cursor_y=2.001,
        px_per_dx=100.0, px_per_dy=100.0,
        model=_model_with_nodes(), grid=sys_grid,
    )
    assert cand is not None and cand.kind == "grid"


# ── MainWindow wiring ────────────────────────────────────────────────────


def _make_window(qt_app, model=None):
    from structural_analysis.gui_qt.app import MainWindow
    w = MainWindow()
    if model is not None:
        w._model = model
    return w


def test_generate_grid_from_nodes_builds_system_matching_node_coords(qt_app):
    w = _make_window(qt_app, _model_with_nodes())
    w._do_generate_grid_from_nodes()
    xs_in_grid = sorted(ln.coord for ln in w._grid.x_lines)
    ys_in_grid = sorted(ln.coord for ln in w._grid.y_lines)
    assert xs_in_grid == [0.0, 6.0]
    assert ys_in_grid == [0.0, 3.0]


def test_generate_grid_from_nodes_does_not_disable_default_grid(qt_app):
    """The user's exact complaint: generating from nodes used to hide
    the default drawing grid. It must stay visible."""
    w = _make_window(qt_app, _model_with_nodes())
    assert w.canvas.show_default_grid is True
    w._do_generate_grid_from_nodes()
    w.canvas.redraw()
    assert w.canvas.show_default_grid is True
    assert any(ln.get_visible() for ln in w.canvas.ax.get_xgridlines())


def test_clear_generated_grid_keeps_default_grid_enabled(qt_app):
    w = _make_window(qt_app, _model_with_nodes())
    w._do_generate_grid_from_nodes()
    assert not w._grid.is_empty()
    w._do_clear_generated_grid()
    assert w._grid.is_empty()
    # Default-grid flag is untouched.
    assert w.canvas.show_default_grid is True
    # The Show-generated and Clear actions disable themselves.
    assert w.act_show_generated_grid.isEnabled() is False
    assert w.act_clear_generated_grid.isEnabled() is False


def test_regenerate_grid_from_nodes_updates_generated_grid(qt_app):
    w = _make_window(qt_app, _model_with_nodes())
    w._do_generate_grid_from_nodes()
    first_x = sorted(ln.coord for ln in w._grid.x_lines)
    # Add a node and regenerate.
    from structural_analysis.model import Node
    w._model.nodes[99] = Node(99, 9.0, 0.0)
    w._do_generate_grid_from_nodes()
    second_x = sorted(ln.coord for ln in w._grid.x_lines)
    assert first_x == [0.0, 6.0]
    assert second_x == [0.0, 6.0, 9.0]


def test_toggle_show_default_grid_action_flips_canvas_flag(qt_app):
    w = _make_window(qt_app, _model_with_nodes())
    assert w.canvas.show_default_grid is True
    w.act_show_default_grid.setChecked(False)
    w.act_show_default_grid.trigger()
    # Qt toggles `checked` BEFORE the triggered signal fires for a
    # checkable action — but setChecked above already flipped it, so
    # trigger() reads the flipped state. To stay decoupled, just call
    # the slot directly with the desired state.
    w.act_show_default_grid.setChecked(False)
    w._toggle_show_default_grid()
    assert w.canvas.show_default_grid is False


def test_toggle_show_generated_grid_action_flips_canvas_flag(qt_app):
    w = _make_window(qt_app, _model_with_nodes())
    w._do_generate_grid_from_nodes()
    assert w.canvas.show_generated_grid is True
    w.act_show_generated_grid.setChecked(False)
    w._toggle_show_generated_grid()
    assert w.canvas.show_generated_grid is False


def test_generated_actions_disabled_until_grid_exists(qt_app):
    w = _make_window(qt_app, _model_with_nodes())
    # Fresh window: no GridSystem yet.
    assert w._grid.is_empty()
    assert w.act_show_generated_grid.isEnabled() is False
    assert w.act_clear_generated_grid.isEnabled() is False
    w._do_generate_grid_from_nodes()
    assert w.act_show_generated_grid.isEnabled() is True
    assert w.act_clear_generated_grid.isEnabled() is True


def test_generate_grid_from_nodes_is_undoable(qt_app):
    w = _make_window(qt_app, _model_with_nodes())
    assert w._grid.is_empty()
    w._do_generate_grid_from_nodes()
    assert not w._grid.is_empty()
    w.act_undo.trigger()
    assert w._grid.is_empty()
    w.act_redo.trigger()
    assert not w._grid.is_empty()


def test_undo_of_grid_change_refreshes_action_enable_state(qt_app):
    """Undoing a grid populate must re-disable the generated-grid
    actions, since the GridSystem is empty again."""
    w = _make_window(qt_app, _model_with_nodes())
    w._do_generate_grid_from_nodes()
    assert w.act_clear_generated_grid.isEnabled() is True
    w.act_undo.trigger()
    assert w.act_clear_generated_grid.isEnabled() is False


def test_show_grid_labels_action_disabled_without_grid(qt_app):
    w = _make_window(qt_app, _model_with_nodes())
    assert w._grid.is_empty()
    assert w.act_show_grid_labels_on_ticks.isEnabled() is False
    w._do_generate_grid_from_nodes()
    assert w.act_show_grid_labels_on_ticks.isEnabled() is True
    w._do_clear_generated_grid()
    assert w.act_show_grid_labels_on_ticks.isEnabled() is False


def test_show_grid_labels_action_flips_canvas_flag(qt_app):
    w = _make_window(qt_app, _model_with_nodes())
    w._do_generate_grid_from_nodes()
    assert w.canvas.show_generated_grid_labels_on_ticks is False
    w.act_show_grid_labels_on_ticks.setChecked(True)
    w._toggle_show_grid_labels_on_ticks()
    assert w.canvas.show_generated_grid_labels_on_ticks is True
    w.act_show_grid_labels_on_ticks.setChecked(False)
    w._toggle_show_grid_labels_on_ticks()
    assert w.canvas.show_generated_grid_labels_on_ticks is False
