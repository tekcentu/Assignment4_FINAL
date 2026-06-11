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
