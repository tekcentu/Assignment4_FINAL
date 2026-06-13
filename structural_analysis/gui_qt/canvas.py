"""Matplotlib QtAgg canvas for the PyQt6 frontend.

``mpl_connect`` events are backend-agnostic, so the drawing and hit-test
code here stays plain matplotlib; only the embedding widget and the
toolbar are Qt-specific.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib
matplotlib.use("QtAgg")  # noqa: E402  must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as _MplPolygon
from matplotlib.collections import PolyCollection as _PolyCollection
from matplotlib import patheffects as _path_effects
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT,
)
from matplotlib.ticker import FixedLocator, FuncFormatter, Locator, ScalarFormatter

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from ..element import FrameElement2D, TrussElement2D
from ..model import (
    FrameTemperatureLoad,
    NodalLoad,
    PointLoad,
    StructuralModel,
    Support,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)
from ..gui_common.geometry import (
    physical_member_polygon as _physical_member_polygon,
    physical_display_thickness as _physical_display_thickness,
    joint_overlap_regions as _joint_overlap_regions,
    resolved_default_depth as _resolved_default_depth,
    PHYSICAL_DEPTH_FRACTION as _DEFAULT_VISUAL_DEPTH_FRACTION,
    PHYSICAL_DEPTH_MIN as _DEFAULT_VISUAL_DEPTH_MIN,
    PHYSICAL_DEPTH_MAX as _DEFAULT_VISUAL_DEPTH_MAX,
)
from .grid import GridSystem
from .snap import SnapCandidate, SnapEngine
from .element_graphics import (
    sample_internal_force as _diagram_ordinates,
    internal_force_at as _diagram_value,
    _split_segments_by_sign as _diagram_sign_split,
    sign_fill_color as _diagram_sign_color,
    _RIGID_ZONE_COLOR,
)


_TOOLBAR_REMOVE = {"Subplots", "Customize"}
# _DEFAULT_VISUAL_DEPTH_* constants are imported from gui_common.geometry above.


class AppNavigationToolbar(NavigationToolbar2QT):
    """Filtered matplotlib navigation toolbar.

    Removes raw plot-configuration actions (Subplots, Customize) that are
    meaningless in a structural-analysis context, and adds a Fit button that
    calls the canvas's fit_to_view method.
    """

    toolitems = [t for t in NavigationToolbar2QT.toolitems
                 if t[0] not in _TOOLBAR_REMOVE]

    def __init__(self, canvas, parent, *, fit_callback=None):
        super().__init__(canvas, parent)
        if fit_callback is not None:
            self.addSeparator()
            act = self.addAction("Fit")
            act.setToolTip("Fit view to model (Home key)")
            act.triggered.connect(fit_callback)
            self._fit_action = act
        else:
            self._fit_action = None


@dataclass
class HitResult:
    """What was under the mouse at the time of an event."""
    x: float
    y: float
    node_id: Optional[int] = None
    element_id: Optional[int] = None
    snap_kind: str = ""    # e.g. "node", "grid", "midpoint", "endpoint", "project"
    snap_label: str = ""   # human-readable target description


class AdaptiveGridLocator(Locator):
    """Coordinate-axis tick locator that stays readable at any zoom.

    Ticks land on multiples of a *base* spacing (the snap grid), but the
    spacing is coarsened in a 1-2-5 progression (base × 1, 2, 5, 10, 20,
    50, …) so that no more than ``max_ticks`` labels fall inside the
    current view. A fixed ``MultipleLocator`` instead emits one tick per
    base step regardless of zoom, so the numbers collide as soon as the
    user zooms out.

    matplotlib re-invokes the locator on *every* draw using the live view
    interval, so this self-adjusts during scroll-zoom — which only calls
    ``draw_idle`` and never rebuilds the grid — as well as on full redraws.
    Ticks are never made finer than ``base``, so zooming in still reveals
    the true snap grid rather than inventing sub-grid coordinates.
    """

    def __init__(self, base: float, max_ticks: int) -> None:
        self._base = float(base)
        self._max_ticks = max(1, int(max_ticks))

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def step_for_span(self, span: float) -> float:
        """Smallest base×{1,2,5,10,…} step keeping ≤ max_ticks in ``span``."""
        step = self._base
        if step <= 0.0 or span <= 0.0:
            return max(step, 0.0)
        seq = (1, 2, 5)
        i = 0
        while span / step > self._max_ticks:
            i += 1
            step = self._base * seq[i % 3] * (10 ** (i // 3))
        return step

    def tick_values(self, vmin, vmax):
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        step = self.step_for_span(vmax - vmin)
        if step <= 0.0:
            return []
        first = math.ceil(vmin / step) * step
        ticks = []
        x = first
        # +0.5·step guards the final tick against floating-point drift.
        while x <= vmax + step * 0.5:
            ticks.append(x)
            x += step
        return self.raise_if_exceeds(ticks)


class ModelCanvas(QWidget):
    """A QWidget containing the matplotlib figure + its navigation toolbar."""

    NODE_PICK_RADIUS_PX = 12
    ELEM_PICK_RADIUS_PX = 8
    MAX_AUTO_NODE_LABELS = 300
    MAX_AUTO_ELEMENT_LABELS = 250
    MAX_LABELED_GRID_LINES = 240
    # Max coordinate numbers per axis before AdaptiveGridLocator coarsens
    # the spacing — keeps the spine labels from colliding when zoomed out.
    MAX_AXIS_LABELS = 12

    def __init__(self, parent: QWidget | None,
                 model_provider: Callable[[], StructuralModel],
                 grid_provider: Callable[[], GridSystem] | None = None) -> None:
        super().__init__(parent)
        self._model = model_provider
        self._grid_provider = grid_provider or (lambda: GridSystem())
        self.grid_spacing: float = 0.5
        self.snap_enabled: bool = True
        self.snap_engine = SnapEngine(tolerance_px=10.0)
        # Two independent grid display layers (snap is unaffected by either):
        #   show_default_grid    — the dotted ``grid_spacing`` reference grid
        #   show_generated_grid  — the labeled GridSystem from _grid_provider
        # Either can be toggled on/off without affecting snap. ``_snap()``
        # always rounds to ``grid_spacing`` when no snap-engine candidate
        # wins, and the snap engine always emits GridSystem-intersection
        # candidates when the system is non-empty — regardless of these
        # display flags.
        self.show_default_grid: bool = True
        self.show_generated_grid: bool = True
        # When True AND a generated grid is visible, axis tick labels on
        # generated-line coordinates render as "<num> (<letter>)" — e.g.
        # "3 (A)". Default OFF: clean numeric spine. Toggled from
        # View → Grid → "Show grid line labels with coords".
        self.show_generated_grid_labels_on_ticks: bool = False

        self.show_deformed: bool = True
        self.show_reactions: bool = True
        self.show_diagrams: bool = False
        self.show_section_labels: bool = False
        # v0.24.0: optional overlay drawing local x/y axis arrows and
        # i/j end labels on each element so users can see element
        # orientation at a glance. Off by default — advanced view.
        self.show_local_axes: bool = False
        self.show_physical_members: bool = False
        # Post-draw counter: how many elements fell back to default depth
        # because section.depth == 0.  Read by the status-bar note in app.py.
        self._physical_members_missing_depth: int = 0
        self.diagram_kind: str = "moment"
        self.deformed_scale: float = 1.0
        self.diagram_scale: float = 1.0
        # Station points are post-processing samples per element, used to
        # draw smooth deformed shapes (cubic-Hermite for frames) and
        # internal-force diagrams. They are NOT model nodes and never
        # affect connectivity, supports, or loads. 21 picks up the
        # midspan exactly and gives smooth UDL / point-load diagrams.
        self.deformed_stations: int = 21
        self.diagram_stations: int = 21
        self._result = None
        # PR-A: which load case the host considers "active". Loads on
        # the canvas whose ``load_case`` matches this string render at
        # full alpha; others dim to ``_inactive_load_alpha`` when
        # ``_active_case_loads_only`` is True (the View → "Active case
        # loads only" toggle). The host (MainWindow) is the source of
        # truth for the toggle — canvas just consumes the boolean.
        self._active_case: str = "DEFAULT"
        self._active_case_loads_only: bool = True
        self._inactive_load_alpha: float = 0.35
        # PR #29: when a load COMBINATION is active, this holds the set
        # of constituent case names so loads from ANY of them render at
        # full alpha (signalling "all these cases contribute") while
        # other-case loads dim — instead of misleadingly highlighting a
        # single case. ``None`` ⇒ a plain load case (or SUM_ALL) is
        # active and the single-case dimming applies.
        self._active_combination_cases: frozenset[str] | None = None
        self._modal_result = None    # ModalResult or None
        self._modal_mode_idx: int = 0
        self._modal_scale: float = 1.0
        self._snap_marker = None  # current SnapCandidate
        # Persistent matplotlib Line2D artists for the hover/snap markers.
        # We re-use them across hover-only repaints (which avoid a full
        # ax.clear()) so the markers can be moved by updating xdata/ydata
        # + draw_idle() instead of rebuilding the whole scene. ax.clear()
        # in redraw() detaches them; _draw_snap_marker rebuilds them
        # lazily on the next paint.
        self._snap_marker_artist = None     # solid snap-target marker
        self._hover_marker_artist = None    # faint "+" ghost crosshair
        # Either anchored at an existing node id (legacy) or a free
        # start point (v0.10.0 — first click landed on empty space and
        # no node has been created yet). Exactly one is non-None at a
        # time.
        self._element_preview: tuple[int, float, float, str] | None = None
        self._element_preview_free: (
            tuple[float, float, float, float, str] | None
        ) = None
        # Multi-select state (v0.13.0). Each set holds all currently
        # selected node / element ids. Single-object selection is just
        # a one-element set; box-select fills these in bulk.
        self._selected_node_ids: set[int] = set()
        self._selected_element_ids: set[int] = set()
        # Active drag-rectangle for box selection. Tuple:
        # ``(x0, y0, x1, y1, is_crossing)`` in world coords, or None
        # when no drag is in progress. ``is_crossing`` is True for
        # right-to-left drags (Crossing mode) and False for
        # left-to-right drags (Window mode).
        self._drag_rect: (
            tuple[float, float, float, float, bool] | None
        ) = None
        # PR #31 — pre-solve validation highlight layer.  Distinct from
        # selection: amber = warning (orphan / advisory), red = error
        # (mechanism / disconnected unsupported component).  Painted
        # behind the selection layer so a selected problem node still
        # shows the gold ring on top.
        self._warning_node_ids: set[int] = set()
        self._error_node_ids: set[int] = set()
        self._warning_element_ids: set[int] = set()
        self._error_element_ids: set[int] = set()
        # Per-element max / min markers on the currently-drawn moment /
        # shear / axial diagram. Populated by _draw_diagrams and fed
        # into the snap engine so the cursor snaps to those points in
        # post mode. Empty when no diagram is showing.
        self._diagram_critical_points: list[dict] = []
        # Fallback hover cursor when no snap candidate is active. This
        # is what the user sees when their cursor is over empty space
        # between grid lines — it marks the point a left-click would
        # actually land on (the rectangular-grid-snapped coords).
        self._hover_xy: tuple[float, float] | None = None
        # Whether the view has been initialised at least once. Until
        # this is True, redraw() auto-fits the model + grid extent. Once
        # set, redraw() preserves the user's current xlim/ylim so
        # placing a node or moving the mouse no longer collapses the
        # zoom level.
        self._view_initialised: bool = False
        # Tracks whether the user has manually adjusted the view (scroll
        # zoom, middle-button pan). While False, a window resize is free
        # to re-fit the data limits to match the new widget aspect so
        # the grid fills the canvas. Once True, the user owns the
        # viewport — resize keeps the current limits and only fit_to_view
        # (View → Fit) takes the wheel back.
        self._user_view_dirty: bool = False
        # Guard flipped to True while we (canvas internals) are
        # programmatically setting xlim/ylim — first fit, redraw's
        # save-and-restore, fit_to_view. The xlim/ylim-changed mpl
        # callbacks consult this flag so the dirty bit only flips for
        # *external* mutations (matplotlib navigation toolbar pan/zoom,
        # programmatic test pokes, etc.).
        self._setting_axes_limits: bool = False

        # Middle-mouse-drag pan state (display coordinates at drag start).
        self._pan_origin: tuple[float, float] | None = None
        self._pan_xlim0: tuple[float, float] = (0.0, 1.0)
        self._pan_ylim0: tuple[float, float] = (0.0, 1.0)

        # The pixel-coord parameter is passed alongside the world-coord
        # HitResult so tools (currently SelectTool) can drive direction-
        # aware drag selection independently of axis flips or zoom.
        self.on_click: (
            Callable[[HitResult, str, tuple[float, float], bool], None] | None
        ) = None
        self.on_motion: (
            Callable[[HitResult, tuple[float, float]], None] | None
        ) = None
        self.on_release: (
            Callable[[HitResult, str, tuple[float, float], bool], None] | None
        ) = None
        # Cache of the most recent hit + event pixel coords so
        # _handle_release can route a HitResult without re-running the
        # full hit-test (mpl release events sometimes arrive with no
        # xdata/ydata, e.g. when the cursor leaves the axes mid-drag).
        self._last_hit: HitResult | None = None
        self._last_event_px: tuple[float, float] | None = None

        self.fig = plt.Figure(figsize=(7.5, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        # adjustable="box" lets the user's xlim/ylim be honored exactly
        # (matplotlib resizes the axes rectangle to keep aspect=1).
        # The legacy "datalim" mode would silently rewrite our limits
        # and emit "Ignoring fixed y limits…" on every redraw.
        self.ax.set_aspect("equal", adjustable="box")
        # Mark the view dirty whenever something *outside* canvas
        # internals changes the limits — most importantly the
        # matplotlib navigation toolbar's pan/zoom modes, which
        # otherwise leave _user_view_dirty False and let the next
        # window resize silently discard the user's view.
        self.ax.callbacks.connect("xlim_changed", self._on_limits_changed)
        self.ax.callbacks.connect("ylim_changed", self._on_limits_changed)

        self._mpl_canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = AppNavigationToolbar(
            self._mpl_canvas, self, fit_callback=self.fit_to_view,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self._mpl_canvas)
        self.setLayout(layout)

        # Allow ESC key events to propagate up to MainWindow regardless
        # of which widget has focus.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._mpl_canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._mpl_canvas.installEventFilter(self)

        self._mpl_canvas.mpl_connect("button_press_event", self._handle_click)
        self._mpl_canvas.mpl_connect("button_release_event", self._handle_release)
        self._mpl_canvas.mpl_connect("motion_notify_event", self._handle_motion)
        self._mpl_canvas.mpl_connect("scroll_event", self._handle_scroll)

    # ── public API ──

    def set_grid_spacing(self, spacing: float) -> None:
        if spacing <= 0:
            raise ValueError("Grid spacing must be > 0.")
        self.grid_spacing = float(spacing)
        self.redraw()

    def toggle_snap(self, enabled: bool) -> None:
        self.snap_enabled = bool(enabled)

    def set_element_preview(
        self, start_node_id: int, end_x: float, end_y: float, kind: str
    ) -> None:
        """Show a temporary member preview anchored at an existing node."""
        self._element_preview = (
            int(start_node_id), float(end_x), float(end_y), str(kind)
        )
        self._element_preview_free = None

    def set_element_preview_free(
        self,
        start_x: float, start_y: float, end_x: float, end_y: float,
        kind: str,
    ) -> None:
        """Show a temporary member preview when the start is empty space.

        v0.10.0: the Frame / Truss tools now accept a first click on
        empty space (a node will be auto-created on the second click).
        While only the first click has landed, there is no existing
        node id to anchor the preview to.
        """
        self._element_preview_free = (
            float(start_x), float(start_y),
            float(end_x), float(end_y),
            str(kind),
        )
        self._element_preview = None

    def clear_element_preview(self) -> None:
        self._element_preview = None
        self._element_preview_free = None

    def select_node(self, node_id: int) -> None:
        """Exclusive single-node selection — clears everything else."""
        self._selected_node_ids = {int(node_id)}
        self._selected_element_ids = set()

    def select_element(self, element_id: int) -> None:
        """Exclusive single-element selection — clears everything else."""
        self._selected_element_ids = {int(element_id)}
        self._selected_node_ids = set()

    def add_node_to_selection(self, node_id: int) -> None:
        self._selected_node_ids.add(int(node_id))

    def remove_node_from_selection(self, node_id: int) -> None:
        self._selected_node_ids.discard(int(node_id))

    def add_element_to_selection(self, element_id: int) -> None:
        self._selected_element_ids.add(int(element_id))

    def remove_element_from_selection(self, element_id: int) -> None:
        self._selected_element_ids.discard(int(element_id))

    def get_selected_nodes(self) -> frozenset[int]:
        return frozenset(self._selected_node_ids)

    def get_selected_elements(self) -> frozenset[int]:
        return frozenset(self._selected_element_ids)

    def clear_selection(self) -> None:
        self._selected_node_ids = set()
        self._selected_element_ids = set()

    def set_drag_rect(
        self, x0: float, y0: float, x1: float, y1: float,
        is_crossing: bool,
    ) -> None:
        """Set the active box-select rectangle (world coords + direction)."""
        self._drag_rect = (
            float(x0), float(y0), float(x1), float(y1), bool(is_crossing),
        )

    def clear_drag_rect(self) -> None:
        self._drag_rect = None

    def set_result(self, result) -> None:
        self._result = result
        # Static and modal results are mutually exclusive on screen.
        self._modal_result = None
        self.redraw()

    def clear_result(self) -> None:
        self._result = None
        self.redraw()

    def set_active_case(self, name: str) -> None:
        """Host signal: a new load case is active. Triggers a redraw so
        the load-dimming and any case-tagged annotations update.
        Idempotent on no-op."""
        if name == self._active_case:
            return
        self._active_case = name
        self.redraw()

    def set_active_case_loads_only(self, on: bool) -> None:
        """Host signal: toggle the "show only active-case loads" mode
        (when off, all loads draw at full alpha)."""
        if on == self._active_case_loads_only:
            return
        self._active_case_loads_only = bool(on)
        self.redraw()

    def set_active_combination_cases(
        self, case_names: "frozenset[str] | set[str] | None",
    ) -> None:
        """Host signal (PR #29): a load COMBINATION is active. ``case_names``
        is the set of constituent case names — loads from any of them
        render at full alpha. Pass ``None`` when switching back to a
        plain case / SUM_ALL selection."""
        new_val = frozenset(case_names) if case_names is not None else None
        if new_val == self._active_combination_cases:
            return
        self._active_combination_cases = new_val
        self.redraw()

    def _load_case_alpha(self, ld) -> float:
        """Return the draw alpha for a load arrow.

        * "active case only" toggle off ⇒ everything full intensity.
        * a COMBINATION is active ⇒ loads belonging to any constituent
          case are full, others dim (PR #29 — no misleading single-case
          highlight).
        * otherwise ⇒ loads matching the single active case are full,
          others dim."""
        if not self._active_case_loads_only:
            return 1.0
        case = getattr(ld, "load_case", "DEFAULT")
        if self._active_combination_cases is not None:
            return (
                1.0 if case in self._active_combination_cases
                else self._inactive_load_alpha
            )
        return 1.0 if case == self._active_case else self._inactive_load_alpha

    def set_modal_result(self, modal_result, mode_idx: int = 0,
                         scale: float = 1.0) -> None:
        """Display a single mode of a ModalResult on the canvas.

        Static-result overlays are cleared while a modal result is shown
        (the canvas displays one analysis kind at a time).
        """
        self._result = None
        self._modal_result = modal_result
        self._modal_mode_idx = max(0, int(mode_idx))
        self._modal_scale = float(scale)
        self.redraw()

    def update_modal_view(self, mode_idx: int, scale: float) -> None:
        """Update the displayed mode index and/or scale for the current
        :class:`ModalResult` without rebuilding it."""
        if self._modal_result is None:
            return
        self._modal_mode_idx = max(0, int(mode_idx))
        self._modal_scale = float(scale)
        self.redraw()

    def clear_modal_result(self) -> None:
        self._modal_result = None
        self.redraw()

    def fit_to_view(self) -> None:
        """Re-fit the axes to enclose the current model + grid extent.

        ``redraw()`` preserves the user's pan/zoom state by default, so
        callers need to invoke this explicitly to reset the view (e.g.
        after loading a file or via the View → Fit action).
        """
        self._view_initialised = False
        self._user_view_dirty = False
        self.redraw()

    def redraw(self) -> None:
        # Preserve the current view across the clear()/redraw cycle so
        # mouse motion and node-placement events don't collapse the
        # zoom level. On the very first redraw we have nothing to
        # preserve — call _set_axes_limits to fit the (possibly empty)
        # model + grid, and remember that we did.
        if self._view_initialised:
            saved_xlim = self.ax.get_xlim()
            saved_ylim = self.ax.get_ylim()
        else:
            saved_xlim = saved_ylim = None

        # ax.clear() resets xlim/ylim and would fire the limit-changed
        # callbacks; the restore call below would also fire them.
        # Both are programmatic, so suppress the dirty-bit toggle for
        # the duration. (_set_axes_limits handles its own guard.)
        self._setting_axes_limits = True
        try:
            self.ax.clear()
            # ax.clear() destroys all artists; drop our cached references
            # so _draw_snap_marker rebuilds them on the next paint.
            self._snap_marker_artist = None
            self._hover_marker_artist = None
            self.ax.set_aspect("equal", adjustable="box")

            if saved_xlim is not None:
                self.ax.set_xlim(saved_xlim)
                self.ax.set_ylim(saved_ylim)
            else:
                self._set_axes_limits()
                self._view_initialised = True
        finally:
            self._setting_axes_limits = False

        self._draw_grid()
        self._draw_origin_axes()
        self._draw_model()
        self._draw_validation_highlights()
        self._draw_selection()
        self._draw_element_preview()
        if self._result is not None and self._result.status == "ok":
            if self.show_deformed:
                self._draw_deformed()
            if self.show_reactions:
                self._draw_reactions()
            if self.show_diagrams:
                self._draw_diagrams()
            else:
                # No diagram on screen → no critical-point snaps active.
                self._diagram_critical_points = []
        elif self._modal_result is not None and self._modal_result.status == "ok":
            self._diagram_critical_points = []
            self._draw_mode_shape()
        else:
            self._diagram_critical_points = []
        self._draw_snap_marker()
        self._draw_drag_rect()
        self._mpl_canvas.draw_idle()

    # ── Qt resize → re-fit while the user hasn't taken the wheel ──

    def resizeEvent(self, event) -> None:
        """When the widget is resized and the user hasn't manually
        panned or zoomed yet, re-fit the data limits so the data box's
        aspect matches the new widget aspect — i.e. the grid keeps
        filling the canvas instead of collapsing to a centred square.

        Once the user pans or scroll-zooms, ``_user_view_dirty`` is
        True and we leave the limits alone. ``fit_to_view`` (View →
        Fit) resets the flag and re-engages auto-fit.
        """
        super().resizeEvent(event)
        if self._user_view_dirty:
            return
        if not self._view_initialised:
            # First-ever paint hasn't happened yet; redraw() will set
            # the limits using the new widget size.
            return
        self._set_axes_limits()
        self._mpl_canvas.draw_idle()

    def eventFilter(self, obj, event) -> bool:
        """Forward ESC key events from the embedded matplotlib canvas up to
        the top-level MainWindow so the existing ESC handler runs regardless
        of which widget currently holds keyboard focus.

        We must also intercept ShortcutOverride: Qt fires this before KeyPress
        to let actions claim a key. MainWindow has act_sel_clear with
        shortcut="Escape"; if we don't claim ESC here, that action fires
        first and the KeyPress never reaches MainWindow.keyPressEvent."""
        if obj is self._mpl_canvas:
            t = event.type()
            if t in (QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride):
                if event.key() == Qt.Key.Key_Escape:
                    if t == QEvent.Type.ShortcutOverride:
                        # Claim ESC so act_sel_clear's Escape shortcut does
                        # not fire before MainWindow.keyPressEvent is reached.
                        event.accept()
                        return True
                    top = self.window()
                    if top is not None and top is not self:
                        top.keyPressEvent(event)
                        return True
        return super().eventFilter(obj, event)

    # ── event forwarding ──

    def _handle_click(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        if event.button == 2 and not self.toolbar.mode:
            # Middle-button drag — start pan.
            self._pan_origin = (event.x, event.y)
            self._pan_xlim0 = tuple(self.ax.get_xlim())
            self._pan_ylim0 = tuple(self.ax.get_ylim())
            return
        if self.on_click is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.toolbar.mode:
            # The matplotlib navigation toolbar is in pan or zoom mode,
            # so it absorbs every left-click on the canvas. Surface a
            # status hint via the on_nav_mode_block callback (if the
            # host has wired one up) — otherwise silently swallow.
            cb = getattr(self, "on_nav_mode_block", None)
            if cb is not None:
                try:
                    cb(str(self.toolbar.mode))
                except Exception:
                    pass
            return
        hit = self._hit_test(event)
        button_name = {1: "left", 2: "middle", 3: "right"}.get(event.button, "left")
        event_px = (float(event.x), float(event.y))
        self._last_hit = hit
        self._last_event_px = event_px
        shift = _shift_pressed()
        self.on_click(hit, button_name, event_px, shift)

    def _handle_motion(self, event) -> None:
        if self._pan_origin is not None:
            # Middle-mouse drag in progress — pan without redrawing the model.
            inv = self.ax.transData.inverted()
            x0d, y0d = inv.transform(self._pan_origin)
            xcd, ycd = inv.transform((event.x, event.y))
            dx, dy = xcd - x0d, ycd - y0d
            xl, xr = self._pan_xlim0
            yb, yt = self._pan_ylim0
            self.ax.set_xlim(xl - dx, xr - dx)
            self.ax.set_ylim(yb - dy, yt - dy)
            self._user_view_dirty = True
            self._mpl_canvas.draw_idle()
            return
        if self.on_motion is None or event.inaxes is not self.ax:
            # Cursor left the axes — drop the hover marker.
            if self._hover_xy is not None:
                self._hover_xy = None
                self._snap_marker = None
                self.repaint_hover_marker()
            return
        if event.xdata is None or event.ydata is None:
            return
        hit = self._hit_test(event)
        # Record the position a click would land on, so the hover
        # marker can be drawn even when the snap engine has no
        # candidate (empty space between labeled grid lines).
        self._hover_xy = (hit.x, hit.y)
        event_px = (float(event.x), float(event.y))
        self._last_hit = hit
        self._last_event_px = event_px
        try:
            self.on_motion(hit, event_px)
        except Exception:
            pass

    def _handle_release(self, event) -> None:
        if event.button == 2:
            self._pan_origin = None
            return
        if event.button != 1 or self.on_release is None:
            return
        # Release events sometimes lack xdata/ydata (cursor outside
        # axes). Reuse the last recorded hit + pixel position so the
        # tool can still finish a drag cleanly.
        hit = self._last_hit
        event_px = self._last_event_px
        if event.xdata is not None and event.ydata is not None:
            event_px = (float(event.x), float(event.y))
            try:
                hit = self._hit_test(event)
                self._last_hit = hit
                self._last_event_px = event_px
            except Exception:
                pass
        if hit is None or event_px is None:
            return
        shift = _shift_pressed()
        try:
            self.on_release(hit, "left", event_px, shift)
        except Exception:
            pass

    def _handle_scroll(self, event) -> None:
        if event.inaxes is not self.ax or self.toolbar.mode:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Scroll up → zoom in (shrink the visible range); scroll down → zoom out.
        factor = 1.0 / 1.15 if event.button == "up" else 1.15
        xl, xr = self.ax.get_xlim()
        yb, yt = self.ax.get_ylim()
        xd, yd = event.xdata, event.ydata
        self.ax.set_xlim(xd - (xd - xl) * factor, xd + (xr - xd) * factor)
        self.ax.set_ylim(yd - (yd - yb) * factor, yd + (yt - yd) * factor)
        self._user_view_dirty = True
        self._mpl_canvas.draw_idle()

    # ── geometry / hit-test ──

    def _snap(self, x: float, y: float) -> tuple[float, float]:
        if not self.snap_enabled:
            return x, y
        s = self.grid_spacing
        return round(x / s) * s, round(y / s) * s

    def _hit_test(self, event) -> HitResult:
        model = self._model()
        grid = self._grid_provider()

        bbox = self.ax.bbox
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        px_per_dx = bbox.width / max(x1 - x0, 1e-9)
        px_per_dy = bbox.height / max(y1 - y0, 1e-9)

        # Run the snap engine.
        candidate = None
        if self.snap_enabled:
            candidate = self.snap_engine.find_snap(
                cursor_x=event.xdata, cursor_y=event.ydata,
                px_per_dx=px_per_dx, px_per_dy=px_per_dy,
                model=model, grid=grid,
                diagram_points=self._diagram_critical_points or None,
            )
        self._snap_marker = candidate

        if candidate is not None:
            hit = HitResult(x=candidate.x, y=candidate.y,
                            snap_kind=candidate.kind,
                            snap_label=candidate.label)
            if candidate.kind == "node":
                hit.node_id = candidate.object_id
            elif candidate.kind in ("endpoint", "midpoint", "project",
                                     "diagram"):
                hit.element_id = candidate.object_id
            else:
                # Non-element snap (e.g. "grid"): the snap engine
                # prefers grid over project, so a click on an
                # element's interior near a grid intersection arrives
                # here. Still attach the nearest element so the
                # NodeTool / _PairTool split path can engage —
                # otherwise the disconnected-component bug returns
                # whenever a labeled grid is configured (PR #21 review,
                # codex P1).
                hit.element_id = self._pick_nearest_element_px(
                    event.xdata, event.ydata, px_per_dx, px_per_dy, model,
                )
            return hit

        # No snap → fall back to rectangular-grid snapping + element pick.
        sx, sy = self._snap(event.xdata, event.ydata)
        hit = HitResult(x=sx, y=sy)
        hit.element_id = self._pick_nearest_element_px(
            event.xdata, event.ydata, px_per_dx, px_per_dy, model,
        )
        return hit

    def _pick_nearest_element_px(
        self, x: float, y: float,
        px_per_dx: float, px_per_dy: float, model,
    ) -> Optional[int]:
        """Return the id of the element closest to ``(x, y)`` within
        :attr:`ELEM_PICK_RADIUS_PX`, or ``None``. Shared by the
        no-snap fallback and the grid-snap branch — keeps the
        element-pick logic in one place (PR #21 review)."""
        best_eid = None
        best_dpx = self.ELEM_PICK_RADIUS_PX
        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            dpx = _point_segment_distance_px(
                x, y, ni.x, ni.y, nj.x, nj.y, px_per_dx, px_per_dy,
            )
            if dpx < best_dpx:
                best_dpx = dpx
                best_eid = elem.id
        return best_eid

    # ── drawing ──

    def _install_axis_locators(self, grid, gen_visible: bool) -> None:
        """Choose the spine tick locator + formatter per axis.

        When the generated grid is visible, an axis that has grid lines
        ticks exactly on those line coordinates (FixedLocator) so the
        spine reads the structural coordinates as constant values; the
        formatter optionally appends the line letter in parentheses
        (e.g. ``"3 (A)"``) when ``show_generated_grid_labels_on_ticks``
        is on. Otherwise — and for an axis the generated grid doesn't
        cover — the AdaptiveGridLocator gives readable default-spacing
        ticks at any zoom, paired with the default ScalarFormatter (so
        ``ticklabel_format(useOffset=False)`` still applies).
        """
        def _install(axis, lines):
            if gen_visible and lines:
                coords = sorted({float(ln.coord) for ln in lines})
                axis.set_major_locator(FixedLocator(coords))
                # FuncFormatter emits plain f"{value:g}" — offset
                # notation is impossible by construction, so no
                # set_useOffset call is needed on this path.
                axis.set_major_formatter(self._grid_letter_formatter(lines))
            else:
                axis.set_major_locator(AdaptiveGridLocator(
                    self.grid_spacing, self.MAX_AXIS_LABELS))
                # Keep the default ScalarFormatter that ax.clear()
                # already installed, but kill the "+1e3" offset corner
                # that grows when coordinates are far from zero.
                fmt = axis.get_major_formatter()
                if isinstance(fmt, ScalarFormatter):
                    fmt.set_useOffset(False)
                    fmt.set_scientific(False)

        _install(self.ax.xaxis, grid.x_lines)
        _install(self.ax.yaxis, grid.y_lines)

    def _grid_letter_formatter(self, lines):
        """Tick formatter that optionally appends ``"(<letter>)"`` to the
        coordinate values on generated-grid lines.

        Lookup uses a 6-decimal rounding so tiny floating-point drift in
        the FixedLocator ticks vs the source ``GridLine.coord`` doesn't
        skip a match.
        """
        by_coord = {round(float(ln.coord), 6): ln.label for ln in lines}

        def _fmt(value, _pos):
            # ``f"{value:g}"`` matches matplotlib's plain-number look and
            # already obeys the ticklabel_format(useOffset=False) intent.
            num = f"{value:g}"
            if not self.show_generated_grid_labels_on_ticks:
                return num
            letter = by_coord.get(round(float(value), 6))
            return f"{num} ({letter})" if letter else num
        return FuncFormatter(_fmt)

    def _draw_grid(self) -> None:
        grid = self._grid_provider()
        # The two layers are drawn independently — populating a GridSystem
        # no longer hides the default reference grid. When both are on,
        # the default grid is faded so the named structural grid stays
        # visually dominant; when only the default is on, it uses its
        # full weight as the user expects.
        gen_visible = self.show_generated_grid and not grid.is_empty()

        # Spine tick locators. When the generated grid is visible we put
        # the ticks ON the structural grid-line coordinates so the spine
        # shows their constant values (e.g. 0, 3, 6 / 0, 3.2) — those are
        # the meaningful coordinates the user wants to read off. With no
        # generated grid we fall back to the AdaptiveGridLocator, which
        # coarsens in a 1-2-5 progression so the default-spacing numbers
        # never collide as the user zooms. ax.clear() (in redraw) resets
        # the locator, so this runs every redraw; matplotlib re-invokes
        # the locator on every draw, which keeps scroll-zoom readable.
        # _install_axis_locators also disables the "+1e3" offset corner
        # on whichever axes carry a ScalarFormatter (the AdaptiveGridLocator
        # path) — the FixedLocator + FuncFormatter path is offset-free
        # by construction, so a blanket ax.ticklabel_format would raise.
        self._install_axis_locators(grid, gen_visible)
        # Mirror tick labels to the top + right spines too. The default
        # axes margins already reserve pixel space on all four sides
        # (no tight_layout / constrained_layout is in use), so enabling
        # the top/right labels is purely additive — it does not affect
        # set_aspect("equal") / _set_axes_limits fitting. ax.clear()
        # resets tick_params, so this lives here next to the locator.
        self.ax.tick_params(axis="x", which="major",
                            top=True, labeltop=True,
                            bottom=True, labelbottom=True,
                            labelsize=8)
        self.ax.tick_params(axis="y", which="major",
                            right=True, labelright=True,
                            left=True, labelleft=True,
                            labelsize=8)

        if self.show_default_grid:
            alpha = 0.4 if gen_visible else 1.0
            if gen_visible:
                # The major ticks are pinned to the generated-grid
                # coordinates (so the spine reads those constant values),
                # which means ax.grid(major) alone would collapse the
                # default reference grid onto just those few lines — the
                # user reported it "never shows" once a grid is generated.
                # Keep it an independent layer by drawing the fine
                # reference grid on the MINOR ticks at adaptive spacing.
                self.ax.xaxis.set_minor_locator(
                    AdaptiveGridLocator(self.grid_spacing, self.MAX_AXIS_LABELS))
                self.ax.yaxis.set_minor_locator(
                    AdaptiveGridLocator(self.grid_spacing, self.MAX_AXIS_LABELS))
                self.ax.tick_params(which="minor", length=0)
                # Reference grid on the MINOR ticks only. The major ticks
                # are the generated coordinates, already marked by the
                # solid light-blue lines; turning on the major grid too
                # (which="both") would draw a second dotted line under the
                # blue one at each generated coord, making those reference
                # lines darker than the rest. "minor" keeps the faint
                # reference grid uniform.
                self.ax.grid(True, which="minor", linestyle=":",
                             linewidth=0.5, color="#cccccc", alpha=alpha)
            else:
                self.ax.grid(True, which="major", linestyle=":",
                             linewidth=0.5, color="#cccccc", alpha=alpha)
        else:
            self.ax.grid(False)
        if not gen_visible:
            return
        # Draw the labeled grid manually as Line2D + text overlays. The
        # default-grid enable above is NOT undone here — both layers can
        # render simultaneously when both flags are on (the default is
        # already drawn lighter so the labeled grid stays prominent).
        # Only draw lines visible in the current viewport so pan/zoom
        # stay responsive on large building grids.
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        x_pad = max(abs(x1 - x0) * 0.02, 1e-9)
        y_pad = max(abs(y1 - y0) * 0.02, 1e-9)
        x_lines = [
            ln for ln in grid.x_lines
            if min(x0, x1) - x_pad <= ln.coord <= max(x0, x1) + x_pad
        ]
        y_lines = [
            ln for ln in grid.y_lines
            if min(y0, y1) - y_pad <= ln.coord <= max(y0, y1) + y_pad
        ]
        total_visible = len(x_lines) + len(y_lines)
        label_stride = max(
            1,
            (total_visible + self.MAX_LABELED_GRID_LINES - 1)
            // self.MAX_LABELED_GRID_LINES,
        )

        # The A/B/1/2 letter labels are gated behind the "show grid line
        # labels" toggle. Previously they were always drawn whenever the
        # generated grid was visible, so the toggle looked like it did
        # nothing ("even without it active it already shows"). Now the
        # toggle is the single control: off → just the coloured grid
        # lines + plain numeric coordinates; on → the letters too (here
        # near the spines, and appended to the axis coords by the
        # FixedLocator formatter). The grid LINES always draw — only the
        # letters are gated.
        show_letters = self.show_generated_grid_labels_on_ticks
        # Anchor letter labels with a MIXED transform so they always sit
        # just inside the top / right spine regardless of the current
        # view interval (data-only anchors went stale on scroll-zoom).
        x_label_tx = self.ax.get_xaxis_transform()  # x: data, y: axes
        y_label_tx = self.ax.get_yaxis_transform()  # x: axes, y: data
        for idx, ln in enumerate(x_lines):
            self.ax.axvline(ln.coord, color="#aac8ff", linewidth=0.7,
                            linestyle="-", alpha=0.6, zorder=0)
            if show_letters and idx % label_stride == 0:
                self.ax.text(ln.coord, 1.0, f"  {ln.label}",
                             transform=x_label_tx,
                             color="#3060c0", fontsize=8, va="bottom",
                             ha="center", zorder=1, clip_on=False)
        for idx, ln in enumerate(y_lines):
            self.ax.axhline(ln.coord, color="#aac8ff", linewidth=0.7,
                            linestyle="-", alpha=0.6, zorder=0)
            if show_letters and idx % label_stride == 0:
                self.ax.text(1.0, ln.coord, f"  {ln.label}",
                             transform=y_label_tx,
                             color="#3060c0", fontsize=8, va="center",
                             ha="left", zorder=1, clip_on=False)

    # Origin-axis arrow length, in screen pixels (zoom-invariant).
    _ORIGIN_AXIS_PX = 46.0

    def _draw_origin_axes(self) -> None:
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        if not (x0 <= 0.0 <= x1 and y0 <= 0.0 <= y1):
            return
        # The arrows are anchored at the origin (data) but sized in screen
        # pixels via offset-points, so they stay a constant on-screen
        # length at any zoom. The previous data-relative length
        # (0.08 × view-span) went stale on scroll-zoom — which only calls
        # draw_idle(), never redraw() — so after zooming in the arrows
        # kept their old, larger data length and shot off the canvas.
        L = self._ORIGIN_AXIS_PX
        arrow = dict(arrowstyle="<-", color="#222222", lw=1.4)
        self.ax.plot(0.0, 0.0, marker="o", markersize=4,
                     color="#222222", zorder=8)
        # arrowstyle "<-" puts the head at the xytext (offset) end, so the
        # arrow runs from the origin outward by L pixels.
        self.ax.annotate(
            "", xy=(0.0, 0.0), xycoords="data",
            xytext=(L, 0.0), textcoords="offset points",
            arrowprops=arrow, zorder=8,
        )
        self.ax.annotate(
            "", xy=(0.0, 0.0), xycoords="data",
            xytext=(0.0, L), textcoords="offset points",
            arrowprops=arrow, zorder=8,
        )
        self.ax.annotate("0,0", (0.0, 0.0), xytext=(4, -14),
                         textcoords="offset points", fontsize=8,
                         color="#222222", zorder=9)
        self.ax.annotate("X", (0.0, 0.0), xytext=(L + 4, -2),
                         textcoords="offset points", fontsize=8,
                         color="#222222", zorder=9)
        self.ax.annotate("Y", (0.0, 0.0), xytext=(4, L + 4),
                         textcoords="offset points", fontsize=8,
                         color="#222222", zorder=9)

    _SNAP_MARKER_STYLES = {
        "node":     ("o", "#ff7f0e"),  # filled circle, orange
        "diagram":  ("*", "#d62728"),  # star, red — post mode
        "grid":     ("s", "#1f77b4"),  # square, blue
        "endpoint": ("^", "#9467bd"),  # triangle, purple
        "midpoint": ("D", "#17becf"),  # diamond, cyan
        "project":  ("x", "#2ca02c"),  # x, green
    }

    def _draw_snap_marker(self) -> None:
        """Update the persistent hover/snap-marker artists in place.

        Called both from ``redraw()`` (after ax.clear, when the cached
        artist refs are None) and from ``repaint_hover_marker()`` between
        redraws. Updating xdata/ydata + visibility on a persistent Line2D
        is orders of magnitude cheaper than ``ax.plot()`` + a full scene
        rebuild, which is what makes mouse-move hover feel snappy on
        large models.
        """
        c = self._snap_marker
        if c is not None:
            marker, color = self._SNAP_MARKER_STYLES.get(c.kind, ("o", "#888"))
            if self._snap_marker_artist is None:
                (self._snap_marker_artist,) = self.ax.plot(
                    [c.x], [c.y], marker=marker, color=color, markersize=12,
                    markerfacecolor="none", markeredgewidth=2,
                    linestyle="None", zorder=10,
                )
            else:
                self._snap_marker_artist.set_data([c.x], [c.y])
                self._snap_marker_artist.set_marker(marker)
                self._snap_marker_artist.set_color(color)
                self._snap_marker_artist.set_visible(True)
            if self._hover_marker_artist is not None:
                self._hover_marker_artist.set_visible(False)
            return
        # No real snap candidate → draw a faint "ghost" crosshair at
        # the rectangular-grid-snapped cursor so the user always knows
        # where a left-click would land.
        if self._snap_marker_artist is not None:
            self._snap_marker_artist.set_visible(False)
        if self._hover_xy is None:
            if self._hover_marker_artist is not None:
                self._hover_marker_artist.set_visible(False)
            return
        x, y = self._hover_xy
        if self._hover_marker_artist is None:
            (self._hover_marker_artist,) = self.ax.plot(
                [x], [y], marker="+", color="#888888", markersize=14,
                markeredgewidth=1.5, alpha=0.7, linestyle="None", zorder=10,
            )
        else:
            self._hover_marker_artist.set_data([x], [y])
            self._hover_marker_artist.set_visible(True)

    def repaint_hover_marker(self) -> None:
        """Update only the hover/snap marker and schedule a paint.

        Used by ``_on_canvas_motion`` so mouse motion does not trigger a
        full ``redraw()`` (which would clear and rebuild every artist —
        grid, all elements, labels, diagrams — on every move).
        """
        self._draw_snap_marker()
        self._mpl_canvas.draw_idle()

    def _draw_element_preview(self) -> None:
        if self._element_preview is not None:
            start_node_id, end_x, end_y, kind = self._element_preview
            start = self._model().nodes.get(start_node_id)
            if start is None:
                return
            start_x, start_y = start.x, start.y
        elif self._element_preview_free is not None:
            start_x, start_y, end_x, end_y, kind = self._element_preview_free
        else:
            return
        is_frame = kind == "frame"
        color = "#1f77b4" if is_frame else "#d62728"
        linestyle = "-" if is_frame else "--"
        self.ax.plot(
            [start_x, end_x], [start_y, end_y],
            color=color, linestyle=linestyle, linewidth=2.4,
            alpha=0.55, zorder=3,
        )
        self.ax.plot(
            end_x, end_y, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor=color,
            alpha=0.85, zorder=9,
        )
        if self._element_preview_free is not None:
            # Mark the free start point too (no real node there yet),
            # so the user has visual confirmation of click 1.
            self.ax.plot(
                start_x, start_y, marker="o", markersize=5,
                markerfacecolor="white", markeredgecolor=color,
                alpha=0.85, zorder=9,
            )

    # ── PR #31 — pre-solve validation highlight layer ──────────────

    def set_validation_highlights(
        self,
        *,
        warning_node_ids: set[int] | None = None,
        error_node_ids: set[int] | None = None,
        warning_element_ids: set[int] | None = None,
        error_element_ids: set[int] | None = None,
    ) -> None:
        """Push a set of problem ids onto the validation highlight
        layer.  Any argument omitted (or ``None``) clears that band.

        The canvas owns its own copies of the sets so later mutation of
        the caller's set doesn't bleed into the painted highlights.
        Triggers a redraw.
        """
        self._warning_node_ids = set(warning_node_ids or ())
        self._error_node_ids = set(error_node_ids or ())
        self._warning_element_ids = set(warning_element_ids or ())
        self._error_element_ids = set(error_element_ids or ())
        self.redraw()

    def clear_validation_highlights(self) -> None:
        """Erase the validation highlight layer.  Called after a
        successful solve, after the user fixes the model (via the
        invalidation surface), or when the user explicitly dismisses
        the report."""
        if (
            self._warning_node_ids or self._error_node_ids
            or self._warning_element_ids or self._error_element_ids
        ):
            self._warning_node_ids = set()
            self._error_node_ids = set()
            self._warning_element_ids = set()
            self._error_element_ids = set()
            self.redraw()

    def has_validation_highlights(self) -> bool:
        return bool(
            self._warning_node_ids or self._error_node_ids
            or self._warning_element_ids or self._error_element_ids
        )

    def _draw_validation_highlights(self) -> None:
        """Paint warning (amber) and error (red) markers on top of
        normal model draw but behind the selection layer.

        Elements get a thick translucent band along their length;
        nodes get a coloured ring/cross marker.  Error styling is
        bolder than warning styling so the user can read severity at
        a glance.

        Every same-severity band/marker is plotted in a single
        ``ax.plot`` call (segments separated by ``None``s for lines;
        marker arrays for nodes) so a model with dozens of flagged
        elements still adds only one ``Line2D`` artist per severity
        instead of N — matters for redraw / pan / zoom responsiveness
        on larger models.
        """
        model = self._model()
        element_by_id = {elem.id: elem for elem in model.elements}
        # ── elements: behind selection (zorder < _draw_selection's 1.5) ──
        warn_xs: list[float | None] = []
        warn_ys: list[float | None] = []
        for eid in self._warning_element_ids:
            elem = element_by_id.get(eid)
            if elem is None:
                continue
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            warn_xs.extend([ni.x, nj.x, None])
            warn_ys.extend([ni.y, nj.y, None])
        if warn_xs:
            self.ax.plot(
                warn_xs, warn_ys,
                color="#f0a030", linewidth=5.0, alpha=0.55,
                solid_capstyle="round", zorder=1.4,
            )
        err_xs: list[float | None] = []
        err_ys: list[float | None] = []
        for eid in self._error_element_ids:
            elem = element_by_id.get(eid)
            if elem is None:
                continue
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            err_xs.extend([ni.x, nj.x, None])
            err_ys.extend([ni.y, nj.y, None])
        if err_xs:
            self.ax.plot(
                err_xs, err_ys,
                color="#e03030", linewidth=5.5, alpha=0.65,
                solid_capstyle="round", zorder=1.45,
            )
        # ── nodes: behind selection rings (selection is zorder 11) ──
        warn_nx: list[float] = []
        warn_ny: list[float] = []
        for nid in self._warning_node_ids:
            node = model.nodes.get(nid)
            if node is None:
                continue
            warn_nx.append(node.x)
            warn_ny.append(node.y)
        if warn_nx:
            self.ax.plot(
                warn_nx, warn_ny, marker="D", markersize=13,
                linestyle="None",
                markerfacecolor="none", markeredgecolor="#f0a030",
                markeredgewidth=2.2, zorder=9.5,
            )
        err_nx: list[float] = []
        err_ny: list[float] = []
        for nid in self._error_node_ids:
            node = model.nodes.get(nid)
            if node is None:
                continue
            err_nx.append(node.x)
            err_ny.append(node.y)
        if err_nx:
            self.ax.plot(
                err_nx, err_ny, marker="X", markersize=15,
                linestyle="None",
                markerfacecolor="#e03030", markeredgecolor="#7a1818",
                markeredgewidth=1.6, zorder=10,
            )

    def _draw_selection(self) -> None:
        model = self._model()
        element_by_id = {elem.id: elem for elem in model.elements}
        # Paint element-band highlights behind the element line so the
        # crisp element stroke still reads through. Use one segmented plot
        # instead of one artist per selected element for smoother redraws
        # after window selections.
        sel_xs: list[float | None] = []
        sel_ys: list[float | None] = []
        for eid in self._selected_element_ids:
            elem = element_by_id.get(eid)
            if elem is None:
                continue
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            sel_xs.extend([ni.x, nj.x, None])
            sel_ys.extend([ni.y, nj.y, None])
        if sel_xs:
            self.ax.plot(
                sel_xs, sel_ys, color="#ffbf00", linewidth=6.0, alpha=0.45,
                solid_capstyle="round", zorder=1.5,
            )
        # Paint node highlights on top (orange ring), again batched into
        # a single marker artist.
        sel_nx: list[float] = []
        sel_ny: list[float] = []
        for nid in self._selected_node_ids:
            node = model.nodes.get(nid)
            if node is None:
                continue
            sel_nx.append(node.x)
            sel_ny.append(node.y)
        if sel_nx:
            self.ax.plot(
                sel_nx, sel_ny, marker="o", markersize=13, linestyle="None",
                markerfacecolor="none", markeredgecolor="#ffbf00",
                markeredgewidth=2.4, zorder=11,
            )

    def _draw_drag_rect(self) -> None:
        if self._drag_rect is None:
            return
        from matplotlib.patches import Rectangle
        x0, y0, x1, y1, is_crossing = self._drag_rect
        rx = min(x0, x1)
        ry = min(y0, y1)
        rw = abs(x1 - x0)
        rh = abs(y1 - y0)
        if is_crossing:
            # Right-to-left drag → Crossing mode: dashed green outline,
            # semi-transparent green fill. Selects anything the rect
            # touches.
            edge = "#2da44e"
            face = "#2da44e"
            ls = "--"
        else:
            # Left-to-right drag → Window mode: solid blue outline,
            # semi-transparent blue fill. Selects only fully enclosed
            # objects.
            edge = "#1f6feb"
            face = "#1f6feb"
            ls = "-"
        rect = Rectangle(
            (rx, ry), rw, rh,
            edgecolor=edge, facecolor=face,
            linestyle=ls, linewidth=1.4, alpha=0.10, fill=True,
            zorder=12,
        )
        # Re-apply edge alpha distinctly from face alpha so the outline
        # stays legible even on busy canvases.
        rect.set_edgecolor(edge)
        rect.set_linewidth(1.4)
        self.ax.add_patch(rect)

    def _draw_model(self) -> None:
        model = self._model()
        # Pre-compute the largest force/UDL/point-load magnitude in the
        # model so every drawn arrow length is proportional to the
        # actual load magnitude relative to the rest of the model. We
        # cap it at the model span so a runaway 1e9 load doesn't fill
        # the screen.
        max_force = 0.0
        max_udl = 0.0
        max_point = 0.0
        for ld in model.nodal_loads:
            max_force = max(max_force, (ld.fx ** 2 + ld.fy ** 2) ** 0.5)
        for elem in model.elements:
            for ml in getattr(elem, "member_loads", []):
                if isinstance(ml, UniformDistributedLoad):
                    max_udl = max(
                        max_udl, abs(ml.wy), abs(getattr(ml, "wx", 0.0)),
                    )
                elif isinstance(ml, PointLoad):
                    max_point = max(
                        max_point, abs(ml.py), abs(getattr(ml, "px", 0.0)),
                    )
        span = self._model_span()
        # Map the largest load in each family to ~12% of the model
        # span on screen; smaller loads scale linearly down from there.
        # If no loads of a given family exist, the scale is 0 so the
        # zero-magnitude check inside each draw call falls through.
        target = 0.12 * span
        load_scales = {
            "force": target / max_force if max_force > 0 else 0.0,
            "udl":   target / max_udl   if max_udl   > 0 else 0.0,
            "point": target / max_point if max_point > 0 else 0.0,
        }

        draw_element_ids = len(model.elements) <= self.MAX_AUTO_ELEMENT_LABELS
        draw_node_ids = len(model.nodes) <= self.MAX_AUTO_NODE_LABELS
        frame_xs: list[float | None] = []
        frame_ys: list[float | None] = []
        truss_xs: list[float | None] = []
        truss_ys: list[float | None] = []
        rigid_xs: list[float | None] = []
        rigid_ys: list[float | None] = []
        release_xs: list[float] = []
        release_ys: list[float] = []
        release_edges: list[str] = []

        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            is_frame = isinstance(elem, FrameElement2D)
            if is_frame:
                frame_xs.extend([ni.x, nj.x, None])
                frame_ys.extend([ni.y, nj.y, None])
                # Rigid end offsets: collect the joint→face stubs so a
                # later pass overdraws them in a distinct colour and a
                # thicker stroke (flexible span keeps the normal line).
                e_i = float(getattr(elem, "offset_i", 0.0) or 0.0)
                e_j = float(getattr(elem, "offset_j", 0.0) or 0.0)
                if e_i > 0.0 or e_j > 0.0:
                    Lx, Ly = nj.x - ni.x, nj.y - ni.y
                    L_e = math.hypot(Lx, Ly)
                    if L_e > 1e-12:
                        tx, ty = Lx / L_e, Ly / L_e
                        if e_i > 0.0:
                            rigid_xs.extend(
                                [ni.x, ni.x + e_i * tx, None])
                            rigid_ys.extend(
                                [ni.y, ni.y + e_i * ty, None])
                        if e_j > 0.0:
                            rigid_xs.extend(
                                [nj.x - e_j * tx, nj.x, None])
                            rigid_ys.extend(
                                [nj.y - e_j * ty, nj.y, None])
            else:
                truss_xs.extend([ni.x, nj.x, None])
                truss_ys.extend([ni.y, nj.y, None])
            color = "#1f77b4" if is_frame else "#d62728"
            mx, my = (ni.x + nj.x) / 2, (ni.y + nj.y) / 2
            if draw_element_ids:
                text = self.ax.annotate(
                    f"e{elem.id}", (mx, my), color=color, fontsize=8,
                    ha="center", va="bottom", zorder=4,
                )
                text.set_path_effects([
                    _path_effects.withStroke(linewidth=2.0, foreground="white"),
                ])
            if self.show_section_labels:
                section = model.sections.get(getattr(elem, "section_id", None))
                if section is not None:
                    material = model.materials.get(section.material_id)
                    sec_name = section.name or f"section {section.id}"
                    mat_name = material.name if material and material.name else (
                        f"material {section.material_id}"
                    )
                    self.ax.annotate(
                        f"{sec_name} / {mat_name}", (mx, my),
                        xytext=(0, -14), textcoords="offset points",
                        fontsize=7, ha="center", va="top", color="#555555",
                        bbox=dict(boxstyle="round,pad=0.18",
                                  fc="white", ec="#dddddd", alpha=0.82),
                        zorder=6,
                    )
            if is_frame:
                if elem.release_i:
                    release_xs.append(ni.x + 0.15 * (nj.x - ni.x))
                    release_ys.append(ni.y + 0.15 * (nj.y - ni.y))
                    release_edges.append(color)
                if elem.release_j:
                    release_xs.append(nj.x - 0.15 * (nj.x - ni.x))
                    release_ys.append(nj.y - 0.15 * (nj.y - ni.y))
                    release_edges.append(color)
            self._draw_member_loads(elem, ni, nj, load_scales)

        if self.show_physical_members:
            self._draw_physical_members(model)
        # Physical view: switch centerlines to thin dashed so the translucent
        # body shows through and the analytical line is still clearly marked.
        _phys = self.show_physical_members
        if frame_xs:
            self.ax.plot(frame_xs, frame_ys, color="#1f77b4",
                         linestyle="--" if _phys else "-",
                         linewidth=1.0 if _phys else 2.0, zorder=2)
        if truss_xs:
            self.ax.plot(truss_xs, truss_ys, color="#d62728",
                         linestyle=":" if _phys else "--",
                         linewidth=1.0 if _phys else 2.0, zorder=2)
        if rigid_xs:
            # Rigid end-offset zones: distinct dark colour + thicker
            # stroke over the joint→face stubs, drawn in EVERY view mode
            # (the physical-view hatching is additional, not a
            # replacement). zorder above the member centerline so the
            # rigid zones are visually obvious at both ends.
            self.ax.plot(rigid_xs, rigid_ys, color=_RIGID_ZONE_COLOR,
                         linestyle="-", linewidth=4.6,
                         solid_capstyle="butt", zorder=2.1)
        for rx, ry, edge in zip(release_xs, release_ys, release_edges):
            self.ax.plot(rx, ry, marker="o", color="white", markersize=7,
                         markeredgecolor=edge, zorder=5)

        if self.show_local_axes:
            self._draw_local_axes(model)

        node_xs = [n.x for n in model.nodes.values()]
        node_ys = [n.y for n in model.nodes.values()]
        if node_xs:
            self.ax.plot(node_xs, node_ys, "o", color="black", markersize=6,
                         linestyle="None", zorder=5)
        for nid, n in model.nodes.items():
            if draw_node_ids:
                text = self.ax.annotate(
                    f"n{nid}", (n.x, n.y), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, zorder=6,
                )
                text.set_path_effects([
                    _path_effects.withStroke(linewidth=2.0, foreground="white"),
                ])
            sup = model.supports.get(nid)
            if sup is not None:
                self._draw_support(sup, n.x, n.y)

        if not draw_element_ids or not draw_node_ids:
            hidden: list[str] = []
            if not draw_element_ids:
                hidden.append("element IDs")
            if not draw_node_ids:
                hidden.append("node IDs")
            self.ax.annotate(
                "Dense view: " + " and ".join(hidden) + " hidden",
                (0.02, 0.02), xycoords="axes fraction", fontsize=8,
                color="#666666", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#dddddd", alpha=0.85),
                zorder=20,
            )

        for ld in model.nodal_loads:
            n = model.nodes.get(ld.node_id)
            if n is None:
                continue
            self._draw_nodal_load(ld, n.x, n.y, load_scales["force"])

    def _draw_local_axes(self, model: StructuralModel) -> None:
        # Local-axis overlay (View → Show local axes). Convention is
        # pinned to ``FrameElement2D.transformation_matrix`` /
        # ``_length_cos_sin`` in ``element.py``:
        #     local x = (nj - ni) / L
        #     local y = (-dy, dx) / L           (i.e. 90° CCW from x)
        # We honour the same dense-view cap as element labels so a
        # 10 000-element model doesn't drown in arrows.
        if len(model.elements) > self.MAX_AUTO_ELEMENT_LABELS:
            return
        # Cap arrow length against the visible diagonal so it stays
        # legible at any zoom; on short elements we additionally cap to
        # 18% of L so the arrow never overshoots the member.
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        view_diag = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        diag_cap = 0.025 * view_diag if view_diag > 0 else 0.0
        gray = "#3a3a3a"
        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            dx = nj.x - ni.x
            dy = nj.y - ni.y
            L = (dx * dx + dy * dy) ** 0.5
            if L < 1e-12:
                continue
            ex_x, ex_y = dx / L, dy / L
            ey_x, ey_y = -dy / L, dx / L
            arrow_len = min(0.18 * L, diag_cap) if diag_cap > 0 else 0.18 * L
            mx = (ni.x + nj.x) / 2
            my = (ni.y + nj.y) / 2
            # Local x arrow (mid → mid + len * ex).
            self.ax.annotate(
                "", xy=(mx + arrow_len * ex_x, my + arrow_len * ex_y),
                xytext=(mx, my),
                arrowprops=dict(
                    arrowstyle="->", color=gray, lw=1.0,
                    shrinkA=0, shrinkB=0,
                ),
                annotation_clip=False, zorder=3.5,
            )
            # Local y arrow (mid → mid + len * ey).
            self.ax.annotate(
                "", xy=(mx + arrow_len * ey_x, my + arrow_len * ey_y),
                xytext=(mx, my),
                arrowprops=dict(
                    arrowstyle="->", color=gray, lw=1.0,
                    shrinkA=0, shrinkB=0,
                ),
                annotation_clip=False, zorder=3.5,
            )
            # Tip labels — placed slightly past the arrowhead.
            tx = mx + arrow_len * 1.12 * ex_x
            ty = my + arrow_len * 1.12 * ex_y
            self.ax.text(
                tx, ty, "x", color=gray, fontsize=7,
                ha="center", va="center", zorder=3.6,
            )
            tx = mx + arrow_len * 1.12 * ey_x
            ty = my + arrow_len * 1.12 * ey_y
            self.ax.text(
                tx, ty, "y", color=gray, fontsize=7,
                ha="center", va="center", zorder=3.6,
            )
            # i / j end labels — 6% in from each end so they don't
            # clash with the node marker or the release marker (15%).
            ix, iy = ni.x + 0.06 * dx, ni.y + 0.06 * dy
            jx, jy = nj.x - 0.06 * dx, nj.y - 0.06 * dy
            for tx, ty, lbl in ((ix, iy, "i"), (jx, jy, "j")):
                t = self.ax.text(
                    tx, ty, lbl, color=gray, fontsize=7,
                    ha="center", va="center", zorder=3.6,
                )
                t.set_path_effects([
                    _path_effects.withStroke(linewidth=1.6, foreground="white"),
                ])

    # ── Physical-member overlay ───────────────────────────────────────────────
    # Pure-geometry helpers (physical_member_polygon, joint_overlap_nodes,
    # resolved_default_depth) live in gui_common/geometry.py — imported at
    # the top of this file as _physical_member_polygon, _joint_overlap_nodes,
    # _resolved_default_depth — so they can be tested without importing the
    # Qt/matplotlib stack.

    def _draw_physical_members(self, model: StructuralModel) -> None:
        """Translucent body rectangles per element + true joint overlaps.

        Frame bodies are drawn in blue at alpha=0.25 (zorder=1.2).
        Truss bodies are drawn in red at alpha=0.18 (zorder=1.2).
        Joint overlap regions (geometry-derived convex intersections of
        the two meeting bodies) are drawn as hatched polygons at
        zorder=1.3.  All sit below validation (1.4) and selection (1.5)
        overlays so error/warning glows and selection highlights remain
        clearly visible.

        Section depth source: ``physical_display_thickness`` resolves
        the in-plane envelope per section (I-sections get
        ``max(depth, width)`` — never web thickness).  When no real
        thickness is available the adaptive ``resolved_default_depth``
        kicks in and the fallback element count is reported on the
        status bar.
        """
        default_depth = _resolved_default_depth(model)
        missing = 0
        frame_polys: list[list[tuple[float, float]]] = []
        truss_polys: list[list[tuple[float, float]]] = []
        # Keyed by elem.id so joint_overlap_regions can pair them up.
        element_polygons: dict[int, list[tuple[float, float]]] = {}
        rigid_zone_polys: list[list[tuple[float, float]]] = []

        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            is_frame = isinstance(elem, FrameElement2D)
            d = _physical_display_thickness(elem, model.sections)
            if d <= 0.0:
                d = default_depth if is_frame else 0.5 * default_depth
                if is_frame:
                    missing += 1
            poly = _physical_member_polygon(ni.x, ni.y, nj.x, nj.y, d)
            if poly is None:
                continue
            if is_frame:
                frame_polys.append(poly)
                element_polygons[elem.id] = poly
                # Rigid end-offset zones: hatched body sub-rectangles
                # over [node, face] so the rigid transfer zones stay
                # distinguishable from the flexible body. The thick
                # centerline marking is drawn separately and persists
                # in every view mode.
                e_i = float(getattr(elem, "offset_i", 0.0) or 0.0)
                e_j = float(getattr(elem, "offset_j", 0.0) or 0.0)
                if e_i > 0.0 or e_j > 0.0:
                    L_e = math.hypot(nj.x - ni.x, nj.y - ni.y)
                    if L_e > 1e-12:
                        tx = (nj.x - ni.x) / L_e
                        ty = (nj.y - ni.y) / L_e
                        zones = []
                        if e_i > 0.0:
                            zones.append(((ni.x, ni.y),
                                          (ni.x + e_i * tx,
                                           ni.y + e_i * ty)))
                        if e_j > 0.0:
                            zones.append(((nj.x - e_j * tx,
                                           nj.y - e_j * ty),
                                          (nj.x, nj.y)))
                        for (ax_, ay_), (bx_, by_) in zones:
                            zpoly = _physical_member_polygon(
                                ax_, ay_, bx_, by_, d,
                            )
                            if zpoly is not None:
                                rigid_zone_polys.append(zpoly)
            else:
                truss_polys.append(poly)

        if frame_polys:
            self.ax.add_collection(_PolyCollection(
                frame_polys, facecolor="#1f77b4", alpha=0.25,
                edgecolor="#1f77b4", linewidth=0.5, zorder=1.2,
            ))
        if truss_polys:
            self.ax.add_collection(_PolyCollection(
                truss_polys, facecolor="#d62728", alpha=0.18,
                edgecolor="#d62728", linewidth=0.5, zorder=1.2,
            ))
        if rigid_zone_polys:
            self.ax.add_collection(_PolyCollection(
                rigid_zone_polys, facecolor=_RIGID_ZONE_COLOR,
                alpha=0.35, edgecolor=_RIGID_ZONE_COLOR,
                hatch="xx", linewidth=0.6, zorder=1.25,
            ))

        # Geometry-derived joint overlap shading (replaces the old
        # fixed-square proxy).  Pair-wise convex intersection of the
        # body rectangles — naturally rectangular when sections differ.
        for poly, (cx, cy), (w, h), _pair in _joint_overlap_regions(
            model, element_polygons,
        ):
            self.ax.add_patch(_MplPolygon(
                poly, closed=True,
                hatch="///", facecolor="#f0c060", edgecolor="#806000",
                alpha=0.30, linewidth=0.8, zorder=1.3,
            ))
            # TEMPORARY debug aid: small label showing the overlap bbox
            # size and centroid so reviewers can verify the geometry
            # comes from real member extents.  Remove after sign-off.
            self.ax.text(
                cx, cy,
                f"{w:.2f}×{h:.2f} m\n@ ({cx:.2f}, {cy:.2f})",
                fontsize=6, color="#604000",
                ha="center", va="center", zorder=1.35,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white", edgecolor="#806000",
                    alpha=0.75, linewidth=0.4,
                ),
            )

        self._physical_members_missing_depth = missing

    def _draw_support(self, sup: Support, x: float, y: float) -> None:
        if sup.ux and sup.uy and sup.rz:
            self.ax.plot(x, y, marker="s", markersize=14, color="black",
                         markerfacecolor="none", zorder=4)
        elif sup.ux and sup.uy:
            self.ax.plot(x, y - 0.15, marker="^", markersize=12, color="black",
                         markerfacecolor="white", zorder=4)
        elif sup.uy and not sup.ux:
            self.ax.plot(x, y - 0.15, marker="o", markersize=10, color="black",
                         markerfacecolor="white", zorder=4)
        elif sup.ux and not sup.uy:
            self.ax.plot(x - 0.15, y, marker="o", markersize=10, color="black",
                         markerfacecolor="white", zorder=4)
        else:
            self.ax.plot(x, y, marker="x", markersize=10, color="black", zorder=4)
        labels = []
        for dof in ("ux", "uy", "rz"):
            v = sup.prescribed(dof)
            if v != 0.0 and getattr(sup, dof):
                labels.append(f"{dof}={v:+.3g}")
        if labels:
            self.ax.annotate(", ".join(labels), (x, y),
                             xytext=(5, -15), textcoords="offset points",
                             fontsize=7, color="#444444", zorder=6)

    def _draw_nodal_load(self, ld: NodalLoad, x: float, y: float,
                          force_scale: float) -> None:
        # ``force_scale`` is "world-units of arrow length per kN" so
        # arrow length is directly proportional to the load magnitude
        # (set by _draw_model from the largest nodal load in the model).
        # ``case_alpha`` dims loads belonging to non-active load cases
        # when the host has the "active case only" overlay on (PR-A).
        case_alpha = self._load_case_alpha(ld)
        if force_scale > 0:
            if ld.fx:
                dx = ld.fx * force_scale
                self.ax.annotate(
                    "",
                    xy=(x, y),
                    xytext=(x - dx, y),
                    arrowprops=dict(
                        arrowstyle="->", color="#2ca02c", lw=2,
                        alpha=case_alpha,
                    ),
                    zorder=5,
                )
                self.ax.annotate(f"Fx={ld.fx:+.3g}", (x - dx, y),
                                 xytext=(0, 5), textcoords="offset points",
                                 fontsize=7, color="#2ca02c", zorder=6,
                                 alpha=case_alpha)
            if ld.fy:
                dy = ld.fy * force_scale
                self.ax.annotate(
                    "",
                    xy=(x, y),
                    xytext=(x, y - dy),
                    arrowprops=dict(
                        arrowstyle="->", color="#2ca02c", lw=2,
                        alpha=case_alpha,
                    ),
                    zorder=5,
                )
                self.ax.annotate(f"Fy={ld.fy:+.3g}", (x, y - dy),
                                 xytext=(5, 0), textcoords="offset points",
                                 fontsize=7, color="#2ca02c", zorder=6,
                                 alpha=case_alpha)
        if ld.mz:
            self.ax.annotate(f"M={ld.mz:+.3g}", (x, y), xytext=(8, -8),
                             textcoords="offset points", fontsize=7,
                             color="#2ca02c", zorder=6,
                             alpha=case_alpha)

    def _draw_member_loads(self, elem, ni, nj, load_scales: dict) -> None:
        """Draw each member load in its TRUE direction:

        * ``coord_system == "local"`` — axial component along the member
          tangent, transverse component perpendicular to the member.
        * ``coord_system == "global"`` — qX and qY components in the true
          global X / Y directions, regardless of the member's
          orientation.
        * ``coord_system == "gravity"`` — single component straight along
          global -Y (positive magnitude = downward), regardless of
          member orientation.

        Each non-zero (direction, magnitude) pair produces one set of
        arrows (UDL: six along the element; PointLoad: one at ``a``).
        The arrow "tail offset" sign convention matches the legacy
        renderer so visual orientation stays consistent for existing
        local loads."""
        if not elem.member_loads:
            return
        L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
        if L < 1e-12:
            return
        tx, ty = (nj.x - ni.x) / L, (nj.y - ni.y) / L   # member tangent
        nx, ny = -ty, tx                                # member +y_local

        labels: list[str] = []
        udl_scale = load_scales.get("udl", 0.0)
        point_scale = load_scales.get("point", 0.0)

        for ml in elem.member_loads:
            # PR-A: dim loads belonging to non-active cases when the
            # host has the "active case only" overlay on.
            case_alpha = self._load_case_alpha(ml)
            if isinstance(ml, UniformDistributedLoad):
                labels.append(_label_for_udl(ml))
                if udl_scale > 0:
                    for dx, dy, mag in _udl_visual_components(
                        ml, tx, ty, nx, ny,
                    ):
                        if mag == 0.0:
                            continue
                        self._draw_udl_arrow_strip(
                            ni, nj, dx, dy, mag, udl_scale,
                            case_alpha=case_alpha,
                        )
            elif isinstance(ml, PointLoad):
                labels.append(_label_for_pointload(ml))
                if point_scale > 0:
                    a = max(0.0, min(L, float(ml.a)))
                    bx = ni.x + tx * a
                    by = ni.y + ty * a
                    for dx, dy, mag in _pointload_visual_components(
                        ml, tx, ty, nx, ny,
                    ):
                        if mag == 0.0:
                            continue
                        h = mag * point_scale
                        # Tail OPPOSITE to load direction so the
                        # arrowhead at xy=(bx,by) visually points along
                        # (dx, dy) — matches the nodal-load convention.
                        self.ax.annotate(
                            "",
                            xy=(bx, by),
                            xytext=(bx - dx * h, by - dy * h),
                            arrowprops=dict(
                                arrowstyle="->", color="#9467bd", lw=2,
                                alpha=case_alpha,
                            ),
                            zorder=5,
                        )
            elif isinstance(ml, TrussTemperatureLoad):
                labels.append(f"ΔT {ml.delta_T:+.3g}°")
            elif isinstance(ml, FrameTemperatureLoad):
                labels.append(f"T {ml.t_top:+.3g}/{ml.t_bottom:+.3g}°")
        if labels:
            mx, my = (ni.x + nj.x) / 2, (ni.y + nj.y) / 2
            self.ax.annotate(", ".join(labels), (mx, my),
                             xytext=(0, -12), textcoords="offset points",
                             fontsize=7, color="#9467bd", ha="center", zorder=6)

    def _draw_udl_arrow_strip(
        self, ni, nj, dx: float, dy: float, magnitude: float,
        udl_scale: float, n_arrows: int = 6,
        *,
        case_alpha: float = 1.0,
    ) -> None:
        """Draw ``n_arrows`` evenly spaced arrows along the element in
        the direction ``(dx, dy)`` with length proportional to
        ``magnitude * udl_scale``. The arrowhead lands on the member
        and the tail sits OPPOSITE to the load direction so the visual
        actually points the way the force acts — matching how nodal
        loads are drawn (see ``_draw_nodal_load`` which also offsets
        the tail by ``-`` the force components).

        ``case_alpha`` multiplies the per-arrow alpha (PR-A) so loads
        from non-active cases render at the host's configured dim
        level."""
        h = magnitude * udl_scale
        if h == 0.0:
            return
        arrow_alpha = 0.85 * case_alpha
        for i in range(n_arrows):
            t = (i + 0.5) / n_arrows
            bx = ni.x + (nj.x - ni.x) * t
            by = ni.y + (nj.y - ni.y) * t
            self.ax.annotate(
                "",
                xy=(bx, by),
                xytext=(bx - dx * h, by - dy * h),
                arrowprops=dict(
                    arrowstyle="->", color="#9467bd", lw=1.0,
                    alpha=arrow_alpha,
                ),
                zorder=4,
            )

    def _node_displacement(self, nid: int) -> tuple[float, float]:
        result = self._result
        if result is None or result.D is None:
            return 0.0, 0.0
        emap = result.E_map.get(nid)
        if emap is None:
            return 0.0, 0.0
        D = result.D
        ux = float(D[emap["ux"]]) if emap["ux"] is not None else 0.0
        uy = float(D[emap["uy"]]) if emap["uy"] is not None else 0.0
        return ux, uy

    def _node_rotation(self, nid: int) -> float:
        """Return θ_z at ``nid`` from the global displacement vector, or
        0.0 if the node has no rotational DOF (e.g. truss-only node)."""
        result = self._result
        if result is None or result.D is None:
            return 0.0
        emap = result.E_map.get(nid)
        if emap is None or emap.get("rz") is None:
            return 0.0
        return float(result.D[emap["rz"]])

    @staticmethod
    def _hermite_v(
        r: float, L: float, vi: float, thi: float, vj: float, thj: float,
    ) -> float:
        """Cubic-Hermite transverse interpolation in local frame.

        ``r`` is the dimensionless position 0..1 along the element. Returns
        the transverse displacement v(r) for end DOFs ``[vi, thi, vj, thj]``.
        """
        N1 = 1.0 - 3.0 * r * r + 2.0 * r * r * r
        N2 = L * (r - 2.0 * r * r + r * r * r)
        N3 = 3.0 * r * r - 2.0 * r * r * r
        N4 = L * (-r * r + r * r * r)
        return N1 * vi + N2 * thi + N3 * vj + N4 * thj

    def _frame_deformed_points(
        self, elem: FrameElement2D, scale: float,
    ) -> tuple[list[float], list[float]]:
        """Sample the frame element's deformed centreline.

        Reads ``d_local`` from ``member_results`` — already in the element's
        local frame and, for moment-released ends, back-calculated via static
        condensation — so the Hermite curve is correct even at hinge joints.
        Pure visualization; solver outputs are not modified.
        """
        nodes = self._model().nodes
        ni = nodes[elem.node_i]
        nj = nodes[elem.node_j]
        L, c, s = elem.length_cos_sin(nodes)
        result = self._result
        mr = result.member_results.get(elem.id) if result else None
        if mr is None or "d_local" not in mr:
            return [ni.x, nj.x], [ni.y, nj.y]
        d = mr["d_local"]
        ui_loc, vi_loc, thi = float(d[0]), float(d[1]), float(d[2])
        uj_loc, vj_loc, thj = float(d[3]), float(d[4]), float(d[5])
        n = max(2, int(self.deformed_stations))
        Xs: list[float] = []
        Ys: list[float] = []
        for k in range(n):
            r = k / (n - 1)
            u_loc = (1.0 - r) * ui_loc + r * uj_loc
            v_loc = self._hermite_v(r, L, vi_loc, thi, vj_loc, thj)
            x_def = r * L + scale * u_loc
            y_def = scale * v_loc
            Xs.append(ni.x + c * x_def - s * y_def)
            Ys.append(ni.y + s * x_def + c * y_def)
        return Xs, Ys

    def _draw_deformed(self) -> None:
        result = self._result
        model = self._model()
        if result.D is None:
            return
        span = self._model_span()
        max_disp = max(
            (abs(self._node_displacement(nid)[0]) for nid in model.nodes),
            default=0.0,
        )
        max_disp = max(
            max_disp,
            max((abs(self._node_displacement(nid)[1]) for nid in model.nodes), default=0.0),
        )
        # Extend max_disp to include Hermite transverse amplitudes for frame
        # elements. Without this, a horizontal SS beam (all nodal ux=uy=0)
        # would have max_disp=0 and the deformed shape would be silenced.
        for elem in model.elements:
            if not isinstance(elem, FrameElement2D):
                continue
            mr = result.member_results.get(elem.id)
            if mr is None or "d_local" not in mr:
                continue
            d = mr["d_local"]
            try:
                L, _c, _s = elem.length_cos_sin(model.nodes)
            except (ValueError, ZeroDivisionError):
                continue
            vi, thi, vj, thj = float(d[1]), float(d[2]), float(d[4]), float(d[5])
            for rk in (0.25, 0.5, 0.75):
                max_disp = max(max_disp, abs(self._hermite_v(rk, L, vi, thi, vj, thj)))
        if max_disp <= 0:
            return
        scale = self.deformed_scale * 0.10 * span / max_disp
        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            if isinstance(elem, FrameElement2D):
                try:
                    Xs, Ys = self._frame_deformed_points(elem, scale)
                except (ValueError, ZeroDivisionError):
                    continue
            else:
                # Truss bar stays straight between displaced endpoints —
                # rotations don't contribute to a pin-jointed member.
                uxi, uyi = self._node_displacement(elem.node_i)
                uxj, uyj = self._node_displacement(elem.node_j)
                Xs = [ni.x + scale * uxi, nj.x + scale * uxj]
                Ys = [ni.y + scale * uyi, nj.y + scale * uyj]
            self.ax.plot(
                Xs, Ys,
                color="#ff7f0e", linestyle="-", linewidth=1.5, alpha=0.7,
                zorder=3,
            )
        self.ax.annotate(
            f"deformed × {scale:.2g} (max |u|={max_disp:.3e} m)",
            (0.02, 0.98), xycoords="axes fraction",
            fontsize=8, color="#ff7f0e", va="top",
        )

    def _mode_displacement(self, node_id: int) -> tuple[float, float]:
        """Return (ux, uy) for ``node_id`` in the currently-shown mode."""
        mr = self._modal_result
        if mr is None or mr.dofs is None:
            return 0.0, 0.0
        emap = mr.dofs.active_map.get(node_id)
        if emap is None:
            return 0.0, 0.0
        col = self._modal_mode_idx
        if col < 0 or col >= mr.modes.shape[1]:
            return 0.0, 0.0
        phi = mr.modes[:, col]
        ux = float(phi[emap["ux"]]) if emap["ux"] is not None else 0.0
        uy = float(phi[emap["uy"]]) if emap["uy"] is not None else 0.0
        return ux, uy

    def _draw_mode_shape(self) -> None:
        mr = self._modal_result
        model = self._model()
        k = self._modal_mode_idx
        if mr is None or k < 0 or k >= mr.n_modes:
            return
        span = self._model_span()
        max_disp = 0.0
        for nid in model.nodes:
            ux, uy = self._mode_displacement(nid)
            max_disp = max(max_disp, abs(ux), abs(uy))
        if max_disp <= 0.0:
            return
        # Auto-scale to a tenth of the model span, multiplied by the
        # user-controlled scale slider on the results pane.
        scale = self._modal_scale * 0.10 * span / max_disp
        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            uxi, uyi = self._mode_displacement(elem.node_i)
            uxj, uyj = self._mode_displacement(elem.node_j)
            self.ax.plot(
                [ni.x, nj.x], [ni.y, nj.y],
                color="#888888", linestyle=":", linewidth=1.0, alpha=0.6,
                zorder=2,
            )
            # Skip red overlay for zero-displacement elements (inactive component).
            if uxi**2 + uyi**2 + uxj**2 + uyj**2 >= 1e-20 * max_disp**2:
                self.ax.plot(
                    [ni.x + scale * uxi, nj.x + scale * uxj],
                    [ni.y + scale * uyi, nj.y + scale * uyj],
                    color="#d62728", linestyle="-", linewidth=1.8, alpha=0.85,
                    zorder=4,
                )
        f = float(mr.frequencies[k])
        T = float(mr.periods[k])
        self.ax.annotate(
            f"mode {k + 1} · f = {f:.4g} Hz · T = {T:.4g} s · scale × {scale:.2g}",
            (0.02, 0.98), xycoords="axes fraction",
            fontsize=8, color="#d62728", va="top",
        )

    def _draw_reactions(self) -> None:
        result = self._result
        model = self._model()
        if not result.reactions:
            return
        span = self._model_span()
        max_r = 0.0
        for r in result.reactions.values():
            max_r = max(max_r,
                        ((r.get("ux", 0)) ** 2 + (r.get("uy", 0)) ** 2) ** 0.5)
        if max_r <= 0:
            return
        arrow_len = 0.10 * span / max_r
        for nid, r in result.reactions.items():
            n = model.nodes.get(nid)
            if n is None:
                continue
            rx = r.get("ux", 0.0)
            ry = r.get("uy", 0.0)
            if rx or ry:
                self.ax.annotate(
                    "",
                    xy=(n.x, n.y),
                    xytext=(n.x - rx * arrow_len, n.y - ry * arrow_len),
                    arrowprops=dict(arrowstyle="->", color="#9467bd", lw=2),
                    zorder=5,
                )
                mag = (rx ** 2 + ry ** 2) ** 0.5
                self.ax.annotate(
                    f"R={mag:.3g} kN",
                    (n.x - rx * arrow_len, n.y - ry * arrow_len),
                    fontsize=7, color="#9467bd", zorder=6,
                )
            mz = r.get("rz", 0.0)
            if mz:
                self.ax.annotate(
                    f"M={mz:+.3g}", (n.x, n.y),
                    xytext=(-10, -15), textcoords="offset points",
                    fontsize=7, color="#9467bd", zorder=6,
                )

    def _draw_diagrams(self) -> None:
        # _diagram_critical_points is consumed by _hit_test → the snap
        # engine, so callers can snap the cursor onto the labelled
        # max/min points. Always reset it before drawing so a stale
        # set from a previous result kind doesn't survive.
        self._diagram_critical_points = []
        result = self._result
        model = self._model()
        if not result.member_results:
            return
        span = self._model_span()
        max_ord = 0.0
        per_elem = []
        for elem in model.elements:
            mr = result.member_results.get(elem.id)
            if mr is None:
                continue
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            # Sample density is configurable via View → Diagram stations.
            # Lower counts give a coarser preview; the critical-point
            # search below operates on the same xs / ys, so very low
            # counts (e.g. 5) may miss the true peak between stations —
            # surfaced to the user via the menu tooltip + status hint.
            n = max(2, int(self.diagram_stations))
            xs, ys = _diagram_ordinates(
                elem, ni, nj, mr["f_local"], self.diagram_kind, n_samples=n,
            )
            if xs is None:
                continue
            per_elem.append((elem, ni, nj, xs, ys))
            max_ord = max(max_ord, max(abs(v) for v in ys))
        if max_ord <= 0 or not per_elem:
            return
        scale = self.diagram_scale * 0.12 * span / max_ord
        # Axial diagram keeps a single legacy fill colour; V and M are
        # split by sign and filled blue (positive) / red (negative).
        axial_color = "#8c564b"
        unit = _DIAGRAM_UNITS[self.diagram_kind]
        # Conventional structural orientation: positive sagging moment
        # plots BELOW the member centerline; positive shear / axial
        # plot on the +normal side as before. The flip is display-only
        # — sampled ys, max_ord, scale, critical-point picks, and the
        # hover read-out (_diagram_value) all use the un-flipped ys.
        if self.diagram_kind == "moment":
            ord_sign = -1.0
        else:
            ord_sign = +1.0

        for elem, ni, nj, xs, ys in per_elem:
            L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
            if L < 1e-12:
                continue
            cx, cy = (nj.x - ni.x) / L, (nj.y - ni.y) / L
            nx, ny = -cy, cx

            def offset_point(xx, yy):
                yy_disp = yy * ord_sign
                return (
                    ni.x + xx * cx + scale * yy_disp * nx,
                    ni.y + xx * cy + scale * yy_disp * ny,
                )

            if self.diagram_kind in ("shear", "moment"):
                # Split the sampled curve at interpolated zero crossings
                # and fill each single-sign segment with the sign colour.
                # Each closed polygon goes from the i-end of the segment
                # along the curve to the j-end, then back along the
                # member centerline so the fill sits between the curve
                # and the member (just like the legacy single-fill).
                segments = _diagram_sign_split(list(xs), list(ys))
                for seg_xs, seg_ys, sign in segments:
                    if len(seg_xs) < 2:
                        continue
                    color = _diagram_sign_color(sign)
                    poly_x = []
                    poly_y = []
                    # Outgoing along the diagram curve.
                    for xx, yy in zip(seg_xs, seg_ys):
                        px, py = offset_point(xx, yy)
                        poly_x.append(px)
                        poly_y.append(py)
                    # Outline the curve portion before appending the
                    # centerline closure (avoids recomputing offset_point).
                    self.ax.plot(
                        poly_x, poly_y, color=color, linewidth=1.0, zorder=2,
                    )
                    # Close the polygon back along the member centerline.
                    poly_x.append(ni.x + seg_xs[-1] * cx)
                    poly_y.append(ni.y + seg_xs[-1] * cy)
                    poly_x.append(ni.x + seg_xs[0] * cx)
                    poly_y.append(ni.y + seg_xs[0] * cy)
                    self.ax.fill(
                        poly_x, poly_y, color=color, alpha=0.25, zorder=1,
                    )
            else:
                # Axial — single colour, no sign split.
                color = axial_color
                poly_x = [ni.x]
                poly_y = [ni.y]
                for xx, yy in zip(xs, ys):
                    px, py = offset_point(xx, yy)
                    poly_x.append(px)
                    poly_y.append(py)
                poly_x.append(nj.x)
                poly_y.append(nj.y)
                self.ax.fill(
                    poly_x, poly_y, color=color, alpha=0.25, zorder=1,
                )
                self.ax.plot(
                    poly_x, poly_y, color=color, linewidth=1.0, zorder=2,
                )

            # Per-element critical points: argmax(ys) and argmin(ys).
            # If the diagram is constant (axial), max == min — label
            # only once at midspan. Skip near-zero peaks (clutter).
            threshold = 0.05 * max_ord
            i_max = max(range(len(ys)), key=lambda i: ys[i])
            i_min = min(range(len(ys)), key=lambda i: ys[i])
            picks: list[int] = []
            if abs(ys[i_max] - ys[i_min]) < 1e-12 * max(max_ord, 1.0):
                # Constant diagram — label once near the midspan.
                picks = [len(xs) // 2]
            else:
                if abs(ys[i_max]) > threshold:
                    picks.append(i_max)
                if abs(ys[i_min]) > threshold and i_min != i_max:
                    picks.append(i_min)
            for i in picks:
                xx = xs[i]
                yy = ys[i]
                yy_disp = yy * ord_sign
                # World-coords on the diagram polyline at this sample,
                # using the same display flip applied to the fill.
                world_x = ni.x + xx * cx + scale * yy_disp * nx
                world_y = ni.y + xx * cy + scale * yy_disp * ny
                # World-coords of the matching point on the element
                # axis itself — that's what the snap engine should
                # snap to (so a left-click in modelling tools still
                # lands on the member geometry, not on the offset
                # diagram polyline).
                axis_x = ni.x + xx * cx
                axis_y = ni.y + xx * cy
                # Marker colour follows the sign convention used for the
                # fill (blue positive / red negative for V & M; legacy
                # brown for axial).
                if self.diagram_kind in ("shear", "moment"):
                    marker_color = _diagram_sign_color(1 if yy >= 0 else -1)
                else:
                    marker_color = axial_color
                self.ax.plot(world_x, world_y, marker="o", color=marker_color,
                             markersize=6, markeredgecolor="#222",
                             markeredgewidth=0.8, zorder=4)
                self.ax.annotate(
                    f"{yy:+.3g} {unit}",
                    (world_x, world_y),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color="#222", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="#ffffffaa", ec=marker_color, lw=0.5),
                )
                # Record a snap target at the element-axis position
                # (not the offset polyline) so cursor snap places a
                # cross/arrow exactly on the member.
                self._diagram_critical_points.append({
                    "x": axis_x,
                    "y": axis_y,
                    "value": yy,
                    "unit": unit,
                    "kind": self.diagram_kind,
                    "elem_id": elem.id,
                    "x_loc": xx,
                    "L": L,
                })

        self.ax.annotate(
            f"{self.diagram_kind} diagram × {scale:.2g}  ·  max |·| = {max_ord:.3g} {unit}",
            (0.02, 0.94), xycoords="axes fraction",
            fontsize=8, color=color, va="top",
        )

    def _model_span(self) -> float:
        model = self._model()
        if not model.nodes:
            return 1.0
        xs = [n.x for n in model.nodes.values()]
        ys = [n.y for n in model.nodes.values()]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        return span if span > 1e-9 else 1.0

    def _set_axes_limits(self) -> None:
        """Fit xlim/ylim to the model + grid, then stretch one axis so
        the data box's aspect matches the canvas widget's pixel aspect.

        Stretching is necessary because ``set_aspect("equal",
        adjustable="box")`` shrinks the axes rectangle to whichever
        dimension is shorter at a 1:1 data scale. If we fed it a
        square data box the user would see the grid as a small square
        floating in the centre of a wide window. By computing xlim/ylim
        whose ratio already matches the widget's pixel ratio, the axes
        rectangle ends up filling the widget with the gridlines and
        origin markers running edge to edge.
        """
        model = self._model()
        grid = self._grid_provider()
        xs: list[float] = [n.x for n in model.nodes.values()]
        ys: list[float] = [n.y for n in model.nodes.values()]
        xs += [ln.coord for ln in grid.x_lines]
        ys += [ln.coord for ln in grid.y_lines]
        if xs and ys:
            x_lo, x_hi = min(xs), max(xs)
            y_lo, y_hi = min(ys), max(ys)
            cx = (x_lo + x_hi) / 2.0
            cy = (y_lo + y_hi) / 2.0
            base = max(x_hi - x_lo, y_hi - y_lo, 1.0) / 2.0 * 1.15
        else:
            cx, cy, base = 5.0, 5.0, 6.0

        size = self._mpl_canvas.size()
        w_px = max(size.width(),  1)
        h_px = max(size.height(), 1)
        if w_px >= h_px:
            x_half = base * (w_px / h_px)
            y_half = base
        else:
            x_half = base
            y_half = base * (h_px / w_px)
        self._setting_axes_limits = True
        try:
            self.ax.set_xlim(cx - x_half, cx + x_half)
            self.ax.set_ylim(cy - y_half, cy + y_half)
        finally:
            self._setting_axes_limits = False

    def _on_limits_changed(self, _ax) -> None:
        """Mark the view as user-owned when xlim/ylim change for any
        reason other than our own programmatic fits. The matplotlib
        navigation toolbar's pan/zoom modes route through here, so
        after the user pans/zooms via the toolbar a subsequent resize
        no longer silently re-fits and throws their view away."""
        if not self._setting_axes_limits:
            self._user_view_dirty = True


def _udl_visual_components(
    ml: UniformDistributedLoad, tx: float, ty: float, nx: float, ny: float,
) -> list[tuple[float, float, float]]:
    """Return ``(direction_x, direction_y, magnitude)`` tuples for each
    drawable component of a UDL, given the element's tangent ``(tx, ty)``
    and +y_local normal ``(nx, ny)``.

    * ``"local"``: two components — axial along the member tangent
      (magnitude ``wx``) and transverse perpendicular to it (magnitude
      ``wy``). Either component may be 0; the caller filters those out.
    * ``"global"``: two components — ``qX`` along true global X
      ``(1, 0)`` and ``qY`` along true global Y ``(0, 1)``. The
      direction vectors are in WORLD axes, independent of the member's
      orientation, so an inclined member draws horizontally / vertically
      not perpendicular to itself.
    * ``"gravity"``: one component straight along global ``-Y`` with
      magnitude ``wy``. Positive magnitude → arrows pointing down."""
    cs = getattr(ml, "coord_system", "local")
    if cs == "local":
        return [(tx, ty, ml.wx), (nx, ny, ml.wy)]
    if cs == "global":
        return [(1.0, 0.0, ml.wx), (0.0, 1.0, ml.wy)]
    if cs == "gravity":
        return [(0.0, -1.0, ml.wy)]
    # Defensive: unknown — fall back to legacy local-y rendering.
    return [(nx, ny, ml.wy)]


def _pointload_visual_components(
    ml: PointLoad, tx: float, ty: float, nx: float, ny: float,
) -> list[tuple[float, float, float]]:
    """Mirror :func:`_udl_visual_components` for a point load."""
    cs = getattr(ml, "coord_system", "local")
    if cs == "local":
        return [(tx, ty, ml.px), (nx, ny, ml.py)]
    if cs == "global":
        return [(1.0, 0.0, ml.px), (0.0, 1.0, ml.py)]
    if cs == "gravity":
        return [(0.0, -1.0, ml.py)]
    return [(nx, ny, ml.py)]


def _label_for_udl(ml: UniformDistributedLoad) -> str:
    """Short magnitude-and-direction tag rendered under the element."""
    cs = getattr(ml, "coord_system", "local")
    if cs == "gravity":
        return f"UDL {ml.wy:+.3g} grav"
    if cs == "global":
        if ml.wx != 0.0 and ml.wy != 0.0:
            return f"UDL ({ml.wx:+.3g},{ml.wy:+.3g}) glob"
        if ml.wx != 0.0:
            return f"UDL qX={ml.wx:+.3g} glob"
        return f"UDL qY={ml.wy:+.3g} glob"
    # local
    if ml.wx != 0.0 and ml.wy != 0.0:
        return f"UDL ({ml.wx:+.3g},{ml.wy:+.3g})"
    if ml.wx != 0.0:
        return f"UDL wx={ml.wx:+.3g}"
    return f"UDL {ml.wy:+.3g}"


def _label_for_pointload(ml: PointLoad) -> str:
    cs = getattr(ml, "coord_system", "local")
    suffix = ""
    if cs == "gravity":
        return f"P {ml.py:+.3g}@{ml.a:.3g} grav"
    if cs == "global":
        suffix = " glob"
    if ml.px != 0.0 and ml.py != 0.0:
        return f"P ({ml.px:+.3g},{ml.py:+.3g})@{ml.a:.3g}{suffix}"
    if ml.px != 0.0:
        return f"P px={ml.px:+.3g}@{ml.a:.3g}{suffix}"
    return f"P {ml.py:+.3g}@{ml.a:.3g}{suffix}"


def _shift_pressed() -> bool:
    """True if the Shift modifier is currently held.

    Read at click/motion/release time so tools see the live keyboard
    state — matplotlib's `event.key` is unreliable for modifier-only
    presses across backends/OSes."""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
    except Exception:
        return False
    app = QApplication.instance()
    if app is None:
        return False
    return bool(app.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)


def _point_in_world_rect(
    px: float, py: float,
    rx0: float, ry0: float, rx1: float, ry1: float,
) -> bool:
    """True iff (px, py) lies inside or on the boundary of the axis-
    aligned rectangle with corners (rx0, ry0)–(rx1, ry1).

    Corner order is not assumed — the rect is normalised here so the
    caller can pass world coords in any order (e.g. press point and
    release point of a right-to-left drag)."""
    lo_x = min(rx0, rx1)
    hi_x = max(rx0, rx1)
    lo_y = min(ry0, ry1)
    hi_y = max(ry0, ry1)
    return lo_x <= px <= hi_x and lo_y <= py <= hi_y


def _segment_intersects_rect(
    x1: float, y1: float, x2: float, y2: float,
    rx0: float, ry0: float, rx1: float, ry1: float,
) -> bool:
    """True iff the segment (x1,y1)→(x2,y2) intersects or lies inside the
    rect. Uses Cohen–Sutherland outcodes — both endpoints inside, any
    endpoint inside, or a clipped segment with positive length all count
    as a hit. Inclusive on the boundary."""
    lo_x = min(rx0, rx1)
    hi_x = max(rx0, rx1)
    lo_y = min(ry0, ry1)
    hi_y = max(ry0, ry1)

    LEFT, RIGHT, BOTTOM, TOP = 1, 2, 4, 8

    def code(x: float, y: float) -> int:
        c = 0
        if x < lo_x:
            c |= LEFT
        elif x > hi_x:
            c |= RIGHT
        if y < lo_y:
            c |= BOTTOM
        elif y > hi_y:
            c |= TOP
        return c

    cx1, cy1 = x1, y1
    cx2, cy2 = x2, y2
    c1 = code(cx1, cy1)
    c2 = code(cx2, cy2)
    while True:
        if c1 == 0 or c2 == 0:
            return True            # at least one endpoint in rect
        if c1 & c2:
            return False           # both share an outside region
        out = c1 or c2
        nx, ny = cx1, cy1
        dx = cx2 - cx1
        dy = cy2 - cy1
        if out & TOP:
            nx = cx1 + dx * (hi_y - cy1) / dy if dy else cx1
            ny = hi_y
        elif out & BOTTOM:
            nx = cx1 + dx * (lo_y - cy1) / dy if dy else cx1
            ny = lo_y
        elif out & RIGHT:
            ny = cy1 + dy * (hi_x - cx1) / dx if dx else cy1
            nx = hi_x
        elif out & LEFT:
            ny = cy1 + dy * (lo_x - cx1) / dx if dx else cy1
            nx = lo_x
        if out == c1:
            cx1, cy1 = nx, ny
            c1 = code(cx1, cy1)
        else:
            cx2, cy2 = nx, ny
            c2 = code(cx2, cy2)


def _point_segment_distance_px(px, py, x1, y1, x2, y2,
                                px_per_dx, px_per_dy) -> float:
    ax = x1 * px_per_dx
    ay = y1 * px_per_dy
    bx = x2 * px_per_dx
    by = y2 * px_per_dy
    qx = px * px_per_dx
    qy = py * px_per_dy
    abx, aby = bx - ax, by - ay
    aqx, aqy = qx - ax, qy - ay
    seg_len_sq = abx * abx + aby * aby
    if seg_len_sq < 1e-9:
        return ((qx - ax) ** 2 + (qy - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, (aqx * abx + aqy * aby) / seg_len_sq))
    cx = ax + t * abx
    cy = ay + t * aby
    return ((qx - cx) ** 2 + (qy - cy) ** 2) ** 0.5


_DIAGRAM_UNITS = {"moment": "kN·m", "shear": "kN", "axial": "kN"}
