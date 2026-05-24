"""Stage-3 3D viewer — extrudes each element along its local axis.

A read-only, geometry-only view. Each element is rendered as a single
:class:`mpl_toolkits.mplot3d.art3d.Poly3DCollection` keyed by element
id in ``_element_meshes``. Future branches will colour those meshes
per face from a force / stress field — the one-mesh-per-element
invariant + the id → mesh mapping are the only forward affordance;
no stub APIs exist yet.

The 2D canvas remains the editing surface; this window has no model
mutations, no solver hooks, and no overlays beyond the undeformed
extruded geometry. Open it from "View → Open 3D viewer" in the main
window.
"""

from __future__ import annotations

from typing import Callable

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QLabel, QMainWindow, QToolBar, QVBoxLayout, QWidget,
)

from ..model import StructuralModel
from ..profiles import section_outline


_FACE_COLOR = "#9ec5e8"
_EDGE_COLOR = "#1f3a5f"
_FALLBACK_SECTION_FRACTION = 0.02   # 2 % of model span when A = 0

# Vertical-axis modes.
#
# Both modes embed the 2D model (which lives in the x-y plane) into a
# 3D world without changing the model coordinates. They differ only in
# which world axis carries the model's elevation (2D model y) and which
# carries the section's out-of-plane width:
#
#   - "y_up" (default): preserves the 2D convention. model x → world X,
#     model y/elevation → world Y, section width → world Z. matplotlib's
#     camera renders Z visually up, so the model lies flat in the X-Y
#     plane like a math/plan extrusion.
#   - "z_up": engineering / structural reading. model x → world X,
#     model y/elevation → world Z, section width → world Y. The
#     building stands upright because matplotlib renders Z visually up.
#
# The mode never reaches the solver / file I/O / model — it only
# changes how nodes are lifted to 3D and which world axis the section's
# extrusion plane lives on.
_ORIENT_Y_UP = "y_up"
_ORIENT_Z_UP = "z_up"
_DEFAULT_ORIENTATION = _ORIENT_Y_UP


def _fallback_size(model: StructuralModel) -> float:
    """Pick a visible size for sections with A=0 — a small fraction of
    the model span so the prism is visible without dominating the view.
    """
    if not model.nodes:
        return 0.1
    xs = [n.x for n in model.nodes.values()]
    ys = [n.y for n in model.nodes.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    return max(span * _FALLBACK_SECTION_FRACTION, 1e-3)


def _world_axes_for(orientation: str
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(in_plane_axis, out_of_plane_axis)`` unit world vectors
    for the given orientation. ``in_plane_axis`` is the world direction
    along which the section's *depth* extrudes (the second component of
    the outline tuple ``(y, z)`` from :func:`section_outline`); for
    completeness ``out_of_plane_axis`` is the *width* direction."""
    if orientation == _ORIENT_Z_UP:
        return (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    # y_up (default).
    return (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))


def _node_world(node_x: float, node_y: float,
                 orientation: str) -> np.ndarray:
    """Lift a 2D node ``(x, y)`` into the 3D world per orientation.
    Elevation goes on Z for z_up, on Y for y_up."""
    if orientation == _ORIENT_Z_UP:
        return np.array([node_x, 0.0, node_y])
    return np.array([node_x, node_y, 0.0])


def _element_frame(p_i: np.ndarray, p_j: np.ndarray,
                    in_plane_axis: np.ndarray,
                    out_of_plane_axis: np.ndarray,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(world_depth, world_width)`` unit vectors that carry
    the section outline's local ``(y, z)`` into the world for an
    element running from ``p_i`` to ``p_j``.

    ``in_plane_axis`` is the orientation's elevation/depth direction
    (Z for z_up, Y for y_up); ``out_of_plane_axis`` is the orientation's
    width direction. For a generic element axis we re-orthonormalise
    so the cross-section is perpendicular to the element axis even
    when the element itself runs along the elevation direction.
    """
    axis = p_j - p_i
    L = float(np.linalg.norm(axis))
    if L < 1e-12:
        raise ValueError("zero-length element")
    local_x = axis / L
    helper = out_of_plane_axis
    if abs(float(np.dot(local_x, helper))) > 0.999:
        helper = in_plane_axis
    world_depth = np.cross(helper, local_x)
    world_depth /= np.linalg.norm(world_depth)
    world_width = np.cross(local_x, world_depth)
    return world_depth, world_width


def _element_mesh(
    p_i: np.ndarray, p_j: np.ndarray,
    outline: list[tuple[float, float]],
    in_plane_axis: np.ndarray,
    out_of_plane_axis: np.ndarray,
) -> list[list[tuple[float, float, float]]]:
    """Build the list of polygon faces (side panels + two end caps) for
    one extruded element. Returned faces feed straight into a single
    :class:`Poly3DCollection`."""
    world_depth, world_width = _element_frame(
        p_i, p_j, in_plane_axis, out_of_plane_axis,
    )

    def world(p: np.ndarray, yc: float, zc: float) -> tuple[float, float, float]:
        v = p + yc * world_depth + zc * world_width
        return (float(v[0]), float(v[1]), float(v[2]))

    cap_i = [world(p_i, yc, zc) for yc, zc in outline]
    cap_j = [world(p_j, yc, zc) for yc, zc in outline]

    faces: list[list[tuple[float, float, float]]] = [cap_i, cap_j]
    n = len(outline)
    for k in range(n):
        k1 = (k + 1) % n
        faces.append([cap_i[k], cap_j[k], cap_j[k1], cap_i[k1]])
    return faces


def _set_equal_aspect(ax, model: StructuralModel,
                      outline_padding: float,
                      orientation: str) -> None:
    """Force equal aspect on a matplotlib 3D axes by inflating the
    short axes to match the longest. matplotlib does not honour
    ``ax.set_aspect('equal')`` in 3D, so we set explicit limits.

    The orientation decides which world axis carries the model's
    elevation (model y), and therefore which axis needs the elevation
    extent vs. only the section-width extent.
    """
    if not model.nodes:
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        return
    xs = [n.x for n in model.nodes.values()]
    ys = [n.y for n in model.nodes.values()]
    pad = outline_padding
    x_lo, x_hi = min(xs) - pad, max(xs) + pad
    elev_lo, elev_hi = min(ys) - pad, max(ys) + pad
    width_lo, width_hi = -pad, pad
    if orientation == _ORIENT_Z_UP:
        y_lo, y_hi = width_lo, width_hi
        z_lo, z_hi = elev_lo, elev_hi
    else:
        y_lo, y_hi = elev_lo, elev_hi
        z_lo, z_hi = width_lo, width_hi
    span = max(x_hi - x_lo, y_hi - y_lo, z_hi - z_lo, 1.0)
    cx = (x_hi + x_lo) / 2
    cy = (y_hi + y_lo) / 2
    cz = (z_hi + z_lo) / 2
    half = span / 2.0
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)


# Default camera angle per orientation (matplotlib elev/azim degrees).
# Z-up gets a slightly head-on tilt so vertical members read as vertical;
# Y-up keeps the existing math/plan-from-above-front angle.
_DEFAULT_VIEW = {
    _ORIENT_Z_UP: (18.0, -65.0),
    _ORIENT_Y_UP: (22.0, -60.0),
}


def _axis_labels_for(orientation: str) -> tuple[str, str, str]:
    if orientation == _ORIENT_Z_UP:
        return ("X (m)", "Y · out-of-plane (m)", "Z · elevation (m)")
    return ("X (m)", "Y · elevation (m)", "Z · out-of-plane (m)")


def _apply_clean_style(ax) -> None:
    """Reduce the default mplot3d clutter: hide back-pane fills, drop
    the inter-tick gridlines, and lighten the spines so the rendered
    structure (not the cube around it) is what the eye lands on."""
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
        try:
            pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        except AttributeError:
            pass
    for line in (ax.xaxis.line, ax.yaxis.line, ax.zaxis.line):
        line.set_color((0.55, 0.55, 0.55, 0.85))
    ax.tick_params(axis="both", colors="#666666", pad=2)


class View3DWindow(QMainWindow):
    """Non-modal 3D viewer for the current model.

    Held as a singleton on the parent MainWindow; re-opening from the
    View menu raises this instance instead of constructing a new one.
    Press the Refresh button to re-read the model after edits.
    """

    def __init__(self, parent: QWidget | None,
                 model_provider: Callable[[], StructuralModel]) -> None:
        super().__init__(parent)
        # Keep the window alive when closed so the singleton on the
        # parent MainWindow remains valid for the next "Open 3D viewer".
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowTitle("3D viewer — extruded geometry")
        self.resize(900, 700)

        self._model_provider = model_provider
        # element id → its Poly3DCollection. Future overlays will look
        # up the mesh by element id to recolour faces — keep this
        # mapping as the public-facing invariant.
        self._element_meshes: dict[int, Poly3DCollection] = {}
        # Active vertical-axis convention. Changing this only re-lifts
        # nodes into 3D and re-applies axis labels / camera; it never
        # mutates the model.
        self._orientation: str = _DEFAULT_ORIENTATION

        self._build_ui()
        # Scroll-wheel zoom is wired exactly once here, not in
        # refresh(), so model edits never accumulate duplicate handlers.
        self._scroll_cid = self.canvas.mpl_connect(
            "scroll_event", self._on_scroll,
        )
        self.refresh()

    # ── layout ──

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(2, 2, 2, 2)

        self._banner = QLabel("", central)
        self._banner.setStyleSheet(
            "color: #7a5a00; background: #fff6dd; padding: 4px;"
            " border: 1px solid #e0c97a;"
        )
        self._banner.setWordWrap(True)
        self._banner.setVisible(False)
        layout.addWidget(self._banner)

        self.fig = Figure(figsize=(8.0, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)

        nav = QToolBar("3D nav", self)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        nav.addWidget(self.toolbar)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, nav)

        actions = QToolBar("3D view", self)
        self.act_refresh = actions.addAction("Refresh", self.refresh)
        self.act_refresh.setStatusTip(
            "Re-read the current model and rebuild the extruded geometry."
        )
        self.act_reset = actions.addAction("Reset view", self._reset_view)
        actions.addSeparator()
        actions.addWidget(QLabel(" Vertical axis: ", actions))
        self._orient_combo = QComboBox(actions)
        self._orient_combo.addItem("Y-up (mathematical)", _ORIENT_Y_UP)
        self._orient_combo.addItem("Z-up (structural)",   _ORIENT_Z_UP)
        # Pick the default — must run *before* connecting the change
        # signal so this initial set doesn't trigger a redundant refresh.
        default_idx = self._orient_combo.findData(_DEFAULT_ORIENTATION)
        self._orient_combo.setCurrentIndex(max(default_idx, 0))
        self._orient_combo.setStatusTip(
            "Switch the vertical-axis convention. "
            "Visualization only — model coordinates, solver, and the "
            "2D canvas are not affected."
        )
        self._orient_combo.currentIndexChanged.connect(
            self._on_orientation_changed,
        )
        actions.addWidget(self._orient_combo)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, actions)

        self.setCentralWidget(central)

    # ── public ──

    def refresh(self) -> None:
        """Rebuild the scene from the current model."""
        model = self._model_provider()
        self.ax.clear()
        _apply_clean_style(self.ax)
        self._element_meshes = {}

        has_manual = any(
            s.shape_type == "manual" for s in model.sections.values()
        )
        if has_manual:
            self._banner.setText(
                "Manual sections (no shape dimensions) are shown using an "
                "approximate square area-equivalent prism (side = √A). For "
                "exact geometry, edit the section and pick a shape."
            )
            self._banner.setVisible(True)
        else:
            self._banner.setVisible(False)

        orientation = self._orientation
        in_plane_axis, out_of_plane_axis = _world_axes_for(orientation)

        fallback = _fallback_size(model)
        max_section = 0.0
        for elem in model.elements:
            section = model.sections.get(getattr(elem, "section_id", None))
            if section is None:
                continue
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            outline = section_outline(section, fallback_size=fallback)
            max_section = max(
                max_section,
                max(abs(y) for y, _ in outline),
                max(abs(z) for _, z in outline),
            )
            p_i = _node_world(ni.x, ni.y, orientation)
            p_j = _node_world(nj.x, nj.y, orientation)
            faces = _element_mesh(
                p_i, p_j, outline, in_plane_axis, out_of_plane_axis,
            )
            # TODO (stage 4): swap face_colors for a per-face array driven by
            # an axial/bending stress field to render a force or stress overlay.
            mesh = Poly3DCollection(
                faces, facecolors=_FACE_COLOR, edgecolors=_EDGE_COLOR,
                linewidths=0.4, alpha=0.92,
            )
            self.ax.add_collection3d(mesh)
            self._element_meshes[elem.id] = mesh

        x_lbl, y_lbl, z_lbl = _axis_labels_for(orientation)
        self.ax.set_xlabel(x_lbl)
        self.ax.set_ylabel(y_lbl)
        self.ax.set_zlabel(z_lbl)
        _set_equal_aspect(
            self.ax, model,
            outline_padding=max(max_section, 0.1),
            orientation=orientation,
        )
        elev, azim = _DEFAULT_VIEW[orientation]
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    # ── private ──

    def _reset_view(self) -> None:
        elev, azim = _DEFAULT_VIEW[self._orientation]
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    def _on_orientation_changed(self, _idx: int) -> None:
        """Switch the world-axis convention and rebuild the geometry.

        Visualisation only — model data, solver, file I/O and the 2D
        canvas are untouched.
        """
        new_mode = self._orient_combo.currentData()
        if new_mode == self._orientation:
            return
        self._orientation = new_mode
        self.refresh()

    def _on_scroll(self, event) -> None:
        """Scroll-wheel zoom around the current view centre.

        Gated so we never conflict with an active toolbar mode (pan or
        zoom-rect): if the user has the toolbar engaged, the toolbar
        owns the interaction and the wheel is a no-op. We also ignore
        events outside our axes (e.g. over the toolbar itself).
        """
        if event.inaxes is not self.ax:
            return
        mode = getattr(self.toolbar, "mode", "")
        if mode:
            return
        # 1.20× per notch; up = zoom in, down = zoom out. Symmetric
        # shrink/grow around the current centre so the camera angle
        # stays put and only the scale changes.
        step = 1.20
        factor = 1.0 / step if event.button == "up" else step
        for lo_hi, setter in (
            (self.ax.get_xlim3d(), self.ax.set_xlim3d),
            (self.ax.get_ylim3d(), self.ax.set_ylim3d),
            (self.ax.get_zlim3d(), self.ax.set_zlim3d),
        ):
            lo, hi = lo_hi
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * factor
            setter(mid - half, mid + half)
        self.canvas.draw_idle()
