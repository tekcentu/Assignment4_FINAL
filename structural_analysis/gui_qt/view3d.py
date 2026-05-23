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
    QLabel, QMainWindow, QToolBar, QVBoxLayout, QWidget,
)

from ..model import StructuralModel
from ..profiles import section_outline


_FACE_COLOR = "#9ec5e8"
_EDGE_COLOR = "#1f3a5f"
_FALLBACK_SECTION_FRACTION = 0.02   # 2 % of model span when A = 0


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


def _element_frame(p_i: np.ndarray, p_j: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(local_y, local_z)`` unit vectors for an element whose
    axis runs from ``p_i`` to ``p_j``. Local z is the global out-of-
    plane axis; local y is in-plane normal to the element axis.

    The 2D model lives in z=0; this keeps the cross-section's depth
    visible in-plane and its width out-of-plane, matching standard
    structural convention.
    """
    axis = p_j - p_i
    L = float(np.linalg.norm(axis))
    if L < 1e-12:
        raise ValueError("zero-length element")
    local_x = axis / L
    local_z = np.array([0.0, 0.0, 1.0])
    # If the element is somehow vertical in 3D (it cannot be from the
    # 2D editor, but guard against it for robustness) flip the helper.
    if abs(np.dot(local_x, local_z)) > 0.999:
        local_z = np.array([0.0, 1.0, 0.0])
    local_y = np.cross(local_z, local_x)
    local_y /= np.linalg.norm(local_y)
    local_z = np.cross(local_x, local_y)
    return local_y, local_z


def _element_mesh(
    p_i: np.ndarray, p_j: np.ndarray,
    outline: list[tuple[float, float]],
) -> list[list[tuple[float, float, float]]]:
    """Build the list of polygon faces (side panels + two end caps) for
    one extruded element. Returned faces feed straight into a single
    :class:`Poly3DCollection`."""
    local_y, local_z = _element_frame(p_i, p_j)

    def world(p: np.ndarray, yc: float, zc: float) -> tuple[float, float, float]:
        v = p + yc * local_y + zc * local_z
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
                      outline_padding: float) -> None:
    """Force equal aspect on a matplotlib 3D axes by inflating the
    short axes to match the longest. matplotlib does not honour
    ``ax.set_aspect('equal')`` in 3D, so we set explicit limits."""
    if not model.nodes:
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        return
    xs = [n.x for n in model.nodes.values()]
    ys = [n.y for n in model.nodes.values()]
    pad = outline_padding
    x_lo, x_hi = min(xs) - pad, max(xs) + pad
    y_lo, y_hi = min(ys) - pad, max(ys) + pad
    z_lo, z_hi = -pad, pad
    span = max(x_hi - x_lo, y_hi - y_lo, z_hi - z_lo, 1.0)
    cx, cy, cz = (x_hi + x_lo) / 2, (y_hi + y_lo) / 2, (z_hi + z_lo) / 2
    half = span / 2.0
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)


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

        self._build_ui()
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

        tb = QToolBar("3D nav", self)
        tb.addWidget(NavigationToolbar2QT(self.canvas, self))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        actions = QToolBar("3D view", self)
        self.act_refresh = actions.addAction("Refresh", self.refresh)
        self.act_refresh.setStatusTip(
            "Re-read the current model and rebuild the extruded geometry."
        )
        self.act_reset = actions.addAction("Reset view", self._reset_view)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, actions)

        self.setCentralWidget(central)

    # ── public ──

    def refresh(self) -> None:
        """Rebuild the scene from the current model."""
        model = self._model_provider()
        self.ax.clear()
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
            p_i = np.array([ni.x, ni.y, 0.0])
            p_j = np.array([nj.x, nj.y, 0.0])
            faces = _element_mesh(p_i, p_j, outline)
            # TODO (stage 4): swap face_colors for a per-face array driven by
            # an axial/bending stress field to render a force or stress overlay.
            mesh = Poly3DCollection(
                faces, facecolors=_FACE_COLOR, edgecolors=_EDGE_COLOR,
                linewidths=0.4, alpha=0.92,
            )
            self.ax.add_collection3d(mesh)
            self._element_meshes[elem.id] = mesh

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        _set_equal_aspect(self.ax, model, outline_padding=max(max_section, 0.1))
        self.canvas.draw_idle()

    # ── private ──

    def _reset_view(self) -> None:
        self.ax.view_init(elev=20.0, azim=-60.0)
        self.canvas.draw_idle()
