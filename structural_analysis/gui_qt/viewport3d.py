"""Interactive OpenGL 3D viewport (v0.33 — beta).

A true 3D scene for the structural model: orbit / pan / zoom (the
pyqtgraph GLViewWidget defaults), cursor picking of nodes and
elements, and direct-in-space geometry creation — clicks are resolved
through a ray cast onto a selectable axis-aligned construction plane
(the 3D generalisation of the 2D canvas's work-plane + working-depth
pair).

Dependency policy: ``pyqtgraph`` + ``PyOpenGL`` are OPTIONAL (the
``gl`` extra in pyproject). When they are missing — or the platform
cannot create a GL context — the host shows a friendly message and
every other feature keeps working; nothing else imports this module
eagerly.

All click-interpretation math (projection, picking, rays) lives in
:mod:`structural_analysis.gui_common.spatial` so it stays unit-
testable without a GL context; this module only owns the Qt/GL shell.

Editing goes through the host's command stack (``host.execute``), so
viewport actions are undoable and instantly visible on the 2D canvas;
a fingerprint poll keeps the scene in sync with edits made anywhere
else (canvas tools, dialogs, undo/redo, file open).
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QLabel, QMainWindow, QMessageBox,
    QPushButton, QToolBar,
)

from ..element import TrussElement2D
from ..gui_common.commands import AddMemberCmd, AddNodeCmd
from ..gui_common import spatial

try:  # Optional dependency — see module docstring.
    import pyqtgraph.opengl as gl
    _GL_IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001 — any import failure disables GL
    gl = None
    _GL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def gl_available() -> tuple[bool, str]:
    """Whether the OpenGL stack imported. ``(ok, reason_if_not)``."""
    if gl is None:
        return False, (
            "The 3D viewport needs the optional 'pyqtgraph' + 'PyOpenGL' "
            f"packages (pip install pyqtgraph PyOpenGL). {_GL_IMPORT_ERROR}"
        )
    return True, ""


_PICK_RADIUS_PX = 14.0
_DRAG_THRESHOLD_PX = 5.0

# Construction-plane choices: label → (axis index, axis letter).
_PLANE_CHOICES: tuple[tuple[str, int], ...] = (
    ("Y = const (storey plane)", 1),
    ("Z = const (front plane)", 2),
    ("X = const (side plane)", 0),
)


def _qmatrix_to_np(m) -> np.ndarray:
    """QMatrix4x4 (column-major data()) → row-major numpy 4×4."""
    return np.array(m.data(), dtype=float).reshape(4, 4).T


if gl is not None:

    class _SceneView(gl.GLViewWidget):
        """GLViewWidget that distinguishes clicks from orbit drags.

        Orbit / pan / zoom stay on the default pyqtgraph bindings; a
        left press-release pair that moves less than
        ``_DRAG_THRESHOLD_PX`` is reported to ``on_click`` instead.
        """

        def __init__(self, on_click) -> None:
            super().__init__()
            self._on_click = on_click
            self._press_pos = None

        def mousePressEvent(self, ev) -> None:
            if ev.buttons() & Qt.MouseButton.LeftButton:
                self._press_pos = ev.position()
            super().mousePressEvent(ev)

        def mouseReleaseEvent(self, ev) -> None:
            press = self._press_pos
            self._press_pos = None
            super().mouseReleaseEvent(ev)
            if press is None:
                return
            delta = ev.position() - press
            if (delta.x() ** 2 + delta.y() ** 2) ** 0.5 \
                    <= _DRAG_THRESHOLD_PX:
                self._on_click(float(ev.position().x()),
                               float(ev.position().y()))

else:

    _SceneView = None  # pragma: no cover — GL stack missing


class Viewport3DWindow(QMainWindow):
    """The interactive 3D viewport window (View → 3D viewport (beta))."""

    def __init__(self, host) -> None:
        super().__init__(host)
        self.setWindowTitle("3D viewport (beta)")
        self.resize(900, 700)
        self._host = host
        self._pending_first: tuple[int | None,
                                   tuple[float, float, float]] | None = None
        self._fingerprint: tuple | None = None
        self._selected_nodes: set[int] = set()
        self._selected_elems: set[int] = set()

        self._build_toolbar()

        self._view = _SceneView(on_click=self._handle_click)
        self.setCentralWidget(self._view)
        self._view.setCameraPosition(distance=20.0, elevation=22.0,
                                     azimuth=-60.0)

        self._grid = gl.GLGridItem()
        self._grid.setSize(20, 20)
        self._grid.setSpacing(1, 1)
        self._view.addItem(self._grid)
        self._axes = gl.GLAxisItem()
        self._axes.setSize(3, 3, 3)
        self._view.addItem(self._axes)
        self._scene_items: list = []

        self._poll = QTimer(self)
        self._poll.setInterval(400)
        self._poll.timeout.connect(self._poll_model)
        self._poll.start()

        self.refresh()
        self._update_grid()
        self._status("Orbit: left-drag · pan: middle-drag · zoom: wheel. "
                     "Short left-clicks use the active tool.")

    # ── UI ──

    def _build_toolbar(self) -> None:
        tb = QToolBar("Viewport tools", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addWidget(QLabel(" Tool: ", self))
        self._tool_combo = QComboBox(self)
        self._tool_combo.addItems(
            ["Select", "Add node", "Frame", "Truss"],
        )
        self._tool_combo.currentTextChanged.connect(
            lambda _t: self._reset_pending(),
        )
        tb.addWidget(self._tool_combo)

        tb.addSeparator()
        tb.addWidget(QLabel(" Construction plane: ", self))
        self._plane_combo = QComboBox(self)
        for label, _axis in _PLANE_CHOICES:
            self._plane_combo.addItem(label)
        self._plane_combo.currentIndexChanged.connect(
            lambda _i: self._update_grid(),
        )
        tb.addWidget(self._plane_combo)

        self._depth_spin = QDoubleSpinBox(self)
        self._depth_spin.setRange(-1e9, 1e9)
        self._depth_spin.setDecimals(3)
        self._depth_spin.setSingleStep(1.0)
        self._depth_spin.setPrefix(" at ")
        self._depth_spin.setSuffix(" m ")
        self._depth_spin.valueChanged.connect(
            lambda _v: self._update_grid(),
        )
        tb.addWidget(self._depth_spin)

        tb.addSeparator()
        fit = QPushButton("Fit", self)
        fit.clicked.connect(self._fit_view)
        tb.addWidget(fit)

        self._status_label = QLabel("", self)
        self.statusBar().addWidget(self._status_label, 1)

    def _status(self, text: str) -> None:
        self._status_label.setText(text)

    # ── plane / camera helpers ──

    def _plane(self) -> tuple[int, float]:
        """(axis index, value) of the active construction plane."""
        axis = _PLANE_CHOICES[self._plane_combo.currentIndex()][1]
        return axis, float(self._depth_spin.value())

    def _update_grid(self) -> None:
        axis, value = self._plane()
        self._grid.resetTransform()
        # GLGridItem lies in the XY plane by default (= Z-const).
        if axis == 1:    # Y-const storey plane
            self._grid.rotate(90, 1, 0, 0)
            self._grid.translate(0, value, 0)
        elif axis == 0:  # X-const side plane
            self._grid.rotate(90, 0, 1, 0)
            self._grid.translate(value, 0, 0)
        else:            # Z-const front plane
            self._grid.translate(0, 0, value)
        self._view.update()

    def _mvp(self) -> np.ndarray | None:
        try:
            proj = _qmatrix_to_np(self._view.projectionMatrix())
            view = _qmatrix_to_np(self._view.viewMatrix())
        except Exception:  # noqa: BLE001 — no context yet
            return None
        return proj @ view

    def _fit_view(self) -> None:
        model = self._host.model()
        if not model.nodes:
            return
        pts = np.array([
            (n.x, n.y, getattr(n, "z", 0.0))
            for n in model.nodes.values()
        ])
        center = pts.mean(axis=0)
        span = float(np.max(np.ptp(pts, axis=0))) if len(pts) > 1 else 4.0
        import pyqtgraph as pg
        self._view.opts["center"] = pg.Vector(*center)
        self._view.setCameraPosition(distance=max(span * 1.8, 4.0))
        self._view.update()

    # ── model sync ──

    def _poll_model(self) -> None:
        fp = spatial.model_fingerprint(self._host.model())
        if fp != self._fingerprint:
            self.refresh()

    def refresh(self) -> None:
        """Rebuild every GL item from the host's current model/result."""
        model = self._host.model()
        self._fingerprint = spatial.model_fingerprint(model)
        for item in self._scene_items:
            self._view.removeItem(item)
        self._scene_items = []

        def _xyz(n) -> tuple[float, float, float]:
            return (n.x, n.y, getattr(n, "z", 0.0))

        # Elements as line segments, coloured per kind.
        frame_pts: list = []
        truss_pts: list = []
        for elem in model.elements:
            ni = model.nodes.get(elem.node_i)
            nj = model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            is_truss = (isinstance(elem, TrussElement2D)
                        or getattr(elem, "kind", "") == "truss3d")
            (truss_pts if is_truss else frame_pts).extend(
                [_xyz(ni), _xyz(nj)],
            )
        if frame_pts:
            self._scene_items.append(gl.GLLinePlotItem(
                pos=np.array(frame_pts), mode="lines",
                color=(0.12, 0.47, 0.71, 1.0), width=2.5, antialias=True,
            ))
        if truss_pts:
            self._scene_items.append(gl.GLLinePlotItem(
                pos=np.array(truss_pts), mode="lines",
                color=(0.84, 0.15, 0.16, 1.0), width=2.0, antialias=True,
            ))

        # Nodes (selected ones gold and bigger), supports orange.
        if model.nodes:
            plain = [_xyz(n) for nid, n in model.nodes.items()
                     if nid not in self._selected_nodes]
            if plain:
                self._scene_items.append(gl.GLScatterPlotItem(
                    pos=np.array(plain), size=8.0,
                    color=(0.1, 0.1, 0.1, 1.0), pxMode=True,
                ))
            sel = [_xyz(model.nodes[nid]) for nid in self._selected_nodes
                   if nid in model.nodes]
            if sel:
                self._scene_items.append(gl.GLScatterPlotItem(
                    pos=np.array(sel), size=14.0,
                    color=(1.0, 0.75, 0.0, 1.0), pxMode=True,
                ))
            sup = [_xyz(model.nodes[nid]) for nid in model.supports
                   if nid in model.nodes]
            if sup:
                self._scene_items.append(gl.GLScatterPlotItem(
                    pos=np.array(sup), size=12.0,
                    color=(1.0, 0.5, 0.05, 0.9), pxMode=True,
                ))

        # Deformed overlay (straight displaced chords, like the canvas
        # fallback for 3D results).
        result = getattr(self._host, "_result", None)
        if (result is not None and getattr(result, "status", "") == "ok"
                and result.D is not None and model.nodes):
            disp = self._node_displacements(model, result)
            if disp:
                max_d = max(np.linalg.norm(v) for v in disp.values())
                if max_d > 0:
                    pts = np.array([
                        (n.x, n.y, getattr(n, "z", 0.0))
                        for n in model.nodes.values()
                    ])
                    span = float(np.max(np.ptp(pts, axis=0))) or 1.0
                    scale = 0.10 * span / max_d
                    def_pts: list = []
                    for elem in model.elements:
                        ni = model.nodes.get(elem.node_i)
                        nj = model.nodes.get(elem.node_j)
                        if ni is None or nj is None:
                            continue
                        di = disp.get(elem.node_i, np.zeros(3))
                        dj = disp.get(elem.node_j, np.zeros(3))
                        def_pts.append(np.array(_xyz(ni)) + scale * di)
                        def_pts.append(np.array(_xyz(nj)) + scale * dj)
                    if def_pts:
                        self._scene_items.append(gl.GLLinePlotItem(
                            pos=np.array(def_pts), mode="lines",
                            color=(1.0, 0.5, 0.05, 0.9), width=1.5,
                            antialias=True,
                        ))

        for item in self._scene_items:
            self._view.addItem(item)
        self._view.update()

    @staticmethod
    def _node_displacements(model, result) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for nid in model.nodes:
            em = result.E_map.get(nid)
            if em is None:
                continue
            vec = np.zeros(3)
            for k, dof in enumerate(("ux", "uy", "uz")):
                idx = em.get(dof)
                if idx is not None:
                    vec[k] = float(result.D[idx])
            out[nid] = vec
        return out

    # ── picking / editing ──

    def _pick(self, sx: float, sy: float):
        """Resolve a click to ``("node", id)`` / ``("element", id)`` /
        ``("plane", (x, y, z))`` / None."""
        model = self._host.model()
        mvp = self._mvp()
        if mvp is None:
            return None
        w = max(self._view.width(), 1)
        h = max(self._view.height(), 1)

        if model.nodes:
            ids = list(model.nodes)
            pts = np.array([
                (model.nodes[i].x, model.nodes[i].y,
                 getattr(model.nodes[i], "z", 0.0))
                for i in ids
            ])
            screen, valid = spatial.project_points(pts, mvp, w, h)
            idx = spatial.nearest_point_index(
                screen, valid, sx, sy, _PICK_RADIUS_PX,
            )
            if idx is not None:
                return ("node", ids[idx])

        if model.elements:
            seg_screen = []
            seg_valid = []
            eids = []
            for elem in model.elements:
                ni = model.nodes.get(elem.node_i)
                nj = model.nodes.get(elem.node_j)
                if ni is None or nj is None:
                    continue
                pts = np.array([
                    (ni.x, ni.y, getattr(ni, "z", 0.0)),
                    (nj.x, nj.y, getattr(nj, "z", 0.0)),
                ])
                screen, valid = spatial.project_points(pts, mvp, w, h)
                seg_screen.append(screen)
                seg_valid.append(bool(valid.all()))
                eids.append(elem.id)
            idx = spatial.nearest_segment_index(
                np.array(seg_screen), np.array(seg_valid),
                sx, sy, _PICK_RADIUS_PX,
            )
            if idx is not None:
                return ("element", eids[idx])

        ray = spatial.ray_from_screen(sx, sy, mvp, w, h)
        if ray is None:
            return None
        axis, value = self._plane()
        hit = spatial.ray_axis_plane_intersection(ray[0], ray[1],
                                                  axis, value)
        if hit is None:
            return None
        return ("plane", hit)

    def _reset_pending(self) -> None:
        self._pending_first = None

    def _handle_click(self, sx: float, sy: float) -> None:
        picked = self._pick(sx, sy)
        tool = self._tool_combo.currentText()
        if picked is None:
            self._status("Click misses the construction plane — orbit "
                         "until the plane faces the camera, or switch "
                         "planes.")
            return
        kind, payload = picked

        if tool == "Select":
            if kind == "node":
                self._selected_nodes = {payload}
                self._selected_elems = set()
                self._safe_host_call("select_node", payload)
                n = self._host.model().nodes[payload]
                self._status(
                    f"Node {payload} at ({n.x:g}, {n.y:g}, "
                    f"{getattr(n, 'z', 0.0):g})."
                )
            elif kind == "element":
                self._selected_elems = {payload}
                self._selected_nodes = set()
                self._safe_host_call("select_element", payload)
                self._status(f"Element {payload}.")
            else:
                self._selected_nodes = set()
                self._selected_elems = set()
                self._safe_host_call("clear_selection")
                self._status("Selection cleared.")
            self.refresh()
            return

        if tool == "Add node":
            if kind == "node":
                self._status(f"Node {payload} already there.")
                return
            if kind != "plane":
                self._status("Click empty space on the construction "
                             "plane to place a node.")
                return
            x, y, z = payload
            self._host.execute(AddNodeCmd(x=x, y=y, z=z))
            self.refresh()
            return

        if tool in ("Frame", "Truss"):
            endpoint = self._click_to_endpoint(kind, payload)
            if endpoint is None:
                return
            if self._pending_first is None:
                self._pending_first = endpoint
                self._status(f"{tool}: first point set — click the "
                             "second point.")
                return
            first, second = self._pending_first, endpoint
            self._pending_first = None
            self._create_member(tool.lower(), first, second)
            return

    def _click_to_endpoint(self, kind, payload):
        if kind == "node":
            model = self._host.model()
            n = model.nodes[payload]
            return (payload, (n.x, n.y, getattr(n, "z", 0.0)))
        if kind == "plane":
            return (None, payload)
        self._status("Member endpoints must be nodes or empty space on "
                     "the construction plane (element splitting lives "
                     "on the 2D canvas).")
        return None

    def _create_member(self, kind: str, first, second) -> None:
        model = self._host.model()
        section_id = self._default_section_id(model)
        if section_id is None:
            QMessageBox.warning(
                self, "No sections defined",
                "Define a section first (Edit → Materials…) before "
                "placing elements.",
            )
            return
        nid_i, (xi, yi, zi) = first
        nid_j, (xj, yj, zj) = second
        self._host.execute(AddMemberCmd(
            x_i=xi, y_i=yi, z_i=zi, node_i=nid_i,
            x_j=xj, y_j=yj, z_j=zj, node_j=nid_j,
            kind=kind, section_id=section_id,
        ))
        self._status(
            f"{kind.capitalize()} placed with section {section_id} "
            "(the viewport uses the remembered/first section — edit "
            "via the element inspector for a different one)."
        )
        self.refresh()

    def _default_section_id(self, model) -> int | None:
        sticky = getattr(self._host, "_sticky_element", None)
        if sticky and sticky.get("section_id") in model.sections:
            return sticky["section_id"]
        if model.sections:
            return min(model.sections)
        return None

    def _safe_host_call(self, name: str, *args) -> None:
        fn = getattr(self._host, name, None)
        if callable(fn):
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 — host sync is best-effort
                pass

    def closeEvent(self, event) -> None:
        self._poll.stop()
        super().closeEvent(event)


def open_viewport3d(host) -> "Viewport3DWindow | None":
    """Factory used by the host. Returns None (after showing a help
    message) when the optional GL stack is unavailable."""
    ok, reason = gl_available()
    if not ok:
        QMessageBox.information(host, "3D viewport unavailable", reason)
        return None
    win = Viewport3DWindow(host)
    win.show()
    return win
