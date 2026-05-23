"""MainWindow — Qt main application: menus, toolbar, undo/redo, file I/O.

Mutations flow through ``execute(command)`` which is the single exception
boundary — internals raise ValueError freely; the boundary turns it into a
``QMessageBox.warning`` without mutating the model.
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QRadioButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..file_io import read_input_file
from ..main import run_analysis
from ..model import AnalysisResult, Material, Section, StructuralModel
from .. import __version__, __what_is_new__
from ..gui_common.commands import (
    AddElementCmd,
    AddMemberLoadCmd,
    AddNodeCmd,
    AddOrUpdateMaterialCmd,
    AddOrUpdateSectionCmd,
    ClearMemberLoadsCmd,
    Command,
    DeleteElementCmd,
    DeleteMaterialCmd,
    DeleteNodeCmd,
    DeleteSectionCmd,
    SetGridSystemCmd,
    SetNodalLoadCmd,
    SetSupportCmd,
    UpdateElementCmd,
)
from ..gui_common.file_writer import write_input_file
from ..gui_common.results_view import format_result

from .canvas import HitResult, ModelCanvas
from .controllers import (
    DeleteTool,
    FrameTool,
    MemberLoadTool,
    NodalLoadTool,
    NodeTool,
    SelectTool,
    SupportTool,
    Tool,
    TrussTool,
)
from .dialogs import (
    ElementDialog,
    ElementPropertiesDialog,
    FineNodeDialog,
    GridDialog,
    GridSpacingDialog,
    MaterialListDialog,
    MemberLoadDialog,
    NodalLoadDialog,
    NodePropertiesDialog,
    SupportDialog,
)
from .grid import GridSystem


def _validate_model_for_solve(model: StructuralModel) -> tuple[list[str], list[str]]:
    """Run lightweight pre-solve checks. Returns (fatal, warnings).

    Fatal issues block the solve outright; warnings are non-blocking.
    The solver itself does a more thorough pass during assembly — this
    pass exists so the GUI can show a friendly summary first.
    """
    fatal: list[str] = []
    warnings: list[str] = []

    if not model.materials:
        fatal.append("No materials defined.")
    if not model.sections:
        fatal.append("No sections defined.")
    for sec in model.sections.values():
        if sec.material_id not in model.materials:
            fatal.append(
                f"Section {sec.id} references missing material "
                f"{sec.material_id}."
            )
    if not model.elements:
        fatal.append("Model has no elements.")
    for elem in model.elements:
        if elem.node_i not in model.nodes:
            fatal.append(f"Element {elem.id} references missing start "
                         f"node {elem.node_i}.")
        if elem.node_j not in model.nodes:
            fatal.append(f"Element {elem.id} references missing end "
                         f"node {elem.node_j}.")
    used_nodes = set()
    for elem in model.elements:
        used_nodes.add(elem.node_i)
        used_nodes.add(elem.node_j)
    isolated = sorted(set(model.nodes) - used_nodes)
    if isolated:
        warnings.append(
            f"Isolated nodes (not connected to any element): {isolated}."
        )
    if not model.supports:
        warnings.append(
            "No supports defined — the stiffness matrix will be singular."
        )
    for ld in model.nodal_loads:
        if ld.node_id not in model.nodes:
            fatal.append(
                f"Nodal load references missing node {ld.node_id}."
            )
    return fatal, warnings


def _build_starter_model() -> StructuralModel:
    """Build an empty model pre-populated with two common Materials and
    two matching Sections so the user can immediately draw elements
    without first walking through the Materials / Sections dialogs.

    Picked deliberately so the modal feature works out of the box:
    both materials carry positive density (kg/m³).
    """
    m = StructuralModel(title="Untitled")
    m.materials = {
        1: Material(id=1, name="Steel_S275", E=2.10e8,
                    alpha=1.20e-5, density=7850.0),
        2: Material(id=2, name="Concrete_C30", E=3.30e7,
                    alpha=1.00e-5, density=2500.0),
    }
    m.sections = {
        1: Section(id=1, name="Steel_IPE200",   material_id=1,
                   A=2.85e-3, I=1.94e-5, depth=0.200),
        2: Section(id=2, name="Concrete_30x50", material_id=2,
                   A=0.150,   I=3.125e-3, depth=0.500),
    }
    return m


class MainWindow(QMainWindow):
    """The Qt main window."""

    def __init__(self, *, initial_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Structural Analysis — GUI (Qt)")
        self.resize(1200, 800)

        self._model = _build_starter_model()
        self._grid: GridSystem = GridSystem()
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._modified = False
        self._current_path: Optional[str] = None
        self._result: Optional[AnalysisResult] = None
        self._modal_result = None
        self._modal_results_dialog = None
        self._view3d_window = None
        # Sticky element-creation defaults (None = ask the user on the
        # next pair-click). When the ElementDialog's "Remember" box is
        # checked these are saved and applied silently for subsequent
        # frame/truss pair clicks until the user clears them.
        self._sticky_element: dict | None = None

        self._build_ui()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()

        self._tools: dict[str, Tool] = {
            "select": SelectTool(self),
            "node": NodeTool(self),
            "frame": FrameTool(self),
            "truss": TrussTool(self),
            "support": SupportTool(self),
            "nodal_load": NodalLoadTool(self),
            "member_load": MemberLoadTool(self),
            "delete": DeleteTool(self),
        }
        self._active_tool: Tool = self._tools["select"]
        self._select_tool("select")

        self.canvas.on_click = self._on_canvas_click
        self.canvas.on_motion = self._on_canvas_motion
        self.canvas.on_nav_mode_block = self._on_nav_mode_block

        self._update_title()
        self.canvas.redraw()

        if initial_path:
            # defer to event loop start
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._open_path(initial_path))

    # ── layout ──

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.canvas = ModelCanvas(splitter, lambda: self._model,
                                    grid_provider=lambda: self._grid)
        splitter.addWidget(self.canvas)

        result_frame = QFrame(splitter)
        rlayout = QVBoxLayout(result_frame)
        rlayout.setContentsMargins(2, 2, 2, 2)
        rlayout.addWidget(QLabel("Analysis report:", result_frame))
        self._result_text = QPlainTextEdit(result_frame)
        self._result_text.setReadOnly(True)
        self._result_text.setStyleSheet(
            "font-family: monospace; font-size: 10pt;"
        )
        self._result_text.setPlainText("(no analysis run yet)")
        rlayout.addWidget(self._result_text)
        splitter.addWidget(result_frame)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        # Status bar
        self.setStatusBar(QStatusBar(self))
        self._status_label = QLabel("", self)
        self._coord_label = QLabel("", self)
        self.statusBar().addWidget(self._status_label, stretch=1)
        self.statusBar().addPermanentWidget(self._coord_label)

    def _build_actions(self) -> None:
        self.act_new = QAction("&New", self, shortcut=QKeySequence.StandardKey.New,
                                triggered=self._do_new)
        self.act_open = QAction("&Open…", self, shortcut=QKeySequence.StandardKey.Open,
                                 triggered=self._do_open)
        self.act_save = QAction("&Save", self, shortcut=QKeySequence.StandardKey.Save,
                                 triggered=self._do_save)
        self.act_save_as = QAction("Save &As…", self,
                                    shortcut=QKeySequence.StandardKey.SaveAs,
                                    triggered=self._do_save_as)
        self.act_quit = QAction("&Quit", self, shortcut=QKeySequence.StandardKey.Quit,
                                 triggered=self.close)

        self.act_undo = QAction("&Undo", self, shortcut=QKeySequence.StandardKey.Undo,
                                 triggered=self._do_undo)
        self.act_redo = QAction("&Redo", self, shortcut=QKeySequence.StandardKey.Redo,
                                 triggered=self._do_redo)
        self.act_materials = QAction("&Materials…", self,
                                       triggered=self._open_material_list)
        self.act_add_node_coords = QAction(
            "Add node at &coordinates…", self,
            shortcut="Shift+N",
            triggered=self._do_add_node_at_coords,
        )
        self.act_grid_spacing = QAction("&Grid spacing…", self,
                                          triggered=self._set_grid_spacing)
        self.act_grid_system = QAction("Grid s&ystem…", self,
                                         triggered=self._edit_grid_system)
        self.act_fit_view = QAction("&Fit to view", self, shortcut="Home",
                                      triggered=self._do_fit_view)
        self.act_open_view3d = QAction(
            "Open &3D viewer", self,
            statusTip="Open a separate 3D window with each element "
                       "extruded along its section profile.",
            triggered=self._open_view3d,
        )
        self.act_forget_elem_defaults = QAction(
            "Forget element defaults", self,
            triggered=self._forget_element_defaults,
        )
        self.act_snap = QAction("Snap to grid", self, checkable=True, checked=True,
                                  triggered=self._toggle_snap)
        # Snap-kind toggles
        self._snap_actions: dict[str, QAction] = {}
        for kind, label in [
            ("node",     "Snap: node"),
            ("grid",     "Snap: grid intersection"),
            ("endpoint", "Snap: element endpoint"),
            ("midpoint", "Snap: element midpoint"),
            ("project",  "Snap: nearest on element"),
        ]:
            a = QAction(label, self, checkable=True, checked=True)
            a.triggered.connect(lambda _checked, k=kind: self._toggle_snap_kind(k))
            self._snap_actions[kind] = a

        # Diagram station-count selector (post-processing samples per
        # element). Pure visualization — changing it never re-runs the
        # solver. Default 21 includes the midspan exactly.
        self._station_tooltip = (
            "Station points are for diagram drawing only. More stations "
            "make diagrams smoother but do not change analysis results. "
            "Very low station counts may miss visual peaks between stations."
        )
        self._station_actions: dict[int, QAction] = {}
        station_group = QActionGroup(self)
        station_group.setExclusive(True)
        for n, label in (
            (5,  "5 (coarse)"),
            (11, "11 (simple)"),
            (21, "21 (default)"),
            (51, "51 (smooth)"),
        ):
            a = QAction(label, self, checkable=True, checked=(n == 21))
            a.setToolTip(self._station_tooltip)
            a.setStatusTip(self._station_tooltip)
            a.triggered.connect(lambda _checked, k=n: self._set_diagram_stations(k))
            station_group.addAction(a)
            self._station_actions[n] = a

        # Deformed shape visual amplification scale.
        self._deformed_scale_tooltip = (
            "Visually amplifies the deformed shape. "
            "Does not change analysis results and does not re-run the solver."
        )
        self._deformed_scale_actions: dict[float, QAction] = {}
        deformed_scale_group = QActionGroup(self)
        deformed_scale_group.setExclusive(True)
        for v, label in (
            (0.5,  "0.5× (reduced)"),
            (1.0,  "1× (default)"),
            (2.0,  "2×"),
            (5.0,  "5×"),
            (10.0, "10× (amplified)"),
        ):
            a = QAction(label, self, checkable=True, checked=(v == 1.0))
            a.setToolTip(self._deformed_scale_tooltip)
            a.setStatusTip(self._deformed_scale_tooltip)
            a.triggered.connect(lambda _checked, k=v: self._set_deformed_scale(k))
            deformed_scale_group.addAction(a)
            self._deformed_scale_actions[v] = a

        self.act_solve = QAction("&Solve", self, shortcut="F5",
                                   triggered=self._do_solve)
        self.act_modal = QAction("&Modal analysis…", self, shortcut="F6",
                                   triggered=self._do_modal)
        self.act_clear_result = QAction("&Clear results", self,
                                          triggered=self._clear_result)

        # Tool actions (mutually exclusive)
        self._tool_actions: dict[str, QAction] = {}
        group = QActionGroup(self)
        group.setExclusive(True)
        for name, label, key in [
            ("select",      "Select",      "S"),
            ("node",        "Node",        "N"),
            ("frame",       "Frame",       "F"),
            ("truss",       "Truss",       "T"),
            ("support",     "Support",     None),
            ("nodal_load",  "Nodal load",  None),
            ("member_load", "Member load", None),
            ("delete",      "Delete",      None),
        ]:
            a = QAction(label, self, checkable=True)
            if key:
                a.setShortcut(key)
            a.triggered.connect(lambda _checked, n=name: self._select_tool(n))
            group.addAction(a)
            self._tool_actions[name] = a
        self._tool_actions["select"].setChecked(True)

    def _build_menus(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        m_file.addAction(self.act_new)
        m_file.addAction(self.act_open)
        self._examples_menu = m_file.addMenu("Open &example…")
        self._populate_examples_menu()
        m_file.addAction(self.act_save)
        m_file.addAction(self.act_save_as)
        m_file.addSeparator()
        m_file.addAction(self.act_quit)

        m_edit = self.menuBar().addMenu("&Edit")
        m_edit.addAction(self.act_undo)
        m_edit.addAction(self.act_redo)
        m_edit.addSeparator()
        m_edit.addAction(self.act_add_node_coords)
        m_edit.addAction(self.act_materials)
        m_edit.addAction(self.act_forget_elem_defaults)

        # Top-right corner of the menu bar: version + what's-new summary
        # so the user always sees which features ship in this build.
        self._version_label = QLabel(
            f"  v{__version__} · {__what_is_new__}  ", self,
        )
        self._version_label.setStyleSheet(
            "color: #555; font-size: 9pt; padding-right: 6px;"
        )
        self._version_label.setToolTip(
            f"Structural Analysis GUI v{__version__}\n"
            f"This release: {__what_is_new__}"
        )
        self.menuBar().setCornerWidget(
            self._version_label, Qt.Corner.TopRightCorner,
        )

        m_view = self.menuBar().addMenu("&View")
        m_view.addAction(self.act_fit_view)
        m_view.addAction(self.act_open_view3d)
        m_view.addSeparator()
        m_view.addAction(self.act_grid_system)
        m_view.addAction(self.act_grid_spacing)
        m_view.addAction(self.act_snap)
        m_view.addSeparator()
        for a in self._snap_actions.values():
            m_view.addAction(a)
        m_view.addSeparator()
        stations_menu = m_view.addMenu("Diagram &stations")
        stations_menu.setToolTip(self._station_tooltip)
        stations_menu.setStatusTip(self._station_tooltip)
        for a in self._station_actions.values():
            stations_menu.addAction(a)
        deformed_scale_menu = m_view.addMenu("Deformed &scale")
        deformed_scale_menu.setToolTip(self._deformed_scale_tooltip)
        deformed_scale_menu.setStatusTip(self._deformed_scale_tooltip)
        for a in self._deformed_scale_actions.values():
            deformed_scale_menu.addAction(a)

        m_run = self.menuBar().addMenu("&Run")
        m_run.addAction(self.act_solve)
        m_run.addAction(self.act_modal)
        m_run.addAction(self.act_clear_result)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Tools", self)
        tb.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tb)
        for name in ("select", "node", "frame", "truss",
                     "support", "nodal_load", "member_load", "delete"):
            tb.addAction(self._tool_actions[name])
        tb.addSeparator()
        tb.addAction(self.act_solve)
        tb.addAction(self.act_materials)
        tb.addSeparator()

        # Overlay toggles
        self._overlay_panel = QWidget(self)
        v = QVBoxLayout(self._overlay_panel)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(QLabel("Overlays:", self._overlay_panel))
        self._cb_deformed = QCheckBox("Deformed shape", self._overlay_panel)
        self._cb_deformed.setChecked(True)
        self._cb_deformed.toggled.connect(self._refresh_overlays)
        self._cb_reactions = QCheckBox("Reactions", self._overlay_panel)
        self._cb_reactions.setChecked(True)
        self._cb_reactions.toggled.connect(self._refresh_overlays)
        self._cb_diagrams = QCheckBox("Force diagrams", self._overlay_panel)
        self._cb_diagrams.toggled.connect(self._refresh_overlays)
        self._cb_section_labels = QCheckBox("Section/material names", self._overlay_panel)
        self._cb_section_labels.toggled.connect(self._refresh_overlays)
        v.addWidget(self._cb_deformed)
        v.addWidget(self._cb_reactions)
        v.addWidget(self._cb_diagrams)
        v.addWidget(self._cb_section_labels)

        self._dia_group = QButtonGroup(self._overlay_panel)
        for label, val in [("M (moment)", "moment"),
                           ("V (shear)",  "shear"),
                           ("N (axial)",  "axial")]:
            rb = QRadioButton(label, self._overlay_panel)
            rb.setProperty("diagram_kind", val)
            if val == "moment":
                rb.setChecked(True)
            rb.toggled.connect(self._refresh_overlays)
            self._dia_group.addButton(rb)
            v.addWidget(rb)
        v.addStretch(1)
        tb.addWidget(self._overlay_panel)

    # ── Host protocol used by tools ──

    def model(self) -> StructuralModel:
        return self._model

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_element_preview(
        self, start_node_id: int, end_x: float, end_y: float, kind: str
    ) -> None:
        self.canvas.set_element_preview(start_node_id, end_x, end_y, kind)
        self.canvas.redraw()

    def clear_element_preview(self) -> None:
        self.canvas.clear_element_preview()
        self.canvas.redraw()

    def select_node(self, node_id: int) -> None:
        node = self._model.nodes.get(node_id)
        if node is None:
            self.clear_selection()
            return
        self.canvas.select_node(node_id)
        support = "support" if node_id in self._model.supports else "no support"
        loads = [ld for ld in self._model.nodal_loads if ld.node_id == node_id]
        load_text = f"{len(loads)} nodal load(s)" if loads else "no nodal load"
        self.set_status(
            f"Selected node {node_id}: x={node.x:.3f} m, y={node.y:.3f} m; "
            f"{support}; {load_text}. Right-click for actions."
        )
        self.canvas.redraw()

    def select_element(self, element_id: int) -> None:
        elem = next((e for e in self._model.elements if e.id == element_id), None)
        if elem is None:
            self.clear_selection()
            return
        self.canvas.select_element(element_id)
        ni = self._model.nodes.get(elem.node_i)
        nj = self._model.nodes.get(elem.node_j)
        length = ""
        if ni is not None and nj is not None:
            L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
            length = f", L={L:.3f} m"
        kind = getattr(elem, "kind", elem.__class__.__name__).lower()
        section = self._model.sections.get(getattr(elem, "section_id", None))
        section_name = section.name if section and section.name else "unnamed section"
        self.set_status(
            f"Selected element {element_id}: {kind}, {section_name}, "
            f"nodes {elem.node_i}-{elem.node_j}{length}. Right-click for actions."
        )
        self.canvas.redraw()

    def clear_selection(self) -> None:
        self.canvas.clear_selection()
        self.set_status("Selection cleared. Click a node or element to inspect it.")
        self.canvas.redraw()

    def execute(self, command: Command) -> None:
        try:
            command.do(self._model)
        except ValueError as e:
            QMessageBox.warning(self, "Cannot apply change", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Internal error",
                                  f"{type(e).__name__}: {e}")
            return
        self._undo.append(command)
        self._redo.clear()
        self._modified = True
        self._invalidate_result()
        self._update_title()
        self.canvas.redraw()

    def open_element_dialog_for_pair(
        self, n_i: int, n_j: int, kind: str | None = None
    ) -> None:
        if not self._model.materials:
            QMessageBox.warning(
                self, "No materials defined",
                "Define a material first (Edit → Materials…) before placing elements.",
            )
            return

        # Sticky path: if a previous element-pair click checked "Remember",
        # reuse the remembered section + releases without re-opening the
        # dialog. The active tool's kind (Frame vs Truss) always wins
        # over the remembered kind — otherwise placing a frame via the
        # Frame tool right after a remembered truss-click would silently
        # produce another truss. Releases are frame-only, so they're
        # cleared when the effective kind is "truss". The setting is
        # invalidated automatically if the remembered section no longer
        # exists.
        sticky = self._sticky_element
        if (
            sticky is not None
            and sticky.get("section_id") in self._model.sections
        ):
            effective_kind = kind or sticky["kind"]
            if effective_kind == "frame":
                release_i = sticky["release_i"]
                release_j = sticky["release_j"]
            else:
                release_i = False
                release_j = False
            self.execute(AddElementCmd(
                node_i=n_i, node_j=n_j,
                section_id=sticky["section_id"],
                kind=effective_kind,
                release_i=release_i,
                release_j=release_j,
            ))
            return

        try:
            d = ElementDialog(
                self, model=self._model,
                existing_kind=kind or (sticky or {}).get("kind"),
                existing_section_id=(sticky or {}).get("section_id"),
            )
        except ValueError as e:
            QMessageBox.warning(self, "Cannot add element", str(e))
            return
        if d.exec() == QDialog.DialogCode.Accepted and d.result_value is not None:
            rv = d.result_value
            if rv.get("remember"):
                self._sticky_element = {
                    "kind": rv["kind"],
                    "section_id": rv["section_id"],
                    "release_i": rv["release_i"],
                    "release_j": rv["release_j"],
                }
                # Hint the user that subsequent pair clicks will skip
                # the dialog until they clear the setting.
                sec = self._model.sections[rv["section_id"]]
                label = (f"{rv['kind']} · section {sec.id}"
                         + (f" ({sec.name})" if sec.name else ""))
                self.set_status(
                    f"Reusing element settings: {label}. "
                    f"Switch tool or use Edit → Forget element defaults to reset."
                )
            else:
                self._sticky_element = None
            self.execute(AddElementCmd(
                node_i=n_i, node_j=n_j,
                section_id=rv["section_id"],
                kind=rv["kind"],
                release_i=rv["release_i"],
                release_j=rv["release_j"],
            ))

    def _forget_element_defaults(self) -> None:
        """Clear sticky element-creation settings so the next frame/truss
        pair click re-opens the ElementDialog."""
        if self._sticky_element is None:
            self.set_status("No remembered element settings to clear.")
            return
        self._sticky_element = None
        self.set_status("Cleared remembered element settings.")

    def show_node_menu(self, node_id: int, action: str | None = None) -> None:
        if action == "support":
            self._edit_support(node_id)
            return
        if action == "nodal_load":
            self._edit_nodal_load(node_id)
            return
        menu = QMenu(self)
        a1 = menu.addAction(f"Node {node_id}: edit support…")
        a2 = menu.addAction(f"Node {node_id}: edit nodal load…")
        a3 = menu.addAction(f"Node {node_id}: delete")
        chosen = menu.exec(self.cursor().pos())
        if chosen is a1:
            self._edit_support(node_id)
        elif chosen is a2:
            self._edit_nodal_load(node_id)
        elif chosen is a3:
            self.execute(DeleteNodeCmd(node_id=node_id))

    def show_element_menu(self, elem_id: int, action: str | None = None) -> None:
        if action == "member_load":
            self._add_member_load(elem_id)
            return
        menu = QMenu(self)
        a1 = menu.addAction(f"Element {elem_id}: edit section/material...")
        a2 = menu.addAction(f"Element {elem_id}: show results / FBD...")
        a3 = menu.addAction(f"Element {elem_id}: add member load...")
        a4 = menu.addAction(f"Element {elem_id}: clear member loads")
        a5 = menu.addAction(f"Element {elem_id}: delete")
        chosen = menu.exec(self.cursor().pos())
        if chosen is a1:
            self._edit_element(elem_id)
        elif chosen is a2:
            self._show_element_results(elem_id)
        elif chosen is a3:
            self._add_member_load(elem_id)
        elif chosen is a4:
            self.execute(ClearMemberLoadsCmd(elem_id=elem_id))
        elif chosen is a5:
            self.execute(DeleteElementCmd(elem_id=elem_id))

    def show_node_details(self, node_id: int) -> None:
        try:
            d = NodePropertiesDialog(self, self._model, node_id, self._result)
        except ValueError as e:
            QMessageBox.warning(self, "Node not found", str(e))
            return
        d.exec()

    def show_element_details(self, elem_id: int) -> None:
        try:
            d = ElementPropertiesDialog(self, self._model, elem_id, self._result)
        except ValueError as e:
            QMessageBox.warning(self, "Element not found", str(e))
            return
        d.exec()

    # ── editing flows ──

    def _edit_support(self, node_id: int) -> None:
        d = SupportDialog(self, existing=self._model.supports.get(node_id),
                           node_id=node_id)
        if d.exec() != QDialog.DialogCode.Accepted:
            return
        action, support = d.result_value
        if action == "remove":
            if node_id in self._model.supports:
                self.execute(SetSupportCmd(support=None, node_id=node_id))
            return
        self.execute(SetSupportCmd(support=support, node_id=node_id))

    def _edit_nodal_load(self, node_id: int) -> None:
        existing = next((ld for ld in self._model.nodal_loads
                         if ld.node_id == node_id), None)
        d = NodalLoadDialog(self, existing=existing, node_id=node_id)
        if d.exec() != QDialog.DialogCode.Accepted:
            return
        fx, fy, mz = d.result_value
        self.execute(SetNodalLoadCmd(node_id=node_id, fx=fx, fy=fy, mz=mz))

    def _add_member_load(self, elem_id: int) -> None:
        try:
            d = MemberLoadDialog(self, model=self._model, elem_id=elem_id)
        except ValueError as e:
            QMessageBox.warning(self, "Cannot edit element", str(e))
            return
        if d.exec() != QDialog.DialogCode.Accepted:
            return
        self.execute(AddMemberLoadCmd(elem_id=elem_id, load=d.result_value))

    def _edit_element(self, elem_id: int) -> None:
        elem = next((e for e in self._model.elements if e.id == elem_id), None)
        if elem is None:
            QMessageBox.warning(self, "Cannot edit element",
                                f"Element {elem_id} does not exist.")
            return
        d = ElementDialog(
            self, model=self._model,
            existing_kind=getattr(elem, "kind", None),
            existing_section_id=getattr(elem, "section_id", None),
            existing_release_i=getattr(elem, "release_i", False),
            existing_release_j=getattr(elem, "release_j", False),
            remember_default=False,
        )
        if d.exec() != QDialog.DialogCode.Accepted or d.result_value is None:
            return
        rv = d.result_value
        self.execute(UpdateElementCmd(
            elem_id=elem_id,
            section_id=rv["section_id"],
            kind=rv["kind"],
            release_i=rv["release_i"],
            release_j=rv["release_j"],
        ))
        self.select_element(elem_id)

    def _show_element_results(self, elem_id: int) -> None:
        elem = next((e for e in self._model.elements if e.id == elem_id), None)
        if elem is None:
            QMessageBox.warning(self, "Element results",
                                f"Element {elem_id} does not exist.")
            return
        if self._result is None or self._result.status != "ok":
            QMessageBox.information(
                self, "Element results",
                "Run static analysis first (F5), then open element results.",
            )
            return
        mr = self._result.member_results.get(elem_id)
        if mr is None:
            QMessageBox.information(
                self, "Element results",
                f"No post-processing result is available for element {elem_id}.",
            )
            return
        f_local = [float(v) for v in mr["f_local"]]
        lines = [
            f"Element {elem_id} free-body / local end forces",
            f"Type: {getattr(elem, 'kind', elem.__class__.__name__)}",
            f"Nodes: {elem.node_i} -> {elem.node_j}",
            "",
            "Local member-end forces:",
            f"  i-end: N={f_local[0]:+.6g} kN, V={f_local[1]:+.6g} kN, "
            f"M={f_local[2]:+.6g} kN*m",
            f"  j-end: N={f_local[3]:+.6g} kN, V={f_local[4]:+.6g} kN, "
            f"M={f_local[5]:+.6g} kN*m",
            "",
            "Free-body convention:",
            "  N is local axial force; V is local transverse shear; "
            "M is local end moment.",
            "  Use the canvas M / V / N diagram overlay for span shape "
            "and critical values.",
        ]
        box = QMessageBox(self)
        box.setWindowTitle("Element results / FBD")
        box.setText(f"Element {elem_id} results")
        box.setInformativeText(
            "Open details to see local end forces and the free-body sign convention."
        )
        box.setDetailedText("\n".join(lines))
        box.exec()

    def _open_material_list(self) -> None:
        d = MaterialListDialog(
            self, model=self._model,
            on_add_or_update_material=lambda mat: self.execute(
                AddOrUpdateMaterialCmd(material=mat)),
            on_delete_material=lambda mid: self.execute(
                DeleteMaterialCmd(material_id=mid)),
            on_add_or_update_section=lambda sec: self.execute(
                AddOrUpdateSectionCmd(section=sec)),
            on_delete_section=lambda sid: self.execute(
                DeleteSectionCmd(section_id=sid)),
        )
        d.exec()

    # ── tool selection ──

    def _select_tool(self, name: str) -> None:
        if name not in self._tools:
            return
        self._active_tool.deactivate()
        if name in self._tool_actions:
            self._tool_actions[name].setChecked(True)
        self._active_tool = self._tools[name]
        # Switching to an editing tool clears the static result overlay
        # so the new geometry isn't covered by the previous deformed
        # shape / reactions / diagrams. The result data itself is kept
        # until an edit fires (_invalidate_result handles that).
        if name in {"node", "frame", "truss", "support", "nodal_load",
                    "member_load", "delete"}:
            if self._result is not None:
                self.canvas.clear_result()
            if self._modal_result is not None:
                self.canvas.clear_modal_result()
        # Heads-up for the most common click-doesn't-do-anything trap:
        # matplotlib navigation toolbar's pan / zoom modes silently
        # absorb every left-click on the canvas. Tell the user.
        nav_mode = getattr(self.canvas.toolbar, "mode", "")
        if nav_mode:
            self.set_status(
                f"{self._active_tool.description}  ·  WARNING: matplotlib "
                f"{nav_mode!s} is active on the canvas toolbar — click it "
                f"again to deactivate, otherwise tool clicks are ignored."
            )
        else:
            self._active_tool.activate()

    def _on_canvas_click(self, hit: HitResult, button: str) -> None:
        if button == "right":
            if hit.node_id is not None:
                self.show_node_menu(hit.node_id)
                return
            if hit.element_id is not None:
                self.show_element_menu(hit.element_id)
                return
        try:
            self._active_tool.on_click(hit, button)
        except Exception as e:
            QMessageBox.critical(self, "Tool error",
                                  f"{type(e).__name__}: {e}")

    def _on_nav_mode_block(self, mode_name: str) -> None:
        # Triggered when the user clicks on the canvas while the
        # matplotlib navigation toolbar is in pan or zoom mode (which
        # silently swallows tool clicks). Surface a status message so
        # the user notices instead of thinking the tool is broken.
        self.set_status(
            f"matplotlib {mode_name!s} mode is active — click the "
            f"toolbar icon to turn it off, then your tool clicks will "
            f"work again."
        )

    def _on_canvas_motion(self, hit: HitResult) -> None:
        parts = [f"({hit.x:.3f}, {hit.y:.3f})"]
        if hit.snap_label:
            parts.append(f"Snap: {hit.snap_label}")
        elif hit.node_id is not None:
            parts.append(f"node {hit.node_id}")
        elif hit.element_id is not None:
            parts.append(f"elem {hit.element_id}")
        # Post-mode bonus: when a static result is displayed with a
        # diagram on screen, surface the moment / shear / axial value
        # at the cursor's projected arc-length on the hovered element.
        # The snap engine already pre-emits a "diagram" candidate at
        # the labelled critical points (their snap_label carries the
        # value text), so skip this extra read when the snap kind is
        # already "diagram" to avoid duplicating the same number.
        if (self._result is not None
                and self.canvas.show_diagrams
                and hit.element_id is not None
                and hit.snap_kind != "diagram"):
            value_text = self._diagram_value_text_for_hit(hit)
            if value_text:
                parts.append(value_text)
        self._coord_label.setText("  |  ".join(parts))
        # Repaint canvas if the snap marker changed.
        self.canvas.redraw()
        try:
            self._active_tool.on_motion(hit)
        except Exception:
            pass

    def _diagram_value_text_for_hit(self, hit: HitResult) -> str | None:
        """Return a "Moment: +12.3 kN·m @ x=4.5 m" tail for the status
        bar when the cursor is near an element and a result is loaded.
        Returns ``None`` when no value is meaningful (truss + shear /
        moment, or the kind doesn't apply)."""
        from .canvas import _diagram_value, _DIAGRAM_UNITS
        if self._result is None or not self._result.member_results:
            return None
        elem = next((e for e in self._model.elements
                     if e.id == hit.element_id), None)
        if elem is None:
            return None
        mr = self._result.member_results.get(elem.id)
        if mr is None:
            return None
        ni = self._model.nodes.get(elem.node_i)
        nj = self._model.nodes.get(elem.node_j)
        if ni is None or nj is None:
            return None
        L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
        if L < 1e-12:
            return None
        # Project (hit.x, hit.y) onto the element axis to get arc-length.
        t = ((hit.x - ni.x) * (nj.x - ni.x)
             + (hit.y - ni.y) * (nj.y - ni.y)) / (L * L)
        t = max(0.0, min(1.0, t))
        x_loc = t * L
        kind = self.canvas.diagram_kind
        value = _diagram_value(elem, ni, nj, mr["f_local"], kind, x_loc)
        if value is None:
            return None
        unit = _DIAGRAM_UNITS.get(kind, "")
        label = {"moment": "M", "shear": "V", "axial": "N"}.get(kind, kind)
        return f"{label}={value:+.4g} {unit} @ x={x_loc:.3f} m on e{elem.id}"

    # ── overlay toggles ──

    def _refresh_overlays(self) -> None:
        self.canvas.show_deformed = self._cb_deformed.isChecked()
        self.canvas.show_reactions = self._cb_reactions.isChecked()
        self.canvas.show_diagrams = self._cb_diagrams.isChecked()
        self.canvas.show_section_labels = self._cb_section_labels.isChecked()
        for btn in self._dia_group.buttons():
            if btn.isChecked():
                self.canvas.diagram_kind = btn.property("diagram_kind")
                break
        self.canvas.redraw()

    # ── grid / snap ──

    def _edit_grid_system(self) -> None:
        d = GridDialog(
            self,
            current=self._grid if not self._grid.is_empty() else None,
            model=self._model,
        )
        if d.exec() == QDialog.DialogCode.Accepted and d.result_value is not None:
            # Route through SetGridSystemCmd so undo/redo works.
            new_grid = d.result_value
            self.execute(SetGridSystemCmd(
                new_grid=new_grid,
                getter=lambda: self._grid,
                setter=lambda g: setattr(self, "_grid", g),
            ))

    def _toggle_snap_kind(self, kind: str) -> None:
        if self._snap_actions[kind].isChecked():
            self.canvas.snap_engine.enabled_kinds.add(kind)
        else:
            self.canvas.snap_engine.enabled_kinds.discard(kind)

    def _do_fit_view(self) -> None:
        """Re-fit the canvas axes to enclose the model and grid extent."""
        self.canvas.fit_to_view()
        self.set_status("View fitted to model.")

    def _open_view3d(self) -> None:
        """Open the non-modal 3D viewer, or raise it if already open.

        A single instance is kept on ``self._view3d_window`` so repeated
        clicks don't pile windows up. The viewer reads the model on
        construction and on its Refresh button — it is read-only and
        holds no references back into the model.
        """
        from .view3d import View3DWindow

        if self._view3d_window is None:
            self._view3d_window = View3DWindow(self, lambda: self._model)
        else:
            # Re-sync after any edits since the window was last shown.
            self._view3d_window.refresh()
        self._view3d_window.show()
        self._view3d_window.raise_()
        self._view3d_window.activateWindow()

    def _populate_examples_menu(self) -> None:
        """Fill the File → Open example submenu from ``inputs/``."""
        self._examples_menu.clear()
        inputs_dir = self._examples_dir()
        if not os.path.isdir(inputs_dir):
            self._examples_menu.setEnabled(False)
            return
        entries = sorted(
            f for f in os.listdir(inputs_dir)
            if f.lower().endswith((".txt", ".spa.json"))
            and not f.startswith(".")
        )
        if not entries:
            self._examples_menu.setEnabled(False)
            return
        self._examples_menu.setEnabled(True)
        for fname in entries:
            full = os.path.join(inputs_dir, fname)
            label = self._example_label(full, fname)
            action = QAction(label, self)
            action.triggered.connect(lambda _checked=False, p=full:
                                     self._open_example(p))
            self._examples_menu.addAction(action)

    def _examples_dir(self) -> str:
        # Repository-root/inputs/. structural_analysis/gui_qt/app.py is
        # three levels deep, so ../../../inputs from this file.
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(here, "..", "..", "inputs"))

    def _example_label(self, path: str, fallback: str) -> str:
        # Use the TITLE line from a .txt model when present so the menu
        # shows a human-readable name instead of just q2a_settlement.txt.
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.rstrip() for ln in f.readlines()[:200]]
        except OSError:
            return fallback
        for i, ln in enumerate(lines):
            if ln.strip().upper() == "TITLE" and i + 1 < len(lines):
                title = lines[i + 1].strip()
                if title and not title.startswith("#"):
                    return f"{fallback} — {title}"
        return fallback

    def _open_example(self, path: str) -> None:
        if not self._confirm_discard():
            return
        self._open_path(path)

    def _set_grid_spacing(self) -> None:
        d = GridSpacingDialog(self, current=self.canvas.grid_spacing)
        if d.exec() == QDialog.DialogCode.Accepted and d.result_value is not None:
            try:
                self.canvas.set_grid_spacing(d.result_value)
            except ValueError as e:
                QMessageBox.warning(self, "Invalid input", str(e))

    def _do_add_node_at_coords(self) -> None:
        d = FineNodeDialog(self, model=self._model)
        if d.exec() != QDialog.DialogCode.Accepted or d.result_value is None:
            return
        x, y = d.result_value
        self.execute(AddNodeCmd(x=x, y=y))

    def _toggle_snap(self, checked: bool) -> None:
        self.canvas.toggle_snap(checked)

    def _set_diagram_stations(self, n: int) -> None:
        """Adjust station-count for diagrams + deformed-shape sampling.

        Post-processing only — the solver is not rerun, only the canvas
        is redrawn. A very coarse setting (5) gets a louder warning so
        the demo audience knows the picture is approximate.
        """
        n = int(n)
        self.canvas.diagram_stations = n
        self.canvas.deformed_stations = n
        # Keep the matching checkable QAction in sync (idempotent if the
        # action triggered this call in the first place).
        for k, action in self._station_actions.items():
            action.setChecked(k == n)
        self.canvas.redraw()
        if n <= 5:
            self.set_status(f"Using {n} stations: coarse diagram preview.")
        else:
            self.set_status(
                f"Diagram stations: {n} per element. "
                f"Solver was not rerun — redraw only."
            )

    def _set_deformed_scale(self, v: float) -> None:
        """Set the visual amplification factor for the deformed shape.

        Redraw only — the solver is not rerun and the cached result is kept.
        """
        v = float(v)
        self.canvas.deformed_scale = v
        for k, action in self._deformed_scale_actions.items():
            action.setChecked(k == v)
        self.canvas.redraw()
        self.set_status(
            f"Deformed scale: {v}× — visual amplification only, no re-solve."
        )

    # ── undo / redo ──

    def _do_undo(self) -> None:
        if not self._undo:
            return
        cmd = self._undo.pop()
        try:
            cmd.undo(self._model)
        except Exception as e:
            QMessageBox.critical(self, "Undo failed",
                                  f"{type(e).__name__}: {e}")
            return
        self._redo.append(cmd)
        self._modified = True
        self._invalidate_result()
        self._update_title()
        self.canvas.redraw()

    def _do_redo(self) -> None:
        if not self._redo:
            return
        cmd = self._redo.pop()
        try:
            cmd.do(self._model)
        except Exception as e:
            QMessageBox.critical(self, "Redo failed",
                                  f"{type(e).__name__}: {e}")
            return
        self._undo.append(cmd)
        self._modified = True
        self._invalidate_result()
        self._update_title()
        self.canvas.redraw()

    # ── file menu actions ──

    def _confirm_discard(self) -> bool:
        if not self._modified:
            return True
        ans = QMessageBox.question(
            self, "Unsaved changes",
            "You have unsaved changes. Save before continuing?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if ans == QMessageBox.StandardButton.Cancel:
            return False
        if ans == QMessageBox.StandardButton.Yes:
            return self._do_save()
        return True

    def _do_new(self) -> None:
        if not self._confirm_discard():
            return
        self._model = _build_starter_model()
        self._grid = GridSystem()
        self._undo.clear()
        self._redo.clear()
        self._modified = False
        self._current_path = None
        self._clear_result()
        self._update_title()
        self.canvas.fit_to_view()

    def _do_open(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open input or project file", "",
            "All supported (*.txt *.spa.json *.json);;"
            "Solver input (*.txt);;GUI project (*.spa.json *.json);;"
            "All files (*.*)",
        )
        if not path:
            return
        self._open_path(path)

    def _open_path(self, path: str) -> None:
        is_json = path.lower().endswith(".spa.json") or path.lower().endswith(".json")
        new_view = None
        try:
            if is_json:
                from .project_io import load_project_json
                project = load_project_json(path)
                new_model = project.model
                new_grid = project.grid
                new_view = project.view
            else:
                new_model = read_input_file(path)
                new_grid = GridSystem()
        except FileNotFoundError:
            QMessageBox.warning(self, "Open failed",
                                  f"File not found: {path}")
            return
        except Exception as e:
            QMessageBox.warning(
                self, "Open failed",
                f"Could not parse {os.path.basename(path)}:\n"
                f"{type(e).__name__}: {e}",
            )
            return
        self._model = new_model
        self._grid = new_grid
        self._undo.clear()
        self._redo.clear()
        self._modified = False
        self._current_path = path
        self._clear_result()
        self._update_title()
        # New file → fit the view by default; if the project carries a
        # saved ViewState, that overrides the fit a moment later.
        self.canvas.fit_to_view()
        if new_view is not None:
            self._apply_view_state(new_view)
        self.set_status(f"Opened {path}")

    def _apply_view_state(self, view) -> None:
        if view.xlim is not None:
            self.canvas.ax.set_xlim(view.xlim)
        if view.ylim is not None:
            self.canvas.ax.set_ylim(view.ylim)
        # We've just established a custom view — future redraws should
        # preserve it instead of auto-fitting back to the model extent.
        if view.xlim is not None or view.ylim is not None:
            self.canvas._view_initialised = True
        enabled = set(view.snap_kinds)
        self.canvas.snap_engine.enabled_kinds = set(enabled)
        for kind, action in self._snap_actions.items():
            action.setChecked(kind in enabled)
        # Repaint so the new xlim/ylim and snap-toggle state show up.
        self.canvas.redraw()

    def _do_save(self) -> bool:
        if self._current_path is None:
            return self._do_save_as()
        return self._save_to(self._current_path)

    def _do_save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", "",
            "GUI project (*.spa.json);;Solver input (*.txt);;All files (*.*)",
        )
        if not path:
            return False
        if self._save_to(path):
            self._current_path = path
            self._update_title()
            return True
        return False

    def _save_to(self, path: str) -> bool:
        is_json = path.lower().endswith(".spa.json") or path.lower().endswith(".json")
        try:
            if is_json:
                from .project_io import Project, ViewState, save_project_json
                view = ViewState(
                    xlim=tuple(self.canvas.ax.get_xlim()),
                    ylim=tuple(self.canvas.ax.get_ylim()),
                    snap_kinds=sorted(self.canvas.snap_engine.enabled_kinds),
                )
                project = Project(
                    model=self._model, grid=self._grid,
                    view=view, title=self._model.title,
                )
                save_project_json(project, path)
            else:
                write_input_file(self._model, path)
        except Exception as e:
            QMessageBox.warning(self, "Save failed",
                                  f"{type(e).__name__}: {e}")
            return False
        self._modified = False
        self._update_title()
        self.set_status(f"Saved to {path}")
        return True

    # ── solve ──

    def _do_solve(self) -> None:
        if not self._model.elements:
            QMessageBox.warning(
                self, "Cannot solve",
                "The model has no elements. Draw some nodes and elements first.",
            )
            return

        # Pre-solve validation — collect both fatal and warning issues.
        fatal, warnings = _validate_model_for_solve(self._model)
        if fatal:
            QMessageBox.critical(
                self, "Model not ready to solve",
                "The following problems must be fixed first:\n\n  - "
                + "\n  - ".join(fatal),
            )
            return
        if warnings:
            ans = QMessageBox.question(
                self, "Warnings before solve",
                "The model has these warnings:\n\n  - "
                + "\n  - ".join(warnings)
                + "\n\nSolve anyway?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            )
            if ans != QMessageBox.StandardButton.Ok:
                return
        try:
            self._result = run_analysis(self._model, verbose=False)
        except Exception as e:
            QMessageBox.critical(
                self, "Analysis failed",
                f"{type(e).__name__}: {e}\n\nThe model is unchanged.",
            )
            return
        self.canvas.set_result(self._result)
        self._update_result_text()
        if self._result.status == "ok":
            self.set_status(
                f"Analysis complete · residual = {self._result.residual:.2e}"
            )
        else:
            self.set_status(f"Analysis status: {self._result.status}")

    def _do_modal(self) -> None:
        if not self._model.elements:
            QMessageBox.warning(
                self, "Cannot run modal analysis",
                "The model has no elements. Draw some nodes and elements first.",
            )
            return
        from .dialogs import ModalAnalysisDialog
        from .modal_view import ModalResultsDialog
        from ..modal import solve_modal

        d = ModalAnalysisDialog(self, default_n_modes=6)
        if d.exec() != QDialog.DialogCode.Accepted or d.result_value is None:
            return
        try:
            modal_result = solve_modal(
                self._model,
                n_modes=d.result_value["n_modes"],
                normalisation=d.result_value["normalisation"],
            )
        except ValueError as e:
            QMessageBox.warning(self, "Modal analysis", str(e))
            return
        except Exception as e:
            QMessageBox.critical(
                self, "Modal analysis failed",
                f"{type(e).__name__}: {e}\n\nThe model is unchanged.",
            )
            return

        self._modal_result = modal_result
        self.canvas.set_modal_result(modal_result, mode_idx=0, scale=1.0)

        def _select(mode_idx: int, scale: float) -> None:
            self.canvas.update_modal_view(mode_idx, scale)

        def _on_close() -> None:
            # Closing the results pane clears the modal overlay so the
            # canvas returns to the plain model view.
            self.canvas.clear_modal_result()
            self._modal_results_dialog = None

        # Keep a reference so the non-modal dialog is not garbage-collected.
        self._modal_results_dialog = ModalResultsDialog(
            self, modal_result, on_select=_select, on_close=_on_close,
        )
        self._modal_results_dialog.show()
        self.set_status(
            f"Modal analysis: {modal_result.n_modes} modes · "
            f"f₁ = {float(modal_result.frequencies[0]):.4g} Hz"
        )

    def _clear_result(self) -> None:
        self._result = None
        self._modal_result = None
        self.canvas.clear_result()
        self.canvas.clear_modal_result()
        self._update_result_text()

    def _invalidate_result(self) -> None:
        if self._result is not None:
            self._result = None
            self.canvas.clear_result()
            self._update_result_text()
        if self._modal_result is not None:
            self._modal_result = None
            self.canvas.clear_modal_result()
            if self._modal_results_dialog is not None:
                self._modal_results_dialog.close()
                self._modal_results_dialog = None

    def _update_result_text(self) -> None:
        text = format_result(self._model, self._result) if self._result \
            else "(no analysis run yet)"
        self._result_text.setPlainText(text)

    # ── title / close ──

    def _update_title(self) -> None:
        path = (os.path.basename(self._current_path)
                if self._current_path else "Untitled")
        mark = "*" if self._modified else ""
        self.setWindowTitle(f"{mark}{path} — Structural Analysis GUI (Qt)")

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        super().closeEvent(event)
