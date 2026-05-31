"""MainWindow — Qt main application: menus, toolbar, undo/redo, file I/O.

Mutations flow through ``execute(command)`` which is the single exception
boundary — internals raise ValueError freely; the boundary turns it into a
``QMessageBox.warning`` without mutating the model.
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QAction, QActionGroup, QColor, QIcon, QKeySequence,
    QPainter, QPen, QPixmap,
)
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
from ..main import run_analysis, run_multi_case_analysis
from ..model import AnalysisResult, Material, Section, StructuralModel
from ..multi_case_result import (
    SUM_ALL_KEY,
    MultiCaseAnalysisResult,
    make_active_case_safe,
)
from .. import __version__, __what_is_new__
from ..gui_common.commands import (
    AddElementCmd,
    AddMemberCmd,
    AddMemberLoadCmd,
    AddNodeCmd,
    AddOrUpdateMaterialCmd,
    AddOrUpdateSectionCmd,
    BatchDeleteCmd,
    BatchUpdateElementsCmd,
    ClearMemberLoadsCmd,
    Command,
    DeleteElementCmd,
    DeleteMaterialCmd,
    DeleteMemberLoadCmd,
    DeleteNodeCmd,
    DeleteSectionCmd,
    DrawMemberWithSplitsCmd,
    ReplaceModelCmd,
    SetGridSystemCmd,
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
    AnalysisSettingsDialog,
    BatchAssignDialog,
    BuildingWizardDialog,
    ElementDialog,
    ElementPropertiesDialog,
    FineNodeDialog,
    GridDialog,
    GridSpacingDialog,
    MaterialListDialog,
    MemberLoadDialog,
    NodalLoadManagerDialog,
    NodePropertiesDialog,
    SupportDialog,
)
from .grid import GridSystem


def _make_building_icon() -> QIcon:
    """Paint a small office-building silhouette for the wizard action."""
    pm = QPixmap(24, 24)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(QPen(QColor("#1f3a5f"), 1.5))
    p.setBrush(QColor("#9ec5e8"))
    p.drawRect(4, 6, 16, 16)
    p.setBrush(QColor("#ffffff"))
    for r in range(3):
        for c in range(3):
            p.drawRect(6 + c * 5, 8 + r * 5, 3, 3)
    p.end()
    return QIcon(pm)


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
        # Single-case "active" result (kept as a plain attribute so all
        # existing consumers — canvas, inspector, overlay handlers —
        # keep working). It's a *view* into ``_multi_result.cases``
        # selected by ``_active_case``; ``_push_active_case_to_canvas``
        # is the canonical write path.
        self._result: Optional[AnalysisResult] = None
        # PR-A: wrapper holding one AnalysisResult per solved case
        # plus the SUM_ALL view machinery. None pre-solve.
        self._multi_result: Optional[MultiCaseAnalysisResult] = None
        # Active case name shown in the toolbar combo / window title.
        # Initialises to DEFAULT — guaranteed to exist on every model
        # (see StructuralModel.__post_init__).
        self._active_case: str = "DEFAULT"
        # PR-A canvas overlay: when True, dim all non-active loads;
        # when False, draw all loads at full intensity. Wired to the
        # View → "Active case loads only" toggle.
        self._active_case_loads_only: bool = True
        self._modal_result = None
        self._modal_results_dialog = None
        self._view3d_window = None
        self._mass_summary_window = None
        self._joint_masses_window = None
        # Singleton element-detail inspector. Held alive across closes
        # so right-clicking a different element reuses the same window
        # (see _open_element_inspector). MainWindow owns the
        # "locked while open" UX so the inspector dialog itself can
        # stay non-modal without losing the safety of "no edits while
        # inspecting".
        self._element_inspector: ElementPropertiesDialog | None = None
        self._lockable_actions: list[QAction] = []
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
        self.canvas.on_release = self._on_canvas_release
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
        self.act_building_wizard = QAction(
            _make_building_icon(),
            "&Building wizard…", self,
            shortcut="Ctrl+B",
            statusTip="Generate a portal-frame building from typed stories, "
                       "bays, and dimensions (Ctrl+B).",
            triggered=self._do_building_wizard,
        )
        self.act_batch_assign = QAction(
            "&Batch assign properties…", self,
            triggered=self._do_batch_assign_selected,
        )
        self.act_batch_assign.setToolTip(
            "Apply section / material override to all selected elements "
            "in one undoable step."
        )
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

        self.act_solve = QAction("&Solve all cases", self, shortcut="F5",
                                   triggered=self._do_solve)
        # Shift+F5 runs only the currently-selected case — useful for
        # iterative model edits on a multi-case model where running
        # every case each time is overkill.
        self.act_solve_active = QAction(
            "Solve &active case only", self, shortcut="Shift+F5",
            triggered=self._do_solve_active_only,
        )
        self.act_modal = QAction("&Modal analysis…", self, shortcut="F6",
                                   triggered=self._do_modal)
        self.act_analysis_settings = QAction(
            "Analysis &settings…", self,
            triggered=self._edit_analysis_settings,
        )
        self.act_mass_summary = QAction(
            "&Mass / self-weight summary…", self,
            triggered=self._show_mass_summary,
        )
        self.act_joint_masses = QAction(
            "&Assembled joint masses…", self,
            triggered=self._show_joint_masses,
        )
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

        # Actions disabled while the element-detail inspector is open.
        # Editing the model from underneath an open inspector would
        # produce a stale view of the very element under inspection;
        # locking these on inspector-open enforces the "view-only
        # while inspecting" invariant the user asked for. Anything
        # view-only (pan, zoom, solve, overlay toggles, View menu,
        # 3D viewer, fit) stays enabled and is *intentionally* not in
        # this list.
        self._lockable_actions = [
            self.act_undo, self.act_redo,
            self.act_building_wizard, self.act_add_node_coords,
            self.act_materials, self.act_forget_elem_defaults,
        ]

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
        m_edit.addAction(self.act_building_wizard)
        m_edit.addAction(self.act_add_node_coords)
        m_edit.addAction(self.act_batch_assign)
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
        m_view.addSeparator()
        # Active-case loads-only toggle (PR-A v0.18).
        self.act_active_case_loads_only = QAction(
            "&Active case loads only", self, checkable=True,
            checked=self._active_case_loads_only,
            triggered=self._on_active_case_loads_only_toggled,
        )
        self.act_active_case_loads_only.setToolTip(
            "When on, loads attached to non-active load cases render "
            "dimmed on the canvas. When off, every load draws at full "
            "intensity regardless of which case is active."
        )
        m_view.addAction(self.act_active_case_loads_only)
        # Load Case Manager dialog (View → Load &cases…).
        self.act_load_cases = QAction(
            "Load &cases…", self,
            triggered=self._show_load_case_manager,
        )
        self.act_load_cases.setToolTip(
            "Add, rename, delete, enable/disable load cases. Also sets "
            "which case absorbs the self-weight contribution."
        )
        m_view.addAction(self.act_load_cases)
        # Load Combination Manager (View → Load com&binations…).
        self.act_load_combinations = QAction(
            "Load com&binations…", self,
            triggered=self._show_load_combination_manager,
        )
        self.act_load_combinations.setToolTip(
            "Add, rename, delete coefficient combinations of load cases "
            "(e.g. 1.2 DEAD + 1.6 LIVE). Combinations are derived views "
            "computed from solved case results."
        )
        m_view.addAction(self.act_load_combinations)

        m_run = self.menuBar().addMenu("&Run")
        m_run.addAction(self.act_solve)
        m_run.addAction(self.act_solve_active)
        m_run.addAction(self.act_modal)
        m_run.addSeparator()
        m_run.addAction(self.act_analysis_settings)
        m_run.addAction(self.act_mass_summary)
        m_run.addAction(self.act_joint_masses)
        m_run.addSeparator()
        m_run.addAction(self.act_clear_result)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Tools", self)
        tb.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tb)
        for name in ("select", "node", "frame", "truss",
                     "support", "nodal_load", "member_load", "delete"):
            tb.addAction(self._tool_actions[name])
        tb.addSeparator()
        tb.addAction(self.act_building_wizard)

        # Case selector — sits IMMEDIATELY above Solve so the
        # toolbar reads "[case combo] → Solve". The combo is hidden
        # behind a thin wrapper widget that adds a label; the combo
        # itself lives on ``self._case_combo`` so tests and code can
        # toggle it directly.
        from PyQt6.QtWidgets import QComboBox
        self._case_combo = QComboBox(self)
        self._case_combo.setToolTip(
            "Active load case — which solved result is shown on the "
            "canvas and in the element-detail inspector. Switching "
            "here updates diagrams, deformed shape, and reactions."
        )
        self._case_combo.setMinimumWidth(110)
        self._case_combo.currentTextChanged.connect(
            self._on_active_case_changed
        )
        case_wrap = QWidget(self)
        case_wrap_layout = QVBoxLayout(case_wrap)
        case_wrap_layout.setContentsMargins(2, 0, 2, 0)
        case_wrap_layout.setSpacing(1)
        case_wrap_layout.addWidget(QLabel("Case:", case_wrap))
        case_wrap_layout.addWidget(self._case_combo)
        tb.addWidget(case_wrap)
        self._refresh_case_selector_combo()

        tb.addAction(self.act_solve)
        tb.addAction(self.act_solve_active)
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

    def set_element_preview_free(
        self,
        start_x: float, start_y: float, end_x: float, end_y: float,
        kind: str,
    ) -> None:
        self.canvas.set_element_preview_free(
            start_x, start_y, end_x, end_y, kind,
        )
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

    # ── v0.13.0 multi-select ────────────────────────────────────────────

    def toggle_node_in_selection(self, node_id: int) -> None:
        if node_id not in self._model.nodes:
            return
        if node_id in self.canvas.get_selected_nodes():
            self.canvas.remove_node_from_selection(node_id)
        else:
            self.canvas.add_node_to_selection(node_id)
        self._update_selection_status()
        self.canvas.redraw()

    def toggle_element_in_selection(self, element_id: int) -> None:
        if not any(e.id == element_id for e in self._model.elements):
            return
        if element_id in self.canvas.get_selected_elements():
            self.canvas.remove_element_from_selection(element_id)
        else:
            self.canvas.add_element_to_selection(element_id)
        self._update_selection_status()
        self.canvas.redraw()

    def set_drag_rect(
        self, x0: float, y0: float, x1: float, y1: float,
        is_crossing: bool,
    ) -> None:
        self.canvas.set_drag_rect(x0, y0, x1, y1, is_crossing)
        self.canvas.redraw()

    def clear_drag_rect(self) -> None:
        self.canvas.clear_drag_rect()
        self.canvas.redraw()

    def apply_box_select(
        self,
        rect: tuple[float, float, float, float],
        shift: bool,
        is_crossing: bool,
    ) -> None:
        """Resolve a finished drag rect into a selection update.

        Nodes are picked on point-in-rect (inclusive) in both modes.
        Elements use Window vs Crossing rules: Window needs both
        endpoints inside; Crossing also accepts segment-intersects.
        Without Shift the existing selection is cleared first; with
        Shift the rect's hits are added to whatever was already selected.
        """
        from .canvas import _point_in_world_rect, _segment_intersects_rect
        if not shift:
            self.canvas.clear_selection()
        rx0, ry0, rx1, ry1 = rect
        for nid, node in self._model.nodes.items():
            if _point_in_world_rect(node.x, node.y, rx0, ry0, rx1, ry1):
                self.canvas.add_node_to_selection(nid)
        for elem in self._model.elements:
            ni = self._model.nodes.get(elem.node_i)
            nj = self._model.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            i_in = _point_in_world_rect(ni.x, ni.y, rx0, ry0, rx1, ry1)
            j_in = _point_in_world_rect(nj.x, nj.y, rx0, ry0, rx1, ry1)
            if is_crossing:
                hit = i_in or j_in or _segment_intersects_rect(
                    ni.x, ni.y, nj.x, nj.y, rx0, ry0, rx1, ry1,
                )
            else:
                hit = i_in and j_in
            if hit:
                self.canvas.add_element_to_selection(elem.id)
        self._update_selection_status()
        self.canvas.redraw()

    def select_to_neutral_mode(self) -> None:
        """Switch the active tool back to Select. Used after ESC."""
        self._select_tool("select")

    def _update_selection_status(self) -> None:
        nn = len(self.canvas.get_selected_nodes())
        elem_ids = self.canvas.get_selected_elements()
        ne = len(elem_ids)
        if nn == 0 and ne == 0:
            self.set_status("No selection.")
            return
        parts: list[str] = []
        if nn:
            parts.append(f"{nn} node{'s' if nn > 1 else ''}")
        if ne:
            parts.append(f"{ne} element{'s' if ne > 1 else ''}")
        text = ", ".join(parts) + " selected."
        # Grouped load counts only when >1 element is selected — for a
        # single element the inspector itself shows the full table.
        if ne > 1:
            from .load_summary import (
                format_selection_load_counts,
                summarize_selection_loads,
            )
            counts = summarize_selection_loads(self._model, elem_ids)
            summary = format_selection_load_counts(counts)
            if summary:
                text += f"  Loads: {summary}."
        self.set_status(text)

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
        # Load-case CRUD (or any command that mutates ``model.load_cases``
        # / ``self_weight_case`` / load ``load_case`` tags) needs the
        # toolbar combo to repaint even on a fresh model where
        # ``_invalidate_result`` was a no-op. Cheap call, runs after
        # every execute so the source of truth stays the model.
        self._refresh_case_selector_combo()
        self._update_title()
        self.canvas.redraw()

    def open_element_dialog_for_pair(
        self, n_i: int, n_j: int, kind: str | None = None
    ) -> None:
        """Compat wrapper for the legacy two-existing-nodes click flow.

        Looks up node coordinates and forwards to
        :meth:`open_element_dialog_for_member`, which is the primary
        entry point as of v0.10.0.
        """
        ni = self._model.nodes.get(n_i)
        nj = self._model.nodes.get(n_j)
        if ni is None or nj is None:
            QMessageBox.warning(
                self, "Cannot add element",
                "One of the requested nodes no longer exists.",
            )
            return
        self.open_element_dialog_for_member(
            first_x=ni.x, first_y=ni.y, first_node_id=n_i,
            second_x=nj.x, second_y=nj.y, second_node_id=n_j,
            kind=kind,
        )

    def open_element_dialog_for_member(
        self,
        *,
        first_x: float, first_y: float, first_node_id: int | None,
        second_x: float, second_y: float, second_node_id: int | None,
        kind: str | None = None,
        first_split_target: tuple[int, float, float] | None = None,
        second_split_target: tuple[int, float, float] | None = None,
    ) -> None:
        """Open the element-properties dialog for a member draw.

        v0.10.0: ``first_node_id`` / ``second_node_id`` may be ``None``
        when the click was on empty space; the underlying
        :class:`AddMemberCmd` will reuse a nearby node (within 1e-9
        world units) or allocate a new one.

        v0.11.0 (post-PR21): when ``first_split_target`` or
        ``second_split_target`` is set, the dispatch builds a
        :class:`DrawMemberWithSplitsCmd` instead of a plain
        :class:`AddMemberCmd` — the split(s) and the member-add then
        share one undo step, and cancelling this dialog leaves the
        model untouched. When both targets are ``None`` (no element
        interior involved) the existing :class:`AddMemberCmd` path
        runs unchanged.
        """
        if not self._model.materials:
            QMessageBox.warning(
                self, "No materials defined",
                "Define a material first (Edit → Materials…) before placing elements.",
            )
            return

        def _dispatch(
            section_id: int,
            effective_kind: str,
            release_i: bool,
            release_j: bool,
            material_override_id: int | None,
        ) -> None:
            if first_split_target is None and second_split_target is None:
                # No splits involved — preserve the exact pre-existing
                # path so the PR #20 / Stage A acceptance tests stay
                # behaviour-identical.
                self.execute(AddMemberCmd(
                    x_i=first_x, y_i=first_y, node_i=first_node_id,
                    x_j=second_x, y_j=second_y, node_j=second_node_id,
                    section_id=section_id,
                    kind=effective_kind,
                    release_i=release_i,
                    release_j=release_j,
                    material_override_id=material_override_id,
                ))
                return
            self.execute(DrawMemberWithSplitsCmd(
                split_target_i=first_split_target,
                split_target_j=second_split_target,
                x_i=first_x, y_i=first_y, node_i_hint=first_node_id,
                x_j=second_x, y_j=second_y, node_j_hint=second_node_id,
                kind=effective_kind, section_id=section_id,
                release_i=release_i, release_j=release_j,
                material_override_id=material_override_id,
            ))

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
            sticky_override = sticky.get("material_override_id")
            if (sticky_override is not None
                    and sticky_override not in self._model.materials):
                sticky_override = None
            _dispatch(
                sticky["section_id"], effective_kind,
                release_i, release_j, sticky_override,
            )
            return

        try:
            d = ElementDialog(
                self, model=self._model,
                existing_kind=kind or (sticky or {}).get("kind"),
                existing_section_id=(sticky or {}).get("section_id"),
                existing_material_override_id=(
                    sticky or {}).get("material_override_id"),
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
                    "material_override_id": rv.get("material_override_id"),
                }
                sec = self._model.sections[rv["section_id"]]
                label = (f"{rv['kind']} · section {sec.id}"
                         + (f" ({sec.name})" if sec.name else ""))
                self.set_status(
                    f"Reusing element settings: {label}. "
                    f"Switch tool or use Edit → Forget element defaults to reset."
                )
            else:
                self._sticky_element = None
            _dispatch(
                rv["section_id"], rv["kind"],
                rv["release_i"], rv["release_j"],
                rv.get("material_override_id"),
            )

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
        # While the element-detail inspector is open the host has the
        # rest of the editing surface locked (see _set_editing_locked).
        # The right-click context menu's edit items are built fresh
        # each time so they're not part of _lockable_actions — disable
        # them here based on the inspector's visibility instead. The
        # "show details" item stays enabled so the user can re-target
        # the open inspector to a different element from any
        # right-click.
        edits_locked = (
            self._element_inspector is not None
            and self._element_inspector.isVisible()
        )
        menu = QMenu(self)
        a_details = menu.addAction(
            f"Element {elem_id}: show details / FBD…"
        )
        menu.addSeparator()
        a_edit = menu.addAction(f"Element {elem_id}: edit section/material…")
        a_add_load = menu.addAction(f"Element {elem_id}: add member load…")
        a_clear_loads = menu.addAction(
            f"Element {elem_id}: clear member loads"
        )
        a_delete = menu.addAction(f"Element {elem_id}: delete")
        for action in (a_edit, a_add_load, a_clear_loads, a_delete):
            action.setEnabled(not edits_locked)
        chosen = menu.exec(self.cursor().pos())
        if chosen is a_details:
            self._open_element_inspector(elem_id)
        elif chosen is a_edit:
            self._edit_element(elem_id)
        elif chosen is a_add_load:
            self._add_member_load(elem_id)
        elif chosen is a_clear_loads:
            self.execute(ClearMemberLoadsCmd(elem_id=elem_id))
        elif chosen is a_delete:
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
        """Open the per-node nodal-load manager (v0.20 — PR #30).

        Replaces the pre-v0.20 single-load editor: a node can now carry
        multiple independent rows (one per case, several per case…), and
        each Add / Edit / Delete is an individual undoable command. The
        dialog dispatches commands immediately through ``self.execute``
        so the model stays in sync with the table while it's open.
        """
        try:
            d = NodalLoadManagerDialog(
                self, host=self, model=self._model, node_id=node_id,
            )
        except ValueError as e:
            QMessageBox.warning(self, "Node not found", str(e))
            return
        d.exec()

    def _ensure_load_case_exists(self, case_name: str) -> None:
        """Auto-create the named load case if the user typed a new one
        in the dialog combo (PR-A redirect #10). DEFAULT is always
        present so no-op for it."""
        if case_name in self._model.load_cases:
            return
        from ..gui_common.commands import AddLoadCaseCmd
        self.execute(AddLoadCaseCmd(name=case_name))

    def _add_member_load(self, elem_id: int) -> None:
        try:
            d = MemberLoadDialog(self, model=self._model, elem_id=elem_id)
        except ValueError as e:
            QMessageBox.warning(self, "Cannot edit element", str(e))
            return
        if d.exec() != QDialog.DialogCode.Accepted:
            return
        load = d.result_value
        self._ensure_load_case_exists(getattr(load, "load_case", "DEFAULT"))
        self.execute(AddMemberLoadCmd(elem_id=elem_id, load=load))

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
            existing_material_override_id=getattr(
                elem, "material_id_override", None),
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
            material_override_id=rv.get("material_override_id"),
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

    def _on_canvas_click(
        self, hit: HitResult, button: str,
        press_px: tuple[float, float] = (0.0, 0.0),
        shift: bool = False,
    ) -> None:
        if button == "right":
            # Right-click on an element shows the context menu (edit /
            # add load / clear loads / delete + the "show details" item
            # that opens the inspector). While the inspector is open
            # the edit items are greyed inside show_element_menu so the
            # user can still pick "show details" to re-target it.
            # Right-click on a node keeps the node context menu.
            if hit.element_id is not None:
                self.show_element_menu(hit.element_id)
                return
            if hit.node_id is not None:
                self.show_node_menu(hit.node_id)
                return
        try:
            self._active_tool.on_click(hit, button, press_px=press_px, shift=shift)
        except Exception as e:
            QMessageBox.critical(self, "Tool error",
                                  f"{type(e).__name__}: {e}")

    def _on_canvas_release(
        self, hit: HitResult, button: str,
        release_px: tuple[float, float] = (0.0, 0.0),
        shift: bool = False,
    ) -> None:
        try:
            self._active_tool.on_release(
                hit, button, release_px=release_px, shift=shift,
            )
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

    def _on_canvas_motion(
        self, hit: HitResult,
        cursor_px: tuple[float, float] = (0.0, 0.0),
    ) -> None:
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
            self._active_tool.on_motion(hit, cursor_px=cursor_px)
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

    def _edit_analysis_settings(self) -> None:
        """Open the modal analysis-settings dialog.

        v0.9.0 exposes a single switch — include self-weight in the
        static solve. Accepting the dialog updates the model flag and
        invalidates any stale results so the user knows to re-solve.
        """
        d = AnalysisSettingsDialog(
            self,
            include_self_weight=bool(
                getattr(self._model, "include_self_weight", False)
            ),
        )
        if d.exec() != QDialog.DialogCode.Accepted or d.result_value is None:
            return
        new_flag = bool(d.result_value["include_self_weight"])
        if new_flag != bool(getattr(self._model, "include_self_weight", False)):
            self._model.include_self_weight = new_flag
            self._modified = True
            self._update_title()
            self._invalidate_result()
        # Even if the flag didn't change, the summary window header
        # may need a refresh (idempotent — cheap).
        if self._mass_summary_window is not None:
            self._mass_summary_window.refresh()
        self.set_status(
            "Self-weight: enabled in solver."
            if new_flag else "Self-weight: disabled in solver."
        )

    def _show_mass_summary(self) -> None:
        """Open the non-modal mass / self-weight summary, or raise it.

        Singleton pattern mirrors :meth:`_open_view3d`.
        """
        from .mass_summary import MassSummaryWindow

        if self._mass_summary_window is None:
            self._mass_summary_window = MassSummaryWindow(
                self, lambda: self._model,
            )
        else:
            self._mass_summary_window.refresh()
        self._mass_summary_window.show()
        self._mass_summary_window.raise_()
        self._mass_summary_window.activateWindow()

    def _show_joint_masses(self) -> None:
        """Open the non-modal Assembled Joint Masses window, or raise it.

        Singleton pattern mirrors :meth:`_show_mass_summary`.
        """
        from .joint_masses import JointMassesWindow

        if self._joint_masses_window is None:
            self._joint_masses_window = JointMassesWindow(
                self, lambda: self._model,
            )
        else:
            self._joint_masses_window.refresh()
        self._joint_masses_window.show()
        self._joint_masses_window.raise_()
        self._joint_masses_window.activateWindow()

    def _open_element_inspector(self, elem_id: int) -> None:
        """Open (or re-target) the non-modal element-detail inspector.

        The inspector is a singleton on ``self._element_inspector`` —
        right-clicking a different element while it's open re-targets
        the same window instead of stacking dialogs. While it's
        visible the host locks the editing actions (see
        :meth:`_set_editing_locked`); the inspector itself stays
        non-modal so pan / zoom / solve / overlay buttons continue to
        work on the main canvas.
        """
        try:
            if self._element_inspector is None:
                self._element_inspector = ElementPropertiesDialog(
                    self, self._model, elem_id, self._result,
                )
                self._element_inspector.finished.connect(
                    self._on_element_inspector_closed
                )
                # Per-row Delete in the loads table delegates here.
                # Wire BEFORE re-rendering the loads table so the
                # buttons render as enabled rather than greyed out (the
                # initial __init__ build ran before this assignment).
                self._element_inspector._host_delete_member_load = (
                    self.delete_member_load
                )
                self._element_inspector.refresh_loads_only(
                    self._model, self._result,
                )
            else:
                self._element_inspector.set_target(
                    self._model, elem_id, self._result,
                )
        except ValueError as e:
            QMessageBox.warning(self, "Element not found", str(e))
            return
        self._set_editing_locked(True)
        self._element_inspector.show()
        self._element_inspector.raise_()
        self._element_inspector.activateWindow()

    def delete_member_load(self, elem_id: int, load_index: int) -> None:
        """Host-side hook for the inspector's per-row Delete buttons.

        Runs ``DeleteMemberLoadCmd`` through :meth:`execute` so it lands
        on the undo stack like any other model mutation, then refreshes
        the inspector. ``execute()`` invalidates ``self._result``, so
        any previously-displayed end-force block / N·V·M diagrams would
        otherwise keep painting forces for a load that no longer exists.
        A full ``refresh()`` rebuilds the body against the now-``None``
        result, clearing the stale diagrams. The loads table picks up
        the row removal as part of that rebuild."""
        cmd = DeleteMemberLoadCmd(
            elem_id=elem_id, load_index=load_index,
        )
        self.execute(cmd)
        if (self._element_inspector is not None
                and self._element_inspector.isVisible()):
            self._element_inspector.refresh(self._model, self._result)

    def _on_element_inspector_closed(self, _result_code=None) -> None:
        """Re-enable the editing surface once the user closes the
        inspector (Close button or window-close). Called by the dialog's
        ``finished`` signal — the connection is wired once at
        construction in :meth:`_open_element_inspector`."""
        self._set_editing_locked(False)

    def _set_editing_locked(self, locked: bool) -> None:
        """Toggle the editing surface — tool palette (everything except
        Select) and the explicit "Edit" QActions registered in
        :attr:`_lockable_actions`. View, solve, modal, fit, snap,
        diagram-station and deformed-scale actions stay enabled so the
        user can still pan / zoom / re-solve / toggle overlays while
        the inspector is open. When locking, also forces the active
        tool back to "select" so pending left-clicks can't add nodes /
        elements / loads."""
        for name, action in self._tool_actions.items():
            action.setEnabled((not locked) or name == "select")
        for action in self._lockable_actions:
            action.setEnabled(not locked)
        if locked and self._active_tool.name != "select":
            self._select_tool("select")

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

    def _do_batch_assign_selected(self) -> None:
        """Open the BatchAssignDialog for the current element selection."""
        elems = list(self.canvas.get_selected_elements())
        if not elems:
            QMessageBox.information(
                self, "Batch assign",
                "Select one or more elements first "
                "(box-drag in Select mode or Shift-click to multi-pick).",
            )
            return
        try:
            d = BatchAssignDialog(
                self, model=self._model, element_count=len(elems),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Batch assign", str(exc))
            return
        if d.exec() != BatchAssignDialog.DialogCode.Accepted:
            return
        result = d.result_value or {}
        cmd = BatchUpdateElementsCmd(
            element_ids=elems,
            section_id=result.get("section_id"),
            material_override_id=result.get("material_override_id"),
        )
        self.execute(cmd)

    def _do_building_wizard(self) -> None:
        try:
            d = BuildingWizardDialog(self, model=self._model)
        except ValueError as exc:
            QMessageBox.warning(self, "Building wizard", str(exc))
            return
        if self._model.nodes or self._model.elements:
            ans = QMessageBox.question(
                self, "Replace model?",
                "The building wizard will replace the current model "
                "(materials and sections are kept). Use Undo to restore.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        if d.exec() != QDialog.DialogCode.Accepted or d.result_value is None:
            return
        self.execute(ReplaceModelCmd(new_model=d.result_value))
        self.canvas.fit_to_view()
        self.set_status(
            f"Building wizard: created {len(self._model.nodes)} nodes, "
            f"{len(self._model.elements)} elements."
        )

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
        """F5 — run every enabled load case."""
        self._run_static_solve(active_only=False)

    def _do_solve_active_only(self) -> None:
        """Shift+F5 — run only the currently-selected case.

        Replaces only that case's slot in ``_multi_result`` (or builds
        a fresh wrapper containing just that case if there is none)
        so the remaining solved cases stay valid."""
        self._run_static_solve(active_only=True)

    def _run_static_solve(self, *, active_only: bool) -> None:
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
            if active_only:
                # Solve only the active case. If a previous multi-result
                # exists, merge in the fresh single-case result so the
                # other cases stay valid.
                active = self._active_case
                if active == SUM_ALL_KEY:
                    # SUM_ALL isn't a real case to solve. Fall back to
                    # full re-run.
                    new_multi = run_multi_case_analysis(
                        self._model, verbose=False,
                    )
                else:
                    fresh = run_multi_case_analysis(
                        self._model, verbose=False, cases=[active],
                        active_case=active,
                    )
                    if self._multi_result is not None:
                        prev = self._multi_result
                        # Build a merged result: prior solved cases +
                        # the fresh single-case one. When the
                        # active-only re-solve FAILS we must drop the
                        # prior result for the active case — otherwise
                        # the canvas would keep showing a stale
                        # success result while the wrapper claims the
                        # case failed (Gemini PR #28 finding).
                        merged_cases = dict(prev.cases)
                        merged_failed = dict(prev.failed_cases)
                        merged_requested = list(prev.requested_cases)
                        if active in merged_failed:
                            merged_failed.pop(active, None)
                        if active in fresh.cases:
                            merged_cases[active] = fresh.cases[active]
                        elif active in fresh.failed_cases:
                            merged_cases.pop(active, None)
                            merged_failed[active] = fresh.failed_cases[active]
                        if active not in merged_requested:
                            merged_requested.append(active)
                        new_multi = MultiCaseAnalysisResult(
                            cases=merged_cases,
                            active_case=active,
                            failed_cases=merged_failed,
                            requested_cases=merged_requested,
                        )
                    else:
                        new_multi = fresh
            else:
                new_multi = run_multi_case_analysis(
                    self._model, verbose=False,
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Analysis failed",
                f"{type(e).__name__}: {e}\n\nThe model is unchanged.",
            )
            return
        self._multi_result = new_multi
        # Pick a sensible active case for the freshly-solved result
        # (may differ from the one the user was looking at if e.g. they
        # had SUM_ALL and an enabled case just failed). A still-defined
        # combination selection is preserved as-is — selecting it after
        # a solve resolves to the combined result (or its placeholder).
        if self._active_case not in self._model.load_combinations:
            self._active_case = make_active_case_safe(
                self._multi_result, self._active_case,
            )
        self._refresh_case_selector_combo()
        self._push_active_case_to_canvas()
        self._update_window_title_with_case()
        # Push the active case's result into an open inspector so
        # the user doesn't have to close-and-reopen after a solve.
        if (self._element_inspector is not None
                and self._element_inspector.isVisible()):
            self._element_inspector.refresh(self._model, self._result)
        n_solved = len(new_multi.cases)
        n_failed = len(new_multi.failed_cases)
        if n_failed:
            failed_names = ", ".join(sorted(new_multi.failed_cases))
            self.set_status(
                f"Solved {n_solved}/{len(new_multi.requested_cases)} "
                f"case(s); failed: {failed_names}"
            )
        elif n_solved == 0:
            self.set_status("No cases were solved.")
        else:
            active_r = new_multi.get(self._active_case)
            if active_r is not None and getattr(active_r, "residual", None) is not None:
                self.set_status(
                    f"Solved {n_solved} case(s) · active = "
                    f"{self._active_case} · "
                    f"residual = {active_r.residual:.2e}"
                )
            else:
                self.set_status(
                    f"Solved {n_solved} case(s) · active = "
                    f"{self._active_case}"
                )

    # ── PR-A: active case + multi-result plumbing ──────────────────

    def _on_active_case_changed(self, new_name: str) -> None:
        """Toolbar combo signal handler.

        Empty signals are dropped (combo is repopulated by clearing →
        adding items, which emits an empty currentTextChanged)."""
        if not new_name:
            return
        if new_name == self._active_case:
            return
        self._active_case = new_name
        self._push_active_case_to_canvas()
        self._update_window_title_with_case()
        if (
            self._element_inspector is not None
            and self._element_inspector.isVisible()
        ):
            self._element_inspector.refresh(self._model, self._result)
        # PR #29: when the user selects an unavailable combination,
        # explain WHY there's no result rather than leaving a bare
        # placeholder.
        if (
            new_name in self._model.load_combinations
            and self._result is None
        ):
            comb = self._model.load_combinations[new_name]
            if self._multi_result is None:
                self.set_status(
                    f"Combination {new_name} needs a solve first (F5)."
                )
            else:
                missing = self._multi_result.missing_cases_for(comb.terms)
                if missing:
                    self.set_status(
                        f"Combination {new_name} requires solved results "
                        f"for {', '.join(missing)}."
                    )

    def _refresh_case_selector_combo(self) -> None:
        """Repopulate the toolbar combo from the model's case dict.

        Includes SUM_ALL last when a multi_result with sum_all_available()
        is in hand. Blocks signals during repopulation so the
        currentTextChanged dance doesn't trigger a spurious
        ``_on_active_case_changed`` mid-rebuild."""
        if not hasattr(self, "_case_combo"):
            return
        combo = self._case_combo
        combo.blockSignals(True)
        try:
            combo.clear()
            # Real cases — sorted alphabetically with DEFAULT pinned
            # to the front so it's always visible at index 0.
            ordered = (
                (["DEFAULT"] if "DEFAULT" in self._model.load_cases else [])
                + sorted(
                    n for n in self._model.load_cases
                    if n != "DEFAULT"
                )
            )
            for name in ordered:
                lc = self._model.load_cases[name]
                label = name if lc.enabled else f"{name}  (disabled)"
                combo.addItem(label, name)
            # SUM_ALL — only when every requested case has solved.
            if (
                self._multi_result is not None
                and self._multi_result.sum_all_available()
                and len(self._multi_result.cases) >= 2
            ):
                combo.addItem(SUM_ALL_KEY, SUM_ALL_KEY)
            # User-defined combinations LAST (PR #29). A combination
            # whose referenced cases aren't all solved is shown with a
            # "(needs solve)" hint and still selectable — selecting it
            # surfaces the placeholder rather than silently hiding it.
            for comb_name in sorted(self._model.load_combinations):
                comb = self._model.load_combinations[comb_name]
                available = (
                    self._multi_result is not None
                    and self._multi_result.combination_available(comb.terms)
                )
                label = (
                    f"{comb_name}  [comb]" if available
                    else f"{comb_name}  [comb · needs solve]"
                )
                combo.addItem(label, comb_name)
            # Restore selection by matching the userData (the raw
            # case name, not the "(disabled)" label).
            idx = combo.findData(self._active_case)
            if idx < 0:
                # Fall back to DEFAULT (or the first item).
                idx = combo.findData("DEFAULT")
                if idx < 0 and combo.count() > 0:
                    idx = 0
                if idx >= 0:
                    self._active_case = combo.itemData(idx)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)

    def _resolve_active_result(self):
        """Resolve ``self._active_case`` to a concrete ``AnalysisResult``
        (or ``None`` for a pre-solve / unavailable selection).

        Handles three kinds of selection:
        * a load-case name → the per-case result;
        * ``SUM_ALL`` → the unit-coefficient superposition view;
        * a combination name → the coefficient-weighted view (or
          ``None`` when a referenced case is unsolved, so the canvas /
          inspector show a clear placeholder)."""
        if self._multi_result is None:
            return None
        name = self._active_case
        if name in self._model.load_combinations:
            comb = self._model.load_combinations[name]
            return self._multi_result.combination(
                comb.terms, name=name,
            )
        return self._multi_result.get(name)

    def _push_active_case_to_canvas(self) -> None:
        """Sync ``self._result`` (the legacy single-case view) and the
        canvas to whatever the active case / combination resolves to."""
        self._result = self._resolve_active_result()
        self.canvas.set_result(self._result)
        if hasattr(self.canvas, "set_active_case"):
            self.canvas.set_active_case(self._active_case)
        # PR #29: when a combination (or SUM_ALL) is active, tell the
        # canvas which cases contribute so its load-dimming highlights
        # all of them rather than a single (misleading) case.
        if hasattr(self.canvas, "set_active_combination_cases"):
            if self._active_case in self._model.load_combinations:
                comb = self._model.load_combinations[self._active_case]
                self.canvas.set_active_combination_cases(
                    set(comb.terms.keys())
                )
            elif (
                self._active_case == SUM_ALL_KEY
                and self._multi_result is not None
            ):
                # SUM_ALL = 1.0 × every solved case → highlight them all.
                self.canvas.set_active_combination_cases(
                    set(self._multi_result.cases.keys())
                )
            else:
                self.canvas.set_active_combination_cases(None)
        self._update_result_text()

    def _on_active_case_loads_only_toggled(self, on: bool) -> None:
        """View → Active case loads only slot."""
        self._active_case_loads_only = bool(on)
        if hasattr(self.canvas, "set_active_case_loads_only"):
            self.canvas.set_active_case_loads_only(self._active_case_loads_only)

    def _show_load_case_manager(self) -> None:
        """Open the Load Case Manager dialog and apply its result.

        The dialog returns a list of CRUD commands the host executes
        through the standard ``execute()`` pipeline so every change is
        undoable and goes through the invalidation surface (which
        clears the multi-case result)."""
        from .dialogs import LoadCaseManagerDialog
        d = LoadCaseManagerDialog(self, model=self._model)
        if d.exec() != QDialog.DialogCode.Accepted:
            return
        for cmd in d.result_value or []:
            try:
                self.execute(cmd)
            except ValueError as e:
                QMessageBox.warning(self, "Load case", str(e))
                return
        self._refresh_case_selector_combo()

    def _show_load_combination_manager(self) -> None:
        """Open the Load Combination Manager dialog and apply its
        result commands through the undoable ``execute()`` pipeline."""
        from .dialogs import LoadCombinationManagerDialog
        d = LoadCombinationManagerDialog(self, model=self._model)
        if d.exec() != QDialog.DialogCode.Accepted:
            return
        for cmd in d.result_value or []:
            try:
                self.execute(cmd)
            except ValueError as e:
                QMessageBox.warning(self, "Load combination", str(e))
                return
        self._refresh_case_selector_combo()

    def _update_window_title_with_case(self) -> None:
        """Reflect the active case / combination in the window title
        when it differs from DEFAULT (so a single-case workflow's title
        stays clean)."""
        base = getattr(self, "_base_window_title", None)
        if base is None:
            base = self.windowTitle()
            for sep in (" — case: ", " — comb: "):
                base = base.split(sep)[0]
            self._base_window_title = base
        if self._active_case == "DEFAULT" or self._multi_result is None:
            self.setWindowTitle(base)
        elif self._active_case in self._model.load_combinations:
            self.setWindowTitle(f"{base} — comb: {self._active_case}")
        else:
            self.setWindowTitle(f"{base} — case: {self._active_case}")

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
                mass_formulation=d.result_value.get(
                    "mass_formulation", "consistent",
                ),
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
        self._multi_result = None
        self._modal_result = None
        self.canvas.clear_result()
        self.canvas.clear_modal_result()
        self._refresh_case_selector_combo()
        self._update_window_title_with_case()
        self._update_result_text()

    def _invalidate_result(self) -> None:
        if self._result is not None or self._multi_result is not None:
            self._result = None
            self._multi_result = None
            self.canvas.clear_result()
            self._refresh_case_selector_combo()
            self._update_window_title_with_case()
            self._update_result_text()
        if self._modal_result is not None:
            self._modal_result = None
            self.canvas.clear_modal_result()
            if self._modal_results_dialog is not None:
                self._modal_results_dialog.close()
                self._modal_results_dialog = None
        # Mass / self-weight summary tracks ρ / A / L from the model;
        # any edit that invalidates a result could also have changed
        # those numbers. The window itself is non-modal and cheap to
        # repopulate, so refresh unconditionally when it's open.
        if self._mass_summary_window is not None:
            self._mass_summary_window.refresh()
        # Same story for the assembled-joint-masses window: any model
        # edit (nodes, supports, sections) can change M's contents.
        # Skip the refresh when the window is hidden — _show_joint_masses
        # always refreshes on re-open, so a hidden singleton can't
        # show stale data. Avoiding the call here means no mass-matrix
        # assembly on every keystroke / drag when the panel isn't open.
        if (
            self._joint_masses_window is not None
            and self._joint_masses_window.isVisible()
        ):
            self._joint_masses_window.refresh()

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

    def keyPressEvent(self, event) -> None:
        """Route ESC to the active tool and Delete/Backspace to batch-delete.

        ESC ALWAYS lands the user in Select mode after the active tool
        has cleaned up its own state (pending pair-tool draws, in-progress
        drag rects, etc.). ESC never touches the model and never pushes
        an undo entry."""
        if event.key() == Qt.Key.Key_Escape:
            try:
                self._active_tool.on_key("escape")
            except Exception:
                pass
            # Always end in Select mode. _select_tool() is idempotent
            # when the active tool is already Select (deactivate +
            # activate just refresh status).
            if self._active_tool is not self._tools["select"]:
                self._select_tool("select")
            self.canvas.redraw()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            inspector_open = (
                self._element_inspector is not None
                and self._element_inspector.isVisible()
            )
            if not inspector_open:
                self._delete_selected_objects()
            event.accept()
            return
        super().keyPressEvent(event)

    def _delete_selected_objects(self) -> None:
        """Delete every currently-selected node/element in one undo step.

        Cascade follows existing DeleteNodeCmd semantics — deleting a
        node still removes its supports, nodal loads, and connected
        elements. We delete elements first to keep already-marked
        elements out of the per-node cascade path."""
        nodes = list(self.canvas.get_selected_nodes())
        elems = list(self.canvas.get_selected_elements())
        if not nodes and not elems:
            self.set_status("Nothing selected to delete.")
            return
        cmd = BatchDeleteCmd(node_ids=nodes, element_ids=elems)
        self.execute(cmd)
        # Only deselect ids that were actually removed from the model;
        # if execute() failed internally the model may be unchanged.
        for nid in nodes:
            if nid not in self._model.nodes:
                self.canvas.remove_node_from_selection(nid)
        for eid in elems:
            if not any(e.id == eid for e in self._model.elements):
                self.canvas.remove_element_from_selection(eid)
        self._update_selection_status()
        self.canvas.redraw()

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        super().closeEvent(event)
