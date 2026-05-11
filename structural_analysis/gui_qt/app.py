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
from ..model import AnalysisResult, StructuralModel
from ..gui.commands import (
    AddElementCmd,
    AddMemberLoadCmd,
    AddOrUpdateMaterialCmd,
    ClearMemberLoadsCmd,
    Command,
    DeleteElementCmd,
    DeleteMaterialCmd,
    DeleteNodeCmd,
    SetNodalLoadCmd,
    SetSupportCmd,
)
from ..gui.file_writer import write_input_file
from ..gui.results_view import format_result

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
    GridSpacingDialog,
    MaterialListDialog,
    MemberLoadDialog,
    NodalLoadDialog,
    SupportDialog,
)


class MainWindow(QMainWindow):
    """The Qt main window."""

    def __init__(self, *, initial_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Structural Analysis — GUI (Qt)")
        self.resize(1200, 800)

        self._model = StructuralModel(title="Untitled")
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._modified = False
        self._current_path: Optional[str] = None
        self._result: Optional[AnalysisResult] = None

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

        self._update_title()
        self.canvas.redraw()

        if initial_path:
            # defer to event loop start
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._open_path(initial_path))

    # ── layout ──

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.canvas = ModelCanvas(splitter, lambda: self._model)
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

        self.act_grid_spacing = QAction("&Grid spacing…", self,
                                          triggered=self._set_grid_spacing)
        self.act_snap = QAction("Snap to grid", self, checkable=True, checked=True,
                                  triggered=self._toggle_snap)

        self.act_solve = QAction("&Solve", self, shortcut="F5",
                                   triggered=self._do_solve)
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
        m_file.addAction(self.act_save)
        m_file.addAction(self.act_save_as)
        m_file.addSeparator()
        m_file.addAction(self.act_quit)

        m_edit = self.menuBar().addMenu("&Edit")
        m_edit.addAction(self.act_undo)
        m_edit.addAction(self.act_redo)
        m_edit.addSeparator()
        m_edit.addAction(self.act_materials)

        m_view = self.menuBar().addMenu("&View")
        m_view.addAction(self.act_grid_spacing)
        m_view.addAction(self.act_snap)

        m_run = self.menuBar().addMenu("&Run")
        m_run.addAction(self.act_solve)
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
        v.addWidget(self._cb_deformed)
        v.addWidget(self._cb_reactions)
        v.addWidget(self._cb_diagrams)

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

    def open_element_dialog_for_pair(self, n_i: int, n_j: int) -> None:
        if not self._model.materials:
            QMessageBox.warning(
                self, "No materials defined",
                "Define a material first (Edit → Materials…) before placing elements.",
            )
            return
        try:
            d = ElementDialog(self, model=self._model)
        except ValueError as e:
            QMessageBox.warning(self, "Cannot add element", str(e))
            return
        if d.exec() == QDialog.DialogCode.Accepted and d.result_value is not None:
            self.execute(AddElementCmd(
                node_i=n_i, node_j=n_j,
                material_id=d.result_value["material_id"],
                kind=d.result_value["kind"],
                release_i=d.result_value["release_i"],
                release_j=d.result_value["release_j"],
            ))

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
        a1 = menu.addAction(f"Element {elem_id}: add member load…")
        a2 = menu.addAction(f"Element {elem_id}: clear member loads")
        a3 = menu.addAction(f"Element {elem_id}: delete")
        chosen = menu.exec(self.cursor().pos())
        if chosen is a1:
            self._add_member_load(elem_id)
        elif chosen is a2:
            self.execute(ClearMemberLoadsCmd(elem_id=elem_id))
        elif chosen is a3:
            self.execute(DeleteElementCmd(elem_id=elem_id))

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

    def _open_material_list(self) -> None:
        d = MaterialListDialog(
            self, model=self._model,
            on_add_or_update=lambda mat: self.execute(AddOrUpdateMaterialCmd(material=mat)),
            on_delete=lambda mid: self.execute(DeleteMaterialCmd(material_id=mid)),
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

    def _on_canvas_motion(self, hit: HitResult) -> None:
        parts = [f"({hit.x:.3f}, {hit.y:.3f})"]
        if hit.node_id is not None:
            parts.append(f"node {hit.node_id}")
        if hit.element_id is not None:
            parts.append(f"elem {hit.element_id}")
        self._coord_label.setText("  |  ".join(parts))
        try:
            self._active_tool.on_motion(hit)
        except Exception:
            pass

    # ── overlay toggles ──

    def _refresh_overlays(self) -> None:
        self.canvas.show_deformed = self._cb_deformed.isChecked()
        self.canvas.show_reactions = self._cb_reactions.isChecked()
        self.canvas.show_diagrams = self._cb_diagrams.isChecked()
        for btn in self._dia_group.buttons():
            if btn.isChecked():
                self.canvas.diagram_kind = btn.property("diagram_kind")
                break
        self.canvas.redraw()

    # ── grid / snap ──

    def _set_grid_spacing(self) -> None:
        d = GridSpacingDialog(self, current=self.canvas.grid_spacing)
        if d.exec() == QDialog.DialogCode.Accepted and d.result_value is not None:
            try:
                self.canvas.set_grid_spacing(d.result_value)
            except ValueError as e:
                QMessageBox.warning(self, "Invalid input", str(e))

    def _toggle_snap(self, checked: bool) -> None:
        self.canvas.toggle_snap(checked)

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
        self._model = StructuralModel(title="Untitled")
        self._undo.clear()
        self._redo.clear()
        self._modified = False
        self._current_path = None
        self._clear_result()
        self._update_title()
        self.canvas.redraw()

    def _do_open(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open input file", "",
            "Text files (*.txt);;All files (*.*)",
        )
        if not path:
            return
        self._open_path(path)

    def _open_path(self, path: str) -> None:
        try:
            new_model = read_input_file(path)
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
        self._undo.clear()
        self._redo.clear()
        self._modified = False
        self._current_path = path
        self._clear_result()
        self._update_title()
        self.canvas.redraw()
        self.set_status(f"Opened {path}")

    def _do_save(self) -> bool:
        if self._current_path is None:
            return self._do_save_as()
        return self._save_to(self._current_path)

    def _do_save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save input file", "",
            "Text files (*.txt);;All files (*.*)",
        )
        if not path:
            return False
        if self._save_to(path):
            self._current_path = path
            self._update_title()
            return True
        return False

    def _save_to(self, path: str) -> bool:
        try:
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
        if not self._model.supports:
            ans = QMessageBox.question(
                self, "No supports defined",
                "The model has no supports — the stiffness matrix will be "
                "singular. Solve anyway?",
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

    def _clear_result(self) -> None:
        self._result = None
        self.canvas.clear_result()
        self._update_result_text()

    def _invalidate_result(self) -> None:
        if self._result is not None:
            self._result = None
            self.canvas.clear_result()
            self._update_result_text()

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
