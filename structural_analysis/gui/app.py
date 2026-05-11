"""MainApplication — the Tk root window: menus, toolbars, undo/redo, file I/O.

All user mutations flow through ``execute(command)`` which:
1. Runs the command's ``do(model)`` inside a try/except.
2. On success, pushes onto the undo stack and triggers a redraw.
3. On ValueError, shows a friendly error dialog without mutating model state.

This is the **single exception boundary** — internals are free to raise.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from ..file_io import read_input_file
from ..main import run_analysis
from ..model import AnalysisResult, StructuralModel
from .canvas import HitResult, ModelCanvas
from .commands import (
    AddElementCmd,
    AddOrUpdateMaterialCmd,
    AddMemberLoadCmd,
    ClearMemberLoadsCmd,
    Command,
    DeleteElementCmd,
    DeleteMaterialCmd,
    DeleteNodeCmd,
    ReplaceModelCmd,
    SetNodalLoadCmd,
    SetSupportCmd,
)
from .dialogs import (
    ElementDialog,
    GridSpacingDialog,
    MaterialDialog,
    MaterialListDialog,
    MemberLoadDialog,
    NodalLoadDialog,
    SupportDialog,
)
from .results_view import format_result
from .file_writer import write_input_file
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


class MainApplication(tk.Tk):
    """The top-level Tk window."""

    def __init__(self, *, initial_path: Optional[str] = None):
        super().__init__()
        self.title("Structural Analysis — GUI")
        self.geometry("1200x780")

        self._model = StructuralModel(title="Untitled")
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._modified = False
        self._current_path: Optional[str] = None
        self._result: Optional[AnalysisResult] = None

        self._build_layout()
        self._build_menubar()
        self._bind_keys()

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
            self.after(50, lambda: self._open_path(initial_path))

    # ── layout ─────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        # Left tool palette
        toolbar = ttk.Frame(self, padding=4)
        toolbar.pack(side="left", fill="y")

        self._tool_var = tk.StringVar(value="select")
        for name, label in [
            ("select",      "Select"),
            ("node",        "Node"),
            ("frame",       "Frame"),
            ("truss",       "Truss"),
            ("support",     "Support"),
            ("nodal_load",  "Nodal load"),
            ("member_load", "Member load"),
            ("delete",      "Delete"),
        ]:
            ttk.Radiobutton(
                toolbar, text=label, value=name, variable=self._tool_var,
                command=lambda n=name: self._select_tool(n),
                width=14,
            ).pack(anchor="w", pady=1)

        ttk.Separator(toolbar, orient="horizontal").pack(fill="x", pady=8)

        ttk.Button(toolbar, text="Solve (F5)",
                   command=self._do_solve).pack(fill="x", pady=2)
        ttk.Button(toolbar, text="Materials…",
                   command=self._open_material_list).pack(fill="x", pady=2)

        ttk.Separator(toolbar, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(toolbar, text="Overlays:").pack(anchor="w")
        self._show_def_var = tk.BooleanVar(value=True)
        self._show_rea_var = tk.BooleanVar(value=True)
        self._show_dia_var = tk.BooleanVar(value=False)
        self._dia_kind_var = tk.StringVar(value="moment")

        ttk.Checkbutton(toolbar, text="Deformed shape",
                        variable=self._show_def_var,
                        command=self._refresh_overlays).pack(anchor="w")
        ttk.Checkbutton(toolbar, text="Reactions",
                        variable=self._show_rea_var,
                        command=self._refresh_overlays).pack(anchor="w")
        ttk.Checkbutton(toolbar, text="Force diagrams",
                        variable=self._show_dia_var,
                        command=self._refresh_overlays).pack(anchor="w")
        for label, val in [("M (moment)", "moment"),
                           ("V (shear)",  "shear"),
                           ("N (axial)",  "axial")]:
            ttk.Radiobutton(toolbar, text=label, value=val,
                            variable=self._dia_kind_var,
                            command=self._refresh_overlays).pack(anchor="w", padx=12)

        # Right side: canvas (top) + result text (bottom)
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)

        self.canvas = ModelCanvas(right, lambda: self._model)
        self.canvas.toolbar.pack(side="top", fill="x")
        self.canvas.widget.pack(side="top", fill="both", expand=True)

        # Status bar
        self._status_var = tk.StringVar(value="")
        status_frame = ttk.Frame(self)
        status_frame.pack(side="bottom", fill="x")
        ttk.Label(status_frame, textvariable=self._status_var,
                  anchor="w", padding=(8, 2)).pack(side="left", fill="x", expand=True)
        self._coord_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self._coord_var,
                  anchor="e", padding=(8, 2)).pack(side="right")

        # Result text panel (below the canvas, collapsible by sash)
        result_frame = ttk.LabelFrame(right, text="Analysis report", padding=4)
        result_frame.pack(side="bottom", fill="x")
        self._result_text = tk.Text(result_frame, height=12, wrap="none",
                                     font=("Courier", 9))
        scroll_y = ttk.Scrollbar(result_frame, orient="vertical",
                                  command=self._result_text.yview)
        scroll_x = ttk.Scrollbar(result_frame, orient="horizontal",
                                  command=self._result_text.xview)
        self._result_text.configure(yscrollcommand=scroll_y.set,
                                     xscrollcommand=scroll_x.set)
        self._result_text.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        self._result_text.insert("end", "(no analysis run yet)\n")
        self._result_text.config(state="disabled")

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New",     accelerator="Ctrl+N", command=self._do_new)
        file_menu.add_command(label="Open…",   accelerator="Ctrl+O", command=self._do_open)
        file_menu.add_command(label="Save",    accelerator="Ctrl+S", command=self._do_save)
        file_menu.add_command(label="Save As…",                       command=self._do_save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Quit",    accelerator="Ctrl+Q", command=self._do_quit)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo",    accelerator="Ctrl+Z", command=self._do_undo)
        edit_menu.add_command(label="Redo",    accelerator="Ctrl+Y", command=self._do_redo)
        edit_menu.add_separator()
        edit_menu.add_command(label="Materials…", command=self._open_material_list)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Grid spacing…", command=self._set_grid_spacing)
        self._snap_var = tk.BooleanVar(value=True)
        view_menu.add_checkbutton(label="Snap to grid", variable=self._snap_var,
                                   command=self._toggle_snap)
        menubar.add_cascade(label="View", menu=view_menu)

        run_menu = tk.Menu(menubar, tearoff=0)
        run_menu.add_command(label="Solve", accelerator="F5", command=self._do_solve)
        run_menu.add_command(label="Clear results",
                              command=self._clear_result)
        menubar.add_cascade(label="Run", menu=run_menu)

        self.config(menu=menubar)

    def _bind_keys(self) -> None:
        self.bind_all("<Control-n>", lambda e: self._do_new())
        self.bind_all("<Control-o>", lambda e: self._do_open())
        self.bind_all("<Control-s>", lambda e: self._do_save())
        self.bind_all("<Control-q>", lambda e: self._do_quit())
        self.bind_all("<Control-z>", lambda e: self._do_undo())
        self.bind_all("<Control-y>", lambda e: self._do_redo())
        self.bind_all("<F5>",         lambda e: self._do_solve())
        # Tool shortcuts
        self.bind_all("s", lambda e: self._select_tool("select"))
        self.bind_all("n", lambda e: self._select_tool("node"))
        self.bind_all("f", lambda e: self._select_tool("frame"))
        self.bind_all("t", lambda e: self._select_tool("truss"))

    # ── Host protocol used by tools ────────────────────────────────

    def model(self) -> StructuralModel:
        return self._model

    def set_status(self, text: str) -> None:
        self._status_var.set(text)

    def execute(self, command: Command) -> None:
        try:
            command.do(self._model)
        except ValueError as e:
            messagebox.showerror("Cannot apply change", str(e), parent=self)
            return
        except Exception as e:  # safety net
            messagebox.showerror("Internal error",
                                  f"{type(e).__name__}: {e}", parent=self)
            return
        self._undo.append(command)
        self._redo.clear()
        self._modified = True
        self._invalidate_result()
        self._update_title()
        self.canvas.redraw()

    def open_element_dialog_for_pair(self, n_i: int, n_j: int) -> None:
        if not self._model.materials:
            messagebox.showwarning(
                "No materials defined",
                "Define a material first (Edit → Materials…) before placing elements.",
                parent=self,
            )
            return
        d = ElementDialog(self, model=self._model)
        self.wait_window(d)
        if d.result is None:
            return
        self.execute(AddElementCmd(
            node_i=n_i, node_j=n_j,
            material_id=d.result["material_id"],
            kind=d.result["kind"],
            release_i=d.result["release_i"],
            release_j=d.result["release_j"],
        ))

    def show_node_menu(self, node_id: int, x: int, y: int,
                        action: str | None = None) -> None:
        if action == "support":
            self._edit_support(node_id)
            return
        if action == "nodal_load":
            self._edit_nodal_load(node_id)
            return
        # generic right-click menu
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label=f"Node {node_id}: edit support…",
                         command=lambda: self._edit_support(node_id))
        menu.add_command(label=f"Node {node_id}: edit nodal load…",
                         command=lambda: self._edit_nodal_load(node_id))
        menu.add_command(label=f"Node {node_id}: delete",
                         command=lambda: self.execute(DeleteNodeCmd(node_id=node_id)))
        try:
            menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        finally:
            menu.grab_release()

    def show_element_menu(self, elem_id: int, x: int, y: int,
                           action: str | None = None) -> None:
        if action == "member_load":
            self._add_member_load(elem_id)
            return
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label=f"Element {elem_id}: add member load…",
                         command=lambda: self._add_member_load(elem_id))
        menu.add_command(label=f"Element {elem_id}: clear member loads",
                         command=lambda: self.execute(ClearMemberLoadsCmd(elem_id=elem_id)))
        menu.add_command(label=f"Element {elem_id}: delete",
                         command=lambda: self.execute(DeleteElementCmd(elem_id=elem_id)))
        try:
            menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        finally:
            menu.grab_release()

    # ── editing flows ─────────────────────────────────────────────

    def _edit_support(self, node_id: int) -> None:
        d = SupportDialog(self, existing=self._model.supports.get(node_id),
                          node_id=node_id)
        self.wait_window(d)
        if d.result is None:
            return  # cancelled
        action, support = d.result
        if action == "remove":
            if node_id in self._model.supports:
                self.execute(SetSupportCmd(support=None, node_id=node_id))
            return
        self.execute(SetSupportCmd(support=support, node_id=node_id))

    def _edit_nodal_load(self, node_id: int) -> None:
        existing = next((ld for ld in self._model.nodal_loads
                         if ld.node_id == node_id), None)
        d = NodalLoadDialog(self, existing=existing, node_id=node_id)
        self.wait_window(d)
        if d.result is None:
            return
        fx, fy, mz = d.result
        self.execute(SetNodalLoadCmd(node_id=node_id, fx=fx, fy=fy, mz=mz))

    def _add_member_load(self, elem_id: int) -> None:
        try:
            d = MemberLoadDialog(self, model=self._model, elem_id=elem_id)
        except ValueError as e:
            messagebox.showerror("Cannot edit element", str(e), parent=self)
            return
        self.wait_window(d)
        if d.result is None:
            return
        self.execute(AddMemberLoadCmd(elem_id=elem_id, load=d.result))

    def _open_material_list(self) -> None:
        def add_or_update(mat):
            self.execute(AddOrUpdateMaterialCmd(material=mat))

        def delete(mid):
            cmd = DeleteMaterialCmd(material_id=mid)
            self.execute(cmd)

        d = MaterialListDialog(
            self, model=self._model,
            on_add_or_update=add_or_update,
            on_delete=delete,
        )
        self.wait_window(d)

    # ── tool selection ────────────────────────────────────────────

    def _select_tool(self, name: str) -> None:
        if name not in self._tools:
            return
        self._active_tool.deactivate()
        self._tool_var.set(name)
        self._active_tool = self._tools[name]
        self._active_tool.activate()

    def _on_canvas_click(self, hit: HitResult, button: str) -> None:
        # Right-click is universal: open the relevant context menu.
        if button == "right":
            if hit.node_id is not None:
                self.show_node_menu(hit.node_id, 0, 0)
                return
            if hit.element_id is not None:
                self.show_element_menu(hit.element_id, 0, 0)
                return
        try:
            self._active_tool.on_click(hit, button)
        except Exception as e:
            messagebox.showerror("Tool error",
                                  f"{type(e).__name__}: {e}", parent=self)

    def _on_canvas_motion(self, hit: HitResult) -> None:
        parts = [f"({hit.x:.3f}, {hit.y:.3f})"]
        if hit.node_id is not None:
            parts.append(f"node {hit.node_id}")
        if hit.element_id is not None:
            parts.append(f"elem {hit.element_id}")
        self._coord_var.set("  |  ".join(parts))
        # Forward to the active tool so it can implement hover/highlight.
        try:
            self._active_tool.on_motion(hit)
        except Exception:
            pass  # tooltips must never raise

    # ── overlay toggles ───────────────────────────────────────────

    def _refresh_overlays(self) -> None:
        self.canvas.show_deformed = self._show_def_var.get()
        self.canvas.show_reactions = self._show_rea_var.get()
        self.canvas.show_diagrams = self._show_dia_var.get()
        self.canvas.diagram_kind = self._dia_kind_var.get()
        self.canvas.redraw()

    # ── grid / snap ───────────────────────────────────────────────

    def _set_grid_spacing(self) -> None:
        d = GridSpacingDialog(self, current=self.canvas.grid_spacing)
        self.wait_window(d)
        if d.result is not None:
            try:
                self.canvas.set_grid_spacing(d.result)
            except ValueError as e:
                messagebox.showerror("Invalid input", str(e), parent=self)

    def _toggle_snap(self) -> None:
        self.canvas.toggle_snap(self._snap_var.get())

    # ── undo / redo ───────────────────────────────────────────────

    def _do_undo(self) -> None:
        if not self._undo:
            return
        cmd = self._undo.pop()
        try:
            cmd.undo(self._model)
        except Exception as e:
            messagebox.showerror("Undo failed",
                                  f"{type(e).__name__}: {e}", parent=self)
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
            messagebox.showerror("Redo failed",
                                  f"{type(e).__name__}: {e}", parent=self)
            return
        self._undo.append(cmd)
        self._modified = True
        self._invalidate_result()
        self._update_title()
        self.canvas.redraw()

    # ── file menu actions ────────────────────────────────────────

    def _confirm_discard(self) -> bool:
        if not self._modified:
            return True
        ans = messagebox.askyesnocancel(
            "Unsaved changes",
            "You have unsaved changes. Save before continuing?",
            parent=self,
        )
        if ans is None:  # cancel
            return False
        if ans:  # save
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
        path = filedialog.askopenfilename(
            title="Open input file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        self._open_path(path)

    def _open_path(self, path: str) -> None:
        try:
            new_model = read_input_file(path)
        except FileNotFoundError:
            messagebox.showerror("Open failed",
                                  f"File not found: {path}", parent=self)
            return
        except Exception as e:
            messagebox.showerror("Open failed",
                                  f"Could not parse {os.path.basename(path)}:\n"
                                  f"{type(e).__name__}: {e}", parent=self)
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
        path = filedialog.asksaveasfilename(
            title="Save input file",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            parent=self,
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
            messagebox.showerror("Save failed",
                                  f"{type(e).__name__}: {e}", parent=self)
            return False
        self._modified = False
        self._update_title()
        self.set_status(f"Saved to {path}")
        return True

    def _do_quit(self) -> None:
        if not self._confirm_discard():
            return
        self.destroy()

    # ── solve ─────────────────────────────────────────────────────

    def _do_solve(self) -> None:
        if not self._model.elements:
            messagebox.showwarning(
                "Cannot solve",
                "The model has no elements. Draw some nodes and elements first.",
                parent=self,
            )
            return
        if not self._model.supports:
            ans = messagebox.askokcancel(
                "No supports defined",
                "The model has no supports — the stiffness matrix will be "
                "singular. Solve anyway?",
                parent=self,
            )
            if not ans:
                return
        try:
            self._result = run_analysis(self._model, verbose=False)
        except Exception as e:
            messagebox.showerror(
                "Analysis failed",
                f"{type(e).__name__}: {e}\n\nThe model is unchanged.",
                parent=self,
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
        # Editing the model invalidates any prior solve.
        if self._result is not None:
            self._result = None
            self.canvas.clear_result()
            self._update_result_text()

    def _update_result_text(self) -> None:
        text = format_result(self._model, self._result) if self._result \
            else "(no analysis run yet)"
        self._result_text.config(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert("end", text + "\n")
        self._result_text.config(state="disabled")

    # ── title / status ─────────────────────────────────────────────

    def _update_title(self) -> None:
        path = (os.path.basename(self._current_path)
                if self._current_path else "Untitled")
        mark = "*" if self._modified else ""
        self.title(f"{mark}{path} — Structural Analysis GUI")
