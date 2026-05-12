"""Tk dialogs for editing model entities.

Each dialog returns its result via ``result`` (None if cancelled).
Numeric fields are validated through ``parse_float`` so users see a friendly
message instead of a Python traceback when they type bad input.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Callable, Optional

from ..element import FrameElement2D, TrussElement2D
from ..model import (
    FrameTemperatureLoad,
    Material,
    NodalLoad,
    PointLoad,
    Section,
    StructuralModel,
    Support,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)


def parse_float(text: str, name: str, *,
                allow_blank: bool = False) -> Optional[float]:
    """Parse a float, raising ValueError("name must be a number, got '...')"""
    s = (text or "").strip()
    if not s:
        if allow_blank:
            return None
        raise ValueError(f"{name} is required.")
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"{name} must be a number, got '{s}'.") from None


def parse_int(text: str, name: str) -> int:
    s = (text or "").strip()
    if not s:
        raise ValueError(f"{name} is required.")
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got '{s}'.") from None


class _ModalDialog(tk.Toplevel):
    """Tk modal dialog base — child windows centred over the parent."""

    def __init__(self, parent: tk.Misc, title: str):
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)
        self.result: Any = None
        self._parent = parent

        body = ttk.Frame(self, padding=10)
        body.grid(row=0, column=0, sticky="nsew")
        self._build_body(body)

        button_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        button_frame.grid(row=1, column=0, sticky="e")
        ttk.Button(button_frame, text="OK", command=self._on_ok).grid(row=0, column=0, padx=4)
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).grid(row=0, column=1, padx=4)

        self.bind("<Return>", lambda e: self._on_ok())
        self.bind("<Escape>", lambda e: self._on_cancel())
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.update_idletasks()
        # centre over parent
        try:
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            w = self.winfo_width()
            h = self.winfo_height()
            self.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")
        except Exception:
            pass
        self.grab_set()

    def _build_body(self, master: ttk.Frame) -> None:
        raise NotImplementedError

    def _accept(self) -> Any:
        """Return the dialog's result, or raise ValueError on bad input."""
        raise NotImplementedError

    def _on_ok(self) -> None:
        try:
            self.result = self._accept()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self)
            return
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()


# ── material editor ──────────────────────────────────────────────────────


class MaterialDialog(_ModalDialog):
    """Add or edit a single Material."""

    def __init__(self, parent, *, existing: Material | None, default_id: int):
        self._existing = existing
        self._default_id = default_id
        super().__init__(parent, "Edit material" if existing else "Add material")

    def __init__(self, parent, *, existing: Material | None,
                 existing_section: Section | None = None,
                 default_id: int = 1):
        # Tk path keeps the combined UI for backwards compatibility — the
        # dialog returns a (Material, Section) pair sharing the same id.
        self._existing = existing
        self._existing_section = existing_section
        self._default_id = default_id
        super().__init__(parent, "Edit material" if existing else "Add material")

    def _build_body(self, master: ttk.Frame) -> None:
        labels = [
            ("ID", "id"), ("E (kN/m²)", "E"), ("A (m²)", "A"),
            ("I (m⁴)", "I"), ("α (1/°C)", "alpha"), ("depth (m)", "depth"),
        ]
        self._entries: dict[str, ttk.Entry] = {}
        for r, (lbl, key) in enumerate(labels):
            ttk.Label(master, text=lbl).grid(row=r, column=0, sticky="w", pady=2)
            e = ttk.Entry(master, width=18)
            e.grid(row=r, column=1, pady=2)
            self._entries[key] = e

        if self._existing:
            m = self._existing
            self._entries["id"].insert(0, str(m.id))
            self._entries["id"].config(state="readonly")
            self._entries["E"].insert(0, repr(m.E))
            self._entries["alpha"].insert(0, repr(m.alpha))
            sec = self._existing_section
            self._entries["A"].insert(0, repr(sec.A if sec else 0.0))
            self._entries["I"].insert(0, repr(sec.I if sec else 0.0))
            self._entries["depth"].insert(0, repr(sec.depth if sec else 0.0))
        else:
            self._entries["id"].insert(0, str(self._default_id))
            self._entries["alpha"].insert(0, "0.0")
            self._entries["depth"].insert(0, "0.0")

    def _accept(self) -> tuple[Material, Section]:
        mid = parse_int(self._entries["id"].get(), "Material ID")
        E = parse_float(self._entries["E"].get(), "E")
        A = parse_float(self._entries["A"].get(), "A")
        I = parse_float(self._entries["I"].get(), "I")
        alpha = parse_float(self._entries["alpha"].get(), "α", allow_blank=True) or 0.0
        depth = parse_float(self._entries["depth"].get(), "depth", allow_blank=True) or 0.0
        if E <= 0:
            raise ValueError("E must be > 0.")
        if A <= 0:
            raise ValueError("A must be > 0.")
        if I < 0:
            raise ValueError("I cannot be negative.")
        return (
            Material(id=mid, E=E, alpha=alpha),
            Section(id=mid, material_id=mid, A=A, I=I, depth=depth),
        )


# ── element properties ──────────────────────────────────────────────────


class ElementDialog(_ModalDialog):
    """Configure a new or existing element (kind, section, releases)."""

    def __init__(self, parent, *,
                 model: StructuralModel,
                 existing_kind: str | None = None,
                 existing_section_id: int | None = None,
                 existing_release_i: bool = False,
                 existing_release_j: bool = False):
        self._model = model
        self._existing_kind = existing_kind
        self._existing_sec = existing_section_id
        self._existing_ri = existing_release_i
        self._existing_rj = existing_release_j
        super().__init__(parent, "Element properties")

    def _build_body(self, master: ttk.Frame) -> None:
        ttk.Label(master, text="Kind:").grid(row=0, column=0, sticky="w", pady=2)
        self._kind_var = tk.StringVar(value=self._existing_kind or "frame")
        kf = ttk.Frame(master)
        kf.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(kf, text="Frame", variable=self._kind_var,
                        value="frame", command=self._refresh_release_state
                        ).pack(side="left")
        ttk.Radiobutton(kf, text="Truss", variable=self._kind_var,
                        value="truss", command=self._refresh_release_state
                        ).pack(side="left")

        ttk.Label(master, text="Section:").grid(row=1, column=0, sticky="w", pady=2)
        sections = sorted(self._model.sections.keys())
        if not sections:
            raise ValueError("No sections defined — add a section first.")
        self._sec_var = tk.StringVar(value=str(self._existing_sec or sections[0]))
        cb = ttk.Combobox(master, textvariable=self._sec_var,
                          values=[str(s) for s in sections], state="readonly", width=10)
        cb.grid(row=1, column=1, sticky="w", pady=2)

        self._ri_var = tk.BooleanVar(value=self._existing_ri)
        self._rj_var = tk.BooleanVar(value=self._existing_rj)
        self._ri_cb = ttk.Checkbutton(master, text="Moment release at start (i)",
                                      variable=self._ri_var)
        self._rj_cb = ttk.Checkbutton(master, text="Moment release at end (j)",
                                      variable=self._rj_var)
        self._ri_cb.grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        self._rj_cb.grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        self._refresh_release_state()

    def _refresh_release_state(self) -> None:
        state = "normal" if self._kind_var.get() == "frame" else "disabled"
        self._ri_cb.config(state=state)
        self._rj_cb.config(state=state)

    def _accept(self) -> dict:
        kind = self._kind_var.get()
        section_id = parse_int(self._sec_var.get(), "Section")
        if section_id not in self._model.sections:
            raise ValueError(f"Section {section_id} does not exist.")
        return {
            "kind": kind,
            "section_id": section_id,
            "release_i": bool(self._ri_var.get()) if kind == "frame" else False,
            "release_j": bool(self._rj_var.get()) if kind == "frame" else False,
        }


# ── support ─────────────────────────────────────────────────────────────


class SupportDialog(_ModalDialog):
    def __init__(self, parent, *, existing: Support | None, node_id: int):
        self._existing = existing
        self._node_id = node_id
        super().__init__(parent, f"Support at node {node_id}")

    def _build_body(self, master: ttk.Frame) -> None:
        s = self._existing
        self._ux = tk.BooleanVar(value=bool(s and s.ux))
        self._uy = tk.BooleanVar(value=bool(s and s.uy))
        self._rz = tk.BooleanVar(value=bool(s and s.rz))
        ttk.Checkbutton(master, text="Restrain ux (translate x)",
                        variable=self._ux).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(master, text="Restrain uy (translate y)",
                        variable=self._uy).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(master, text="Restrain rz (rotation)",
                        variable=self._rz).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(master, text="Settlement (blank = none):").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 2))
        labels = [("Δux (m)", "settle_ux"), ("Δuy (m)", "settle_uy"),
                  ("Δrz (rad)", "settle_rz")]
        self._settle: dict[str, ttk.Entry] = {}
        for r, (lbl, key) in enumerate(labels, start=4):
            ttk.Label(master, text=lbl).grid(row=r, column=0, sticky="w", pady=2)
            e = ttk.Entry(master, width=18)
            e.grid(row=r, column=1, pady=2)
            if s is not None:
                v = getattr(s, key)
                if v is not None:
                    e.insert(0, repr(v))
            self._settle[key] = e

        self._remove = tk.BooleanVar(value=False)
        ttk.Checkbutton(master, text="Remove support at this node",
                        variable=self._remove).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(10, 2))

    def _accept(self) -> tuple[str, Support | None]:
        if self._remove.get():
            return ("remove", None)
        ux = self._ux.get()
        uy = self._uy.get()
        rz = self._rz.get()
        if not (ux or uy or rz):
            raise ValueError("Select at least one restrained DOF, "
                             "or check 'Remove support'.")
        s_ux = parse_float(self._settle["settle_ux"].get(), "Δux", allow_blank=True)
        s_uy = parse_float(self._settle["settle_uy"].get(), "Δuy", allow_blank=True)
        s_rz = parse_float(self._settle["settle_rz"].get(), "Δrz", allow_blank=True)
        # only meaningful at restrained DOFs
        if s_ux is not None and not ux:
            raise ValueError("Δux is set but ux is not restrained.")
        if s_uy is not None and not uy:
            raise ValueError("Δuy is set but uy is not restrained.")
        if s_rz is not None and not rz:
            raise ValueError("Δrz is set but rz is not restrained.")
        return ("set", Support(
            node_id=self._node_id, ux=ux, uy=uy, rz=rz,
            settle_ux=s_ux if s_ux not in (None, 0.0) else None,
            settle_uy=s_uy if s_uy not in (None, 0.0) else None,
            settle_rz=s_rz if s_rz not in (None, 0.0) else None,
        ))


# ── nodal load ──────────────────────────────────────────────────────────


class NodalLoadDialog(_ModalDialog):
    def __init__(self, parent, *, existing: NodalLoad | None, node_id: int):
        self._existing = existing
        self._node_id = node_id
        super().__init__(parent, f"Nodal load at node {node_id}")

    def _build_body(self, master: ttk.Frame) -> None:
        labels = [("Fx (kN)", "fx"), ("Fy (kN)", "fy"), ("Mz (kN·m)", "mz")]
        self._entries: dict[str, ttk.Entry] = {}
        for r, (lbl, key) in enumerate(labels):
            ttk.Label(master, text=lbl).grid(row=r, column=0, sticky="w", pady=2)
            e = ttk.Entry(master, width=18)
            e.grid(row=r, column=1, pady=2)
            if self._existing is not None:
                e.insert(0, repr(getattr(self._existing, key)))
            else:
                e.insert(0, "0.0")
            self._entries[key] = e

    def _accept(self) -> tuple[float, float, float]:
        fx = parse_float(self._entries["fx"].get(), "Fx", allow_blank=True) or 0.0
        fy = parse_float(self._entries["fy"].get(), "Fy", allow_blank=True) or 0.0
        mz = parse_float(self._entries["mz"].get(), "Mz", allow_blank=True) or 0.0
        return (fx, fy, mz)


# ── member load ─────────────────────────────────────────────────────────


class MemberLoadDialog(_ModalDialog):
    """Add a UDL / point / thermal member load to an element."""

    def __init__(self, parent, *, model: StructuralModel, elem_id: int):
        self._model = model
        self._elem_id = elem_id
        elem = next((e for e in model.elements if e.id == elem_id), None)
        if elem is None:
            raise ValueError(f"Element {elem_id} does not exist.")
        self._elem = elem
        super().__init__(parent, f"Member load on element {elem_id}")

    def _build_body(self, master: ttk.Frame) -> None:
        is_truss = isinstance(self._elem, TrussElement2D)
        choices = [("UDL (transverse)", "udl"),
                   ("Point load", "point"),
                   ("Thermal (truss ΔT)", "truss_t"),
                   ("Thermal (frame top/bottom)", "frame_t")]
        if is_truss:
            choices = [c for c in choices if c[1] in ("truss_t",)]
        else:
            choices = [c for c in choices if c[1] != "truss_t"]

        ttk.Label(master, text="Type:").grid(row=0, column=0, sticky="w", pady=2)
        self._type_var = tk.StringVar(value=choices[0][1])
        for i, (lbl, val) in enumerate(choices):
            ttk.Radiobutton(master, text=lbl, value=val,
                            variable=self._type_var,
                            command=self._refresh_fields).grid(
                row=0 + i, column=1, sticky="w", pady=1)

        # Field area below
        self._field_frame = ttk.Frame(master)
        self._field_frame.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._refresh_fields()

    def _refresh_fields(self) -> None:
        for w in self._field_frame.winfo_children():
            w.destroy()
        kind = self._type_var.get()
        self._fields: dict[str, ttk.Entry] = {}
        if kind == "udl":
            self._add_field("wy (kN/m, +local-y)", "wy")
        elif kind == "point":
            self._add_field("py (kN, +local-y)", "py")
            self._add_field("a (m from start node)", "a")
        elif kind == "truss_t":
            self._add_field("ΔT (°C)", "delta_T")
        elif kind == "frame_t":
            self._add_field("t_top (°C)", "t_top")
            self._add_field("t_bottom (°C)", "t_bottom")

    def _add_field(self, label: str, key: str) -> None:
        r = len(self._fields)
        ttk.Label(self._field_frame, text=label).grid(row=r, column=0, sticky="w", pady=2)
        e = ttk.Entry(self._field_frame, width=18)
        e.grid(row=r, column=1, pady=2)
        e.insert(0, "0.0")
        self._fields[key] = e

    def _accept(self) -> Any:
        kind = self._type_var.get()
        if kind == "udl":
            wy = parse_float(self._fields["wy"].get(), "wy")
            return UniformDistributedLoad(wy=wy)
        if kind == "point":
            py = parse_float(self._fields["py"].get(), "py")
            a = parse_float(self._fields["a"].get(), "a")
            L, _, _ = self._elem.length_cos_sin(self._model.nodes)
            if a < 0 or a > L:
                raise ValueError(f"a must lie within [0, {L:.3g}] (element length).")
            return PointLoad(py=py, a=a)
        if kind == "truss_t":
            dT = parse_float(self._fields["delta_T"].get(), "ΔT")
            return TrussTemperatureLoad(delta_T=dT)
        if kind == "frame_t":
            t_top = parse_float(self._fields["t_top"].get(), "t_top")
            t_bot = parse_float(self._fields["t_bottom"].get(), "t_bottom")
            return FrameTemperatureLoad(t_top=t_top, t_bottom=t_bot)
        raise ValueError(f"Unknown load type {kind!r}.")


# ── grid spacing ────────────────────────────────────────────────────────


class GridSpacingDialog(_ModalDialog):
    def __init__(self, parent, *, current: float):
        self._current = current
        super().__init__(parent, "Grid spacing")

    def _build_body(self, master: ttk.Frame) -> None:
        ttk.Label(master, text="Grid spacing (m):").grid(row=0, column=0, sticky="w", pady=2)
        self._entry = ttk.Entry(master, width=12)
        self._entry.grid(row=0, column=1, pady=2)
        self._entry.insert(0, repr(self._current))

    def _accept(self) -> float:
        v = parse_float(self._entry.get(), "Grid spacing")
        if v <= 0:
            raise ValueError("Grid spacing must be > 0.")
        return v


# ── material list editor (composes MaterialDialog) ──────────────────────


class MaterialListDialog(_ModalDialog):
    """Browse / add / edit / delete the list of materials."""

    def __init__(self, parent, *, model: StructuralModel,
                 on_add_or_update: Callable[[Material], None],
                 on_delete: Callable[[int], None]):
        self._model = model
        self._on_add_or_update = on_add_or_update
        self._on_delete = on_delete
        super().__init__(parent, "Materials")

    def _build_body(self, master: ttk.Frame) -> None:
        cols = ("id", "E", "A", "I", "alpha", "depth")
        self._tree = ttk.Treeview(master, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (50, 110, 90, 110, 90, 80)):
            self._tree.heading(c, text=c)
            self._tree.column(c, width=w, anchor="e")
        self._tree.grid(row=0, column=0, columnspan=4, pady=2)
        self._refresh()
        ttk.Button(master, text="Add",  command=self._add).grid(row=1, column=0, pady=4)
        ttk.Button(master, text="Edit", command=self._edit).grid(row=1, column=1, pady=4)
        ttk.Button(master, text="Delete", command=self._delete).grid(row=1, column=2, pady=4)

    def _refresh(self) -> None:
        for it in self._tree.get_children():
            self._tree.delete(it)
        # Show one row per Material, pulling A/I/depth from the matching
        # Section (1:1 association is the legacy norm).
        for mid in sorted(self._model.materials):
            m = self._model.materials[mid]
            sec = self._model.sections.get(mid)
            A = sec.A if sec else 0.0
            I = sec.I if sec else 0.0
            depth = sec.depth if sec else 0.0
            self._tree.insert("", "end", iid=str(mid), values=(
                m.id, f"{m.E:g}", f"{A:g}", f"{I:g}",
                f"{m.alpha:g}", f"{depth:g}",
            ))

    def _selected_id(self) -> int | None:
        sel = self._tree.selection()
        return int(sel[0]) if sel else None

    def _add(self) -> None:
        existing_ids = list(self._model.materials.keys())
        next_id = (max(existing_ids) + 1) if existing_ids else 1
        d = MaterialDialog(self, existing=None, default_id=next_id)
        self.wait_window(d)
        if d.result is not None:
            # ``_on_add_or_update`` is wired to MainApplication.execute(), which
            # already routes ValueError to a messagebox — don't duplicate the
            # catch here, just refresh the list view.
            self._on_add_or_update(d.result)
            self._refresh()

    def _edit(self) -> None:
        mid = self._selected_id()
        if mid is None:
            return
        d = MaterialDialog(
            self,
            existing=self._model.materials[mid],
            existing_section=self._model.sections.get(mid),
            default_id=mid,
        )
        self.wait_window(d)
        if d.result is not None:
            self._on_add_or_update(d.result)
            self._refresh()

    def _delete(self) -> None:
        mid = self._selected_id()
        if mid is None:
            return
        # Errors are reported by MainApplication.execute() via its own dialog.
        self._on_delete(mid)
        self._refresh()

    def _accept(self) -> None:
        return None  # changes were applied incrementally
