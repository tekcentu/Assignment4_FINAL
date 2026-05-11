"""PyQt6 modal dialogs for editing model entities.

Numeric fields go through ``parse_float`` so users see a friendly message
instead of a Python traceback when they type bad input.
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..element import FrameElement2D, TrussElement2D
from ..model import (
    FrameTemperatureLoad,
    Material,
    NodalLoad,
    PointLoad,
    StructuralModel,
    Support,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)


def parse_float(text: str, name: str, *, allow_blank: bool = False) -> Optional[float]:
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


class _ModalDialog(QDialog):
    """Shared base — wires OK/Cancel buttons and centres an _accept() hook."""

    def __init__(self, parent: QWidget | None, title: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.result_value: Any = None

        layout = QVBoxLayout(self)
        self._body = QWidget(self)
        layout.addWidget(self._body)
        self._build_body(self._body)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_body(self, body: QWidget) -> None:
        raise NotImplementedError

    def _accept(self) -> Any:
        raise NotImplementedError

    def _on_ok(self) -> None:
        try:
            self.result_value = self._accept()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid input", str(e))
            return
        self.accept()


# ── material editor ──


class MaterialDialog(_ModalDialog):
    def __init__(self, parent, *, existing: Material | None, default_id: int):
        self._existing = existing
        self._default_id = default_id
        super().__init__(parent, "Edit material" if existing else "Add material")

    def _build_body(self, body: QWidget) -> None:
        form = QFormLayout(body)
        self._entries: dict[str, QLineEdit] = {}
        labels = [("ID", "id"), ("E (kN/m²)", "E"), ("A (m²)", "A"),
                  ("I (m⁴)", "I"), ("α (1/°C)", "alpha"), ("depth (m)", "depth")]
        for label, key in labels:
            e = QLineEdit(body)
            form.addRow(label, e)
            self._entries[key] = e

        if self._existing:
            m = self._existing
            self._entries["id"].setText(str(m.id))
            self._entries["id"].setReadOnly(True)
            self._entries["E"].setText(repr(m.E))
            self._entries["A"].setText(repr(m.A))
            self._entries["I"].setText(repr(m.I))
            self._entries["alpha"].setText(repr(m.alpha))
            self._entries["depth"].setText(repr(m.depth))
        else:
            self._entries["id"].setText(str(self._default_id))
            self._entries["alpha"].setText("0.0")
            self._entries["depth"].setText("0.0")

    def _accept(self) -> Material:
        mid = parse_int(self._entries["id"].text(), "Material ID")
        E = parse_float(self._entries["E"].text(), "E")
        A = parse_float(self._entries["A"].text(), "A")
        I = parse_float(self._entries["I"].text(), "I")
        alpha = parse_float(self._entries["alpha"].text(), "α", allow_blank=True) or 0.0
        depth = parse_float(self._entries["depth"].text(), "depth", allow_blank=True) or 0.0
        if E <= 0:
            raise ValueError("E must be > 0.")
        if A <= 0:
            raise ValueError("A must be > 0.")
        if I < 0:
            raise ValueError("I cannot be negative.")
        return Material(id=mid, E=E, A=A, I=I, alpha=alpha, depth=depth)


# ── element properties ──


class ElementDialog(_ModalDialog):
    def __init__(self, parent, *, model: StructuralModel,
                 existing_kind: str | None = None,
                 existing_material_id: int | None = None,
                 existing_release_i: bool = False,
                 existing_release_j: bool = False):
        self._model = model
        if not model.materials:
            raise ValueError("No materials defined — add a material first.")
        self._existing_kind = existing_kind
        self._existing_mat = existing_material_id
        self._existing_ri = existing_release_i
        self._existing_rj = existing_release_j
        super().__init__(parent, "Element properties")

    def _build_body(self, body: QWidget) -> None:
        form = QFormLayout(body)
        kind_box = QWidget(body)
        h = QHBoxLayout(kind_box)
        h.setContentsMargins(0, 0, 0, 0)
        self._rb_frame = QRadioButton("Frame", kind_box)
        self._rb_truss = QRadioButton("Truss", kind_box)
        h.addWidget(self._rb_frame)
        h.addWidget(self._rb_truss)
        h.addStretch(1)
        if (self._existing_kind or "frame") == "truss":
            self._rb_truss.setChecked(True)
        else:
            self._rb_frame.setChecked(True)
        self._rb_frame.toggled.connect(self._refresh_release_state)
        form.addRow("Kind:", kind_box)

        self._mat_combo = QComboBox(body)
        for mid in sorted(self._model.materials):
            self._mat_combo.addItem(str(mid), mid)
        if self._existing_mat is not None:
            idx = self._mat_combo.findData(self._existing_mat)
            if idx >= 0:
                self._mat_combo.setCurrentIndex(idx)
        form.addRow("Material:", self._mat_combo)

        self._cb_ri = QCheckBox("Moment release at start (i)", body)
        self._cb_rj = QCheckBox("Moment release at end (j)", body)
        self._cb_ri.setChecked(self._existing_ri)
        self._cb_rj.setChecked(self._existing_rj)
        form.addRow(self._cb_ri)
        form.addRow(self._cb_rj)
        self._refresh_release_state()

    def _refresh_release_state(self) -> None:
        is_frame = self._rb_frame.isChecked()
        self._cb_ri.setEnabled(is_frame)
        self._cb_rj.setEnabled(is_frame)

    def _accept(self) -> dict:
        kind = "frame" if self._rb_frame.isChecked() else "truss"
        material_id = self._mat_combo.currentData()
        if material_id not in self._model.materials:
            raise ValueError(f"Material {material_id} does not exist.")
        return {
            "kind": kind,
            "material_id": int(material_id),
            "release_i": self._cb_ri.isChecked() if kind == "frame" else False,
            "release_j": self._cb_rj.isChecked() if kind == "frame" else False,
        }


# ── support ──


class SupportDialog(_ModalDialog):
    def __init__(self, parent, *, existing: Support | None, node_id: int):
        self._existing = existing
        self._node_id = node_id
        super().__init__(parent, f"Support at node {node_id}")

    def _build_body(self, body: QWidget) -> None:
        v = QVBoxLayout(body)
        s = self._existing
        self._cb_ux = QCheckBox("Restrain ux (translate x)", body)
        self._cb_uy = QCheckBox("Restrain uy (translate y)", body)
        self._cb_rz = QCheckBox("Restrain rz (rotation)", body)
        if s:
            self._cb_ux.setChecked(s.ux)
            self._cb_uy.setChecked(s.uy)
            self._cb_rz.setChecked(s.rz)
        v.addWidget(self._cb_ux)
        v.addWidget(self._cb_uy)
        v.addWidget(self._cb_rz)

        v.addWidget(QLabel("Settlement (blank = none):", body))
        form = QFormLayout()
        self._settle: dict[str, QLineEdit] = {}
        for label, key in [("Δux (m)", "settle_ux"),
                           ("Δuy (m)", "settle_uy"),
                           ("Δrz (rad)", "settle_rz")]:
            e = QLineEdit(body)
            if s is not None:
                val = getattr(s, key)
                if val is not None:
                    e.setText(repr(val))
            form.addRow(label, e)
            self._settle[key] = e
        v.addLayout(form)

        self._cb_remove = QCheckBox("Remove support at this node", body)
        v.addWidget(self._cb_remove)

    def _accept(self) -> tuple[str, Support | None]:
        if self._cb_remove.isChecked():
            return ("remove", None)
        ux = self._cb_ux.isChecked()
        uy = self._cb_uy.isChecked()
        rz = self._cb_rz.isChecked()
        if not (ux or uy or rz):
            raise ValueError("Select at least one restrained DOF, "
                             "or check 'Remove support'.")
        s_ux = parse_float(self._settle["settle_ux"].text(), "Δux", allow_blank=True)
        s_uy = parse_float(self._settle["settle_uy"].text(), "Δuy", allow_blank=True)
        s_rz = parse_float(self._settle["settle_rz"].text(), "Δrz", allow_blank=True)
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


# ── nodal load ──


class NodalLoadDialog(_ModalDialog):
    def __init__(self, parent, *, existing: NodalLoad | None, node_id: int):
        self._existing = existing
        self._node_id = node_id
        super().__init__(parent, f"Nodal load at node {node_id}")

    def _build_body(self, body: QWidget) -> None:
        form = QFormLayout(body)
        self._entries: dict[str, QLineEdit] = {}
        for label, key in [("Fx (kN)", "fx"), ("Fy (kN)", "fy"), ("Mz (kN·m)", "mz")]:
            e = QLineEdit(body)
            if self._existing is not None:
                e.setText(repr(getattr(self._existing, key)))
            else:
                e.setText("0.0")
            form.addRow(label, e)
            self._entries[key] = e

    def _accept(self) -> tuple[float, float, float]:
        fx = parse_float(self._entries["fx"].text(), "Fx", allow_blank=True) or 0.0
        fy = parse_float(self._entries["fy"].text(), "Fy", allow_blank=True) or 0.0
        mz = parse_float(self._entries["mz"].text(), "Mz", allow_blank=True) or 0.0
        return (fx, fy, mz)


# ── member load ──


class MemberLoadDialog(_ModalDialog):
    def __init__(self, parent, *, model: StructuralModel, elem_id: int):
        self._model = model
        self._elem_id = elem_id
        elem = next((e for e in model.elements if e.id == elem_id), None)
        if elem is None:
            raise ValueError(f"Element {elem_id} does not exist.")
        self._elem = elem
        super().__init__(parent, f"Member load on element {elem_id}")

    def _build_body(self, body: QWidget) -> None:
        v = QVBoxLayout(body)
        is_truss = isinstance(self._elem, TrussElement2D)
        choices = [("UDL (transverse)", "udl"),
                   ("Point load", "point"),
                   ("Thermal (truss ΔT)", "truss_t"),
                   ("Thermal (frame top/bottom)", "frame_t")]
        if is_truss:
            choices = [c for c in choices if c[1] == "truss_t"]
        else:
            choices = [c for c in choices if c[1] != "truss_t"]

        # Create field container first — the toggled signal handler reads it.
        v.addWidget(QLabel("Type:", body))
        self._group = QButtonGroup(body)
        self._radios: dict[str, QRadioButton] = {}
        radio_buttons: list[tuple[str, QRadioButton]] = []
        for label, val in choices:
            rb = QRadioButton(label, body)
            v.addWidget(rb)
            self._group.addButton(rb)
            self._radios[val] = rb
            radio_buttons.append((val, rb))

        self._field_container = QWidget(body)
        self._field_form = QFormLayout(self._field_container)
        v.addWidget(self._field_container)
        self._fields: dict[str, QLineEdit] = {}

        # Now safely wire the toggled signal and default-select the first option.
        for val, rb in radio_buttons:
            rb.toggled.connect(self._refresh_fields)
        first_key = choices[0][1]
        self._radios[first_key].setChecked(True)
        self._refresh_fields()

    def _current_kind(self) -> str:
        for key, rb in self._radios.items():
            if rb.isChecked():
                return key
        return next(iter(self._radios))

    def _refresh_fields(self) -> None:
        # clear
        while self._field_form.count():
            item = self._field_form.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._fields = {}
        kind = self._current_kind()
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
        e = QLineEdit(self._field_container)
        e.setText("0.0")
        self._field_form.addRow(label, e)
        self._fields[key] = e

    def _accept(self) -> Any:
        kind = self._current_kind()
        if kind == "udl":
            return UniformDistributedLoad(wy=parse_float(self._fields["wy"].text(), "wy"))
        if kind == "point":
            py = parse_float(self._fields["py"].text(), "py")
            a = parse_float(self._fields["a"].text(), "a")
            L, _, _ = self._elem.length_cos_sin(self._model.nodes)
            if a < 0 or a > L:
                raise ValueError(f"a must lie within [0, {L:.3g}] (element length).")
            return PointLoad(py=py, a=a)
        if kind == "truss_t":
            return TrussTemperatureLoad(
                delta_T=parse_float(self._fields["delta_T"].text(), "ΔT"))
        if kind == "frame_t":
            return FrameTemperatureLoad(
                t_top=parse_float(self._fields["t_top"].text(), "t_top"),
                t_bottom=parse_float(self._fields["t_bottom"].text(), "t_bottom"),
            )
        raise ValueError(f"Unknown load type {kind!r}.")


# ── grid spacing ──


class GridSpacingDialog(_ModalDialog):
    def __init__(self, parent, *, current: float):
        self._current = current
        super().__init__(parent, "Grid spacing")

    def _build_body(self, body: QWidget) -> None:
        form = QFormLayout(body)
        self._entry = QLineEdit(body)
        self._entry.setText(repr(self._current))
        form.addRow("Grid spacing (m):", self._entry)

    def _accept(self) -> float:
        v = parse_float(self._entry.text(), "Grid spacing")
        if v <= 0:
            raise ValueError("Grid spacing must be > 0.")
        return v


# ── material list editor ──


class MaterialListDialog(_ModalDialog):
    """Browse / add / edit / delete materials."""

    def __init__(self, parent, *, model: StructuralModel,
                 on_add_or_update, on_delete) -> None:
        self._model = model
        self._on_add_or_update = on_add_or_update
        self._on_delete = on_delete
        super().__init__(parent, "Materials")

    def _build_body(self, body: QWidget) -> None:
        v = QVBoxLayout(body)
        self._tree = QTreeWidget(body)
        self._tree.setHeaderLabels(["id", "E", "A", "I", "α", "depth"])
        self._tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        v.addWidget(self._tree)
        self._refresh()

        btns = QHBoxLayout()
        b_add = QPushButton("Add", body)
        b_edit = QPushButton("Edit", body)
        b_del = QPushButton("Delete", body)
        b_add.clicked.connect(self._add)
        b_edit.clicked.connect(self._edit)
        b_del.clicked.connect(self._delete)
        btns.addWidget(b_add)
        btns.addWidget(b_edit)
        btns.addWidget(b_del)
        btns.addStretch(1)
        v.addLayout(btns)

    def _refresh(self) -> None:
        self._tree.clear()
        for mid in sorted(self._model.materials):
            m = self._model.materials[mid]
            QTreeWidgetItem(self._tree, [
                str(m.id), f"{m.E:g}", f"{m.A:g}", f"{m.I:g}",
                f"{m.alpha:g}", f"{m.depth:g}",
            ])

    def _selected_id(self) -> int | None:
        items = self._tree.selectedItems()
        if not items:
            return None
        return int(items[0].text(0))

    def _add(self) -> None:
        existing_ids = list(self._model.materials.keys())
        next_id = (max(existing_ids) + 1) if existing_ids else 1
        d = MaterialDialog(self, existing=None, default_id=next_id)
        if d.exec() == QDialog.DialogCode.Accepted and d.result_value is not None:
            self._on_add_or_update(d.result_value)
            self._refresh()

    def _edit(self) -> None:
        mid = self._selected_id()
        if mid is None:
            return
        d = MaterialDialog(self, existing=self._model.materials[mid], default_id=mid)
        if d.exec() == QDialog.DialogCode.Accepted and d.result_value is not None:
            self._on_add_or_update(d.result_value)
            self._refresh()

    def _delete(self) -> None:
        mid = self._selected_id()
        if mid is None:
            return
        # Errors are reported by MainWindow.execute() via its own dialog.
        self._on_delete(mid)
        self._refresh()

    def _accept(self) -> None:
        return None
