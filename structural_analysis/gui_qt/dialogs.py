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

from PyQt6.QtWidgets import QTabWidget

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
    """Edit a single Material (id, name, E, α). Section properties are
    edited separately via :class:`SectionDialog`."""

    def __init__(self, parent, *, existing: Material | None, default_id: int):
        self._existing = existing
        self._default_id = default_id
        super().__init__(parent, "Edit material" if existing else "Add material")

    def _build_body(self, body: QWidget) -> None:
        form = QFormLayout(body)
        self._entries: dict[str, QLineEdit] = {}
        for label, key in [("ID", "id"), ("Name", "name"),
                           ("E (kN/m²)", "E"), ("α (1/°C)", "alpha"),
                           ("density (kg/m³)", "density")]:
            e = QLineEdit(body)
            form.addRow(label, e)
            self._entries[key] = e

        if self._existing:
            m = self._existing
            self._entries["id"].setText(str(m.id))
            self._entries["id"].setReadOnly(True)
            self._entries["name"].setText(m.name)
            self._entries["E"].setText(repr(m.E))
            self._entries["alpha"].setText(repr(m.alpha))
            self._entries["density"].setText(repr(m.density))
        else:
            self._entries["id"].setText(str(self._default_id))
            self._entries["alpha"].setText("0.0")
            self._entries["density"].setText("0.0")

    def _accept(self) -> Material:
        mid = parse_int(self._entries["id"].text(), "Material ID")
        name = self._entries["name"].text().strip()
        E = parse_float(self._entries["E"].text(), "E")
        alpha = parse_float(self._entries["alpha"].text(), "α", allow_blank=True) or 0.0
        density = parse_float(self._entries["density"].text(), "density",
                              allow_blank=True) or 0.0
        if E <= 0:
            raise ValueError("E must be > 0.")
        if density < 0:
            raise ValueError("density cannot be negative.")
        return Material(id=mid, name=name, E=E, alpha=alpha, density=density)


class SectionDialog(_ModalDialog):
    """Edit a single Section (id, name, material_id, A, I, depth)."""

    def __init__(self, parent, *,
                 model: StructuralModel,
                 existing: Section | None,
                 default_id: int):
        self._model = model
        self._existing = existing
        self._default_id = default_id
        if not model.materials:
            raise ValueError("No materials defined — add a material first.")
        super().__init__(parent, "Edit section" if existing else "Add section")

    def _build_body(self, body: QWidget) -> None:
        form = QFormLayout(body)
        self._id_entry = QLineEdit(body)
        form.addRow("ID", self._id_entry)
        self._name_entry = QLineEdit(body)
        form.addRow("Name", self._name_entry)

        self._mat_combo = QComboBox(body)
        for mid in sorted(self._model.materials):
            m = self._model.materials[mid]
            label = m.name if m.name else f"material {mid}"
            self._mat_combo.addItem(label, mid)
        form.addRow("Material", self._mat_combo)

        self._a_entry = QLineEdit(body)
        self._i_entry = QLineEdit(body)
        self._d_entry = QLineEdit(body)
        form.addRow("A (m²)", self._a_entry)
        form.addRow("I (m⁴)", self._i_entry)
        form.addRow("depth (m)", self._d_entry)

        if self._existing:
            s = self._existing
            self._id_entry.setText(str(s.id))
            self._id_entry.setReadOnly(True)
            self._name_entry.setText(s.name)
            idx = self._mat_combo.findData(s.material_id)
            if idx >= 0:
                self._mat_combo.setCurrentIndex(idx)
            self._a_entry.setText(repr(s.A))
            self._i_entry.setText(repr(s.I))
            self._d_entry.setText(repr(s.depth))
        else:
            self._id_entry.setText(str(self._default_id))
            self._d_entry.setText("0.0")

    def _accept(self) -> Section:
        sid = parse_int(self._id_entry.text(), "Section ID")
        name = self._name_entry.text().strip()
        material_id = self._mat_combo.currentData()
        if material_id not in self._model.materials:
            raise ValueError(f"Material {material_id} does not exist.")
        A = parse_float(self._a_entry.text(), "A")
        I = parse_float(self._i_entry.text(), "I")
        depth = parse_float(self._d_entry.text(), "depth", allow_blank=True) or 0.0
        if A <= 0:
            raise ValueError("A must be > 0.")
        if I < 0:
            raise ValueError("I cannot be negative.")
        return Section(id=sid, name=name, material_id=int(material_id),
                       A=A, I=I, depth=depth)


# ── element properties ──


class ElementDialog(_ModalDialog):
    def __init__(self, parent, *, model: StructuralModel,
                 existing_kind: str | None = None,
                 existing_section_id: int | None = None,
                 existing_release_i: bool = False,
                 existing_release_j: bool = False,
                 remember_default: bool = True):
        self._model = model
        if not model.sections:
            raise ValueError("No sections defined — add a section first.")
        self._existing_kind = existing_kind
        self._existing_sec = existing_section_id
        self._existing_ri = existing_release_i
        self._existing_rj = existing_release_j
        self._remember_default = bool(remember_default)
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

        self._sec_combo = QComboBox(body)
        for sid in sorted(self._model.sections):
            s = self._model.sections[sid]
            mat = self._model.materials.get(s.material_id)
            mat_name = (mat.name if mat and mat.name
                         else f"material {s.material_id}")
            sec_name = s.name if s.name else f"section {sid}"
            label = f"{sec_name} / {mat_name}"
            self._sec_combo.addItem(label, sid)
        if self._existing_sec is not None:
            idx = self._sec_combo.findData(self._existing_sec)
            if idx >= 0:
                self._sec_combo.setCurrentIndex(idx)
        form.addRow("Section / material:", self._sec_combo)

        self._cb_ri = QCheckBox("Moment release at start (i)", body)
        self._cb_rj = QCheckBox("Moment release at end (j)", body)
        self._cb_ri.setChecked(self._existing_ri)
        self._cb_rj.setChecked(self._existing_rj)
        form.addRow(self._cb_ri)
        form.addRow(self._cb_rj)

        self._cb_remember = QCheckBox(
            "Remember and reuse these settings for subsequent elements",
            body,
        )
        self._cb_remember.setChecked(self._remember_default)
        form.addRow(self._cb_remember)

        self._refresh_release_state()

    def _refresh_release_state(self) -> None:
        is_frame = self._rb_frame.isChecked()
        self._cb_ri.setEnabled(is_frame)
        self._cb_rj.setEnabled(is_frame)

    def _accept(self) -> dict:
        kind = "frame" if self._rb_frame.isChecked() else "truss"
        section_id = self._sec_combo.currentData()
        if section_id not in self._model.sections:
            raise ValueError(f"Section {section_id} does not exist.")
        return {
            "kind": kind,
            "section_id": int(section_id),
            "release_i": self._cb_ri.isChecked() if kind == "frame" else False,
            "release_j": self._cb_rj.isChecked() if kind == "frame" else False,
            "remember": self._cb_remember.isChecked(),
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


# ── labeled grid system ──


class GridDialog(_ModalDialog):
    """Define the X and Y grid lines (SAP2000-style)."""

    def __init__(self, parent, *, current=None):
        # ``current`` is an optional GridSystem (None ⇒ blank).
        self._current = current
        super().__init__(parent, "Grid system")

    def _build_body(self, body: QWidget) -> None:
        v = QVBoxLayout(body)
        v.addWidget(QLabel(
            "Enter labels and coordinates as comma-separated pairs.\n"
            "Example: A=0, B=4, C=8, D=12  /  1=0, 2=3, 3=6",
            body,
        ))
        form = QFormLayout()
        self._x_entry = QLineEdit(body)
        self._y_entry = QLineEdit(body)
        form.addRow("X lines:", self._x_entry)
        form.addRow("Y lines:", self._y_entry)
        v.addLayout(form)

        if self._current is not None:
            self._x_entry.setText(", ".join(
                f"{ln.label}={ln.coord:g}" for ln in self._current.x_lines))
            self._y_entry.setText(", ".join(
                f"{ln.label}={ln.coord:g}" for ln in self._current.y_lines))

    def _parse_axis(self, text: str, axis_name: str):
        from .grid import GridLine
        lines: list = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    f"{axis_name} entry '{part}' is not in 'label=value' form."
                )
            label, coord_s = part.split("=", 1)
            label = label.strip()
            if not label:
                raise ValueError(f"{axis_name} entry has empty label.")
            coord = parse_float(coord_s.strip(), f"{axis_name} '{label}' coord")
            lines.append(GridLine(label=label, coord=coord))
        return lines

    def _accept(self):
        from .grid import GridSystem
        x_lines = self._parse_axis(self._x_entry.text(), "X")
        y_lines = self._parse_axis(self._y_entry.text(), "Y")
        if not x_lines and not y_lines:
            # Empty grid is permitted (treats canvas as blank).
            return GridSystem(x_lines=[], y_lines=[])
        return GridSystem(x_lines=x_lines, y_lines=y_lines)


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
    """Tabbed Materials + Sections editor.

    The callbacks are wired by the host:
        on_add_or_update_material(Material)
        on_delete_material(int)
        on_add_or_update_section(Section)
        on_delete_section(int)
    Each callback dispatches the matching command via MainWindow.execute().
    """

    def __init__(self, parent, *, model: StructuralModel,
                 on_add_or_update_material,
                 on_delete_material,
                 on_add_or_update_section,
                 on_delete_section) -> None:
        self._model = model
        self._on_add_or_update_material = on_add_or_update_material
        self._on_delete_material = on_delete_material
        self._on_add_or_update_section = on_add_or_update_section
        self._on_delete_section = on_delete_section
        super().__init__(parent, "Materials and Sections")

    def _build_body(self, body: QWidget) -> None:
        v = QVBoxLayout(body)
        self._tabs = QTabWidget(body)
        v.addWidget(self._tabs)

        # ── Materials tab ──
        mat_page = QWidget(self._tabs)
        ml = QVBoxLayout(mat_page)
        self._mat_tree = QTreeWidget(mat_page)
        self._mat_tree.setHeaderLabels(
            ["id", "name", "E (kN/m²)", "α (1/°C)", "ρ (kg/m³)"]
        )
        self._mat_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        ml.addWidget(self._mat_tree)
        mb = QHBoxLayout()
        for label, slot in [("Add", self._add_mat),
                             ("Edit", self._edit_mat),
                             ("Delete", self._delete_mat)]:
            b = QPushButton(label, mat_page)
            b.clicked.connect(slot)
            mb.addWidget(b)
        mb.addStretch(1)
        ml.addLayout(mb)
        self._tabs.addTab(mat_page, "Materials")

        # ── Sections tab ──
        sec_page = QWidget(self._tabs)
        sl = QVBoxLayout(sec_page)
        self._sec_tree = QTreeWidget(sec_page)
        self._sec_tree.setHeaderLabels(
            ["id", "name", "material", "A (m²)", "I (m⁴)", "depth (m)"]
        )
        self._sec_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        sl.addWidget(self._sec_tree)
        sb = QHBoxLayout()
        for label, slot in [("Add", self._add_sec),
                             ("Edit", self._edit_sec),
                             ("Delete", self._delete_sec)]:
            b = QPushButton(label, sec_page)
            b.clicked.connect(slot)
            sb.addWidget(b)
        sb.addStretch(1)
        sl.addLayout(sb)
        self._tabs.addTab(sec_page, "Sections")

        self._refresh()

    def _refresh(self) -> None:
        self._mat_tree.clear()
        for mid in sorted(self._model.materials):
            m = self._model.materials[mid]
            QTreeWidgetItem(self._mat_tree, [
                str(m.id), m.name, f"{m.E:g}", f"{m.alpha:g}",
                f"{m.density:g}",
            ])
        self._sec_tree.clear()
        for sid in sorted(self._model.sections):
            s = self._model.sections[sid]
            QTreeWidgetItem(self._sec_tree, [
                str(s.id), s.name, str(s.material_id),
                f"{s.A:g}", f"{s.I:g}", f"{s.depth:g}",
            ])

    def _selected_id(self, tree: QTreeWidget) -> int | None:
        items = tree.selectedItems()
        if not items:
            return None
        return int(items[0].text(0))

    # ── Materials tab handlers ──
    def _add_mat(self) -> None:
        existing = list(self._model.materials.keys())
        next_id = (max(existing) + 1) if existing else 1
        d = MaterialDialog(self, existing=None, default_id=next_id)
        if d.exec() == QDialog.DialogCode.Accepted and d.result_value is not None:
            self._on_add_or_update_material(d.result_value)
            self._refresh()

    def _edit_mat(self) -> None:
        mid = self._selected_id(self._mat_tree)
        if mid is None:
            QMessageBox.information(self, "No selection",
                                      "Select a material row in the table first.")
            return
        d = MaterialDialog(self, existing=self._model.materials[mid], default_id=mid)
        if d.exec() == QDialog.DialogCode.Accepted and d.result_value is not None:
            self._on_add_or_update_material(d.result_value)
            self._refresh()

    def _delete_mat(self) -> None:
        mid = self._selected_id(self._mat_tree)
        if mid is None:
            QMessageBox.information(self, "No selection",
                                      "Select a material row in the table first.")
            return
        self._on_delete_material(mid)
        self._refresh()

    # ── Sections tab handlers ──
    def _add_sec(self) -> None:
        existing = list(self._model.sections.keys())
        next_id = (max(existing) + 1) if existing else 1
        try:
            d = SectionDialog(self, model=self._model, existing=None,
                               default_id=next_id)
        except ValueError as e:
            QMessageBox.warning(self, "Cannot add section", str(e))
            return
        if d.exec() == QDialog.DialogCode.Accepted and d.result_value is not None:
            self._on_add_or_update_section(d.result_value)
            self._refresh()

    def _edit_sec(self) -> None:
        sid = self._selected_id(self._sec_tree)
        if sid is None:
            QMessageBox.information(self, "No selection",
                                      "Select a section row in the table first.")
            return
        d = SectionDialog(self, model=self._model,
                           existing=self._model.sections[sid], default_id=sid)
        if d.exec() == QDialog.DialogCode.Accepted and d.result_value is not None:
            self._on_add_or_update_section(d.result_value)
            self._refresh()

    def _delete_sec(self) -> None:
        sid = self._selected_id(self._sec_tree)
        if sid is None:
            QMessageBox.information(self, "No selection",
                                      "Select a section row in the table first.")
            return
        self._on_delete_section(sid)
        self._refresh()

    def _accept(self) -> None:
        return None


# ── modal analysis input dialog ────────────────────────────────


class ModalAnalysisDialog(_ModalDialog):
    """Ask the user how many modes to extract and the normalisation."""

    def __init__(self, parent, *, default_n_modes: int = 6) -> None:
        self._default_n_modes = max(1, int(default_n_modes))
        super().__init__(parent, "Modal analysis")

    def _build_body(self, body: QWidget) -> None:
        form = QFormLayout(body)
        self._n_modes = QLineEdit(body)
        self._n_modes.setText(str(self._default_n_modes))
        form.addRow("Number of modes", self._n_modes)

        self._norm_combo = QComboBox(body)
        self._norm_combo.addItem("Mass-orthonormal  (φᵀ·M·φ = 1)", "mass")
        self._norm_combo.addItem("Max-component = 1", "max")
        form.addRow("Normalisation", self._norm_combo)

        note = QLabel(
            "Modal analysis requires a positive density on every "
            "element's material.\nSet density (kg/m³) on each Material "
            "via Edit → Materials and sections.",
            body,
        )
        note.setWordWrap(True)
        form.addRow(note)

    def _accept(self) -> dict:
        n = parse_int(self._n_modes.text(), "Number of modes")
        if n < 1:
            raise ValueError("Number of modes must be at least 1.")
        norm = self._norm_combo.currentData()
        return {"n_modes": n, "normalisation": norm}
