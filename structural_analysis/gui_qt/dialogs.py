"""PyQt6 modal dialogs for editing model entities.

Numeric fields go through ``parse_float`` so users see a friendly message
instead of a Python traceback when they type bad input.
"""

from __future__ import annotations

from typing import Any, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PyQt6.QtWidgets import QTabWidget

from ..element import FrameElement2D, TrussElement2D
from ..model import (
    FrameTemperatureLoad,
    LoadCase,
    Material,
    NodalLoad,
    Node,
    PointLoad,
    Section,
    StructuralModel,
    Support,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)
from ..profiles import (
    MATERIAL_TEMPLATES,
    SECTION_SHAPES,
    properties_for_shape,
    section_outline,
)

from PyQt6.QtWidgets import QStackedWidget

from .table_copy import install_table_copy


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
    """Edit a single Material.

    Adds a "Template" combo that fills E/α/ρ/ν from
    :data:`structural_analysis.profiles.MATERIAL_TEMPLATES`, a ν
    (Poisson) field validated to ``0 ≤ ν < 0.5``, and a read-only
    derived G display (``G = E / (2·(1+ν))``).
    """

    def __init__(self, parent, *, existing: Material | None, default_id: int):
        self._existing = existing
        self._default_id = default_id
        super().__init__(parent, "Edit material" if existing else "Add material")

    def _build_body(self, body: QWidget) -> None:
        form = QFormLayout(body)

        # Template combo
        self._template_combo = QComboBox(body)
        self._template_combo.addItem("(custom)", "")
        for key in MATERIAL_TEMPLATES:
            self._template_combo.addItem(key, key)
        form.addRow("Template", self._template_combo)

        self._entries: dict[str, QLineEdit] = {}
        for label, key in [("ID", "id"), ("Name", "name"),
                           ("E (kN/m²)", "E"), ("α (1/°C)", "alpha"),
                           ("density (kg/m³)", "density"),
                           ("ν (Poisson)", "nu")]:
            e = QLineEdit(body)
            form.addRow(label, e)
            self._entries[key] = e

        # Read-only derived G
        self._g_label = QLabel("—", body)
        form.addRow("G (kN/m²) [derived]", self._g_label)

        # Inline validation message for ν
        self._nu_status = QLabel("", body)
        self._nu_status.setStyleSheet("color: #b00;")
        form.addRow("", self._nu_status)

        if self._existing:
            m = self._existing
            self._entries["id"].setText(str(m.id))
            self._entries["id"].setReadOnly(True)
            self._entries["name"].setText(m.name)
            self._entries["E"].setText(repr(m.E))
            self._entries["alpha"].setText(repr(m.alpha))
            self._entries["density"].setText(repr(m.density))
            self._entries["nu"].setText(repr(m.nu))
            if m.template:
                idx = self._template_combo.findData(m.template)
                if idx >= 0:
                    self._template_combo.setCurrentIndex(idx)
        else:
            self._entries["id"].setText(str(self._default_id))
            self._entries["alpha"].setText("0.0")
            self._entries["density"].setText("0.0")
            self._entries["nu"].setText("0.0")

        # Wire updates
        self._template_combo.currentIndexChanged.connect(self._on_template_changed)
        self._entries["E"].textChanged.connect(self._refresh_derived)
        self._entries["nu"].textChanged.connect(self._refresh_derived)
        self._refresh_derived()

    def _on_template_changed(self) -> None:
        key = self._template_combo.currentData()
        if not key:
            return
        preset = MATERIAL_TEMPLATES[key]
        self._entries["name"].setText(str(preset.get("name", key)))
        self._entries["E"].setText(repr(float(preset["E"])))
        self._entries["alpha"].setText(repr(float(preset["alpha"])))
        self._entries["density"].setText(repr(float(preset["density"])))
        self._entries["nu"].setText(repr(float(preset["nu"])))

    def _refresh_derived(self) -> None:
        # Compute G and gate the OK button. Rules:
        #   - E must parse and be > 0 to enable OK.
        #   - ν empty is fine (defaults to 0 on _accept); a non-empty ν
        #     must parse AND lie in [0, 0.5).
        e_text = self._entries["E"].text().strip()
        nu_text = self._entries["nu"].text().strip()

        E: float | None
        try:
            E = float(e_text) if e_text else None
        except ValueError:
            E = None
        e_ok = E is not None and E > 0.0

        nu: float | None
        if not nu_text:
            nu = 0.0
            nu_ok = True
        else:
            try:
                nu = float(nu_text)
            except ValueError:
                nu = None
                nu_ok = False
            else:
                nu_ok = 0.0 <= nu < 0.5

        if not nu_ok and nu_text:
            self._nu_status.setText("ν must be in [0, 0.5).")
        else:
            self._nu_status.setText("")

        ok = self._ok_button()
        if ok is not None:
            ok.setEnabled(e_ok and nu_ok)

        if e_ok and nu_ok and nu is not None:
            self._g_label.setText(f"{E / (2.0 * (1.0 + nu)):g}")
        else:
            self._g_label.setText("—")

    def _ok_button(self):
        # findChildren walks the widget tree; cache the result since
        # _refresh_derived runs on every keystroke in E or ν.
        if not hasattr(self, "_cached_ok_button"):
            cached = None
            for child in self.findChildren(QDialogButtonBox):
                cached = child.button(QDialogButtonBox.StandardButton.Ok)
                break
            self._cached_ok_button = cached
        return self._cached_ok_button

    def _accept(self) -> Material:
        mid = parse_int(self._entries["id"].text(), "Material ID")
        name = self._entries["name"].text().strip()
        E = parse_float(self._entries["E"].text(), "E")
        alpha = parse_float(self._entries["alpha"].text(), "α", allow_blank=True) or 0.0
        density = parse_float(self._entries["density"].text(), "density",
                              allow_blank=True) or 0.0
        nu = parse_float(self._entries["nu"].text(), "ν", allow_blank=True) or 0.0
        if E <= 0:
            raise ValueError("E must be > 0.")
        if density < 0:
            raise ValueError("density cannot be negative.")
        if not (0.0 <= nu < 0.5):
            raise ValueError("Poisson's ratio must be in [0, 0.5).")
        template = self._template_combo.currentData() or ""
        return Material(id=mid, name=name, E=E, alpha=alpha,
                        density=density, nu=nu, template=template)


class SectionDialog(_ModalDialog):
    """Edit a single Section with a shape wizard.

    The "Shape" combo switches a stacked widget between dimension
    inputs:

      - manual: A, I, depth, width — user types raw values.
      - rectangle: b, h drive A/I/depth/width (read-only previews).
      - square: h drives the rest.
      - i_section: h/b/tf/tw drive A/I/J/depth/width.

    A "Manual override" button on the shape pages copies the computed
    values to the manual page and switches the combo, so the user can
    deviate from the calculator when needed.
    """

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

        # Shape combo + stacked pages
        self._shape_combo = QComboBox(body)
        for s in SECTION_SHAPES:
            self._shape_combo.addItem(s, s)
        form.addRow("Shape", self._shape_combo)

        self._stack = QStackedWidget(body)
        # Explicit shape_type → stack-page-index map so the dialog
        # doesn't silently break if SECTION_SHAPES is reordered or a
        # new shape is appended in a different position.
        self._page_index: dict[str, int] = {}
        self._page_index["manual"] = self._stack.addWidget(
            self._build_manual_page()
        )
        self._page_index["rectangle"] = self._stack.addWidget(
            self._build_rect_page()
        )
        self._page_index["square"] = self._stack.addWidget(
            self._build_square_page()
        )
        self._page_index["i_section"] = self._stack.addWidget(
            self._build_i_page()
        )
        form.addRow(self._stack)

        # Storage-only J note
        note = QLabel(
            "J (torsion constant) is storage-only — the current 2D "
            "solver does not use it. Rectangle and square shapes leave "
            "J = 0; I-section provides an approximate thin-walled value "
            "for future 3D / reporting use.",
            body,
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555; font-size: 10px;")
        form.addRow(note)

        # Status label for live calculator validation errors
        self._status = QLabel("", body)
        self._status.setStyleSheet("color: #b00;")
        form.addRow(self._status)

        # Live cross-section preview. A small square matplotlib figure
        # rendered next to the existing per-page text read-out. The
        # widget is built once and re-drawn from _refresh_preview() —
        # which is already on the textChanged path for every
        # dimension field and on the shape combo's index-changed
        # signal — so the preview follows whatever the user types.
        self._preview_fig = Figure(figsize=(2.6, 2.6), dpi=96)
        self._preview_fig.patch.set_facecolor("white")
        self._preview_ax = self._preview_fig.add_subplot(111)
        self._preview_canvas = FigureCanvasQTAgg(self._preview_fig)
        self._preview_canvas.setMinimumSize(220, 220)
        form.addRow("Preview", self._preview_canvas)

        # Initialise from existing section if any
        if self._existing:
            s = self._existing
            self._id_entry.setText(str(s.id))
            self._id_entry.setReadOnly(True)
            self._name_entry.setText(s.name)
            idx = self._mat_combo.findData(s.material_id)
            if idx >= 0:
                self._mat_combo.setCurrentIndex(idx)
            shape_key = s.shape_type or "manual"
            shape_idx = self._shape_combo.findData(shape_key)
            if shape_idx >= 0:
                self._shape_combo.setCurrentIndex(shape_idx)
            self._stack.setCurrentIndex(self._page_index.get(shape_key, 0))
            # Populate the relevant page
            self._a_entry.setText(repr(s.A))
            self._i_entry.setText(repr(s.I))
            self._d_entry.setText(repr(s.depth))
            self._w_entry.setText(repr(s.width))
            self._rect_b.setText(repr(s.b or 0.0))
            self._rect_h.setText(repr(s.h or 0.0))
            self._sq_h.setText(repr(s.h or 0.0))
            self._i_h.setText(repr(s.h or 0.0))
            self._i_b.setText(repr(s.b or 0.0))
            self._i_tf.setText(repr(s.tf or 0.0))
            self._i_tw.setText(repr(s.tw or 0.0))
        else:
            self._id_entry.setText(str(self._default_id))
            self._d_entry.setText("0.0")
            self._w_entry.setText("0.0")

        self._shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        self._refresh_preview()

    # ── page builders ──

    def _build_manual_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self._a_entry = QLineEdit(page)
        self._i_entry = QLineEdit(page)
        self._d_entry = QLineEdit(page)
        self._w_entry = QLineEdit(page)
        form.addRow("A (m²)", self._a_entry)
        form.addRow("I (m⁴)", self._i_entry)
        form.addRow("depth (m)", self._d_entry)
        form.addRow("width (m)", self._w_entry)
        return page

    def _build_rect_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self._rect_b = QLineEdit(page)
        self._rect_h = QLineEdit(page)
        form.addRow("b — width (m)", self._rect_b)
        form.addRow("h — depth (m)", self._rect_h)
        self._rect_preview = QLabel("—", page)
        form.addRow("Derived", self._rect_preview)
        btn = QPushButton("↻ Manual override", page)
        btn.clicked.connect(self._copy_to_manual)
        form.addRow(btn)
        self._rect_b.textChanged.connect(self._refresh_preview)
        self._rect_h.textChanged.connect(self._refresh_preview)
        return page

    def _build_square_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self._sq_h = QLineEdit(page)
        form.addRow("h — side (m)", self._sq_h)
        self._sq_preview = QLabel("—", page)
        form.addRow("Derived", self._sq_preview)
        btn = QPushButton("↻ Manual override", page)
        btn.clicked.connect(self._copy_to_manual)
        form.addRow(btn)
        self._sq_h.textChanged.connect(self._refresh_preview)
        return page

    def _build_i_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self._i_h = QLineEdit(page)
        self._i_b = QLineEdit(page)
        self._i_tf = QLineEdit(page)
        self._i_tw = QLineEdit(page)
        form.addRow("h — depth (m)", self._i_h)
        form.addRow("b — flange width (m)", self._i_b)
        form.addRow("tf — flange thickness (m)", self._i_tf)
        form.addRow("tw — web thickness (m)", self._i_tw)
        self._i_preview = QLabel("—", page)
        form.addRow("Derived", self._i_preview)
        btn = QPushButton("↻ Manual override", page)
        btn.clicked.connect(self._copy_to_manual)
        form.addRow(btn)
        for w in (self._i_h, self._i_b, self._i_tf, self._i_tw):
            w.textChanged.connect(self._refresh_preview)
        return page

    # ── interaction ──

    def _on_shape_changed(self) -> None:
        shape = self._current_shape()
        self._stack.setCurrentIndex(self._page_index.get(shape, 0))
        self._refresh_preview()

    def _current_shape(self) -> str:
        return self._shape_combo.currentData() or "manual"

    def _ok_button(self):
        # Cache once found. The button box is added AFTER _build_body
        # runs, so the first calls (during construction) return None;
        # cache only positive lookups so we re-search until it exists.
        cached = getattr(self, "_cached_ok_button", None)
        if cached is not None:
            return cached
        for child in self.findChildren(QDialogButtonBox):
            btn = child.button(QDialogButtonBox.StandardButton.Ok)
            if btn is not None:
                self._cached_ok_button = btn
                return btn
        return None

    def _compute_derived(self) -> dict[str, float] | None:
        """Try to compute A/I/depth/width(/J) for the selected shape.

        Returns the dict on success, None when input is incomplete or
        invalid. Side-effect: updates self._status with the error
        message (cleared on success).
        """
        shape = self._current_shape()
        try:
            if shape == "rectangle":
                b = float(self._rect_b.text() or "0")
                h = float(self._rect_h.text() or "0")
                return properties_for_shape("rectangle", b=b, h=h)
            if shape == "square":
                h = float(self._sq_h.text() or "0")
                return properties_for_shape("square", h=h)
            if shape == "i_section":
                h = float(self._i_h.text() or "0")
                b = float(self._i_b.text() or "0")
                tf = float(self._i_tf.text() or "0")
                tw = float(self._i_tw.text() or "0")
                return properties_for_shape(
                    "i_section", h=h, b=b, tf=tf, tw=tw,
                )
        except ValueError as e:
            self._status.setText(str(e))
            return None
        return None

    def _refresh_preview(self) -> None:
        shape = self._current_shape()
        ok = self._ok_button()
        if shape == "manual":
            self._status.setText("")
            if ok is not None:
                ok.setEnabled(True)
            self._draw_section_preview(None)
            return
        derived = self._compute_derived()
        if derived is None:
            if ok is not None:
                ok.setEnabled(False)
            self._draw_section_preview(None)
            return
        self._status.setText("")
        if ok is not None:
            ok.setEnabled(True)
        text = (f"A = {derived['A']:g}  I = {derived['I']:g}\n"
                f"depth = {derived['depth']:g}  width = {derived['width']:g}"
                f"  J = {derived.get('J', 0.0):g}")
        if shape == "rectangle":
            self._rect_preview.setText(text)
        elif shape == "square":
            self._sq_preview.setText(text)
        elif shape == "i_section":
            self._i_preview.setText(text)
        self._draw_section_preview(derived)

    def _draw_section_preview(self, derived: dict[str, float] | None) -> None:
        """Re-render the small cross-section thumbnail.

        ``derived`` is the dict returned by :meth:`_compute_derived` or
        ``None`` when the input is incomplete / invalid (or when the
        manual shape is selected, which has no geometric dimensions).
        The graphical preview is additive — the per-page text read-out
        (``_rect_preview`` / ``_sq_preview`` / ``_i_preview``) and the
        red ``_status`` label keep doing their existing jobs.
        """
        ax = self._preview_ax
        ax.clear()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        shape = self._current_shape()

        if shape == "manual":
            # Manual sections have no geometric inputs in this dialog
            # — the user types A / I / depth / width directly. When
            # the dialog opens for a new section the canvas would
            # otherwise be empty, which gives the user no idea what
            # the preview is for. Render a small example rectangle
            # with b / h dimension labels as a visual hint. For an
            # *existing* manual section we keep the previous
            # "no shape preview" message — the user is editing real
            # numbers and doesn't want a fake outline interfering.
            if self._existing is None:
                self._draw_example_outline(ax, "rectangle")
            else:
                ax.text(
                    0.5, 0.5,
                    "manual section\n(no shape preview)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#555",
                )
            self._preview_canvas.draw_idle()
            return

        if derived is None:
            # Input is incomplete or invalid. For a brand-new section
            # render the canonical example of the *currently selected*
            # shape so the canvas always tells the user what they're
            # about to build (an I-section after switching to "i" is
            # far more useful than the previous "invalid dimensions"
            # placeholder). For an existing section we still show
            # "invalid dimensions", because the user's own valid
            # input was just broken and we shouldn't paper over it.
            if self._existing is None:
                self._draw_example_outline(ax, shape)
            else:
                ax.text(
                    0.5, 0.5,
                    "invalid dimensions",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#b00",
                )
            self._preview_canvas.draw_idle()
            return

        try:
            section = self._section_from_inputs(shape, derived)
            pts = section_outline(section)
        except (ValueError, KeyError):
            ax.text(
                0.5, 0.5,
                "invalid dimensions",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#b00",
            )
            self._preview_canvas.draw_idle()
            return

        # Outline tuples are (y, z) with depth on local y, width on
        # local z. Plot width on the horizontal screen axis and depth
        # on the vertical screen axis so the thumbnail matches the
        # user's mental "wide horizontally, deep vertically" picture.
        zs = [p[1] for p in pts]
        ys = [p[0] for p in pts]
        ax.fill(zs, ys, facecolor="#cfe3f6", edgecolor="#1f3a5f",
                linewidth=1.2, alpha=0.95)

        # Dimension labels — small, positioned just outside the outline.
        # b = width along z, h = depth along y; tf / tw only for I.
        b = derived.get("width", 0.0)
        h = derived.get("depth", 0.0)
        if b > 0 and h > 0:
            ax.annotate(
                f"b = {b:g}",
                xy=(0.0, -h / 2.0), xytext=(0.0, -h / 2.0 - 0.18 * h),
                ha="center", va="top", fontsize=8, color="#333",
            )
            ax.annotate(
                f"h = {h:g}",
                xy=(b / 2.0, 0.0), xytext=(b / 2.0 + 0.18 * b, 0.0),
                ha="left", va="center", fontsize=8, color="#333",
            )
        if shape == "i_section":
            # section.tf / section.tw are already populated from the
            # same input fields by _section_from_inputs (which only
            # runs when the text parses cleanly), so re-parsing the
            # widgets here would only repeat work and risk drifting
            # from the geometry we just drew.
            tf, tw = section.tf, section.tw
            if tf > 0:
                ax.annotate(
                    f"tf = {tf:g}",
                    xy=(-b / 2.0, h / 2.0 - tf / 2.0),
                    xytext=(-b / 2.0 - 0.22 * b, h / 2.0 - tf / 2.0),
                    ha="right", va="center", fontsize=8, color="#333",
                )
            if tw > 0:
                ax.annotate(
                    f"tw = {tw:g}",
                    xy=(tw / 2.0, 0.0),
                    xytext=(tw / 2.0 + 0.18 * b, -0.25 * h),
                    ha="left", va="center", fontsize=8, color="#333",
                )

        # A little breathing room around the outline so the dimension
        # labels don't run off the edge of the figure.
        ax.relim()
        ax.autoscale_view()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        pad_x = (x1 - x0) * 0.30 + 1e-6
        pad_y = (y1 - y0) * 0.25 + 1e-6
        ax.set_xlim(x0 - pad_x, x1 + pad_x)
        ax.set_ylim(y0 - pad_y, y1 + pad_y)

        self._preview_canvas.draw_idle()

    # Canonical example dimensions per shape. Used by the preview
    # when no valid user input exists yet — picked so each outline
    # is visually readable at the dialog's 220×220 px thumbnail size.
    _EXAMPLE_DIMS: dict[str, dict[str, float]] = {
        "rectangle": dict(b=0.30, h=0.50),
        "square":    dict(h=0.40),
        "i_section": dict(h=0.20, b=0.10, tf=0.0085, tw=0.0056),
    }

    def _draw_example_outline(self, ax, shape: str) -> None:
        """Render the canonical example outline for ``shape`` on the
        preview canvas so a freshly-opened Add Section dialog never
        shows a blank canvas — even before the user types any
        dimensions. The geometry comes from :func:`section_outline`
        (same path the live preview uses), drawn dashed in a muted
        palette and tagged "(example)" so it can't be mistaken for
        the user's own input. Manual + new defaults to "rectangle";
        the other shapes use their dedicated example dims.
        """
        shape_key = shape if shape in self._EXAMPLE_DIMS else "rectangle"
        dims = self._EXAMPLE_DIMS[shape_key]
        if shape_key == "i_section":
            section = Section(
                id=0, shape_type="i_section",
                b=dims["b"], h=dims["h"],
                tf=dims["tf"], tw=dims["tw"],
            )
        elif shape_key == "square":
            h = dims["h"]
            section = Section(
                id=0, shape_type="square", b=h, h=h,
            )
        else:  # rectangle
            section = Section(
                id=0, shape_type="rectangle",
                b=dims["b"], h=dims["h"],
            )
        pts = section_outline(section)
        zs = [p[1] for p in pts]
        ys = [p[0] for p in pts]
        ax.fill(zs, ys, facecolor="#e8eef5", edgecolor="#9aa9bf",
                linewidth=1.2, alpha=0.9, linestyle="--")

        # Dimension annotations — same set the real preview shows,
        # so the example reads as "this is what your input will draw".
        b_ann = dims.get("b", dims.get("h", 0.0))
        h_ann = dims["h"]
        ax.annotate(
            f"b = {b_ann:g}",
            xy=(0.0, -h_ann / 2.0),
            xytext=(0.0, -h_ann / 2.0 - 0.18 * h_ann),
            ha="center", va="top", fontsize=8, color="#666",
        )
        ax.annotate(
            f"h = {h_ann:g}",
            xy=(b_ann / 2.0, 0.0),
            xytext=(b_ann / 2.0 + 0.18 * b_ann, 0.0),
            ha="left", va="center", fontsize=8, color="#666",
        )
        if shape_key == "i_section":
            ax.annotate(
                f"tf = {dims['tf']:g}",
                xy=(-b_ann / 2.0, h_ann / 2.0 - dims["tf"] / 2.0),
                xytext=(-b_ann / 2.0 - 0.22 * b_ann,
                         h_ann / 2.0 - dims["tf"] / 2.0),
                ha="right", va="center", fontsize=8, color="#666",
            )
            ax.annotate(
                f"tw = {dims['tw']:g}",
                xy=(dims["tw"] / 2.0, 0.0),
                xytext=(dims["tw"] / 2.0 + 0.18 * b_ann, -0.25 * h_ann),
                ha="left", va="center", fontsize=8, color="#666",
            )

        ax.text(
            0.5, 0.97,
            f"example {shape_key} (type dimensions to override)",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=8, color="#888", style="italic",
        )
        ax.relim()
        ax.autoscale_view()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        pad_x = (x1 - x0) * 0.30 + 1e-6
        pad_y = (y1 - y0) * 0.25 + 1e-6
        ax.set_xlim(x0 - pad_x, x1 + pad_x)
        ax.set_ylim(y0 - pad_y, y1 + pad_y)

    def _section_from_inputs(self, shape: str,
                              derived: dict[str, float]) -> Section:
        """Build a transient Section from the current dialog inputs so
        :func:`section_outline` can render it. ID / material_id are
        placeholders — this object is only used for outline geometry
        and never reaches the model.

        The shape calculators copy their ``b`` / ``h`` inputs through
        to the ``width`` / ``depth`` keys of the returned dict
        (see ``profiles.rectangle_properties`` / ``i_section_properties``),
        so we read them out of ``derived`` here instead of re-parsing
        the text widgets. ``tf`` / ``tw`` only exist on the I-section
        page and still come from the widget — that's the one input
        the calculator does not echo back.
        """
        b = derived["width"]
        h = derived["depth"]
        if shape == "rectangle":
            return Section(
                id=0, shape_type="rectangle",
                b=b, h=h,
                A=derived["A"], I=derived["I"],
                depth=h, width=b,
            )
        if shape == "square":
            return Section(
                id=0, shape_type="square",
                b=h, h=h,
                A=derived["A"], I=derived["I"],
                depth=h, width=h,
            )
        if shape == "i_section":
            return Section(
                id=0, shape_type="i_section",
                b=b, h=h,
                tf=float(self._i_tf.text() or "0"),
                tw=float(self._i_tw.text() or "0"),
                A=derived["A"], I=derived["I"],
                depth=h, width=b,
                J=derived.get("J", 0.0),
            )
        raise ValueError(f"unsupported shape {shape!r}")

    def _copy_to_manual(self) -> None:
        derived = self._compute_derived()
        if derived is None:
            return
        self._a_entry.setText(repr(derived["A"]))
        self._i_entry.setText(repr(derived["I"]))
        self._d_entry.setText(repr(derived["depth"]))
        self._w_entry.setText(repr(derived["width"]))
        idx = self._shape_combo.findData("manual")
        if idx >= 0:
            self._shape_combo.setCurrentIndex(idx)

    # ── accept ──

    def _accept(self) -> Section:
        sid = parse_int(self._id_entry.text(), "Section ID")
        name = self._name_entry.text().strip()
        material_id = self._mat_combo.currentData()
        if material_id not in self._model.materials:
            raise ValueError(f"Material {material_id} does not exist.")
        shape = self._current_shape()

        if shape == "manual":
            A = parse_float(self._a_entry.text(), "A")
            I = parse_float(self._i_entry.text(), "I")
            depth = parse_float(self._d_entry.text(), "depth",
                                 allow_blank=True) or 0.0
            width = parse_float(self._w_entry.text(), "width",
                                 allow_blank=True) or 0.0
            if A <= 0:
                raise ValueError("A must be > 0.")
            if I < 0:
                raise ValueError("I cannot be negative.")
            return Section(
                id=sid, name=name, material_id=int(material_id),
                A=A, I=I, depth=depth, width=width,
                J=0.0, shape_type="manual",
            )

        # Shape-driven path
        derived = self._compute_derived()
        if derived is None:
            raise ValueError(
                "Cross-section dimensions are incomplete or invalid; "
                "see the preview area."
            )
        b = h = tf = tw = 0.0
        if shape == "rectangle":
            b = parse_float(self._rect_b.text(), "b", allow_blank=True) or 0.0
            h = parse_float(self._rect_h.text(), "h", allow_blank=True) or 0.0
        elif shape == "square":
            h = parse_float(self._sq_h.text(), "h", allow_blank=True) or 0.0
            b = h
        elif shape == "i_section":
            h = parse_float(self._i_h.text(), "h", allow_blank=True) or 0.0
            b = parse_float(self._i_b.text(), "b", allow_blank=True) or 0.0
            tf = parse_float(self._i_tf.text(), "tf", allow_blank=True) or 0.0
            tw = parse_float(self._i_tw.text(), "tw", allow_blank=True) or 0.0
        return Section(
            id=sid, name=name, material_id=int(material_id),
            A=derived["A"], I=derived["I"],
            depth=derived["depth"], width=derived["width"],
            J=derived.get("J", 0.0),
            shape_type=shape,
            b=b, h=h, tf=tf, tw=tw,
        )


# ── element properties ──


class ElementDialog(_ModalDialog):
    def __init__(self, parent, *, model: StructuralModel,
                 existing_kind: str | None = None,
                 existing_section_id: int | None = None,
                 existing_release_i: bool = False,
                 existing_release_j: bool = False,
                 existing_material_override_id: int | None = None,
                 remember_default: bool = True):
        self._model = model
        if not model.sections:
            raise ValueError("No sections defined — add a section first.")
        self._existing_kind = existing_kind
        self._existing_sec = existing_section_id
        self._existing_ri = existing_release_i
        self._existing_rj = existing_release_j
        self._existing_mat_override = existing_material_override_id
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

        # Material override combo. First entry "Use section default"
        # carries data=None (the common case); remaining entries
        # enumerate every Material in the model. Changing this is
        # independent of the Section combo — it just overrides the
        # section's default material for this single element.
        self._mat_combo = QComboBox(body)
        self._mat_combo.addItem("Use section default", None)
        for mid in sorted(self._model.materials):
            m = self._model.materials[mid]
            label = f"{m.name or f'material {mid}'}  (id {mid})"
            self._mat_combo.addItem(label, mid)
        if self._existing_mat_override is not None:
            idx = self._mat_combo.findData(self._existing_mat_override)
            if idx >= 0:
                self._mat_combo.setCurrentIndex(idx)
        form.addRow("Material:", self._mat_combo)

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
        mat_override = self._mat_combo.currentData()
        return {
            "kind": kind,
            "section_id": int(section_id),
            "release_i": self._cb_ri.isChecked() if kind == "frame" else False,
            "release_j": self._cb_rj.isChecked() if kind == "frame" else False,
            "material_override_id": (int(mat_override)
                                      if mat_override is not None else None),
            "remember": self._cb_remember.isChecked(),
        }


# ── batch assign (v0.13.0) ──


class BatchAssignDialog(_ModalDialog):
    """Batch-assign section / material override to many elements.

    Both dropdowns default to "(leave unchanged)" so a mixed selection
    isn't silently overwritten with a single default. The user only
    changes the field(s) they explicitly want batched.

    Returns ``{"section_id": int | None, "material_override_id": int | None}``
    where ``None`` means "do not touch this field". The
    ``material_override_id`` value :data:`-1` (mapped from the dialog's
    explicit "Use section default (clear override)" item) is the
    sentinel ``CLEAR_MATERIAL_OVERRIDE`` that the command translates to
    "set override = None" — required because plain ``None`` already
    means "leave alone."
    """

    def __init__(self, parent: QWidget | None, *,
                 model: StructuralModel,
                 element_count: int) -> None:
        self._model = model
        self._count = int(element_count)
        if not model.sections:
            raise ValueError("No sections defined — add a section first.")
        super().__init__(parent, "Batch assign element properties")

    def _build_body(self, body: QWidget) -> None:
        form = QFormLayout(body)
        heading = QLabel(
            f"Applying to {self._count} element"
            f"{'s' if self._count != 1 else ''}.",
            body,
        )
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        form.addRow(heading)

        # Section dropdown — first entry is "leave unchanged".
        self._sec_combo = QComboBox(body)
        self._sec_combo.addItem("(leave unchanged)", None)
        for sid in sorted(self._model.sections):
            s = self._model.sections[sid]
            mat = self._model.materials.get(s.material_id)
            mat_name = (mat.name if mat and mat.name
                        else f"material {s.material_id}")
            sec_name = s.name if s.name else f"section {sid}"
            self._sec_combo.addItem(f"{sec_name} / {mat_name}", sid)
        form.addRow("Section / material:", self._sec_combo)

        # Material override dropdown — three classes of item:
        # "(leave unchanged)" → None
        # "Use section default (clear override)" → -1 sentinel
        # each material id → that material as override
        self._mat_combo = QComboBox(body)
        self._mat_combo.addItem("(leave unchanged)", None)
        self._mat_combo.addItem(
            "Use section default (clear override)", -1,
        )
        for mid in sorted(self._model.materials):
            m = self._model.materials[mid]
            self._mat_combo.addItem(
                f"{m.name or f'material {mid}'}  (id {mid})", mid,
            )
        form.addRow("Material override:", self._mat_combo)

        hint = QLabel(
            "Fields left unchanged keep their per-element value.",
            body,
        )
        hint.setStyleSheet("color: #555; font-style: italic;")
        form.addRow(hint)

    def _accept(self) -> dict:
        section_id = self._sec_combo.currentData()
        mat_override = self._mat_combo.currentData()
        if section_id is None and mat_override is None:
            raise ValueError(
                "Pick a section or a material override before clicking OK "
                "(both are currently set to 'leave unchanged')."
            )
        return {
            "section_id": (int(section_id) if section_id is not None
                           else None),
            "material_override_id": (int(mat_override)
                                     if mat_override is not None else None),
        }


# ── renumber elements ──


class RenumberElementsDialog(_ModalDialog):
    """Pick an ordering, preview the old → new id mapping, confirm.

    Three orderings:
      - "By current id" — compact to 1..N, preserving relative order.
      - "By geometry" — sort by midpoint (y descending, then x ascending),
        i.e. top-to-bottom, left-to-right. Helps SAP-style models.
      - "Selected first, then rest by current id" — selected elements
        get 1..k sorted by their current id; the rest follow by current
        id starting from k+1. Only meaningful when a selection exists.

    Returns a ``{old_id: new_id}`` dict suitable for
    :class:`RenumberElementsCmd`.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        model: StructuralModel,
        selected_ids: frozenset[int] | set[int] = frozenset(),
    ) -> None:
        self._model = model
        self._selected_ids = frozenset(selected_ids)
        if not model.elements:
            raise ValueError("No elements to renumber.")
        super().__init__(parent, "Renumber elements")
        self.resize(420, 360)

    def _build_body(self, body: QWidget) -> None:
        layout = QVBoxLayout(body)
        heading = QLabel(
            f"Renumber {len(self._model.elements)} element"
            f"{'s' if len(self._model.elements) != 1 else ''}.",
            body,
        )
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)

        from PyQt6.QtWidgets import QButtonGroup, QRadioButton, QTableWidget
        from PyQt6.QtWidgets import QTableWidgetItem

        self._rb_current = QRadioButton(
            "By current ID (compact to 1..N)", body,
        )
        self._rb_geometry = QRadioButton(
            "By geometry (top-to-bottom, left-to-right)", body,
        )
        self._rb_selection = QRadioButton(
            "Selected first, then remaining by current ID", body,
        )
        self._rb_selection.setEnabled(bool(self._selected_ids))
        if not self._selected_ids:
            self._rb_selection.setToolTip(
                "No active element selection. Close, select some "
                "elements on the canvas, then reopen this dialog."
            )
        self._rb_current.setChecked(True)
        group = QButtonGroup(body)
        for rb in (self._rb_current, self._rb_geometry, self._rb_selection):
            group.addButton(rb)
            layout.addWidget(rb)
            rb.toggled.connect(self._refresh_preview)

        self._table = QTableWidget(len(self._model.elements), 3, body)
        self._table.setHorizontalHeaderLabels(["Old ID", "New ID", "Note"])
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)
        self._QTableWidgetItem = QTableWidgetItem
        self._refresh_preview()

    def _strategy(self) -> str:
        if self._rb_geometry.isChecked():
            return "geometry"
        if self._rb_selection.isChecked():
            return "selection"
        return "current"

    def _compute_mapping(self) -> dict[int, int]:
        """Compute old→new mapping for the active strategy."""
        elements = self._model.elements
        nodes = self._model.nodes
        strategy = self._strategy()
        if strategy == "geometry":
            def _mid(e):
                ni = nodes.get(e.node_i)
                nj = nodes.get(e.node_j)
                if ni is None or nj is None:
                    return (0.0, 0.0)
                return ((ni.x + nj.x) / 2.0, (ni.y + nj.y) / 2.0)
            ordered = sorted(
                elements,
                key=lambda e: (-(_mid(e)[1]), _mid(e)[0], e.id),
            )
        elif strategy == "selection":
            in_sel = sorted(
                (e for e in elements if e.id in self._selected_ids),
                key=lambda e: e.id,
            )
            out_sel = sorted(
                (e for e in elements if e.id not in self._selected_ids),
                key=lambda e: e.id,
            )
            ordered = in_sel + out_sel
        else:  # "current"
            ordered = sorted(elements, key=lambda e: e.id)
        return {e.id: new_id for new_id, e in enumerate(ordered, start=1)}

    def _refresh_preview(self) -> None:
        from PyQt6.QtCore import Qt
        mapping = self._compute_mapping()
        rows = sorted(mapping.items(), key=lambda kv: kv[1])
        for r, (old, new) in enumerate(rows):
            note = ""
            if old == new:
                note = "(unchanged)"
            elif old in self._selected_ids and self._strategy() == "selection":
                note = "(selected)"
            for c, txt in enumerate((str(old), str(new), note)):
                item = self._QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()

    def _accept(self) -> dict[int, int]:
        return self._compute_mapping()


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


# ── load_case combo (shared by Nodal + Member dialogs) ──

#: Built-in suggestions for the load-case combo. The combo is editable
#: so users can type custom case names; these are just convenient
#: starting points.
_LOAD_CASE_SUGGESTIONS = ("DEFAULT", "DEAD", "LIVE", "WIND", "THERMAL")


def _make_load_case_combo(
    parent: QWidget, current: str = "DEFAULT",
    *,
    model: "StructuralModel | None" = None,
) -> QComboBox:
    """Return a populated, editable load-case combo with a sensible
    initial value. Callers normalize the final value via
    :func:`_normalize_load_case` on accept.

    PR-A: when ``model`` is provided, the existing case names from
    ``model.load_cases`` are appended after the built-in suggestions
    (deduplicated, sorted) so a user assigning a load to a custom case
    can pick it from the dropdown instead of retyping. Typing a new
    case name still works — the dialog auto-creates it on accept via
    :class:`AddLoadCaseCmd`."""
    combo = QComboBox(parent)
    combo.setEditable(True)
    # Built-in suggestions first (DEFAULT, DEAD, LIVE, WIND, THERMAL),
    # then any model-defined case names that aren't already listed.
    items = list(_LOAD_CASE_SUGGESTIONS)
    if model is not None:
        for name in sorted(model.load_cases.keys()):
            if name not in items:
                items.append(name)
    combo.addItems(items)
    current_norm = _normalize_load_case(current)
    if current_norm in items:
        combo.setCurrentText(current_norm)
    else:
        # Custom case name that isn't in the built-in suggestions.
        combo.setEditText(current_norm)
    combo.setToolTip(
        "User-facing tag for this load. Pick an existing case or type "
        "a new name — typing a new name auto-creates the case on OK. "
        "Blank / whitespace falls back to DEFAULT."
    )
    return combo


def _normalize_load_case(raw: str | None) -> str:
    """Trim + uppercase a user-entered case name. Empty → 'DEFAULT'.

    Whitespace or ``#`` inside the case name is rejected: whitespace
    would break the writer's single-token storage, and ``#`` starts a
    comment in the input-file format and would silently truncate the
    saved row on reload.
    """
    if raw is None:
        return "DEFAULT"
    stripped = raw.strip()
    if not stripped:
        return "DEFAULT"
    if any(ch.isspace() or ch == "#" for ch in stripped):
        raise ValueError(
            f"Load case {stripped!r} contains invalid characters "
            "(whitespace or '#'); case names must be a single token "
            "and cannot contain '#'. Use underscores or hyphens."
        )
    return stripped.upper()


# ── nodal load ──


class NodalLoadDialog(_ModalDialog):
    def __init__(
        self, parent, *, existing: NodalLoad | None, node_id: int,
        model: "StructuralModel | None" = None,
    ):
        self._existing = existing
        self._node_id = node_id
        # PR-A: when ``model`` is provided, the load-case combo lists
        # the model's case names in addition to the built-in
        # suggestions. None preserves the v0.17 behaviour for any
        # legacy caller / unit-test that constructs the dialog
        # without a model.
        self._model_for_cases = model
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
        existing_case = (
            getattr(self._existing, "load_case", "DEFAULT")
            if self._existing is not None else "DEFAULT"
        )
        self._case_combo = _make_load_case_combo(
            body, existing_case, model=self._model_for_cases,
        )
        form.addRow("Load case:", self._case_combo)

    def _accept(self) -> tuple[float, float, float, str]:
        fx = parse_float(self._entries["fx"].text(), "Fx", allow_blank=True) or 0.0
        fy = parse_float(self._entries["fy"].text(), "Fy", allow_blank=True) or 0.0
        mz = parse_float(self._entries["mz"].text(), "Mz", allow_blank=True) or 0.0
        load_case = _normalize_load_case(self._case_combo.currentText())
        return (fx, fy, mz, load_case)


# ── nodal-load manager (v0.20 — PR #30) ──


class NodalLoadManagerDialog(QDialog):
    """List + Add / Edit / Delete the nodal loads at a single node.

    Unlike most editors in this module the manager does **not** collect
    a batch of intents on accept; each Add / Edit / Delete is dispatched
    immediately through ``host.execute()`` so every row mutation
    becomes an individually undoable command on the host's stack — that
    matches the user-facing promise (PR #30) that each row action is
    independently undoable / redoable.

    The dialog re-reads ``model.nodal_loads`` after every action so the
    table is always in sync with the live model. Row → global index
    mapping is captured at render time and stored on each
    :class:`QTableWidgetItem` so Edit / Delete know which entry in
    ``model.nodal_loads`` to address.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        host: Any,
        model: StructuralModel,
        node_id: int,
    ) -> None:
        super().__init__(parent)
        if node_id not in model.nodes:
            raise ValueError(f"Node {node_id} does not exist.")
        self._host = host
        self._model = model
        self._node_id = node_id
        self.setWindowTitle(f"Nodal loads at node {node_id}")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Each row is one independent nodal load at node {node_id}. "
            "Add appends a row; Edit / Delete operate on the selected "
            "row. Every action is individually undoable.",
            self,
        ))

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(
            ["Case", "Fx (kN)", "Fy (kN)", "Mz (kN·m)"]
        )
        install_table_copy(self._table, include_headers=True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add Load…", self)
        self._add_btn.clicked.connect(self._on_add)
        btn_row.addWidget(self._add_btn)
        self._edit_btn = QPushButton("Edit Selected Load…", self)
        self._edit_btn.clicked.connect(self._on_edit)
        btn_row.addWidget(self._edit_btn)
        self._delete_btn = QPushButton("Delete Selected Load", self)
        self._delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self._delete_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close, parent=self,
        )
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)

        self._table.itemSelectionChanged.connect(self._refresh_buttons)
        self._rebuild_table()

    # ── table sync ──

    def _rows_for_node(self) -> list[tuple[int, NodalLoad]]:
        """Return (global_index, NodalLoad) pairs for this node only.

        The global index — position in ``model.nodal_loads`` — is what
        :class:`EditNodalLoadRowCmd` / :class:`DeleteNodalLoadRowCmd`
        need as their ``row_index`` argument.
        """
        return [
            (i, ld) for i, ld in enumerate(self._model.nodal_loads)
            if ld.node_id == self._node_id
        ]

    def _rebuild_table(self) -> None:
        rows = self._rows_for_node()
        self._table.setRowCount(len(rows))
        for visible_idx, (global_idx, ld) in enumerate(rows):
            case_item = QTableWidgetItem(
                getattr(ld, "load_case", "DEFAULT") or "DEFAULT"
            )
            # Stash the global index on the first cell so Edit / Delete
            # can read it back without re-deriving from the visible row.
            case_item.setData(Qt.ItemDataRole.UserRole, global_idx)
            self._table.setItem(visible_idx, 0, case_item)
            self._table.setItem(
                visible_idx, 1, QTableWidgetItem(f"{ld.fx:g}")
            )
            self._table.setItem(
                visible_idx, 2, QTableWidgetItem(f"{ld.fy:g}")
            )
            self._table.setItem(
                visible_idx, 3, QTableWidgetItem(f"{ld.mz:g}")
            )
        self._refresh_buttons()

    def _selected_global_index(self) -> int | None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self._table.item(rows[0].row(), 0)
        if item is None:
            return None
        return int(item.data(Qt.ItemDataRole.UserRole))

    def _refresh_buttons(self) -> None:
        has_selection = self._selected_global_index() is not None
        self._edit_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)

    # ── actions ──

    def _open_form(
        self, existing: NodalLoad | None,
    ) -> tuple[float, float, float, str] | None:
        d = NodalLoadDialog(
            self,
            existing=existing,
            node_id=self._node_id,
            model=self._model,
        )
        if d.exec() != QDialog.DialogCode.Accepted:
            return None
        return d.result_value

    def _on_add(self) -> None:
        from ..gui_common.commands import AddNodalLoadCmd
        values = self._open_form(existing=None)
        if values is None:
            return
        fx, fy, mz, load_case = values
        if fx == 0.0 and fy == 0.0 and mz == 0.0:
            QMessageBox.information(
                self, "Empty load",
                "Fx, Fy, Mz are all zero — nothing to add.",
            )
            return
        # PR-A: ensure unknown load-case names appear in model.load_cases
        # before pushing the load command, mirroring _edit_nodal_load.
        ensure = getattr(self._host, "_ensure_load_case_exists", None)
        if callable(ensure):
            ensure(load_case)
        self._host.execute(AddNodalLoadCmd(
            node_id=self._node_id, fx=fx, fy=fy, mz=mz,
            load_case=load_case,
        ))
        self._rebuild_table()

    def _on_edit(self) -> None:
        from ..gui_common.commands import EditNodalLoadRowCmd
        global_idx = self._selected_global_index()
        if global_idx is None:
            QMessageBox.information(
                self, "No selection",
                "Select a row in the table before clicking Edit.",
            )
            return
        existing = self._model.nodal_loads[global_idx]
        values = self._open_form(existing=existing)
        if values is None:
            return
        fx, fy, mz, load_case = values
        if fx == 0.0 and fy == 0.0 and mz == 0.0:
            QMessageBox.information(
                self, "Empty load",
                "Fx, Fy, Mz are all zero — use Delete to remove the row.",
            )
            return
        ensure = getattr(self._host, "_ensure_load_case_exists", None)
        if callable(ensure):
            ensure(load_case)
        self._host.execute(EditNodalLoadRowCmd(
            row_index=global_idx, fx=fx, fy=fy, mz=mz,
            load_case=load_case,
        ))
        self._rebuild_table()

    def _on_delete(self) -> None:
        from ..gui_common.commands import DeleteNodalLoadRowCmd
        global_idx = self._selected_global_index()
        if global_idx is None:
            QMessageBox.information(
                self, "No selection",
                "Select a row in the table before clicking Delete.",
            )
            return
        self._host.execute(DeleteNodalLoadRowCmd(row_index=global_idx))
        self._rebuild_table()


# ── member load ──


class MemberLoadDialog(_ModalDialog):
    """Add (default) or edit (when ``existing_load`` is provided) one
    member-load row.

    Edit mode is a strict UI prefill — radio state and field text are
    seeded from ``existing_load`` so the user can amend any field and
    accept; the dialog still returns a freshly-built load instance, and
    the caller decides whether to land that as an
    :class:`AddMemberLoadCmd` (Add) or
    :class:`UpdateMemberLoadCmd` (Edit).  ``existing_index`` is stored
    for the caller's convenience and does not affect dialog logic.
    """

    def __init__(
        self, parent, *,
        model: StructuralModel,
        elem_id: int,
        existing_load: object | None = None,
        existing_index: int | None = None,
    ):
        self._model = model
        self._elem_id = elem_id
        elem = next((e for e in model.elements if e.id == elem_id), None)
        if elem is None:
            raise ValueError(f"Element {elem_id} does not exist.")
        self._elem = elem
        self._existing_load = existing_load
        self._existing_index = existing_index
        title = (
            f"Edit member load on element {elem_id}"
            if existing_load is not None
            else f"Member load on element {elem_id}"
        )
        super().__init__(parent, title)

    def _build_body(self, body: QWidget) -> None:
        v = QVBoxLayout(body)
        self._is_truss = isinstance(self._elem, TrussElement2D)

        # ── Load category: Mechanical vs Thermal ───────────────────
        # One radio per top-level category. The body swap below hides
        # all controls that don't belong to the active category so
        # neither half can leak into the other (thermal never sees the
        # mechanical Direction radios; mechanical never sees the
        # Thermal-type radios).
        #
        # Truss elements have no bending DOFs and their solver path
        # explicitly rejects UDL / PointLoad (see
        # ``TrussElement2D.local_load_vector``). The pre-v0.17 dialog
        # filtered trusses down to thermal-only; v0.17 preserves that
        # contract by disabling the Mechanical radio with a tooltip
        # explaining why, rather than silently letting users build a
        # load the solver would later reject.
        v.addWidget(QLabel("Load category:", body))
        self._cat_group = QButtonGroup(body)
        self._rb_cat_mechanical = QRadioButton("Mechanical", body)
        if self._is_truss:
            self._rb_cat_mechanical.setEnabled(False)
            self._rb_cat_mechanical.setToolTip(
                "Truss elements support thermal loads only. "
                "Distributed and point loads require frame bending "
                "DOFs — use a frame element instead."
            )
        else:
            self._rb_cat_mechanical.setToolTip(
                "Force loads: distributed (UDL) or concentrated "
                "(point). Choose a Type below and then a Direction "
                "(Local / Global / Gravity)."
            )
        self._rb_cat_thermal = QRadioButton("Thermal", body)
        self._rb_cat_thermal.setToolTip(
            "Temperature change. Uniform ΔT produces axial thermal "
            "strain. A top/bottom gradient produces curvature "
            "(bending if restrained) and requires a frame element."
        )
        self._cat_group.addButton(self._rb_cat_mechanical)
        self._cat_group.addButton(self._rb_cat_thermal)
        v.addWidget(self._rb_cat_mechanical)
        v.addWidget(self._rb_cat_thermal)
        self._rb_cat_mechanical.toggled.connect(self._refresh_fields)
        self._rb_cat_thermal.toggled.connect(self._refresh_fields)

        # ── Mechanical Type selector (UDL / Point Load) ────────────
        self._mech_widget = QWidget(body)
        mech_layout = QVBoxLayout(self._mech_widget)
        mech_layout.setContentsMargins(0, 4, 0, 0)
        mech_layout.addWidget(QLabel("Type:", self._mech_widget))
        self._mech_group = QButtonGroup(self._mech_widget)
        self._rb_udl = QRadioButton(
            "Uniform Distributed Load", self._mech_widget,
        )
        self._rb_point = QRadioButton("Point Load", self._mech_widget)
        self._mech_group.addButton(self._rb_udl)
        self._mech_group.addButton(self._rb_point)
        mech_layout.addWidget(self._rb_udl)
        mech_layout.addWidget(self._rb_point)
        self._rb_udl.setChecked(True)
        self._rb_udl.toggled.connect(self._refresh_fields)
        self._rb_point.toggled.connect(self._refresh_fields)
        v.addWidget(self._mech_widget)

        # ── Mechanical Direction (Local / Global / Gravity) ────────
        self._coord_widget = QWidget(body)
        coord_layout = QVBoxLayout(self._coord_widget)
        coord_layout.setContentsMargins(0, 4, 0, 0)
        coord_layout.addWidget(QLabel("Direction:", self._coord_widget))
        self._coord_group = QButtonGroup(self._coord_widget)
        self._rb_local = QRadioButton(
            "Local (element axes)", self._coord_widget,
        )
        self._rb_local.setToolTip(
            "Components act along the element's local axes — qy / py "
            "transverse, qx / px axial. The interpretation rotates "
            "with the element."
        )
        self._rb_global = QRadioButton(
            "Global (X / Y axes)", self._coord_widget,
        )
        self._rb_global.setToolTip(
            "Components act along global X / Y. Force per unit MEMBER "
            "length (not per horizontal projection). The solver "
            "projects to local axes so inclined members pick up both "
            "axial and transverse fixed-end forces."
        )
        self._rb_gravity = QRadioButton(
            "Gravity (global -Y)", self._coord_widget,
        )
        self._rb_gravity.setToolTip(
            "Single magnitude — positive acts DOWN (global -Y). "
            "Distinct from Global Y: Global Y +10 means global +Y; "
            "Gravity +10 means global -Y."
        )
        self._coord_group.addButton(self._rb_local)
        self._coord_group.addButton(self._rb_global)
        self._coord_group.addButton(self._rb_gravity)
        coord_layout.addWidget(self._rb_local)
        coord_layout.addWidget(self._rb_global)
        coord_layout.addWidget(self._rb_gravity)
        self._rb_local.setChecked(True)
        self._rb_local.toggled.connect(self._refresh_fields)
        self._rb_global.toggled.connect(self._refresh_fields)
        self._rb_gravity.toggled.connect(self._refresh_fields)
        v.addWidget(self._coord_widget)

        # Helper text shown only when Local direction is active —
        # reminds the user that local axes rotate with the element's
        # i→j orientation (so +local-y is NOT "up" on a vertical
        # member). Use Global / Gravity when an absolute direction is
        # wanted. Visibility is toggled in _refresh_fields.
        self._local_help = QLabel(
            "Local directions follow the element i→j orientation "
            "(local y rotates with the member). Use Global or Gravity "
            "for absolute directions.",
            body,
        )
        self._local_help.setWordWrap(True)
        self._local_help.setStyleSheet("color: #666; font-size: 11px;")
        v.addWidget(self._local_help)

        # ── Thermal Type selector (Uniform ΔT / Gradient) ──────────
        # Gradient is frame-only; on a truss element the gradient
        # radio is disabled with a tooltip explaining why.
        self._thermal_widget = QWidget(body)
        therm_layout = QVBoxLayout(self._thermal_widget)
        therm_layout.setContentsMargins(0, 4, 0, 0)
        therm_layout.addWidget(QLabel("Thermal type:", self._thermal_widget))
        self._therm_group = QButtonGroup(self._thermal_widget)
        self._rb_t_uniform = QRadioButton(
            "Uniform ΔT", self._thermal_widget,
        )
        self._rb_t_uniform.setToolTip(
            "Single temperature change applied uniformly through the "
            "section. Produces axial thermal strain N_T = E·A·α·ΔT."
        )
        self._rb_t_gradient = QRadioButton(
            "Top / bottom gradient", self._thermal_widget,
        )
        if self._is_truss:
            self._rb_t_gradient.setEnabled(False)
            self._rb_t_gradient.setToolTip(
                "Thermal gradient requires frame bending DOFs. "
                "Truss elements support uniform ΔT only."
            )
        else:
            self._rb_t_gradient.setToolTip(
                "Specify top and bottom fiber temperatures. The mean "
                "produces axial strain; the difference produces "
                "curvature (bending if restrained)."
            )
        self._therm_group.addButton(self._rb_t_uniform)
        self._therm_group.addButton(self._rb_t_gradient)
        therm_layout.addWidget(self._rb_t_uniform)
        therm_layout.addWidget(self._rb_t_gradient)
        self._rb_t_uniform.setChecked(True)
        self._rb_t_uniform.toggled.connect(self._refresh_fields)
        self._rb_t_gradient.toggled.connect(self._refresh_fields)
        v.addWidget(self._thermal_widget)

        # ── Numeric field container (rebuilt on every selection) ────
        self._field_container = QWidget(body)
        self._field_form = QFormLayout(self._field_container)
        v.addWidget(self._field_container)
        self._fields: dict[str, QLineEdit] = {}

        # ── Load case combo (always visible) ───────────────────────
        case_label = QLabel("Load case:", body)
        v.addWidget(case_label)
        self._case_combo = _make_load_case_combo(
            body, "DEFAULT", model=self._model,
        )
        v.addWidget(self._case_combo)

        # Truss elements land on Thermal (Mechanical is disabled
        # above); frames default to Mechanical so the most common
        # action ("add a UDL") is one click away.
        if self._is_truss:
            self._rb_cat_thermal.setChecked(True)
        else:
            self._rb_cat_mechanical.setChecked(True)
        self._refresh_fields()

        # Edit mode: seed every radio + field + the case combo from the
        # caller-supplied existing load. The radio toggles already trigger
        # _refresh_fields via signal hookups, so by the time we fill the
        # QLineEdits the right keys exist in self._fields.
        if self._existing_load is not None:
            self._apply_existing_load(self._existing_load)

    def _apply_existing_load(self, load: object) -> None:
        """Seed radios + field text + case combo from an existing load.

        Called once from :meth:`_build_body` when the dialog is opened in
        edit mode. After this returns, the user can amend any field and
        click OK — :meth:`_accept` constructs a brand-new load instance
        the same way it does for Add.
        """
        # Match the case combo first — independent of category/type.
        case_name = getattr(load, "load_case", "DEFAULT") or "DEFAULT"
        idx = self._case_combo.findText(case_name)
        if idx >= 0:
            self._case_combo.setCurrentIndex(idx)

        if isinstance(load, (FrameTemperatureLoad, TrussTemperatureLoad)):
            self._rb_cat_thermal.setChecked(True)
            if isinstance(load, TrussTemperatureLoad):
                self._rb_t_uniform.setChecked(True)
                self._refresh_fields()
                if "delta_T" in self._fields:
                    self._fields["delta_T"].setText(self._fmt(load.delta_T))
                return
            # FrameTemperatureLoad: uniform iff t_top == t_bottom
            if load.t_top == load.t_bottom:
                self._rb_t_uniform.setChecked(True)
                self._refresh_fields()
                if "delta_T" in self._fields:
                    self._fields["delta_T"].setText(self._fmt(load.t_top))
            else:
                self._rb_t_gradient.setChecked(True)
                self._refresh_fields()
                if "t_top" in self._fields:
                    self._fields["t_top"].setText(self._fmt(load.t_top))
                if "t_bottom" in self._fields:
                    self._fields["t_bottom"].setText(self._fmt(load.t_bottom))
            return

        # Mechanical (UDL or PointLoad).
        self._rb_cat_mechanical.setChecked(True)
        cs = getattr(load, "coord_system", "local")
        if cs == "gravity":
            self._rb_gravity.setChecked(True)
        elif cs == "global":
            self._rb_global.setChecked(True)
        else:
            self._rb_local.setChecked(True)

        if isinstance(load, UniformDistributedLoad):
            self._rb_udl.setChecked(True)
            self._refresh_fields()
            # Gravity hides wx — the load's wx is 0 by construction; skip.
            if cs != "gravity" and "wx" in self._fields:
                self._fields["wx"].setText(self._fmt(load.wx))
            if "wy" in self._fields:
                self._fields["wy"].setText(self._fmt(load.wy))
            return

        if isinstance(load, PointLoad):
            self._rb_point.setChecked(True)
            self._refresh_fields()
            if cs != "gravity" and "px" in self._fields:
                self._fields["px"].setText(self._fmt(load.px))
            if "py" in self._fields:
                self._fields["py"].setText(self._fmt(load.py))
            if "a" in self._fields:
                self._fields["a"].setText(self._fmt(load.a))

    @staticmethod
    def _fmt(v: float) -> str:
        """Compact textual rendering for prefilled numeric fields. Uses
        ``%g`` so 0.0 → "0" and 3.14 → "3.14" without the noisy default
        ``str(float)`` exponent. Negative zero is normalised to zero so a
        round-trip ``-0.0`` doesn't surprise the user."""
        if v == 0:
            v = 0.0
        return f"{v:g}"

    def _current_category(self) -> str:
        return "thermal" if self._rb_cat_thermal.isChecked() else "mechanical"

    def _current_mechanical_kind(self) -> str:
        return "point" if self._rb_point.isChecked() else "udl"

    def _current_thermal_kind(self) -> str:
        return "gradient" if self._rb_t_gradient.isChecked() else "uniform"

    def _current_coord_system(self) -> str:
        if self._rb_gravity.isChecked():
            return "gravity"
        if self._rb_global.isChecked():
            return "global"
        return "local"

    def _refresh_fields(self) -> None:
        # Clear the field form. We ``hide()`` each old widget BEFORE
        # ``deleteLater`` rather than reparenting to ``None``:
        #   * ``deleteLater`` alone leaves the widget visible (ghosting
        #     behind the new fields) until the event loop runs;
        #   * ``setParent(None)`` removes the ghost but momentarily
        #     promotes the child to a TOP-LEVEL window, which flashes as
        #     a tiny pop-up that immediately disappears (the reported
        #     flicker when toggling Mechanical/Thermal, direction, or
        #     Uniform/Gradient).
        # ``hide()`` makes the widget invisible immediately while it
        # stays parented to ``_field_container`` (never top-level), so
        # there is neither a ghost nor a flash; ``deleteLater`` then
        # reclaims it on the next event-loop turn.
        while self._field_form.count():
            item = self._field_form.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self._fields = {}

        cat = self._current_category()
        # Helper text is Local-mechanical only.
        self._local_help.setVisible(
            cat == "mechanical" and self._current_coord_system() == "local"
        )
        if cat == "mechanical":
            self._mech_widget.setVisible(True)
            self._coord_widget.setVisible(True)
            self._thermal_widget.setVisible(False)
            kind = self._current_mechanical_kind()
            cs = self._current_coord_system()
            if kind == "udl":
                if cs == "local":
                    self._add_field(
                        "wx (kN/m, local x / axial / along i→j)", "wx",
                    )
                    self._add_field(
                        "wy (kN/m, local y / transverse / ⊥ member)", "wy",
                    )
                elif cs == "global":
                    self._add_field("qX (kN/m, global X)", "wx")
                    self._add_field("qY (kN/m, global Y)", "wy")
                else:  # gravity
                    self._add_field(
                        "qg (kN/m, +ve downward · global -Y)", "wy",
                    )
            else:  # point
                if cs == "local":
                    self._add_field(
                        "Px (kN, local x / axial / along i→j)", "px",
                    )
                    self._add_field(
                        "Py (kN, local y / transverse / ⊥ member)", "py",
                    )
                elif cs == "global":
                    self._add_field("PX (kN, global X)", "px")
                    self._add_field("PY (kN, global Y)", "py")
                else:  # gravity
                    self._add_field(
                        "Pg (kN, +ve downward · global -Y)", "py",
                    )
                self._add_field("a (m from start node)", "a")
        else:  # thermal
            self._mech_widget.setVisible(False)
            self._coord_widget.setVisible(False)
            self._thermal_widget.setVisible(True)
            # Truss elements may not use the gradient mode; force
            # uniform back on if the user (or a previous selection)
            # somehow landed on gradient.
            if self._is_truss and self._rb_t_gradient.isChecked():
                self._rb_t_uniform.setChecked(True)
            tkind = self._current_thermal_kind()
            if tkind == "uniform":
                self._add_field("ΔT (°C, uniform through depth)", "delta_T")
            else:
                self._add_field("t_top (°C)", "t_top")
                self._add_field("t_bottom (°C)", "t_bottom")

    def _add_field(self, label: str, key: str) -> None:
        e = QLineEdit(self._field_container)
        e.setText("0.0")
        self._field_form.addRow(label, e)
        self._fields[key] = e

    def _accept(self) -> Any:
        load_case = _normalize_load_case(self._case_combo.currentText())
        cat = self._current_category()
        if cat == "mechanical" and self._is_truss:
            # Defensive: the Mechanical radio is disabled for trusses
            # (see _build_body) so the user can't reach this branch
            # through the UI, but a programmatic
            # ``_rb_cat_mechanical.setChecked(True)`` could. Reject
            # here too rather than building a load the solver will
            # later reject.
            raise ValueError(
                "Truss elements do not support mechanical loads "
                "(distributed or point). Switch to Thermal — only "
                "uniform ΔT is valid on a truss."
            )
        if cat == "mechanical":
            kind = self._current_mechanical_kind()
            cs = self._current_coord_system()
            if kind == "udl":
                # parse_float's error name must match the on-screen field
                # label so "Invalid number for qY" replaces a confusing
                # "Invalid number for wy" when Global is selected.
                y_name = (
                    "qg" if cs == "gravity"
                    else ("qY" if cs == "global" else "wy")
                )
                wy = parse_float(self._fields["wy"].text(), y_name)
                # Gravity hides the wx field — pass wx=0 so the load
                # class __post_init__ accepts it. Local and global
                # show wx.
                wx = 0.0
                if cs != "gravity":
                    x_name = "qX" if cs == "global" else "wx"
                    wx = parse_float(
                        self._fields["wx"].text(), x_name, allow_blank=True,
                    ) or 0.0
                return UniformDistributedLoad(
                    wy=wy, wx=wx, coord_system=cs, load_case=load_case,
                )
            else:  # point
                y_name = (
                    "Pg" if cs == "gravity"
                    else ("PY" if cs == "global" else "Py")
                )
                py = parse_float(self._fields["py"].text(), y_name)
                px = 0.0
                if cs != "gravity":
                    x_name = "PX" if cs == "global" else "Px"
                    px = parse_float(
                        self._fields["px"].text(), x_name, allow_blank=True,
                    ) or 0.0
                a = parse_float(self._fields["a"].text(), "a")
                L, _, _ = self._elem.length_cos_sin(self._model.nodes)
                if a < 0 or a > L:
                    raise ValueError(
                        f"a must lie within [0, {L:.3g}] (element length)."
                    )
                return PointLoad(
                    py=py, a=a, px=px, coord_system=cs,
                    load_case=load_case,
                )
        # thermal
        tkind = self._current_thermal_kind()
        if tkind == "uniform":
            dT = parse_float(self._fields["delta_T"].text(), "ΔT")
            if self._is_truss:
                return TrussTemperatureLoad(
                    delta_T=dT, load_case=load_case,
                )
            # Frame uniform ΔT → store as FrameTemperatureLoad with
            # t_top == t_bottom == ΔT so the rest of the pipeline
            # (solver, format_element_loads) treats it as a pure
            # uniform load.
            return FrameTemperatureLoad(
                t_top=dT, t_bottom=dT, load_case=load_case,
            )
        # gradient — guarded by truss-disable above; defensive raise.
        if self._is_truss:
            raise ValueError(
                "Thermal gradient requires frame bending DOFs. "
                "Truss elements support uniform ΔT only."
            )
        return FrameTemperatureLoad(
            t_top=parse_float(self._fields["t_top"].text(), "t_top"),
            t_bottom=parse_float(self._fields["t_bottom"].text(), "t_bottom"),
            load_case=load_case,
        )


# ── labeled grid system ──


class GridDialog(_ModalDialog):
    """Define the X and Y grid lines (SAP2000-style)."""

    def __init__(self, parent, *, current=None, model: StructuralModel | None = None):
        # ``current`` is an optional GridSystem (None ⇒ blank).
        self._current = current
        self._model = model
        super().__init__(parent, "Grid system")

    def _build_body(self, body: QWidget) -> None:
        v = QVBoxLayout(body)
        v.addWidget(QLabel(
            "Enter label=coordinate pairs, or plain coordinates to auto-label.",
            body,
        ))
        form = QFormLayout()
        self._x_entry = QLineEdit(body)
        self._y_entry = QLineEdit(body)
        self._x_entry.setPlaceholderText("A=0, B=6, C=12")
        self._y_entry.setPlaceholderText("1=0, 2=4, 3=8")
        form.addRow("X lines:", self._x_entry)
        form.addRow("Y lines:", self._y_entry)
        v.addLayout(form)

        self._preview = QLabel("", body)
        self._preview.setWordWrap(True)
        v.addWidget(self._preview)

        fill = QPushButton("From model nodes", body)
        fill.clicked.connect(self._fill_from_model_nodes)
        v.addWidget(fill)

        if self._current is not None:
            self._x_entry.setText(", ".join(
                f"{ln.label}={ln.coord:g}" for ln in self._current.x_lines))
            self._y_entry.setText(", ".join(
                f"{ln.label}={ln.coord:g}" for ln in self._current.y_lines))

        self._x_entry.textChanged.connect(self._update_preview)
        self._y_entry.textChanged.connect(self._update_preview)
        self._update_preview()

    @staticmethod
    def _auto_label(index: int, axis_name: str) -> str:
        from .grid import _label
        return _label(index, "numeric" if axis_name == "Y" else "alpha")

    @staticmethod
    def _format_axis(lines) -> str:
        return ", ".join(f"{ln.label}={ln.coord:g}" for ln in lines)

    def _parse_axis(self, text: str, axis_name: str):
        from .grid import GridLine
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if not parts:
            return []
        has_pairs = any("=" in part for part in parts)
        lines: list = []
        if has_pairs:
            for part in parts:
                if "=" not in part:
                    raise ValueError(
                        f"{axis_name} token '{part}' is invalid: use either "
                        "all label=value pairs or all plain numbers."
                    )
                label, coord_s = part.split("=", 1)
                label = label.strip()
                if not label:
                    raise ValueError(
                        f"{axis_name} token '{part}' is invalid: label is empty."
                    )
                try:
                    coord = parse_float(coord_s.strip(),
                                        f"{axis_name} '{label}' coordinate")
                except ValueError as e:
                    raise ValueError(
                        f"{axis_name} token '{part}' is invalid: {e}"
                    ) from None
                lines.append(GridLine(label=label, coord=coord))
        else:
            coords: set[float] = set()
            for part in parts:
                try:
                    coord = parse_float(part, f"{axis_name} coordinate")
                except ValueError as e:
                    raise ValueError(
                        f"{axis_name} token '{part}' is invalid: {e}"
                    ) from None
                coords.add(coord)
            for idx, coord in enumerate(sorted(coords)):
                lines.append(GridLine(label=self._auto_label(idx, axis_name),
                                      coord=coord))
        return sorted(lines, key=lambda ln: ln.coord)

    def _fill_from_model_nodes(self) -> None:
        if self._model is None or not self._model.nodes:
            QMessageBox.information(
                self, "No model nodes",
                "Draw or open a model with nodes before using this shortcut.",
            )
            return
        xs = sorted({float(n.x) for n in self._model.nodes.values()})
        ys = sorted({float(n.y) for n in self._model.nodes.values()})
        self._x_entry.setText(", ".join(f"{x:g}" for x in xs))
        self._y_entry.setText(", ".join(f"{y:g}" for y in ys))

    def _update_preview(self) -> None:
        try:
            x_lines = self._parse_axis(self._x_entry.text(), "X")
            y_lines = self._parse_axis(self._y_entry.text(), "Y")
        except ValueError as e:
            self._preview.setText(f"Preview: {e}")
            return
        x_text = self._format_axis(x_lines) if x_lines else "(none)"
        y_text = self._format_axis(y_lines) if y_lines else "(none)"
        self._preview.setText(f"Preview: X: {x_text} | Y: {y_text}")

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


class LoadCaseManagerDialog(_ModalDialog):
    """CRUD + enable/disable + self-weight-case selection for the
    model's :class:`LoadCase` objects (PR-A — v0.18).

    The dialog NEVER mutates the model directly. It collects a list of
    commands describing the user's intent and exposes it via
    ``result_value`` on accept. The host (``MainWindow``) dispatches
    each command through ``execute()`` so every change is undoable and
    the standard invalidation surface runs (clears the multi-case
    result, refreshes the toolbar combo).
    """

    def __init__(self, parent, *, model: StructuralModel):
        self._model = model
        # Working copy keyed by current-name -> {name, enabled,
        # description, original_name}. ``original_name`` is None for
        # rows the user has freshly added (so we know to emit an
        # AddLoadCaseCmd rather than a Rename + Set).
        self._rows: list[dict] = [
            {
                "name": name,
                "enabled": lc.enabled,
                "description": lc.description,
                "original_name": name,
                "deleted": False,
            }
            for name, lc in sorted(model.load_cases.items())
        ]
        self._self_weight_case_initial = model.self_weight_case
        super().__init__(parent, "Load cases")

    def _build_body(self, body: QWidget) -> None:
        v = QVBoxLayout(body)
        v.addWidget(QLabel(
            "User-defined load cases for the multi-case static run. "
            "Disabled cases are skipped by 'Solve all cases' (F5).",
            body,
        ))
        # Table: Name · Enabled · Self-weight · Notes (DEFAULT is
        # decorated with a (default) marker and isn't deletable).
        from PyQt6.QtWidgets import QPushButton
        self._table = QTableWidget(0, 4, body)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Enabled", "Self-weight", "Notes"]
        )
        install_table_copy(self._table, include_headers=True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        v.addWidget(self._table)
        # Edit row: name field + Add / Rename / Delete / Toggle buttons
        # operate on the row currently highlighted via "Edit" links per
        # row. To keep the dialog simple we expose just an Add button
        # and per-row Delete / Enable-toggle controls; renaming is done
        # in place via the Name column (the cell is editable).
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.SelectedClicked
        )
        # Self-weight chooser as a single combo below the table.
        from PyQt6.QtWidgets import QHBoxLayout
        sw_row = QHBoxLayout()
        sw_row.addWidget(QLabel("Self-weight applied to:", body))
        self._sw_combo = QComboBox(body)
        sw_row.addWidget(self._sw_combo)
        sw_row.addStretch()
        v.addLayout(sw_row)
        # Add new case row.
        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Add case:", body))
        self._new_name = QLineEdit(body)
        self._new_name.setPlaceholderText("e.g. WIND_X")
        add_row.addWidget(self._new_name)
        add_btn = QPushButton("Add", body)
        add_btn.clicked.connect(self._on_add_clicked)
        add_row.addWidget(add_btn)
        v.addLayout(add_row)
        # Populate the table from the current row list + sw combo.
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        """Render ``self._rows`` into the table widget."""
        from PyQt6.QtWidgets import QPushButton, QCheckBox
        # Block table itemChanged during rebuild so synthetic
        # setItem calls don't fire spurious rename intents.
        try:
            self._table.itemChanged.disconnect(self._on_item_changed)
        except (TypeError, RuntimeError):
            pass
        live_rows = [r for r in self._rows if not r["deleted"]]
        self._table.setRowCount(len(live_rows))
        for i, row in enumerate(live_rows):
            name_item = QTableWidgetItem(row["name"])
            if row["name"] == "DEFAULT":
                # DEFAULT cannot be renamed.
                name_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                name_item.setForeground(QColor("#444"))
            self._table.setItem(i, 0, name_item)
            cb = QCheckBox()
            cb.setChecked(row["enabled"])
            cb.stateChanged.connect(
                lambda state, r=row: r.update(enabled=bool(state))
            )
            self._table.setCellWidget(i, 1, cb)
            sw_label = QLabel(
                "✓" if self._sw_combo_value() == row["name"] else ""
            )
            sw_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setCellWidget(i, 2, sw_label)
            if row["name"] == "DEFAULT":
                self._table.setItem(i, 3, QTableWidgetItem("(default — cannot delete)"))
            else:
                del_btn = QPushButton("Delete")
                del_btn.clicked.connect(
                    lambda _checked=False, r=row: self._on_delete_clicked(r)
                )
                self._table.setCellWidget(i, 3, del_btn)
        # Repopulate self-weight combo with all live names.
        existing = self._sw_combo_value() or self._self_weight_case_initial
        self._sw_combo.blockSignals(True)
        self._sw_combo.clear()
        names = [r["name"] for r in live_rows]
        self._sw_combo.addItems(names)
        if existing in names:
            self._sw_combo.setCurrentText(existing)
        elif "DEFAULT" in names:
            self._sw_combo.setCurrentText("DEFAULT")
        self._sw_combo.blockSignals(False)
        self._table.itemChanged.connect(self._on_item_changed)

    def _sw_combo_value(self) -> str:
        if not hasattr(self, "_sw_combo"):
            return self._self_weight_case_initial
        return self._sw_combo.currentText() or self._self_weight_case_initial

    def _on_add_clicked(self) -> None:
        raw = self._new_name.text()
        try:
            name = _normalize_load_case(raw)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid case name", str(e))
            return
        if name == "DEFAULT":
            QMessageBox.warning(
                self, "Invalid case name",
                "DEFAULT already exists.",
            )
            return
        live_names = {r["name"] for r in self._rows if not r["deleted"]}
        if name in live_names:
            QMessageBox.warning(
                self, "Invalid case name",
                f"Case {name!r} is already defined.",
            )
            return
        self._rows.append({
            "name": name,
            "enabled": True,
            "description": "",
            "original_name": None,   # freshly added
            "deleted": False,
        })
        self._new_name.clear()
        self._rebuild_table()

    def _on_delete_clicked(self, row: dict) -> None:
        if row["name"] == "DEFAULT":
            return
        ans = QMessageBox.question(
            self, "Delete load case",
            f"Delete load case {row['name']!r}? Loads tagged with this "
            "case will be re-tagged to DEFAULT.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if ans != QMessageBox.StandardButton.Ok:
            return
        row["deleted"] = True
        self._rebuild_table()

    def _set_item_text_silent(
        self, item: QTableWidgetItem, text: str,
    ) -> None:
        """Update a table item's text without re-firing
        ``itemChanged`` — used by ``_on_item_changed`` to roll back
        invalid edits or to write the normalised name back. Defensive
        block-signals-on-table during the setText so the recursive
        call doesn't even need to early-return (Gemini PR #28
        finding)."""
        self._table.blockSignals(True)
        try:
            item.setText(text)
        finally:
            self._table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        # Only the Name column (0) is editable, and only for non-DEFAULT.
        if item.column() != 0:
            return
        live_rows = [r for r in self._rows if not r["deleted"]]
        r = live_rows[item.row()]
        new_text = item.text()
        try:
            normalised = _normalize_load_case(new_text)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid case name", str(e))
            self._set_item_text_silent(item, r["name"])
            return
        if normalised == r["name"]:
            return
        if normalised == "DEFAULT":
            QMessageBox.warning(
                self, "Invalid case name",
                "DEFAULT already exists.",
            )
            self._set_item_text_silent(item, r["name"])
            return
        live_names = {row["name"] for row in live_rows if row is not r}
        if normalised in live_names:
            QMessageBox.warning(
                self, "Invalid case name",
                f"Case {normalised!r} is already defined.",
            )
            self._set_item_text_silent(item, r["name"])
            return
        r["name"] = normalised
        self._set_item_text_silent(item, normalised)

    def _accept(self) -> list:
        """Return the list of commands needed to apply the dialog's
        edits. The host dispatches them in order."""
        from ..gui_common.commands import (
            AddLoadCaseCmd,
            DeleteLoadCaseCmd,
            RenameLoadCaseCmd,
            SetLoadCaseEnabledCmd,
            SetSelfWeightCaseCmd,
        )
        cmds: list = []
        # 1. Deletes first (so a rename can't collide with a deleted
        # name that's about to come back).
        for r in self._rows:
            if r["deleted"] and r["original_name"] is not None:
                cmds.append(DeleteLoadCaseCmd(name=r["original_name"]))
        # 2. Renames + Adds, then enable-toggles.
        for r in self._rows:
            if r["deleted"]:
                continue
            orig = r["original_name"]
            if orig is None:
                cmds.append(AddLoadCaseCmd(
                    name=r["name"], enabled=r["enabled"],
                ))
            elif orig != r["name"]:
                cmds.append(RenameLoadCaseCmd(
                    old_name=orig, new_name=r["name"],
                ))
                # Apply enabled-flag too in case it also changed.
                if (
                    r["name"] in self._model.load_cases
                    and r["enabled"] != self._model.load_cases[orig].enabled
                ):
                    cmds.append(SetLoadCaseEnabledCmd(
                        name=r["name"], enabled=r["enabled"],
                    ))
            else:
                # Pure enable toggle.
                if orig in self._model.load_cases:
                    if (
                        self._model.load_cases[orig].enabled != r["enabled"]
                    ):
                        cmds.append(SetLoadCaseEnabledCmd(
                            name=r["name"], enabled=r["enabled"],
                        ))
        # 3. Self-weight case (after deletes/renames so the target
        # exists post-cascade).
        sw = self._sw_combo_value()
        if sw and sw != self._self_weight_case_initial:
            cmds.append(SetSelfWeightCaseCmd(case_name=sw))
        return cmds


def _parse_terms_expression(text: str) -> dict[str, float]:
    """Parse a combination terms expression into ``{case: coeff}``.

    Accepted forms (``+`` separates terms; ``*`` or whitespace separates
    coefficient from case)::

        1.2*DEAD + 1.6*LIVE
        1.0 DEAD + 0.7 WIND_X

    Case names are normalised (uppercased). Raises ``ValueError`` on a
    malformed term, a duplicate case, a non-finite/zero coefficient, or
    an empty expression."""
    import math
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Enter at least one term, e.g. 1.2*DEAD + 1.6*LIVE.")
    terms: dict[str, float] = {}
    for chunk in raw.split("+"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "*" in chunk:
            coeff_s, _, case_s = chunk.partition("*")
        else:
            parts = chunk.split(None, 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Cannot parse term {chunk!r}; use "
                    "'coefficient*CASE' (e.g. 1.2*DEAD)."
                )
            coeff_s, case_s = parts
        coeff_s = coeff_s.strip()
        case_s = _normalize_load_case(case_s.strip())
        try:
            coeff = float(coeff_s)
        except ValueError:
            raise ValueError(
                f"Coefficient {coeff_s!r} in term {chunk!r} is not a number."
            )
        if not math.isfinite(coeff):
            raise ValueError(
                f"Coefficient in term {chunk!r} must be finite."
            )
        if coeff == 0.0:
            raise ValueError(
                f"Term {chunk!r} has a zero coefficient; remove it instead."
            )
        if case_s in terms:
            raise ValueError(
                f"Case {case_s!r} appears more than once in the expression."
            )
        terms[case_s] = coeff
    if not terms:
        raise ValueError("Enter at least one term, e.g. 1.2*DEAD + 1.6*LIVE.")
    return terms


def _format_terms_expression(terms: dict[str, float]) -> str:
    """Inverse of :func:`_parse_terms_expression` for display / editing."""
    return " + ".join(
        f"{coeff:g}*{case}" for case, coeff in terms.items()
    )


class LoadCombinationManagerDialog(_ModalDialog):
    """Add / rename / delete coefficient combinations + edit their terms
    (PR #29 — v0.19).

    Like :class:`LoadCaseManagerDialog`, this dialog never mutates the
    model directly. It collects the intent as a list of commands exposed
    via ``result_value`` on accept; the host dispatches them through the
    undoable ``execute()`` pipeline."""

    def __init__(self, parent, *, model: StructuralModel):
        self._model = model
        # Working rows: {name, terms (dict), description, original_name,
        # deleted}. original_name is None for freshly-added rows.
        self._rows: list[dict] = [
            {
                "name": name,
                "terms": dict(c.terms),
                "description": c.description,
                "original_name": name,
                "deleted": False,
            }
            for name, c in sorted(model.load_combinations.items())
        ]
        super().__init__(parent, "Load combinations")

    def _build_body(self, body: QWidget) -> None:
        from PyQt6.QtWidgets import QHBoxLayout, QPushButton
        v = QVBoxLayout(body)
        cases = ", ".join(sorted(self._model.load_cases.keys()))
        v.addWidget(QLabel(
            "Coefficient combinations of solved load cases (derived "
            "views — not separately solved).\n"
            f"Available cases: {cases}",
            body,
        ))
        self._table = QTableWidget(0, 4, body)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Terms", "Description", ""]
        )
        install_table_copy(self._table, include_headers=True)
        self._table.verticalHeader().setVisible(False)
        # The Name column is editable in place (double-click) so a
        # combination can be RENAMED — _on_item_changed turns the edit
        # into a RenameLoadCombinationCmd on accept. Terms / Description
        # stay read-only (edit them via the form below).
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.SelectedClicked
        )
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self._table)

        # Add / update form.
        form_box = QWidget(body)
        form = QFormLayout(form_box)
        self._name_edit = QLineEdit(form_box)
        self._name_edit.setPlaceholderText("e.g. COMB_STRENGTH")
        form.addRow("Name:", self._name_edit)
        self._terms_edit = QLineEdit(form_box)
        self._terms_edit.setPlaceholderText("1.2*DEAD + 1.6*LIVE")
        form.addRow("Terms:", self._terms_edit)
        self._desc_edit = QLineEdit(form_box)
        form.addRow("Description:", self._desc_edit)
        v.addWidget(form_box)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add / update combination", form_box)
        add_btn.clicked.connect(self._on_add_or_update_clicked)
        btn_row.addWidget(add_btn)
        btn_row.addStretch()
        v.addLayout(btn_row)

        self._rebuild_table()

    def _rebuild_table(self) -> None:
        from PyQt6.QtWidgets import QPushButton
        # Suppress itemChanged while we synthesise cells.
        try:
            self._table.itemChanged.disconnect(self._on_item_changed)
        except (TypeError, RuntimeError):
            pass
        live = [r for r in self._rows if not r["deleted"]]
        self._table.setRowCount(len(live))
        for i, row in enumerate(live):
            self._table.setItem(i, 0, QTableWidgetItem(row["name"]))
            # Terms / Description are read-only cells (edit via the form).
            terms_item = QTableWidgetItem(
                _format_terms_expression(row["terms"])
            )
            terms_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            self._table.setItem(i, 1, terms_item)
            desc_item = QTableWidgetItem(row["description"])
            desc_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            self._table.setItem(i, 2, desc_item)
            del_btn = QPushButton("Delete")
            del_btn.clicked.connect(
                lambda _checked=False, r=row: self._on_delete_clicked(r)
            )
            self._table.setCellWidget(i, 3, del_btn)
        self._table.itemChanged.connect(self._on_item_changed)

    def _set_item_text_silent(self, item, text: str) -> None:
        self._table.blockSignals(True)
        try:
            item.setText(text)
        finally:
            self._table.blockSignals(False)

    def _on_item_changed(self, item) -> None:
        """In-place rename via the Name column (column 0)."""
        if item.column() != 0:
            return
        live = [r for r in self._rows if not r["deleted"]]
        r = live[item.row()]
        try:
            new_name = _normalize_load_case(item.text())
        except ValueError as e:
            QMessageBox.warning(self, "Invalid combination name", str(e))
            self._set_item_text_silent(item, r["name"])
            return
        if new_name == r["name"]:
            return
        if new_name == "SUM_ALL":
            QMessageBox.warning(
                self, "Invalid combination name",
                "SUM_ALL is a built-in derived view and cannot be used.",
            )
            self._set_item_text_silent(item, r["name"])
            return
        if new_name in self._model.load_cases:
            QMessageBox.warning(
                self, "Invalid combination name",
                f"{new_name!r} is already a load-case name.",
            )
            self._set_item_text_silent(item, r["name"])
            return
        live_names = {row["name"] for row in live if row is not r}
        if new_name in live_names:
            QMessageBox.warning(
                self, "Invalid combination name",
                f"Combination {new_name!r} is already defined.",
            )
            self._set_item_text_silent(item, r["name"])
            return
        r["name"] = new_name
        self._set_item_text_silent(item, new_name)

    def _on_delete_clicked(self, row: dict) -> None:
        row["deleted"] = True
        self._rebuild_table()

    def _on_add_or_update_clicked(self) -> None:
        try:
            name = _normalize_load_case(self._name_edit.text())
        except ValueError as e:
            QMessageBox.warning(self, "Invalid combination name", str(e))
            return
        if name == "SUM_ALL":
            QMessageBox.warning(
                self, "Invalid combination name",
                "SUM_ALL is a built-in derived view and cannot be used "
                "as a combination name.",
            )
            return
        if name in self._model.load_cases:
            QMessageBox.warning(
                self, "Invalid combination name",
                f"{name!r} is already a load-case name; combination "
                "names must be distinct from case names.",
            )
            return
        try:
            terms = _parse_terms_expression(self._terms_edit.text())
        except ValueError as e:
            QMessageBox.warning(self, "Invalid terms", str(e))
            return
        # Referenced cases must exist.
        missing = sorted(c for c in terms if c not in self._model.load_cases)
        if missing:
            QMessageBox.warning(
                self, "Invalid terms",
                "Combination references unknown load case(s): "
                + ", ".join(missing),
            )
            return
        desc = self._desc_edit.text().strip()
        # Update an existing live row with this name, else append a new.
        for r in self._rows:
            if not r["deleted"] and r["name"] == name:
                r["terms"] = terms
                r["description"] = desc
                break
        else:
            self._rows.append({
                "name": name,
                "terms": terms,
                "description": desc,
                "original_name": None,
                "deleted": False,
            })
        self._name_edit.clear()
        self._terms_edit.clear()
        self._desc_edit.clear()
        self._rebuild_table()

    def _accept(self) -> list:
        from ..gui_common.commands import (
            AddLoadCombinationCmd,
            DeleteLoadCombinationCmd,
            RenameLoadCombinationCmd,
            SetLoadCombinationTermsCmd,
        )
        cmds: list = []
        # Deletes first.
        for r in self._rows:
            if r["deleted"] and r["original_name"] is not None:
                cmds.append(DeleteLoadCombinationCmd(name=r["original_name"]))
        # Adds + renames + term edits.
        for r in self._rows:
            if r["deleted"]:
                continue
            orig = r["original_name"]
            if orig is None:
                cmds.append(AddLoadCombinationCmd(
                    name=r["name"], terms=dict(r["terms"]),
                    combo_description=r["description"],
                ))
            else:
                if orig != r["name"]:
                    cmds.append(RenameLoadCombinationCmd(
                        old_name=orig, new_name=r["name"],
                    ))
                original = self._model.load_combinations.get(orig)
                terms_changed = (
                    original is None or original.terms != r["terms"]
                    or original.description != r["description"]
                )
                if terms_changed:
                    cmds.append(SetLoadCombinationTermsCmd(
                        name=r["name"], terms=dict(r["terms"]),
                        combo_description=r["description"],
                    ))
        return cmds


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
        install_table_copy(self._mat_tree, include_headers=True)
        self._mat_tree.setHeaderLabels(
            ["id", "name", "E (kN/m²)", "α (1/°C)", "ρ (kg/m³)",
             "ν", "G (derived)"]
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
        install_table_copy(self._sec_tree, include_headers=True)
        self._sec_tree.setHeaderLabels(
            ["id", "name", "material", "A (m²)", "I (m⁴)", "depth (m)",
             "width (m)", "shape"]
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
                f"{m.density:g}", f"{m.nu:g}", f"{m.G:g}",
            ])
        self._sec_tree.clear()
        for sid in sorted(self._model.sections):
            s = self._model.sections[sid]
            QTreeWidgetItem(self._sec_tree, [
                str(s.id), s.name, str(s.material_id),
                f"{s.A:g}", f"{s.I:g}", f"{s.depth:g}",
                f"{s.width:g}", s.shape_type,
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

        # Mass formulation. Consistent is the default and unchanged
        # from v0.9.1; lumped translational is a comparison aid that
        # condenses rotational DOFs out of the modal eigenproblem.
        self._mass_combo = QComboBox(body)
        self._mass_combo.addItem(
            "Consistent element mass", "consistent",
        )
        self._mass_combo.addItem(
            "Lumped translational mass  (comparison aid)", "lumped",
        )
        form.addRow("Mass formulation", self._mass_combo)

        note = QLabel(
            "Modal analysis requires a positive density on every "
            "element's material.\nSet density (kg/m³) on each Material "
            "via Edit → Materials and sections.\n\n"
            "Lumped translational mass is a comparison aid. Agreement "
            "with external software depends on matching units, "
            "density/mass source, section properties, mesh, boundary "
            "conditions, restraints, and mass formulation.",
            body,
        )
        note.setWordWrap(True)
        form.addRow(note)

    def _accept(self) -> dict:
        n = parse_int(self._n_modes.text(), "Number of modes")
        if n < 1:
            raise ValueError("Number of modes must be at least 1.")
        norm = self._norm_combo.currentData()
        mass_formulation = self._mass_combo.currentData()
        return {
            "n_modes": n,
            "normalisation": norm,
            "mass_formulation": mass_formulation,
        }


# ── read-only property inspectors (left-click in Select tool) ──


def _support_summary(support: Support | None) -> str:
    if support is None:
        return "free"
    flags = (bool(support.ux), bool(support.uy), bool(support.rz))
    if flags == (True, True, True):
        kind = "fixed"
    elif flags == (True, True, False):
        kind = "pin"
    elif flags == (False, True, False):
        kind = "roller (uy)"
    elif flags == (True, False, False):
        kind = "roller (ux)"
    elif flags == (False, False, False):
        kind = "free"
    else:
        kind = "custom"
    parts = [f"ux={support.ux}", f"uy={support.uy}", f"rz={support.rz}"]
    settle = []
    for dof in ("ux", "uy", "rz"):
        v = getattr(support, f"settle_{dof}")
        if v is not None and abs(v) > 0.0:
            settle.append(f"Δ{dof}={v:g}")
    body = "  ".join(parts) + (f"  ·  settle: {', '.join(settle)}" if settle else "")
    return f"{kind}  ·  {body}"


def _nodal_load_summary(model: StructuralModel, node_id: int) -> str:
    """Multi-row summary of every nodal load attached to ``node_id``.

    Pre-v0.20 the inspector only showed a single load per node (the
    storage layer allowed multiples but the GUI consolidated). The
    manager dialog now exposes the full list; this helper formats it
    for the read-only :class:`NodePropertiesDialog` view.
    """
    loads = [ld for ld in model.nodal_loads if ld.node_id == node_id]
    if not loads:
        return "(none)"

    def _row(ld: NodalLoad) -> str:
        base = (
            f"Fx = {ld.fx:g} kN,  Fy = {ld.fy:g} kN,  "
            f"Mz = {ld.mz:g} kN·m"
        )
        case = getattr(ld, "load_case", "DEFAULT")
        if case and case != "DEFAULT":
            base += f"  ·  case: {case}"
        return base

    if len(loads) == 1:
        return _row(loads[0])
    return "\n".join(f"• {_row(ld)}" for ld in loads)


def _member_loads_summary(elem) -> list[str]:
    loads = list(getattr(elem, "member_loads", []) or [])
    if not loads:
        return ["(none)"]
    out: list[str] = []
    for ld in loads:
        if isinstance(ld, UniformDistributedLoad):
            out.append(f"UDL: wy = {ld.wy:g} kN/m")
        elif isinstance(ld, PointLoad):
            out.append(f"Point: py = {ld.py:g} kN at a = {ld.a:g} m")
        elif isinstance(ld, FrameTemperatureLoad):
            out.append(
                f"Thermal frame: t_top = {ld.t_top:g} °C, "
                f"t_bottom = {ld.t_bottom:g} °C"
            )
        elif isinstance(ld, TrussTemperatureLoad):
            out.append(f"Thermal truss: ΔT = {ld.delta_T:g} °C")
        else:
            out.append(repr(ld))
    return out


class NodePropertiesDialog(QDialog):
    """Read-only inspector for a node — opened by left-click in Select tool."""

    def __init__(self, parent, model: StructuralModel, node_id: int,
                 result=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Node {node_id} properties")
        self.setModal(True)
        if node_id not in model.nodes:
            raise ValueError(f"Node {node_id} does not exist.")

        node = model.nodes[node_id]
        support = model.supports.get(node_id)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        form.addRow("Node ID:", QLabel(str(node_id)))
        form.addRow("Coordinates:",
                     QLabel(f"x = {node.x:g} m,  y = {node.y:g} m"))
        form.addRow("Support:", QLabel(_support_summary(support)))
        form.addRow("Nodal load:", QLabel(_nodal_load_summary(model, node_id)))

        # Result rows (only if a successful static result is available).
        if result is not None and getattr(result, "status", None) == "ok":
            sep = QLabel("── Result ──")
            sep.setStyleSheet("font-weight: bold;")
            layout.addWidget(sep)
            res_form = QFormLayout()
            layout.addLayout(res_form)

            disp = _node_displacement(result, node_id)
            res_form.addRow("Displacement:", QLabel(disp))
            reac = _node_reaction(result, node_id)
            if reac is not None:
                res_form.addRow("Reaction:", QLabel(reac))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close,
                                     parent=self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        # Close button on a Close-only box fires `rejected`; wire it to accept too.
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)
        layout.addWidget(buttons)


def _node_displacement(result, node_id: int) -> str:
    nm = result.E_map.get(node_id)
    if nm is None or result.D is None:
        return "(not available)"
    parts = []
    for dof in ("ux", "uy", "rz"):
        idx = nm.get(dof)
        if idx is None:
            parts.append(f"{dof} = 0  (restrained)")
        else:
            val = float(result.D[idx])
            unit = "rad" if dof == "rz" else "m"
            parts.append(f"{dof} = {val:.6e} {unit}")
    return ",  ".join(parts)


def _node_reaction(result, node_id: int) -> str | None:
    reactions = getattr(result, "reactions", None) or {}
    r = reactions.get(node_id)
    if not r:
        return None
    parts = []
    for dof, label, unit in (("ux", "Rx", "kN"),
                              ("uy", "Ry", "kN"),
                              ("rz", "Mz", "kN·m")):
        if dof in r:
            parts.append(f"{label} = {r[dof]:.4f} {unit}")
    if not parts:
        return None
    return ",  ".join(parts)


class ElementPropertiesDialog(QDialog):
    """Read-only / loads-edit inspector for an element (SAP2000-style tabs).

    PR #35: three tabs on a ``QTabWidget`` — **Properties** (geometry +
    section thumbnail + material/E/A/I + releases), **Results** (N/V/M
    diagrams with a dialog-local case/combo selector, end-force table,
    crosshair readout), and **Load Assignments** (member-loads table with
    per-row Edit / Delete buttons and an Add button at the bottom).

    Non-modal so the main window stays usable for view-only operations
    (pan / zoom / solve / overlay toggles). The host
    (:class:`MainWindow`) is responsible for locking edit actions
    while the inspector is visible — see
    :meth:`MainWindow._set_editing_locked`.

    Constructed as a singleton on ``MainWindow._element_inspector``;
    re-opening from another element calls :meth:`set_target` to swap
    the contents in place. After a solve the host calls
    :meth:`refresh` so the diagrams pick up the new result.

    Public API preserved across the PR #35 rewrite:
    ``set_target``, ``refresh``, ``refresh_loads_only``, ``_elem_id``,
    ``_detail_axes`` (still a 4-key dict with sketch/fbd/diagrams/section
    keys — sketch+fbd+section come from the Properties tab figure,
    diagrams come from the Results tab figure).

    Host callbacks (set by MainWindow after construction):
      _host_delete_member_load(elem_id, load_index)
      _host_add_member_load(elem_id)
      _host_edit_member_load(elem_id, load_index)
    All three remain ``None`` in unit-test construction so the row
    buttons render disabled and the model is never mutated implicitly.
    """

    _TAB_PROPERTIES = 0
    _TAB_RESULTS = 1
    _TAB_LOADS = 2
    _TAB_NAME_TO_INDEX = {
        "properties": _TAB_PROPERTIES,
        "results": _TAB_RESULTS,
        "loads": _TAB_LOADS,
    }

    def __init__(
        self, parent, model: StructuralModel, elem_id: int,
        result=None, *, multi_result=None,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._outer = QVBoxLayout(self)
        # Body widget — a QTabWidget wholly replaced by set_target on
        # each refresh.  Keep the variable name ``_body_widget`` so the
        # set_target / refresh_loads_only logic doesn't change shape.
        self._body_widget: QWidget = QWidget(self)
        self._outer.addWidget(self._body_widget)

        # Buttons live at the bottom permanently.
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close, parent=self,
        )
        self._buttons.rejected.connect(self.close)
        self._buttons.accepted.connect(self.close)
        close_btn = self._buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.close)
        self._outer.addWidget(self._buttons)

        self._elem_id: int = elem_id
        # Host callbacks (set by MainWindow after construction).
        self._host_delete_member_load = None
        self._host_add_member_load = None
        self._host_edit_member_load = None

        # Per-rebuild widget handles — every tab body is wholly rebuilt
        # by set_target so these get reassigned on each refresh.
        # ``_loads_widget`` keeps the legacy semantics (points at the
        # QTableWidget itself); ``_loads_tab_widget`` holds the whole
        # tab body inserted into the QTabWidget.
        self._tabs: QTabWidget | None = None
        self._props_widget: QWidget | None = None
        self._results_widget: QWidget | None = None
        self._loads_tab_widget: QWidget | None = None
        self._loads_widget: QWidget | None = None
        # "Show Maxima" is ON by default and its state is remembered
        # across rebuilds (set_target / refresh / Results-tab case switch)
        # for the lifetime of the dialog — switching the case must NOT
        # reset this user choice.
        self._show_maxima_on: bool = True

        # Dialog-local result selection (raw case / SUM_ALL / combination
        # name).  Initialised to the host's active case at open time so
        # the dialog mirrors what the user is currently viewing on the
        # canvas, then drifts independently as the user picks a different
        # case in the Results tab.
        active_case = getattr(parent, "_active_case", "DEFAULT")
        self._results_selection: str = active_case

        self.set_target(model, elem_id, result, multi_result=multi_result)

    def set_target(
        self, model: StructuralModel, elem_id: int, result=None,
        *, multi_result=None,
    ) -> None:
        """Swap the inspector to show ``elem_id``. Raises ``ValueError``
        if the element does not exist. All three tabs are rebuilt; the
        Properties / Loads tab bodies are inexpensive, and Results is
        the only one that depends on the result so it's the natural
        rebuild boundary even when only the result changed."""
        elem = next((e for e in model.elements if e.id == elem_id), None)
        if elem is None:
            raise ValueError(f"Element {elem_id} does not exist.")
        # Preserve the focused tab across the full rebuild so a
        # refresh() triggered from the Load Assignments tab (after an
        # Add / Edit / Delete) doesn't bounce the user back to
        # Properties.  The host calls set_initial_tab() AFTER set_target
        # when it wants a specific tab (open / right-click routing), so
        # that explicit intent still wins.
        prev_index = (
            self._tabs.currentIndex() if self._tabs is not None else None
        )
        new_body = self._build_tabs(model, elem, result, multi_result)
        self._outer.replaceWidget(self._body_widget, new_body)
        self._body_widget.setParent(None)
        self._body_widget.deleteLater()
        self._body_widget = new_body
        self._elem_id = elem_id
        self.setWindowTitle(f"Element {elem_id} properties")
        if (
            prev_index is not None
            and self._tabs is not None
            and 0 <= prev_index < self._tabs.count()
        ):
            self._tabs.setCurrentIndex(prev_index)

    def refresh(
        self, model: StructuralModel, result=None, *, multi_result=None,
    ) -> None:
        """Re-render the current element. Called by the host after solve /
        case change / model mutation. Silently no-ops if the current
        element id no longer exists.

        Preserves :attr:`_results_selection` so the user's pinned dialog
        case stays put across solves; the new result is re-resolved
        against the same selection in the rebuilt Results tab."""
        if not any(e.id == self._elem_id for e in model.elements):
            self.close()
            return
        self.set_target(
            model, self._elem_id, result, multi_result=multi_result,
        )

    def set_initial_tab(self, name: str) -> None:
        """Switch the focused tab to ``name`` (``"properties"``,
        ``"results"``, or ``"loads"``). Used by the host when right-click
        routes directly to "Edit member loads…" or "Element Details"."""
        idx = self._TAB_NAME_TO_INDEX.get(name)
        if idx is not None and self._tabs is not None:
            self._tabs.setCurrentIndex(idx)

    # ── tab assembly ──────────────────────────────────────────────────

    def _build_tabs(
        self, model: StructuralModel, elem, result, multi_result,
    ) -> QWidget:
        from .element_graphics import (
            internal_force_at,
            sample_internal_force,
        )

        # Element-wide cached data the interactive crosshair layer needs.
        ni = model.nodes.get(elem.node_i)
        nj = model.nodes.get(elem.node_j)
        if ni is None or nj is None:
            length = 0.0
        else:
            length = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
        self._elem_ref = elem
        self._ni_ref = ni
        self._nj_ref = nj
        self._L = length
        self._internal_force_at = internal_force_at
        self._sample_internal_force = sample_internal_force

        # Tab widget is the entire body.  We never let the QTabWidget
        # itself be the outer body widget because set_target() replaces
        # _body_widget wholesale — keep the same pattern (wrap in a
        # QWidget) so the replace logic doesn't need to special-case.
        container = QWidget(self)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget(container)
        self._tabs = tabs

        self._props_widget = self._build_properties_tab(model, elem)
        tabs.addTab(self._props_widget, "Properties")

        self._results_widget = self._build_results_tab(
            model, elem, result, multi_result,
        )
        tabs.addTab(self._results_widget, "Results")

        self._loads_tab_widget = self._build_loads_tab(model, elem)
        tabs.addTab(self._loads_tab_widget, "Load Assignments")

        container_layout.addWidget(tabs)
        return container

    # ── Properties tab ────────────────────────────────────────────────

    def _build_properties_tab(
        self, model: StructuralModel, elem,
    ) -> QWidget:
        """Build the Properties tab: geometry + section + material form,
        member-sketch / FBD figure, side-by-side section thumbnail."""
        from .element_graphics import draw_element_detail

        body = QWidget(self)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)

        elem_id = elem.id
        section = model.sections.get(getattr(elem, "section_id", None) or -1)
        default_material = (
            model.materials.get(section.material_id)
            if section is not None else None
        )
        override_id = getattr(elem, "material_id_override", None)
        if override_id is not None:
            effective_mat = model.materials.get(override_id)
        else:
            effective_mat = default_material

        ni = self._ni_ref
        nj = self._nj_ref
        length = self._L

        # Top row: form on the left, section thumbnail on the right.
        top_row = QHBoxLayout()
        form_widget = QWidget(body)
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        top_row.addWidget(form_widget, stretch=3)

        self._section_fig = Figure(figsize=(2.2, 2.2), dpi=92)
        self._section_fig.patch.set_facecolor("white")
        self._section_canvas = FigureCanvasQTAgg(self._section_fig)
        self._section_canvas.setMinimumSize(180, 180)
        self._section_canvas.setMaximumSize(260, 260)
        top_row.addWidget(self._section_canvas, stretch=1)
        layout.addLayout(top_row)

        form.addRow("Element ID:", QLabel(str(elem_id)))
        form.addRow("Kind:", QLabel(elem.kind.capitalize()))
        form.addRow("Nodes:", QLabel(f"{elem.node_i} → {elem.node_j}"))
        form.addRow("Length:", QLabel(f"{length:g} m"))
        if section is not None:
            sec_text = section.name or f"section {section.id}"
            form.addRow("Section:", QLabel(f"{sec_text}  (id {section.id})"))
        else:
            form.addRow("Section:", QLabel("(none)"))
        if effective_mat is not None:
            eff_name = effective_mat.name or f"material {effective_mat.id}"
            if override_id is None:
                tag = "— section default"
                mat_line = f"{eff_name}  (id {effective_mat.id})  {tag}"
            else:
                default_name = (
                    default_material.name
                    if default_material is not None and default_material.name
                    else f"material "
                         f"{section.material_id if section else '?'}"
                )
                default_id = (
                    default_material.id
                    if default_material is not None
                    else (section.material_id if section else None)
                )
                mat_line = (
                    f"{eff_name}  (id {effective_mat.id})  — override "
                    f"(default: {default_name}, id {default_id})"
                )
            form.addRow("Material:", QLabel(mat_line))

        form.addRow("E:", QLabel(f"{elem.E:g} kN/m²"))
        form.addRow("A:", QLabel(f"{elem.A:g} m²"))
        if isinstance(elem, FrameElement2D):
            form.addRow("I:", QLabel(f"{elem.I:g} m⁴"))
            form.addRow(
                "Releases:",
                QLabel(f"i={elem.release_i},  j={elem.release_j}"),
            )

        # Sketch + FBD figure (no diagrams here — those live in the
        # Results tab).  Section thumbnail is drawn into the side
        # mini-figure via section_fig=.
        self._props_fig = Figure(figsize=(6.5, 3.0), dpi=92)
        self._props_fig.patch.set_facecolor("white")
        self._props_canvas = FigureCanvasQTAgg(self._props_fig)
        self._props_canvas.setMinimumSize(520, 220)
        layout.addWidget(self._props_canvas)
        props_axes = draw_element_detail(
            self._props_fig, elem, model, result=None,
            section_fig=self._section_fig, panels="properties",
        )
        self._props_canvas.draw_idle()
        self._section_canvas.draw_idle()
        self._props_axes = props_axes

        return body

    # ── Results tab ───────────────────────────────────────────────────

    def _build_results_tab(
        self, model: StructuralModel, elem, result, multi_result,
    ) -> QWidget:
        """Build the Results tab: local case/combo selector + N/V/M
        diagrams + crosshair readout + end-force table.

        Pre-solve / failed / unsolved selections paint a placeholder and
        leave **zero** data lines on the diagrams figure — never stale.
        """
        from PyQt6.QtGui import QFont
        from ..gui_common.results_view import (
            case_combo_entries, resolve_view,
        )
        from .element_graphics import draw_element_detail

        body = QWidget(self)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)

        # ── Local case/combo selector ──
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Case / Combination:"))
        self._results_combo = QComboBox(body)
        # Always store the RAW identifier in userData so we never
        # substring-match the display label (avoiding the legacy bug
        # where "COMB1  [comb]" leaked into the internal key).
        entries = case_combo_entries(model, multi_result)
        for label, raw_name in entries:
            self._results_combo.addItem(label, raw_name)
        if entries:
            idx = self._results_combo.findData(self._results_selection)
            if idx < 0:
                # The saved selection isn't available anymore — fall
                # back to DEFAULT or the first entry.
                idx = self._results_combo.findData("DEFAULT")
                if idx < 0:
                    idx = 0
                self._results_selection = self._results_combo.itemData(idx)
            self._results_combo.setCurrentIndex(idx)
            self._results_combo.currentIndexChanged.connect(
                self._on_results_selection_changed
            )
        sel_row.addWidget(self._results_combo, stretch=1)
        layout.addLayout(sel_row)

        # ── Status placeholder ──
        # Resolve the selection to a result (None if pre-solve / failed /
        # combination needs solve).  status_msg drives the placeholder
        # QLabel shown above the diagrams figure.
        if multi_result is not None:
            resolved, status_msg = resolve_view(
                model, multi_result, self._results_selection,
            )
        else:
            # Backward-compat path for callers that construct the dialog
            # with just (model, elem_id, result) — no multi_result in
            # hand. Treat the passed-in ``result`` as the active view so
            # legacy tests and any non-MainWindow callers keep working.
            resolved = result
            status_msg = (
                ""
                if result is not None
                   and getattr(result, "status", None) == "ok"
                else "No analysis results yet. "
                     "Run analysis to show N/V/M diagrams."
            )
        ok_result = (
            resolved
            if resolved is not None
               and getattr(resolved, "status", None) == "ok"
            else None
        )
        if ok_result is None and not status_msg:
            status_msg = (
                "No analysis results yet. "
                "Run analysis to show N/V/M diagrams."
            )
        self._results_status = QLabel(status_msg, body)
        self._results_status.setStyleSheet("color: #555; font-style: italic;")
        self._results_status.setVisible(bool(status_msg))
        layout.addWidget(self._results_status)

        # ── Diagrams figure ──
        self._detail_fig = Figure(figsize=(7.0, 4.5), dpi=92)
        self._detail_fig.patch.set_facecolor("white")
        self._detail_canvas = FigureCanvasQTAgg(self._detail_fig)
        self._detail_canvas.setMinimumSize(560, 320)
        layout.addWidget(self._detail_canvas)
        results_axes = draw_element_detail(
            self._detail_fig, elem, model, ok_result, panels="diagrams",
        )
        self._detail_canvas.draw_idle()

        # Cache the three N/V/M sub-panel axes for the crosshair layer.
        self._ax_n = results_axes.ax_n
        self._ax_v = results_axes.ax_v
        self._ax_m = results_axes.ax_m
        self._diagram_axes_set = {self._ax_n, self._ax_v, self._ax_m}

        # Assemble the public _detail_axes dict that smoke tests read.
        # ``diagrams`` is the N (axial) axis (the test's anchor for the
        # pre-solve "Run analysis" text + post-solve line count) and the
        # sketch / fbd / section keys come from the Properties tab.
        from .element_graphics import ElementDetailAxes
        self._detail_axes = ElementDetailAxes(
            sketch=getattr(self, "_props_axes", None) and self._props_axes.get("sketch"),
            fbd=getattr(self, "_props_axes", None) and self._props_axes.get("fbd"),
            diagrams=self._ax_n,
            section=getattr(self, "_props_axes", None) and self._props_axes.get("section"),
        )
        self._detail_axes.ax_n = self._ax_n
        self._detail_axes.ax_v = self._ax_v
        self._detail_axes.ax_m = self._ax_m

        # ── Crosshair + readout strip ──
        f_local_raw = (
            _element_local_forces(ok_result, elem.id)
            if ok_result is not None else None
        )
        self._f_local_ref = (
            list(f_local_raw) if f_local_raw is not None else None
        )
        self._cursors = []
        self._maxima_annotations = []
        if self._f_local_ref is not None:
            _kw = dict(
                color="red", linestyle="--", linewidth=0.9,
                alpha=0.0, zorder=5,
            )
            self._cursors = [
                self._ax_n.axvline(x=0, **_kw),
                self._ax_v.axvline(x=0, **_kw),
                self._ax_m.axvline(x=0, **_kw),
            ]
            self._detail_canvas.mpl_connect(
                "motion_notify_event", self._on_diagram_motion,
            )
            self._detail_canvas.mpl_connect(
                "button_press_event", self._on_diagram_motion,
            )

        _mono = QFont("Courier")
        _mono.setPointSize(8)
        val_row = QHBoxLayout()
        self._lbl_x = QLabel("x: —  m")
        self._lbl_N = QLabel("N: —  kN")
        self._lbl_V = QLabel("V: —  kN")
        self._lbl_M = QLabel("M: —  kN·m")
        for lbl in (self._lbl_x, self._lbl_N, self._lbl_V, self._lbl_M):
            lbl.setFont(_mono)
            lbl.setMinimumWidth(115)
            lbl.setStyleSheet("border: 1px solid #ccc; padding: 2px 4px;")
            val_row.addWidget(lbl)
        val_row.addStretch()
        layout.addLayout(val_row)

        # Show Maxima checkbox — defaults ON and restores the persisted
        # per-dialog state so a case switch (which rebuilds this body)
        # doesn't reset the user's choice. Connecting before setChecked
        # means the restore fires _toggle_maxima, drawing the annotations
        # for the freshly-rendered (current-case) diagrams — so there are
        # never stale maxima values from the previous case.
        maxima_row = QHBoxLayout()
        self._show_maxima_cb = QCheckBox("Show Maxima")
        self._show_maxima_cb.setEnabled(self._f_local_ref is not None)
        self._show_maxima_cb.toggled.connect(self._toggle_maxima)
        maxima_row.addWidget(self._show_maxima_cb)
        maxima_row.addStretch()
        layout.addLayout(maxima_row)
        if self._f_local_ref is not None and self._show_maxima_on:
            self._show_maxima_cb.setChecked(True)

        # ── End-force table (only when a solved result is shown) ──
        if ok_result is not None:
            f_local = _element_local_forces(ok_result, elem.id)
            if f_local is not None:
                sep = QLabel("── End forces (local) ──")
                sep.setStyleSheet("font-weight: bold;")
                layout.addWidget(sep)
                ef_form = QFormLayout()
                layout.addLayout(ef_form)
                Ni, Vi, Mi, Nj, Vj, Mj = f_local
                if isinstance(elem, TrussElement2D):
                    ef_form.addRow("Node i:", QLabel(f"N = {Ni:+.4f} kN"))
                    ef_form.addRow("Node j:", QLabel(f"N = {Nj:+.4f} kN"))
                else:
                    ef_form.addRow(
                        "Node i:",
                        QLabel(
                            f"N = {Ni:+.4f} kN,  V = {Vi:+.4f} kN,  "
                            f"M = {Mi:+.4f} kN·m"
                        ),
                    )
                    ef_form.addRow(
                        "Node j:",
                        QLabel(
                            f"N = {Nj:+.4f} kN,  V = {Vj:+.4f} kN,  "
                            f"M = {Mj:+.4f} kN·m"
                        ),
                    )

        # Keep handles for re-resolution when the user changes the local
        # selector (cheap rebuild via _on_results_selection_changed).
        self._results_model_ref = model
        self._results_multi_ref = multi_result
        return body

    def _on_results_selection_changed(self, _idx: int) -> None:
        """Local case/combo combo changed — re-render the Results tab
        in place (the rest of the dialog is untouched, and the host's
        canvas active-case is NOT modified).  Uses currentData() so the
        raw identifier flows through; never the display label."""
        if self._results_combo is None:
            return
        new_sel = self._results_combo.currentData()
        if not new_sel or new_sel == self._results_selection:
            return
        self._results_selection = new_sel
        elem = next(
            (e for e in self._results_model_ref.elements
             if e.id == self._elem_id),
            None,
        )
        if elem is None:
            return
        new_widget = self._build_results_tab(
            self._results_model_ref, elem,
            None,  # result arg is ignored by the rebuild path —
                   # resolve_view() picks the right result for the
                   # dialog's selection from multi_result.
            self._results_multi_ref,
        )
        if self._tabs is not None and self._results_widget is not None:
            self._tabs.removeTab(self._TAB_RESULTS)
            self._results_widget.setParent(None)
            self._results_widget.deleteLater()
            self._tabs.insertTab(
                self._TAB_RESULTS, new_widget, "Results",
            )
            self._tabs.setCurrentIndex(self._TAB_RESULTS)
            self._results_widget = new_widget

    # ── Load Assignments tab ──────────────────────────────────────────

    def _build_loads_tab(
        self, model: StructuralModel, elem,
    ) -> QWidget:
        """Build the Load Assignments tab: loads table + Add button."""
        body = QWidget(self)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)

        loads_header = QLabel("Member loads on this element")
        loads_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(loads_header)

        # _loads_widget stays pointing at the QTableWidget itself (the
        # existing inspector tests treat it as the table); the entire
        # tab body is held by the QTabWidget via self._loads_tab_widget.
        self._loads_widget = self._build_loads_table(model, elem)
        layout.addWidget(self._loads_widget)

        # Bottom row: Add button (Edit / Delete are per-row in the table).
        add_row = QHBoxLayout()
        add_row.addStretch()
        self._add_load_btn = QPushButton("Add member load…")
        self._add_load_btn.setEnabled(
            self._host_add_member_load is not None
        )
        self._add_load_btn.clicked.connect(self._on_add_load_clicked)
        add_row.addWidget(self._add_load_btn)
        layout.addLayout(add_row)

        return body

    def _on_add_load_clicked(self) -> None:
        if self._host_add_member_load is None:
            return
        self._host_add_member_load(self._elem_id)

    def _on_edit_load_clicked(self, load_index: int) -> None:
        if self._host_edit_member_load is None:
            return
        self._host_edit_member_load(self._elem_id, load_index)

    # ── Loads table ──────────────────────────────────────────────────

    def _build_loads_table(
        self, model: StructuralModel, elem,
    ) -> QWidget:
        """Build the per-element member-loads table widget.

        Columns: # | Type | Direction | Magnitude | Position / Notes |
        Case | Edit | Delete.  Edit and Delete are routed through the
        host callbacks (``_host_edit_member_load`` /
        ``_host_delete_member_load``); when those callbacks are missing
        — e.g. in unit tests that construct the dialog directly — the
        buttons render disabled so the model is never mutated implicitly.
        """
        from .load_summary import format_element_loads

        rows = format_element_loads(model, elem)
        table = QTableWidget(0, 8)
        table.setHorizontalHeaderLabels([
            "#", "Type", "Direction", "Magnitude",
            "Position / Notes", "Case", "", "",
        ])
        # Spreadsheet copy: read-only / NoSelection table, so Ctrl+C and
        # right-click Copy fall back to the whole table (Case/Type/...
        # columns; the trailing Edit/Delete-button columns copy as empty).
        install_table_copy(table, include_headers=True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )
        table.setVerticalScrollMode(
            QTableWidget.ScrollMode.ScrollPerPixel,
        )
        header = table.horizontalHeader()
        for c in (0, 1, 2, 3, 5, 6, 7):
            header.setSectionResizeMode(
                c, QHeaderView.ResizeMode.ResizeToContents,
            )
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        if not rows:
            table.setRowCount(1)
            none_item = QTableWidgetItem("(no member loads)")
            none_item.setFlags(Qt.ItemFlag.NoItemFlags)
            table.setItem(0, 0, QTableWidgetItem(""))
            table.setSpan(0, 0, 1, 8)
            table.setItem(0, 0, none_item)
        else:
            table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                table.setItem(i, 0, QTableWidgetItem(str(row.index + 1)))
                t_item = QTableWidgetItem(row.type_label)
                t_item.setToolTip(row.meaning)
                table.setItem(i, 1, t_item)
                table.setItem(
                    i, 2, QTableWidgetItem(_direction_label_for_row(row)),
                )
                table.setItem(i, 3, QTableWidgetItem(row.magnitude))
                table.setItem(i, 4, QTableWidgetItem(row.position))
                # Case column: dim "—" placeholder for the default case
                # so legacy data doesn't visually shout a case tag it
                # never had.
                case_text = (
                    "—" if row.load_case == "DEFAULT" else row.load_case
                )
                case_item = QTableWidgetItem(case_text)
                if row.load_case == "DEFAULT":
                    case_item.setForeground(QColor("#888"))
                table.setItem(i, 5, case_item)
                # Capture row.index at lambda definition time so later
                # additions / removals don't change the button targets.
                load_idx = row.index
                edit_btn = QPushButton("Edit")
                edit_btn.setEnabled(
                    self._host_edit_member_load is not None
                )
                edit_btn.clicked.connect(
                    lambda _checked=False, idx=load_idx:
                    self._on_edit_load_clicked(idx)
                )
                table.setCellWidget(i, 6, edit_btn)
                del_btn = QPushButton("Delete")
                del_btn.setEnabled(
                    self._host_delete_member_load is not None
                )
                del_btn.clicked.connect(
                    lambda _checked=False, idx=load_idx:
                    self._on_delete_load_clicked(idx)
                )
                table.setCellWidget(i, 7, del_btn)
        # Height: always reserve at least 3 rows so a single-load
        # element still reads as a table (not one cramped row), and cap
        # at 6 visible rows — beyond that the vertical scrollbar (set
        # above) takes over so many loads stay inspectable.
        n_visible = max(3, min(max(len(rows), 1), 6))
        row_h = table.verticalHeader().defaultSectionSize()
        header_h = table.horizontalHeader().height()
        table_h = header_h + (row_h * n_visible) + 4
        table.setMinimumHeight(table_h)
        table.setMaximumHeight(table_h)
        return table

    def _on_delete_load_clicked(self, load_index: int) -> None:
        """Forward a Delete-button click to the host callback.

        Guarded against missing host (defensive — Delete buttons are
        also disabled in that case, but a stale slot connection could
        in principle fire after the host went away)."""
        if self._host_delete_member_load is None:
            return
        self._host_delete_member_load(self._elem_id, load_index)

    def refresh_loads_only(
        self, model: StructuralModel, result=None,
    ) -> None:
        """Re-render just the loads table inside the Load Assignments
        tab without rebuilding the Properties / Results tabs (which
        would flash the diagrams figure). The host calls this after a
        successful :class:`DeleteMemberLoadCmd` so the row disappears
        immediately while the diagrams stay put."""
        elem = next(
            (e for e in model.elements if e.id == self._elem_id), None,
        )
        if elem is None:
            # Element vanished (e.g. cascade delete via another path) —
            # close the inspector so the user isn't staring at stale data.
            self.close()
            return
        # Rebuild the Loads tab body in place; full set_target on
        # fallback paths.
        if self._tabs is None or self._loads_tab_widget is None:
            self.set_target(model, self._elem_id, result)
            return
        # removeTab on the active tab makes QTabWidget auto-switch focus
        # elsewhere; save and restore the current index so editing /
        # deleting a row from the Load Assignments tab keeps the user
        # on that tab.
        current_idx = self._tabs.currentIndex()
        new_widget = self._build_loads_tab(model, elem)
        self._tabs.removeTab(self._TAB_LOADS)
        self._loads_tab_widget.setParent(None)
        self._loads_tab_widget.deleteLater()
        self._tabs.insertTab(
            self._TAB_LOADS, new_widget, "Load Assignments",
        )
        self._loads_tab_widget = new_widget
        if 0 <= current_idx < self._tabs.count():
            self._tabs.setCurrentIndex(current_idx)

    # ── Interactive handlers ──────────────────────────────────────────

    def _on_diagram_motion(self, event) -> None:
        """Synchronise the crosshair across N/V/M axes and update the
        readout strip.  Bound to motion_notify_event and button_press_event.
        """
        if not self._cursors or self._f_local_ref is None:
            return
        if event.inaxes not in self._diagram_axes_set:
            for c in self._cursors:
                c.set_alpha(0.0)
            for lbl, base in (
                (self._lbl_x, "x"),
                (self._lbl_N, "N"),
                (self._lbl_V, "V"),
                (self._lbl_M, "M"),
            ):
                lbl.setText(f"{base}: —")
            self._detail_canvas.draw_idle()
            return
        if event.xdata is None:
            return
        x = max(0.0, min(self._L, float(event.xdata)))
        for c in self._cursors:
            c.set_xdata([x, x])
            c.set_alpha(0.7)
        n_val = self._internal_force_at(
            self._elem_ref, self._ni_ref, self._nj_ref,
            self._f_local_ref, "axial", x)
        v_val = self._internal_force_at(
            self._elem_ref, self._ni_ref, self._nj_ref,
            self._f_local_ref, "shear", x)
        m_val = self._internal_force_at(
            self._elem_ref, self._ni_ref, self._nj_ref,
            self._f_local_ref, "moment", x)
        self._lbl_x.setText(f"x: {x:.3f} m")
        self._lbl_N.setText(f"N: {n_val:.3f} kN"
                            if n_val is not None else "N: —")
        self._lbl_V.setText(f"V: {v_val:.3f} kN"
                            if v_val is not None else "V: —")
        self._lbl_M.setText(f"M: {m_val:.3f} kN·m"
                            if m_val is not None else "M: —")
        self._detail_canvas.draw_idle()

    def _toggle_maxima(self, checked: bool) -> None:
        """Add / remove absolute-peak annotations on each diagram axis.

        Records the user's choice in ``self._show_maxima_on`` so it
        survives a body rebuild (case switch / refresh)."""
        self._show_maxima_on = bool(checked)
        for ann in self._maxima_annotations:
            try:
                ann.remove()
            except ValueError:
                pass
        self._maxima_annotations.clear()
        if checked and self._f_local_ref is not None:
            for kind, ax in (
                ("axial",  self._ax_n),
                ("shear",  self._ax_v),
                ("moment", self._ax_m),
            ):
                xs, ys = self._sample_internal_force(
                    self._elem_ref, self._ni_ref, self._nj_ref,
                    self._f_local_ref, kind, n_samples=101,
                )
                if xs is None or ys is None:
                    continue
                peak_i = max(range(len(ys)), key=lambda i: abs(ys[i]))
                ann = ax.annotate(
                    f"{ys[peak_i]:.3g}",
                    xy=(xs[peak_i], ys[peak_i]),
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=7, color="#222", ha="center",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#444"),
                )
                self._maxima_annotations.append(ann)
        self._detail_canvas.draw_idle()


def _direction_label_for_row(row) -> str:
    """Direction column text for the Load Assignments table.

    Mechanical loads carry a direction string; thermal loads have none
    (their effect is a strain field, not a force vector) — show an em
    dash so the column reads cleanly."""
    if row.direction:
        return row.direction
    return "—"

    # ── Interactive handlers ──────────────────────────────────────────

    def _on_diagram_motion(self, event) -> None:
        """Synchronise the crosshair cursor across N/V/M axes and update
        the readout strip.  Bound to both motion_notify and button_press."""
        if not self._cursors or self._f_local_ref is None:
            return
        if event.inaxes not in self._diagram_axes_set:
            for c in self._cursors:
                c.set_alpha(0.0)
            for lbl, base in (
                (self._lbl_x, "x"),
                (self._lbl_N, "N"),
                (self._lbl_V, "V"),
                (self._lbl_M, "M"),
            ):
                lbl.setText(f"{base}: —")
            self._detail_canvas.draw_idle()
            return

        if event.xdata is None:
            return
        x = max(0.0, min(self._L, float(event.xdata)))
        for c in self._cursors:
            c.set_xdata([x, x])
            c.set_alpha(0.7)

        n_val = self._internal_force_at(
            self._elem_ref, self._ni_ref, self._nj_ref,
            self._f_local_ref, "axial", x)
        v_val = self._internal_force_at(
            self._elem_ref, self._ni_ref, self._nj_ref,
            self._f_local_ref, "shear", x)
        m_val = self._internal_force_at(
            self._elem_ref, self._ni_ref, self._nj_ref,
            self._f_local_ref, "moment", x)

        self._lbl_x.setText(f"x: {x:.3f} m")
        self._lbl_N.setText(f"N: {n_val:.3f} kN"
                            if n_val is not None else "N: —")
        self._lbl_V.setText(f"V: {v_val:.3f} kN"
                            if v_val is not None else "V: —")
        self._lbl_M.setText(f"M: {m_val:.3f} kN·m"
                            if m_val is not None else "M: —")
        self._detail_canvas.draw_idle()

    def _toggle_maxima(self, state: int) -> None:
        """Add / remove absolute-peak annotations on each diagram axis."""
        for ann in self._maxima_annotations:
            try:
                ann.remove()
            except ValueError:
                pass
        self._maxima_annotations.clear()

        if state and self._f_local_ref is not None:
            for kind, ax in (
                ("axial",  self._ax_n),
                ("shear",  self._ax_v),
                ("moment", self._ax_m),
            ):
                xs, ys = self._sample_internal_force(
                    self._elem_ref, self._ni_ref, self._nj_ref,
                    self._f_local_ref, kind, n_samples=101,
                )
                if xs is None or ys is None:
                    continue
                peak_i = max(range(len(ys)), key=lambda i: abs(ys[i]))
                ann = ax.annotate(
                    f"{ys[peak_i]:.3g}",
                    xy=(xs[peak_i], ys[peak_i]),
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=7, color="#222", ha="center",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#444"),
                )
                self._maxima_annotations.append(ann)

        self._detail_canvas.draw_idle()


def _element_local_forces(result, elem_id: int):
    member = getattr(result, "member_results", None) or {}
    entry = member.get(elem_id)
    if not entry:
        return None
    f = entry.get("f_local")
    if f is None or len(f) < 6:
        return None
    return tuple(float(v) for v in f[:6])


class FineNodeDialog(_ModalDialog):
    """Add a node at typed (x, y) coordinates — alternative to canvas click."""

    def __init__(self, parent, *, model: StructuralModel) -> None:
        self._model = model
        super().__init__(parent, "Add node at coordinates")

    def _build_body(self, body: QWidget) -> None:
        form = QFormLayout(body)
        self._x_entry = QLineEdit(body)
        self._y_entry = QLineEdit(body)
        self._x_entry.setText("0.0")
        self._y_entry.setText("0.0")
        form.addRow("X (m):", self._x_entry)
        form.addRow("Y (m):", self._y_entry)
        hint = QLabel(
            "The node is created via the same Add-Node command used by\n"
            "canvas clicks — undo / duplicate detection still apply.",
            body,
        )
        hint.setWordWrap(True)
        form.addRow(hint)

    def _accept(self) -> tuple[float, float]:
        x = parse_float(self._x_entry.text(), "X")
        y = parse_float(self._y_entry.text(), "Y")
        return (x, y)


class BuildingWizardDialog(_ModalDialog):
    """Generate a 2D portal-frame building from typed dimensions.

    On accept, returns a fresh :class:`StructuralModel` containing the new
    geometry — materials and sections are copied from the source model so
    the user's section/material library is preserved. The host applies it
    via ``ReplaceModelCmd`` so the whole wizard is undoable in one step.
    """

    def __init__(self, parent, *, model: StructuralModel) -> None:
        self._source_model = model
        if not model.sections:
            raise ValueError(
                "No sections defined — add a section before running the "
                "building wizard."
            )
        super().__init__(parent, "Building wizard")

    def _build_body(self, body: QWidget) -> None:
        form = QFormLayout(body)

        self._stories = QSpinBox(body)
        self._stories.setRange(1, 30)
        self._stories.setValue(3)

        self._story_h = QDoubleSpinBox(body)
        self._story_h.setRange(0.5, 50.0)
        self._story_h.setDecimals(2)
        self._story_h.setSingleStep(0.5)
        self._story_h.setSuffix(" m")
        self._story_h.setValue(3.0)

        self._bays = QSpinBox(body)
        self._bays.setRange(1, 30)
        self._bays.setValue(3)

        self._bay_w = QDoubleSpinBox(body)
        self._bay_w.setRange(0.5, 50.0)
        self._bay_w.setDecimals(2)
        self._bay_w.setSingleStep(0.5)
        self._bay_w.setSuffix(" m")
        self._bay_w.setValue(5.0)

        self._sec_combo = QComboBox(body)
        for sid in sorted(self._source_model.sections):
            s = self._source_model.sections[sid]
            mat = self._source_model.materials.get(s.material_id)
            mat_name = (
                mat.name if (mat and mat.name) else f"mat {s.material_id}"
            )
            label = f"{s.name or 'unnamed'} / {mat_name}"
            self._sec_combo.addItem(label, sid)

        self._fixed_base = QCheckBox(
            "Fixed (ux, uy, rz) supports at every ground node", body,
        )
        self._fixed_base.setChecked(True)

        form.addRow("Stories:", self._stories)
        form.addRow("Story height:", self._story_h)
        form.addRow("Bays (X direction):", self._bays)
        form.addRow("Bay width:", self._bay_w)
        form.addRow("Section (cols & beams):", self._sec_combo)
        form.addRow("", self._fixed_base)

        hint = QLabel(
            "Generates a planar moment-frame: vertical columns at each "
            "column line, horizontal beams at every floor above ground. "
            "Replaces the current model — materials and sections are "
            "preserved. Use Undo (Ctrl+Z) to restore.",
            body,
        )
        hint.setWordWrap(True)
        form.addRow(hint)

    def _accept(self) -> StructuralModel:
        stories = int(self._stories.value())
        h = float(self._story_h.value())
        bays = int(self._bays.value())
        bw = float(self._bay_w.value())
        sid = self._sec_combo.currentData()
        if sid is None or sid not in self._source_model.sections:
            raise ValueError("No valid section selected.")

        section = self._source_model.sections[sid]
        mat = self._source_model.materials.get(section.material_id)
        if mat is None:
            raise ValueError(
                f"Section {sid} references missing material "
                f"{section.material_id}."
            )

        m = StructuralModel(title=f"Building {stories}s × {bays}b")
        m.materials = dict(self._source_model.materials)
        m.sections = dict(self._source_model.sections)

        # (bays+1) × (stories+1) node grid: node_grid[j][i] is the node id
        # at column line i (0..bays) on floor j (0..stories, j=0 is ground).
        node_grid: list[list[int]] = []
        nid = 1
        for j in range(stories + 1):
            row: list[int] = []
            for i in range(bays + 1):
                m.nodes[nid] = Node(nid, i * bw, j * h)
                row.append(nid)
                nid += 1
            node_grid.append(row)

        eid = 1
        # Columns: each column line, each story.
        for i in range(bays + 1):
            for j in range(stories):
                m.elements.append(FrameElement2D(
                    id=eid,
                    node_i=node_grid[j][i],
                    node_j=node_grid[j + 1][i],
                    E=mat.E, A=section.A, I=section.I,
                    alpha=mat.alpha, rho=mat.density,
                    depth=section.depth,
                    section_id=section.id,
                ))
                eid += 1
        # Beams: every floor above ground, between adjacent columns.
        for j in range(1, stories + 1):
            for i in range(bays):
                m.elements.append(FrameElement2D(
                    id=eid,
                    node_i=node_grid[j][i],
                    node_j=node_grid[j][i + 1],
                    E=mat.E, A=section.A, I=section.I,
                    alpha=mat.alpha, rho=mat.density,
                    depth=section.depth,
                    section_id=section.id,
                ))
                eid += 1

        if self._fixed_base.isChecked():
            for nid_base in node_grid[0]:
                m.supports[nid_base] = Support(
                    nid_base, ux=True, uy=True, rz=True,
                )

        return m


# ── analysis settings (v0.9.0) ──


class AnalysisSettingsDialog(_ModalDialog):
    """Edit the model's analysis settings.

    v0.9.0 exposes a single switch — "Include self-weight in static
    analysis". Gravity is fixed at g = 9.81 m/s² in global -Y; future
    versions may expose those as user-editable controls.
    """

    def __init__(self, parent, *, include_self_weight: bool) -> None:
        self._initial = bool(include_self_weight)
        super().__init__(parent, "Analysis settings")

    def _build_body(self, body: QWidget) -> None:
        layout = QVBoxLayout(body)
        self._sw_check = QCheckBox(
            "Include self-weight in static analysis", body,
        )
        self._sw_check.setChecked(self._initial)
        layout.addWidget(self._sw_check)
        note = QLabel(
            "Gravity acts in global −Y at g = 9.81 m/s².\n"
            "Future versions will let you change these.",
            body,
        )
        note.setStyleSheet("color: #555; font-size: 9pt;")
        note.setWordWrap(True)
        layout.addWidget(note)

    def _accept(self) -> dict:
        return {"include_self_weight": bool(self._sw_check.isChecked())}


# ── modal mass source (v0.25 — PR #40) ──────────────────────────────────


class ModalMassSourceDialog(_ModalDialog):
    """Configure which contributions are assembled into the modal mass matrix.

    Three checkboxes (self-mass, joint masses, load-case mass) plus a
    table for per-case multipliers.  A live label warns about potential
    double-counting while the OK button remains enabled (per the brief:
    warn, not block).

    ``result_value`` on accept is the new :class:`ModalMassSource`.
    """

    def __init__(self, parent, *, model: StructuralModel) -> None:
        from ..model import ModalMassSource
        self._model = model
        self._src = getattr(model, "modal_mass_source", ModalMassSource())
        super().__init__(parent, "Modal mass source settings")
        self.resize(480, 420)

    def _build_body(self, body: QWidget) -> None:
        from PyQt6.QtWidgets import (
            QFormLayout, QGroupBox, QTableWidget, QTableWidgetItem,
            QHeaderView, QScrollArea, QSizePolicy,
        )
        from PyQt6.QtCore import Qt

        v = QVBoxLayout(body)
        v.setContentsMargins(4, 4, 4, 4)

        # ── Three checkboxes ──────────────────────────────────────────
        n_jm = len(self._model.joint_masses) if hasattr(self._model, "joint_masses") else 0
        self._cb_self = QCheckBox(
            "Include element self-mass from material density", body,
        )
        self._cb_self.setChecked(self._src.include_self_mass)

        self._cb_jm = QCheckBox(
            f"Include nodal / joint masses  (current model: {n_jm} "
            f"{'entry' if n_jm == 1 else 'entries'})",
            body,
        )
        self._cb_jm.setChecked(self._src.include_joint_masses)

        self._cb_lc = QCheckBox(
            "Include selected load cases as mass  (|Fy|/g added to "
            "both translational DOFs)",
            body,
        )
        self._cb_lc.setChecked(self._src.include_load_cases)

        v.addWidget(self._cb_self)
        v.addWidget(self._cb_jm)
        v.addWidget(self._cb_lc)

        # ── Case-factor table ─────────────────────────────────────────
        cases = [
            name for name in sorted(self._model.load_cases)
            if name != "SUM_ALL"
        ]
        self._table = QTableWidget(len(cases), 2, body)
        self._table.setHorizontalHeaderLabels(["Case", "Multiplier"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch,
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents,
        )
        self._table.verticalHeader().setVisible(False)
        for row, name in enumerate(cases):
            item_name = QTableWidgetItem(name)
            item_name.setFlags(item_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            factor = self._src.load_case_factors.get(name, 0.0)
            item_mult = QTableWidgetItem(f"{factor:g}")
            self._table.setItem(row, 0, item_name)
            self._table.setItem(row, 1, item_mult)
        self._table.setEnabled(self._cb_lc.isChecked())
        self._case_names = cases
        v.addWidget(self._table)

        # ── Warning label ─────────────────────────────────────────────
        self._warn_label = QLabel("", body)
        self._warn_label.setWordWrap(True)
        self._warn_label.setStyleSheet(
            "color: #a06000; font-size: 9pt; padding: 2px 0;"
        )
        v.addWidget(self._warn_label)

        # ── Wire signals ──────────────────────────────────────────────
        self._cb_self.toggled.connect(self._refresh_warnings)
        self._cb_jm.toggled.connect(self._refresh_warnings)
        self._cb_lc.toggled.connect(self._on_lc_toggled)
        self._table.itemChanged.connect(self._refresh_warnings)

        self._refresh_warnings()

    def _on_lc_toggled(self, checked: bool) -> None:
        self._table.setEnabled(checked)
        self._refresh_warnings()

    def _read_factors(self, *, strict: bool = False) -> dict[str, float]:
        factors: dict[str, float] = {}
        for row, name in enumerate(self._case_names):
            item = self._table.item(row, 1)
            if item is None:
                continue
            text = item.text().strip()
            try:
                val = float(text) if text else 0.0
            except ValueError:
                if strict:
                    raise ValueError(
                        f"Multiplier for case {name!r} is not a valid number: {text!r}"
                    )
                val = 0.0
            factors[name] = val
        return factors

    def _refresh_warnings(self) -> None:
        msg = self._double_count_message()
        self._warn_label.setText(msg)

    def _double_count_message(self) -> str:
        if not self._cb_self.isChecked():
            return ""
        if not self._cb_lc.isChecked():
            return ""
        sw_enabled = getattr(self._model, "include_self_weight", False)
        if not sw_enabled:
            return ""
        sw_case = getattr(self._model, "self_weight_case", "DEFAULT")
        mult = self._read_factors().get(sw_case, 0.0)
        if mult > 0.0:
            return (
                f"Note: Generated self-weight in {sw_case!r} is automatically "
                "excluded from mass conversion because element density already "
                "provides self-mass. Only manually assigned loads in that case "
                "contribute additional mass."
            )
        return ""

    def _accept(self):
        from ..model import ModalMassSource
        try:
            factors = self._read_factors(strict=True) if self._cb_lc.isChecked() else {}
        except ValueError as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid multiplier", str(exc))
            return None
        try:
            src = ModalMassSource(
                include_self_mass=self._cb_self.isChecked(),
                include_joint_masses=self._cb_jm.isChecked(),
                include_load_cases=self._cb_lc.isChecked(),
                load_case_factors=factors,
            )
        except ValueError as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid mass source", str(exc))
            return None
        return src


# ── batch load dialogs (v0.26 — PR #41) ─────────────────────────────────


class BatchMemberLoadDialog(_ModalDialog):
    """Define one member-load spec to be applied to N selected elements.

    Mirrors :class:`MemberLoadDialog`'s field set but emits a spec dict
    (not a load instance) because the same spec turns into different
    per-element loads — most importantly, batch point loads use a
    relative position ``ratio ∈ [0, 1]`` that is converted to the
    correct absolute ``a = ratio * L`` per element in the app handler.

    Compatibility is decided ONCE at open time from the selection's
    element kinds:
      - all frame   → mechanical + thermal (uniform + gradient)
      - all truss   → thermal uniform only (mechanical + gradient blocked)
      - mixed       → OK button disabled; warning label names the rule.

    ``result_value`` on accept is a dict; see :meth:`_accept` for keys.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        model: StructuralModel,
        element_ids: "frozenset[int] | set[int]",
    ) -> None:
        if len(element_ids) < 2:
            raise ValueError(
                "Batch member-load dialog requires two or more selected "
                f"elements (got {len(element_ids)})."
            )
        self._model = model
        self._element_ids = frozenset(element_ids)
        # Classify the selection up-front so _build_body can disable
        # incompatible radios cleanly.
        elements = [e for e in model.elements if e.id in self._element_ids]
        missing = self._element_ids - {e.id for e in elements}
        if missing:
            raise ValueError(
                f"Elements not in model: {sorted(missing)}"
            )
        self._elements = elements
        self._has_frame = any(isinstance(e, FrameElement2D) for e in elements)
        self._has_truss = any(isinstance(e, TrussElement2D) for e in elements)
        self._is_mixed = self._has_frame and self._has_truss
        self._all_truss = self._has_truss and not self._has_frame
        super().__init__(
            parent,
            f"Batch member load — {len(self._element_ids)} elements",
        )

    def _ok_button(self):
        if not hasattr(self, "_cached_ok_button"):
            cached = None
            for child in self.findChildren(QDialogButtonBox):
                cached = child.button(QDialogButtonBox.StandardButton.Ok)
                break
            self._cached_ok_button = cached
        return self._cached_ok_button

    def _build_body(self, body: QWidget) -> None:
        from PyQt6.QtWidgets import QButtonGroup, QRadioButton, QFormLayout
        v = QVBoxLayout(body)

        # Heading + selection summary.
        v.addWidget(QLabel(
            f"<b>Apply the same member load to "
            f"{len(self._element_ids)} selected elements.</b>",
            body,
        ))
        kinds: list[str] = []
        if self._has_frame:
            kinds.append(f"{sum(isinstance(e, FrameElement2D) for e in self._elements)} frame")
        if self._has_truss:
            kinds.append(f"{sum(isinstance(e, TrussElement2D) for e in self._elements)} truss")
        v.addWidget(QLabel(f"Selection: {', '.join(kinds)}.", body))

        # If mixed, block at the dialog: no field choice can make a
        # mixed batch valid (every member-load type is rejected by at
        # least one of the two element kinds).
        if self._is_mixed:
            warn = QLabel(
                "Batch member loads require a uniform element type. "
                "The current selection mixes frame and truss elements.\n"
                "Tip: Use Selection → Keep Frames Only, Keep Trusses Only, "
                "or Select Group first.",
                body,
            )
            warn.setWordWrap(True)
            warn.setStyleSheet(
                "color: #b00020; font-weight: bold; padding: 6px;"
            )
            v.addWidget(warn)
            # Build no further fields; disable OK after the buttonbox
            # exists (deferred via singleShot since _ok_button can only
            # find the buttonbox after _build_body returns).
            QTimer.singleShot(0, lambda: self._ok_button() and self._ok_button().setEnabled(False))
            return

        # ── Category radios (Mechanical / Thermal) ────────────────────
        # Mechanical is disabled for all-truss selections (mirrors the
        # single-element MemberLoadDialog rule at dialogs.py:1679).
        v.addWidget(QLabel("Load category:", body))
        self._cat_group = QButtonGroup(body)
        self._rb_mech = QRadioButton("Mechanical", body)
        self._rb_therm = QRadioButton("Thermal", body)
        if self._all_truss:
            self._rb_mech.setEnabled(False)
            self._rb_mech.setToolTip(
                "Truss elements support thermal loads only — UDL / point "
                "loads require frame bending DOFs."
            )
            self._rb_therm.setChecked(True)
        else:
            self._rb_mech.setChecked(True)
        self._cat_group.addButton(self._rb_mech)
        self._cat_group.addButton(self._rb_therm)
        v.addWidget(self._rb_mech)
        v.addWidget(self._rb_therm)
        self._rb_mech.toggled.connect(self._refresh)
        self._rb_therm.toggled.connect(self._refresh)

        # ── Mechanical subform: UDL / Point + Local/Global/Gravity ────
        self._mech_widget = QWidget(body)
        mv = QVBoxLayout(self._mech_widget)
        mv.setContentsMargins(0, 4, 0, 0)
        mv.addWidget(QLabel("Type:", self._mech_widget))
        self._kind_group = QButtonGroup(self._mech_widget)
        self._rb_udl = QRadioButton("Uniform Distributed Load", self._mech_widget)
        self._rb_point = QRadioButton("Point Load (relative position)", self._mech_widget)
        self._kind_group.addButton(self._rb_udl)
        self._kind_group.addButton(self._rb_point)
        self._rb_udl.setChecked(True)
        mv.addWidget(self._rb_udl)
        mv.addWidget(self._rb_point)
        self._rb_udl.toggled.connect(self._refresh)
        self._rb_point.toggled.connect(self._refresh)
        mv.addWidget(QLabel("Direction:", self._mech_widget))
        self._coord_group = QButtonGroup(self._mech_widget)
        self._rb_local = QRadioButton("Local", self._mech_widget)
        self._rb_global = QRadioButton("Global", self._mech_widget)
        self._rb_gravity = QRadioButton("Gravity (global -Y)", self._mech_widget)
        self._rb_local.setChecked(True)
        for rb in (self._rb_local, self._rb_global, self._rb_gravity):
            self._coord_group.addButton(rb)
            mv.addWidget(rb)
            rb.toggled.connect(self._refresh)
        v.addWidget(self._mech_widget)

        # ── Thermal subform ───────────────────────────────────────────
        self._therm_widget = QWidget(body)
        tv = QVBoxLayout(self._therm_widget)
        tv.setContentsMargins(0, 4, 0, 0)
        tv.addWidget(QLabel("Thermal type:", self._therm_widget))
        self._therm_group = QButtonGroup(self._therm_widget)
        self._rb_t_uniform = QRadioButton("Uniform ΔT", self._therm_widget)
        self._rb_t_gradient = QRadioButton("Top / bottom gradient", self._therm_widget)
        if self._all_truss:
            self._rb_t_gradient.setEnabled(False)
            self._rb_t_gradient.setToolTip(
                "Thermal gradient requires frame bending DOFs."
            )
        self._therm_group.addButton(self._rb_t_uniform)
        self._therm_group.addButton(self._rb_t_gradient)
        self._rb_t_uniform.setChecked(True)
        tv.addWidget(self._rb_t_uniform)
        tv.addWidget(self._rb_t_gradient)
        self._rb_t_uniform.toggled.connect(self._refresh)
        self._rb_t_gradient.toggled.connect(self._refresh)
        v.addWidget(self._therm_widget)

        # ── Numeric fields container ──────────────────────────────────
        self._field_container = QWidget(body)
        self._field_form = QFormLayout(self._field_container)
        v.addWidget(self._field_container)
        self._fields: dict[str, QLineEdit] = {}

        # ── Load case combo (mirrors MemberLoadDialog) ────────────────
        v.addWidget(QLabel("Load case:", body))
        self._case_combo = _make_load_case_combo(body, "DEFAULT", model=self._model)
        v.addWidget(self._case_combo)

        self._refresh()

    def _current_category(self) -> str:
        return "thermal" if self._rb_therm.isChecked() else "mechanical"

    def _current_coord(self) -> str:
        if self._rb_gravity.isChecked():
            return "gravity"
        if self._rb_global.isChecked():
            return "global"
        return "local"

    def _refresh(self) -> None:
        # Clear the field container.
        while self._field_form.count():
            item = self._field_form.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self._fields = {}

        cat = self._current_category()
        self._mech_widget.setVisible(cat == "mechanical")
        self._therm_widget.setVisible(cat == "thermal")

        if cat == "mechanical":
            cs = self._current_coord()
            kind = "point" if self._rb_point.isChecked() else "udl"
            if kind == "udl":
                if cs == "gravity":
                    self._add_field("qg (kN/m, +ve downward)", "wy")
                else:
                    xname = "qX (kN/m, global X)" if cs == "global" else "wx (kN/m, local x)"
                    yname = "qY (kN/m, global Y)" if cs == "global" else "wy (kN/m, local y)"
                    self._add_field(xname, "wx")
                    self._add_field(yname, "wy")
            else:
                if cs == "gravity":
                    self._add_field("Pg (kN, +ve downward)", "py")
                else:
                    xname = "PX (kN, global X)" if cs == "global" else "Px (kN, local x)"
                    yname = "PY (kN, global Y)" if cs == "global" else "Py (kN, local y)"
                    self._add_field(xname, "px")
                    self._add_field(yname, "py")
                # Relative position is the only mode in V1 — the value
                # is converted to absolute a = ratio * L per element by
                # the app-layer handler.
                self._add_field(
                    "Position ratio (0.0 = start, 1.0 = end)",
                    "ratio",
                )
                self._fields["ratio"].setText("0.5")
        else:
            tkind = "gradient" if self._rb_t_gradient.isChecked() else "uniform"
            if tkind == "uniform":
                self._add_field("ΔT (°C)", "delta_T")
            else:
                self._add_field("t_top (°C)", "t_top")
                self._add_field("t_bottom (°C)", "t_bottom")

    def _add_field(self, label: str, key: str) -> None:
        le = QLineEdit(self._field_container)
        self._field_form.addRow(label, le)
        self._fields[key] = le

    def _accept(self) -> dict:
        if self._is_mixed:
            # Defensive — OK is disabled in the mixed branch, but reject
            # programmatic accepts too rather than emitting garbage.
            raise ValueError("Selection mixes frame and truss elements.")
        load_case = _normalize_load_case(self._case_combo.currentText())
        cat = self._current_category()
        if cat == "mechanical":
            cs = self._current_coord()
            kind = "point" if self._rb_point.isChecked() else "udl"
            if kind == "udl":
                if cs == "gravity":
                    wy = parse_float(self._fields["wy"].text(), "qg")
                    return {
                        "kind": "udl", "coord_system": cs, "load_case": load_case,
                        "wx": 0.0, "wy": wy,
                    }
                xname = "qX" if cs == "global" else "wx"
                yname = "qY" if cs == "global" else "wy"
                wx = parse_float(self._fields["wx"].text(), xname, allow_blank=True) or 0.0
                wy = parse_float(self._fields["wy"].text(), yname)
                return {
                    "kind": "udl", "coord_system": cs, "load_case": load_case,
                    "wx": wx, "wy": wy,
                }
            # Point load (relative position).
            if cs == "gravity":
                py = parse_float(self._fields["py"].text(), "Pg")
                px = 0.0
            else:
                xname = "PX" if cs == "global" else "Px"
                yname = "PY" if cs == "global" else "Py"
                px = parse_float(self._fields["px"].text(), xname, allow_blank=True) or 0.0
                py = parse_float(self._fields["py"].text(), yname)
            ratio = parse_float(self._fields["ratio"].text(), "Position ratio")
            if ratio < 0.0 or ratio > 1.0:
                raise ValueError(
                    f"Position ratio must lie within [0.0, 1.0]; got {ratio}."
                )
            return {
                "kind": "point", "coord_system": cs, "load_case": load_case,
                "px": px, "py": py, "ratio": ratio,
            }
        # Thermal.
        tkind = "gradient" if self._rb_t_gradient.isChecked() else "uniform"
        if tkind == "uniform":
            dT = parse_float(self._fields["delta_T"].text(), "ΔT")
            return {
                "kind": "thermal_uniform", "load_case": load_case,
                "delta_T": dT,
            }
        # Gradient — disallowed on truss; OK is enabled only when all-frame.
        return {
            "kind": "thermal_gradient", "load_case": load_case,
            "t_top": parse_float(self._fields["t_top"].text(), "t_top"),
            "t_bottom": parse_float(self._fields["t_bottom"].text(), "t_bottom"),
        }


class BatchNodalLoadDialog(_ModalDialog):
    """Define one nodal-load row to be appended to each of N selected nodes.

    Appends — does NOT overwrite existing nodal loads on the target
    nodes. ``result_value`` on accept is a dict: ``{fx, fy, mz, case}``.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        model: StructuralModel,
        node_ids: "frozenset[int] | set[int]",
    ) -> None:
        if len(node_ids) < 2:
            raise ValueError(
                "Batch nodal-load dialog requires two or more selected "
                f"nodes (got {len(node_ids)})."
            )
        self._model = model
        self._node_ids = frozenset(node_ids)
        missing = self._node_ids - set(model.nodes)
        if missing:
            raise ValueError(f"Nodes not in model: {sorted(missing)}")
        super().__init__(
            parent,
            f"Batch nodal load — {len(self._node_ids)} nodes",
        )

    def _build_body(self, body: QWidget) -> None:
        from PyQt6.QtWidgets import QFormLayout
        v = QVBoxLayout(body)
        v.addWidget(QLabel(
            f"<b>Append the same nodal load to "
            f"{len(self._node_ids)} selected nodes.</b>",
            body,
        ))
        v.addWidget(QLabel(
            "Existing nodal loads on each node are preserved — this dialog "
            "appends a new row.",
            body,
        ))
        form_widget = QWidget(body)
        form = QFormLayout(form_widget)
        self._fx = QLineEdit(form_widget); self._fx.setText("0")
        self._fy = QLineEdit(form_widget); self._fy.setText("0")
        self._mz = QLineEdit(form_widget); self._mz.setText("0")
        form.addRow("Fx (kN, global X):", self._fx)
        form.addRow("Fy (kN, global Y):", self._fy)
        form.addRow("Mz (kN·m, +CCW):", self._mz)
        v.addWidget(form_widget)
        v.addWidget(QLabel("Load case:", body))
        self._case_combo = _make_load_case_combo(body, "DEFAULT", model=self._model)
        v.addWidget(self._case_combo)

    def _accept(self) -> dict:
        fx = parse_float(self._fx.text(), "Fx", allow_blank=True) or 0.0
        fy = parse_float(self._fy.text(), "Fy", allow_blank=True) or 0.0
        mz = parse_float(self._mz.text(), "Mz", allow_blank=True) or 0.0
        case = _normalize_load_case(self._case_combo.currentText())
        if fx == 0.0 and fy == 0.0 and mz == 0.0:
            raise ValueError(
                "At least one of Fx, Fy, Mz must be non-zero."
            )
        return {"fx": fx, "fy": fy, "mz": mz, "load_case": case}


# ── Group Manager dialog (v0.27.0) ─────────────────────────────────────────


class GroupManagerDialog(QDialog):
    """Manage named selection groups.

    Lists all groups in a table (Name / Nodes / Elements) and provides
    buttons to Create, Edit Members, Rename, Delete, Select, and adjust
    canvas selection from groups.  Every group mutation is reflected
    immediately in the table.  Groups are GUI/project metadata only —
    the solver is never involved.
    """

    def __init__(
        self,
        parent: "QWidget | None",
        *,
        host: "Any",
        groups: "dict[str, Any]",
        model: "StructuralModel",
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._groups = groups
        self._model = model
        self.setWindowTitle("Group Manager")
        self.setModal(True)
        self.resize(520, 380)
        self._build_ui()
        self._rebuild_table()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Named groups let you save and recall selections of nodes and elements.\n"
            "Groups are stored with the .spa.json project file.",
            self,
        ))

        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Name", "Nodes", "Elements"])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

        # ── action buttons ──
        btn_row1 = QHBoxLayout()
        self._btn_create = QPushButton("Create from Selection", self)
        self._btn_create.clicked.connect(self._on_create)
        btn_row1.addWidget(self._btn_create)
        self._btn_add_sel = QPushButton("Add Current Selection", self)
        self._btn_add_sel.clicked.connect(self._on_add_selection)
        btn_row1.addWidget(self._btn_add_sel)
        self._btn_replace = QPushButton("Replace with Selection", self)
        self._btn_replace.clicked.connect(self._on_replace)
        btn_row1.addWidget(self._btn_replace)
        self._btn_remove_sel = QPushButton("Remove Current Selection", self)
        self._btn_remove_sel.clicked.connect(self._on_remove_selection)
        btn_row1.addWidget(self._btn_remove_sel)
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self._btn_select = QPushButton("Select", self)
        self._btn_select.clicked.connect(self._on_select)
        btn_row2.addWidget(self._btn_select)
        self._btn_add_to = QPushButton("Add to Selection", self)
        self._btn_add_to.clicked.connect(self._on_add_to_selection)
        btn_row2.addWidget(self._btn_add_to)
        self._btn_rem_from = QPushButton("Remove from Selection", self)
        self._btn_rem_from.clicked.connect(self._on_remove_from_selection)
        btn_row2.addWidget(self._btn_rem_from)
        btn_row2.addStretch()
        self._btn_rename = QPushButton("Rename…", self)
        self._btn_rename.clicked.connect(self._on_rename)
        btn_row2.addWidget(self._btn_rename)
        self._btn_delete = QPushButton("Delete", self)
        self._btn_delete.clicked.connect(self._on_delete)
        btn_row2.addWidget(self._btn_delete)
        layout.addLayout(btn_row2)

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        close_box.rejected.connect(self.reject)
        layout.addWidget(close_box)

    def _rebuild_table(self) -> None:
        self._table.setRowCount(0)
        for row, (name, g) in enumerate(sorted(self._groups.items())):
            self._table.insertRow(row)
            item_name = QTableWidgetItem(name)
            item_name.setData(Qt.ItemDataRole.UserRole, name)
            self._table.setItem(row, 0, item_name)
            self._table.setItem(row, 1, QTableWidgetItem(str(len(g.node_ids))))
            self._table.setItem(row, 2, QTableWidgetItem(str(len(g.element_ids))))

    def _selected_group_name(self) -> "str | None":
        row = self._table.currentRow()
        if row < 0:
            return None
        item = self._table.item(row, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    # ── button handlers ──

    def _on_create(self) -> None:
        self._host._group_create_from_selection()
        self._rebuild_table()

    def _on_add_selection(self) -> None:
        name = self._selected_group_name()
        if name is None:
            QMessageBox.information(self, "Add current selection", "Select a group row first.")
            return
        self._host._group_add_selection(name)
        self._rebuild_table()

    def _on_replace(self) -> None:
        name = self._selected_group_name()
        if name is None:
            QMessageBox.information(self, "Replace group", "Select a group row first.")
            return
        self._host._group_replace_with_selection(name)
        self._rebuild_table()

    def _on_remove_selection(self) -> None:
        name = self._selected_group_name()
        if name is None:
            QMessageBox.information(self, "Remove current selection", "Select a group row first.")
            return
        self._host._group_remove_selection(name)
        self._rebuild_table()

    def _on_select(self) -> None:
        name = self._selected_group_name()
        if name is None:
            QMessageBox.information(self, "Select group", "Select a group row first.")
            return
        self._host._group_select(name)

    def _on_add_to_selection(self) -> None:
        name = self._selected_group_name()
        if name is None:
            QMessageBox.information(self, "Add to selection", "Select a group row first.")
            return
        self._host._group_add_to_selection(name)

    def _on_remove_from_selection(self) -> None:
        name = self._selected_group_name()
        if name is None:
            QMessageBox.information(self, "Remove from selection", "Select a group row first.")
            return
        self._host._group_remove_from_selection(name)

    def _on_rename(self) -> None:
        self._host._group_rename()
        self._rebuild_table()

    def _on_delete(self) -> None:
        self._host._group_delete()
        self._rebuild_table()
