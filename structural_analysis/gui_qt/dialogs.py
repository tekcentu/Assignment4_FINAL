"""PyQt6 modal dialogs for editing model entities.

Numeric fields go through ``parse_float`` so users see a friendly message
instead of a Python traceback when they type bad input.
"""

from __future__ import annotations

from typing import Any, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt
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
    load = next((ld for ld in model.nodal_loads if ld.node_id == node_id), None)
    if load is None:
        return "(none)"
    return f"Fx = {load.fx:g} kN,  Fy = {load.fy:g} kN,  Mz = {load.mz:g} kN·m"


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
    """Read-only inspector for an element.

    Non-modal so the main window stays usable for view-only operations
    (pan / zoom / solve / overlay toggles). The host
    (:class:`MainWindow`) is responsible for locking edit actions
    while the inspector is visible — see
    :meth:`MainWindow._set_editing_locked`.

    Constructed as a singleton on ``MainWindow._element_inspector``;
    re-opening from another element calls :meth:`set_target` to swap
    the contents in place. After a solve the host calls
    :meth:`refresh` so the diagrams pick up the new result.
    """

    def __init__(self, parent, model: StructuralModel, elem_id: int,
                 result=None) -> None:
        super().__init__(parent)
        self.setModal(False)
        # Persist across close so MainWindow's singleton stays valid;
        # tests + the host both rely on _element_inspector being
        # reusable across right-clicks.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._outer = QVBoxLayout(self)
        # Body widget — wholly replaced by set_target on each refresh.
        self._body_widget: QWidget = QWidget(self)
        self._outer.addWidget(self._body_widget)

        # Buttons live at the bottom permanently — only the body swaps.
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
        self.set_target(model, elem_id, result)

    def set_target(
        self, model: StructuralModel, elem_id: int, result=None,
    ) -> None:
        """Swap the inspector to show ``elem_id``. Raises ``ValueError``
        if the element does not exist. The figure / form widgets are
        rebuilt from scratch — simpler than wiring every field for
        individual updates, and the dialog is hardly hot-path."""
        elem = next((e for e in model.elements if e.id == elem_id), None)
        if elem is None:
            raise ValueError(f"Element {elem_id} does not exist.")
        new_body = self._build_body(model, elem, result)
        self._outer.replaceWidget(self._body_widget, new_body)
        self._body_widget.setParent(None)
        self._body_widget.deleteLater()
        self._body_widget = new_body
        self._elem_id = elem_id
        self.setWindowTitle(f"Element {elem_id} properties")

    def refresh(self, model: StructuralModel, result=None) -> None:
        """Re-render the current element against ``model`` / ``result``.
        Called by the host after :meth:`MainWindow._do_solve` so the
        N/V/M traces and the end-force block pick up the new result.
        Silently no-ops if the current element id no longer exists."""
        if not any(e.id == self._elem_id for e in model.elements):
            self.close()
            return
        self.set_target(model, self._elem_id, result)

    def _build_body(
        self, model: StructuralModel, elem, result,
    ) -> QWidget:
        from .element_graphics import draw_element_detail

        body = QWidget(self)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        layout.addLayout(form)

        elem_id = elem.id
        section = model.sections.get(getattr(elem, "section_id", None) or -1)
        material = (model.materials.get(section.material_id)
                     if section is not None else None)

        ni = model.nodes.get(elem.node_i)
        nj = model.nodes.get(elem.node_j)
        if ni is None or nj is None:
            length = 0.0
        else:
            length = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5

        form.addRow("Element ID:", QLabel(str(elem_id)))
        form.addRow("Kind:", QLabel(elem.kind.capitalize()))
        form.addRow("Nodes:", QLabel(f"{elem.node_i} → {elem.node_j}"))
        form.addRow("Length:", QLabel(f"{length:g} m"))

        if section is not None:
            sec_text = section.name or f"section {section.id}"
            form.addRow("Section:", QLabel(f"{sec_text}  (id {section.id})"))
        else:
            form.addRow("Section:", QLabel("(none)"))
        if material is not None:
            mat_text = material.name or f"material {material.id}"
            form.addRow("Material:", QLabel(f"{mat_text}  (id {material.id})"))

        form.addRow("E:", QLabel(f"{elem.E:g} kN/m²"))
        form.addRow("A:", QLabel(f"{elem.A:g} m²"))
        if isinstance(elem, FrameElement2D):
            form.addRow("I:", QLabel(f"{elem.I:g} m⁴"))
            form.addRow("Releases:",
                         QLabel(f"i={elem.release_i},  j={elem.release_j}"))

        loads = _member_loads_summary(elem)
        form.addRow("Member loads:", QLabel(loads[0]))
        for line in loads[1:]:
            form.addRow("", QLabel(line))

        # Graphical detail block — single source of truth in
        # element_graphics, shared with the main canvas.
        self._detail_fig = Figure(figsize=(6.4, 4.6), dpi=96)
        self._detail_fig.patch.set_facecolor("white")
        self._detail_canvas = FigureCanvasQTAgg(self._detail_fig)
        self._detail_canvas.setMinimumSize(520, 360)
        layout.addWidget(self._detail_canvas)
        ok_result = (
            result if (result is not None
                       and getattr(result, "status", None) == "ok")
            else None
        )
        self._detail_axes = draw_element_detail(
            self._detail_fig, elem, model, ok_result,
        )
        self._detail_canvas.draw_idle()

        if result is not None and getattr(result, "status", None) == "ok":
            f_local = _element_local_forces(result, elem_id)
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

        return body


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
