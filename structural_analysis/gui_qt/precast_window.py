"""Precast Handling Stage window (V1).

A non-modal singleton ``QMainWindow`` that shows ONE selected frame
element as a temporary, isolated precast member and lets the user check
it through three handling stages (lifting, stock / storage, truck /
transport). Left side is a lightweight matplotlib canvas (member sketch
+ V/M diagrams); right side holds the stage controls, a results table,
warnings, and a copy-report button.

The window never mutates the main model — it snapshots the member into a
:class:`~structural_analysis.gui_qt.precast.MemberSpec`. Stage data is
temporary UI state only and is NOT saved to the project file in V1.

All statics live in :mod:`structural_analysis.gui_qt.precast`; the V/M
diagrams come from the shared ``element_graphics`` helpers via that
engine (no second BMD/SFD formula here).
"""

from __future__ import annotations

import math
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..model import StructuralModel
from . import precast as P
from .precast import (
    ORIENT_CUSTOM,
    ORIENT_HORIZONTAL,
    ORIENT_MODEL,
    SCHEME_ONE_POINT,
    SCHEME_TWO_POINT,
    STAGE_LABELS,
    STAGE_LIFTING,
    STAGE_STOCK,
    STAGE_TRUCK,
    StageInput,
    compute_handling,
    format_report,
    member_spec_from_element,
    resolve_single_frame,
)
from .table_copy import install_table_copy


class PrecastHandlingWindow(QMainWindow):
    """Temporary precast handling-stage checker for one frame element."""

    def __init__(
        self,
        parent: QWidget | None,
        model_provider: Callable[[], StructuralModel],
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowTitle("Precast Handling Stages")
        self.resize(1100, 680)

        self._model_provider = model_provider
        self._member: P.MemberSpec | None = None
        self._stage_inputs: dict[str, StageInput] = {}
        self._updating = False   # guards programmatic control updates
        self._last_report = ""

        self._build_ui()

    # ── construction ──

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Left: matplotlib canvas with member sketch + V + M.
        self._fig = Figure(figsize=(6.0, 6.0), dpi=100)
        self._ax_member = self._fig.add_subplot(3, 1, 1)
        self._ax_v = self._fig.add_subplot(3, 1, 2)
        self._ax_m = self._fig.add_subplot(3, 1, 3)
        self._canvas = FigureCanvasQTAgg(self._fig)
        splitter.addWidget(self._canvas)

        # Right: controls + results.
        right = QWidget(self)
        col = QVBoxLayout(right)
        col.setContentsMargins(8, 8, 8, 8)

        self._member_label = QLabel("No member selected", right)
        self._member_label.setStyleSheet("font-weight: bold;")
        col.addWidget(self._member_label)

        form = QFormLayout()

        self._stage_combo = QComboBox(right)
        for key in (STAGE_LIFTING, STAGE_STOCK, STAGE_TRUCK):
            self._stage_combo.addItem(STAGE_LABELS[key], key)
        self._stage_combo.currentIndexChanged.connect(self._on_stage_changed)
        form.addRow("Stage", self._stage_combo)

        self._orient_combo = QComboBox(right)
        self._orient_combo.addItem("Horizontal handling", ORIENT_HORIZONTAL)
        self._orient_combo.addItem("Original model angle", ORIENT_MODEL)
        self._orient_combo.addItem("Custom angle", ORIENT_CUSTOM)
        self._orient_combo.currentIndexChanged.connect(self._on_input_changed)
        form.addRow("Display orientation", self._orient_combo)

        self._custom_angle = self._spin(right, -180.0, 180.0, 0.0, 1.0, " °")
        form.addRow("Custom angle", self._custom_angle)

        self._scheme_combo = QComboBox(right)
        self._scheme_combo.addItem("Two-point lift", SCHEME_TWO_POINT)
        self._scheme_combo.addItem("One-point lift", SCHEME_ONE_POINT)
        self._scheme_combo.currentIndexChanged.connect(self._on_scheme_changed)
        form.addRow("Lifting scheme", self._scheme_combo)

        self._p1 = self._spin(right, 0.0, 1.0, 0.0, 0.1, " m")
        self._p2 = self._spin(right, 0.0, 1.0, 0.0, 0.1, " m")
        form.addRow("Point 1 position", self._p1)
        form.addRow("Point 2 position", self._p2)

        self._sling_angle = self._spin(right, 1.0, 90.0, 60.0, 1.0, " °")
        form.addRow("Sling angle (from horiz.)", self._sling_angle)

        self._daf = self._spin(right, 0.1, 5.0, 1.0, 0.05, "")
        self._daf.setDecimals(2)
        form.addRow("DAF", self._daf)

        self._auto_weight = QCheckBox("Auto (from section)", right)
        self._auto_weight.setChecked(True)
        self._auto_weight.toggled.connect(self._on_input_changed)
        form.addRow("Self-weight", self._auto_weight)
        self._manual_weight = self._spin(right, 0.0, 1e6, 0.0, 0.5, " kN/m")
        form.addRow("Manual weight", self._manual_weight)

        self._suction = self._spin(right, 0.0, 1e6, 0.0, 0.5, " kN/m")
        form.addRow("Suction / adhesion", self._suction)

        self._extra_udl = self._spin(right, 0.0, 1e6, 0.0, 0.5, " kN/m")
        form.addRow("Extra handling UDL", self._extra_udl)

        col.addLayout(form)

        self._results = QTableWidget(0, 4, right)
        self._results.setHorizontalHeaderLabels(
            ["Point x (m)", "Reaction (kN)", "Sling T (kN)", "Sling H (kN)"],
        )
        self._results.verticalHeader().setVisible(False)
        self._results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        install_table_copy(self._results, include_headers=True)
        col.addWidget(self._results)

        self._summary_label = QLabel("", right)
        col.addWidget(self._summary_label)

        self._warn_label = QLabel("", right)
        self._warn_label.setWordWrap(True)
        self._warn_label.setStyleSheet("color: #b00;")
        col.addWidget(self._warn_label)

        self._note_label = QLabel(P.DISPLAY_ONLY_NOTE, right)
        self._note_label.setWordWrap(True)
        self._note_label.setStyleSheet("color: #555; font-style: italic;")
        col.addWidget(self._note_label)

        btn_row = QHBoxLayout()
        self._copy_btn = QPushButton("Copy report", right)
        self._copy_btn.clicked.connect(self._copy_report)
        btn_row.addStretch(1)
        btn_row.addWidget(self._copy_btn)
        close_btn = QPushButton("Close", right)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        col.addLayout(btn_row)

        temp = QLabel(
            "Handling stages are temporary in V1 — not saved with the project.",
            right,
        )
        temp.setStyleSheet("color: #777; font-size: 8pt;")
        temp.setWordWrap(True)
        col.addWidget(temp)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        for sp in (self._p1, self._p2, self._sling_angle, self._daf,
                   self._manual_weight, self._suction, self._extra_udl,
                   self._custom_angle):
            sp.valueChanged.connect(self._on_input_changed)

    @staticmethod
    def _spin(parent, lo, hi, val, step, suffix) -> QDoubleSpinBox:
        sp = QDoubleSpinBox(parent)
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(3)
        sp.setValue(val)
        if suffix:
            sp.setSuffix(suffix)
        return sp

    # ── public API ──

    def set_target(self, elem_id: int) -> None:
        """Point the window at element ``elem_id`` (must be a frame)."""
        model = self._model_provider()
        elem = resolve_single_frame(model, [elem_id])
        self._member = member_spec_from_element(model, elem)
        L = self._member.length
        self._stage_inputs = {
            STAGE_LIFTING: StageInput(
                stage=STAGE_LIFTING, lifting_scheme=SCHEME_TWO_POINT,
                points=(round(0.2 * L, 3), round(0.8 * L, 3)),
            ),
            STAGE_STOCK: StageInput(
                stage=STAGE_STOCK, points=(round(0.2 * L, 3), round(0.8 * L, 3)),
            ),
            STAGE_TRUCK: StageInput(
                stage=STAGE_TRUCK, points=(round(0.1 * L, 3), round(0.9 * L, 3)),
            ),
        }
        for sp in (self._p1, self._p2):
            sp.setRange(0.0, L)
        self._member_label.setText(
            f"Element {self._member.elem_id}"
            + (f" · {self._member.section_name}" if self._member.section_name
               else "")
            + f" · L = {L:.3g} m · self-weight {self._member.self_weight:.4g} kN/m"
        )
        self._stage_combo.setCurrentIndex(0)
        self._load_stage_into_controls(self._current_stage_key())
        self._recompute()

    def refresh(self) -> None:
        """Re-snapshot the current member if it still exists."""
        if self._member is None:
            return
        try:
            self.set_target(self._member.elem_id)
        except (ValueError, TypeError):
            # The element was deleted / changed type while the window was
            # open — leave the last snapshot in place.
            pass

    # ── control <-> state ──

    def _current_stage_key(self) -> str:
        return self._stage_combo.currentData()

    def _load_stage_into_controls(self, key: str) -> None:
        st = self._stage_inputs[key]
        self._updating = True
        try:
            idx = self._orient_combo.findData(st.orientation)
            self._orient_combo.setCurrentIndex(max(0, idx))
            self._custom_angle.setValue(st.custom_angle_deg)
            sidx = self._scheme_combo.findData(st.lifting_scheme)
            self._scheme_combo.setCurrentIndex(max(0, sidx))
            pts = list(st.points) + [0.0, 0.0]
            self._p1.setValue(pts[0])
            self._p2.setValue(pts[1])
            self._sling_angle.setValue(st.sling_angle_deg)
            self._daf.setValue(st.daf)
            self._auto_weight.setChecked(st.manual_weight is None)
            self._manual_weight.setValue(
                st.manual_weight if st.manual_weight is not None
                else (self._member.self_weight if self._member else 0.0)
            )
            self._suction.setValue(st.suction)
            self._extra_udl.setValue(st.extra_udl)
        finally:
            self._updating = False
        self._apply_enabled_state(key)

    def _apply_enabled_state(self, key: str) -> None:
        is_lift = key == STAGE_LIFTING
        one_pt = (is_lift
                  and self._scheme_combo.currentData() == SCHEME_ONE_POINT)
        self._scheme_combo.setEnabled(is_lift)
        self._sling_angle.setEnabled(is_lift)
        self._suction.setEnabled(is_lift)
        self._p2.setEnabled(not one_pt)
        self._manual_weight.setEnabled(not self._auto_weight.isChecked())
        self._custom_angle.setEnabled(
            self._orient_combo.currentData() == ORIENT_CUSTOM,
        )

    def _read_controls_into_stage(self, key: str) -> None:
        is_lift = key == STAGE_LIFTING
        scheme = self._scheme_combo.currentData()
        if is_lift and scheme == SCHEME_ONE_POINT:
            points = (self._p1.value(),)
        else:
            points = (self._p1.value(), self._p2.value())
        self._stage_inputs[key] = StageInput(
            stage=key,
            points=points,
            lifting_scheme=scheme,
            sling_angle_deg=self._sling_angle.value(),
            daf=self._daf.value(),
            manual_weight=(None if self._auto_weight.isChecked()
                           else self._manual_weight.value()),
            suction=self._suction.value(),
            extra_udl=self._extra_udl.value(),
            orientation=self._orient_combo.currentData(),
            custom_angle_deg=self._custom_angle.value(),
        )

    # ── signal handlers ──

    def _on_stage_changed(self) -> None:
        if self._updating or self._member is None:
            return
        self._load_stage_into_controls(self._current_stage_key())
        self._recompute()

    def _on_scheme_changed(self) -> None:
        if self._updating:
            return
        self._apply_enabled_state(self._current_stage_key())
        self._on_input_changed()

    def _on_input_changed(self) -> None:
        if self._updating or self._member is None:
            return
        self._apply_enabled_state(self._current_stage_key())
        self._recompute()

    # ── compute + render ──

    def _recompute(self) -> None:
        if self._member is None:
            return
        key = self._current_stage_key()
        self._read_controls_into_stage(key)
        stage = self._stage_inputs[key]
        try:
            result = compute_handling(self._member, stage)
        except (ValueError, TypeError) as exc:
            self._warn_label.setText(f"⚠ {exc}")
            self._results.setRowCount(0)
            self._summary_label.setText("")
            self._last_report = ""
            return
        self._render(stage, result)

    def _render(self, stage: StageInput, result: P.HandlingResult) -> None:
        # Results table.
        self._results.setRowCount(len(result.reactions))
        for r, (x, rr) in enumerate(result.reactions):
            vals = [f"{x:.3f}", f"{rr:.4g}"]
            if result.sling_tensions:
                vals += [f"{result.sling_tensions[r]:.4g}",
                         f"{result.sling_horizontal[r]:.4g}"]
            else:
                vals += ["—", "—"]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
                self._results.setItem(r, c, item)

        self._summary_label.setText(
            f"Total load: {result.total_load:.4g} kN   ·   "
            f"|V|max: {result.v_max:.4g} kN   ·   "
            f"M+: {result.m_pos_max:.4g}   M−: {result.m_neg_max:.4g} kN·m"
        )
        self._warn_label.setText(
            "\n".join(f"⚠ {w}" for w in result.warnings)
        )
        self._last_report = format_report(self._member, stage, result)
        self._draw(stage, result)

    def _draw(self, stage: StageInput, result: P.HandlingResult) -> None:
        L = self._member.length
        # Display angle (display-only; calc is always horizontal).
        if stage.orientation == ORIENT_MODEL:
            ang = math.radians(self._member.model_angle_deg)
        elif stage.orientation == ORIENT_CUSTOM:
            ang = math.radians(stage.custom_angle_deg)
        else:
            ang = 0.0
        ca, sa = math.cos(ang), math.sin(ang)

        ax = self._ax_member
        ax.clear()
        ax.plot([0.0, L * ca], [0.0, L * sa], color="#1f3a5f", lw=4)
        # Support / lift markers.
        for x, _r in result.reactions:
            px, py = x * ca, x * sa
            if result.stage == STAGE_LIFTING:
                # Sling line up to a hook above the midpoint.
                hx, hy = (L / 2.0) * ca, (L / 2.0) * sa + 0.25 * L
                ax.plot([px, hx], [py, hy], color="#1a7f37", lw=1.2)
                ax.plot([px], [py], marker="v", color="#1a7f37", ms=9)
            else:
                ax.plot([px], [py - 0.04 * L], marker="^", color="#444", ms=12)
        if result.stage == STAGE_LIFTING and result.reactions:
            hx, hy = (L / 2.0) * ca, (L / 2.0) * sa + 0.25 * L
            ax.plot([hx], [hy], marker="o", color="#1a7f37", ms=7)
        # Self-weight arrows.
        for i in range(1, 6):
            x = i * L / 6.0
            ax.annotate("", xy=(x * ca, x * sa - 0.06 * L),
                        xytext=(x * ca, x * sa),
                        arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))
        ax.set_title(f"{STAGE_LABELS[result.stage]} — element "
                     f"{self._member.elem_id}", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.15)
        ax.tick_params(labelsize=7)

        xs = [s[0] for s in result.stations]
        vs = [s[1] for s in result.stations]
        ms = [s[2] for s in result.stations]
        for axd, ys, title, color in (
            (self._ax_v, vs, "Shear V (kN)", "#0f9d58"),
            (self._ax_m, ms, "Moment M (kN·m)", "#d24c4c"),
        ):
            axd.clear()
            if xs:
                axd.axhline(0.0, color="#bbb", lw=0.6)
                axd.plot(xs, ys, color=color, lw=1.4)
                axd.fill_between(xs, ys, 0.0, color=color, alpha=0.15)
            axd.set_title(title, fontsize=9)
            axd.tick_params(labelsize=7)
        self._ax_m.set_xlabel("x along member (m)", fontsize=8)

        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ── report ──

    def _copy_report(self) -> None:
        if not self._last_report:
            return
        from PyQt6.QtWidgets import QApplication
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(self._last_report)
