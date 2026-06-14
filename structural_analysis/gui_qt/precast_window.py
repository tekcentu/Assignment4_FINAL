"""Precast Handling Stage window (V2).

A non-modal singleton ``QMainWindow`` that shows ONE selected frame
element as a temporary, isolated precast member and lets the user check
it through all three handling stages — lifting, stock / storage, and
truck / transport — laid out together on a single scrolling sheet.

The window never mutates the main model — it snapshots the member into a
:class:`~structural_analysis.gui_qt.precast.MemberSpec`. Stage data is
temporary UI state only and is NOT saved to the project file in V2.

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
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
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
    STAGE_LABELS,
    STAGE_LIFTING,
    STAGES,
    StageInput,
    auto_even_points,
    compute_handling,
    format_report,
    member_spec_from_element,
    resolve_single_frame,
)


def _spin(parent, lo, hi, val, step, suffix, *, decimals=3) -> QDoubleSpinBox:
    sp = QDoubleSpinBox(parent)
    sp.setRange(lo, hi)
    sp.setSingleStep(step)
    sp.setDecimals(decimals)
    sp.setValue(val)
    if suffix:
        sp.setSuffix(suffix)
    return sp


class _StageRow(QFrame):
    """One row of the precast sheet — controls + member sketch + V + M.

    Owned by :class:`PrecastHandlingWindow`. The window injects:
    * ``on_changed`` — fired whenever the user edits any spinbox in this
      row (positions, sling, suction).
    """

    def __init__(
        self,
        parent: QWidget,
        stage_key: str,
        on_changed: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self.stage_key = stage_key
        self._on_changed = on_changed
        self._updating = False
        self.setFrameShape(QFrame.Shape.StyledPanel)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        title = QLabel(f"<b>{STAGE_LABELS[stage_key]}</b>", self)
        outer.addWidget(title)

        controls = QGridLayout()
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(4)
        row = 0
        controls.addWidget(QLabel("Point 1 (m)", self), row, 0)
        self.p1 = _spin(self, 0.0, 1.0, 0.0, 0.1, "")
        controls.addWidget(self.p1, row, 1)
        controls.addWidget(QLabel("Point 2 (m)", self), row, 2)
        self.p2 = _spin(self, 0.0, 1.0, 0.0, 0.1, "")
        controls.addWidget(self.p2, row, 3)
        self.auto_btn = QPushButton("Auto-space", self)
        self.auto_btn.clicked.connect(self._auto_space_clicked)
        controls.addWidget(self.auto_btn, row, 4)

        # Lifting-only controls live on the lifting row.
        self.sling_angle: QDoubleSpinBox | None = None
        self.suction: QDoubleSpinBox | None = None
        if stage_key == STAGE_LIFTING:
            row += 1
            controls.addWidget(QLabel("Sling angle (°)", self), row, 0)
            self.sling_angle = _spin(self, 1.0, 90.0, 60.0, 1.0, " °",
                                     decimals=1)
            controls.addWidget(self.sling_angle, row, 1)
            controls.addWidget(QLabel("Suction (kN/m)", self), row, 2)
            self.suction = _spin(self, 0.0, 1e6, 0.0, 0.5, " kN/m")
            controls.addWidget(self.suction, row, 3)

        outer.addLayout(controls)

        # Matplotlib figure: 1 × 3 — member sketch, V, M.
        self._fig = Figure(figsize=(9.0, 2.4), dpi=100)
        self._ax_member = self._fig.add_subplot(1, 3, 1)
        self._ax_v = self._fig.add_subplot(1, 3, 2)
        self._ax_m = self._fig.add_subplot(1, 3, 3)
        self.canvas = FigureCanvasQTAgg(self._fig)
        self.canvas.setMinimumHeight(220)
        outer.addWidget(self.canvas)

        self.summary = QLabel("", self)
        self.summary.setWordWrap(True)
        outer.addWidget(self.summary)

        self.warning = QLabel("", self)
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #b00;")
        outer.addWidget(self.warning)

        # Wire up edits.
        for sp in (self.p1, self.p2):
            sp.valueChanged.connect(self._fire)
        if self.sling_angle is not None:
            self.sling_angle.valueChanged.connect(self._fire)
        if self.suction is not None:
            self.suction.valueChanged.connect(self._fire)

    # ── helpers ──

    def _fire(self) -> None:
        if self._updating:
            return
        self._on_changed()

    def _auto_space_clicked(self) -> None:
        # The window injects the length via set_length(); use it here.
        length = self.p1.maximum()
        a, b = auto_even_points(self.stage_key, length)
        self._updating = True
        try:
            self.p1.setValue(a)
            self.p2.setValue(b)
        finally:
            self._updating = False
        self._on_changed()

    def set_length(self, length: float) -> None:
        """Update spinbox ranges when the member length changes."""
        self._updating = True
        try:
            for sp in (self.p1, self.p2):
                sp.setRange(0.0, length)
        finally:
            self._updating = False

    def load_inputs(self, stage: StageInput) -> None:
        self._updating = True
        try:
            x1, x2 = stage.points
            self.p1.setValue(x1)
            self.p2.setValue(x2)
            if self.sling_angle is not None:
                self.sling_angle.setValue(stage.sling_angle_deg)
            if self.suction is not None:
                self.suction.setValue(stage.suction)
        finally:
            self._updating = False

    def read_inputs(self, template: StageInput) -> StageInput:
        """Update ``template`` with the row's current control values."""
        points = (self.p1.value(), self.p2.value())
        sling_angle = (self.sling_angle.value() if self.sling_angle is not None
                       else template.sling_angle_deg)
        suction = (self.suction.value() if self.suction is not None
                   else template.suction)
        return StageInput(
            stage=template.stage,
            points=points,
            sling_angle_deg=sling_angle,
            daf=template.daf,
            manual_weight=template.manual_weight,
            suction=suction,
            extra_udl=template.extra_udl,
            orientation=template.orientation,
            custom_angle_deg=template.custom_angle_deg,
        )

    # ── rendering ──

    def show_error(self, msg: str) -> None:
        self.summary.setText("")
        self.warning.setText(f"⚠ {msg}")
        for ax in (self._ax_member, self._ax_v, self._ax_m):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        self.canvas.draw_idle()

    def show_result(
        self,
        member: P.MemberSpec,
        stage: StageInput,
        result: P.HandlingResult,
    ) -> None:
        # Summary line — reactions and slings inline.
        parts = [f"UDL {result.udl_per_m:.4g} kN/m · "
                 f"W {result.total_load:.4g} kN"]
        for i, (x, r) in enumerate(result.reactions):
            if result.sling_tensions:
                parts.append(
                    f"R@{x:.3g}m = {r:.4g} kN · "
                    f"T = {result.sling_tensions[i]:.4g} kN · "
                    f"H = {result.sling_horizontal[i]:.4g} kN"
                )
            else:
                parts.append(f"R@{x:.3g}m = {r:.4g} kN")
        parts.append(
            f"|V|max = {result.v_max:.4g} kN · "
            f"M+ {result.m_pos_max:.4g} · M− {result.m_neg_max:.4g} kN·m"
        )
        self.summary.setText("   |   ".join(parts))
        self.warning.setText(
            "\n".join(f"⚠ {w}" for w in result.warnings)
        )
        self._draw(member, stage, result)

    def _draw(
        self,
        member: P.MemberSpec,
        stage: StageInput,
        result: P.HandlingResult,
    ) -> None:
        L = member.length
        # Display angle (display-only; calc is always horizontal).
        if stage.orientation == ORIENT_MODEL:
            ang = math.radians(member.model_angle_deg)
        elif stage.orientation == ORIENT_CUSTOM:
            ang = math.radians(stage.custom_angle_deg)
        else:
            ang = 0.0
        ca, sa = math.cos(ang), math.sin(ang)

        ax = self._ax_member
        ax.clear()
        ax.plot([0.0, L * ca], [0.0, L * sa], color="#1f3a5f", lw=4)

        for x, _r in result.reactions:
            px, py = x * ca, x * sa
            if result.stage == STAGE_LIFTING:
                hx, hy = (L / 2.0) * ca, (L / 2.0) * sa + 0.25 * L
                ax.plot([px, hx], [py, hy], color="#1a7f37", lw=1.2)
                ax.plot([px], [py], marker="v", color="#1a7f37", ms=9)
            else:
                ax.plot([px], [py - 0.04 * L], marker="^",
                        color="#444", ms=12)
        if result.stage == STAGE_LIFTING and result.reactions:
            hx, hy = (L / 2.0) * ca, (L / 2.0) * sa + 0.25 * L
            ax.plot([hx], [hy], marker="o", color="#1a7f37", ms=7)

        for i in range(1, 6):
            x = i * L / 6.0
            ax.annotate("", xy=(x * ca, x * sa - 0.06 * L),
                        xytext=(x * ca, x * sa),
                        arrowprops=dict(arrowstyle="->",
                                        color="#999", lw=0.8))
        ax.set_title("Member", fontsize=9)
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
        self._ax_m.set_xlabel("x (m)", fontsize=8)
        self._ax_v.set_xlabel("x (m)", fontsize=8)

        self._fig.tight_layout()
        self.canvas.draw_idle()


class PrecastHandlingWindow(QMainWindow):
    """Temporary precast handling-stage checker for one frame element.

    All three stages are visible at once on a scrollable sheet. Global
    inputs (orientation, DAF, weight, extra UDL) at the top; each stage
    row owns its own positions, and the lifting row owns the sling and
    suction inputs.
    """

    def __init__(
        self,
        parent: QWidget | None,
        model_provider: Callable[[], StructuralModel],
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowTitle("Precast Handling Stages")
        self.resize(1180, 820)

        self._model_provider = model_provider
        self._member: P.MemberSpec | None = None
        self._stage_inputs: dict[str, StageInput] = {}
        self._last_results: dict[str, P.HandlingResult] = {}
        self._updating = False
        self._last_report = ""

        self._build_ui()

    # ── construction ──

    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        col = QVBoxLayout(content)
        col.setContentsMargins(10, 10, 10, 10)
        col.setSpacing(10)

        self._member_label = QLabel("No member selected", content)
        self._member_label.setStyleSheet("font-weight: bold;")
        col.addWidget(self._member_label)

        # Global controls.
        glob = QFrame(content)
        glob.setFrameShape(QFrame.Shape.StyledPanel)
        form = QFormLayout(glob)
        form.setContentsMargins(8, 8, 8, 8)

        self._orient_combo = QComboBox(glob)
        self._orient_combo.addItem("Horizontal handling", ORIENT_HORIZONTAL)
        self._orient_combo.addItem("Original model angle", ORIENT_MODEL)
        self._orient_combo.addItem("Custom angle", ORIENT_CUSTOM)
        self._orient_combo.currentIndexChanged.connect(self._on_global_changed)
        form.addRow("Display orientation", self._orient_combo)

        self._custom_angle = _spin(glob, -180.0, 180.0, 0.0, 1.0, " °",
                                   decimals=1)
        form.addRow("Custom angle", self._custom_angle)

        self._daf = _spin(glob, 0.1, 5.0, 1.0, 0.05, "", decimals=2)
        form.addRow("DAF", self._daf)

        self._auto_weight = QCheckBox("Auto (from section)", glob)
        self._auto_weight.setChecked(True)
        self._auto_weight.toggled.connect(self._on_global_changed)
        form.addRow("Self-weight", self._auto_weight)
        self._manual_weight = _spin(glob, 0.0, 1e6, 0.0, 0.5, " kN/m")
        form.addRow("Manual weight", self._manual_weight)

        self._extra_udl = _spin(glob, 0.0, 1e6, 0.0, 0.5, " kN/m")
        form.addRow("Extra handling UDL", self._extra_udl)

        for sp in (self._custom_angle, self._daf,
                   self._manual_weight, self._extra_udl):
            sp.valueChanged.connect(self._on_global_changed)

        col.addWidget(glob)

        # Stage rows — keep references keyed by stage.
        self._rows: dict[str, _StageRow] = {}
        for key in STAGES:
            row = _StageRow(content, key, self._on_row_changed)
            col.addWidget(row)
            self._rows[key] = row

        # Bottom buttons.
        btn_row = QHBoxLayout()
        self._note_label = QLabel(P.DISPLAY_ONLY_NOTE, content)
        self._note_label.setStyleSheet("color: #555; font-style: italic;")
        btn_row.addWidget(self._note_label)
        btn_row.addStretch(1)
        self._copy_btn = QPushButton("Copy report", content)
        self._copy_btn.clicked.connect(self._copy_report)
        btn_row.addWidget(self._copy_btn)
        close_btn = QPushButton("Close", content)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        col.addLayout(btn_row)

        temp = QLabel(
            "Handling stages are temporary in V2 — not saved with the project.",
            content,
        )
        temp.setStyleSheet("color: #777; font-size: 8pt;")
        temp.setWordWrap(True)
        col.addWidget(temp)

        col.addStretch(1)
        scroll.setWidget(content)
        self.setCentralWidget(scroll)
        self._apply_enabled_state()

    # ── public API ──

    def set_target(self, elem_id: int) -> None:
        """Point the window at element ``elem_id`` (must be a frame)."""
        model = self._model_provider()
        elem = resolve_single_frame(model, [elem_id])
        self._member = member_spec_from_element(model, elem)
        L = self._member.length
        self._stage_inputs = {
            key: StageInput(stage=key, points=auto_even_points(key, L))
            for key in STAGES
        }
        self._updating = True
        try:
            self._manual_weight.setValue(self._member.self_weight)
            for key, row in self._rows.items():
                row.set_length(L)
                row.load_inputs(self._stage_inputs[key])
        finally:
            self._updating = False
        self._member_label.setText(
            f"Element {self._member.elem_id}"
            + (f" · {self._member.section_name}" if self._member.section_name
               else "")
            + f" · L = {L:.3g} m · self-weight "
              f"{self._member.self_weight:.4g} kN/m"
        )
        self._apply_enabled_state()
        self._recompute_all()

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

    # ── control wiring ──

    def _apply_enabled_state(self) -> None:
        self._manual_weight.setEnabled(not self._auto_weight.isChecked())
        self._custom_angle.setEnabled(
            self._orient_combo.currentData() == ORIENT_CUSTOM,
        )

    def _global_template(self, stage_key: str) -> StageInput:
        """Build a baseline StageInput from the global controls only.

        The per-row controls (points, sling, suction) are layered on top
        in :meth:`_recompute_one`.
        """
        return StageInput(
            stage=stage_key,
            points=(0.0, 0.0),  # overwritten by row
            daf=self._daf.value(),
            manual_weight=(None if self._auto_weight.isChecked()
                           else self._manual_weight.value()),
            extra_udl=self._extra_udl.value(),
            orientation=self._orient_combo.currentData(),
            custom_angle_deg=self._custom_angle.value(),
        )

    # ── signal handlers ──

    def _on_global_changed(self) -> None:
        if self._updating:
            return
        self._apply_enabled_state()
        self._recompute_all()

    def _on_row_changed(self) -> None:
        if self._updating or self._member is None:
            return
        self._recompute_all()

    # ── compute + render ──

    def _recompute_all(self) -> None:
        if self._member is None:
            return
        ordered: list[tuple[StageInput, P.HandlingResult]] = []
        for key in STAGES:
            stage, result = self._recompute_one(key)
            if result is not None:
                ordered.append((stage, result))
        if ordered:
            self._last_report = format_report(self._member, ordered)
        else:
            self._last_report = ""

    def _recompute_one(
        self, key: str,
    ) -> tuple[StageInput, P.HandlingResult | None]:
        template = self._global_template(key)
        stage = self._rows[key].read_inputs(template)
        self._stage_inputs[key] = stage
        try:
            result = compute_handling(self._member, stage)
        except (ValueError, TypeError) as exc:
            self._rows[key].show_error(str(exc))
            self._last_results.pop(key, None)
            return stage, None
        self._last_results[key] = result
        self._rows[key].show_result(self._member, stage, result)
        return stage, result

    # ── report ──

    def _copy_report(self) -> None:
        if not self._last_report:
            return
        from PyQt6.QtWidgets import QApplication
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(self._last_report)
