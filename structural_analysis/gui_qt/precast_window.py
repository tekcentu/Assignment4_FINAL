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


SLING_ANGLE_TOOLTIP = (
    "Sling angle from horizontal — T/H only.\n\n"
    "Used for rigging force calculation only: it changes the sling "
    "tension T and horizontal component H. It does NOT change vertical "
    "reactions, shear, or moment in the simplified horizontal handling "
    "model.\n\n"
    "Not the same as the concrete insert / anchor / embedded-loop angle.\n\n"
    "Auto mode: angle = atan(hook_height / ((x2 − x1) / 2))."
)


class _NoScrollSpinBox(QDoubleSpinBox):
    """A spin box that ignores the mouse wheel.

    The window lives inside a :class:`QScrollArea`; without this, hovering
    a spin box while scrolling the sheet would silently change its value
    instead of scrolling the page. Ignoring the wheel lets the event
    bubble up to the scroll area. Values are still editable by typing or
    the up/down arrows.
    """

    def wheelEvent(self, event):  # noqa: N802 (Qt override)
        event.ignore()


class _NoScrollComboBox(QComboBox):
    """A combo box that ignores the mouse wheel (see _NoScrollSpinBox)."""

    def wheelEvent(self, event):  # noqa: N802 (Qt override)
        event.ignore()


def _spin(parent, lo, hi, val, step, suffix, *, decimals=3) -> QDoubleSpinBox:
    sp = _NoScrollSpinBox(parent)
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

        # Title bar: stage name · status chip · enabled toggle.
        title_bar = QHBoxLayout()
        title = QLabel(f"<b>{STAGE_LABELS[stage_key]}</b>", self)
        title_bar.addWidget(title)
        title_bar.addStretch(1)
        self.status_chip = QLabel("", self)
        self._set_chip("OK", "OK")
        title_bar.addWidget(self.status_chip)
        self.enabled_cb = QCheckBox("Enabled", self)
        self.enabled_cb.setChecked(True)
        self.enabled_cb.toggled.connect(self._fire)
        title_bar.addWidget(self.enabled_cb)
        outer.addLayout(title_bar)

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
        controls.addWidget(QLabel("DAF", self), row, 5)
        self.daf = _spin(self, 0.1, 5.0, 1.0, 0.05, "", decimals=2)
        controls.addWidget(self.daf, row, 6)

        # Lifting-only controls live on the lifting row.
        self.sling_angle: QDoubleSpinBox | None = None
        self.suction: QDoubleSpinBox | None = None
        self.angle_mode: QComboBox | None = None
        self.hook_height: QDoubleSpinBox | None = None
        if stage_key == STAGE_LIFTING:
            row += 1
            angle_label = QLabel(
                "Sling angle from horizontal (T/H only)", self)
            angle_label.setToolTip(SLING_ANGLE_TOOLTIP)
            controls.addWidget(angle_label, row, 0)
            self.sling_angle = _spin(self, 1.0, 90.0, 60.0, 1.0, " °",
                                     decimals=1)
            self.sling_angle.setToolTip(SLING_ANGLE_TOOLTIP)
            controls.addWidget(self.sling_angle, row, 1)
            controls.addWidget(
                QLabel("Bed adhesion / suction (kN/m)", self), row, 2)
            self.suction = _spin(self, 0.0, 1e6, 0.0, 0.5, " kN/m")
            self.suction.setToolTip(
                "Downward bed adhesion / form suction on the lifting "
                "stage only. 0.0 = off.")
            controls.addWidget(self.suction, row, 3)

            row += 1
            mode_label = QLabel("Angle mode", self)
            mode_label.setToolTip(SLING_ANGLE_TOOLTIP)
            controls.addWidget(mode_label, row, 0)
            self.angle_mode = _NoScrollComboBox(self)
            self.angle_mode.addItem("Manual angle", "manual")
            self.angle_mode.addItem("Auto from hook height", "auto")
            self.angle_mode.setToolTip(SLING_ANGLE_TOOLTIP)
            controls.addWidget(self.angle_mode, row, 1)
            controls.addWidget(QLabel("Hook height above member (m)", self),
                               row, 2)
            self.hook_height = _spin(self, 0.0, 1e3, 2.0, 0.1, " m",
                                     decimals=2)
            self.hook_height.setToolTip(
                "Vertical distance from the lift points up to the hook. "
                "Used in Auto mode: angle = atan(hook_height / "
                "((x2 − x1) / 2)). Ignored in Manual mode.")
            controls.addWidget(self.hook_height, row, 3)

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

        self.stress = QLabel("", self)
        self.stress.setWordWrap(True)
        outer.addWidget(self.stress)

        self.warning = QLabel("", self)
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #b00;")
        outer.addWidget(self.warning)

        # Wire up edits.
        for sp in (self.p1, self.p2, self.daf):
            sp.valueChanged.connect(self._fire)
        if self.sling_angle is not None:
            self.sling_angle.valueChanged.connect(self._fire)
        if self.suction is not None:
            self.suction.valueChanged.connect(self._fire)
        if self.hook_height is not None:
            self.hook_height.valueChanged.connect(self._fire)
        if self.angle_mode is not None:
            self.angle_mode.currentIndexChanged.connect(self._fire)
        self._sync_auto_angle_state()

    # ── helpers ──

    def _set_chip(self, text: str, kind: str) -> None:
        """Style the title-bar status chip. ``kind`` ∈ {OK, WARNING, DISABLED}."""
        palette = {
            "OK": ("#1a7f37", "#e6f4ea"),
            "WARNING": ("#b00020", "#fdecea"),
            "DISABLED": ("#777", "#eeeeee"),
        }
        fg, bg = palette.get(kind, palette["OK"])
        self.status_chip.setText(text)
        self.status_chip.setStyleSheet(
            f"color: {fg}; background: {bg}; border-radius: 7px; "
            "padding: 1px 9px; font-weight: bold;"
        )

    def _fire(self) -> None:
        if self._updating:
            return
        self._sync_auto_angle_state()
        self._sync_auto_angle_value()
        self._on_changed()

    def _sync_auto_angle_state(self) -> None:
        """Disable the manual sling-angle spinbox when Auto mode is active."""
        if self.sling_angle is None or self.angle_mode is None:
            return
        auto = self.angle_mode.currentData() == "auto"
        self.sling_angle.setEnabled(not auto)
        if self.hook_height is not None:
            self.hook_height.setEnabled(auto)

    def _sync_auto_angle_value(self) -> None:
        """In Auto mode, recompute the sling angle from hook height + spacing
        and push it into the (read-only) angle spinbox."""
        if (self.sling_angle is None or self.angle_mode is None
                or self.hook_height is None):
            return
        if self.angle_mode.currentData() != "auto":
            return
        half_spacing = abs(self.p2.value() - self.p1.value()) / 2.0
        h = self.hook_height.value()
        if half_spacing <= 1e-9:
            # Coincident points are rejected by the engine anyway; pin to
            # vertical so the spinbox stays sane meanwhile.
            ang = 90.0
        else:
            ang = math.degrees(math.atan(h / half_spacing))
        ang = max(min(ang, 90.0), 1.0)  # spinbox bounds
        self._updating = True
        try:
            self.sling_angle.setValue(ang)
        finally:
            self._updating = False

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

    @property
    def enabled(self) -> bool:
        return self.enabled_cb.isChecked()

    def load_inputs(self, stage: StageInput) -> None:
        self._updating = True
        try:
            x1, x2 = stage.points
            self.p1.setValue(x1)
            self.p2.setValue(x2)
            self.daf.setValue(stage.daf)
            self.enabled_cb.setChecked(stage.enabled)
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
            daf=self.daf.value(),
            enabled=self.enabled_cb.isChecked(),
            manual_weight=template.manual_weight,
            suction=suction,
            extra_udl=template.extra_udl,
            orientation=template.orientation,
            custom_angle_deg=template.custom_angle_deg,
            stress_check_enabled=template.stress_check_enabled,
            allowable_tensile_mpa=template.allowable_tensile_mpa,
            manual_y_top=template.manual_y_top,
            manual_y_bottom=template.manual_y_bottom,
        )

    def set_greyed(self, greyed: bool) -> None:
        """Enable/disable every input except the Enabled toggle itself."""
        live = not greyed
        for sp in (self.p1, self.p2, self.daf, self.auto_btn):
            sp.setEnabled(live)
        if self.sling_angle is not None:
            self.sling_angle.setEnabled(live)
        if self.suction is not None:
            self.suction.setEnabled(live)
        if self.hook_height is not None:
            self.hook_height.setEnabled(live)
        if self.angle_mode is not None:
            self.angle_mode.setEnabled(live)
        # Auto mode keeps the angle spinbox disabled (it shows a computed
        # value) — re-apply that after the bulk re-enable.
        if live:
            self._sync_auto_angle_state()

    def show_disabled(self) -> None:
        """Grey the row out: clear text, chip → DISABLED, blank the axes."""
        self.summary.setText("")
        self.stress.setText("")
        self.warning.setText("")
        self._set_chip("DISABLED", "DISABLED")
        for ax in (self._ax_member, self._ax_v, self._ax_m):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        self._ax_member.text(
            0.5, 0.5, "Stage disabled", ha="center", va="center",
            transform=self._ax_member.transAxes, color="#999", fontsize=10,
        )
        self.canvas.draw_idle()

    # ── rendering ──

    def show_error(self, msg: str) -> None:
        self.summary.setText("")
        self.stress.setText("")
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
        spacing = abs(result.reactions[1][0] - result.reactions[0][0])
        parts.append(f"Support spacing = {spacing:.4g} m")
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
        cracking = (result.stress_check.cracking_status == "CRACKING WARNING")
        if result.warnings or cracking:
            self._set_chip("WARNING", "WARNING")
        else:
            self._set_chip("OK", "OK")
        self._render_stress(result.stress_check)
        self._draw(member, stage, result)

    def _render_stress(self, sc: P.StressCheck) -> None:
        if sc.skipped:
            self.stress.setText(
                f"Cracking check: skipped — {sc.skip_reason}"
            )
            self.stress.setStyleSheet("color: #777;")
            return
        peak_top = (
            f"σ_top_max = {sc.max_top_tensile_mpa:.3g} MPa "
            f"@ x = {sc.max_top_tensile_x:.3g} m"
        )
        peak_bot = (
            f"σ_bot_max = {sc.max_bottom_tensile_mpa:.3g} MPa "
            f"@ x = {sc.max_bottom_tensile_x:.3g} m"
        )
        gate = (
            f"σ_allow = {sc.allowable_tensile_mpa:.3g} MPa · "
            f"ratio = {sc.cracking_ratio:.3g} · {sc.cracking_status}"
        )
        self.stress.setText(f"{peak_top}   |   {peak_bot}   |   {gate}")
        self.stress.setStyleSheet(
            "color: #b00; font-weight: bold;"
            if sc.cracking_status == "CRACKING WARNING"
            else "color: #1a7f37;"
        )

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
        # Member line.
        ax.plot([0.0, L * ca], [0.0, L * sa], color="#1f3a5f", lw=4, zorder=4)

        # UDL band: downward gravity arrows along the span + a label. Arrows
        # point straight down (gravity) regardless of the display angle.
        band = 0.13 * L
        for i in range(0, 9):
            x = i * L / 8.0
            px, py = x * ca, x * sa
            ax.annotate(
                "", xy=(px, py), xytext=(px, py + band),
                arrowprops=dict(arrowstyle="->", color="#c0508a", lw=0.9),
                zorder=3,
            )
        ax.plot([0.0, L * ca], [band, L * sa + band],
                color="#c0508a", lw=1.0, alpha=0.8, zorder=3)
        ax.text((L / 2.0) * ca, (L / 2.0) * sa + band * 1.25,
                f"w = {result.udl_per_m:.3g} kN/m",
                ha="center", va="bottom", fontsize=7, color="#c0508a")

        # Supports / lift points, upward reaction arrows + value labels, and
        # (lifting only) sling lines with T / H labels.
        arr = 0.22 * L
        hx, hy = (L / 2.0) * ca, (L / 2.0) * sa + 0.32 * L
        for i, (x, r) in enumerate(result.reactions):
            px, py = x * ca, x * sa
            if result.stage == STAGE_LIFTING:
                ax.plot([px, hx], [py, hy], color="#1a7f37", lw=1.2, zorder=3)
                ax.plot([px], [py], marker="v", color="#1a7f37", ms=9,
                        zorder=5)
                if result.sling_tensions:
                    mx, my = (px + hx) / 2.0, (py + hy) / 2.0
                    ax.annotate(
                        f"T={result.sling_tensions[i]:.3g} kN\n"
                        f"H={result.sling_horizontal[i]:.3g} kN",
                        xy=(mx, my), fontsize=6.5, color="#1a7f37",
                        ha="left", va="center",
                    )
            else:
                ax.plot([px], [py - 0.05 * L], marker="^", color="#444",
                        ms=12, zorder=5)
            # Upward reaction arrow + value (the support's vertical reaction).
            ax.annotate(
                "", xy=(px, py), xytext=(px, py - arr),
                arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.6),
                zorder=4,
            )
            ax.text(px, py - arr, f"R={r:.3g} kN", ha="center", va="top",
                    fontsize=7, color="#1565c0")
        if result.stage == STAGE_LIFTING and result.reactions:
            ax.plot([hx], [hy], marker="o", color="#1a7f37", ms=7, zorder=5)

        status = "WARNING" if (result.warnings or result.stress_check
                               .cracking_status == "CRACKING WARNING") else "OK"
        ax.set_title(f"{STAGE_LABELS.get(result.stage, result.stage)} — "
                     f"{status}", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.18)
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
            # Pin the x-axis to the member span. Without this, switching to
            # a shorter element leaves the previous longer member's xlim
            # stale (matplotlib doesn't auto-shrink past axhline's extent).
            axd.set_xlim(-0.02 * L, 1.02 * L)
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
        # Global Units V1: the precast window keeps its own copy of the
        # active preset so set_units_preset can be called either by the
        # parent MainWindow when the user flips the global selector or
        # directly by tests. Affects the copied report only — the
        # numeric engine and the in-window sketch numbers stay in kN/m.
        self._units_preset: str = "kN_m"
        if parent is not None and hasattr(parent, "_units_preset"):
            self._units_preset = parent._units_preset

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

        self._orient_combo = _NoScrollComboBox(glob)
        self._orient_combo.addItem("Horizontal handling", ORIENT_HORIZONTAL)
        self._orient_combo.addItem("Original model angle", ORIENT_MODEL)
        self._orient_combo.addItem("Custom angle", ORIENT_CUSTOM)
        self._orient_combo.currentIndexChanged.connect(self._on_global_changed)
        form.addRow("Display orientation", self._orient_combo)

        self._custom_angle = _spin(glob, -180.0, 180.0, 0.0, 1.0, " °",
                                   decimals=1)
        form.addRow("Custom angle", self._custom_angle)

        self._auto_weight = QCheckBox("Auto (from section)", glob)
        self._auto_weight.setChecked(True)
        self._auto_weight.toggled.connect(self._on_global_changed)
        form.addRow("Self-weight", self._auto_weight)
        self._manual_weight = _spin(glob, 0.0, 1e6, 0.0, 0.5, " kN/m")
        form.addRow("Manual weight", self._manual_weight)

        self._extra_udl = _spin(glob, 0.0, 1e6, 0.0, 0.5, " kN/m")
        form.addRow("Extra handling UDL", self._extra_udl)

        # ── Flexural cracking check (V1) ──
        self._stress_enabled = QCheckBox("Run elastic σ = M·y / I check", glob)
        self._stress_enabled.setChecked(True)
        self._stress_enabled.toggled.connect(self._on_global_changed)
        form.addRow("Cracking check", self._stress_enabled)

        self._allowable_tensile = _spin(glob, 0.0, 100.0, 2.6, 0.1,
                                        " MPa", decimals=2)
        form.addRow("Allowable σ_t", self._allowable_tensile)

        self._manual_y = QCheckBox("Manual y_top / y_bottom", glob)
        self._manual_y.setChecked(False)
        self._manual_y.toggled.connect(self._on_global_changed)
        form.addRow("Fiber distances", self._manual_y)

        self._y_top = _spin(glob, 0.0001, 100.0, 0.2, 0.01, " m", decimals=4)
        form.addRow("y_top", self._y_top)
        self._y_bottom = _spin(glob, 0.0001, 100.0, 0.2, 0.01, " m", decimals=4)
        form.addRow("y_bottom", self._y_bottom)

        for sp in (self._custom_angle,
                   self._manual_weight, self._extra_udl,
                   self._allowable_tensile, self._y_top, self._y_bottom):
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
            # Seed the auto y-distances from section depth so the user sees
            # the values that will be used if they later toggle to manual.
            half_depth = (self._member.depth / 2.0
                          if self._member.depth > 0.0 else 0.2)
            self._y_top.setValue(half_depth)
            self._y_bottom.setValue(half_depth)
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

    def set_units_preset(self, preset_id: str) -> None:
        """Update the unit preset used for the copied report. The in-window
        sketch and numeric engine stay in kN-m (V1 scope: report only)."""
        self._units_preset = preset_id
        # Re-render the copied report so a subsequent _copy_report uses
        # the new units. The sketch axes don't depend on the preset in V1.
        if self._member is not None:
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
        check_on = self._stress_enabled.isChecked()
        self._allowable_tensile.setEnabled(check_on)
        self._manual_y.setEnabled(check_on)
        manual = check_on and self._manual_y.isChecked()
        self._y_top.setEnabled(manual)
        self._y_bottom.setEnabled(manual)

    def _global_template(self, stage_key: str) -> StageInput:
        """Build a baseline StageInput from the global controls only.

        The per-row controls (points, sling, suction) are layered on top
        in :meth:`_recompute_one`.
        """
        manual_y = self._manual_y.isChecked()
        return StageInput(
            stage=stage_key,
            points=(0.0, 0.0),  # overwritten by row
            manual_weight=(None if self._auto_weight.isChecked()
                           else self._manual_weight.value()),
            extra_udl=self._extra_udl.value(),
            orientation=self._orient_combo.currentData(),
            custom_angle_deg=self._custom_angle.value(),
            stress_check_enabled=self._stress_enabled.isChecked(),
            allowable_tensile_mpa=self._allowable_tensile.value(),
            manual_y_top=self._y_top.value() if manual_y else None,
            manual_y_bottom=self._y_bottom.value() if manual_y else None,
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
            row = self._rows[key]
            if not row.enabled:
                # Skip from calculation + report; grey the row out.
                row.set_greyed(True)
                row.show_disabled()
                self._stage_inputs[key] = row.read_inputs(
                    self._global_template(key))
                self._last_results.pop(key, None)
                continue
            row.set_greyed(False)
            stage, result = self._recompute_one(key)
            if result is not None:
                ordered.append((stage, result))
        if ordered:
            self._last_report = format_report(
                self._member, ordered, unit_preset=self._units_preset)
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
