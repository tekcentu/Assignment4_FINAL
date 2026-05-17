"""PyQt6 results window for free-vibration modal analysis.

Shows a frequency/period table per mode and lets the user pick which
mode to display on the canvas (with a deformation-scale slider). The
dialog is non-modal so the user can keep interacting with the model.

The optional time-domain animation toggle is wired here; if matplotlib
animation is not available, the toggle silently keeps the static
overlay (the static plot is the required deliverable per the proposal).
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..modal import ModalResult


class ModalResultsDialog(QDialog):
    """Non-modal window displaying a :class:`ModalResult`."""

    def __init__(
        self,
        parent: QWidget | None,
        result: ModalResult,
        on_select: Callable[[int, float], None],
        on_close: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Modal analysis results")
        self.setModal(False)
        self._result = result
        self._on_select = on_select
        self._on_close = on_close

        v = QVBoxLayout(self)

        v.addWidget(QLabel(
            f"<b>{result.title or 'Model'}</b> · {result.n_modes} modes · "
            f"normalisation: {result.normalisation}",
            self,
        ))

        self._tree = QTreeWidget(self)
        self._tree.setHeaderLabels(
            ["mode", "f (Hz)", "T (s)", "ω (rad/s)"]
        )
        self._tree.header().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        for k in range(result.n_modes):
            QTreeWidgetItem(self._tree, [
                str(k + 1),
                f"{float(result.frequencies[k]):.6g}",
                f"{float(result.periods[k]):.6g}",
                f"{float(result.omegas[k]):.6g}",
            ])
        v.addWidget(self._tree)

        # Mode selector — built before connecting the tree's selection
        # signal so the very first setCurrentItem can read it safely.
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:", self))
        self._mode_spin = QSpinBox(self)
        self._mode_spin.setRange(1, max(1, result.n_modes))
        self._mode_spin.setValue(1)
        self._mode_spin.valueChanged.connect(self._on_mode_spun)
        row.addWidget(self._mode_spin)

        # Connect the tree signal now that _mode_spin exists, then set
        # the initial selection.
        self._tree.currentItemChanged.connect(self._on_row_changed)
        if result.n_modes > 0:
            self._tree.setCurrentItem(self._tree.topLevelItem(0))

        row.addSpacing(20)
        row.addWidget(QLabel("Scale ×:", self))
        self._scale_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._scale_slider.setRange(1, 100)      # 0.1× … 10×
        self._scale_slider.setValue(10)          # → ×1.0 default
        self._scale_label = QLabel("1.0", self)
        self._scale_slider.valueChanged.connect(self._on_scale_changed)
        row.addWidget(self._scale_slider, 1)
        row.addWidget(self._scale_label)
        v.addLayout(row)

        # Optional animation toggle (polish, opt-in).
        self._anim_box = QCheckBox(
            "Animate (sinusoidal sweep at ω) — optional polish", self,
        )
        self._anim_box.stateChanged.connect(self._on_anim_toggled)
        v.addWidget(self._anim_box)

        # Close button
        bb = QHBoxLayout()
        bb.addStretch(1)
        close = QPushButton("Close", self)
        close.clicked.connect(self._on_close_clicked)
        bb.addWidget(close)
        v.addLayout(bb)

        # Animation timer (created lazily — see _on_anim_toggled).
        self._anim_timer: QTimer | None = None
        self._anim_phase: float = 0.0

        # Push the initial selection out.
        self._push_view()

    # ── slot helpers ──

    def _on_row_changed(self, current, _previous) -> None:
        if current is None:
            return
        idx = self._tree.indexOfTopLevelItem(current) + 1
        if idx != self._mode_spin.value():
            self._mode_spin.setValue(idx)

    def _on_mode_spun(self, value: int) -> None:
        # Keep tree selection in sync with the spinner.
        item = self._tree.topLevelItem(value - 1)
        if item is not None and self._tree.currentItem() is not item:
            self._tree.setCurrentItem(item)
        self._push_view()

    def _on_scale_changed(self, value: int) -> None:
        scale = value / 10.0
        self._scale_label.setText(f"{scale:.1f}")
        self._push_view()

    def _on_close_clicked(self) -> None:
        if self._anim_timer is not None:
            self._anim_timer.stop()
        self._on_close()
        self.accept()

    def closeEvent(self, event) -> None:
        if self._anim_timer is not None:
            self._anim_timer.stop()
        self._on_close()
        super().closeEvent(event)

    # ── view-pushing ──

    def _push_view(self, *, anim_factor: float = 1.0) -> None:
        mode_idx = self._mode_spin.value() - 1
        base_scale = self._scale_slider.value() / 10.0
        self._on_select(mode_idx, base_scale * anim_factor)

    # ── optional animation (polish) ──

    def _on_anim_toggled(self, state) -> None:
        if state == Qt.CheckState.Checked.value:
            if self._anim_timer is None:
                self._anim_timer = QTimer(self)
                self._anim_timer.timeout.connect(self._tick_anim)
            self._anim_phase = 0.0
            # ~25 fps; the sinusoidal phase is independent of the
            # mode's actual frequency to keep the visual readable.
            self._anim_timer.start(40)
        else:
            if self._anim_timer is not None:
                self._anim_timer.stop()
            # Snap back to the user's chosen static scale.
            self._push_view(anim_factor=1.0)

    def _tick_anim(self) -> None:
        import math
        self._anim_phase += 0.15
        factor = math.cos(self._anim_phase)
        self._push_view(anim_factor=factor)
