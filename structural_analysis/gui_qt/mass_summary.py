"""Mass / self-weight summary window (v0.9.0).

A non-modal singleton ``QMainWindow`` showing one row per element with
its effective material, density, area, length, mass, and self-weight,
plus a global totals footer and a header strip with the gravity
constant, direction, and whether self-weight is currently enabled in
the solver.

The window holds a callable that returns the current
:class:`StructuralModel`; clicking Refresh re-reads the model and
re-populates the table. The host is responsible for calling
``refresh()`` after edits that could change ρ / A / L or the element
set.
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..element import FrameElement2D, TrussElement2D
from ..model import STANDARD_GRAVITY, StructuralModel, effective_material


_COLUMNS = [
    "Element",
    "Kind",
    "Section",
    "Effective material",
    "ρ (kg/m³)",
    "A (m²)",
    "L (m)",
    "Mass (kg)",
    "Self-weight (kN/m)",
    "Total weight (kN)",
]


class MassSummaryWindow(QMainWindow):
    """Read-only mass / self-weight breakdown for the current model."""

    def __init__(
        self,
        parent: QWidget | None,
        model_provider: Callable[[], StructuralModel],
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowTitle("Mass / self-weight summary")
        self.resize(900, 480)

        self._model_provider = model_provider

        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        header_row = QHBoxLayout()
        self._fixed_label = QLabel(
            f"g = {STANDARD_GRAVITY:g} m/s²  ·  Gravity: Global −Y",
            central,
        )
        self._fixed_label.setStyleSheet("font-size: 9pt; color: #444;")
        header_row.addWidget(self._fixed_label)
        header_row.addStretch(1)
        self._refresh_btn = QPushButton("Refresh", central)
        self._refresh_btn.clicked.connect(self.refresh)
        header_row.addWidget(self._refresh_btn)
        self._close_btn = QPushButton("Close", central)
        self._close_btn.clicked.connect(self.close)
        header_row.addWidget(self._close_btn)
        root.addLayout(header_row)

        self._status_label = QLabel("", central)
        self._status_label.setStyleSheet("font-weight: bold;")
        root.addWidget(self._status_label)

        self._table = QTableWidget(0, len(_COLUMNS), central)
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows,
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(True)
        root.addWidget(self._table)

        self._totals_label = QLabel("", central)
        self._totals_label.setStyleSheet(
            "font-size: 10pt; padding-top: 6px;"
        )
        root.addWidget(self._totals_label)

        self.setCentralWidget(central)
        self.refresh()

    # ── public API ──

    def refresh(self) -> None:
        """Re-read the model and re-populate the table + totals."""
        model = self._model_provider()
        self._update_status(model)

        rows = self._collect_rows(model)
        self._table.setRowCount(len(rows))
        total_mass = 0.0
        total_weight_kN = 0.0
        for r, row in enumerate(rows):
            for c, val in enumerate(row["display"]):
                item = QTableWidgetItem(val)
                if c >= 4:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter,
                    )
                self._table.setItem(r, c, item)
            total_mass += row["mass_kg"]
            total_weight_kN += row["weight_kN"]

        self._totals_label.setText(
            f"Total mass: {total_mass:.3f} kg     "
            f"Total self-weight: {total_weight_kN:.3f} kN"
        )

    # ── helpers ──

    def _update_status(self, model: StructuralModel) -> None:
        if getattr(model, "include_self_weight", False):
            self._status_label.setText("Self-weight: ENABLED in solver")
            self._status_label.setStyleSheet(
                "color: #1a6b1a; font-weight: bold;"
            )
        else:
            self._status_label.setText("Self-weight: DISABLED in solver")
            self._status_label.setStyleSheet(
                "color: #b00; font-weight: bold;"
            )

    def _collect_rows(self, model: StructuralModel) -> list[dict]:
        rows: list[dict] = []
        for elem in model.elements:
            if isinstance(elem, FrameElement2D):
                kind = "FRAME"
            elif isinstance(elem, TrussElement2D):
                kind = "TRUSS"
            else:
                kind = type(elem).__name__

            section = model.sections.get(getattr(elem, "section_id", None))
            section_name = (
                section.name if section and section.name
                else (f"id {section.id}" if section else "—")
            )

            try:
                mat = effective_material(model, elem)
                mat_name = mat.name or f"id {mat.id}"
                rho = float(mat.density)
            except (KeyError, AttributeError):
                # Mid-edit dangling reference. Fall back to the
                # element's snapshot so the window stays useful.
                mat_name = "—"
                rho = float(getattr(elem, "rho", 0.0))

            A = float(getattr(elem, "A", 0.0))
            try:
                L, _, _ = elem.length_cos_sin(model.nodes)
            except KeyError:
                L = 0.0

            mass = rho * A * L  # kg
            w_per_m = rho * A * STANDARD_GRAVITY / 1000.0  # kN/m
            weight_kN = w_per_m * L

            rows.append({
                "display": [
                    str(elem.id),
                    kind,
                    section_name,
                    mat_name,
                    f"{rho:.3f}",
                    f"{A:.6g}",
                    f"{L:.4f}",
                    f"{mass:.4f}",
                    f"{w_per_m:.4f}",
                    f"{weight_kN:.4f}",
                ],
                "mass_kg": mass,
                "weight_kN": weight_kN,
            })
        return rows
