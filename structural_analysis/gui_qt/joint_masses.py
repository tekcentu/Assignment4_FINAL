"""Assembled Joint Masses inspection window (v0.9.1).

Non-modal singleton ``QMainWindow`` that renders the table produced by
:func:`structural_analysis.mass_inspect.joint_mass_table`. The user can
toggle between the row-sum equivalent (SAP-style) and the raw diagonal
summary; both are pure read-outs of the global mass matrix.

The host (:class:`MainWindow`) owns the singleton and calls
:meth:`refresh` whenever the model changes (same pattern as the
mass / self-weight summary window).
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..mass import MassFormulation
from ..mass_inspect import JointMassReport, Method, joint_mass_table
from ..model import StructuralModel


_COLUMNS = [
    "Node",
    "Mx / ux (kg)",
    "My / uy (kg)",
    "Mrz / rz (kg·m²)",
    "Notes",
]

_DISCLOSURE = (
    "Displayed values are inspection summaries derived from the "
    "assembled global mass matrix. Modal analysis still uses the "
    "full mass matrix."
)


def _format_cell(v: float | str, is_rotation: bool) -> str:
    """Format a NodeMassRow cell for the table.

    Uses scientific notation for very small rotational inertias so they
    stay readable without truncating to 0.000.
    """
    if isinstance(v, str):
        return "—"
    if is_rotation and 0.0 < abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.4f}"


class JointMassesWindow(QMainWindow):
    """Read-only assembled-joint-masses inspector.

    Stays open across model edits; host calls :meth:`refresh` to
    re-build the table from the current model.
    """

    def __init__(
        self,
        parent: QWidget | None,
        model_provider: Callable[[], StructuralModel],
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowTitle("Assembled Joint Masses")
        self.resize(720, 520)

        self._model_provider = model_provider
        self._method: Method = "row_sum"
        # Lumped is the only modal mass formulation in the final-submission
        # build. The Joint Masses inspector inherits that choice so its
        # diagnostic table matches what the modal solver actually sees.
        self._mass_formulation: MassFormulation = "lumped"

        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        header_row = QHBoxLayout()
        self._formulation_label = QLabel("", central)
        self._formulation_label.setStyleSheet("font-size: 9pt; color: #444;")
        header_row.addWidget(self._formulation_label)
        header_row.addStretch(1)

        self._method_group = QButtonGroup(self)
        # ``method`` is a TABLE-DISPLAY toggle (Row-sum / raw diagonal of M),
        # NOT a mass formulation. The mass formulation lives in a separate
        # row below (Consistent / Lumped). The label clarifies this for
        # reviewers (final-submission cleanup) — see docs/mass_inspect for
        # the SAP-mirrored definitions.
        self._rb_rowsum = QRadioButton(
            "Row-sum (Σ block) — SAP-style", central)
        self._rb_diag = QRadioButton(
            "Diagonal — raw M[i,i]", central)
        self._rb_rowsum.setChecked(True)
        self._method_group.addButton(self._rb_rowsum)
        self._method_group.addButton(self._rb_diag)
        self._rb_rowsum.toggled.connect(self._on_method_changed)
        self._rb_diag.toggled.connect(self._on_method_changed)
        header_row.addWidget(QLabel("Table view:", central))
        header_row.addWidget(self._rb_rowsum)
        header_row.addWidget(self._rb_diag)

        header_row.addSpacing(12)
        self._refresh_btn = QPushButton("Refresh", central)
        self._refresh_btn.clicked.connect(self.refresh)
        header_row.addWidget(self._refresh_btn)
        self._close_btn = QPushButton("Close", central)
        self._close_btn.clicked.connect(self.close)
        header_row.addWidget(self._close_btn)
        root.addLayout(header_row)

        # The modal mass-formulation selector was removed for the
        # final-submission build (modal analysis uses lumped / row-sum
        # only). The "Table view" row above is a diagnostic display
        # toggle, not a mass formulation. Backward-compat: the
        # ``_on_formulation_changed`` slot and ``_mass_formulation``
        # attribute are kept; the formulation is fixed to "lumped" so
        # the report matches what the modal solver actually uses.

        self._disclosure_label = QLabel(_DISCLOSURE, central)
        self._disclosure_label.setWordWrap(True)
        self._disclosure_label.setStyleSheet(
            "font-size: 9pt; color: #5a5a5a; font-style: italic; "
            "padding: 4px 0;"
        )
        root.addWidget(self._disclosure_label)

        # Tri-state banner — green (healthy), amber (degenerate
        # element-mass contributions), red (assembly raised). Same
        # idiom as mass_summary._update_status.
        self._status_label = QLabel("", central)
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        self._table = QTableWidget(0, len(_COLUMNS), central)
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        from .table_copy import install_table_copy
        install_table_copy(self._table, include_headers=True)
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
        try:
            report = joint_mass_table(
                model,
                method=self._method,
                mass_formulation=self._mass_formulation,
            )
        except Exception as exc:
            # Mid-edit dangling references (missing node / zero-length
            # element) shouldn't crash a non-modal window.
            self._table.setRowCount(0)
            self._formulation_label.setText("(mass assembly unavailable)")
            self._totals_label.setText("")
            self._set_status(
                f"Error: {type(exc).__name__}: {exc}", level="error",
            )
            return

        self._formulation_label.setText(
            f"Mass formulation: {report.formulation}  ·  "
            f"Summary method: "
            f"{'Row-sum equivalent' if report.method == 'row_sum' else 'Diagonal'}"
        )

        if report.warning is None:
            self._set_status(
                "Mass matrix assembled from current model. "
                "No modal solve was run.",
                level="ok",
            )
        else:
            self._set_status(report.warning, level="warn")

        self._populate(report)

    def _set_status(
        self,
        text: str,
        *,
        level: str,  # "ok" | "warn" | "error"
    ) -> None:
        """Paint the tri-state status banner.

        Colour idiom mirrors ``mass_summary._update_status`` so the two
        diagnostic windows feel consistent.
        """
        colour = {"ok": "#1a6b1a", "warn": "#a06000", "error": "#b00"}[level]
        self._status_label.setStyleSheet(
            f"color: {colour}; font-weight: bold; "
            "font-size: 10pt; padding: 2px 0;"
        )
        self._status_label.setText(text)

    # ── helpers ──

    def _on_method_changed(self, checked: bool) -> None:
        # Both radios in the exclusive group fire `toggled` on every
        # click (one going off, one going on). Acting on both would
        # trigger two full refresh() calls — i.e. two mass-matrix
        # assemblies — per user click. Early-return on the "off" edge.
        if not checked:
            return
        self._method = "row_sum" if self._rb_rowsum.isChecked() else "diagonal"
        self.refresh()

    def _on_formulation_changed(self, checked: bool) -> None:
        # Stub kept for backward compatibility; the formulation row was
        # removed in the lumped-only build, so this slot is no longer
        # connected to any widget. The diagnostic table is always built
        # from the lumped mass matrix (same as the modal solver).
        del checked  # unused
        self._mass_formulation = "lumped"
        self.refresh()

    def _populate(self, report: JointMassReport) -> None:
        self._table.setRowCount(len(report.rows))
        for r, row in enumerate(report.rows):
            cells = [
                str(row.node_id),
                _format_cell(row.values["ux"], is_rotation=False),
                _format_cell(row.values["uy"], is_rotation=False),
                _format_cell(row.values["rz"], is_rotation=True),
                row.notes(),
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c in (1, 2, 3):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter,
                    )
                self._table.setItem(r, c, item)

        t = report.totals_kg
        self._totals_label.setText(
            f"Σ Mx = {t['ux']:.4f} kg     "
            f"Σ My = {t['uy']:.4f} kg     "
            f"Σ Mrz = {t['rz']:.4e} kg·m²     "
            f"Active modal DOFs = {report.n_free_dofs} "
            f"(of {report.n_total_dofs} assembled)"
        )
