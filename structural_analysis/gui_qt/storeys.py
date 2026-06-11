"""Storey manager (v0.33) — named z-levels for the working depth.

A storey is GUI metadata only: ``(name, z)`` pairs the user defines
once and then jumps between, instead of retyping View → Working
depth values. Persisted in the ``.spa.json`` project's view state;
never passed to the solver.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)


def storey_name_for_depth(
    storeys: list[tuple[str, float]], depth: float,
    tol: float = 1e-9,
) -> str | None:
    """Name of the storey at ``depth``, or None. Qt-free for tests."""
    for name, z in storeys:
        if abs(z - depth) < tol:
            return name
    return None


def normalized_storeys(
    rows: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Validate + sort storey rows (by z ascending).

    Raises ValueError for empty/duplicate names or duplicate levels —
    a storey list with two "Level 1" entries (or two storeys at the
    same z) can only confuse the depth jump.
    """
    seen_names: set[str] = set()
    seen_z: list[float] = []
    out: list[tuple[str, float]] = []
    for name, z in rows:
        name = name.strip()
        if not name:
            raise ValueError("Storey names must be non-empty.")
        if name in seen_names:
            raise ValueError(f"Duplicate storey name {name!r}.")
        if any(abs(z - other) < 1e-9 for other in seen_z):
            raise ValueError(
                f"Two storeys share the level z = {z:g} m."
            )
        seen_names.add(name)
        seen_z.append(z)
        out.append((name, float(z)))
    return sorted(out, key=lambda row: row[1])


class StoreyManagerDialog(QDialog):
    """Edit the storey list and optionally jump the working depth.

    ``result_storeys`` holds the accepted list; ``activated_depth`` is
    the z the user asked to jump to (None when no jump requested).
    """

    def __init__(self, parent, *, storeys: list[tuple[str, float]],
                 current_depth: float) -> None:
        super().__init__(parent)
        self.setWindowTitle("Storeys (named z-levels)")
        self.result_storeys: list[tuple[str, float]] | None = None
        self.activated_depth: float | None = None
        self._current_depth = current_depth

        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            "Storeys name the z-levels you build on. 'Set working "
            "depth' jumps View → Working depth to the selected storey.",
            self,
        ))

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["Name", "z (m)"])
        self._table.horizontalHeader().setStretchLastSection(True)
        for name, z in storeys:
            self._append_row(name, z)
        v.addWidget(self._table)

        add_row = QHBoxLayout()
        self._name_entry = QLineEdit(self)
        self._name_entry.setPlaceholderText("e.g. Level 1")
        self._z_spin = QDoubleSpinBox(self)
        self._z_spin.setRange(-1e9, 1e9)
        self._z_spin.setDecimals(3)
        add_btn = QPushButton("Add", self)
        add_btn.clicked.connect(self._on_add)
        add_row.addWidget(self._name_entry, 2)
        add_row.addWidget(self._z_spin, 1)
        add_row.addWidget(add_btn)
        v.addLayout(add_row)

        btn_row = QHBoxLayout()
        remove_btn = QPushButton("Remove selected", self)
        remove_btn.clicked.connect(self._on_remove)
        activate_btn = QPushButton("Set working depth", self)
        activate_btn.clicked.connect(self._on_activate)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(activate_btn)
        btn_row.addStretch(1)
        v.addLayout(btn_row)

        self._error = QLabel("", self)
        self._error.setStyleSheet("color: #c03030;")
        v.addWidget(self._error)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    # ── rows ──

    def _append_row(self, name: str, z: float) -> None:
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._table.setItem(r, 0, QTableWidgetItem(name))
        self._table.setItem(r, 1, QTableWidgetItem(f"{z:g}"))

    def _rows(self) -> list[tuple[str, float]]:
        rows: list[tuple[str, float]] = []
        for r in range(self._table.rowCount()):
            name_item = self._table.item(r, 0)
            z_item = self._table.item(r, 1)
            name = name_item.text() if name_item else ""
            try:
                z = float(z_item.text()) if z_item else 0.0
            except ValueError:
                raise ValueError(
                    f"Row {r + 1}: z = {z_item.text()!r} is not a number."
                )
            rows.append((name, z))
        return rows

    # ── actions ──

    def _on_add(self) -> None:
        name = self._name_entry.text().strip()
        if not name:
            self._error.setText("Enter a storey name first.")
            return
        self._append_row(name, float(self._z_spin.value()))
        self._name_entry.clear()
        self._error.setText("")

    def _on_remove(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)

    def _on_activate(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            self._error.setText("Select a storey row first.")
            return
        z_item = self._table.item(row, 1)
        try:
            self.activated_depth = float(z_item.text())
        except (TypeError, ValueError):
            self._error.setText("Selected row has an invalid z value.")
            return
        self._error.setText("")
        self._on_accept()

    def _on_accept(self) -> None:
        try:
            self.result_storeys = normalized_storeys(self._rows())
        except ValueError as e:
            self._error.setText(str(e))
            self.result_storeys = None
            return
        self.accept()
