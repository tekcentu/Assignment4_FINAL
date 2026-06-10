"""Read-only Qt dialogs for result export tables."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..element import FrameElement2D
from ..gui_common.results_view import case_combo_entries, resolve_view
from ..model import AnalysisResult, StructuralModel
from ..multi_case_result import MultiCaseAnalysisResult
from .result_export_data import (
    MEMBER_STATION_HEADERS,
    NODE_RESULT_HEADERS,
    assert_clean_headers,
    format_export_value,
    member_station_metadata,
    member_station_rows,
    node_result_rows,
    write_csv,
)
from .table_copy import copy_view_to_clipboard, install_table_copy


def _set_item(table: QTableWidget, row: int, col: int, value: object) -> None:
    item = QTableWidgetItem(format_export_value(value))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    table.setItem(row, col, item)


def _populate_table(
    table: QTableWidget,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    assert_clean_headers(headers)
    table.clear()
    table.setColumnCount(len(headers))
    table.setRowCount(len(rows))
    table.setHorizontalHeaderLabels(list(headers))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            _set_item(table, r, c, value)
    table.resizeColumnsToContents()


class _ResultExportDialogBase(QDialog):
    """Shared read-only result table dialog plumbing."""

    headers: Sequence[str] = ()

    def __init__(
        self,
        parent: QWidget | None,
        *,
        model: StructuralModel,
        multi_result: MultiCaseAnalysisResult | None,
        active_case: str = "DEFAULT",
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._multi_result = multi_result
        self._active_case = active_case
        self._rows: list[list[object]] = []

        self._outer = QVBoxLayout(self)
        self._case_combo = QComboBox(self)
        case_row = QHBoxLayout()
        case_row.addWidget(QLabel("Case / Combination:"))
        entries = case_combo_entries(model, multi_result)
        for label, raw in entries:
            self._case_combo.addItem(label, raw)
        if entries:
            idx = self._case_combo.findData(active_case)
            self._case_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._case_combo.currentIndexChanged.connect(self._refresh_rows)
        case_row.addWidget(self._case_combo, stretch=1)
        self._outer.addLayout(case_row)

        self._metadata_label = QLabel("")
        self._metadata_label.setWordWrap(True)
        self._outer.addWidget(self._metadata_label)

        self._table = QTableWidget(self)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        install_table_copy(self._table, include_headers=True)
        self._outer.addWidget(self._table, stretch=1)

        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._outer.addWidget(self._note)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._copy_btn = QPushButton("Copy TSV", self)
        self._copy_btn.clicked.connect(self.copy_tsv)
        btn_row.addWidget(self._copy_btn)
        self._csv_btn = QPushButton("Export CSV…", self)
        self._csv_btn.clicked.connect(self._export_csv_dialog)
        btn_row.addWidget(self._csv_btn)
        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self._buttons.rejected.connect(self.close)
        btn_row.addWidget(self._buttons)
        self._outer.addLayout(btn_row)

    @property
    def rows(self) -> list[list[object]]:
        return self._rows

    def selected_case(self) -> str:
        return str(self._case_combo.currentData() or "DEFAULT")

    def selected_result(self) -> tuple[AnalysisResult | None, str]:
        return resolve_view(self._model, self._multi_result, self.selected_case())

    def copy_tsv(self) -> str:
        return copy_view_to_clipboard(self._table, include_headers=True)

    def export_csv(self, path: str | Path) -> None:
        write_csv(path, self.headers, self._rows)

    def _export_csv_dialog(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            self.export_csv(path)
        except OSError as exc:
            QMessageBox.warning(self, "Export CSV", str(exc))

    def _refresh_rows(self) -> None:
        raise NotImplementedError


class MemberStationForceTableDialog(_ResultExportDialogBase):
    headers = MEMBER_STATION_HEADERS

    def __init__(
        self,
        parent: QWidget | None,
        *,
        model: StructuralModel,
        element_ids: Sequence[int],
        multi_result: MultiCaseAnalysisResult | None,
        active_case: str = "DEFAULT",
        n_stations: int = 21,
    ) -> None:
        self._element_ids = list(element_ids)
        self._n_stations_default = int(n_stations)
        super().__init__(
            parent, model=model, multi_result=multi_result, active_case=active_case,
        )
        self.setWindowTitle("Station Force Table")
        station_row = QHBoxLayout()
        station_row.addWidget(QLabel("Stations per element:"))
        self._station_spin = QSpinBox(self)
        self._station_spin.setRange(2, 1001)
        self._station_spin.setValue(max(2, self._n_stations_default))
        self._station_spin.valueChanged.connect(lambda _v: self._refresh_rows())
        station_row.addWidget(self._station_spin)
        station_row.addStretch()
        self._outer.insertLayout(1, station_row)
        self._refresh_rows()

    def _metadata_text(self) -> str:
        by_id = {e.id: e for e in self._model.elements}
        metas = []
        for eid in self._element_ids:
            elem = by_id.get(eid)
            if isinstance(elem, FrameElement2D):
                metas.append(member_station_metadata(self._model, elem))
        if not metas:
            return "No frame elements selected."
        if len(metas) == 1:
            m = metas[0]
            return (
                f"Element {m.element_id}: L_total={m.L_total:.6g} m, "
                f"offset_i={m.offset_i:.6g} m, offset_j={m.offset_j:.6g} m, "
                f"L_flex={m.L_flex:.6g} m, station range=[{m.x_start:.6g}, {m.x_end:.6g}] m."
            )
        return f"{len(metas)} frame elements selected; each row includes element_id and station coordinates."

    def _refresh_rows(self) -> None:
        result, status_msg = self.selected_result()
        self._metadata_label.setText(self._metadata_text())
        if result is None or result.status != "ok":
            self._rows = []
            _populate_table(self._table, self.headers, self._rows)
            self._note.setText(status_msg or "Run static analysis first.")
            return
        try:
            self._rows = member_station_rows(
                self._model,
                result,
                self._element_ids,
                case_or_combination=self.selected_case(),
                n_stations=int(self._station_spin.value()),
            )
        except ValueError as exc:
            self._rows = []
            _populate_table(self._table, self.headers, self._rows)
            self._note.setText(str(exc))
            return
        _populate_table(self._table, self.headers, self._rows)
        has_offsets = any(
            isinstance(e, FrameElement2D)
            and (float(getattr(e, "offset_i", 0.0) or 0.0) != 0.0
                 or float(getattr(e, "offset_j", 0.0) or 0.0) != 0.0)
            for e in self._model.elements
            if e.id in self._element_ids
        )
        offset_note = (
            " Rigid offsets active: internal forces shown on the flexible span."
            if has_offsets else ""
        )
        self._note.setText(
            "CSV/TSV is Excel-friendly and can be used for manual SAP2000 comparison. "
            "Sign convention: local x is element i → j; N, V, and M follow this "
            "program's member-force convention. SAP2000 V2/M3 signs may differ."
            + offset_note
        )


class NodeResultTableDialog(_ResultExportDialogBase):
    headers = NODE_RESULT_HEADERS

    def __init__(
        self,
        parent: QWidget | None,
        *,
        model: StructuralModel,
        multi_result: MultiCaseAnalysisResult | None,
        active_case: str = "DEFAULT",
    ) -> None:
        super().__init__(
            parent, model=model, multi_result=multi_result, active_case=active_case,
        )
        self.setWindowTitle("Node Result Table")
        self._refresh_rows()

    def _refresh_rows(self) -> None:
        result, status_msg = self.selected_result()
        self._metadata_label.setText(
            f"{len(self._model.nodes)} nodes. Displacements are global UX/UY/RZ; "
            "reactions are shown at restrained DOFs when available."
        )
        if result is None or result.status != "ok":
            self._rows = []
            _populate_table(self._table, self.headers, self._rows)
            self._note.setText(status_msg or "Run static analysis first.")
            return
        self._rows = node_result_rows(
            self._model, result, case_or_combination=self.selected_case(),
        )
        _populate_table(self._table, self.headers, self._rows)
        self._note.setText(
            "CSV/TSV is Excel-friendly and can be used for manual SAP2000 comparison. "
            "Displacements are global UX/UY/RZ. Reactions are global components "
            "reported where available."
        )
