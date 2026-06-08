"""Unit tests for the reusable spreadsheet-style table-copy helper.

Headless Qt (offscreen). Each test builds a tiny QTableWidget /
QTreeWidget, drives a selection, and asserts the tab-delimited clipboard
text the helper produces.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
)

from structural_analysis.gui_qt.table_copy import (
    copy_view_to_clipboard,
    install_table_copy,
    tsv_for_view,
)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _table(rows: list[list[str]]) -> QTableWidget:
    t = QTableWidget(len(rows), len(rows[0]) if rows else 0)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            t.setItem(r, c, QTableWidgetItem(val))
    return t


def test_single_cell_copy(qt_app):
    from PyQt6.QtWidgets import QTableWidgetSelectionRange
    t = _table([["a", "b"], ["c", "d"]])
    t.setRangeSelected(QTableWidgetSelectionRange(0, 0, 0, 0), True)
    assert tsv_for_view(t) == "a"


def test_multi_cell_tab_delimited(qt_app):
    from PyQt6.QtWidgets import QTableWidgetSelectionRange
    t = _table([["a", "b"], ["c", "d"]])
    t.setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 1), True)
    assert tsv_for_view(t) == "a\tb\nc\td"


def test_multi_row_single_col_preserves_rows(qt_app):
    from PyQt6.QtWidgets import QTableWidgetSelectionRange
    t = _table([["a"], ["b"], ["c"]])
    t.setRangeSelected(QTableWidgetSelectionRange(0, 0, 2, 0), True)
    assert tsv_for_view(t) == "a\nb\nc"


def test_no_selection_copies_whole_table(qt_app):
    t = _table([["a", "b"], ["c", "d"]])
    # NoSelection table (like the element-loads table) — Ctrl+C should
    # still copy the full contents.
    t.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    assert tsv_for_view(t) == "a\tb\nc\td"


def test_cellwidget_column_copies_empty(qt_app):
    from PyQt6.QtWidgets import QTableWidgetSelectionRange
    t = _table([["a", ""], ["c", ""]])
    # Embed buttons in column 1 (mirrors the loads-table Edit/Delete).
    t.setCellWidget(0, 1, QPushButton("Delete"))
    t.setCellWidget(1, 1, QPushButton("Delete"))
    t.setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 1), True)
    # Button text must NOT appear; column 1 copies as empty.
    assert tsv_for_view(t) == "a\t\nc\t"


def test_include_headers_prepends_header_row(qt_app):
    from PyQt6.QtWidgets import QTableWidgetSelectionRange
    t = _table([["a", "b"]])
    t.setHorizontalHeaderLabels(["Case", "Type"])
    t.setRangeSelected(QTableWidgetSelectionRange(0, 0, 0, 1), True)
    assert tsv_for_view(t, include_headers=True) == "Case\tType\na\tb"


def test_include_headers_aligns_with_selected_column_subset(qt_app):
    """Selecting only some columns must slice the header row to match,
    so the TSV stays a proper rectangle when pasted into Excel."""
    from PyQt6.QtWidgets import QTableWidgetSelectionRange
    t = _table([["a", "b", "c"], ["d", "e", "f"]])
    t.setHorizontalHeaderLabels(["C0", "C1", "C2"])
    # Select only columns 1..2 across both rows.
    t.setRangeSelected(QTableWidgetSelectionRange(0, 1, 1, 2), True)
    tsv = tsv_for_view(t, include_headers=True)
    assert tsv == "C1\tC2\nb\tc\ne\tf"


def test_install_is_idempotent(qt_app):
    t = _table([["a"]])
    install_table_copy(t)
    install_table_copy(t)  # must not raise / double-install
    assert getattr(t, "_table_copy_installed", False) is True


def test_copy_to_clipboard_sets_clipboard_text(qt_app):
    from PyQt6.QtWidgets import QTableWidgetSelectionRange
    t = _table([["x", "y"]])
    t.setRangeSelected(QTableWidgetSelectionRange(0, 0, 0, 1), True)
    out = copy_view_to_clipboard(t)
    assert out == "x\ty"
    assert QGuiApplication.clipboard().text() == "x\ty"


def test_works_on_qtreewidget(qt_app):
    tree = QTreeWidget()
    tree.setColumnCount(2)
    QTreeWidgetItem(tree, ["a", "b"])
    QTreeWidgetItem(tree, ["c", "d"])
    # No selection → whole tree.
    assert tsv_for_view(tree) == "a\tb\nc\td"


def test_context_menu_policy_set_on_install(qt_app):
    t = _table([["a"]])
    install_table_copy(t)
    assert (
        t.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    )
