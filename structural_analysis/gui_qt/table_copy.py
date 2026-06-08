"""Spreadsheet-style copy for any Qt item view (tables, trees, lists).

A single reusable helper, :func:`install_table_copy`, wires **Ctrl+C**
and a right-click **Copy** action onto any ``QAbstractItemView`` so the
current selection (or, when nothing is selected, the whole view) is
placed on the clipboard as **tab-delimited text** — ready to paste
directly into Excel / LibreOffice Calc / Google Sheets as separate
cells.

Design goals:

* **Reusable** — one call per view; works on ``QTableWidget``,
  ``QTableView``, ``QTreeWidget`` and ``QTreeView`` (anything exposing a
  ``QAbstractItemModel`` and a ``selectionModel``).
* **Non-intrusive** — read-only: it never mutates the model. Editable
  cells keep working; while a cell editor is open the inner ``QLineEdit``
  grabs Ctrl+C first (desired — copies the text being edited).
* **Excel-faithful** — rows joined by ``\n``, cells by ``\t``; a
  multi-cell rectangular selection preserves row/column structure.
* **Embedded-widget aware** — table cells that hold a widget (e.g. the
  Edit / Delete buttons in the element-loads table) copy as an empty
  string instead of leaking the button text into the spreadsheet.
* **Idempotent** — calling twice on the same view installs nothing the
  second time.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import QAbstractItemView, QMenu


_INSTALLED_FLAG = "_table_copy_installed"


def install_table_copy(
    view: QAbstractItemView, *, include_headers: bool = False,
) -> None:
    """Install Ctrl+C and a right-click 'Copy' action on ``view``.

    ``include_headers`` prepends the horizontal header labels as the
    first TSV row (useful for result tables the user pastes as a
    standalone block). Idempotent.
    """
    if getattr(view, _INSTALLED_FLAG, False):
        return
    setattr(view, _INSTALLED_FLAG, True)

    # Ctrl+C / Cmd+C — scoped to the view (and its children, so an open
    # cell editor still gets first dibs while editing).
    shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Copy), view)
    shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
    shortcut.activated.connect(
        lambda: copy_view_to_clipboard(view, include_headers=include_headers)
    )

    # Right-click → Copy. None of the app's views use a custom context
    # menu today, so installing one here is safe (verified by audit).
    view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    view.customContextMenuRequested.connect(
        lambda pos, v=view, h=include_headers: _show_copy_menu(v, pos, h)
    )


def copy_view_to_clipboard(
    view: QAbstractItemView, *, include_headers: bool = False,
) -> str:
    """Copy the current selection (or the whole view if nothing is
    selected) to the clipboard as tab-delimited text. Returns the text
    that was placed on the clipboard (also handy for tests)."""
    text = tsv_for_view(view, include_headers=include_headers)
    QGuiApplication.clipboard().setText(text)
    return text


# ── TSV assembly ─────────────────────────────────────────────────────


def _index_text(view: QAbstractItemView, index) -> str:
    """Display text for one model index. Cells holding an embedded
    widget (buttons) copy as an empty string so 'Edit' / 'Delete' never
    leak into the spreadsheet. ``indexWidget`` is the generic
    ``QAbstractItemView`` API and works for tables, trees, and lists."""
    if view.indexWidget(index) is not None:
        return ""
    data = index.data(Qt.ItemDataRole.DisplayRole)
    return "" if data is None else str(data)


def _header_text(view: QAbstractItemView) -> list[str] | None:
    model = view.model()
    if model is None:
        return None
    out: list[str] = []
    for c in range(model.columnCount()):
        label = model.headerData(
            c, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole,
        )
        out.append("" if label is None else str(label))
    return out


def tsv_for_view(
    view: QAbstractItemView, *, include_headers: bool = False,
) -> str:
    """Render ``view``'s selection (or its full contents) as TSV."""
    model = view.model()
    if model is None:
        return ""
    sel_model = view.selectionModel()
    selected = sel_model.selectedIndexes() if sel_model is not None else []

    lines: list[str] = []
    col_range: tuple[int, int] | None = None
    if selected:
        # Group by row, remember the spanned column range so a
        # rectangular selection keeps its shape; gaps inside the range
        # become empty cells (matches Excel's contiguous block paste).
        by_row: dict[int, dict[int, str]] = {}
        for idx in selected:
            by_row.setdefault(idx.row(), {})[idx.column()] = _index_text(
                view, idx,
            )
        all_cols = [c for cols in by_row.values() for c in cols]
        c_min, c_max = min(all_cols), max(all_cols)
        col_range = (c_min, c_max)
        for r in sorted(by_row):
            cols = by_row[r]
            lines.append(
                "\t".join(cols.get(c, "") for c in range(c_min, c_max + 1))
            )
    else:
        # Nothing selected → whole view (so read-only / no-selection
        # tables are still copyable with a single Ctrl+C).
        for r in range(model.rowCount()):
            lines.append(
                "\t".join(
                    _index_text(view, model.index(r, c))
                    for c in range(model.columnCount())
                )
            )

    if include_headers:
        headers = _header_text(view)
        if headers is not None:
            # When the user copied only a column subset, slice the
            # header row to the same range so the TSV stays a true
            # rectangle (Excel paste keeps header ↔ data alignment).
            if col_range is not None:
                c_min, c_max = col_range
                headers = headers[c_min:c_max + 1]
            lines.insert(0, "\t".join(headers))
    return "\n".join(lines)


def _show_copy_menu(
    view: QAbstractItemView, pos, include_headers: bool,
) -> None:
    menu = QMenu(view)
    act_copy = menu.addAction("Copy")
    chosen = menu.exec(view.viewport().mapToGlobal(pos))
    if chosen is act_copy:
        copy_view_to_clipboard(view, include_headers=include_headers)
