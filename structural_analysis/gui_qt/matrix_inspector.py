"""Matrix / DOF Inspector — read-only stiffness and DOF numbering viewer.

Non-modal window with four tabs:
  1. DOF Map — equation numbering, free/restrained status
  2. Element Matrix — k_local, T, k_global for a selected element
  3. Global K — full assembled stiffness matrix
  4. Kff — free-DOF partition (boundary-reduced)

All data is extracted via existing public APIs; no solver math is duplicated.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

_DISPLAY_GUARD = 400   # max n_total to render K / Kff grid cells
_RANK_GUARD = 100      # max n_free for SVD rank / condition estimate


# ── formatting helpers ────────────────────────────────────────────────────────


def _fmt(v: float) -> str:
    """'0' for near-zero values, scientific notation otherwise."""
    if abs(v) < 1e-30:
        return "0"
    return f"{v:.3e}"


def _build_dof_labels(dofs) -> dict[int, str]:
    """Map global DOF index → short label 'n{nid}.{dof}'."""
    out: dict[int, str] = {}
    for nid, m in dofs.active_map.items():
        for dof_name, idx in m.items():
            if idx is not None:
                out[idx] = f"n{nid}.{dof_name}"
    return out


def _elem_dof_labels(elem, dofs) -> list[str]:
    """6-entry label list matching element local DOF order.

    Uses the element's DOF address vector so that inactive DOFs (e.g. truss rz)
    are clearly marked '–' rather than silently labelled.
    """
    mapping = dofs.element_dof_map(elem)          # 6-entry, None for inactive
    node_ids = [
        elem.node_i, elem.node_i, elem.node_i,
        elem.node_j, elem.node_j, elem.node_j,
    ]
    dof_names = ["ux", "uy", "rz", "ux", "uy", "rz"]
    return [
        f"n{nid}.{dof}" if global_idx is not None else "–"
        for nid, dof, global_idx in zip(node_ids, dof_names, mapping)
    ]


# ── widget factories ──────────────────────────────────────────────────────────


def _make_matrix_table(
    mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    parent: QWidget,
) -> QTableWidget:
    """Read-only, copyable QTableWidget for a 2-D numeric matrix."""
    from .table_copy import install_table_copy

    nrows, ncols = mat.shape
    table = QTableWidget(nrows, ncols, parent)
    table.setHorizontalHeaderLabels(col_labels)
    table.setVerticalHeaderLabels(row_labels)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    for i in range(nrows):
        for j in range(ncols):
            item = QTableWidgetItem(_fmt(float(mat[i, j])))
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            table.setItem(i, j, item)
    if ncols > 20:
        table.horizontalHeader().setDefaultSectionSize(85)
    else:
        table.resizeColumnsToContents()
    install_table_copy(table, include_headers=True)
    return table


def _make_placeholder(msg: str, parent: QWidget) -> QWidget:
    """Simple label widget used when a tab cannot display data."""
    w = QWidget(parent)
    lbl = QLabel(msg, w)
    lbl.setWordWrap(True)
    lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    lay = QVBoxLayout(w)
    lay.addWidget(lbl)
    lay.addStretch()
    return w


def _html(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setTextFormat(Qt.TextFormat.RichText)
    return lbl


# ── main window ───────────────────────────────────────────────────────────────


class MatrixDofInspectorWindow(QWidget):
    """Non-modal stiffness / DOF inspector.

    Instantiate once and call :meth:`refresh` to update after model edits.
    Use :meth:`set_selected_element` to pre-select an element in tab 2.
    """

    def __init__(self, parent: QWidget, model_fn) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Matrix / DOF Inspector")
        self.resize(820, 640)
        self._model_fn = model_fn
        self._sel_elem_id: int | None = None

        self._tabs = QTabWidget(self)

        close_btn = QPushButton("Close", self)
        close_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        close_btn.clicked.connect(self.close)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(self._tabs)
        lay.addLayout(btn_row)

        self.refresh()

    # ── public API ────────────────────────────────────────────────────────

    def set_selected_element(self, elem_id: int | None) -> None:
        """Tell the inspector which canvas element is currently selected."""
        self._sel_elem_id = elem_id

    def refresh(self) -> None:
        """Rebuild all four tabs from the current model."""
        model = self._model_fn()

        try:
            from ..assembler import assemble_global_system
            K, F, dofs, _warnings, elem_data = assemble_global_system(model)
            free = dofs.free_indices
            Kff = K[np.ix_(free, free)] if free else np.zeros((0, 0))
            dof_labels = _build_dof_labels(dofs)
            error: str | None = None
        except Exception as exc:
            dofs = K = F = Kff = dof_labels = elem_data = None
            error = str(exc)

        for i in range(self._tabs.count()):
            widget = self._tabs.widget(i)
            if widget is not None:
                widget.deleteLater()
        self._tabs.clear()
        self._tabs.addTab(self._build_dof_tab(dofs, error), "DOF Map")
        self._tabs.addTab(
            self._build_elem_tab(model, dofs, dof_labels, error), "Element Matrix"
        )
        self._tabs.addTab(
            self._build_global_k_tab(K, dofs, dof_labels, error), "Global K"
        )
        self._tabs.addTab(
            self._build_kff_tab(Kff, dofs, dof_labels, error), "Kff"
        )

    # ── Tab 1: DOF Map ────────────────────────────────────────────────────

    def _build_dof_tab(self, dofs, error: str | None) -> QWidget:
        if error is not None:
            return _make_placeholder(f"Cannot assemble model:\n{error}", self)
        if dofs is None:
            return _make_placeholder("No model loaded.", self)

        from .table_copy import install_table_copy

        free_set = set(dofs.free_indices)
        free_pos = {idx: pos for pos, idx in enumerate(dofs.free_indices)}

        rows: list[tuple] = []
        for nid, m in sorted(dofs.active_map.items()):
            for dof_name in ("ux", "uy", "rz"):
                idx = m.get(dof_name)
                if idx is None:
                    continue
                status = "free" if idx in free_set else "restrained"
                free_eq: str = str(free_pos[idx]) if idx in free_set else "—"
                rows.append((idx, str(nid), dof_name, status, free_eq))
        rows.sort(key=lambda r: r[0])

        table = QTableWidget(len(rows), 5, self)
        table.setHorizontalHeaderLabels(
            ["Eq# (global)", "Node", "DOF", "Status", "Free Eq#"]
        )
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row_idx, (eq, node, dof, status, feq) in enumerate(rows):
            for col, val in enumerate((str(eq), node, dof, status, feq)):
                item = QTableWidgetItem(val)
                if col in (0, 4):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row_idx, col, item)
        table.resizeColumnsToContents()
        install_table_copy(table, include_headers=True)

        container = QWidget(self)
        lay = QVBoxLayout(container)
        lay.addWidget(QLabel(
            f"n_total = {dofs.n_total}   "
            f"free = {len(dofs.free_indices)}   "
            f"restrained = {len(dofs.restrained_indices)}"
        ))
        lay.addWidget(table)
        return container

    # ── Tab 2: Element Matrix ─────────────────────────────────────────────

    def _build_elem_tab(self, model, dofs, dof_labels, error: str | None) -> QWidget:
        if error is not None:
            return _make_placeholder(f"Cannot assemble model:\n{error}", self)
        if not model.elements:
            return _make_placeholder("No elements in model.", self)
        if dofs is None:
            return _make_placeholder("DOF manager unavailable.", self)

        container = QWidget(self)
        lay = QVBoxLayout(container)

        combo = QComboBox(container)
        for elem in model.elements:
            kind = getattr(elem, "kind", elem.__class__.__name__)
            combo.addItem(f"Element {elem.id}  ({kind})", userData=elem.id)

        default_idx = 0
        if self._sel_elem_id is not None:
            for i in range(combo.count()):
                if combo.itemData(i) == self._sel_elem_id:
                    default_idx = i
                    break
        combo.setCurrentIndex(default_idx)

        info_lbl = QLabel("", container)
        info_lbl.setWordWrap(True)

        scroll = QScrollArea(container)
        scroll.setWidgetResizable(True)
        mat_widget = QWidget()
        mat_lay = QVBoxLayout(mat_widget)
        scroll.setWidget(mat_widget)

        def _render(combo_idx: int) -> None:
            while mat_lay.count():
                item = mat_lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            eid = combo.itemData(combo_idx)
            elem = next((e for e in model.elements if e.id == eid), None)
            if elem is None:
                mat_lay.addWidget(QLabel("Element not found."))
                return

            try:
                L, c, s = elem.length_cos_sin(model.nodes)
            except Exception as exc:
                mat_lay.addWidget(QLabel(f"Cannot compute geometry: {exc}"))
                return

            kind = getattr(elem, "kind", elem.__class__.__name__)
            theta_deg = float(np.degrees(np.arctan2(s, c)))
            info_lbl.setText(
                f"Element {elem.id} · {kind} · nodes {elem.node_i}–{elem.node_j}"
                f" · L = {L:.4g} m · c = {c:.4f} · s = {s:.4f}"
                f" · θ = {theta_deg:.2f}°"
            )
            self._sel_elem_id = eid

            lbls = _elem_dof_labels(elem, dofs)

            # k_local raw
            k_raw = elem.raw_local_stiffness(model.nodes)
            mat_lay.addWidget(_html("<b>k_local (raw)</b>"))
            mat_lay.addWidget(_make_matrix_table(k_raw, lbls, lbls, mat_widget))

            # k_local condensed — only if releases change it
            try:
                k_cond, _ = elem.assembled_local_stiffness_and_load(model.nodes)
            except Exception:
                k_cond = k_raw
            if float(np.max(np.abs(k_cond - k_raw))) > 1e-30:
                mat_lay.addWidget(_html("<b>k_local (condensed — after moment releases)</b>"))
                mat_lay.addWidget(_make_matrix_table(k_cond, lbls, lbls, mat_widget))
            else:
                mat_lay.addWidget(_html(
                    "<i>k_local (condensed): no releases — identical to raw</i>"
                ))

            # Transformation matrix T  (global→local)
            T = elem.transformation_matrix(model.nodes)
            mat_lay.addWidget(_html("<b>T  (global→local)</b>"))
            mat_lay.addWidget(_make_matrix_table(T, lbls, lbls, mat_widget))

            # k_global = Tᵀ k_local T
            k_global, _ = elem.global_stiffness_and_load(model.nodes)
            mat_lay.addWidget(_html("<b>k_global = T<sup>T</sup> k_local T</b>"))
            mat_lay.addWidget(_make_matrix_table(k_global, lbls, lbls, mat_widget))

            mat_lay.addStretch()

        combo.currentIndexChanged.connect(_render)
        _render(default_idx)

        top = QHBoxLayout()
        top.addWidget(QLabel("Element:"))
        top.addWidget(combo)
        top.addStretch()
        lay.addLayout(top)
        lay.addWidget(info_lbl)
        lay.addWidget(scroll)
        return container

    # ── Tab 3: Global K ───────────────────────────────────────────────────

    def _build_global_k_tab(self, K, dofs, dof_labels, error: str | None) -> QWidget:
        if error is not None:
            return _make_placeholder(f"Cannot assemble model:\n{error}", self)
        if K is None:
            return _make_placeholder("No model loaded.", self)

        n = dofs.n_total
        sym = float(np.max(np.abs(K - K.T)))

        container = QWidget(self)
        lay = QVBoxLayout(container)
        lay.addWidget(QLabel(f"Size: {n} × {n}"))
        lay.addWidget(QLabel(f"Symmetry: max|K − Kᵀ| = {sym:.2e}"))

        nf = len(dofs.free_indices)
        if nf <= _RANK_GUARD and nf > 0:
            Kff_loc = K[np.ix_(dofs.free_indices, dofs.free_indices)]
            try:
                sv = np.linalg.svd(Kff_loc, compute_uv=False)
                tol = max(Kff_loc.shape) * sv[0] * 1e-12
                rank = int(np.sum(sv > tol))
                cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
                lay.addWidget(QLabel(f"Kff rank: {rank}/{nf}"))
                lay.addWidget(QLabel(f"Kff condition estimate: {cond:.2e}"))
                if rank < nf:
                    lay.addWidget(QLabel(
                        "⚠ WARNING: Singular or near-singular "
                        "(mechanism in free DOFs)"
                    ))
            except Exception as exc:
                lay.addWidget(QLabel(f"Rank/condition: error ({exc})"))
        elif nf == 0:
            lay.addWidget(QLabel("All DOFs restrained — Kff is empty."))
        else:
            lay.addWidget(QLabel(
                f"Rank/condition: skipped (n_free = {nf} > {_RANK_GUARD})"
            ))

        if n <= _DISPLAY_GUARD:
            lbls = [dof_labels.get(i, str(i)) for i in range(n)]
            lay.addWidget(_make_matrix_table(K, lbls, lbls, container))
        else:
            lay.addWidget(QLabel(
                f"Matrix too large to display ({n} × {n} > {_DISPLAY_GUARD}).\n"
                "Use a smaller model to view the full grid."
            ))
        lay.addStretch()
        return container

    # ── Tab 4: Kff ────────────────────────────────────────────────────────

    def _build_kff_tab(self, Kff, dofs, dof_labels, error: str | None) -> QWidget:
        if error is not None:
            return _make_placeholder(f"Cannot assemble model:\n{error}", self)
        if Kff is None:
            return _make_placeholder("No model loaded.", self)

        nf = len(dofs.free_indices)
        if nf == 0:
            return _make_placeholder(
                "All DOFs are restrained — no free DOFs (Kff is empty).", self
            )

        sym = float(np.max(np.abs(Kff - Kff.T)))
        restrained_lbls = [
            dof_labels.get(i, str(i)) for i in dofs.restrained_indices
        ]

        container = QWidget(self)
        lay = QVBoxLayout(container)
        lay.addWidget(QLabel(f"Kff size: {nf} × {nf}  (free DOFs only)"))
        lay.addWidget(QLabel(f"Symmetry: max|Kff − Kffᵀ| = {sym:.2e}"))
        lay.addWidget(QLabel(
            f"Free DOFs: {nf}   "
            f"Restrained DOFs: {len(dofs.restrained_indices)}"
        ))
        if restrained_lbls:
            lay.addWidget(QLabel("Restrained: " + ", ".join(restrained_lbls)))

        if nf <= _RANK_GUARD:
            try:
                sv = np.linalg.svd(Kff, compute_uv=False)
                tol = max(Kff.shape) * sv[0] * 1e-12
                rank = int(np.sum(sv > tol))
                cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
                lay.addWidget(QLabel(f"Rank: {rank}/{nf}"))
                lay.addWidget(QLabel(f"Condition estimate: {cond:.2e}"))
                if rank < nf:
                    lay.addWidget(QLabel(
                        "⚠ WARNING: Singular or near-singular "
                        "(mechanism in free DOFs)"
                    ))
            except Exception as exc:
                lay.addWidget(QLabel(f"Rank/condition: error ({exc})"))
        else:
            lay.addWidget(QLabel(
                f"Rank/condition: skipped (n_free = {nf} > {_RANK_GUARD})"
            ))

        if nf <= _DISPLAY_GUARD:
            free_lbls = [dof_labels.get(i, str(i)) for i in dofs.free_indices]
            lay.addWidget(_make_matrix_table(Kff, free_lbls, free_lbls, container))
        else:
            lay.addWidget(QLabel(
                f"Kff too large to display ({nf} × {nf} > {_DISPLAY_GUARD}).\n"
                "Use a smaller model to view the full grid."
            ))
        lay.addStretch()
        return container
