"""Regression lock — lumped-only modal workflow (final-submission build).

These tests pin the user-facing contract that the lumped / row-sum mass
formulation is the ONLY one exposed by:

* the public ``solve_modal()`` entry point (default + only accepted
  result; passing ``"consistent"`` emits a :class:`DeprecationWarning`
  and is transparently mapped to lumped, no silent crash);
* the Modal Analysis dialog (combo box has a single, disabled item);
* the Modal viewer header (always reads "lumped (row-sum)" for fresh
  results, with a "(legacy)" suffix for older saved ``ModalResult``
  objects that were tagged differently);
* the Joint Masses inspection window (formulation row removed; the
  Row-sum / Diagonal table-view toggle is unchanged because that is a
  diagnostic display mode, not a mass formulation).

The internal element-level helper ``consistent_mass_local`` and the
low-level ``assemble_mass_matrix(formulation="consistent")`` are kept
because they are still used by:

* the diagnostic ``mass_inspect.joint_mass_table`` when a developer
  explicitly asks for the consistent rotational diagonals;
* element-level unit tests pinning the consistent identities (e.g.
  ``L² · ρAL / 420`` row-sum value).

This separation is what lets the GUI be simple while keeping the
low-level mass library available for diagnostics — the user explicitly
asked us not to delete consistent-mass code blindly.
"""

from __future__ import annotations

import os
import warnings

import pytest

from structural_analysis.element import FrameElement2D
from structural_analysis.model import (
    Material, Node, Section, StructuralModel, Support,
)
from structural_analysis.modal import solve_modal


# ── 1. Engine: solve_modal default + deprecation ────────────────────────


def _cantilever():
    m = StructuralModel(title="lumped-only check")
    m.materials[1] = Material(id=1, name="C", E=2.0e8, density=7850.0)
    m.sections[1] = Section(id=1, name="S", material_id=1,
                            A=0.02, I=8.0e-4, depth=0.3)
    m.nodes = {1: Node(1, 0.0, 0.0), 2: Node(2, 5.0, 0.0)}
    m.elements = [FrameElement2D(
        id=1, node_i=1, node_j=2, E=2.0e8,
        A=0.02, I=8.0e-4, section_id=1, rho=7850.0,
    )]
    m.supports = {1: Support(node_id=1, ux=True, uy=True, rz=True)}
    return m


def test_solve_modal_default_is_lumped():
    r = solve_modal(_cantilever(), n_modes=3)
    assert r.mass_formulation == "lumped"


def test_solve_modal_consistent_warns_and_maps_to_lumped():
    """Saved-file / older-script compatibility: passing
    ``"consistent"`` must not crash, must emit a clear DeprecationWarning
    naming the policy change, and must produce a working lumped
    result (the cleanest of the two graceful-fallback options the
    spec offered)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = solve_modal(_cantilever(), n_modes=3, mass_formulation="consistent")
    matching = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert matching, "expected a DeprecationWarning on consistent fallback"
    msg = str(matching[0].message)
    assert "Consistent mass" in msg
    assert "lumped" in msg.lower()
    # The result is a working lumped result — same as default.
    r_default = solve_modal(_cantilever(), n_modes=3)
    assert r.mass_formulation == r_default.mass_formulation == "lumped"
    assert list(r.frequencies) == list(r_default.frequencies)


def test_solve_modal_unknown_formulation_raises():
    with pytest.raises(ValueError, match="lumped"):
        solve_modal(_cantilever(), n_modes=3, mass_formulation="banana")  # type: ignore[arg-type]


# ── 2. GUI: Modal dialog combo only offers lumped ───────────────────────

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication  # noqa: E402
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"PyQt6 unavailable: {exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_modal_dialog_combo_only_offers_lumped(qt_app):
    from structural_analysis.gui_qt.dialogs import ModalAnalysisDialog
    d = ModalAnalysisDialog(parent=None)
    items = [d._mass_combo.itemData(i) for i in range(d._mass_combo.count())]
    assert items == ["lumped"]
    assert not d._mass_combo.isEnabled()      # single option → read-only
    result = d._accept()
    assert result["mass_formulation"] == "lumped"


def test_modal_dialog_does_not_expose_consistent_label(qt_app):
    """The Modal Analysis dialog must not show 'Consistent' anywhere in
    the combo item texts (label text is what the user sees)."""
    from structural_analysis.gui_qt.dialogs import ModalAnalysisDialog
    d = ModalAnalysisDialog(parent=None)
    labels = [d._mass_combo.itemText(i) for i in range(d._mass_combo.count())]
    assert not any("Consistent" in s for s in labels)


# ── 3. GUI: Joint Masses window no longer has a formulation selector ────


def test_joint_masses_window_has_no_formulation_radios(qt_app):
    """The Joint Masses inspection window had a 'Consistent / Lumped'
    radio pair before the lumped-only cleanup. Those radios were
    removed; only the Row-sum / Diagonal *table view* radios remain
    (those are diagnostic display modes, not mass formulations)."""
    from structural_analysis.gui_qt.joint_masses import JointMassesWindow
    w = JointMassesWindow(
        parent=None,
        model_provider=lambda: _cantilever(),
    )
    # No formulation radios should exist on the window.
    assert not hasattr(w, "_rb_consistent")
    assert not hasattr(w, "_rb_lumped")
    # The default formulation it asks joint_mass_table for is lumped.
    assert w._mass_formulation == "lumped"
    # Diagnostic table-view radios are still there.
    assert hasattr(w, "_rb_rowsum")
    assert hasattr(w, "_rb_diag")
