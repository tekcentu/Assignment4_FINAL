"""Tests for PR #43 — Undo/Redo Clarity (v0.28.0)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QKeySequence
    _has_qt = True
except Exception:
    _has_qt = False


@pytest.fixture(scope="module")
def qt_app():
    if not _has_qt:
        pytest.skip("PyQt6 unavailable")
    app = QApplication.instance() or QApplication([])
    yield app


def _window(qt_app):
    from structural_analysis.gui_qt.app import MainWindow
    return MainWindow()


def _push_add_node(w, node_id=999, x=10.0, y=0.0):
    from structural_analysis.gui_common.commands import AddNodeCmd
    cmd = AddNodeCmd(node_id=node_id, x=x, y=y)
    w.execute(cmd)
    return cmd


# ── empty-stack defaults ─────────────────────────────────────────────────────


def test_empty_stack_plain_disabled_labels(qt_app):
    w = _window(qt_app)
    assert w.act_undo.text() == "&Undo"
    assert w.act_redo.text() == "&Redo"
    assert not w.act_undo.isEnabled()
    assert not w.act_redo.isEnabled()


def test_empty_stack_tooltips_plain(qt_app):
    w = _window(qt_app)
    assert w.act_undo.toolTip() == "Undo"
    assert w.act_redo.toolTip() == "Undo" or w.act_redo.toolTip() == "Redo"
    # specifically, redo tooltip is "Redo"
    assert w.act_redo.toolTip() == "Redo"


# ── after execute ────────────────────────────────────────────────────────────


def test_after_execute_undo_label_updates(qt_app):
    w = _window(qt_app)
    _push_add_node(w)
    assert w.act_undo.text() == "&Undo Add Node"
    assert w.act_undo.isEnabled()
    # redo cleared on new execute
    assert w.act_redo.text() == "&Redo"
    assert not w.act_redo.isEnabled()


def test_tooltip_matches_label(qt_app):
    w = _window(qt_app)
    _push_add_node(w)
    assert w.act_undo.toolTip() == "Undo Add Node"


# ── after undo ───────────────────────────────────────────────────────────────


def test_after_undo_redo_label_updates(qt_app):
    w = _window(qt_app)
    _push_add_node(w)
    w._do_undo()
    assert w.act_undo.text() == "&Undo"
    assert not w.act_undo.isEnabled()
    assert w.act_redo.text() == "&Redo Add Node"
    assert w.act_redo.isEnabled()
    assert w.act_redo.toolTip() == "Redo Add Node"


def test_after_redo_labels_restore(qt_app):
    w = _window(qt_app)
    _push_add_node(w)
    w._do_undo()
    w._do_redo()
    assert w.act_undo.text() == "&Undo Add Node"
    assert w.act_undo.isEnabled()
    assert w.act_redo.text() == "&Redo"
    assert not w.act_redo.isEnabled()


# ── overrides ────────────────────────────────────────────────────────────────


def test_override_batch_assign(qt_app):
    w = _window(qt_app)
    from structural_analysis.gui_common.commands import BatchUpdateElementsCmd
    # Use the formatter directly — constructing a real BatchUpdateElementsCmd
    # needs valid target elements and a non-trivial diff. The label comes
    # from ``description``, not from instance state.
    inst = BatchUpdateElementsCmd.__new__(BatchUpdateElementsCmd)
    assert w._command_label(inst) == "Batch Edit Element Properties"


def test_override_edit_nodal_load_row(qt_app):
    w = _window(qt_app)
    # Directly test the formatter, since constructing a valid
    # EditNodalLoadRowCmd needs an existing row index.
    from structural_analysis.gui_common.commands import (
        EditNodalLoadRowCmd, DeleteNodalLoadRowCmd, ReplaceModelCmd,
    )
    e = EditNodalLoadRowCmd.__new__(EditNodalLoadRowCmd)
    assert w._command_label(e) == "Edit Nodal Load"
    d = DeleteNodalLoadRowCmd.__new__(DeleteNodalLoadRowCmd)
    assert w._command_label(d) == "Delete Nodal Load"
    r = ReplaceModelCmd.__new__(ReplaceModelCmd)
    assert w._command_label(r) == "Replace Model"


# ── status bar feedback ──────────────────────────────────────────────────────


def test_status_after_execute(qt_app):
    w = _window(qt_app)
    _push_add_node(w)
    assert "Add Node" in w._status_label.text()
    assert "Undo: Add Node" in w._status_label.text()


def test_status_after_undo(qt_app):
    w = _window(qt_app)
    _push_add_node(w)
    w._do_undo()
    text = w._status_label.text()
    assert text.startswith("Undid Add Node")
    assert "Redo: Add Node" in text


def test_status_after_redo(qt_app):
    w = _window(qt_app)
    _push_add_node(w)
    w._do_undo()
    w._do_redo()
    text = w._status_label.text()
    assert text.startswith("Redid Add Node")
    assert "Undo: Add Node" in text


# ── shortcuts unchanged ──────────────────────────────────────────────────────


def test_shortcut_still_undo_standard_key(qt_app):
    w = _window(qt_app)
    assert w.act_undo.shortcut() == QKeySequence(QKeySequence.StandardKey.Undo)
    assert w.act_redo.shortcut() == QKeySequence(QKeySequence.StandardKey.Redo)


def test_shortcut_trigger_still_undoes(qt_app):
    w = _window(qt_app)
    _push_add_node(w, node_id=998, x=20.0)
    assert 998 in w._model.nodes
    w.act_undo.trigger()
    assert 998 not in w._model.nodes
    assert w.act_redo.isEnabled()


# ── editing-lock reconciliation ──────────────────────────────────────────────


def test_lock_disables_undo_even_with_stack(qt_app):
    w = _window(qt_app)
    _push_add_node(w)
    assert w.act_undo.isEnabled()
    w._set_editing_locked(True)
    assert not w.act_undo.isEnabled()
    # label still describes what would be undone
    assert w.act_undo.text() == "&Undo Add Node"
    w._set_editing_locked(False)
    assert w.act_undo.isEnabled()


def test_lock_does_not_enable_empty_undo(qt_app):
    w = _window(qt_app)
    w._set_editing_locked(False)
    assert not w.act_undo.isEnabled()
    assert w.act_undo.text() == "&Undo"


# ── reset on new file ────────────────────────────────────────────────────────


def test_labels_reset_on_new_file(qt_app):
    w = _window(qt_app)
    _push_add_node(w)
    assert w.act_undo.isEnabled()
    # Simulate _do_new without the confirm-discard dialog
    w._undo.clear()
    w._redo.clear()
    w._refresh_undo_redo()
    assert w.act_undo.text() == "&Undo"
    assert w.act_redo.text() == "&Redo"
    assert not w.act_undo.isEnabled()
    assert not w.act_redo.isEnabled()


# ── audit: every Command subclass renders cleanly ────────────────────────────


def _all_command_classes():
    from structural_analysis.gui_common import commands as cm
    out = []
    for name in dir(cm):
        obj = getattr(cm, name)
        if (isinstance(obj, type)
                and issubclass(obj, cm.Command)
                and obj is not cm.Command):
            out.append(obj)
    return out


def test_all_command_descriptions_render_titlecase(qt_app):
    w = _window(qt_app)
    forbidden = {"cmd", "row"}
    for cls in _all_command_classes():
        inst = cls.__new__(cls)
        label = w._command_label(inst)
        assert label, f"{cls.__name__}: empty label"
        # No leading lowercase
        assert label[0].isupper(), f"{cls.__name__}: not Title Case: {label!r}"
        # No internal jargon
        tokens = {t.lower() for t in label.split()}
        bad = tokens & forbidden
        assert not bad, f"{cls.__name__}: bad token in label {label!r}: {bad}"


def test_specific_command_labels(qt_app):
    """Spot-check the commands the PR brief calls out by name."""
    w = _window(qt_app)
    from structural_analysis.gui_common import commands as cm
    cases = {
        cm.AddMemberLoadCmd:           "Add Member Load",
        cm.UpdateMemberLoadCmd:        "Edit Member Load",
        cm.DeleteMemberLoadCmd:        "Delete Member Load",
        cm.BatchAddMemberLoadsCmd:     "Batch Add Member Loads",
        cm.AddNodalLoadCmd:            "Add Nodal Load",
        cm.EditNodalLoadRowCmd:        "Edit Nodal Load",
        cm.DeleteNodalLoadRowCmd:      "Delete Nodal Load",
        cm.BatchAddNodalLoadsCmd:      "Batch Add Nodal Loads",
        cm.RenumberElementsCmd:        "Renumber Elements",
        cm.MergeAdjacentElementsCmd:   "Merge Adjacent Elements",
        cm.UpdateModalMassSourceCmd:   "Update Modal Mass Source",
        cm.BatchDeleteCmd:             "Delete Selected",
        cm.BatchUpdateElementsCmd:     "Batch Edit Element Properties",
    }
    for cls, expected in cases.items():
        inst = cls.__new__(cls)
        assert w._command_label(inst) == expected, cls.__name__


def test_unknown_description_falls_back_to_command(qt_app):
    """A command with an empty/missing description should still render."""
    w = _window(qt_app)

    class Dummy:
        description = ""

    assert w._command_label(Dummy()) == "Command"
