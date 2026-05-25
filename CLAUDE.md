# Project rules — read every session

## Always bump the user-visible version on a shipping change

Any PR that adds, removes, or visibly changes a feature in the PyQt6 GUI MUST also:

- Set `structural_analysis/__init__.py :: __version__` (semver-ish, e.g. 0.7.0)
- Set `structural_analysis/__init__.py :: __what_is_new__` (≤ 3 short clauses separated
  by ` · `, shown in the MainWindow menu-bar badge — keep it terse)

Do NOT ship a GUI change without these two lines. Previous sessions forgot; this rule exists
so future sessions don't.

## N/V/M diagram math — single source of truth

`structural_analysis/gui_qt/element_graphics.py` owns:

  evaluate_internal_force(elem, ni, nj, f_local, kind)
  sample_internal_force(elem, ni, nj, f_local, kind, n_samples)
  internal_force_at(elem, ni, nj, f_local, kind, x_loc)

All callers — canvas overlays, hover read-out, detail dialog, future stress ribbons —
MUST go through these helpers. Never write a second BMD/SFD formula in a dialog or
viewer; the `dM/dx = V` sign convention is only tested on these helpers.

## No solver / model / I/O changes from GUI PRs

PRs scoped to the GUI must not modify:
  structural_analysis/{solver,assembler,postprocessor,modal,element,model,
                        profiles,file_io,mass}.py
  structural_analysis/gui_common/{file_writer,commands}.py
  structural_analysis/{project_io,main}.py
  inputs/*.txt

If a GUI PR needs solver work, split it.
