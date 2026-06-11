# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CE4011 Assignment 4 — a 2D/3D structural-analysis solver (frames + trusses,
thermal loads, support settlements, modal analysis [2D-only]) with a PyQt6 GUI
on top of a pure NumPy/SciPy core. Python ≥ 3.11.

3D (v0.32): a model with out-of-plane content (node z ≠ 0, 3D supports/loads,
native `Element3D` instances, or `model.force_3d`) solves through a
6-DOF-per-node pipeline. `element3d.py` owns the 12-DOF space frame / space
truss and the 2D→3D promotion applied at assembly time
(`assembler.model_is_3d` / `prepare_solve_elements`); planar models keep the
legacy 2D pipeline bit-identical. The GUI canvas shows 3D models via
work-plane projection (View → Work plane: XY/XZ/ZY/isometric); geometry
creation happens on the XY plane at the View → Working depth z level.

## Commands

```bash
# Run the full test suite (GUI smoke tests need an offscreen Qt platform)
QT_QPA_PLATFORM=offscreen python -m pytest -q

# A single file / test / keyword
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_validation.py -q
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_qt_smoke.py::test_solve_all_skips_empty_load_case -q
QT_QPA_PLATFORM=offscreen python -m pytest -q -k "validation or solve"

# Lint (ruff is installed; there is no in-repo config, so it runs on defaults)
ruff check structural_analysis tests

# Launch the GUI
python -m structural_analysis.gui_qt

# Run the CLI solver on an input file
python -m structural_analysis.main inputs/q2a_settlement.txt
```

There are ~670 tests across 26 files in `tests/`. Example input files live in
`inputs/` (the `example_*` set plus the graded `q2a_settlement.txt` /
`q2b_thermal.txt`).

## Architecture

Three layers, strictly separated (see the **No solver/model/I/O changes from GUI
PRs** rule below). Diagrams: `docs/architecture.md`.

**1. Analysis core** (`structural_analysis/*.py`) — pure, no Qt/matplotlib:
- `model.py` — `StructuralModel` aggregates frozen dataclasses: `Node`, `Material`,
  `Section`, `Support` (with prescribed settlements), `Element2D` (abstract) →
  `FrameElement2D` / `TrussElement2D`, loads (`NodalLoad`, `UniformDistributedLoad`,
  `PointLoad`, `FrameTemperatureLoad`, `TrussTemperatureLoad`), `LoadCase`,
  `LoadCombination`.
- Pipeline orchestrated by `main.run_analysis` / `run_multi_case_analysis`:
  `file_io.read_input_file` → `assembler.assemble_global_system` (builds K, F via
  `DofManager`) → `solver.solve_system` (partitioned `K_ff·D_f = F_f − K_fs·D_s`,
  so settlements are prescribed restrained-DOF displacements, not fake loads) →
  `postprocessor` (member forces, reactions, equilibrium) → `AnalysisResult`.
- `multi_case_result.py` runs many cases and linearly combines them per
  `LoadCombination`. `modal.py` / `mass.py` / `profiles.py` add eigenanalysis,
  mass matrices, and the section-profile library.
- Polymorphism is the design spine: each element subtype specializes stiffness,
  load handling, and force recovery. Element/thermal-load mismatches raise
  `TypeError` (a frame rejects `TrussTemperatureLoad` and vice-versa).

**2. `gui_common/`** — Qt-free shared logic usable by any front end:
- `commands.py` — every model mutation is a `Command` with `do(model)`/`undo(model)`;
  this is what makes the GUI undoable.
- `validation.py` — the **pre-solve UX** validator (orphan nodes, unsupported
  components, truss free-end mechanisms). Distinct from `assembler.validate_model`,
  which is the core DOF/singularity check. Two `validate_model`s, different layers,
  on purpose.
- `geometry.py`, `results_view.py`, `file_writer.py`.

**3. `gui_qt/`** — PyQt6 presentation:
- `app.MainWindow` owns the model, the undo/redo stack, and orchestrates everything.
- `canvas.py` (matplotlib draw + selection + validation-highlight layer),
  `controllers.py` (Tool state machines: click/key/release), `dialogs.py`,
  `grid.py`/`snap.py`, `view3d.py`, `modal_view.py`, and the `*_summary` panels.
- `element_graphics.py` — see the single-source-of-truth rule below.

---

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

## Input file format

Plain-text sections parsed by `file_io.read_input_file`. Key Assignment-4 additions:

- `MATERIALS`: `<id> <A> <I> <E> [alpha] [depth]` — `alpha`/`depth` drive thermal effects.
- `SUPPORTS`: `<node_id> <ux> <uy> <rz> [settle_ux settle_uy settle_rz]`.
- `FRAME_TEMPERATURE`: `<element_id> <t_top> <t_bottom>` (mean → axial, difference → bending gradient).
- `TRUSS_TEMPERATURE`: `<element_id> <delta_T>` (uniform ΔT along the bar).

3D additions (v0.32 — see `inputs/example_3d_grillage.txt`):

- `NODES` rows accept an optional 4th column: `<id> <x> <y> [z]`.
- `SUPPORTS3D`: `<node_id> <ux> <uy> <uz> <rx> <ry> <rz> [6 settlements]`
  (separate keyword — a widened SUPPORTS row would be ambiguous with the
  legacy 3-flags + 3-settlements form).
- `LOADS3D`: `<node_id> <fx> <fy> <fz> <mx> <my> <mz> [case=NAME]`.
- `MEMBER_UDL` / `MEMBER_POINT_LOADS` accept an optional third numeric
  component (`wz` / `pz`).
- `ANALYSIS_OPTIONS`: `force_3d=<bool>` forces the 6-DOF pipeline on a
  planar model.
