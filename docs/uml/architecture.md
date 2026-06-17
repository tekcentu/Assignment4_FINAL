# Architecture — CE4011 2D Structural Analysis

A short, source-grounded architecture note for the final report. Every class
and module named here exists in `structural_analysis/`. Diagrams:
[`class_diagram.mmd`](class_diagram.mmd) (Mermaid),
[`class_diagram.dot`](class_diagram.dot) / `class_diagram.png` (Graphviz).

> **Naming note.** The assignment brief refers to `solver/`, `editor/`, and
> `gui/` packages. The real repository uses different directory names for the
> same three layers. This document uses the **real** names and maps them to the
> brief's vocabulary so nothing is invented:
>
> | Brief term | Real package | Role |
> |------------|--------------|------|
> | `solver/`  | `structural_analysis/` (top-level `.py` modules) | computational engine |
> | `editor/`  | `structural_analysis/gui_common/` | Qt-free shared logic (commands, validation, file writer) |
> | `gui/`     | `structural_analysis/gui_qt/` | PyQt6 presentation |

---

## 1. Package split and ownership

### Computational engine — `structural_analysis/*.py` (pure NumPy/SciPy, no Qt)

| Module | Owns |
|--------|------|
| `model.py` | The domain model: `StructuralModel` aggregate plus the frozen dataclasses `Node`, `Material`, `Section`, `Support`, `NodalLoad`, `UniformDistributedLoad`, `PointLoad`, `TrussTemperatureLoad`, `FrameTemperatureLoad`, `LoadCase`, `LoadCombination`, `JointMass`, `ModalMassSource`, and the `AnalysisResult` container. |
| `element.py` | The element class hierarchy: abstract `Element2D` → `FrameElement2D`, `TrussElement2D`. Local stiffness, consistent load vectors, moment-release condensation, rigid end offsets, mass matrices, and end-force recovery. |
| `assembler.py` | `DofManager` (dynamic equation numbering) + `assemble_global_system` + the core `validate_model` (DOF/singularity/connectivity audit). |
| `solver.py` | `solve_system` — the partitioned linear solve with SVD singularity detection and prescribed-displacement (settlement) support. |
| `postprocessor.py` | `compute_member_forces`, `compute_reactions`, `equilibrium_check`. |
| `main.py` | Orchestration: `run_analysis`, `run_multi_case_analysis`, `run_from_file`, and the per-load-case filtering context manager. |
| `multi_case_result.py` | `MultiCaseAnalysisResult` — per-case results, `SUM_ALL`, and linear `LoadCombination` superposition. |
| `file_io.py` | `read_input_file` — the plain-text input parser. |
| `modal.py`, `mass.py`, `profiles.py` | Eigen/modal analysis, mass matrices, and the section-profile library. |

### Editor layer — `structural_analysis/gui_common/` (Qt-free, reusable by any front end)

| Module | Owns |
|--------|------|
| `commands.py` | The `Command` base class and ~40 concrete commands (`AddNodeCmd`, `AddMemberCmd`, `SetSupportCmd`, `AddNodalLoadCmd`, `AddMemberLoadCmd`, `UpdateElementCmd`, …). Each implements `do(model)` / `undo(model)` — this is what makes every GUI edit undoable. |
| `validation.py` | The pre-solve UX validator: `validate_model` → `ModelValidationResult` of `ValidationIssue`s (orphan nodes, unsupported components, truss free-end mechanisms). Distinct from `assembler.validate_model`. |
| `file_writer.py` | `write_input_file` — serializes a model back to the text format. |
| `results_view.py`, `geometry.py`, `units.py` | Result formatting, geometry helpers, unit presets. |

### GUI layer — `structural_analysis/gui_qt/` (PyQt6 presentation)

| Module | Owns |
|--------|------|
| `app.py` | `MainWindow` — owns the model, the undo/redo stack, and orchestrates solving and display. |
| `canvas.py` | `ModelCanvas` — matplotlib draw, selection, validation highlight overlay. |
| `controllers.py` | The `Tool` state machines: `SelectTool`, `NodeTool`, `FrameTool`, `TrussTool`, `SupportTool`, `NodalLoadTool`, `MemberLoadTool`, `DeleteTool`. |
| `element_graphics.py` | **The single source of truth for N/V/M diagram math** — `evaluate_internal_force`, `sample_internal_force`, `internal_force_at`. |
| `dialogs.py`, `grid.py`, `snap.py`, `view3d.py`, `modal_view.py`, `*_summary.py` | Property dialogs, grid/snap, 3D view, modal results, summary panels. |
| `project_io.py` | `Project` — the `.spa.json` save/load format (model text + view state + selection groups). |

---

## 2. Data flow: GUI input → model → assembler → solver → post-processing

```
 user draws/edits          Command.do(model)            read_input_file(path)
 (Tool on canvas)  ─────►  mutates StructuralModel  ◄─── (CLI / File→Open .txt)
                                   │
                                   ▼
                    assemble_global_system(model)
                    ├─ DofManager.from_model(model)   (equation numbering)
                    ├─ validate_model(model, dofs)    (connectivity / singularity)
                    └─ build K, F, elem_data
                                   │
                                   ▼
            solve_system(K, F, dofs, D_prescribed)     K_ff·D_f = F_f − K_fs·D_s
                                   │
                                   ▼
            compute_member_forces / compute_reactions / equilibrium_check
                                   │
                                   ▼
                    AnalysisResult  (or MultiCaseAnalysisResult)
                                   │
                                   ▼
            MainWindow → ModelCanvas + element_graphics  (deformed shape, N/V/M,
                                                          reactions, tables)
```

The same engine path serves both front ends: the CLI calls `run_from_file`
→ `read_input_file` → `run_analysis`; the GUI calls `MainWindow.solve()` →
`run_multi_case_analysis`. Neither path is privileged — the engine never imports
Qt.

---

## 3. Key design decisions (one-line rationale each)

- **Abstract `Element2D` with `FrameElement2D` / `TrussElement2D` subclasses** —
  polymorphism lets the assembler treat every member uniformly while each
  subtype specializes stiffness, load handling, and force recovery.
- **Frozen dataclasses for the domain model** — model data is immutable, so a
  result can never silently diverge from the geometry it was solved on.
- **`Material` (E, α, ρ) split from `Section` (A, I, depth)** — these are
  independent physical concerns; one material drives many sections.
- **Settlements as prescribed restrained-DOF displacements** (partitioned
  `K_ff·D_f = F_f − K_fs·D_s`) — physically correct, not faked with large
  springs or dummy loads.
- **Element/thermal-load type checking raises `TypeError`** — a frame rejects
  `TrussTemperatureLoad` and vice-versa, so a modeling mistake fails loudly.
- **Dynamic rotational-DOF omission in `DofManager`** — pure-truss nodes get no
  `rz` DOF, avoiding spurious singular matrices without user intervention.
- **SVD rank check in the solver** — instability is reported with the offending
  mechanism DOFs instead of a cryptic `LinAlgError`.
- **Command pattern for every GUI edit** — uniform undo/redo and a clean
  Qt-free boundary (`gui_common` knows nothing about widgets).
- **N/V/M math centralized in `element_graphics.py`** — the `dM/dx = V` sign
  convention is defined and tested in exactly one place.
- **Load cases solved independently, combinations derived by superposition** —
  exact for a linear-elastic solver and avoids re-solving.

---

## 4. How engine OOP is separated from GUI presentation

The dependency direction is strictly one-way: `gui_qt → gui_common → engine`.
The engine modules import only NumPy/SciPy; `gui_common` imports the engine but
no Qt; `gui_qt` imports both. This is enforced as a project rule (see
`CLAUDE.md`: *"No solver/model/I/O changes from GUI PRs"*). The practical proof
is that the entire analysis pipeline runs headless from the CLI
(`python -m structural_analysis.main <file>`) with no Qt installed.

---

## 5. Architecture observations found in the current code (reported honestly)

- **Two `validate_model` functions, by design.** `assembler.validate_model` is
  the core DOF/singularity check; `gui_common.validation.validate_model` is the
  pre-solve UX pass that highlights problems on the canvas. They live in
  different layers and are intentionally separate, but the shared name can
  confuse a first read.
- **The engine is module-oriented, not class-oriented, at the service level.**
  `assembler`, `solver`, `postprocessor`, `main`, `file_io`, and `modal` expose
  functions rather than service classes. The diagrams render them with a
  `«module»` stereotype to stay faithful to the code rather than inventing
  wrapper classes.
- **`element_graphics.py` (a `gui_qt` module) holds engineering math, not just
  drawing.** It is the single source of truth for N/V/M diagram values. This is
  a deliberate documented invariant, but it means one piece of mechanics lives
  in the GUI package rather than the engine. It reads `f_local` from the engine
  and never mutates the model, so the one-way dependency rule still holds.
- **The repo predates this brief's `solver/`/`editor/`/`gui/` names.** The
  mapping table in §0 reconciles them; no renaming was done as part of this
  submission package (it would violate the no-refactor rule).
