# Architecture Overview for Final Report

## Package split

The current codebase uses a three-layer split under `structural_analysis/`:

1. **Computational engine (`structural_analysis/*.py`)**
   - Owns the structural data model, element stiffness/load behavior, matrix assembly, solving, modal/mass utilities, post-processing, and text file I/O.
   - Important classes/functions: `StructuralModel`, `Node`, `Material`, `Section`, `Support`, load dataclasses, `Element2D`, `FrameElement2D`, `TrussElement2D`, `DofManager`, `assemble_global_system()`, `solve_system()`, `run_analysis()`, and post-processing functions.

2. **Editor/common GUI support (`structural_analysis/gui_common/`)**
   - Owns Qt-free GUI logic that can be shared or tested without rendering widgets.
   - Important responsibilities: undoable `Command` classes, model validation results, file-writing helpers, geometry helpers, units, and result-view formatting support.

3. **PyQt6 GUI (`structural_analysis/gui_qt/`)**
   - Owns presentation and interaction: `MainWindow`, drawing canvas, dialogs, tools/controllers, project JSON I/O, results windows, matrix inspection, load summaries, and modal views.
   - GUI code creates/edits a `StructuralModel`, then calls the same analysis pipeline used by file-based examples.

The user prompt referred to `solver/`, `editor/`, and `gui/`; in this repository those responsibilities map to `structural_analysis/*.py`, `structural_analysis/gui_common/`, and `structural_analysis/gui_qt/` respectively.

## Data flow

1. **GUI input or file input**
   - GUI tools/dialogs and command objects mutate a `StructuralModel` by adding nodes, elements, materials, sections, supports, and loads.
   - Text examples are parsed by `read_input_file()` into the same `StructuralModel` shape.

2. **Model validation and equation numbering**
   - GUI-side validation can report user-facing issues before solving.
   - `DofManager.from_model()` assigns active DOFs, restrained DOFs, and free equation numbers based on supports, element types, releases, and moment loads.

3. **Assembly**
   - `assemble_global_system()` visits the model's elements, asks each element for stiffness/load contributions, transforms local quantities to global coordinates, and assembles the global stiffness matrix `K` and load vector `F`.

4. **Solver**
   - `solve_system()` partitions the system into free and restrained DOFs and solves `K_ff D_f = F_f - K_fs D_s`.
   - Singular or unstable systems are reported as errors/warnings; the solver does not add hidden restraints.

5. **Post-processing**
   - Member end forces are recovered from solved displacements and element loads.
   - Support reactions and equilibrium checks are computed for reporting.
   - GUI result panels and diagrams display these computed outputs rather than re-solving the model.

## Key design decisions and rationale

- **Pure solver core separated from PyQt6:** keeps engineering calculations testable without GUI dependencies.
- **Dataclass model objects:** makes model state explicit and easy to serialize, inspect, and test.
- **Element inheritance (`Element2D` → `FrameElement2D` / `TrussElement2D`):** shares coordinate transformation and recovery behavior while keeping frame and truss stiffness rules separate.
- **Material/section split:** lets multiple sections reference a material and lets element stiffness use effective material plus assigned section properties.
- **`DofManager` centralizes DOF activation:** avoids scattering equation-numbering logic across elements, assembler, and GUI.
- **Partitioned solve with prescribed support displacements:** handles support settlements directly in the DSM equations without altering element stiffness.
- **Undoable command objects in `gui_common`:** keeps editing operations reversible and testable outside the Qt widget layer.
- **Validation distinct from solving:** GUI validation improves user feedback, while the assembler/solver still protect the computational engine from invalid or unstable models.
- **Diagrams as post-processing/presentation:** GUI rendering shows analysis results but does not duplicate solver math or change sign conventions.

## Diagram accuracy notes

The UML diagrams intentionally focus on major classes and relationships for the report. They are not exhaustive: many dialog classes, specialized windows, load-combination helpers, modal classes, and test-only helpers are omitted to keep the diagram readable. The class names and package locations were verified against the current source tree before writing the diagrams.
