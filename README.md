# CE4011 Assignment 4 - 2D Structural Analysis Program

```mermaid
flowchart LR
    SM[StructuralModel]
    DM[DofManager]
    EL[Element Hierarchy<br/>Element2D → Frame / Truss]
    TL[Thermal Loads<br/>FrameTemperatureLoad<br/>TrussTemperatureLoad]
    SP[Support<br/>+ prescribed settlements]
    AS[Assembler]
    SO[Solver<br/>Kff Df = Ff - Kfs Ds]
    PP[Postprocessor]
    AR[AnalysisResult]

    SM --> DM
    SM --> EL
    SM --> SP
    TL --> EL
    DM --> AS
    EL --> AS
    SP --> SO
    AS --> SO
    SO --> PP
    PP --> AR
```

**Ali Utku Tekin - 2744076**  
MSc. Structural Engineering, 3rd Semester | Spring 2025-2026

This package extends the Assignment 3 2D frame-truss solver with the two capabilities required by Assignment 4:

1. **Thermal loading**
   - `TrussTemperatureLoad` for uniform temperature change in truss members
   - `FrameTemperatureLoad` for top/bottom-fiber temperatures in frame members, with the mean-temperature axial effect and the through-depth gradient bending effect derived internally
2. **Support settlements**
   - prescribed displacements at restrained support DOFs handled through the partitioned system
     `K_ff * D_f = F_f - K_fs * D_s`

# Architecture

Diagrams of the `structural_analysis` package: the layered design, the
analysis pipeline, the domain model, and the GUI command/undo flow.

All diagrams are [Mermaid](https://mermaid.js.org/) — they render natively on
GitHub and in most IDEs.

---

## 1. Layered architecture

A pure analysis core with two layers of GUI on top. `CLAUDE.md` enforces the
boundary: GUI-scoped PRs may not modify the solver / model / I/O core.

```mermaid
flowchart TB
    subgraph GUI["gui_qt — PyQt6 presentation"]
        app["app.MainWindow<br/>(owns model + undo stack)"]
        canvas["canvas.py<br/>matplotlib draw, selection,<br/>validation highlights"]
        controllers["controllers.py<br/>Tool state machines"]
        egfx["element_graphics.py<br/>N/V/M math — single source of truth"]
        dialogs["dialogs · grid · snap · view3d<br/>modal_view · *_summary panels"]
    end

    subgraph COMMON["gui_common — Qt-free shared logic"]
        commands["commands.py<br/>Command.do / undo"]
        guival["validation.py<br/>pre-solve UX checks"]
        geometry["geometry.py"]
        resultsview["results_view.py"]
        filewriter["file_writer.py"]
    end

    subgraph CORE["Analysis core — pure, no Qt"]
        main["main.py<br/>run_analysis · run_multi_case_analysis"]
        fileio["file_io.read_input_file"]
        model["model.StructuralModel"]
        assembler["assembler.py<br/>assemble_global_system · DofManager<br/>validate_model (DOF/singularity)"]
        solver["solver.solve_system"]
        postproc["postprocessor.py<br/>member_forces · reactions · equilibrium"]
        multicase["multi_case_result.py<br/>MultiCaseAnalysisResult + combine"]
        modal["modal.py · mass.py · profiles.py"]
    end

    GUI --> COMMON --> CORE
    app --> main
    commands --> model
    main --> fileio --> model
    main --> assembler --> solver --> postproc
    main --> multicase
    egfx -.->|reads| model
```

---

## 2. Analysis pipeline (data flow)

The single-case solve path through the core, as orchestrated by
`main.run_analysis`.

```mermaid
flowchart LR
    A["input .txt"] -->|read_input_file| B[StructuralModel]
    B -->|assemble_global_system| C["K, F, DofManager,<br/>warnings, elem_data"]
    C -->|solve_system| D["D (displacements),<br/>residual, warnings"]
    D --> E[compute_member_forces]
    D --> F[compute_reactions]
    E --> G[AnalysisResult]
    F --> G
    G -->|equilibrium_check| H["equilibrium residuals"]

    B -. "run_multi_case_analysis<br/>(per case)" .-> I[MultiCaseAnalysisResult]
    I -. "LoadCombination<br/>linear combine" .-> I
```

---

## 3. Domain model (class diagram)

Frozen dataclasses hang off `StructuralModel`. `Element2D` is the abstract base
for the two element kinds.

```mermaid
classDiagram
    class StructuralModel {
        +dict nodes
        +dict materials
        +dict sections
        +dict supports
        +list elements
        +list nodal_loads
        +dict load_cases
        +list load_combinations
    }

    class Node {
        +int id
        +float x
        +float y
    }
    class Material {
        +int id
        +float E
    }
    class Section {
        +int id
        +int material_id
        +float A
        +float I
    }
    class Support {
        +int node_id
        +bool ux
        +bool uy
        +bool rz
    }
    class Element2D {
        <<abstract>>
        +int id
        +int node_i
        +int node_j
        +int section_id
    }
    class FrameElement2D
    class TrussElement2D
    class NodalLoad {
        +int node_id
        +str load_case
    }
    class LoadCase {
        +str name
        +bool enabled
    }
    class LoadCombination {
        +str name
        +dict factors
    }
    class AnalysisResult
    class MultiCaseAnalysisResult

    Element2D <|-- FrameElement2D
    Element2D <|-- TrussElement2D

    StructuralModel "1" *-- "*" Node
    StructuralModel "1" *-- "*" Material
    StructuralModel "1" *-- "*" Section
    StructuralModel "1" *-- "*" Support
    StructuralModel "1" *-- "*" Element2D
    StructuralModel "1" *-- "*" NodalLoad
    StructuralModel "1" *-- "*" LoadCase
    StructuralModel "1" *-- "*" LoadCombination

    Section --> Material : material_id
    Element2D --> Section : section_id
    NodalLoad --> LoadCase : load_case
    LoadCombination ..> LoadCase : factors over cases

    StructuralModel ..> AnalysisResult : run_analysis
    StructuralModel ..> MultiCaseAnalysisResult : run_multi_case_analysis
```

---

## 4. GUI command / undo flow

Every model mutation in the GUI goes through a `Command` so it can be undone.
`app.MainWindow` owns the model and the undo/redo stacks.

```mermaid
flowchart TB
    user([User action]) --> tool["controllers.Tool<br/>(on_click / on_key / on_release)"]
    tool --> host["app.MainWindow (host)"]
    host -->|push| cmd["Command.do(model)"]
    cmd --> model[(StructuralModel)]
    host -->|append| undo["undo stack"]
    host -->|invalidate| inval["clear stale result +<br/>canvas validation highlights"]
    model --> redraw["canvas.redraw()"]

    undo -. "Ctrl+Z" .-> undocmd["Command.undo(model)"]
    undocmd --> model
```

---

## Invariants worth knowing

- **N/V/M math lives only in `gui_qt/element_graphics.py`** (`evaluate_internal_force`,
  `sample_internal_force`, `internal_force_at`). Every diagram caller routes through it;
  the `dM/dx = V` sign convention is tested only there.
- **Two `validate_model` functions, on purpose:** `assembler.validate_model` (core
  DOF/singularity check) vs. `gui_common.validation.validate_model` (pre-solve UX pass
  that highlights problems on the canvas). Different layers.
- **GUI changes bump the version:** any visible GUI change sets `__version__` and
  `__what_is_new__` in `structural_analysis/__init__.py` (currently `0.22.0`).
- **Solver isolation:** GUI-scoped PRs must not touch `solver`, `assembler`,
  `postprocessor`, `modal`, `element`, `model`, `profiles`, `file_io`, `mass`,
  `gui_common/{file_writer,commands}`, `project_io`, `main`, or `inputs/*.txt`.

## Package contents

- `structural_analysis/` - source code
- `tests/` - automated tests
- `inputs/` - Assignment 4 Q2 input files
- `outputs/` - generated console/text outputs for the reported Q2 runs
  


## OOP / architecture highlights

- `Element2D` is the abstract base class for all structural members.
- `FrameElement2D` and `TrussElement2D` specialize stiffness, compatible load handling, and force recovery polymorphically.
- `Material` stores `E`, `A`, `I`, `alpha`, and `depth` because the thermal expansion coefficient and the section depth are intrinsic material/section properties, not properties of a specific load.
- Strict validation rejects incompatible thermal load-element combinations with a clear `TypeError`.
- Support settlements are modeled as prescribed restrained-DOF displacements rather than fake external loads.

## How to run

From the package root:

```bash
python -m structural_analysis.main inputs/q2a_settlement.txt
python -m structural_analysis.main inputs/q2b_thermal.txt
python -m pytest -q
```

## Current automated test status

The included suite contains **59 automated tests**, all passing:

- 53 preserved Assignment 3 tests
- 6 new Assignment 4 tests (4 unit, 2 integration)

## Input format notes

### `MATERIALS`

```text
MATERIALS n
<id> <A> <I> <E> [alpha] [depth]
```

### `SUPPORTS`

```text
SUPPORTS n
<node_id> <ux_fix> <uy_fix> <rz_fix> [settle_ux settle_uy settle_rz]
```

### `FRAME_TEMPERATURE`

```text
FRAME_TEMPERATURE n
<element_id> <t_top> <t_bottom>
```

### `TRUSS_TEMPERATURE`

```text
TRUSS_TEMPERATURE n
<element_id> <delta_T>
```

## Assignment 4 modeling note for Q2(b)

The reported Q2(b) solution uses the **standard centerline idealization**. The thermal gradient of the concrete beam is modeled explicitly with `t_top = 0` and `t_bottom = +50`, which gives both a mean-temperature axial effect and a bending-gradient effect. Any secondary eccentricity between the beam centroidal axis and the bottom-fiber truss attachment is neglected as a higher-order refinement.

## GitHub

The public repository for this assignment lives at:

`https://github.com/tekcentu/StructureGUI`

Update the report cover page accordingly.
