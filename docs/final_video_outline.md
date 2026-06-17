# Final Video Outline (8–9 minutes)

Purpose: record a practical demonstration of the 2D structural analysis program without making it look hardcoded. The live demo should create the portal frame through the normal GUI input workflow; saved examples under `examples/final_demo/` are for reproducibility and backup.

> Exact line to say: "This model is created through the normal user input workflow. The saved example file is only included for reproducibility and backup."

## 0:00–0:40 — Intro: purpose, scope, and DSM

**Show on screen**
- Repository README and/or the launched application window.
- A simple frame/truss model view if the GUI is already open.

**Speaker notes**
- State that the project is a Python 2D structural analysis tool for frames and trusses.
- Mention the Direct Stiffness Method: element stiffness matrices are transformed and assembled into a global system, supports are applied, displacements are solved, and forces/reactions are recovered.
- Keep scope clear: linear elastic 2D frame/truss analysis, not a commercial design-code package.

**Key sentences**
- “This program models 2D frame and truss structures using the Direct Stiffness Method.”
- “The main outputs are nodal displacements, support reactions, member end forces, and plotted deformed shape/internal force diagrams.”
- “The goal of the demo is to show the normal workflow from model input through analysis and verification.”

## 0:40–2:00 — Program structure and OOP in the computational engine

**Show on screen**
- `structural_analysis/model.py` for core dataclasses.
- `structural_analysis/element.py` for `Element2D`, `FrameElement2D`, and `TrussElement2D`.
- `structural_analysis/assembler.py`, `solver.py`, and `postprocessor.py` for the analysis pipeline.

**Speaker notes**
- Explain that the actual package split is `structural_analysis/` for the solver engine, `structural_analysis/gui_common/` for Qt-free GUI support/commands/validation, and `structural_analysis/gui_qt/` for PyQt6 presentation.
- Core model objects include `StructuralModel`, `Node`, `Material`, `Section`, `Support`, `NodalLoad`, `UniformDistributedLoad`, `PointLoad`, `TrussTemperatureLoad`, `FrameTemperatureLoad`, `LoadCase`, and `LoadCombination`.
- `Element2D` is the shared base class; `FrameElement2D` handles axial plus bending frame behavior, and `TrussElement2D` handles axial-only behavior.
- `DofManager` owns equation numbering and active/free/restrained DOFs. `assemble_global_system()` builds global `K` and `F`. `solve_system()` solves the partitioned system. Post-processing computes member forces, reactions, and equilibrium checks.

**Key sentences**
- “The computational engine is separated from the PyQt6 GUI; the solver files do not depend on GUI widgets.”
- “The OOP center of the solver is the element hierarchy: `Element2D` defines shared transformation, load, mass, and recovery behavior, while `FrameElement2D` and `TrussElement2D` specialize the stiffness and active DOFs.”
- “`StructuralModel` is the container passed from input/GUI into assembly, solution, and post-processing.”

## 2:00–4:30 — GUI input: draw a model from scratch

**Show on screen**
- Launch the GUI.
- Create the live portal frame model from scratch:
  - Nodes: `(0,0)`, `(6,0)`, `(0,4)`, `(6,4)` in meters.
  - Frame elements: left column, right column, top beam.
  - Material: steel demo, `E = 210000000 kN/m²`, `alpha = 1.2e-05 1/°C`, density `7850 kg/m³`.
  - Sections: column `A = 0.006 m²`, `I = 8.0e-5 m⁴`, depth `0.30 m`; beam `A = 0.005 m²`, `I = 6.0e-5 m⁴`, depth `0.25 m`.
  - Fixed supports at both base nodes.
  - Loads: top-left vertical `Fy = -15 kN`; top-right `Fx = +30 kN`, `Fy = -15 kN`; beam UDL `wy = -8 kN/m` in local coordinates.

**Speaker notes**
- Emphasize that the model is created through normal GUI tools.
- Say the required exact line before or after creating the model.
- Mention that `examples/final_demo/demo_portal_frame.txt` has the same model for backup/reproducibility.

**Key sentences**
- “I am drawing the model through the GUI tools: nodes first, then frame members, then materials/sections, supports, and loads.”
- “This model is created through the normal user input workflow. The saved example file is only included for reproducibility and backup.”
- “The saved example uses the same text input format as the rest of the project, so it is not a separate hardcoded demo path.”

## 4:30–5:30 — Run analysis

**Show on screen**
- Click the normal Analyze/Run control.
- Show status/result panel after the solve.
- If there is a validation warning, explain it directly; do not skip it.

**Speaker notes**
- State that the GUI passes the model into the same solver pipeline used by file examples/tests.
- Mention the core sequence: model → DOF manager → assembly → partitioned solve → post-processing.

**Key sentences**
- “The analysis is run by assembling the global stiffness matrix and load vector, applying supports, solving for displacements, and recovering forces.”
- “The same engine is used whether the model came from GUI input or a saved input file.”

## 5:30–6:40 — Results: deformed shape, diagrams, reactions, displacements

**Show on screen**
- Deformed shape view.
- Bending moment diagram (BMD), shear force diagram (SFD), and axial force diagram (AFD) for selected members.
- Results tables/panels for nodal displacements and support reactions.

**Speaker notes**
- Interpret only the main trends: lateral load causes frame sway; gravity load causes beam bending/shear; fixed bases develop reactions and moments.
- Avoid claiming exact values unless reading them directly from the result panel.

**Key sentences**
- “The deformed shape provides a qualitative check on the expected sway and bending behavior.”
- “The internal force diagrams are post-processed from the solved element displacements and loads.”
- “The reaction and displacement tables are the numerical outputs I would use for engineering checks.”

## 6:40–8:20 — Verification

**Show on screen**
- Open `examples/final_demo/verification_cantilever_or_simple_beam.txt` or recreate it quickly.
- Show `docs/verification/final_verification.md`.
- Compare program results with hand calculations for a cantilever beam with tip load.

**Speaker notes**
- Verification model: fixed cantilever, `L = 4 m`, point load `P = 10 kN` downward at the free end, `E = 210000000 kN/m²`, `I = 8.0e-6 m⁴`.
- Reference equations:
  - Tip deflection: `δ = P L³ / (3 E I) = 0.126984 m` downward.
  - Tip rotation: `θ = P L² / (2 E I) = 0.047619 rad` clockwise/negative in the program convention.
  - Fixed vertical reaction: `+10 kN`.
  - Fixed end moment magnitude: `40 kN·m`.
- Compare signs using the program’s displayed convention; use magnitude where the verification table says magnitude.

**Key sentences**
- “This verification case is deliberately simple so the reference values come from hand calculation, not from another black-box program.”
- “The program displacement, rotation, reaction, and fixed-end moment match the hand-calculation values to numerical precision for this model.”
- “This does not prove every feature, but it validates the basic DSM beam behavior used by the larger frame example.”

## 8:20–9:00 — Limitations and future improvements

**Show on screen**
- README known limitations section or a final slide/document.

**Speaker notes**
- Keep limitations honest and bounded.
- Suggested limitations: linear elastic small-displacement behavior; 2D frame/truss only; no design-code checks; input quality and support stability still matter; GUI is a student project interface.
- Future improvements: more validation messages, additional benchmark examples, richer export/reporting, and design-code modules.

**Key sentences**
- “The project focuses on the analysis workflow, not code-based member design.”
- “The solver detects unstable/singular systems rather than adding artificial restraints.”
- “Future improvements would be better reporting, more benchmark cases, and expanded validation around user input.”
