# Final Video Outline — CE 4011 Demonstration

**Target length:** 5–10 minutes (aim for **8–9 minutes**).
**Format:** screen recording with voice-over. Not a word-for-word script — these
are cues, talking points, and a few key sentences.

> ⚠️ **The demo must NOT look hardcoded.** At least one model must be built
> live, from scratch, through the normal GUI input workflow (drawing nodes,
> elements, supports, loads). The saved example files are a backup/reproducibility
> aid only — never the centrepiece.
>
> 🎙️ **Say this line on camera when you reference the saved file:**
> *"This model is created through the normal user input workflow. The saved
> example file is only included for reproducibility and backup."*

---

## 0:00 – 0:40 — Intro: purpose, scope, DSM

**Show:** title slide or the app's main window at rest.

**Talking points**
- This is a 2D structural-analysis program for frames and trusses, built on the
  **Direct Stiffness Method**.
- It computes nodal displacements, member internal forces (N, V, M), and support
  reactions, and draws the deformed shape and force diagrams.
- It also handles thermal loads, support settlements, load combinations, and
  modal analysis.

**Key sentence:** *"The program uses the direct stiffness method to solve plane
frames and trusses, and reports displacements, internal forces, reactions, and
their diagrams."*

---

## 0:40 – 2:00 — Program structure + OOP in the engine

**Show:** the class diagram (`docs/uml/class_diagram.png`) and briefly the
source tree (`structural_analysis/`, `gui_common/`, `gui_qt/`).

**Talking points**
- Three layers, one-way dependency: **engine** (pure NumPy/SciPy) → **editor**
  (Qt-free commands + validation) → **GUI** (PyQt6).
- The engine is object-oriented around an abstract **`Element2D`** base, with
  **`FrameElement2D`** and **`TrussElement2D`** subclasses that specialize
  stiffness, load handling, and force recovery.
- The **`StructuralModel`** aggregates nodes, materials, sections, supports,
  elements, and loads. **`DofManager`** numbers the equations; **`assembler`**,
  **`solver`**, and **`postprocessor`** run the pipeline.
- Mention type-safety: a truss thermal load on a frame raises an error by
  design.

**Key sentence:** *"An abstract element base class lets the assembler treat every
member the same way, while each element type provides its own stiffness and
force recovery — that's the polymorphism at the heart of the engine."*

---

## 2:00 – 4:30 — GUI input: build a model FROM SCRATCH (the important part)

**Show:** start from File ▸ New, then draw the portal frame live.

**Do, on camera (this is the live, non-hardcoded part):**
1. **Geometry:** Node tool → place 4 nodes `(0,0)`, `(6,0)`, `(0,4)`, `(6,4)`.
   Frame tool → connect `1→3`, `2→4`, `3→4`.
2. **Material/section:** Materials… → add Concrete C30; add section 30×50
   (`A = 0.15`, `I = 3.125e-3`, depth `0.5`) and assign to all elements.
3. **Supports:** Support tool → fix nodes 1 and 2.
4. **Loads:** Nodal load → 30 kN lateral at node 3; Member load → −15 kN/m UDL
   on the beam (element 3).

**Talking points**
- Narrate units as you go (metres, kN, kN·m).
- Point out snapping/grid for clean geometry and that every edit is undoable.

**Key sentence:** *"I'm building this portal frame from scratch using the normal
drawing tools — nodes, members, supports, then loads."*

> If anything goes wrong while drawing, recover gracefully: open the backup
> (`examples/final_demo/backup_demo_model.spa.json`) via File ▸ Open and say the
> reproducibility/backup line above.

---

## 4:30 – 5:30 — Run the analysis

**Show:** Analyze ▸ Solve all cases (F5).

**Talking points**
- The pipeline numbers DOFs, validates connectivity, assembles `K` and `F`,
  solves `K_ff·D_f = F_f − K_fs·D_s`, then recovers forces and reactions.
- If the model were unstable, the SVD check would flag the mechanism instead of
  crashing.

**Key sentence:** *"One click solves the model: assemble the global stiffness
matrix, apply boundary conditions, solve, and recover member forces."*

---

## 5:30 – 6:40 — Results: deformed shape, diagrams, reactions

**Show:** toggle the overlay panel checkboxes one at a time.

**Do**
- **Deformed shape** (bump the deformed scale so the sway/sag is visible).
- **Diagrams ▸ M (moment)**, then **V (shear)**, then **N (axial)**.
- **Reactions** — read out base reactions
  (node 1 ≈ `Rx −2.48`, `Ry 37.0`, `M 19.6`; node 2 ≈ `Rx −27.5`, `Ry 53.0`,
  `M 52.5`).
- Hover over a member to show the live internal-force read-out.

**Talking points**
- Global equilibrium check: `ΣFy = 90 kN` equals the total UDL (15 × 6); the two
  horizontal reactions sum to −30 kN, balancing the lateral load.
- All diagrams come from one shared N/V/M routine, so hover/diagram/detail always
  agree.

**Key sentence:** *"The deformed shape and the moment, shear, and axial diagrams
are all drawn from the same internal-force routine, and the reactions satisfy
global equilibrium."*

---

## 6:40 – 8:20 — Verification

**Show:** open the cantilever model
(`examples/final_demo/verification_cantilever.txt`) — this is where the saved
file is legitimately used — **and** the verification table
(`docs/verification/final_verification.md`). Optionally run it in the CLI.

**Do**
- State the hand calc: tip deflection `PL³/3EI`, fixed-end moment `PL = 40 kN·m`,
  reaction `10 kN`.
- Show the program reproduces `δ = 5.236e-2 m`, `M = 40 kN·m`, `R = 10 kN`.
- 🎙️ **Say the reproducibility/backup line here** when you open the saved file.

**Talking points**
- Agreement is to machine precision; equilibrium residual ~1e-13.
- Mention ~670 automated tests back the engine.

**Key sentence:** *"The cantilever matches the closed-form hand calculation
exactly — tip deflection, fixed-end moment, and reaction all agree, which
confirms the assembly, boundary conditions, and force recovery are correct."*

---

## 8:20 – 9:00 — Limitations + future work

**Show:** the limitations section of the report or just speak over the app.

**Talking points**
- Honest scope: 2D, linear-elastic, static + modal; no nonlinearity, no P-Δ, no
  design-code checks; self-weight on rigid end zones is neglected.
- Future: 3D extension, geometric stiffness / P-Δ, response-spectrum dynamics,
  sparse storage.

**Key sentence:** *"The program is intentionally scoped to linear-elastic 2D
analysis; the natural next steps are 3D, geometric nonlinearity, and dynamic
analysis."*

**Close:** thank the viewer; mention the repository and that the report,
manuals, and verification are all included in the submission.

---

### Recording tips
- Do a 30-second dry run of the live drawing so it's smooth on camera.
- Keep the backup model one click away (File ▸ Open) in case live input slips.
- Speak the units out loud; keep the cursor movements deliberate.
- Upload as **YouTube unlisted**, paste the link into `video_link.txt`, and
  verify it plays in a private/incognito window.
