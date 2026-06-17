# User Manual — CE4011 2D Structural Analysis

This manual covers everyday use: building a model (in the GUI or as a text
file), running the analysis, and reading the results. It ends with one complete
worked example, from an empty canvas to the deformed shape and bending-moment
diagram.

> Units throughout are the consistent **kN – m** system: lengths in metres,
> forces in kN, moments in kN·m, `E` in kN/m², `A` in m², `I` in m⁴.

---

## 1. Two ways to create a model

| | GUI | Text input file |
|--|-----|-----------------|
| Start | `python -m structural_analysis.gui_qt` | edit a `.txt` file |
| Best for | drawing, demos, interactive editing | reproducible/graded runs, version control |
| Run | **Analyze ▸ Solve all cases** (F5) | `python -m structural_analysis.main file.txt` |

Both feed the **same** analysis engine, so results are identical.

---

## 2. Loading and saving

- **New model:** File ▸ New.
- **Open:** File ▸ Open… accepts both solver inputs (`*.txt`) and GUI projects
  (`*.spa.json`).
- **Open example:** File ▸ Open example… lists the bundled `inputs/` files.
- **Save:** File ▸ Save / Save As… — `*.spa.json` keeps the model **and** the
  view/grid state; `*.txt` writes a plain solver input.

---

## 3. Building a model in the GUI

The toolbar holds the drawing **tools** (single-key shortcuts in brackets):

| Tool | Shortcut | What it does |
|------|----------|--------------|
| Select | `S` | pick nodes/elements; drag to move; opens property dialogs on double-click |
| Node | `N` | click to place a node (snaps to grid) |
| Frame | `F` | click two nodes to create a frame (beam-column) element |
| Truss | `T` | click two nodes to create a truss (axial-only) element |
| Support | — | click a node to set/edit restraints |
| Nodal load | — | click a node to add a force/moment |
| Member load | — | click an element to add a UDL or point load |
| Delete | — | click to remove a node/element |

Undo/redo (`Ctrl+Z` / `Ctrl+Y`) work for **every** edit.

### 3.1 Geometry (nodes and elements)

1. Choose the **Node** tool and click to place nodes (use the grid + snapping
   for clean coordinates). You can also use **Edit ▸ Add node by coordinates…**
   for exact positions.
2. Choose **Frame** or **Truss** and click a start node then an end node to
   create a member.

### 3.2 Materials and sections

1. Open **Edit ▸ Materials…** (also reachable from the element dialog).
2. A **Material** holds `E`, thermal coefficient `α`, and density `ρ`
   (e.g. Steel S275, Concrete C30).
3. A **Section** holds `A`, `I`, and `depth`, and points to a material
   (e.g. an IPE200 or a 300×500 rectangle). The section dialog includes shape
   calculators (rectangle / square / I-section) that fill `A` and `I` for you.
4. Assign a section to elements via the element dialog or **Batch assign…**.

### 3.3 Supports

Pick the **Support** tool and click a node, or double-click a node with the
Select tool. In the dialog, tick the restrained DOFs (`ux`, `uy`, `rz`):

- Pin = `ux`, `uy` restrained; Roller = one translation restrained;
  Fixed = all three restrained.
- **Support settlement:** enter a non-zero prescribed displacement in the
  settlement fields to impose a known support movement (Assignment-4 feature).

### 3.4 Loads

- **Nodal load** tool → click a node → enter `Fx`, `Fy`, `Mz`.
- **Member load** tool → click an element → add a **UDL** (`wy` transverse,
  optionally `wx` axial; local/global/gravity frame) or a **point load** at a
  distance `a` along the member.
- **Thermal loads** (Assignment-4): a frame member takes top/bottom fibre
  temperatures (mean → axial, difference → bending gradient); a truss member
  takes a uniform `ΔT`.
- **Load cases / combinations:** loads can be tagged into named cases, and
  combinations (e.g. `1.2·DEAD + 1.6·LIVE`) are formed by superposition.

---

## 4. Running the analysis

- **Analyze ▸ Solve all cases** (`F5`) — solves every enabled load case.
- **Analyze ▸ Solve active case** — solves just the case shown in the results
  selector.
- **Analyze ▸ Modal analysis…** (`F6`) — natural frequencies and mode shapes.
- **Analyze ▸ Analysis settings…** — toggle self-weight, etc.

If the model is invalid, the pre-solve validator highlights the problem on the
canvas (orphan nodes, unsupported parts, mechanisms) instead of producing a
silent wrong answer.

---

## 5. Reading the results

After solving, the right-hand **overlay panel** controls what is drawn:

| Toggle | Shows |
|--------|-------|
| Deformed shape | the displaced structure (use **View ▸ Deformed scale** to amplify) |
| Reactions | support reaction arrows/moments |
| Diagrams | internal-force diagram on each member, choose one of: |
| ↳ M (moment) | bending-moment diagram (BMD) |
| ↳ V (shear) | shear-force diagram (SFD) |
| ↳ N (axial) | axial-force diagram (AFD) |
| Section labels / Physical | annotation and true-thickness rendering |

- **Hover** over a member to read the internal force value at the cursor.
- **Double-click** a member (Select tool) for a detail dialog with its end
  forces and full N/V/M plot.
- All diagram values come from one shared routine
  (`element_graphics.evaluate_internal_force`), so the hover read-out, the
  canvas diagram, and the detail dialog always agree (`dM/dx = V`).
- **File ▸ Export station results…** writes a CSV of N/V/M sampled at 21
  stations per member (handy for comparing against SAP2000).

The CLI prints the same information as a text report: equation numbering, the
assembled `K`/`F`, nodal displacements, member end forces, support reactions,
and an equilibrium check.

---

## 6. Complete example workflow (portal frame)

This reproduces the bundled `examples/final_demo/demo_portal_frame.txt`: a
single-bay portal frame, 6 m wide × 4 m tall, with a 30 kN lateral load and a
15 kN/m gravity UDL on the beam.

**A. Geometry**

1. Node tool → place 4 nodes: `(0,0)`, `(6,0)`, `(0,4)`, `(6,4)`.
2. Frame tool → connect `1→3` (left column), `2→4` (right column),
   `3→4` (beam).

**B. Material + section**

3. Materials… → add **Concrete C30** (`E = 3.3×10⁷ kN/m²`, `α = 1e-5`,
   `ρ = 2500`).
4. Add a section **Concrete 30×50** (`A = 0.15 m²`, `I = 3.125×10⁻³ m⁴`,
   `depth = 0.5 m`) and assign it to all three elements.

**C. Supports**

5. Support tool → set nodes **1 and 2** to **Fixed** (`ux, uy, rz` all ticked).

**D. Loads**

6. Nodal load tool → node **3** → `Fx = 30 kN` (lateral).
7. Member load tool → element **3** (the beam) → UDL `wy = −15 kN/m`
   (downward).

**E. Solve**

8. Analyze ▸ Solve all cases (`F5`).

**F. Results** (verify against the committed output
`examples/final_demo/outputs/demo_portal_frame_output.txt`)

9. Enable **Deformed shape** — the frame sways right and the beam sags.
10. Enable **Diagrams ▸ M (moment)** — peak beam/column moments appear at the
    joints (~57.5 kN·m at the right joint).
11. Enable **Reactions** — base reactions are:
    - Node 1: `Rx = −2.48 kN`, `Ry = 37.01 kN`, `M = 19.57 kN·m`
    - Node 2: `Rx = −27.52 kN`, `Ry = 52.99 kN`, `M = 52.52 kN·m`
12. Check global equilibrium: `ΣFy = 90 kN` (= 15 kN/m × 6 m) and
    `ΣFx = 0` (the two `Rx` sum to −30 kN, balancing the applied 30 kN). The
    program reports an equilibrium residual ≈ 1e-13.

You now have a complete model-to-visualization cycle. For a hand-checkable case,
repeat with `examples/final_demo/verification_cantilever.txt` and compare to
`docs/verification/final_verification.md`.
