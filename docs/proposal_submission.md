# CE 4011 Term Project — Proposal

**Ali Utku Tekin — 2744076**
MSc. Structural Engineering · Spring 2025–2026

## Structural system and analysis method
The program analyses 2-D plane frames and trusses by the Direct Stiffness
Method. Frame members are Euler–Bernoulli beam-columns with three DOFs
per node (uₓ, u_y, θ_z); truss members are pin-pin axial bars. The solver
supports nodal loads, support settlements (partitioned solve), uniform
distributed and in-span point loads on frames, axial-thermal and
through-depth thermal-gradient loading, and internal hinges via Schur
static condensation.

## Planned advanced feature — free-vibration modal analysis
The program will assemble a global mass matrix from element-level
consistent-mass matrices and material density ρ, then solve the
generalised eigenvalue problem `(K − ω²M)φ = 0` on the active free-DOF
block via `scipy.linalg.eigh`. Outputs are natural angular frequencies
ω_n, frequencies f_n = ω_n / 2π, periods T_n, and mass-orthonormal
mode shapes φ_n. Density is read in SI kg/m³ and converted internally
to the solver's kN-m-s system.

## Input / output
The existing `.txt` input format is preserved; density on `MATERIALS`
becomes the only new field and defaults to 0 for backward compatibility.
Outputs: free-DOF displacements, support reactions, element end-forces,
an equilibrium-residual check, and — for modal runs — a frequency /
period table plus the selected mode shape.

## Visualisation (PyQt6 + matplotlib)
The GUI provides model creation, undeformed and (cubic-Hermite) deformed
shapes, bending-moment / shear-force / axial-force diagrams, reaction
arrows, and a modal-results pane with a mode selector and a deformation
scale slider. A time-domain mode animation is included as optional
polish.

## Architecture and verification
Class layout: `StructuralModel`, `Node`, `Material`, `Section`,
`Element2D` ↳ {`FrameElement2D`, `TrussElement2D`}, with `assembler`,
`solver`, `postprocessor`, `mass`, and `modal` modules and a separate
`ModalResult` dataclass. Verification: pytest unit tests, closed-form
checks (cantilever PL³/3EI, simply-supported PL/4), modal references
(clamped-free β_n L = 1.875, 4.694, 7.855; Chopra two-DOF shear
building), and a mode-orthogonality check φᵢᵀMφⱼ = 0.
