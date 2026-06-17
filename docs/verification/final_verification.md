# Verification — CE4011 2D Structural Analysis

This document verifies the solver against closed-form hand calculations. The
numbers in the "Our Program" column are taken directly from the CLI output of
the bundled demo models and can be reproduced with:

```bash
python -m structural_analysis.main examples/final_demo/verification_cantilever.txt
python -m structural_analysis.main inputs/example_02_simply_supported_point_load.txt
```

All quantities are in the consistent **kN – m** unit system.

---

## What the verification checks

A correct displacement-method result requires every stage of the pipeline to be
right. These two examples exercise:

1. **Stiffness matrix assembly** — `EA/L`, `12EI/L³`, `6EI/L²`, `4EI/L`,
   `2EI/L` terms placed into the global `K` (`assembler.assemble_global_system`).
   A wrong term would corrupt the displacements below.
2. **Boundary conditions** — the fixed/pinned/roller restraints partition the
   system correctly (`DofManager`, dynamic rotational-DOF handling).
3. **Load vector formation** — nodal loads and the consistent fixed-end vector
   for the member point load (`FrameElement2D.local_consistent_load`).
4. **Result recovery** — displacements solved (`solver.solve_system`) and member
   end forces recovered via `q = K·d − p` (`postprocessor.compute_member_forces`).
5. **Post-processing sign conventions** — reactions from `R = K·D − F`, and the
   `dM/dx = V` convention used by the diagram helpers.

Both models additionally pass the program's internal equilibrium check
(`equilibrium_check`): max residual at free nodes ≈ 1e-13 (machine precision).

---

## Example 1 — Steel cantilever, tip point load (primary)

**Model** (`examples/final_demo/verification_cantilever.txt`):

- Length `L = 4 m`, fixed at the base, free at the tip.
- Section/material: steel IPE200, `E = 2.10×10⁸ kN/m²`, `I = 1.94×10⁻⁵ m⁴`
  ⇒ `EI = 4074 kN·m²`.
- Load: `P = 10 kN` downward at the tip.

**Closed-form solution (Euler–Bernoulli):**

| Quantity | Formula | Hand value |
|----------|---------|-----------|
| Tip deflection | `δ = P·L³ / (3·EI)` | `10·4³ / (3·4074) = 0.0523646 m` |
| Tip rotation | `θ = P·L² / (2·EI)` | `10·4² / (2·4074) = 0.0196367 rad` |
| Fixed-end moment | `M = P·L` | `40 kN·m` |
| Base shear | `V = P` | `10 kN` |
| Base axial | `N = 0` | `0 kN` |

**Comparison:**

| Quantity | Our Program | Reference (hand) | Difference | Status |
|----------|-------------|------------------|------------|--------|
| Tip vertical deflection (m) | −5.236459e−02 | −5.236460e−02 | < 0.001 % | ✅ PASS |
| Tip rotation (rad) | −1.963672e−02 | −1.963672e−02 | < 0.001 % | ✅ PASS |
| Fixed-end moment Mᵢ (kN·m) | 40.0000 | 40.0000 | 0.0 % | ✅ PASS |
| Base shear Vᵢ (kN) | 10.0000 | 10.0000 | 0.0 % | ✅ PASS |
| Base axial Nᵢ (kN) | 0.0000 | 0.0000 | 0.0 % | ✅ PASS |
| Base vertical reaction Rᵧ (kN) | 10.0000 | 10.0000 | 0.0 % | ✅ PASS |

The agreement is exact to machine precision because beam theory and the
2-node frame stiffness element share the same cubic displacement assumption for
this load case.

---

## Example 2 — Simply supported beam, central point load (secondary)

**Model** (`inputs/example_02_simply_supported_point_load.txt`):

- Length `L = 10 m`, pin at node 1, roller at node 2.
- Same steel IPE200 section, `EI = 4074 kN·m²`.
- Load: `P = 10 kN` downward at midspan (`a = 5 m`).

**Closed-form solution:**

| Quantity | Formula | Hand value |
|----------|---------|-----------|
| Support reactions | `R = P/2` | `5 kN` each |
| End rotation | `θ = P·L² / (16·EI)` | `10·10² / (16·4074) = 0.0153413 rad` |
| Max moment (midspan) | `M = P·L/4` | `10·10 / 4 = 25 kN·m` |

**Comparison:**

| Quantity | Our Program | Reference (hand) | Difference | Status |
|----------|-------------|------------------|------------|--------|
| Reaction at node 1, Rᵧ (kN) | 5.0000 | 5.0000 | 0.0 % | ✅ PASS |
| Reaction at node 2, Rᵧ (kN) | 5.0000 | 5.0000 | 0.0 % | ✅ PASS |
| End rotation at node 1 (rad) | −1.534119e−02 | −1.534130e−02 | < 0.001 % | ✅ PASS |
| End rotation at node 2 (rad) | +1.534119e−02 | +1.534130e−02 | < 0.001 % | ✅ PASS |
| Max bending moment (kN·m) | 25.0000¹ | 25.0000 | 0.0 % | ✅ PASS |

¹ Midspan moment is not a member *end* force (both end moments are 0 for this
single-element model); it is read from the bending-moment diagram, which the
GUI samples through `element_graphics.sample_internal_force`. It also equals the
member end shear `Vᵢ = 5 kN` integrated over the 5 m half-span (`5 × 5 = 25`),
i.e. the program's own end forces give the same value by statics.

---

## Optional: SAP2000 / textbook cross-check (to fill after the demo)

If you run the same two models in SAP2000 (or compare against a textbook table),
record the third-party values here. Until then they are marked `TODO` so no
unverified number is implied.

| Model | Quantity | Our Program | SAP2000 / Textbook | Difference | Status |
|-------|----------|-------------|--------------------|-----------|--------|
| Cantilever | Tip deflection (m) | −0.0523646 | `TODO` | `TODO` | `TODO` |
| Cantilever | Fixed-end moment (kN·m) | 40.0 | `TODO` | `TODO` | `TODO` |
| SS beam | Max moment (kN·m) | 25.0 | `TODO` | `TODO` | `TODO` |

---

## Conclusion

For both hand-verifiable benchmarks the program reproduces the analytical
displacements, reactions, shears, and moments to machine precision, and the
internal equilibrium residual is at the level of floating-point round-off. This
confirms correct stiffness assembly, boundary-condition handling, load-vector
formation, solution, and force recovery for the bending-dominant frame element.
