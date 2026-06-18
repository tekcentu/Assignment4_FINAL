# Final Video Recording Checklist

Tick each item before, during, and after recording. The goal: a smooth 8–9
minute demo that shows a model built **live** (not hardcoded) and verified
against a hand calculation.

## Before recording
- [ ] Open the project folder and activate the virtual environment.
- [ ] Run the tests and confirm they pass:
      `QT_QPA_PLATFORM=offscreen python -m pytest -q`
- [ ] Launch the GUI: `python -m structural_analysis.gui_qt`
- [ ] Do a quick dry run of the live drawing so it's smooth on camera.
- [ ] Have the backup model ready to open if live input fails:
      `examples/final_demo/backup_demo_model.spa.json`

## Model creation (LIVE — must not look hardcoded)
- [ ] Start from **File ▸ New** (empty canvas, on camera).
- [ ] **Define geometry:** place nodes and draw frame/truss members with the tools.
- [ ] **Define material/section:** add a material (E, α, ρ) and a section (A, I, depth).
- [ ] **Define supports:** set restraints (fixed/pin/roller) on the support nodes.
- [ ] **Define loads:** add nodal load(s) and a member UDL/point load.
- [ ] Say the required line when referencing any saved file:
      *"This model is created through the normal user input workflow. The saved
      example file is only included for reproducibility and backup."*

## Analysis
- [ ] Run **Analyze ▸ Solve all cases** (F5).

## Results
- [ ] Show the **deformed shape** (amplify with Deformed scale).
- [ ] Show internal-force diagrams: **M (moment)**, **V (shear)**, **N (axial)**.
- [ ] Show **reactions** and read out the values.
- [ ] (Optional) Hover a member to show the live internal-force read-out.

## Verification
- [ ] Open the cantilever model (`examples/final_demo/verification_cantilever.txt`).
- [ ] Show the **verification comparison** table
      (`docs/verification/final_verification.md`): program vs. hand calc.
- [ ] State the match (tip deflection, fixed-end moment, reaction).

## Wrap-up
- [ ] **Mention limitations** (2D, linear-elastic, no nonlinearity/P-Δ/design checks).
- [ ] Mention future improvements briefly.

## After recording
- [ ] Upload the video to **YouTube as Unlisted**.
- [ ] Paste the link into `video_link.txt` (replace the placeholder).
- [ ] Confirm `video_link.txt` exists and contains the real link.
- [ ] Open the link in an **incognito/private browser** to confirm it plays
      without being logged in.
