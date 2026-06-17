# Final Video Recording Checklist

## Pre-recording setup
- [ ] Open the project repository.
- [ ] Confirm the branch/PR contains final submission materials.
- [ ] Install dependencies if needed.
- [ ] Run the test suite and note any failures honestly.
- [ ] Launch the PyQt6 GUI.
- [ ] Confirm `video_link.txt` exists.

## Live GUI model creation
- [ ] Create at least one model from scratch through the GUI; do not only load an example.
- [ ] Define geometry: portal frame nodes and frame elements.
- [ ] Define material properties with explicit units.
- [ ] Define section properties with explicit units.
- [ ] Assign fixed supports at the base nodes.
- [ ] Add nodal loads.
- [ ] Add the beam distributed load.
- [ ] Say: “This model is created through the normal user input workflow. The saved example file is only included for reproducibility and backup.”

## Analysis and results
- [ ] Run analysis using the normal GUI action.
- [ ] Show the deformed shape.
- [ ] Show bending moment diagram (BMD).
- [ ] Show shear force diagram (SFD).
- [ ] Show axial force diagram (AFD).
- [ ] Show support reactions.
- [ ] Show nodal displacements.
- [ ] If anything fails or warns, state it clearly in the recording.

## Verification segment
- [ ] Open or recreate `examples/final_demo/verification_cantilever_or_simple_beam.txt`.
- [ ] Show `docs/verification/final_verification.md`.
- [ ] Compare program values against hand calculations.
- [ ] Explain sign convention/magnitude where needed.

## Closing
- [ ] Mention known limitations.
- [ ] Mention future improvements.
- [ ] Confirm the final video link will be pasted into `video_link.txt` before submission.
