# Final submission checklist

Project: **CE 4011 — Assignment 4** · 2D Frame and Truss Structural Analysis Program with GUI
Repository HEAD: `cb48a19` (branch `main`) · Package version `0.40.2`
Build date: 2026-06-17

---

## 1. Deliverables (under `report/`)

| File | Purpose | Status |
|------|---------|--------|
| `report/final_report.md`  | Editable Markdown source of the final report | DONE |
| `report/final_report.docx` | Submission-format Word document (1.2 MB) | DONE |
| `report/final_report.pdf`  | Submission-format PDF (1.3 MB, 17 pages, A4) | DONE |
| `report/figures/`          | All embedded figures as PNG @ 200 dpi | DONE |
| `report/final_submission_checklist.md` | This checklist | DONE |
| `report/build/make_figures.py` | Reproducible figure-generation script (real solver output) | DONE |
| `report/build/make_pdf.py`     | Markdown → PDF generator (reportlab) | DONE |
| `report/build/make_docx.py`    | Markdown → DOCX generator (python-docx) | DONE |

Regenerate everything end-to-end:

```bash
PYTHONPATH=. python report/build/make_figures.py
PYTHONPATH=. python report/build/make_docx.py
PYTHONPATH=. python report/build/make_pdf.py
```

---

## 2. Report sections (17 pages, A4)

| # | Section | Status |
|---|---------|--------|
| — | Title page | included |
| — | Abstract | included |
| 1 | Introduction | included |
| 2 | Structural System and Analysis Method | included (incl. 6×6 frame stiffness, 4×4 truss stiffness, partitioned solve, member loads, sign convention, units) |
| 3 | Program Architecture | included (3 layers + 4 architecture figures) |
| 4 | Implemented Features | included (10 subsections grounded in main-branch code) |
| 5 | GUI Demonstration | included (5 script-generated figures of true repository examples) |
| 6 | Verification and Validation | included (3 quantitative cases + automated test summary) |
| 7 | Example Case Study | included (portal frame, key results table, interpretation) |
| 8 | Error Handling and User Safety | included |
| 9 | Limitations | included |
| 10 | Future Improvements | included |
| 11 | Conclusion | included |
| 12 | Appendix | included (input format, example input, test command, public API table) |

---

## 3. Figures generated (`report/figures/`)

All 12 figures are written by `report/build/make_figures.py`. The analysis-result
figures (5–12) are produced by running real example inputs through the actual solver
and rendering with matplotlib — they are not mock-ups.

| File | Figure # | Subject | Status |
|------|----------|---------|--------|
| `architecture_overview.png`     | 1 | Layered architecture (GUI / common / core) | done |
| `analysis_pipeline.png`         | 2 | `run_analysis` pipeline + multi-case branch | done |
| `data_model_diagram.png`        | 3 | StructuralModel + result containers          | done |
| `gui_workflow.png`              | 4 | Command / undo flow                         | done |
| `example_model.png`             | 5 | Portal frame model + supports + loads       | done |
| `example_deformed_shape.png`    | 6 | Deformed shape after solve                  | done |
| `example_axial_diagram.png`     | 7 | N(x) overlay                                | done |
| `example_shear_diagram.png`     | 8 | V(x) overlay                                | done |
| `example_moment_diagram.png`    | 9 | M(x) overlay                                | done |
| `verification_case_1.png`       | 10 | Cantilever: model + program M(x) vs closed | done |
| `verification_case_2.png`       | 11 | Simply-supported: model + V/M vs textbook  | done |
| `verification_case_3.png`       | 12 | Portal: deformed + reactions table          | done |

---

## 4. Tests run for this report

```
QT_QPA_PLATFORM=offscreen python -m pytest -q --ignore=tests/test_gui_qt_smoke.py -p no:cacheprovider
1136 passed in 27.72s
```

The PyQt6 GUI smoke suite (`tests/test_gui_qt_smoke.py`) is present and runs against
the offscreen Qt platform during feature work, but was intentionally skipped for the
report-only run — the verification surface this report depends on (analysis core,
postprocessor, multi-case combine, diagram math, validators, file I/O, command/undo,
station export, units) is exercised by the 1136-test non-smoke suite.

---

## 5. Honest limitations of this report

* **Live PyQt6 screenshots not used.** No interactive Qt desktop in the build
  environment. The GUI figures (5–9) are **script-generated visualisations of true
  repository examples** rendered with matplotlib through the actual solver, not the
  PyQt6 canvas — they show the same geometry the GUI would draw on screen. This is
  stated explicitly in §5 of the report.
* **PyQt6 smoke suite not rerun for this build.** See §4. Sign convention, multi-case
  combine correctness, rigid-offset behaviour, station-export fidelity, and load-case
  filtering are all locked by the non-smoke regression suite that was rerun.
* **SAP2000 comparison numbers not included.** No actual SAP2000 reference data exists
  in the repository, so the report does not claim a SAP2000 cross-check (the
  station-export CSV is *built* to be SAP-comparable, but the comparison itself is
  not performed in this report).
* **PDF math rendering is plain-text / monospace.** LaTeX-quality math is not
  available in the reportlab build environment, so equations and matrices are
  rendered as monospace text in code blocks. The semantics and the closed-form
  numbers are correct.

---

## 6. Verification numbers locked in the report

* **Case 1 — Cantilever** (`example_01_cantilever_tip_load.txt`):
  closed-form tip deflection **δ = P L³ / (3 E I) = 52.3646 mm**; program returns
  exactly the same to floating-point round-off (5.3 × 10⁻¹⁴ %).
* **Case 2 — Simply-supported beam** (`example_02_simply_supported_point_load.txt`):
  closed-form mid-span moment **M\_max = P L / 4 = 25 kN·m**; program returns exactly
  25 kN·m and end shear ±5 kN.
* **Case 3 — Portal frame** (`example_03_portal_frame_lateral_load.txt`): closed-form
  global equilibrium **ΣR\_x = −50 kN, ΣR\_y = +40 kN**; program matches; free-DOF
  equilibrium residual = 2.7 × 10⁻¹³ (machine precision).

---

## 7. Things explicitly out of scope (project rule, not laziness)

* No solver / model / I/O / postprocessor changes were made for this report.
* No GUI behaviour changes were made.
* No fake screenshots, fake test numbers, or fabricated SAP comparisons.
* No restraint / stiffness "tweaks" to flatter the verification numbers.

---

## 8. Submission readiness

The package is **ready for review/submission**: every required deliverable is
produced, every figure is reproducible from the repository, all verification numbers
are sourced from running the real solver, and the test command + result is the
genuine output of the suite at HEAD.
