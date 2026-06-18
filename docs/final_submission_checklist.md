# Final Submission Checklist (ZIP package)

Use this to assemble the final submission archive. The required naming format is:

```
NameSurname_ID_Report.zip
```

Replace `NameSurname` and `ID` with your own (e.g. `AliTekin_2744076_Report.zip`).

---

## Contents of the ZIP

| # | Item | Where it is in this repo | Status |
|---|------|--------------------------|--------|
| 1 | Source code (or GitHub link) | `structural_analysis/`, `tests/` (+ repo URL on the cover page) | ✅ included |
| 2 | Executable / runnable project folder | the whole project — run with `python -m structural_analysis.gui_qt` or `python -m structural_analysis.main <file>` | ✅ runnable (see installation manual) |
| 3 | Installation manual / installer | `docs/installation_manual.md` | ✅ included |
| 4 | User manual | `docs/user_manual.md` | ✅ included |
| 5 | Project report | `report/final_project_report.md` (+ `.pdf` if rendered) | ✅ included |
| 6 | Example input files | `examples/final_demo/*.txt`, `examples/final_demo/backup_demo_model.spa.json`, `inputs/*.txt` | ✅ included |
| 7 | Example output files | `examples/final_demo/outputs/*.txt`, `outputs/*.txt` | ✅ included |
| 8 | Verification examples | `docs/verification/final_verification.md` | ✅ included |
| 9 | UML / architecture docs | `docs/uml/class_diagram.mmd`, `class_diagram.dot`, `class_diagram.png`, `architecture.md` | ✅ included |
| 10 | External files required to run | none beyond the pip dependencies in `pyproject.toml` | ✅ documented |
| 11 | `video_link.txt` | repo root | ⚠️ paste the real YouTube unlisted link before zipping |

---

## Pre-zip verification
- [ ] Tests pass: `QT_QPA_PLATFORM=offscreen python -m pytest -q`
- [ ] A demo model runs:
      `python -m structural_analysis.main examples/final_demo/demo_portal_frame.txt`
- [ ] `video_link.txt` contains the real (unlisted) YouTube link, verified in a
      private browser window.
- [ ] `report/final_project_report.pdf` is present (or the export note in the
      report has been followed to generate it).
- [ ] Cover page of the report is filled in (name, ID, date, repo URL).
- [ ] The `report/final_project_report.md` exports cleanly to PDF (≤ ~10 pages
      excluding cover/refs/appendices).

## Build the archive
- [ ] Optionally exclude `.venv/`, `__pycache__/`, and `.git/` to keep the ZIP small.
- [ ] Name it `NameSurname_ID_Report.zip`.
- [ ] Open the ZIP on a clean machine and re-run the installation manual's
      one-minute smoke test to confirm it is self-contained.
