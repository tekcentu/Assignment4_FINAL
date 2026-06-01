---
name: civil-dev-modernizer
description: Safely modernize Python civil/structural-analysis repositories with pure solver cores and GUI layers using a pilot-first Codex workflow. Use for planning, refactoring, migration, validation, PR review, or implementation involving finite-element solver code, PyQt/Tk GUI code, gui_common boundaries, mechanism validation, load cases, modal analysis, file formats, tests, ExecPlans, parity checks, or code-modernization guides.
---

# CIVIL DEV Modernizer

Use this skill to modernize or review this structural-analysis repository without blurring architecture boundaries or making unstable structures pass silently.

## Required workflow

1. **Bound the pilot.** Pick one flow or bug class. Do not modernize solver, GUI, file formats, docs, and formatting all at once.
2. **Inventory before changing code.** Identify files, current behavior, data flow, structural assumptions, and existing tests.
3. **Write or update pilot docs.** Use the templates in `assets/templates/` when the user asks for planning or modernization docs.
4. **Define validation/parity first.** List tests, independent numerical checks, backwards-compatibility cases, and GUI behavior before implementation.
5. **Implement in small steps.** Add or update tests first for behavior changes, then change production code.
6. **Review architecture and physical correctness.** Check that boundaries are preserved and unstable mechanisms are blocked rather than hidden.
7. **Update the ExecPlan.** Record progress, decisions, surprises, and remaining risks after discovery or implementation.

## Repository architecture rules

Treat this repo as three layers:

- `structural_analysis/*.py`: pure analysis core. Do not import Qt, matplotlib canvas objects, dialogs, or GUI commands here.
- `structural_analysis/gui_common/`: Qt-free shared GUI logic. Put reusable validation/command/file-writing behavior here when it is front-end independent.
- `structural_analysis/gui_qt/`: PyQt6 presentation. Keep drawing, dialogs, controllers, highlights, and user interaction here.

Enforce these modernization constraints:

- Do not add hidden stiffness, artificial springs, fake supports, or fake restraints to make a singular model solve.
- Do not suppress singular-matrix errors or downgrade fatal instability to a warning.
- Do not duplicate solver math, release-condensation logic, settlement handling, thermal-load math, modal math, or N/V/M diagram formulas in GUI presentation code.
- Preserve `gui_common.validation.validate_model` as a pre-solve UX validator distinct from core assembly/solver validation.
- For GUI-visible feature changes, update `structural_analysis/__init__.py` version metadata according to repo instructions.

## Pilot documentation outputs

When asked to create a modernization plan, create or update:

```text
.agents/modernization/<pilot>/pilot_execplan.md
.agents/modernization/<pilot>/pilot_overview.md
.agents/modernization/<pilot>/pilot_design.md
.agents/modernization/<pilot>/pilot_validation.md
```

Start from these templates:

- `assets/templates/template_modernization_execplan.md`
- `assets/templates/template_pilot_overview.md`
- `assets/templates/template_pilot_design.md`
- `assets/templates/template_pilot_validation.md`

Keep docs concise but concrete: cite actual files, functions/classes, tests, and model scenarios.

## Implementation guardrails

Before editing production code:

- Run `python .agents/skills/civil-dev-modernizer/scripts/repo_snapshot.py` if a quick map of repo layers is useful.
- Search with `rg`, not recursive grep.
- Read existing tests near the target behavior.
- For solver/validation changes, create a regression test that fails or demonstrates missing coverage.

During implementation:

- Prefer the smallest layer that owns the behavior.
- Keep public file formats backwards compatible unless the pilot explicitly changes them.
- Keep command/undo mutations in command objects, not scattered through GUI handlers.
- Keep display/highlight code separate from validation rules.

After implementation:

- Run targeted tests first, then `QT_QPA_PLATFORM=offscreen python -m pytest -q` when feasible.
- Run `ruff check structural_analysis tests` when linting is in scope.
- Summarize risks, skipped checks, and independent validation gaps.

## Review checklist

Use this checklist for PR reviews or self-review:

- Architecture: Does the change preserve pure core -> `gui_common` -> `gui_qt` separation?
- Stability: Could any unstable structure solve silently?
- Physics: Are releases, internal hinges, trusses, supports, settlements, thermal loads, and modal mass assumptions handled consistently?
- Numerical proof: Is there an independent expected result for solver math changes?
- Tests: Are there targeted regression tests and enough parity/backwards-compatibility coverage?
- GUI: Is visible behavior versioned and are highlights driven by structured validation data?
- Scope: Should unrelated solver, GUI, docs, or formatting changes be split?

## Helpful references

Read only when needed:

- `references/modernization_workflow.md`: adapted pilot-first workflow from OpenAI's Codex modernization Cookbook.
- `references/repo_architecture_rules.md`: repo-specific boundaries, commands, and structural-analysis guardrails.
