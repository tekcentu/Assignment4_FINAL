# Codex Code Modernization Guide for CIVIL DEV

This guide adapts OpenAI's Codex code-modernization workflow to this repository: a Python 2D structural-analysis solver with a pure numerical core, Qt-free shared GUI logic, and a PyQt6 presentation layer.

Source basis:

- OpenAI Cookbook, **Modernizing your Codebase with Codex**: <https://developers.openai.com/cookbook/examples/codex/code_modernization>
- OpenAI Codex docs, **Agent Skills**: <https://developers.openai.com/codex/skills>

The Cookbook pattern is pilot-first: choose one flow, create an ExecPlan, draft inventory/overview, design the target behavior, define parity validation, implement in bounded steps, and keep the plan synchronized. This guide translates that pattern from a legacy COBOL example into a safer modernization workflow for this repo.

## 1. Principle: do not modernize everything at once

Avoid broad tasks such as:

```text
Modernize the entire solver and GUI.
```

Use bounded pilot flows instead:

```text
Modernize pre-solve validation for orphan nodes and single-member mechanisms.
```

Good pilot candidates in this repo include:

- mechanism detection in `gui_common.validation`;
- orphan-node and unsupported-component validation;
- load case and load combination flows;
- modal-analysis setup and result display;
- command/undo behavior in `gui_common.commands` plus `gui_qt` controllers;
- project JSON or text input/output compatibility;
- result reporting, member-force diagrams, and equilibrium checks.

## 2. Preserve the repository architecture

This repository has three layers:

1. **Analysis core**: `structural_analysis/*.py`, pure Python/NumPy/SciPy with no Qt or matplotlib dependency.
2. **GUI common**: `structural_analysis/gui_common/`, Qt-free shared logic such as commands, validation, geometry, results view, and file writing.
3. **PyQt GUI**: `structural_analysis/gui_qt/`, presentation, canvas, dialogs, controllers, and panels.

Modernization work must respect these boundaries:

- Core solver/model/I/O code must not import PyQt, Tkinter, dialogs, canvas objects, or GUI command objects.
- GUI code must not duplicate stiffness, release-condensation, settlement, thermal-load, N/V/M, or modal math.
- GUI-scoped work must not change solver/model/I/O behavior. Split the PR if core behavior needs to change.
- Visible GUI changes must update `structural_analysis/__init__.py` version metadata according to `CLAUDE.md`.

## 3. Recommended modernization workspace

Use the repo-scoped Codex location `.agents/` so Codex Cloud, CLI, and IDE sessions can discover checked-in skills and task artifacts:

```text
.agents/
  skills/
    civil-dev-modernizer/
      SKILL.md
      agents/openai.yaml
      references/
      assets/templates/
      scripts/
  modernization/
    validation-mechanisms/
      pilot_execplan.md
      pilot_overview.md
      pilot_design.md
      pilot_validation.md
```

Use one folder per pilot. Do not mix unrelated solver, GUI, docs, and formatting changes in a single pilot.

## 4. Phase 0: set the agent contract

Before implementation, ensure each task repeats the non-negotiable rules:

```markdown
Preserve the architecture: pure core -> gui_common -> gui_qt.
Do not add hidden stiffness, fake restraints, or artificial springs to make singular structures pass.
Do not suppress singular-matrix errors.
Add regression tests before changing solver or validation behavior.
For solver math changes, include an independent expected result.
For GUI changes, do not duplicate N/V/M or stiffness formulas in presentation code.
```

## 5. Phase 1: choose a pilot

A good pilot is small enough to understand and test, but important enough to reduce future risk.

### Good pilot example

```text
Detect and block solving a frame member whose released support end and free far node form a one-member mechanism.
```

### Poor pilot example

```text
Improve all validation, diagrams, solver stability, and GUI warnings.
```

### Pilot selection checklist

- Is there a clear current behavior and target behavior?
- Can tests prove correctness or parity?
- Can the work fit in one focused PR?
- Is the boundary clear between core, `gui_common`, and `gui_qt`?
- Are structural assumptions explicit?

## 6. Phase 2: inventory and discovery

Create `pilot_overview.md` before editing production code.

Template:

```markdown
# Pilot overview - <pilot name>

## Goal
Describe the user-visible or engineering outcome in one paragraph.

## Files involved
| Layer | Files | Notes |
|---|---|---|
| Core | `structural_analysis/...` | Solver/model/element behavior, if in scope |
| GUI common | `structural_analysis/gui_common/...` | Commands, validation, file writer, geometry |
| GUI | `structural_analysis/gui_qt/...` | Canvas, controllers, dialogs, summaries |
| Tests | `tests/...` | Existing and missing coverage |

## Current flow
1. User/model input path.
2. Validation or assembly path.
3. Solve/result path.
4. GUI display or file-output path.

## Known risks
- Structural assumptions.
- Backwards compatibility.
- Numerical parity.
- GUI highlighting or undo behavior.

## Edge cases to inspect
- Orphan nodes.
- Disconnected components.
- Unsupported components.
- Truss free-end mechanisms.
- Frame releases and internal hinges.
- Thermal loads and support settlements.
- Empty or inactive load cases.
```

## 7. Phase 3: design and validation plan

Create `pilot_design.md` and `pilot_validation.md` before implementing.

### Design guidance

- Put reusable non-Qt behavior in `gui_common` or the pure core, not in dialogs.
- Keep structural validation separate from canvas presentation.
- Return affected node IDs, element IDs, and DOFs when validation messages need highlighting.
- Fatal mechanisms must block solve.
- Stable benchmark models must continue to solve.
- Solver singularity errors should remain visible and diagnosable.

### Validation table template

```markdown
| Scenario | Expected result | Test type | Independent check |
|---|---|---|---|
| Orphan node exists | Solve blocked; node identified | Regression/interface | Existing UX rule |
| Released support end + free far node | Solve blocked as mechanism | Regression | Hand stability reasoning |
| Stable portal frame | Solve allowed | Regression | Existing input benchmark |
| Internal hinge benchmark | Hinge moments/reactions match expected | Numerical regression | Hand or trusted reference |
```

## 8. Phase 4: implement in small, reviewable steps

Use this order:

1. Add or update failing tests that capture the pilot behavior.
2. Implement the smallest production change.
3. Run targeted tests.
4. Run the full test suite.
5. Run lint if available.
6. Update the ExecPlan with progress, discoveries, and decisions.
7. Keep the PR focused.

For numerical solver changes, do not rely only on internal equilibrium. A result can balance globally while still using an incorrect stiffness or recovery formulation. Prefer an independent hand calculation, textbook benchmark, trusted tool comparison, or previously validated regression case.

## 9. Phase 5: review before merge

Review structural correctness before style:

- Does the PR preserve the core -> `gui_common` -> `gui_qt` boundary?
- Could an unstable model solve silently?
- Are releases, hinges, trusses, supports, settlements, thermal loads, and modal mass assumptions handled physically?
- Are tests targeted and independent enough?
- Is GUI-visible behavior versioned?
- Should the PR be split?

## 10. Ready-to-paste Codex prompts

### Pick a pilot

```markdown
Use $civil-dev-modernizer. Inspect this repository and propose 2 modernization pilot flows.
For each pilot, list files/modules involved, current behavior, bounded scope, risks, missing tests, and your recommended first pilot.
Do not edit production code.
```

### Create pilot docs

```markdown
Use $civil-dev-modernizer. Create `.agents/modernization/<pilot>/pilot_execplan.md`, `pilot_overview.md`, `pilot_design.md`, and `pilot_validation.md` for <specific pilot>.
Preserve the pure core / gui_common / gui_qt boundaries.
Do not edit production code yet.
```

### Implement a validation fix

```markdown
Use $civil-dev-modernizer.
Goal: detect and block solve for <specific mechanism case>.
Constraints:
- Add or update a regression test first.
- Do not add hidden stiffness, fake supports, or artificial springs.
- Do not suppress singular-matrix errors.
- Preserve pure core / gui_common / gui_qt separation.
Required behavior:
- Fatal validation error is returned.
- Affected node/element IDs are included for GUI highlighting where relevant.
- Stable models still solve.
Run targeted tests and the full suite if feasible.
```

### PR review

```markdown
Use $civil-dev-modernizer. Review this PR as a structural-analysis software reviewer.
Focus on architecture boundaries, physical correctness, unstable mechanisms, test sufficiency, independent numerical checks, GUI version metadata, and whether the PR should be split.
Give a merge/request-changes verdict and a ready-to-paste follow-up prompt.
```

## 11. How to use the included skill

This repository includes the skill at:

```text
.agents/skills/civil-dev-modernizer/
```

Codex can discover repo skills from `.agents/skills` when you run Codex in this repository. In CLI or IDE, run `/skills` or type `$civil-dev-modernizer` in your prompt. In Codex Cloud/App, use this checked-in repository skill by opening a task on this repo and explicitly starting with `Use $civil-dev-modernizer`.
