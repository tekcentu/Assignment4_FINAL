# Pilot-first modernization workflow

Adapted from OpenAI's Codex code-modernization Cookbook.

## Phases

1. **Pilot selection**: choose one bounded flow, not the whole system.
2. **ExecPlan**: create a living plan that tracks goal, scope, steps, validation, decisions, discoveries, and progress.
3. **Inventory and discovery**: document current files, control/data flow, business or engineering behavior, and risks.
4. **Design and validation**: define target architecture, external behavior, parity tests, and acceptance criteria before code changes.
5. **Implementation**: add tests first, implement in focused steps, run targeted and broad checks.
6. **Review and repeat**: update docs/plan, review risks, and use the completed pilot as a repeatable pattern.

## CIVIL DEV adaptation

For this repo, business behavior usually means structural-analysis or user-modeling behavior:

- whether a model is stable;
- whether loads, settlements, releases, hinges, and modal masses are interpreted correctly;
- whether GUI validation blocks bad models before solve;
- whether file input/output and project JSON remain compatible;
- whether diagrams/results display the same numerical meaning as the core.

Prefer pilot names such as:

- `validation-mechanisms`
- `load-case-manager`
- `modal-analysis-display`
- `project-io-compatibility`
- `member-force-diagrams`
