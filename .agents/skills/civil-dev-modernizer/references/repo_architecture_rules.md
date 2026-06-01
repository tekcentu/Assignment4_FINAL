# CIVIL DEV architecture and commands

## Layers

- Core: `structural_analysis/*.py`, pure solver/model/I/O/postprocessing/modal code.
- Common GUI services: `structural_analysis/gui_common/`, Qt-free commands, validation, geometry, results, file writer.
- PyQt GUI: `structural_analysis/gui_qt/`, presentation and interaction.

## Important repo rules

- Keep GUI-scoped PRs out of solver/model/I/O files unless the task is explicitly split.
- Use `gui_qt/element_graphics.py` helpers as the single source of truth for N/V/M diagram math.
- Keep `gui_common.validation.validate_model` separate from core validation/assembly.
- Do not add hidden restraints or fake stiffness to make singular models solve.
- Do not suppress numerical singularity errors.
- Update `structural_analysis/__init__.py` user-visible version metadata for GUI-visible feature changes.

## Common commands

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_validation.py -q
QT_QPA_PLATFORM=offscreen python -m pytest -q -k "validation or solve"
ruff check structural_analysis tests
python -m structural_analysis.main inputs/q2a_settlement.txt
python -m structural_analysis.gui_qt
```
