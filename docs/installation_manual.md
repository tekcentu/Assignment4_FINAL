# Installation Manual — CE4011 2D Structural Analysis

This guide gets the program running from a clean checkout on Windows, macOS, or
Linux. It is written so another student or the instructor can reproduce the
results without prior knowledge of the project.

---

## 1. Requirements

| Requirement | Version |
|-------------|---------|
| Python | **3.11 or newer** (the code uses 3.11+ syntax) |
| pip | bundled with Python |
| OS | Windows 10+, macOS 12+, or a modern Linux |

**Python packages** (installed in step 4):

| Package | Why |
|---------|-----|
| `numpy` ≥ 1.24 | linear algebra / matrices |
| `scipy` ≥ 1.10 | eigen-solver for modal analysis |
| `matplotlib` ≥ 3.7 | model canvas + diagram drawing |
| `PyQt6` ≥ 6.6 | the desktop GUI |
| `pytest` ≥ 7.0 | running the test suite (development only) |

> The **analysis engine and the CLI** need only NumPy + SciPy + matplotlib.
> **PyQt6 is required only for the GUI.** If you only want to run input files
> from the command line, you can skip PyQt6.

---

## 2. Get the source

Either clone from GitHub:

```bash
git clone <repository-url>
cd Assignment4_FINAL
```

…or unzip the submitted archive and `cd` into the project folder (the one that
contains `pyproject.toml` and the `structural_analysis/` directory).

---

## 3. Create and activate a virtual environment

Keeping dependencies in a venv avoids clashing with system packages.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now be prefixed with `(.venv)`.

---

## 4. Install the dependencies

The project ships a `pyproject.toml`, so the simplest install is:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

This installs the runtime dependencies **and** `pytest`. If you prefer an
explicit install without the editable package:

```bash
pip install "numpy>=1.24" "scipy>=1.10" "matplotlib>=3.7" "PyQt6>=6.6" "pytest>=7.0"
```

---

## 5. Launch the program

### GUI (recommended for the demo)

```bash
python -m structural_analysis.gui_qt
```

The main window opens with a drawing canvas, a toolbar (Select / Node / Frame /
Truss / Support / Load / Delete), and menus for materials, sections, supports,
loads, analysis, and results.

### Command-line solver (no GUI needed)

```bash
python -m structural_analysis.main inputs/q2a_settlement.txt
python -m structural_analysis.main examples/final_demo/demo_portal_frame.txt
```

This prints the full step-by-step report (equation numbering, K/F, solve,
member forces, reactions, equilibrium).

---

## 6. Run the tests

The suite has ~670 tests. GUI smoke tests need an **offscreen Qt platform**, so
set `QT_QPA_PLATFORM=offscreen`:

```bash
# Whole suite
QT_QPA_PLATFORM=offscreen python -m pytest -q

# A single file or keyword
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_validation.py -q
QT_QPA_PLATFORM=offscreen python -m pytest -q -k "validation or solve"
```

On Windows PowerShell, set the variable first:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -q
```

---

## 7. Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `ModuleNotFoundError: No module named 'numpy'` | The venv isn't active, or step 4 was skipped. Activate the venv and reinstall. |
| `ImportError: Failed to import any of the following Qt binding modules: PyQt6` | PyQt6 isn't installed. Run `pip install PyQt6`. Engine/CLI still work without it. |
| `ImportError: libEGL.so.1: cannot open shared object file` (headless Linux) | Qt needs system graphics libraries. Install them, e.g. on Debian/Ubuntu: `sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libdbus-1-3 libglib2.0-0 libfontconfig1`. |
| GUI tests fail to launch a display on a server | Prefix the command with `QT_QPA_PLATFORM=offscreen`. |
| GUI window doesn't appear over SSH | Use a local machine, or enable X11 forwarding / a virtual display. |
| `python` runs Python 2 | Use `python3` (and `python3 -m venv`). Verify with `python --version`. |
| Wrong Python version (< 3.11) | Install Python 3.11+ from python.org and recreate the venv. |
| `dot: command not found` when re-rendering the UML PNG | Graphviz isn't installed. The PNG is already committed; you only need Graphviz to regenerate it (`sudo apt-get install graphviz` or `brew install graphviz`). |

---

## 8. Quick smoke test (one minute)

```bash
# 1. engine works
python -m structural_analysis.main examples/final_demo/verification_cantilever.txt
# expect: tip uy ≈ -5.236e-2 m, base moment 40 kN·m, reaction 10 kN

# 2. GUI launches
python -m structural_analysis.gui_qt
```

If both run, the installation is complete.
