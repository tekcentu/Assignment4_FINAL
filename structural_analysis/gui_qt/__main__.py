"""Entry point: ``python -m structural_analysis.gui_qt [input_file]``."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from .app import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    window = MainWindow(initial_path=initial)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
