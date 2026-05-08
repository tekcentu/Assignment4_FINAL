"""Entry point: ``python -m structural_analysis.gui [input_file]``."""

from __future__ import annotations

import sys

from .app import MainApplication


def main() -> int:
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    app = MainApplication(initial_path=initial)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
