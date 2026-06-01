#!/usr/bin/env python3
"""Print a compact CIVIL DEV repository snapshot for modernization planning."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

SECTIONS = {
    "core": ROOT / "structural_analysis",
    "gui_common": ROOT / "structural_analysis" / "gui_common",
    "gui_qt": ROOT / "structural_analysis" / "gui_qt",
    "tests": ROOT / "tests",
    "docs": ROOT / "docs",
    "inputs": ROOT / "inputs",
}

EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def iter_files(path: Path, *, recursive: bool = True) -> list[Path]:
    if not path.exists():
        return []
    iterator = path.rglob("*") if recursive else path.glob("*")
    files: list[Path] = []
    for child in sorted(iterator):
        if any(part in EXCLUDE_DIRS for part in child.parts):
            continue
        if child.is_file():
            files.append(child.relative_to(ROOT))
    return files


def main() -> int:
    print(f"Repository: {ROOT}")
    for name, path in SECTIONS.items():
        files = iter_files(path, recursive=(name != "core"))
        print(f"\n[{name}] {len(files)} files")
        for rel in files[:40]:
            print(f"  - {rel}")
        if len(files) > 40:
            print(f"  ... {len(files) - 40} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
