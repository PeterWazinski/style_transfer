"""Small utility helpers shared across the stylist package."""
from __future__ import annotations

import sys
from pathlib import Path


def _get_project_root() -> Path:
    """Return the project root for resolving relative model paths.

    When frozen (PyInstaller), the root is the directory that contains the
    executable (i.e. ``dist/PetersPictureStyler/``).  In development the root
    is three levels above this file (``src/stylist/_utils.py`` → repo root).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent.parent
