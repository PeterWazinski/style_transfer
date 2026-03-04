"""Stylist app entry point.

Usage::

    python main_image_styler.py

Thin wrapper that delegates to :func:`src.stylist.app.main`.
All application logic lives in ``src/stylist/``.
"""
from __future__ import annotations

import sys

from src.stylist.app import main

if __name__ == "__main__":
    sys.exit(main())
