"""Style Transfer app entry point (bin/ stub).

Thin wrapper — all logic lives in :mod:`src.stylist.app`.

Usage::

    python bin/main_image_styler.py

See ``src/stylist/app.py`` for the full application.
"""
from __future__ import annotations

import sys

from src.stylist.app import main

if __name__ == "__main__":
    sys.exit(main())
