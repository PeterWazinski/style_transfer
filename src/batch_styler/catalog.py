"""Catalog helpers and REPO_ROOT for BatchStyler.

REPO_ROOT is defined here so that both commands.py and app.py can access it
via module-attribute lookup (``import src.batch_styler.catalog as _catalog;
_catalog.REPO_ROOT``).  Using attribute access (rather than a local binding)
means a single ``patch("src.batch_styler.catalog.REPO_ROOT", ...)`` in tests
covers all callers simultaneously.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# REPO_ROOT
# ---------------------------------------------------------------------------
# When compiled with PyInstaller, sys.executable is BatchStyler.exe which
# lives alongside styles\ in the dist\PetersPictureStyler\ folder.
# In dev mode, __file__ is src/batch_styler/catalog.py → up 3 = repo root.

if getattr(sys, "frozen", False):
    REPO_ROOT: Path = Path(sys.executable).parent
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def _style_name_to_filename(style_name: str) -> str:
    """Convert a style display name to a safe filename stem."""
    safe = style_name.lower()
    for ch in " /\\:*?\"<>|":
        safe = safe.replace(ch, "_")
    return safe


def _list_styles_for_help() -> str:
    """Return a formatted list of available style names for the usage string."""
    catalog_path = REPO_ROOT / "styles" / "catalog.json"
    try:
        with open(catalog_path, encoding="utf-8") as f:
            catalog = json.load(f)
        names = sorted(s.get("name", s["id"]) for s in catalog.get("styles", []))
        return "\n".join(f"  {n}" for n in names) if names else "  (no styles found)"
    except Exception:
        return "  (catalog not found)"


def filter_styles_by_name(
    styles: list[dict],
    style_name: str,
) -> list[dict]:
    """Return the subset of *styles* whose ``name`` matches *style_name* (case-insensitive).

    Raises:
        SystemExit: If no style with the given name is found.
    """
    needle = style_name.strip().casefold()
    matches = [s for s in styles if s.get("name", "").casefold() == needle]
    if not matches:
        available = ", ".join(f"'{s.get('name', s['id'])}' " for s in styles)
        sys.exit(
            f"Error: style '{style_name}' not found in catalog.\n"
            f"Available styles: {available}"
        )
    return matches
