"""Backend logic for scripts/add_style.ipynb.

Provides all non-UI functions so the notebook remains a thin cockpit.

Functions
---------
setup()
    Locate repo root, assert catalog + content image exist, load catalog.
pick_onnx_file()
    Open a tkinter file dialog and return the three related paths.
report_model_files(onnx_path, pth_path, data_path)
    Print a human-readable report of which model files are present.
validate_style_id(name, existing_ids)
    Validate a proposed style name and return ``(style_id, message)``.
install_style(...)
    Copy model files, generate preview, append to catalog.json.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

class CatalogContext(NamedTuple):
    repo_root: Path
    styles_dir: Path
    catalog_path: Path
    content_image: Path
    catalog: dict
    existing_ids: set[str]


def setup(repo_root: Path | None = None) -> CatalogContext:
    """Locate the repo, assert required files exist, and load catalog.json.

    Args:
        repo_root: Explicit repo root.  When *None* the parent of the
                   ``scripts/`` folder is used (works when the CWD is
                   ``scripts/`` or the repo root).

    Returns:
        :class:`CatalogContext` named-tuple with all resolved paths plus the
        parsed catalog dict and set of existing style IDs.
    """
    if repo_root is None:
        # works from scripts/ (notebook cwd) or from repo root
        _here = Path(__file__).resolve().parent
        repo_root = _here.parent

    styles_dir = repo_root / "styles"
    catalog_path = styles_dir / "catalog.json"
    content_image = repo_root / "sample_images" / "arch.png"

    assert catalog_path.exists(), f"Catalog not found: {catalog_path}"
    assert content_image.exists(), f"Content image not found: {content_image}"

    with open(catalog_path, encoding="utf-8") as fh:
        catalog: dict = json.load(fh)

    existing_ids: set[str] = {s["id"] for s in catalog["styles"]}

    print(f"Repository     : {repo_root}")
    print(f"Existing styles: {sorted(existing_ids)}")

    return CatalogContext(
        repo_root=repo_root,
        styles_dir=styles_dir,
        catalog_path=catalog_path,
        content_image=content_image,
        catalog=catalog,
        existing_ids=existing_ids,
    )


# ---------------------------------------------------------------------------
# File picker
# ---------------------------------------------------------------------------

class ModelPaths(NamedTuple):
    onnx_path: Path
    pth_path: Path
    data_path: Path


def pick_onnx_file() -> ModelPaths:
    """Open a tkinter file dialog so the user can select an ``.onnx`` file.

    Returns:
        :class:`ModelPaths` with the selected ``.onnx`` path and the
        co-located ``.pth`` / ``.onnx.data`` paths (which may not exist).

    Raises:
        SystemExit: When the user cancels the dialog.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    onnx_str = filedialog.askopenfilename(
        title="Select the .onnx model file",
        filetypes=[("ONNX model", "*.onnx")],
    )
    root.destroy()

    if not onnx_str:
        raise SystemExit("No file selected — re-run this cell to try again.")

    onnx_path = Path(onnx_str)
    pth_path = onnx_path.with_suffix(".pth")
    data_path = Path(str(onnx_path) + ".data")

    print(f".onnx : {onnx_path}")
    print(f".pth  : {pth_path}")

    return ModelPaths(onnx_path=onnx_path, pth_path=pth_path, data_path=data_path)


# ---------------------------------------------------------------------------
# File validation report
# ---------------------------------------------------------------------------

def report_model_files(onnx_path: Path, pth_path: Path, data_path: Path) -> None:
    """Assert the ``.onnx`` file exists and print a size report for all three paths.

    Args:
        onnx_path:  Path to the ``.onnx`` file (must exist).
        pth_path:   Path to the optional ``.pth`` file.
        data_path:  Path to the optional ``.onnx.data`` file.

    Raises:
        AssertionError: When *onnx_path* does not exist.
    """
    assert onnx_path.exists(), f"ONNX file not found: {onnx_path}"

    print(f"  .onnx      OK  ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    if pth_path.exists():
        print(f"  .pth       OK  ({pth_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print("  .pth       -- not present (optional; only needed for NST models)")

    if data_path.exists():
        print(f"  .onnx.data OK  ({data_path.stat().st_size / 1e6:.1f} MB)  -- will be copied too")
    else:
        print("  .onnx.data -- not present (weights embedded in .onnx)")


# ---------------------------------------------------------------------------
# Style ID validation
# ---------------------------------------------------------------------------

def validate_style_id(name: str, existing_ids: set[str]) -> tuple[str, str]:
    """Derive and validate the style ID from a display name.

    Args:
        name:         Raw display name, e.g. ``"Anime Hayao"``.
        existing_ids: Set of IDs already present in the catalog.

    Returns:
        ``(style_id, message)`` where *message* starts with ``"OK"`` or
        ``"Warning"``.
    """
    style_id = name.strip().lower().replace(" ", "_")
    if not style_id:
        return ("", "")
    if style_id in existing_ids:
        return (style_id, f"Warning: ID '{style_id}' already exists in catalog -- choose a different name.")
    return (style_id, f"OK  ID will be: '{style_id}'")


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def install_style(
    *,
    onnx_path: Path,
    pth_path: Path,
    data_path: Path,
    style_name: str,
    style_desc: str,
    style_author: str,
    tensor_layout: str,
    styles_dir: Path,
    catalog_path: Path,
    catalog: dict,
    existing_ids: set[str],
    content_image: Path,
    repo_root: Path,
    preview_size: int = 256,
) -> str:
    """Copy model files, generate preview, and append the entry to catalog.json.

    Args:
        onnx_path:      Source ``.onnx`` file selected by the user.
        pth_path:       Co-located ``.pth`` (may not exist; skipped if absent).
        data_path:      Co-located ``.onnx.data`` (may not exist; skipped if absent).
        style_name:     Human-readable display name (e.g. ``"Anime Hayao"``).
        style_desc:     Short description for the gallery card.
        style_author:   Author name for the catalog entry.
        tensor_layout:  ``"nchw"`` or ``"nhwc_tanh"``.
        styles_dir:     Repo ``styles/`` directory.
        catalog_path:   Path to ``styles/catalog.json``.
        catalog:        Already-parsed catalog dict (will be mutated and saved).
        existing_ids:   Set of IDs already in the catalog (for pre-check).
        content_image:  Content photo used to generate the preview thumbnail.
        repo_root:      Repo root (added to sys.path so ``src/`` is importable).
        preview_size:   Edge length of the preview thumbnail in pixels.

    Returns:
        The new ``style_id`` string.

    Raises:
        AssertionError: On empty name or duplicate ID.
    """
    style_name = style_name.strip()
    style_author = style_author.strip() or "PeterWazinski"

    assert style_name, "Style name is empty -- fill in the widgets first."
    style_id = style_name.lower().replace(" ", "_")
    assert style_id not in existing_ids, f"'{style_id}' already exists in the catalog."

    dest_dir = styles_dir / style_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # -- Copy model files
    shutil.copy2(onnx_path, dest_dir / "model.onnx")
    if pth_path.exists():
        shutil.copy2(pth_path, dest_dir / "model.pth")
        print("Copied model.pth")
    if data_path.exists():
        shutil.copy2(data_path, dest_dir / "model.onnx.data")
        print("Copied model.onnx.data")

    # -- Generate preview
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.trainer.preview import generate_preview  # noqa: PLC0415

    preview_path = dest_dir / "preview.jpg"
    generate_preview(
        onnx_path=dest_dir / "model.onnx",
        preview_path=preview_path,
        content_image=content_image,
        size=preview_size,
        tensor_layout=tensor_layout,
    )
    print(f"Preview generated: {preview_path}")

    # -- Update catalog.json
    catalog["styles"].append({
        "id":            style_id,
        "name":          style_name,
        "description":   style_desc,
        "author":        style_author,
        "model_path":    f"styles/{style_id}/model.onnx",
        "preview_path":  f"styles/{style_id}/preview.jpg",
        "is_builtin":    False,
        "tensor_layout": tensor_layout,
    })
    with open(catalog_path, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=2)

    print(f"\nOK  '{style_name}'  (id='{style_id}', layout='{tensor_layout}')  added to gallery.")
    print("Files:")
    for p in sorted(dest_dir.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    print("\nNext: git add -A ; git commit -m 'feat: add <name> style'")

    return style_id
