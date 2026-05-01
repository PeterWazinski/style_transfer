"""Batch style transfer — apply styles to one image.

Two modes (exactly one must be specified):

``--pdfoverview``
    Creates a DIN-A4 landscape PDF contact sheet (2 rows x 3 columns per page).
    The original image appears in the top-left cell; every style then gets three
    consecutive cells at strengths 100 %, 150 %, and 200 %.  Inference runs once
    per style; the extra strength variants are derived by pixel-level blending.
    Output: ``<stem>_thumbnails.pdf`` next to the source image.

``--replay``
    Applies a saved style-chain YAML (produced by the interactive app) to one
    image, executing every step in order.  Suitable for batch re-application of
    a chain that was tuned interactively.
    Output: ``<stem>_<chain_stem>.jpg`` next to the source image.

Optional arguments for ``--replay``:

``--strength-override N``
    Scale every step's strength by N percent without editing the file.
    E.g. ``--strength-override 60`` turns 150 %/75 % into 90 %/45 %.

Optional filter (``--pdfoverview`` only):

``--style "MyStyle"``
    Apply only the named style (case-insensitive).  Aborts with an error
    message listing available styles if the name is not found in the catalog.

Usage::

    python scripts/batch_styler.py --pdfoverview path/to/photo.jpg
    python scripts/batch_styler.py --pdfoverview path/to/photo.jpg --style "Anime Hayao"
    python scripts/batch_styler.py --replay .\\my_chain.yml path/to/photo.jpg
    python scripts/batch_styler.py --replay .\\my_chain.yml path/to/photo.jpg --strength-override 60
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Repository root ─────────────────────────────────────────────────────────
# When compiled with PyInstaller, sys.executable points to BatchStyler.exe
# which lives alongside styles\ in the dist\PetersPictureStyler\ folder.
# In dev mode, __file__ is scripts/batch_styler.py → parent.parent = repo root.
if getattr(sys, "frozen", False):
    REPO_ROOT: Path = Path(sys.executable).parent
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT))

from src.core.engine import StyleTransferEngine  # noqa: E402

logging.basicConfig(level=logging.WARNING)  # suppress engine INFO noise to stderr

import numpy as np  # noqa: E402  (after sys.path setup)

# ── PDF layout at 150 DPI (DIN A4 landscape = 297 × 210 mm) ─────────────────
DPI: int = 150
A4_W: int = int(297 / 25.4 * DPI)   # ≈ 1752 px
A4_H: int = int(210 / 25.4 * DPI)   # ≈ 1240 px
MARGIN: int = int(12 / 25.4 * DPI)  # ≈  71 px  (border around grid)
GAP: int    = int(7  / 25.4 * DPI)  # ≈  41 px  (gutter between cells)
LABEL_H: int = int(7 / 25.4 * DPI)  # ≈  41 px  (caption below each thumb)
COLS: int = 3
ROWS: int = 2
CELLS_PER_PAGE: int = COLS * ROWS

# Per-cell dimensions
CELL_W: int = (A4_W - 2 * MARGIN - (COLS - 1) * GAP) // COLS
CELL_H: int = (A4_H - 2 * MARGIN - (ROWS - 1) * GAP) // ROWS
IMG_H: int  = CELL_H - LABEL_H      # pixel height reserved for the thumbnail

# Strength levels rendered per style in --pdfoverview mode
PDF_STRENGTHS: list[float] = [1.0, 1.5, 2.0]


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _load_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a PIL font, falling back to the built-in bitmap font."""
    for candidate in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _fit_into(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Return a copy of *image* scaled to fit within (max_w, max_h), aspect-preserved."""
    copy = image.copy()
    copy.thumbnail((max_w, max_h), Image.LANCZOS)
    return copy


def _blend_to_strength(
    original: Image.Image,
    styled: Image.Image,
    strength: float,
) -> Image.Image:
    """Re-apply a strength factor to a pre-computed fully-styled image.

    Uses the same formula as ``StyleTransferEngine.apply()``:
    - strength == 1.0 → return *styled* unchanged.
    - strength  < 1.0 → linear interpolation toward *original*.
    - strength  > 1.0 → extrapolation beyond *styled* (amplified effect).

    Args:
        original: Source image (RGB, any size).
        styled:   Styled image at strength 1.0, same size as *original*.
        strength: Target blend factor.

    Returns:
        A new PIL image blended/extrapolated to *strength*.
    """
    if strength == 1.0:
        return styled.copy()
    arr_orig   = np.array(original.convert("RGB"), dtype=np.float32)
    arr_styled = np.array(styled, dtype=np.float32)
    result = np.clip(arr_orig + strength * (arr_styled - arr_orig), 0, 255)
    return Image.fromarray(result.astype(np.uint8))


def _make_page(
    cells: list[tuple[str, Image.Image]],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Compose one A4-landscape page from up to CELLS_PER_PAGE (name, image) pairs."""
    page = Image.new("RGB", (A4_W, A4_H), color=(255, 255, 255))
    draw = ImageDraw.Draw(page)

    for idx, (style_name, img) in enumerate(cells):
        if img is None:
            continue  # empty cell (e.g. placeholder columns after the original)
        row = idx // COLS
        col = idx % COLS
        x0 = MARGIN + col * (CELL_W + GAP)
        y0 = MARGIN + row * (CELL_H + GAP)

        # Thumbnail centred horizontally in its cell, top-aligned
        thumb = _fit_into(img, CELL_W, IMG_H)
        x_img = x0 + (CELL_W - thumb.width) // 2
        page.paste(thumb, (x_img, y0))

        # Caption below thumbnail
        caption_y = y0 + IMG_H + 3
        draw.text((x0, caption_y), style_name, fill=(50, 50, 50), font=font)

    return page


# ---------------------------------------------------------------------------
# Cell list builder (exported for unit tests)
# ---------------------------------------------------------------------------

def build_cell_list(
    original: Image.Image,
    styled_results: list[tuple[str, Image.Image]],
) -> list[tuple[str, Image.Image]]:
    """Return the ordered cell list: original first, then styled results.

    Args:
        original:       The unmodified source image.
        styled_results: List of (style_name, styled_image) pairs.

    Returns:
        List starting with ``("Original", original)`` followed by all
        *styled_results* in their original order.
    """
    return [("Original", original)] + list(styled_results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_name_to_filename(style_name: str) -> str:
    """Convert a style display name to a safe filename stem."""
    safe = style_name.lower()
    for ch in " /\\:*?\"<>|":
        safe = safe.replace(ch, "_")
    return safe


# ---------------------------------------------------------------------------
# Style filtering
# ---------------------------------------------------------------------------

def filter_styles_by_name(
    styles: list[dict],
    style_name: str,
) -> list[dict]:
    """Return the subset of *styles* whose ``name`` matches *style_name* (case-insensitive).

    Args:
        styles:     Full list of style dicts from the catalog.
        style_name: The requested style name (case-insensitive).

    Returns:
        A single-element list with the matching style dict.

    Raises:
        SystemExit: If no style with the given name is found, prints an error
                    listing available names and exits with code 1.
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


# ---------------------------------------------------------------------------
# Command: --pdfoverview
# ---------------------------------------------------------------------------

def cmd_pdfoverview(
    image_path: Path,
    styles: list[dict],
    tile_size: int,
    overlap: int,
    strength: float,  # noqa: ARG001 — ignored; pdfoverview uses PDF_STRENGTHS
    use_float16: bool,
    out_dir: Path | None = None,
) -> None:
    """Apply all styles at each PDF_STRENGTHS level and write a DIN-A4-landscape PDF.

    Layout (3 columns × 2 rows per page):
    - Page 1, row 1: original image in col 1 only; cols 2 and 3 are left blank.
    - From row 2 onward: each style occupies exactly one row with
      len(PDF_STRENGTHS) cells labelled ``"Style (100%)"`` / ``"(150%)"`` etc.
    - Styles are ordered alphabetically (case-insensitive).

    Inference runs once per style; the extra strength variants are derived by
    pixel-level blending at no additional compute cost.
    """
    source = Image.open(image_path).convert("RGB")
    engine = StyleTransferEngine()

    # Sort styles alphabetically by display name
    styles_sorted = sorted(styles, key=lambda s: s.get("name", s["id"]).casefold())

    # Row 1: original in col 1, cols 2 and 3 intentionally blank
    cells: list[tuple[str, Image.Image | None]] = [
        ("Original", source.copy()),
        ("", None),
        ("", None),
    ]
    n_applied: int = 0

    for style in styles_sorted:
        style_id:   str  = style["id"]
        style_name: str  = style.get("name", style_id)
        model_path: Path = REPO_ROOT / style["model_path"]

        if not model_path.exists():
            print(f"  Skipping '{style_name}' — model not found: {model_path}")
            continue

        tensor_layout: str = style.get("tensor_layout", "nchw")
        print(f"Processing style '{style_name}' ...", flush=True)
        try:
            engine.load_model(style_id, model_path, tensor_layout=tensor_layout)
            styled_full = engine.apply(
                source,
                style_id,
                strength=1.0,
                tile_size=tile_size,
                overlap=overlap,
                use_float16=use_float16,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  Error applying '{style_name}': {exc}")
            continue
        finally:
            engine._sessions.pop(style_id, None)  # noqa: SLF001

        for s in PDF_STRENGTHS:
            label = f"{style_name} ({int(s * 100)}%)"
            cells.append((label, _blend_to_strength(source, styled_full, s)))
        n_applied += 1

    if n_applied == 0:
        sys.exit("No styles were applied successfully — nothing to write.")

    font = _load_font(int(LABEL_H * 0.60))
    n_pages = (len(cells) + CELLS_PER_PAGE - 1) // CELLS_PER_PAGE
    print(f"\nComposing {n_pages} PDF page(s) ...", flush=True)

    pages: list[Image.Image] = []
    for i in range(0, len(cells), CELLS_PER_PAGE):
        pages.append(_make_page(cells[i : i + CELLS_PER_PAGE], font))

    dir_out = out_dir if out_dir is not None else image_path.parent
    pdf_path = dir_out / (image_path.stem + "_thumbnails.pdf")
    pages[0].save(
        pdf_path,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=DPI,
    )

    print(
        f"\nOK  PDF written: {pdf_path}"
        f"  ({len(pages)} page(s), {n_applied} style(s) × {len(PDF_STRENGTHS)} strengths + original)"
    )



# ---------------------------------------------------------------------------
# Command: --replay
# ---------------------------------------------------------------------------

def cmd_replay(
    image_path: Path,
    replay_path: Path,
    tile_size: int | None,
    overlap: int | None,
    use_float16: bool,
    strength_override: int | None = None,
    out_dir: Path | None = None,
) -> None:
    """Apply a saved style-chain YAML to *image_path*, step by step."""
    from src.core.replay_schema import load_replay_log  # noqa: PLC0415

    try:
        replay = load_replay_log(replay_path)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    # CLI args take precedence; fall back to values stored in the YAML, then hard defaults.
    effective_tile_size: int = tile_size if tile_size is not None else (replay.tile_size if replay.tile_size is not None else 1024)
    effective_overlap: int = overlap if overlap is not None else (replay.tile_overlap if replay.tile_overlap is not None else 128)

    catalog_path = REPO_ROOT / "styles" / "catalog.json"
    if not catalog_path.exists():
        sys.exit(f"Error: catalog not found: {catalog_path}")

    with open(catalog_path, encoding="utf-8") as f:
        catalog: dict = json.load(f)
    styles: list[dict] = catalog.get("styles", [])

    print(f"Tile size    : {effective_tile_size} px  overlap: {effective_overlap} px")
    print()

    engine = StyleTransferEngine()
    result = Image.open(image_path).convert("RGB")
    scale = (strength_override / 100.0) if strength_override is not None else 1.0

    for i, step in enumerate(replay.steps, start=1):
        matched = filter_styles_by_name(styles, step.style)
        catalog_style = matched[0]
        model_path = REPO_ROOT / catalog_style["model_path"]
        if not model_path.exists():
            sys.exit(f"Step {i}: model not found for '{step.style}': {model_path}")
        tensor_layout: str = catalog_style.get("tensor_layout", "nchw")
        strength = (step.strength * scale) / 100.0
        effective_pct = int(step.strength * scale)
        print(
            f"Step {i}/{len(replay.steps)}: '{step.style}' @ {effective_pct}% ...",
            flush=True,
        )
        engine.load_model(catalog_style["id"], model_path, tensor_layout=tensor_layout)
        result = engine.apply(
            result,
            catalog_style["id"],
            strength=strength,
            tile_size=effective_tile_size,
            overlap=effective_overlap,
            use_float16=use_float16,
        )
        engine._sessions.pop(catalog_style["id"], None)  # noqa: SLF001

    dir_out = out_dir if out_dir is not None else image_path.parent
    if strength_override is not None:
        fname = f"{image_path.stem}_{replay_path.stem}_{strength_override}.jpg"
    else:
        fname = f"{image_path.stem}_{replay_path.stem}.jpg"
    out_path = dir_out / fname
    result.save(out_path, format="JPEG", quality=92)
    print(f"\nOK  Result written: {out_path}")





# ---------------------------------------------------------------------------
# Main / argument parsing
# ---------------------------------------------------------------------------

_USAGE = r"""
Usage:
  batch_styler.exe --pdfoverview <image>  [options]
  batch_styler.exe --replay      <chain.yml> <image>  [options]

Modes (exactly one required):
  --pdfoverview  Create a DIN-A4 landscape PDF contact sheet with all styles.
                 Each style gets 3 cells: 100 %, 150 %, 200 % strength.
                 Output: <image-dir>/<stem>_thumbnails.pdf
  --replay FILE  Apply a saved style-chain YAML to the image, executing every
                 step in order.
                 Output: <image-dir>/<stem>_<chain-stem>.jpg

Options for --pdfoverview:
  --style NAME           Apply only the named style (case-insensitive).
                         Aborts with an error if the name is not in the catalog.

Options for --replay:
  --strength-override N  Scale every step's strength by N percent (0–300).
                         E.g. --strength-override 60 turns 150% → 90%.
                         Output filename gets a _<N> suffix, e.g. photo_chain_60.jpg.

Common options:
  --tile-size N  Tile size for ONNX inference in pixels (default: 1024)
  --overlap N    Tile overlap in pixels (default: 128)
  --float16      Enable float16 inference (faster on GPU/DML)
  --outdir DIR   Write output file(s) to DIR instead of the source image folder.
                 DIR must already exist.

Examples:
  batch_styler.exe --pdfoverview photos\portrait.jpg
  batch_styler.exe --pdfoverview photos\portrait.jpg --style "Anime Hayao"
  batch_styler.exe --replay my_chain.yml photos\portrait.jpg
  batch_styler.exe --replay my_chain.yml photos\portrait.jpg --strength-override 60
  batch_styler.exe --replay my_chain.yml photos\portrait.jpg --strength-override 80 --outdir C:\output
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch style transfer.",
        add_help=True,
    )
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--pdfoverview", action="store_true",
        help="Create a PDF contact sheet with all styles.",
    )
    mode_group.add_argument(
        "--replay", type=Path, metavar="CHAIN",
        help="Apply a saved style-chain YAML to the image.",
    )
    parser.add_argument("image", type=Path, help="Source image file (JPEG or PNG)")
    parser.add_argument(
        "--tile-size", type=int, default=None,
        help="Tile size for inference in pixels. Overrides the value stored in the replay YAML. Default: use YAML value or 1024.",
    )
    parser.add_argument(
        "--overlap", type=int, default=None,
        help="Tile overlap in pixels. Overrides the value stored in the replay YAML. Default: use YAML value or 128.",
    )
    parser.add_argument(
        "--strength-override", type=int, default=None, metavar="PCT",
        help="Scale all replay step strengths by this percentage (0–300). Only used with --replay.",
    )
    parser.add_argument(
        "--float16", action="store_true", default=False,
        help="Use float16 inference (faster on GPU/DML)",
    )
    parser.add_argument(
        "--style", type=str, default=None, metavar="NAME",
        help="Apply only this style (case-insensitive name). Aborts if not found.",
    )
    parser.add_argument(
        "--outdir", type=Path, default=None, metavar="DIR",
        help="Write output file(s) to DIR instead of the source image folder. DIR must already exist.",
    )
    args = parser.parse_args()

    if not args.pdfoverview and not args.replay:
        print(_USAGE)
        sys.exit(1)

    image_path: Path = args.image.resolve()
    if not image_path.exists():
        sys.exit(f"Error: image not found: {image_path}")

    # Validate --outdir
    out_dir: Path | None = None
    if args.outdir is not None:
        out_dir = args.outdir.resolve()
        if not out_dir.is_dir():
            sys.exit(f"Error: --outdir directory does not exist: {out_dir}")

    # Validate --strength-override range
    if args.strength_override is not None and not (0 <= args.strength_override <= 300):
        sys.exit("Error: --strength-override must be between 0 and 300.")

    if args.replay:
        cmd_replay(
            image_path, args.replay.resolve(),
            tile_size=args.tile_size,
            overlap=args.overlap,
            use_float16=args.float16,
            strength_override=args.strength_override,
            out_dir=out_dir,
        )
        return

    catalog_path = REPO_ROOT / "styles" / "catalog.json"
    if not catalog_path.exists():
        sys.exit(f"Error: catalog not found: {catalog_path}")

    with open(catalog_path, encoding="utf-8") as f:
        catalog: dict = json.load(f)

    styles: list[dict] = catalog.get("styles", [])
    if not styles:
        sys.exit("No styles found in catalog.")

    if args.style:
        styles = filter_styles_by_name(styles, args.style)

    pdf_tile_size: int = args.tile_size if args.tile_size is not None else 1024
    pdf_overlap: int = args.overlap if args.overlap is not None else 128

    print(f"Source image : {image_path}")
    print(f"Styles       : {len(styles)} style(s)" + (f" (filtered: '{args.style}')" if args.style else ""))
    print(f"Tile size    : {pdf_tile_size} px  overlap: {pdf_overlap} px")
    print()

    cmd_pdfoverview(
        image_path, styles,
        tile_size=pdf_tile_size,
        overlap=pdf_overlap,
        strength=1.0,
        use_float16=args.float16,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
