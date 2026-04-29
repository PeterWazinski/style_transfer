"""Batch style transfer — apply every catalog style to one image.

Two modes (exactly one must be specified):

``--pdfoverview``
    Creates a DIN-A4 landscape PDF contact sheet (2 rows x 3 columns per page).
    The original image appears in the top-left cell; all styles follow.
    Output: ``<stem>_thumbnails.pdf`` next to the source image.

``--fullimage``
    Saves a full-resolution styled JPEG for every style.
    Output: ``<stem>_<style_name>.jpg`` next to the source image.
    The original is not duplicated.

Optional filter:

``--style "MyStyle"``
    Apply only the named style (case-insensitive).  Aborts with an error
    message listing available styles if the name is not found in the catalog.

Usage::

    python scripts/batch_styler.py --pdfoverview path/to/photo.jpg
    python scripts/batch_styler.py --fullimage   path/to/photo.jpg --strength 0.9
    python scripts/batch_styler.py --fullimage   path/to/photo.jpg --style "Anime Hayao"
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


def _make_page(
    cells: list[tuple[str, Image.Image]],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Compose one A4-landscape page from up to CELLS_PER_PAGE (name, image) pairs."""
    page = Image.new("RGB", (A4_W, A4_H), color=(255, 255, 255))
    draw = ImageDraw.Draw(page)

    for idx, (style_name, img) in enumerate(cells):
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
# Shared: run inference for all styles, return list of (name, image) pairs
# ---------------------------------------------------------------------------

def _apply_all_styles(
    source: Image.Image,
    styles: list[dict],
    tile_size: int,
    overlap: int,
    strength: float,
    use_float16: bool,
) -> list[tuple[str, Image.Image]]:
    """Apply every style in *styles* to *source* and return results.

    Skips styles whose model file is missing.  Prints a progress line for
    each style.  Returns a list of ``(style_name, result_image)`` pairs.
    """
    engine = StyleTransferEngine()
    results: list[tuple[str, Image.Image]] = []

    for style in styles:
        style_id: str   = style["id"]
        style_name: str = style.get("name", style_id)
        model_path: Path = REPO_ROOT / style["model_path"]

        if not model_path.exists():
            print(f"  Skipping '{style_name}' — model not found: {model_path}")
            continue

        tensor_layout: str = style.get("tensor_layout", "nchw")
        print(f"Processing style '{style_name}' ...", flush=True)
        try:
            engine.load_model(style_id, model_path, tensor_layout=tensor_layout)
            result = engine.apply(
                source,
                style_id,
                strength=strength,
                tile_size=tile_size,
                overlap=overlap,
                use_float16=use_float16,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  Error applying '{style_name}': {exc}")
            continue
        finally:
            engine._sessions.pop(style_id, None)  # noqa: SLF001

        results.append((style_name, result))

    return results


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
    strength: float,
    use_float16: bool,
) -> None:
    """Apply all styles and write a DIN-A4-landscape PDF contact sheet."""
    source = Image.open(image_path).convert("RGB")
    raw_styled = _apply_all_styles(
        source, styles, tile_size, overlap, strength, use_float16
    )

    if not raw_styled:
        sys.exit("No styles were applied successfully — nothing to write.")

    styled_results = build_cell_list(source, raw_styled)
    font = _load_font(int(LABEL_H * 0.60))

    print()
    n_pages = (len(styled_results) + CELLS_PER_PAGE - 1) // CELLS_PER_PAGE
    print(f"Composing {n_pages} PDF page(s) ...", flush=True)

    pages: list[Image.Image] = []
    for i in range(0, len(styled_results), CELLS_PER_PAGE):
        chunk = styled_results[i : i + CELLS_PER_PAGE]
        pages.append(_make_page(chunk, font))

    pdf_path = image_path.parent / (image_path.stem + "_thumbnails.pdf")
    pages[0].save(
        pdf_path,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=DPI,
    )

    print(
        f"\nOK  PDF written: {pdf_path}"
        f"  ({len(pages)} page(s), {len(raw_styled)} style(s) + original)"
    )


# ---------------------------------------------------------------------------
# Command: --fullimage
# ---------------------------------------------------------------------------

def cmd_fullimage(
    image_path: Path,
    styles: list[dict],
    tile_size: int,
    overlap: int,
    strength: float,
    use_float16: bool,
) -> None:
    """Apply all styles and save a full-resolution JPEG per style."""
    source = Image.open(image_path).convert("RGB")
    raw_styled = _apply_all_styles(
        source, styles, tile_size, overlap, strength, use_float16
    )

    if not raw_styled:
        sys.exit("No styles were applied successfully — nothing to write.")

    print()
    written: list[Path] = []
    for style_name, result in raw_styled:
        stem = _style_name_to_filename(style_name)
        out_path = image_path.parent / f"{image_path.stem}_{stem}.jpg"
        result.save(out_path, format="JPEG", quality=92)
        print(f"  Saved: {out_path}")
        written.append(out_path)

    print(f"\nOK  {len(written)} image(s) written to {image_path.parent}")


# ---------------------------------------------------------------------------
# Main / argument parsing
# ---------------------------------------------------------------------------

_USAGE = r"""
Usage:
  batch_styler.ps1 -pdfoverview <image>  [options]
  batch_styler.ps1 -fullimage   <image>  [options]

Modes (exactly one required):
  -pdfoverview   Create a DIN-A4 landscape PDF contact sheet with all styles.
                 Output: <image-dir>/<stem>_thumbnails.pdf
  -fullimage     Save a full-resolution JPEG for each style.
                 Output: <image-dir>/<stem>_<stylename>.jpg  (one per style)

Options:
  --style NAME   Apply only the named style (case-insensitive).
                 Aborts with an error if the name is not in the catalog.
  --tile-size N  Tile size for ONNX inference in pixels (default: 1024)
  --overlap N    Tile overlap in pixels (default: 128)
  --strength F   Style blend strength 0.0-1.0 (default: 1.0)
  --float16      Enable float16 inference (faster on GPU/DML)

Examples:
  batch_styler.ps1 -pdfoverview photos\portrait.jpg
  batch_styler.ps1 -fullimage   photos\portrait.jpg --strength 0.85
  batch_styler.ps1 -fullimage   photos\portrait.jpg --style "Anime Hayao"
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
        "--fullimage", action="store_true",
        help="Save a full-resolution JPEG for each style.",
    )
    parser.add_argument("image", type=Path, help="Source image file (JPEG or PNG)")
    parser.add_argument(
        "--tile-size", type=int, default=1024,
        help="Tile size for inference (default: 1024)",
    )
    parser.add_argument(
        "--overlap", type=int, default=128,
        help="Tile overlap in pixels (default: 128)",
    )
    parser.add_argument(
        "--strength", type=float, default=1.0,
        help="Style blend strength 0.0-1.0 (default: 1.0)",
    )
    parser.add_argument(
        "--float16", action="store_true", default=False,
        help="Use float16 inference (faster on GPU/DML)",
    )
    parser.add_argument(
        "--style", type=str, default=None, metavar="NAME",
        help="Apply only this style (case-insensitive name). Aborts if not found.",
    )
    args = parser.parse_args()

    if not args.pdfoverview and not args.fullimage:
        print(_USAGE)
        sys.exit(1)

    image_path: Path = args.image.resolve()
    if not image_path.exists():
        sys.exit(f"Error: image not found: {image_path}")

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

    print(f"Source image : {image_path}")
    print(f"Styles       : {len(styles)} style(s)" + (f" (filtered: '{args.style}')" if args.style else ""))
    print(f"Tile size    : {args.tile_size} px  overlap: {args.overlap} px")
    print()

    if args.pdfoverview:
        cmd_pdfoverview(
            image_path, styles,
            tile_size=args.tile_size,
            overlap=args.overlap,
            strength=args.strength,
            use_float16=args.float16,
        )
    else:
        cmd_fullimage(
            image_path, styles,
            tile_size=args.tile_size,
            overlap=args.overlap,
            strength=args.strength,
            use_float16=args.float16,
        )


if __name__ == "__main__":
    main()
