"""Batch style transfer — apply every catalog style to one image.

Produces a DIN-A4 landscape PDF contact sheet (2 rows × 3 columns per page).

Usage::

    python scripts/batch_styler.py path/to/photo.jpg [--tile-size 1024]
    python scripts/batch_styler.py path/to/photo.jpg --strength 0.9

The output PDF is written next to the source image as
``<stem>_thumbnails.pdf``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Repository root: scripts/ → repo root ───────────────────────────────────
SCRIPTS_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = SCRIPTS_DIR.parent
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply all catalog styles to an image and write a PDF contact sheet."
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
        help="Style blend strength 0.0–1.0 (default: 1.0)",
    )
    parser.add_argument(
        "--float16", action="store_true", default=False,
        help="Use float16 inference (faster on GPU/DML)",
    )
    args = parser.parse_args()

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

    print(f"Source image : {image_path}")
    print(f"Styles found : {len(styles)}")
    print(f"Tile size    : {args.tile_size} px  overlap: {args.overlap} px")
    print()

    source = Image.open(image_path).convert("RGB")
    engine = StyleTransferEngine()
    font = _load_font(int(LABEL_H * 0.60))

    # Original image is always the first cell
    raw_styled: list[tuple[str, Image.Image]] = []

    for style in styles:
        style_id: str   = style["id"]
        style_name: str = style.get("name", style_id)
        model_path: Path = REPO_ROOT / style["model_path"]

        if not model_path.exists():
            print(f"  Skipping '{style_name}' — model not found: {model_path}")
            continue

        print(f"Processing style '{style_name}' …", flush=True)
        try:
            engine.load_model(style_id, model_path)
            result = engine.apply(
                source,
                style_id,
                strength=args.strength,
                tile_size=args.tile_size,
                overlap=args.overlap,
                use_float16=args.float16,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  Error applying '{style_name}': {exc}")
            continue
        finally:
            # Release the ONNX session to keep GPU/CPU memory manageable
            engine._sessions.pop(style_id, None)  # noqa: SLF001

        raw_styled.append((style_name, result))

    if not raw_styled:
        sys.exit("No styles were applied successfully — nothing to write.")

    styled_results = build_cell_list(source, raw_styled)

    # ── Compose PDF pages ────────────────────────────────────────────────────
    print()
    n_styles = len(raw_styled)  # excludes the "Original" cell
    n_pages = (len(styled_results) + CELLS_PER_PAGE - 1) // CELLS_PER_PAGE
    print(f"Composing {n_pages} PDF page(s) …", flush=True)

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
        f"  ({len(pages)} page(s), {n_styles} style(s) + original)"
    )


if __name__ == "__main__":
    main()
