"""PDF layout constants and image-composition helpers for BatchStyler.

All helpers are pure (no I/O, no REPO_ROOT dependency) so they can be
imported and tested without any file-system fixtures.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

# Strength levels rendered per style in --style-overview mode
PDF_STRENGTHS: list[float] = [1.0, 1.5, 2.0]

# ── PDF layout at 150 DPI (DIN A4 portrait = 210 × 297 mm) ─────────────────
A4P_W: int = int(210 / 25.4 * DPI)   # ≈ 1240 px
A4P_H: int = int(297 / 25.4 * DPI)   # ≈ 1754 px
CHAIN_ROWS: int = 2
CHAIN_COLS: int = 1


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
            continue
        row = idx // COLS
        col = idx % COLS
        x0 = MARGIN + col * (CELL_W + GAP)
        y0 = MARGIN + row * (CELL_H + GAP)

        thumb = _fit_into(img, CELL_W, IMG_H)
        x_img = x0 + (CELL_W - thumb.width) // 2
        page.paste(thumb, (x_img, y0))

        caption_y = y0 + IMG_H + 3
        draw.text((x0, caption_y), style_name, fill=(50, 50, 50), font=font)

    return page


def _make_chain_page(
    cells: list[tuple[str, Image.Image | None]],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Compose one A4-portrait page with up to CHAIN_ROWS (1 col x 2 rows) cells."""
    page = Image.new("RGB", (A4P_W, A4P_H), color=(255, 255, 255))
    draw = ImageDraw.Draw(page)
    chain_cell_h = (A4P_H - 2 * MARGIN - GAP) // CHAIN_ROWS
    chain_img_h = chain_cell_h - LABEL_H
    for idx, (name, img) in enumerate(cells[:CHAIN_ROWS]):
        if img is None:
            continue
        row = idx
        y0 = MARGIN + row * (chain_cell_h + GAP)
        thumb = _fit_into(img, A4P_W - 2 * MARGIN, chain_img_h)
        x_img = MARGIN + (A4P_W - 2 * MARGIN - thumb.width) // 2
        page.paste(thumb, (x_img, y0))
        draw.text((MARGIN, y0 + chain_img_h + 3), name, fill=(50, 50, 50), font=font)
    return page


def build_cell_list(
    original: Image.Image,
    styled_results: list[tuple[str, Image.Image]],
) -> list[tuple[str, Image.Image]]:
    """Return the ordered cell list: original first, then styled results."""
    return [("Original", original)] + list(styled_results)
