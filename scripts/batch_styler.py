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

# ── PDF layout at 150 DPI (DIN A4 landscape = 297 × 210 mm) ──────────────────────
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

# ── PDF layout at 150 DPI (DIN A4 portrait = 210 × 297 mm) ─────────────────────
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
# Command: --style-overview
# ---------------------------------------------------------------------------

def cmd_style_overview(
    image_path: Path,
    styles: list[dict],
    tile_size: int,
    overlap: int,
    strength: float,  # noqa: ARG001 — ignored; style_overview uses PDF_STRENGTHS
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
            engine.unload_model(style_id)

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
    pdf_path = dir_out / (image_path.stem + "_style_overview.pdf")
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
# Command: --apply-style-chain
# ---------------------------------------------------------------------------

def _apply_chain_to_image(
    source: Image.Image,
    chain,  # ReplayLog
    styles: list[dict],
    engine: StyleTransferEngine,
    tile_size: int,
    overlap: int,
    use_float16: bool,
    strength_scale: int | None,
) -> Image.Image:
    """Apply all steps of *chain* to *source* and return the final result."""
    result = source.copy()
    for step in chain.steps:
        matched = filter_styles_by_name(styles, step.style)
        catalog_style = matched[0]
        model_path = REPO_ROOT / catalog_style["model_path"]
        if strength_scale is not None:
            effective_pct = min(300, round(step.strength * strength_scale / 100))
        else:
            effective_pct = step.strength
        strength = effective_pct / 100.0
        tensor_layout: str = catalog_style.get("tensor_layout", "nchw")
        engine.load_model(catalog_style["id"], model_path, tensor_layout=tensor_layout)
        result = engine.apply(
            result, catalog_style["id"],
            strength=strength, tile_size=tile_size, overlap=overlap, use_float16=use_float16,
        )
        engine.unload_model(catalog_style["id"])
    return result


def cmd_apply_style_chain(
    image_path: Path,
    replay_path: Path,
    tile_size: int | None,
    overlap: int | None,
    use_float16: bool,
    strength_scale: int | None = None,
    out_dir: Path | None = None,
) -> None:
    """Apply a saved style-chain YAML to *image_path*, step by step."""
    from src.core.style_chain_schema import load_style_chain  # noqa: PLC0415

    try:
        replay = load_style_chain(replay_path)
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

    # Pre-flight: verify all style names before starting inference
    unknown = []
    for step in replay.steps:
        needle = step.style.strip().casefold()
        if not any(s.get("name", "").casefold() == needle for s in styles):
            unknown.append(step.style)
    if unknown:
        sys.exit("Error: the following style(s) were not found in the catalog:\n" +
                 "\n".join(f"  - {n}" for n in unknown))

    print(f"Tile size    : {effective_tile_size} px  overlap: {effective_overlap} px")
    print()

    engine = StyleTransferEngine()
    source = Image.open(image_path).convert("RGB")

    print(f"Applying {len(replay.steps)} step(s) ...", flush=True)
    result = _apply_chain_to_image(
        source, replay, styles, engine,
        tile_size=effective_tile_size,
        overlap=effective_overlap,
        use_float16=use_float16,
        strength_scale=strength_scale,
    )

    dir_out = out_dir if out_dir is not None else image_path.parent
    if strength_scale is not None:
        fname = f"{image_path.stem}_{replay_path.stem}_{strength_scale}.jpg"
    else:
        fname = f"{image_path.stem}_{replay_path.stem}.jpg"
    out_path = dir_out / fname
    result.save(out_path, format="JPEG", quality=92)
    print(f"\nOK  Result written: {out_path}")


# ---------------------------------------------------------------------------
# Command: --style-chain-overview
# ---------------------------------------------------------------------------

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


def cmd_style_chain_overview(
    image_path: Path,
    chain_dir: Path,
    tile_size: int | None,
    overlap: int | None,
    use_float16: bool,
    strength_scale: int | None = None,
    out_dir: Path | None = None,
) -> None:
    """Apply all .yml chains in *chain_dir* and write a portrait A4 PDF overview."""
    from src.core.style_chain_schema import load_style_chain  # noqa: PLC0415

    chain_files = sorted(chain_dir.glob("*.yml")) + sorted(chain_dir.glob("*.yaml"))
    chain_files = sorted(set(chain_files))
    if not chain_files:
        sys.exit(f"Error: no .yml/.yaml files found in {chain_dir}")

    catalog_path = REPO_ROOT / "styles" / "catalog.json"
    if not catalog_path.exists():
        sys.exit(f"Error: catalog not found: {catalog_path}")
    with open(catalog_path, encoding="utf-8") as f:
        catalog: dict = json.load(f)
    styles: list[dict] = catalog.get("styles", [])

    effective_tile_size: int = tile_size if tile_size is not None else 1024
    effective_overlap: int = overlap if overlap is not None else 128

    source = Image.open(image_path).convert("RGB")
    engine = StyleTransferEngine()

    # Cells: (label, image) pairs — original first, then one per chain
    cells: list[tuple[str, Image.Image]] = [("Original", source.copy())]

    for chain_file in chain_files:
        try:
            chain = load_style_chain(chain_file)
        except ValueError as exc:
            print(f"  Warning: skipping '{chain_file.name}' — invalid schema: {exc}")
            continue
        # Pre-flight check
        unknown = []
        for step in chain.steps:
            needle = step.style.strip().casefold()
            if not any(s.get("name", "").casefold() == needle for s in styles):
                unknown.append(step.style)
        if unknown:
            print(f"  Warning: skipping '{chain_file.name}' — unknown style(s): {', '.join(unknown)}")
            continue
        print(f"Applying chain '{chain_file.stem}' ...", flush=True)
        try:
            result = _apply_chain_to_image(
                source, chain, styles, engine,
                tile_size=effective_tile_size,
                overlap=effective_overlap,
                use_float16=use_float16,
                strength_scale=strength_scale,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: skipping '{chain_file.name}' — error during apply: {exc}")
            continue
        cells.append((chain_file.stem, result))

    if len(cells) <= 1:
        sys.exit("No chains were applied successfully — nothing to write.")

    font = _load_font(int(LABEL_H * 0.60))
    pages: list[Image.Image] = []
    for i in range(0, len(cells), CHAIN_ROWS):
        pages.append(_make_chain_page(cells[i : i + CHAIN_ROWS], font))

    dir_out = out_dir if out_dir is not None else image_path.parent
    pdf_path = dir_out / f"{image_path.stem}_{chain_dir.name}_overview.pdf"
    pages[0].save(
        pdf_path,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=DPI,
    )
    print(f"\nOK  PDF written: {pdf_path}  ({len(pages)} page(s), {len(cells) - 1} chain(s))")
# ---------------------------------------------------------------------------
# Main / argument parsing
# ---------------------------------------------------------------------------

_USAGE = (
    "Usage:\n"
    "  BatchStyler.exe --style-overview <image>  [options]\n"
    "  BatchStyler.exe --apply-style-chain <chain.yml> <image>  [options]\n"
    "  BatchStyler.exe --style-chain-overview <chain-dir> <image>  [options]\n"
    "\n"
    "Modes (exactly one required):\n"
    "  --style-overview       Create a DIN-A4 landscape PDF contact sheet with all styles.\n"
    "                         Each style gets 3 cells: 100 %, 150 %, 200 % strength.\n"
    "                         Output: <image-dir>/<stem>_style_overview.pdf\n"
    "  --apply-style-chain FILE  Apply a saved style-chain YAML to the image.\n"
    "                         Output: <image-dir>/<stem>_<chain-stem>.jpg\n"
    "  --style-chain-overview CHAIN_DIR\n"
    "                         Apply all .yml chains in CHAIN_DIR and produce a portrait A4 PDF.\n"
    "                         Output: <image-dir>/<stem>_<chain-dir-name>_overview.pdf\n"
    "\n"
    "Options for --style-overview:\n"
    "  --apply-style NAME     Apply only the named style (case-insensitive).\n"
    "\n"
    "Options for --apply-style-chain and --style-chain-overview:\n"
    "  --strength-scale N     Scale each step's strength by N% (1\u2013300). Capped at 300%.\n"
    "                         E.g. --strength-scale 50 turns 100%\u219250%, 200%\u2192100%.\n"
    "\n"
    "Common options:\n"
    "  --tile-size N  Tile size for ONNX inference in pixels (default: 1024)\n"
    "  --overlap N    Tile overlap in pixels (default: 128)\n"
    "  --float16      Enable float16 inference (faster on GPU/DML)\n"
    "  --outdir DIR   Write output file(s) to DIR instead of the source image folder.\n"
    "                 DIR must already exist.\n"
    "\n"
    "Available styles:\n"
    + _list_styles_for_help()
    + "\n"
    "\n"
    "Examples:\n"
    "  BatchStyler.exe --style-overview portrait.jpg\n"
    "  BatchStyler.exe --style-overview portrait.jpg --apply-style \"Candy\"\n"
    "  BatchStyler.exe --apply-style-chain my_chain.yml portrait.jpg\n"
    "  BatchStyler.exe --apply-style-chain my_chain.yml portrait.jpg --strength-scale 80\n"
    "  BatchStyler.exe --apply-style-chain my_chain.yml portrait.jpg --outdir C:\\\\output\n"
    "  BatchStyler.exe --style-chain-overview C:\\\\chains portrait.jpg\n"
)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch style transfer.",
        add_help=True,
    )
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--style-overview", action="store_true", dest="style_overview",
        help="Create a PDF contact sheet with all styles.",
    )
    mode_group.add_argument(
        "--apply-style-chain", type=Path, metavar="CHAIN", dest="apply_style_chain",
        help="Apply a saved style-chain YAML to the image.",
    )
    mode_group.add_argument(
        "--style-chain-overview", type=Path, metavar="CHAIN_DIR", dest="style_chain_overview",
        help="Apply all .yml chains in CHAIN_DIR and produce a portrait A4 PDF.",
    )
    parser.add_argument("image", type=Path, help="Source image file (JPEG or PNG)")
    parser.add_argument(
        "--tile-size", type=int, default=None,
        help="Tile size for inference in pixels. Default: use YAML value or 1024.",
    )
    parser.add_argument(
        "--overlap", type=int, default=None,
        help="Tile overlap in pixels. Default: use YAML value or 128.",
    )
    parser.add_argument(
        "--strength-scale", type=int, default=None, metavar="PCT", dest="strength_scale",
        help="Scale all chain step strengths by this percentage (1\u20133300). Capped at 300%%.",
    )
    parser.add_argument(
        "--float16", action="store_true", default=False,
        help="Use float16 inference (faster on GPU/DML)",
    )
    parser.add_argument(
        "--apply-style", type=str, default=None, metavar="NAME", dest="apply_style",
        help="Apply only this style (case-insensitive name). Only for --style-overview.",
    )
    parser.add_argument(
        "--outdir", type=Path, default=None, metavar="DIR",
        help="Write output file(s) to DIR instead of the source image folder. DIR must already exist.",
    )
    args = parser.parse_args()

    if not args.style_overview and not args.apply_style_chain and not args.style_chain_overview:
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

    # Validate --strength-scale range
    if args.strength_scale is not None and not (1 <= args.strength_scale <= 300):
        sys.exit("Error: --strength-scale must be between 1 and 300.")

    # --apply-style is only valid with --style-overview
    if args.apply_style and not args.style_overview:
        sys.exit("Error: --apply-style can only be used with --style-overview.")

    if args.apply_style_chain:
        cmd_apply_style_chain(
            image_path, args.apply_style_chain.resolve(),
            tile_size=args.tile_size,
            overlap=args.overlap,
            use_float16=args.float16,
            strength_scale=args.strength_scale,
            out_dir=out_dir,
        )
        return

    if args.style_chain_overview:
        chain_dir = args.style_chain_overview.resolve()
        if not chain_dir.is_dir():
            sys.exit(f"Error: chain directory does not exist: {chain_dir}")
        cmd_style_chain_overview(
            image_path, chain_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            use_float16=args.float16,
            strength_scale=args.strength_scale,
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

    if args.apply_style:
        styles = filter_styles_by_name(styles, args.apply_style)

    pdf_tile_size: int = args.tile_size if args.tile_size is not None else 1024
    pdf_overlap: int = args.overlap if args.overlap is not None else 128

    print(f"Source image : {image_path}")
    print(f"Styles       : {len(styles)} style(s)" + (f" (filtered: '{args.apply_style}')" if args.apply_style else ""))
    print(f"Tile size    : {pdf_tile_size} px  overlap: {pdf_overlap} px")
    print()

    cmd_style_overview(
        image_path, styles,
        tile_size=pdf_tile_size,
        overlap=pdf_overlap,
        strength=1.0,
        use_float16=args.float16,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
