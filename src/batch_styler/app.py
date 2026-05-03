"""BatchStyler CLI entry point.

Usage::

    python -m src.batch_styler.app --style-overview path/to/photo.jpg
    python -m src.batch_styler.app --apply-style-chain my_chain.yml path/to/photo.jpg
    python -m src.batch_styler.app --style-chain-overview chains/ path/to/photo.jpg

When compiled with PyInstaller the entry point is ``BatchStyler.exe``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Repository root (must come before any src.* import) ─────────────────────
# When compiled with PyInstaller, sys.executable points to BatchStyler.exe.
# In dev mode, __file__ is src/batch_styler/app.py → up 3 levels = repo root.
if not getattr(sys, "frozen", False):
    _repo = Path(__file__).resolve().parent.parent.parent
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))

import src.batch_styler.catalog as _catalog  # noqa: E402
from src.batch_styler.commands import (  # noqa: E402
    cmd_apply_style_chain,
    cmd_style_chain_overview,
    cmd_style_overview,
)
from src.core.registry import StyleRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# CLI help / usage string
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
    + _catalog._list_styles_for_help()
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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

    out_dir: Path | None = None
    if args.outdir is not None:
        out_dir = args.outdir.resolve()
        if not out_dir.is_dir():
            sys.exit(f"Error: --outdir directory does not exist: {out_dir}")

    if args.strength_scale is not None and not (1 <= args.strength_scale <= 300):
        sys.exit("Error: --strength-scale must be between 1 and 300.")

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

    catalog_path = _catalog.REPO_ROOT / "styles" / "catalog.json"
    if not catalog_path.exists():
        sys.exit(f"Error: catalog not found: {catalog_path}")

    registry = StyleRegistry(catalog_path)
    styles = registry.list_styles()
    if not styles:
        sys.exit("No styles found in catalog.")

    if args.apply_style:
        matched = registry.find_by_name(args.apply_style)
        if matched is None:
            available = ", ".join(f"'{s.name}'" for s in registry.list_styles())
            sys.exit(
                f"Error: style '{args.apply_style}' not found in catalog.\n"
                f"Available styles: {available}"
            )
        styles = [matched]

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
