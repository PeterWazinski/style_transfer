"""Batch style-transfer command implementations.

All commands access REPO_ROOT via ``_catalog.REPO_ROOT`` (module-attribute
lookup) so that a single ``patch("src.batch_styler.catalog.REPO_ROOT", ...)``
in tests covers this module as well.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from PIL import Image

import src.batch_styler.catalog as _catalog
from src.batch_styler.pdf_layout import (
    CELLS_PER_PAGE,
    DPI,
    LABEL_H,
    PDF_STRENGTHS,
    _blend_to_strength,
    _fit_into,
    _load_font,
    _make_page,
    build_cell_list,
)
from src.core.engine import StyleTransferEngine
from src.core.models import StyleModel
from src.core.registry import StyleRegistry

logging.basicConfig(level=logging.WARNING)

# JPEG quality used when saving styled output images.
JPEG_QUALITY: int = 92


# ---------------------------------------------------------------------------
# Command: --style-overview
# ---------------------------------------------------------------------------

def cmd_style_overview(
    image_path: Path,
    styles: list[StyleModel],
    tile_size: int,
    overlap: int,
    strength: float,  # noqa: ARG001 — ignored; style_overview uses PDF_STRENGTHS
    use_float16: bool,
    out_dir: Path | None = None,
) -> None:
    """Apply all styles at each PDF_STRENGTHS level and write a DIN-A4-landscape PDF."""
    source = Image.open(image_path).convert("RGB")
    engine = StyleTransferEngine()

    styles_sorted = sorted(styles, key=lambda s: s.name.casefold())
    total = len(styles_sorted)

    cells: list[tuple[str, Image.Image | None]] = [
        ("Original", source.copy()),
        ("", None),
        ("", None),
    ]
    n_applied: int = 0

    for idx, style in enumerate(styles_sorted, 1):
        model_path: Path = style.model_path_resolved(_catalog.REPO_ROOT)

        if not model_path.exists():
            print(f"({idx}/{total}) Skipping '{style.name}' — model not found: {model_path}")
            continue

        print(f"({idx}/{total}) Processing style '{style.name}' ...", end="", flush=True)
        t0 = time.monotonic()
        try:
            engine.load_model(style.id, model_path, tensor_layout=style.tensor_layout)
            styled_full = engine.apply(
                source,
                style.id,
                strength=1.0,
                tile_size=tile_size,
                overlap=overlap,
                use_float16=use_float16,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n  Error applying '{style.name}': {exc}")
            continue
        finally:
            engine.unload_model(style.id)

        elapsed = round(time.monotonic() - t0)
        print(f"\r({idx}/{total}) Processing style '{style.name}' in {elapsed} seconds.")

        for s in PDF_STRENGTHS:
            label = f"{style.name} ({int(s * 100)}%)"
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

    print(f"PDF written: {pdf_path}")


# ---------------------------------------------------------------------------
# Command: --apply-style-chain
# ---------------------------------------------------------------------------

def _apply_chain_to_image(
    source: Image.Image,
    chain,  # StyleChain
    registry: StyleRegistry,
    engine: StyleTransferEngine,
    tile_size: int,
    overlap: int,
    use_float16: bool,
    strength_scale: int | None,
) -> Image.Image:
    """Apply all steps of *chain* to *source* and return the final result."""
    result = source.copy()
    for step in chain.steps:
        style_model = registry.find_by_name(step.style)
        if style_model is None:
            sys.exit(f"Error: style '{step.style}' not found in catalog.")
        model_path = style_model.model_path_resolved(_catalog.REPO_ROOT)
        if strength_scale is not None:
            effective_pct = min(300, round(step.strength * strength_scale / 100))
        else:
            effective_pct = step.strength
        strength = effective_pct / 100.0
        engine.load_model(style_model.id, model_path, tensor_layout=style_model.tensor_layout)
        result = engine.apply(
            result, style_model.id,
            strength=strength, tile_size=tile_size, overlap=overlap, use_float16=use_float16,
        )
        engine.unload_model(style_model.id)
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

    effective_tile_size: int = tile_size if tile_size is not None else (replay.tile_size if replay.tile_size is not None else 1024)
    effective_overlap: int = overlap if overlap is not None else (replay.tile_overlap if replay.tile_overlap is not None else 128)

    catalog_path = _catalog.REPO_ROOT / "styles" / "catalog.json"
    if not catalog_path.exists():
        sys.exit(f"Error: catalog not found: {catalog_path}")

    registry = StyleRegistry(catalog_path)
    unknown = [step.style for step in replay.steps if registry.find_by_name(step.style) is None]
    if unknown:
        sys.exit("Error: the following style(s) were not found in the catalog:\n" +
                 "\n".join(f"  - {n}" for n in unknown))

    print(f"Tile size    : {effective_tile_size} px  overlap: {effective_overlap} px")
    print()

    engine = StyleTransferEngine()
    source = Image.open(image_path).convert("RGB")

    print(f"Applying {len(replay.steps)} step(s) ...", flush=True)
    result = _apply_chain_to_image(
        source, replay, registry, engine,
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
    result.save(out_path, format="JPEG", quality=JPEG_QUALITY)
    print(f"\nOK  Result written: {out_path}")


# ---------------------------------------------------------------------------
# Command: --style-chain-overview
# ---------------------------------------------------------------------------

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

    catalog_path = _catalog.REPO_ROOT / "styles" / "catalog.json"
    if not catalog_path.exists():
        sys.exit(f"Error: catalog not found: {catalog_path}")
    registry = StyleRegistry(catalog_path)

    effective_tile_size: int = tile_size if tile_size is not None else 1024
    effective_overlap: int = overlap if overlap is not None else 128

    source = Image.open(image_path).convert("RGB")
    engine = StyleTransferEngine()

    cells: list[tuple[str, Image.Image]] = [("Original", source.copy())]
    total_chains = len(chain_files)

    for idx, chain_file in enumerate(chain_files, 1):
        try:
            chain = load_style_chain(chain_file)
        except ValueError as exc:
            print(f"({idx}/{total_chains}) Skipping '{chain_file.name}' — invalid schema: {exc}")
            continue
        unknown = [step.style for step in chain.steps if registry.find_by_name(step.style) is None]
        if unknown:
            print(f"({idx}/{total_chains}) Skipping '{chain_file.name}' — unknown style(s): {', '.join(unknown)}")
            continue
        print(f"({idx}/{total_chains}) Applying chain '{chain_file.stem}' ...", end="", flush=True)
        t0 = time.monotonic()
        try:
            result = _apply_chain_to_image(
                source, chain, registry, engine,
                tile_size=effective_tile_size,
                overlap=effective_overlap,
                use_float16=use_float16,
                strength_scale=strength_scale,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n  Warning: skipping '{chain_file.name}' — error during apply: {exc}")
            continue
        elapsed = round(time.monotonic() - t0)
        print(f"\r({idx}/{total_chains}) Applying chain '{chain_file.stem}' in {elapsed} seconds.")
        cells.append((chain_file.stem, result))

    if len(cells) <= 1:
        sys.exit("No chains were applied successfully — nothing to write.")

    font = _load_font(int(LABEL_H * 0.60))
    pages: list[Image.Image] = []
    for i in range(0, len(cells), CELLS_PER_PAGE):
        pages.append(_make_page(cells[i : i + CELLS_PER_PAGE], font))

    dir_out = out_dir if out_dir is not None else image_path.parent
    pdf_path = dir_out / f"{image_path.stem}_{chain_dir.name}_overview.pdf"
    pages[0].save(
        pdf_path,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=DPI,
    )
    print(f"PDF written: {pdf_path}")
