"""Backend logic for training/add_style_chain.ipynb.

Provides all non-UI functions so the notebook remains a thin cockpit.

Functions
---------
setup()
    Locate repo root, assert catalog + content image exist, load catalogs.
validate_chain_styles(chain, styles_catalog)
    Return a list of style names referenced by *chain* that are absent from
    the styles catalog.  An empty list means the chain is valid.
install_chain(...)
    Write the chain YAML + preview, and append the entry to
    ``style_chains/catalog.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

class ChainCatalogContext(NamedTuple):
    repo_root: Path
    styles_dir: Path
    chains_dir: Path
    styles_catalog_path: Path
    chains_catalog_path: Path
    content_image: Path
    styles_catalog: dict
    chains_catalog: dict
    existing_chain_ids: set[str]


def setup(repo_root: Path | None = None) -> ChainCatalogContext:
    """Locate the repo, assert required files exist, and load both catalogs.

    Args:
        repo_root: Explicit repo root.  When *None* the parent of the
                   ``training/`` folder is used.

    Returns:
        :class:`ChainCatalogContext` named-tuple with all resolved paths
        plus the parsed catalog dicts and the set of existing chain IDs.
    """
    if repo_root is None:
        _here = Path(__file__).resolve().parent
        repo_root = _here.parent

    styles_dir = repo_root / "styles"
    chains_dir = repo_root / "style_chains"
    styles_catalog_path = styles_dir / "catalog.json"
    chains_catalog_path = chains_dir / "catalog.json"
    content_image = repo_root / "sample_images" / "arch.png"

    assert styles_catalog_path.exists(), f"Styles catalog not found: {styles_catalog_path}"
    assert content_image.exists(), f"Content image not found: {content_image}"

    with open(styles_catalog_path, encoding="utf-8") as fh:
        styles_catalog: dict = json.load(fh)

    if chains_catalog_path.exists():
        with open(chains_catalog_path, encoding="utf-8") as fh:
            chains_catalog: dict = json.load(fh)
    else:
        chains_catalog = {"chains": []}

    existing_chain_ids: set[str] = {c["id"] for c in chains_catalog.get("chains", [])}

    print(f"Repository        : {repo_root}")
    print(f"Existing styles   : {sorted(s['name'] for s in styles_catalog.get('styles', []))}")
    print(f"Existing chains   : {sorted(existing_chain_ids)}")

    return ChainCatalogContext(
        repo_root=repo_root,
        styles_dir=styles_dir,
        chains_dir=chains_dir,
        styles_catalog_path=styles_catalog_path,
        chains_catalog_path=chains_catalog_path,
        content_image=content_image,
        styles_catalog=styles_catalog,
        chains_catalog=chains_catalog,
        existing_chain_ids=existing_chain_ids,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_chain_styles(
    steps: list[dict],
    styles_catalog: dict,
) -> list[str]:
    """Return style names in *steps* that are absent from the styles catalog.

    Args:
        steps:          List of ``{"style": str, "strength": int}`` dicts
                        (the ``steps`` section of a chain YAML / the author's
                        draft list).
        styles_catalog: Parsed ``styles/catalog.json`` dict.

    Returns:
        List of unknown style names.  Empty list means the chain is valid.
    """
    known_names: set[str] = {s["name"] for s in styles_catalog.get("styles", [])}
    return [
        step["style"]
        for step in steps
        if step.get("style") not in known_names
    ]


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def install_chain(
    *,
    steps: list[dict],
    chain_name: str,
    chain_desc: str,
    chain_tags: list[str],
    chains_dir: Path,
    chains_catalog_path: Path,
    chains_catalog: dict,
    existing_chain_ids: set[str],
    content_image: Path,
    repo_root: Path,
    preview_size: int = 256,
) -> str:
    """Apply the chain to *content_image*, write files, and update the catalog.

    Args:
        steps:               List of ``{"style": str, "strength": int}`` dicts.
        chain_name:          Human-readable display name (e.g. ``"Pastel"``).
        chain_desc:          Short description for the gallery card.
        chain_tags:          List of tag strings (may be empty).
        chains_dir:          Repo ``style_chains/`` directory.
        chains_catalog_path: Path to ``style_chains/catalog.json``.
        chains_catalog:      Already-parsed catalog dict (will be mutated and saved).
        existing_chain_ids:  Set of IDs already in the chain catalog (for pre-check).
        content_image:       Content photo used to generate the preview thumbnail.
        repo_root:           Repo root (added to ``sys.path`` so ``src/`` is importable).
        preview_size:        Edge length of the preview thumbnail in pixels.

    Returns:
        The new ``chain_id`` string.

    Raises:
        AssertionError: On empty name, duplicate ID, or preview generation failure.
    """
    import shutil

    chain_name = chain_name.strip()
    assert chain_name, "Chain name is empty — fill in the name first."
    chain_id = chain_name.strip().lower().replace(" ", "-")
    assert chain_id not in existing_chain_ids, (
        f"'{chain_id}' already exists in the chain catalog."
    )
    assert steps, "No steps defined — add at least one style step."

    dest_dir = chains_dir / chain_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # -- Write chain.yml
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.core.style_chain_schema import dump_style_chain, StyleChain, ChainStep  # noqa: PLC0415

    chain_obj = StyleChain(
        steps=[ChainStep(style=s["style"], strength=s["strength"]) for s in steps]
    )
    yml_path = dest_dir / "chain.yml"
    yml_path.write_text(dump_style_chain(chain_obj), encoding="utf-8")
    print(f"Chain YAML written : {yml_path}")

    # -- Generate preview by applying the chain to the content image
    from src.core.engine import StyleTransferEngine          # noqa: PLC0415
    from src.core.registry import StyleRegistry               # noqa: PLC0415
    from src.core.style_chain_schema import load_style_chain  # noqa: PLC0415
    from PIL import Image                                       # noqa: PLC0415

    styles_catalog_path = repo_root / "styles" / "catalog.json"
    registry = StyleRegistry(catalog_path=styles_catalog_path)
    engine = StyleTransferEngine()

    content_img = Image.open(content_image).convert("RGB")
    # Resize so the longer side == preview_size (keeps aspect ratio)
    w, h = content_img.size
    scale = preview_size / max(w, h)
    content_img = content_img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
    )

    current = content_img
    for i, step in enumerate(steps):
        style = registry.find_by_name(step["style"])
        assert style is not None, f"Style not found: {step['style']}"
        style_id = style.id
        if not engine.is_loaded(style_id):
            engine.load_model(
                style_id,
                style.model_path_resolved(repo_root),
                tensor_layout=style.tensor_layout,
            )
        current = engine.apply(current, style_id, strength=step["strength"] / 100.0)
        assert current is not None, f"Style transfer returned None at step {i + 1}"

    preview_path = dest_dir / "preview.jpg"
    current.save(preview_path, "JPEG", quality=85)
    print(f"Preview generated  : {preview_path}")

    # -- Update catalog.json
    rel_chain = f"style_chains/{chain_id}/chain.yml"
    rel_preview = f"style_chains/{chain_id}/preview.jpg"
    chains_catalog["chains"].append({
        "id":           chain_id,
        "name":         chain_name,
        "description":  chain_desc,
        "chain_path":   rel_chain,
        "preview_path": rel_preview,
        "step_count":   len(steps),
        "tags":         chain_tags,
    })
    chains_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chains_catalog_path, "w", encoding="utf-8") as fh:
        json.dump(chains_catalog, fh, indent=2)

    print(f"\nOK  '{chain_name}'  (id='{chain_id}',  {len(steps)} step(s))  added to chain gallery.")
    print(f"Files:")
    for p in sorted(dest_dir.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1e3:.0f} kB)")
    print("\nNext: git add -A ; git commit -m 'feat: add <name> style chain'")

    return chain_id
