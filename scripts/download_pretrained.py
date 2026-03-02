"""Download pretrained ONNX models from GitHub Releases.

Usage
-----
    python scripts/download_pretrained.py [--styles candy mosaic ...]

All models are saved to the ``styles/`` directory next to this script.
Run with ``--help`` for full options.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry: map style_id → (download_url, sha256_hex)
# Update these URLs once models are attached to GitHub Releases.
# ---------------------------------------------------------------------------
RELEASE_BASE = (
    "https://github.com/PeterWazinski/style_transfer/releases/download/v1.0.0"
)

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "candy":         (f"{RELEASE_BASE}/candy.onnx",         ""),
    "mosaic":        (f"{RELEASE_BASE}/mosaic.onnx",        ""),
    "rain_princess": (f"{RELEASE_BASE}/rain_princess.onnx", ""),
    "udnie":         (f"{RELEASE_BASE}/udnie.onnx",         ""),
    "starry_night":  (f"{RELEASE_BASE}/starry_night.onnx",  ""),
    "abstract":      (f"{RELEASE_BASE}/abstract.onnx",      ""),
}

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
STYLES_DIR: Path = PROJECT_ROOT / "styles"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(style_id: str, *, force: bool = False) -> None:
    if style_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown style '{style_id}'. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )

    url, expected_sha256 = MODEL_REGISTRY[style_id]
    dest: Path = STYLES_DIR / style_id / "model.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        if expected_sha256 and _sha256(dest) == expected_sha256:
            print(f"[{style_id}] Already downloaded and verified — skipping.")
            return
        print(f"[{style_id}] File exists but hash mismatch or unverified — re-downloading.")

    print(f"[{style_id}] Downloading from {url} …")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            bar = "#" * (pct // 5) + " " * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress bar
    except Exception as exc:
        print(f"\n  ERROR: {exc}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        raise

    if expected_sha256:
        actual = _sha256(dest)
        if actual != expected_sha256:
            dest.unlink()
            raise RuntimeError(
                f"[{style_id}] Hash mismatch!\n"
                f"  expected: {expected_sha256}\n"
                f"  got:      {actual}"
            )
        print(f"[{style_id}] SHA-256 verified ✓")

    print(f"[{style_id}] Saved to {dest}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download pretrained ONNX style models."
    )
    parser.add_argument(
        "--styles",
        nargs="*",
        default=list(MODEL_REGISTRY.keys()),
        metavar="STYLE",
        help="Which styles to download (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists and matches.",
    )
    args = parser.parse_args(argv)

    failed: list[str] = []
    for style_id in args.styles:
        try:
            download_model(style_id, force=args.force)
        except Exception as exc:
            print(f"[{style_id}] FAILED: {exc}", file=sys.stderr)
            failed.append(style_id)

    if failed:
        print(f"\nFailed to download: {failed}", file=sys.stderr)
        sys.exit(1)
    print("\nAll models downloaded successfully.")


if __name__ == "__main__":
    main()
