"""Benchmark ONNX inference latency for each available style model.

Usage
-----
    python scripts/benchmark.py [--style candy] [--tile-size 1024] [--runs 5]
                                [--float16]

Measures per-tile inference time and asserts it stays below 2 s on the
configured hardware (Intel Arc / DirectML, or CPU fallback).
Results are appended to ``benchmarks.log`` in the project root.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.engine import StyleModelNotFoundError, StyleTransferEngine  # noqa: E402

CATALOG_PATH: Path = PROJECT_ROOT / "styles" / "catalog.json"
BENCHMARK_LOG: Path = PROJECT_ROOT / "benchmarks.log"
MAX_LATENCY_SECONDS: float = 2.0


def _load_catalog() -> list[dict]:
    with open(CATALOG_PATH) as f:
        return json.load(f)["styles"]


def _synthetic_tile(tile_size: int) -> Image.Image:
    arr = np.random.randint(0, 255, (tile_size, tile_size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def benchmark_style(
    style_id: str,
    model_path: Path,
    tile_size: int,
    runs: int,
    *,
    use_float16: bool = False,
    assert_latency: bool = True,
) -> dict:
    engine = StyleTransferEngine()
    try:
        engine.load_model(style_id, model_path)
    except StyleModelNotFoundError:
        return {"style_id": style_id, "status": "SKIPPED (model not found)"}
    except ImportError as exc:
        return {"style_id": style_id, "status": f"SKIPPED ({exc})"}

    tile = _synthetic_tile(tile_size)
    latencies: list[float] = []

    # Warm-up run (not timed)
    engine.apply(
        tile, style_id, strength=1.0, tile_size=tile_size + 1,
        overlap=0, use_float16=use_float16,
    )

    for _ in range(runs):
        t0 = time.perf_counter()
        engine.apply(
            tile, style_id, strength=1.0, tile_size=tile_size + 1,
            overlap=0, use_float16=use_float16,
        )
        latencies.append(time.perf_counter() - t0)

    mean_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)

    status = "PASS" if mean_lat <= MAX_LATENCY_SECONDS else "FAIL"
    result = {
        "style_id":    style_id,
        "tile_size":   tile_size,
        "runs":        runs,
        "float16":     use_float16,
        "mean_s":      round(mean_lat, 3),
        "min_s":       round(min_lat, 3),
        "max_s":       round(max_lat, 3),
        "threshold_s": MAX_LATENCY_SECONDS,
        "status":      status,
    }

    if assert_latency and mean_lat > MAX_LATENCY_SECONDS:
        print(
            f"  [FAIL] {style_id}: mean latency {mean_lat:.3f}s "
            f"exceeds {MAX_LATENCY_SECONDS}s threshold",
            file=sys.stderr,
        )

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX inference latency per tile."
    )
    parser.add_argument("--style", default=None, help="Benchmark a single style ID.")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Cast input tiles to float16 before inference.",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="Report results without failing on slow models.",
    )
    args = parser.parse_args(argv)

    catalog = _load_catalog()
    if args.style:
        catalog = [s for s in catalog if s["id"] == args.style]
        if not catalog:
            print(f"Style '{args.style}' not found in catalog.", file=sys.stderr)
            sys.exit(1)

    print(f"\nBenchmark: tile_size={args.tile_size}px, {args.runs} runs each"
          f"{', float16' if args.float16 else ''}\n")
    print(f"{'Style':<20} {'Mean':>8} {'Min':>8} {'Max':>8}  Status")
    print("-" * 58)

    failures: list[str] = []
    for entry in catalog:
        style_id: str = entry["id"]
        model_path = PROJECT_ROOT / entry["model_path"]
        result = benchmark_style(
            style_id,
            model_path,
            args.tile_size,
            args.runs,
            use_float16=args.float16,
            assert_latency=not args.no_assert,
        )
        if result.get("status") in ("SKIPPED (model not found)", ) or result["status"].startswith("SKIPPED"):
            print(f"  {style_id:<20}  {result['status']}")
            continue
        mean = result["mean_s"]
        mn   = result["min_s"]
        mx   = result["max_s"]
        status = result["status"]
        flag = "✓" if status == "PASS" else "✗"
        print(f"  {style_id:<20} {mean:>7.3f}s {mn:>7.3f}s {mx:>7.3f}s  {flag} {status}")
        if status == "FAIL":
            failures.append(style_id)

    print()
    # --- Append to benchmarks.log ---
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    with open(BENCHMARK_LOG, "a", encoding="utf-8") as logf:
        logf.write(f"\n--- {timestamp}  tile={args.tile_size}  float16={args.float16} ---\n")
        for entry in catalog:
            sid = entry["id"]
            model_path = PROJECT_ROOT / entry["model_path"]
            r = benchmark_style(
                sid, model_path, args.tile_size, args.runs,
                use_float16=args.float16, assert_latency=False,
            )
            logf.write(json.dumps(r) + "\n")

    if failures and not args.no_assert:
        print(f"FAILED styles (latency > {MAX_LATENCY_SECONDS}s): {failures}", file=sys.stderr)
        sys.exit(1)
    print("Benchmark complete. Results appended to benchmarks.log.")


if __name__ == "__main__":
    main()
