"""Style Trainer CLI — full entry-point logic.

Usage::

    python -m src.trainer.app train \\
        --style  my_style.jpg \\
        --coco   data/train2017 \\
        --out    styles/my_style \\
        --id     my_style \\
        --name   "My Style"

    python -m src.trainer.app preview \\
        --model  styles/my_style/model.onnx \\
        --content sample_images/Ball-4MP.jpg \\
        --out    styles/my_style/preview.jpg

All imports come from ``src/trainer`` and ``src/core`` only.
The Stylist Qt app (``src/stylist/``) is never imported here.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    """Train a new TransformerNet and export it to ONNX."""
    from src.trainer.style_trainer import StyleTrainer  # noqa: PLC0415

    style_paths = [Path(p) for p in (args.style if isinstance(args.style, list) else [args.style])]
    coco_path = Path(args.coco)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pth_path = out_dir / "model.pth"
    onnx_path = out_dir / "model.onnx"
    preview_path = out_dir / "preview.jpg"

    def _progress(done: int, total: int, loss: float) -> None:
        pct = 100.0 * done / max(total, 1)
        print(f"\r  {pct:6.2f}%  images={done}/{total}  loss={loss:.4f}", end="", flush=True)

    trainer = StyleTrainer(device=args.device)

    logger.info("Training: style=%s  coco=%s  out=%s", style_paths, coco_path, pth_path)
    trainer.train(
        style_images=style_paths,
        coco_dataset_path=coco_path,
        output_model_path=pth_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        tv_weight=getattr(args, "tv_weight", 0.0),
        progress_callback=_progress,
        max_batches=args.max_batches,
    )
    print()  # newline after progress

    logger.info("Exporting ONNX…")
    trainer.export_onnx(pth_path, onnx_path)

    if args.content and Path(args.content).exists():
        from src.trainer.preview import generate_preview  # noqa: PLC0415
        generate_preview(onnx_path, preview_path, Path(args.content))
        logger.info("Preview saved to %s", preview_path)

    if args.id and args.name:
        from src.core.models import StyleModel  # noqa: PLC0415
        from src.core.registry import StyleRegistry  # noqa: PLC0415

        catalog = out_dir.parent / "catalog.json"
        if catalog.exists():
            registry = StyleRegistry(catalog_path=catalog)
            model = StyleModel(
                id=args.id,
                name=args.name,
                model_path=str(onnx_path.relative_to(catalog.parent)),
                preview_path=str(preview_path.relative_to(catalog.parent))
                if preview_path.exists() else "",
                is_builtin=False,
            )
            try:
                registry.add(model)
                logger.info("Registered style '%s' in %s", args.name, catalog)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not register style: %s", exc)

    logger.info("Done.")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    """Generate a preview thumbnail from an existing ONNX model."""
    from src.trainer.preview import generate_preview  # noqa: PLC0415

    generate_preview(
        onnx_path=Path(args.model),
        preview_path=Path(args.out),
        content_image=Path(args.content),
        size=args.size,
    )
    logger.info("Preview saved to %s", args.out)
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainer",
        description="Style Trainer -- developer CLI for training new ONNX style models",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train --
    p_train = sub.add_parser("train", help="Train a new style model")
    p_train.add_argument("--style",          required=True,  nargs="+",
                         help="Style reference image(s) — multiple paths for mean-Gram training")
    p_train.add_argument("--coco",           required=True,  help="MS-COCO dataset root")
    p_train.add_argument("--out",            required=True,  help="Output directory (e.g. styles/my_style)")
    p_train.add_argument("--id",             default="",     help="Style ID for catalog registration")
    p_train.add_argument("--name",           default="",     help="Style display name for catalog registration")
    p_train.add_argument("--content",        default="",     help="Content image for preview generation")
    p_train.add_argument("--device",         default="auto", help="Training device: auto | cpu | cuda")
    p_train.add_argument("--epochs",         type=int,   default=2)
    p_train.add_argument("--batch-size",     type=int,   default=4, dest="batch_size")
    p_train.add_argument("--image-size",     type=int,   default=256, dest="image_size")
    p_train.add_argument("--style-weight",   type=float, default=1e8, dest="style_weight")
    p_train.add_argument("--content-weight", type=float, default=1e5, dest="content_weight")
    p_train.add_argument("--tv-weight",      type=float, default=0.0, dest="tv_weight",
                         help="Total Variation loss weight (default 0.0 = off for CLI; 1e-6 recommended)")
    p_train.add_argument("--max-batches",    type=int,   default=None, dest="max_batches",
                         help="Stop after N gradient steps (smoke-test mode; default: run fully)")

    # -- preview --
    p_prev = sub.add_parser("preview", help="Generate a preview thumbnail from an ONNX model")
    p_prev.add_argument("--model",   required=True, help="Path to .onnx model file")
    p_prev.add_argument("--content", required=True, help="Content image path")
    p_prev.add_argument("--out",     required=True, help="Output preview.jpg path")
    p_prev.add_argument("--size",    type=int, default=256, help="Preview size in pixels")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "preview":
        return cmd_preview(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
