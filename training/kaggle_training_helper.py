"""Kaggle training helper — backend logic extracted from kaggle_style_training.ipynb.

Provides:

* :class:`TrainingConfig` — typed dataclass for all training hyperparameters,
  with ``save()`` / ``load()`` JSON persistence so resume is automatic.
* :class:`KaggleStyleRunner` — encapsulates verify, analyse, smoke-test,
  full-train, resume, and package steps.
* CLI entry point — each step is a sub-command:

    python scripts/kaggle_training_helper.py verify
    python scripts/kaggle_training_helper.py analyse  --style /kaggle/working/my_style.jpg
    python scripts/kaggle_training_helper.py smoke    --style ... --id my_style --coco ...
    python scripts/kaggle_training_helper.py train    --style ... --id my_style --coco ...
    python scripts/kaggle_training_helper.py resume   --id my_style
    python scripts/kaggle_training_helper.py package  --id my_style

The helper is designed to run on Kaggle after ``git clone`` of the repo — it
imports from ``src/`` directly without an install step.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field

import numpy as np

# ── ensure repo root is on sys.path when executed directly ───────────────────
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.trainer.style_analyser import analyse_style, hist_overlap, recommend_weights  # noqa: E402


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All hyperparameters for one style-transfer training job."""

    style_images: list[pathlib.Path]
    style_id: str
    style_name: str
    coco_path: pathlib.Path
    style_weight: float = 1e10
    content_weight: float = 1e5
    tv_weight: float = 1e-6       # Total Variation loss weight; set 0.0 to disable
    epochs: int = 2
    batch_size: int = 4
    image_size: int = 256
    smoke_batches: int = 2000  # validated on T4 (P3-1: mean_diff=57 on candy)
    device: str = "cuda"

    # If set, auto-expands *.jpg/jpeg/png in this dir to style_images on post_init
    style_images_dir: pathlib.Path | None = None

    # populated by run_full_training(); used by resume_training()
    output_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("."))

    _CONFIG_FILE: str = field(default="config.json", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Expand style_images_dir to style_images list if provided."""
        if self.style_images_dir is not None and self.style_images_dir.is_dir():
            exts = {".jpg", ".jpeg", ".png"}
            found = sorted(
                p for p in self.style_images_dir.iterdir()
                if p.suffix.lower() in exts
            )
            if found:
                self.style_images = found

    def save(self, out_dir: pathlib.Path) -> None:
        """Serialise config to ``<out_dir>/config.json``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        # Convert Path objects and lists of Paths to strings for JSON
        for k, v in d.items():
            if isinstance(v, pathlib.Path):
                d[k] = str(v)
            elif isinstance(v, list):
                d[k] = [str(item) if isinstance(item, pathlib.Path) else item for item in v]
        (out_dir / "config.json").write_text(
            json.dumps(d, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, out_dir: pathlib.Path) -> "TrainingConfig":
        """Deserialise config from ``<out_dir>/config.json``."""
        cfg_path = out_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"No config.json found in {out_dir}")
        d = json.loads(cfg_path.read_text(encoding="utf-8"))
        # Remove internal fields not accepted by __init__
        d.pop("_CONFIG_FILE", None)
        path_fields = {"coco_path", "output_dir", "style_images_dir"}
        for k in path_fields:
            if k in d and d[k] is not None:
                d[k] = pathlib.Path(d[k])
        # style_images is a list of paths
        if "style_images" in d:
            d["style_images"] = [pathlib.Path(p) for p in d["style_images"]]
        # Back-compat: old configs may have style_image (singular)
        if "style_image" in d and "style_images" not in d:
            d["style_images"] = [pathlib.Path(d.pop("style_image"))]
        else:
            d.pop("style_image", None)
        return cls(**d)


# ---------------------------------------------------------------------------
# KaggleStyleRunner
# ---------------------------------------------------------------------------

class KaggleStyleRunner:
    """Orchestrates each phase of a Kaggle style-transfer training job."""

    _COCO_VAL_CONTENT = pathlib.Path(
        "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017/000000000139.jpg"
    )

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self._repo_dir = _REPO_ROOT

    # ── Step 0: environment checks ────────────────────────────────────────────

    def verify_environment(self) -> None:
        """Check GPU availability, COCO mount, and internet access."""
        import torch  # type: ignore[import-untyped]

        print("─── Environment verification ───")
        cuda_ok = torch.cuda.is_available()
        print(f"  CUDA available : {cuda_ok}")
        if cuda_ok:
            print(f"  GPU            : {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠  No GPU — training will be very slow on CPU.")

        imgs = list(self.cfg.coco_path.glob("*.jpg"))
        print(f"  COCO images    : {len(imgs):,}  (need > 1 000)")
        if len(imgs) < 1000:
            raise RuntimeError(
                f"Too few COCO images in {self.cfg.coco_path}.\n"
                "Add awsaf49/coco-2017-dataset in Kaggle notebook settings."
            )

        try:
            urllib.request.urlopen("https://api.github.com", timeout=5)
            print("  Internet       : ✓ available")
        except Exception:
            raise RuntimeError(
                "No internet access — VGG16 weights (~550 MB) cannot be downloaded.\n"
                "Enable: Notebook sidebar → Settings → Internet → On"
            )
        print("  All checks passed.")

    # ── Step 1: style analysis ────────────────────────────────────────────────

    def analyse_style(self) -> list[dict]:
        """Analyse all style images and print per-image metric table."""
        print("─── Style image analysis ───")
        n = len(self.cfg.style_images)
        if n == 1:
            print("  ⚠  Only 1 style image — consider using kaggle_trainer for single-image training.")
        print(f"  Images     : {n}")
        print()
        header = f"  {'#':>3}  {'File':<30}  {'flat%':>6}  {'p_std':>6}  {'edge':>6}  {'l_var':>6}  {'SW_rec':>8}  Verdict"
        print(header)
        print("  " + "─" * (len(header) - 2))

        results = []
        for i, img_path in enumerate(self.cfg.style_images, 1):
            m = analyse_style(img_path)
            sw_rec, cw_rec, verdict = recommend_weights(m)
            flag = "⚠" if "flat" in verdict.lower() or m["flat_pct"] > 60 else " "
            print(
                f"  {i:>3}  {img_path.name:<30}  "
                f"{m['flat_pct']:>6.1f}  {m['mean_patch_std']:>6.1f}  "
                f"{m['edge_density']:>6.1f}  {m['local_var']:>6.1f}  "
                f"{sw_rec:>8.0e}  {flag} {verdict}"
            )
            results.append({"path": img_path, "metrics": m, "sw_rec": sw_rec, "verdict": verdict})

        print()
        print(f"  Using fixed  →  STYLE_WEIGHT = {self.cfg.style_weight:.0e}   "
              f"CONTENT_WEIGHT = {self.cfg.content_weight:.0e}   "
              f"TV_WEIGHT = {self.cfg.tv_weight:.0e}")
        return results

    # ── Step 2: smoke test ────────────────────────────────────────────────────

    def run_smoke_test(self) -> dict:
        """Train for ``cfg.smoke_batches`` batches, export ONNX, score on content photo.

        Returns:
            ``{"mean_diff": float, "colour_shift": float, "verdict": str}``
        """
        from src.trainer.style_trainer import StyleTrainer  # noqa: PLC0415

        import onnxruntime as ort  # type: ignore[import-untyped]
        from PIL import Image  # noqa: PLC0415

        n = self.cfg.smoke_batches
        print(f"─── Smoke test ({n} batches, device={self.cfg.device}) ───")

        with tempfile.TemporaryDirectory() as _tmp:
            pth  = pathlib.Path(_tmp) / "smoke.pth"
            onnx_path = pathlib.Path(_tmp) / "smoke.onnx"

            _last_loss: list[float] = [0.0]

            def _cb(done: int, total: int, loss: float) -> None:
                _last_loss[0] = loss
                b = (done + self.cfg.batch_size - 1) // self.cfg.batch_size
                print(f"\r  batch {b:>4}/{n}  loss={loss:.4f}", end="", flush=True)

            trainer = StyleTrainer(device=self.cfg.device)
            trainer.train(
                style_images=self.cfg.style_images,
                coco_dataset_path=self.cfg.coco_path,
                output_model_path=pth,
                epochs=999,
                batch_size=self.cfg.batch_size,
                image_size=self.cfg.image_size,
                style_weight=self.cfg.style_weight,
                content_weight=self.cfg.content_weight,
                tv_weight=self.cfg.tv_weight,
                checkpoint_interval=0,
                max_batches=n,
                progress_callback=_cb,
            )
            print()
            trainer.export_onnx(pth, onnx_path)

            # Score on a neutral content photo (not the style image — see inline note)
            check = (
                self._COCO_VAL_CONTENT
                if self._COCO_VAL_CONTENT.exists()
                else self.cfg.style_image
            )
            img = Image.open(check).convert("RGB").resize((256, 256))
            arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis]
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.cfg.device == "cuda"
                else ["CPUExecutionProvider"]
            )
            sess = ort.InferenceSession(str(onnx_path), providers=providers)
            out = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
            out_img = np.clip(out[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)

            mean_diff = float(np.abs(arr[0].transpose(1, 2, 0) - out_img).mean())

            style_ref = np.array(
                Image.open(self.cfg.style_images[0]).convert("RGB").resize((256, 256)),
                dtype=np.float32,
            )
            orig_arr = arr[0].transpose(1, 2, 0)
            colour_shift = float(
                hist_overlap(out_img.astype(np.float32), style_ref)
                - hist_overlap(orig_arr, style_ref)
            )

            # Capture PIL images before the temp dir is cleaned up
            content_pil   = img
            styled_pil    = Image.fromarray(out_img)
            style_ref_pil = Image.fromarray(style_ref.astype(np.uint8))

        print(f"  Mean pixel change : {mean_diff:.1f}")
        print(f"  Colour shift      : {colour_shift:+.3f}")
        print(f"  Style images used : {len(self.cfg.style_images)}")
        if mean_diff < 8 and colour_shift < 0.02:
            verdict = "⚠ WEAK"
            print(f"  {verdict} — style barely visible.")
            print(f"     ➜  Try STYLE_WEIGHT = {self.cfg.style_weight * 10:.0e}  (10×)")
        elif mean_diff < 20 or colour_shift < 0.05:
            verdict = "~ MODERATE"
            print(f"  {verdict} — some effect; consider higher SW.")
        else:
            verdict = "✓ GOOD"
            print(f"  {verdict} — style clearly visible after {n} batches.")
            print("     ➜  Safe to proceed to full training.")
        return {
            "mean_diff":       mean_diff,
            "colour_shift":    colour_shift,
            "verdict":         verdict,
            "n_style_images":  len(self.cfg.style_images),
            "content_img":     content_pil,    # PIL Image 256×256 — content photo used for scoring
            "styled_img":      styled_pil,     # PIL Image 256×256 — styled output
            "style_ref_img":   style_ref_pil,  # PIL Image 256×256 — style reference (first image, resized)
        }

    # ── Step 3: full training ─────────────────────────────────────────────────

    def run_full_training(self) -> None:
        """Spawn ``main_style_trainer.py train`` subprocess and save config.json."""
        out_dir = self._repo_dir / "styles" / self.cfg.style_id
        self.cfg.output_dir = out_dir

        content_image = (
            self._COCO_VAL_CONTENT
            if self._COCO_VAL_CONTENT.exists()
            else self.cfg.style_image
        )

        cmd = [
            sys.executable, str(self._repo_dir / "bin" / "main_style_trainer.py"), "train",
            "--style",          *[str(p) for p in self.cfg.style_images],
            "--coco",           str(self.cfg.coco_path),
            "--out",            str(out_dir),
            "--id",             self.cfg.style_id,
            "--name",           self.cfg.style_name,
            "--content",        str(content_image),
            "--style-weight",   str(int(self.cfg.style_weight)),
            "--content-weight", str(int(self.cfg.content_weight)),
            "--tv-weight",      str(self.cfg.tv_weight),
            "--device",         self.cfg.device,
            "--epochs",         str(self.cfg.epochs),
            "--batch-size",     str(self.cfg.batch_size),
            "--image-size",     str(self.cfg.image_size),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self._repo_dir) + os.pathsep + env.get("PYTHONPATH", "")

        print("─── Full training ───")
        print("  Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(self._repo_dir), env=env)

        # Remove intermediate checkpoints
        checkpoints = sorted(out_dir.glob("*.ckpt_*.pth"))
        for ckpt in checkpoints:
            ckpt.unlink()
        if checkpoints:
            print(f"  Deleted {len(checkpoints)} checkpoint file(s).")

        # Persist config so resume_training() can reload it without re-entry
        self.cfg.save(out_dir)
        print(f"  Config saved to {out_dir / 'config.json'}")

        print("\n  Output files:")
        for f in sorted(out_dir.iterdir()):
            print(f"    {f.name:30s}  {f.stat().st_size // 1024:6d} KB")

    # ── Step 4: resume ────────────────────────────────────────────────────────

    def resume_training(self) -> None:
        """Resume from the latest checkpoint in the output directory.

        If ``config.json`` exists in the output directory it is loaded
        automatically — no need to re-enter hyperparameters.
        """
        out_dir = self._repo_dir / "styles" / self.cfg.style_id
        config_path = out_dir / "config.json"
        if config_path.exists():
            saved = TrainingConfig.load(out_dir)
            # Merge: prefer saved hyperparameters but keep runtime device
            saved.device = self.cfg.device
            self.cfg = saved
            print(f"  Loaded config from {config_path}")

        checkpoints = sorted(out_dir.glob("model.ckpt_*.pth"))
        if not checkpoints:
            print(f"  No checkpoints found in {out_dir} — nothing to resume.")
            return
        latest = checkpoints[-1]
        print(f"─── Resume from {latest.name} ───")

        from src.trainer.style_trainer import StyleTrainer  # noqa: PLC0415
        from src.trainer.preview import generate_preview     # noqa: PLC0415

        content_image = (
            self._COCO_VAL_CONTENT
            if self._COCO_VAL_CONTENT.exists()
            else self.cfg.style_image
        )

        def _progress(done: int, total: int, loss: float) -> None:
            print(f"\r  images {done:,}/{total:,}  loss={loss:.4f}", end="", flush=True)

        trainer = StyleTrainer(device=self.cfg.device)
        trainer.train(
            style_images=self.cfg.style_images,
            coco_dataset_path=self.cfg.coco_path,
            output_model_path=out_dir / "model.pth",
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            image_size=self.cfg.image_size,
            style_weight=self.cfg.style_weight,
            content_weight=self.cfg.content_weight,
            tv_weight=self.cfg.tv_weight,
            checkpoint_path=latest,
            progress_callback=_progress,
        )
        print("\n  Training complete.")

        trainer.export_onnx(out_dir / "model.pth", out_dir / "model.onnx")
        print("  ONNX exported.")

        generate_preview(
            onnx_path=out_dir / "model.onnx",
            preview_path=out_dir / "preview.jpg",
            content_image=content_image,
            size=256,
        )
        print(f"  Preview updated: {out_dir / 'preview.jpg'}")

        for ckpt in sorted(out_dir.glob("model.ckpt_*.pth")):
            ckpt.unlink()
            print(f"  Deleted {ckpt.name}")

    # ── Step 5: package ───────────────────────────────────────────────────────

    def package_output(self) -> pathlib.Path:
        """Copy output files to ``/kaggle/output/<style_id>/`` and create a zip.

        Returns:
            Path to the created ``.zip`` file.
        """
        out_dir = self._repo_dir / "styles" / self.cfg.style_id
        download_dir = pathlib.Path("/kaggle/output") / self.cfg.style_id
        download_dir.mkdir(parents=True, exist_ok=True)

        for f in out_dir.iterdir():
            shutil.copy2(f, download_dir / f.name)

        zip_path = pathlib.Path("/kaggle/output") / f"{self.cfg.style_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(download_dir.iterdir()):
                zf.write(f, f.name)

        print(f"─── Package ───")
        print(f"  Zip: {zip_path}  ({zip_path.stat().st_size // 1024} KB)")
        print("  Download from the Output tab (right panel) in the Kaggle notebook UI.")
        return zip_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Kaggle style-transfer training helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── shared style/coco args ────────────────────────────────────────────────
    def _add_style_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--style", type=pathlib.Path, required=True, nargs="+",
            help="Path(s) to style image(s) — supply multiple for mean-Gram training"
        )
        sp.add_argument("--id",     required=True,                    help="Style slug (folder name)")
        sp.add_argument("--name",   default="",                       help="Display name")
        sp.add_argument("--coco",   type=pathlib.Path,
                        default=pathlib.Path(
                            "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017"
                        ),
                        help="COCO train2017 directory")
        sp.add_argument("--device", default="cuda",                   help="torch device")

    def _add_weight_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--style-weight",   type=float, default=1e10)
        sp.add_argument("--content-weight", type=float, default=1e5)
        sp.add_argument("--tv-weight",      type=float, default=1e-6,
                        help="Total Variation loss weight (0.0 = disabled)")

    # verify
    sv = sub.add_parser("verify", help="Check GPU, COCO mount, and internet")
    _add_style_args(sv)

    # analyse
    sa = sub.add_parser("analyse", help="Analyse style image and print recommended weights")
    _add_style_args(sa)

    # smoke
    ss = sub.add_parser("smoke", help="Run a quick smoke test (N batches)")
    _add_style_args(ss)
    _add_weight_args(ss)
    ss.add_argument("--smoke-batches", type=int, default=2000)

    # train
    st = sub.add_parser("train", help="Full training run (2 epochs by default)")
    _add_style_args(st)
    _add_weight_args(st)
    st.add_argument("--epochs",     type=int, default=2)
    st.add_argument("--batch-size", type=int, default=4)
    st.add_argument("--image-size", type=int, default=256)

    # resume
    sr = sub.add_parser("resume", help="Resume training from last checkpoint")
    sr.add_argument("--id",     required=True, help="Style slug")
    sr.add_argument("--style",  type=pathlib.Path, nargs="+",
                    default=[pathlib.Path("/kaggle/working/my_style.jpg")])
    sr.add_argument("--coco",   type=pathlib.Path,
                    default=pathlib.Path(
                        "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017"
                    ))
    sr.add_argument("--device", default="cuda")

    # package
    spkg = sub.add_parser("package", help="Copy output to /kaggle/output/ and zip")
    spkg.add_argument("--id",    required=True, help="Style slug")
    spkg.add_argument("--style", type=pathlib.Path, nargs="+",
                      default=[pathlib.Path("/kaggle/working/my_style.jpg")])
    spkg.add_argument("--coco",  type=pathlib.Path,
                      default=pathlib.Path(
                          "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017"
                      ))

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = TrainingConfig(
        style_images=getattr(args, "style", [pathlib.Path("/kaggle/working/my_style.jpg")]),
        style_id=getattr(args, "id", ""),
        style_name=getattr(args, "name", getattr(args, "id", "")),
        coco_path=getattr(args, "coco", pathlib.Path(".")),
        style_weight=getattr(args, "style_weight", 1e10),
        content_weight=getattr(args, "content_weight", 1e5),
        tv_weight=getattr(args, "tv_weight", 1e-6),
        epochs=getattr(args, "epochs", 2),
        batch_size=getattr(args, "batch_size", 4),
        image_size=getattr(args, "image_size", 256),
        smoke_batches=getattr(args, "smoke_batches", 200),
        device=getattr(args, "device", "cuda"),
    )

    runner = KaggleStyleRunner(cfg)
    dispatch = {
        "verify":  runner.verify_environment,
        "analyse": runner.analyse_style,
        "smoke":   runner.run_smoke_test,
        "train":   runner.run_full_training,
        "resume":  runner.resume_training,
        "package": runner.package_output,
    }
    dispatch[args.cmd]()


if __name__ == "__main__":
    main()
