"""Integration tests for the full training pipeline (toy run)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.core.trainer import StyleTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_toy_coco(directory: Path, n_images: int = 10) -> Path:
    """Write *n_images* synthetic JPEG images into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(directory / f"img_{i:04d}.jpg")
    return directory


def _make_style_image(path: Path) -> Path:
    """Write a synthetic style PNG at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


# ---------------------------------------------------------------------------
# Single gradient step
# ---------------------------------------------------------------------------

def test_one_step_loss_is_finite(tmp_path: Path) -> None:
    """One forward-backward step must produce a finite, positive loss."""
    coco_dir = _make_toy_coco(tmp_path / "coco", n_images=4)
    style_path = _make_style_image(tmp_path / "style.png")
    out_path = tmp_path / "model.pth"

    losses: list[float] = []

    trainer = StyleTrainer(device="cpu")
    trainer.train(
        style_images=[style_path],
        coco_dataset_path=coco_dir,
        output_model_path=out_path,
        epochs=1,
        batch_size=2,
        image_size=64,
        checkpoint_interval=0,  # no checkpoints
        progress_callback=lambda done, total, loss: losses.append(loss),
    )

    assert len(losses) >= 1
    for loss_val in losses:
        assert math.isfinite(loss_val), f"Non-finite loss: {loss_val}"
        assert loss_val >= 0.0, f"Negative loss: {loss_val}"


def test_output_pth_saved(tmp_path: Path) -> None:
    """The .pth file must exist after training completes."""
    coco_dir = _make_toy_coco(tmp_path / "coco", n_images=4)
    style_path = _make_style_image(tmp_path / "style.png")
    out_path = tmp_path / "model.pth"

    trainer = StyleTrainer(device="cpu")
    returned = trainer.train(
        style_images=[style_path],
        coco_dataset_path=coco_dir,
        output_model_path=out_path,
        epochs=1,
        batch_size=2,
        image_size=64,
        checkpoint_interval=0,
    )

    assert returned == out_path
    assert out_path.exists()


def test_pth_contains_model_state(tmp_path: Path) -> None:
    """The .pth checkpoint must include a 'model_state' key."""
    coco_dir = _make_toy_coco(tmp_path / "coco", n_images=4)
    style_path = _make_style_image(tmp_path / "style.png")
    out_path = tmp_path / "model.pth"

    trainer = StyleTrainer(device="cpu")
    trainer.train(
        style_images=[style_path],
        coco_dataset_path=coco_dir,
        output_model_path=out_path,
        epochs=1,
        batch_size=2,
        image_size=64,
        checkpoint_interval=0,
    )

    ckpt: dict = torch.load(str(out_path), map_location="cpu")
    assert "model_state" in ckpt
    assert len(ckpt["model_state"]) > 0


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def test_export_onnx_produces_file(tmp_path: Path) -> None:
    """export_onnx() must write a non-empty .onnx file."""
    coco_dir = _make_toy_coco(tmp_path / "coco", n_images=4)
    style_path = _make_style_image(tmp_path / "style.png")
    pth_path = tmp_path / "model.pth"
    onnx_path = tmp_path / "model.onnx"

    trainer = StyleTrainer(device="cpu")
    trainer.train(
        style_images=[style_path],
        coco_dataset_path=coco_dir,
        output_model_path=pth_path,
        epochs=1,
        batch_size=2,
        image_size=64,
        checkpoint_interval=0,
    )
    returned = trainer.export_onnx(pth_path, onnx_path, image_size=64)

    assert returned == onnx_path
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Full-epoch test (marked slow)
# ---------------------------------------------------------------------------

def test_full_epoch_takes_long(tmp_path: Path) -> None:
    """Run a full epoch on a small synthetic dataset; loss must decrease."""
    n_images = 40
    coco_dir = _make_toy_coco(tmp_path / "coco", n_images=n_images)
    style_path = _make_style_image(tmp_path / "style.png")
    out_path = tmp_path / "model.pth"

    losses: list[float] = []

    trainer = StyleTrainer(device="cpu")
    trainer.train(
        style_images=[style_path],
        coco_dataset_path=coco_dir,
        output_model_path=out_path,
        epochs=2,
        batch_size=4,
        image_size=64,
        checkpoint_interval=0,
        progress_callback=lambda done, total, loss: losses.append(loss),
    )

    assert all(math.isfinite(l) for l in losses), "Non-finite loss detected"
    # Rough sanity: final loss should be lower than peak loss
    assert losses[-1] < max(losses), "Loss never decreased"
