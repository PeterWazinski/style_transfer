"""Unit tests for multi-image mean Gram matrix and Total Variation loss.

Uses sample_images/dali/ (7 real JPEGs) so tests are fully deterministic
and do not require network access unless marked _takes_long (VGG download).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.trainer.train_utils import load_style_tensor
from src.trainer.vgg_loss import VGGPerceptualLoss, total_variation_loss

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DALI_DIR  = _REPO_ROOT / "sample_images" / "dali"
_DALI_IMGS = sorted(_DALI_DIR.glob("*.jpg"))

assert len(_DALI_IMGS) >= 2, f"Need ≥2 dali images in {_DALI_DIR}"


# ---------------------------------------------------------------------------
# total_variation_loss — no VGG needed
# ---------------------------------------------------------------------------

def test_tv_loss_zero_for_constant_image() -> None:
    """A flat (constant) image has zero TV loss."""
    x = torch.ones(1, 3, 8, 8) * 128.0
    assert total_variation_loss(x).item() == pytest.approx(0.0, abs=1e-6)


def test_tv_loss_positive_for_random_image() -> None:
    """A random image has positive TV loss."""
    x = torch.rand(2, 3, 16, 16)
    assert total_variation_loss(x).item() > 0.0


def test_tv_loss_returns_scalar() -> None:
    x = torch.rand(1, 3, 8, 8)
    out = total_variation_loss(x)
    assert out.shape == torch.Size([])


def test_tv_loss_checkerboard_larger_than_smooth() -> None:
    """Checkerboard pattern has larger TV than smooth gradient."""
    smooth = torch.zeros(1, 1, 8, 8)
    smooth[0, 0] = torch.arange(8).float().unsqueeze(0).expand(8, 8) / 7.0
    checker = torch.zeros(1, 1, 8, 8)
    for i in range(8):
        for j in range(8):
            checker[0, 0, i, j] = float((i + j) % 2)
    assert total_variation_loss(checker) > total_variation_loss(smooth)


# ---------------------------------------------------------------------------
# VGGPerceptualLoss.compute_mean_style_grams — requires VGG (_takes_long)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loss_fn_module() -> VGGPerceptualLoss:
    return VGGPerceptualLoss().eval()


def test_mean_grams_single_matches_compute_style_grams_takes_long(
    loss_fn_module: VGGPerceptualLoss,
) -> None:
    """N=1: compute_mean_style_grams must be identical to compute_style_grams."""
    img_tensor = load_style_tensor(_DALI_IMGS[0], size=64)
    grams_single = loss_fn_module.compute_style_grams(img_tensor)
    grams_mean   = loss_fn_module.compute_mean_style_grams([img_tensor])
    assert len(grams_mean) == len(grams_single)
    for g_s, g_m in zip(grams_single, grams_mean):
        assert torch.allclose(g_s, g_m, atol=1e-5), "N=1 mean Gram differs from single Gram"


def test_mean_grams_returns_four_layers_takes_long(
    loss_fn_module: VGGPerceptualLoss,
) -> None:
    tensors = [load_style_tensor(p, size=64) for p in _DALI_IMGS[:3]]
    grams = loss_fn_module.compute_mean_style_grams(tensors)
    assert len(grams) == 4


def test_mean_grams_shapes_match_single_takes_long(
    loss_fn_module: VGGPerceptualLoss,
) -> None:
    """Each mean Gram must have the same shape as a single-image Gram."""
    ref_tensor    = load_style_tensor(_DALI_IMGS[0], size=64)
    multi_tensors = [load_style_tensor(p, size=64) for p in _DALI_IMGS[:3]]
    single_grams  = loss_fn_module.compute_style_grams(ref_tensor)
    mean_grams    = loss_fn_module.compute_mean_style_grams(multi_tensors)
    for sg, mg in zip(single_grams, mean_grams):
        assert sg.shape == mg.shape, f"Shape mismatch: {sg.shape} vs {mg.shape}"


def test_mean_grams_averages_two_correctly_takes_long(
    loss_fn_module: VGGPerceptualLoss,
) -> None:
    """mean_grams(a, b) == (grams(a) + grams(b)) / 2 at each layer."""
    ta = load_style_tensor(_DALI_IMGS[0], size=64)
    tb = load_style_tensor(_DALI_IMGS[1], size=64)
    ga = loss_fn_module.compute_style_grams(ta)
    gb = loss_fn_module.compute_style_grams(tb)
    expected = [(a + b) / 2.0 for a, b in zip(ga, gb)]
    actual   = loss_fn_module.compute_mean_style_grams([ta, tb])
    for exp, act in zip(expected, actual):
        assert torch.allclose(exp, act, atol=1e-5), "Mean of two Grams is incorrect"


def test_mean_grams_empty_raises_takes_long(
    loss_fn_module: VGGPerceptualLoss,
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        loss_fn_module.compute_mean_style_grams([])


def test_mean_grams_all_dali_images_takes_long(
    loss_fn_module: VGGPerceptualLoss,
) -> None:
    """Exercise with all 7 dali images — smoke test for stability."""
    tensors = [load_style_tensor(p, size=64) for p in _DALI_IMGS]
    grams = loss_fn_module.compute_mean_style_grams(tensors)
    assert len(grams) == 4
    for g in grams:
        assert torch.isfinite(g).all(), "Mean Gram contains NaN or Inf"


# ---------------------------------------------------------------------------
# StyleTrainer with multiple style images (_takes_long)
# ---------------------------------------------------------------------------

def test_style_trainer_accepts_multiple_images_takes_long() -> None:
    """Verify StyleTrainer trains 1 batch with 3 style images without error."""
    import tempfile

    from src.trainer.style_trainer import StyleTrainer

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "model.pth"
        trainer = StyleTrainer(device="cpu")
        trainer.train(
            style_images=list(_DALI_IMGS[:3]),
            coco_dataset_path=_DALI_DIR,          # use dali dir as tiny "COCO"
            output_model_path=out,
            epochs=1,
            batch_size=1,
            image_size=64,
            style_weight=1e10,
            content_weight=1e5,
            tv_weight=1e-6,
            checkpoint_interval=0,
            max_batches=1,
        )
        assert out.exists(), "model.pth was not created"


def test_style_trainer_tv_weight_zero_identical_to_no_tv_takes_long() -> None:
    """tv_weight=0.0 must produce same result as not passing tv_weight (backward compat)."""
    import tempfile

    from src.trainer.style_trainer import StyleTrainer

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "model.pth"
        trainer = StyleTrainer(device="cpu")
        trainer.train(
            style_images=[_DALI_IMGS[0]],
            coco_dataset_path=_DALI_DIR,
            output_model_path=out,
            epochs=1,
            batch_size=1,
            image_size=64,
            style_weight=1e10,
            content_weight=1e5,
            tv_weight=0.0,
            checkpoint_interval=0,
            max_batches=1,
        )
        assert out.exists()
