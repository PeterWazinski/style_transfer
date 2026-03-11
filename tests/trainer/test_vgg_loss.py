"""Unit tests for VGGPerceptualLoss and gram_matrix helper."""
from __future__ import annotations

import pytest
import torch

from src.trainer.vgg_loss import VGGPerceptualLoss, gram_matrix


# ---------------------------------------------------------------------------
# gram_matrix
# ---------------------------------------------------------------------------

def test_gram_matrix_shape() -> None:
    feat = torch.rand(2, 16, 8, 8)   # batch=2, channels=16, h=8, w=8
    g = gram_matrix(feat)
    assert g.shape == (2, 16, 16), f"Expected (2,16,16), got {g.shape}"


def test_gram_matrix_is_symmetric() -> None:
    feat = torch.rand(1, 8, 4, 4)
    g = gram_matrix(feat)
    assert torch.allclose(g, g.transpose(1, 2), atol=1e-5)


def test_gram_matrix_is_positive_semidefinite() -> None:
    feat = torch.rand(1, 4, 4, 4)
    g = gram_matrix(feat)
    eigvals = torch.linalg.eigvalsh(g[0])
    assert (eigvals >= -1e-4).all(), "Gram matrix has negative eigenvalues"


# ---------------------------------------------------------------------------
# VGGPerceptualLoss — construction
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loss_fn() -> VGGPerceptualLoss:
    """Shared loss module (VGG weights downloaded once per test session)."""
    return VGGPerceptualLoss().eval()


def test_vgg_loss_creates_without_error_takes_long(loss_fn: VGGPerceptualLoss) -> None:
    assert loss_fn is not None


def test_vgg_weights_are_frozen_takes_long(loss_fn: VGGPerceptualLoss) -> None:
    for param in loss_fn.extractor.parameters():
        assert not param.requires_grad, "VGG weights must be frozen"


# ---------------------------------------------------------------------------
# VGGPerceptualLoss — compute_style_grams
# ---------------------------------------------------------------------------

def test_style_grams_returns_four_tensors_takes_long(loss_fn: VGGPerceptualLoss) -> None:
    style = torch.rand(1, 3, 64, 64) * 255.0
    grams = loss_fn.compute_style_grams(style)
    assert len(grams) == 4, f"Expected 4 Gram matrices, got {len(grams)}"


def test_style_grams_are_finite_takes_long(loss_fn: VGGPerceptualLoss) -> None:
    style = torch.rand(1, 3, 64, 64) * 255.0
    grams = loss_fn.compute_style_grams(style)
    for i, g in enumerate(grams):
        assert torch.isfinite(g).all(), f"Gram matrix {i} has non-finite values"


# ---------------------------------------------------------------------------
# VGGPerceptualLoss — forward losses
# ---------------------------------------------------------------------------

def test_content_loss_is_positive_scalar_takes_long(loss_fn: VGGPerceptualLoss) -> None:
    output = torch.rand(1, 3, 64, 64) * 255.0
    content = torch.rand(1, 3, 64, 64) * 255.0
    style = torch.rand(1, 3, 64, 64) * 255.0
    grams = loss_fn.compute_style_grams(style)
    c_loss, s_loss = loss_fn(output, content, grams)
    assert c_loss.shape == (), "content_loss must be a scalar"
    assert float(c_loss) > 0.0, "content_loss must be positive"


def test_style_loss_is_positive_scalar_takes_long(loss_fn: VGGPerceptualLoss) -> None:
    output = torch.rand(1, 3, 64, 64) * 255.0
    content = torch.rand(1, 3, 64, 64) * 255.0
    style = torch.rand(1, 3, 64, 64) * 255.0
    grams = loss_fn.compute_style_grams(style)
    _c_loss, s_loss = loss_fn(output, content, grams)
    assert s_loss.shape == (), "style_loss must be a scalar"
    assert float(s_loss) > 0.0, "style_loss must be positive"


def test_identical_output_and_content_gives_zero_content_loss_takes_long(
    loss_fn: VGGPerceptualLoss,
) -> None:
    """When output == content, content loss should be ~0."""
    img = torch.rand(1, 3, 64, 64) * 255.0
    style = torch.rand(1, 3, 64, 64) * 255.0
    grams = loss_fn.compute_style_grams(style)
    c_loss, _ = loss_fn(img, img, grams)
    assert float(c_loss) < 1e-3, f"Content loss for identical images: {float(c_loss)}"


def test_losses_are_finite_takes_long(loss_fn: VGGPerceptualLoss) -> None:
    output = torch.rand(1, 3, 64, 64) * 255.0
    content = torch.rand(1, 3, 64, 64) * 255.0
    style = torch.rand(1, 3, 64, 64) * 255.0
    grams = loss_fn.compute_style_grams(style)
    c_loss, s_loss = loss_fn(output, content, grams)
    assert torch.isfinite(c_loss), "content_loss is not finite"
    assert torch.isfinite(s_loss), "style_loss is not finite"
