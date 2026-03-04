"""Unit tests for TransformerNet: shapes, dtypes, and forward pass invariants."""
from __future__ import annotations

import pytest
import torch

from src.trainer.transformer_net import TransformerNet


@pytest.fixture()
def net() -> TransformerNet:
    return TransformerNet().eval()


# ---------------------------------------------------------------------------
# Forward pass — shape checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b,h,w", [
    (1, 64, 64),
    (2, 128, 128),
    (1, 256, 384),   # non-square
])
def test_output_shape_matches_input(net: TransformerNet, b: int, h: int, w: int) -> None:
    """Output spatial dimensions must equal input."""
    x = torch.zeros(b, 3, h, w)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (b, 3, h, w), f"Expected {(b, 3, h, w)}, got {out.shape}"


def test_output_dtype_is_float32(net: TransformerNet) -> None:
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        out = net(x)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Output range
# ---------------------------------------------------------------------------

def test_output_clamped_to_0_255(net: TransformerNet) -> None:
    """All output pixels must be in [0, 255] after clamping."""
    x = torch.rand(1, 3, 64, 64) * 255.0
    with torch.no_grad():
        out = net(x)
    assert float(out.min()) >= 0.0, "Output below 0"
    assert float(out.max()) <= 255.0, "Output above 255"


def test_zero_input_gives_bounded_output(net: TransformerNet) -> None:
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (1, 3, 64, 64)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 255.0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow_through_network() -> None:
    """Backprop must not raise and loss.item() must be finite."""
    net_train = TransformerNet().train()
    x = torch.rand(1, 3, 64, 64, requires_grad=False) * 255.0
    out = net_train(x)
    loss = out.mean()
    loss.backward()
    for name, param in net_train.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"


# ---------------------------------------------------------------------------
# Parameter count sanity
# ---------------------------------------------------------------------------

def test_parameter_count_reasonable(net: TransformerNet) -> None:
    """Network should have between 50 k and 2 M parameters."""
    n = sum(p.numel() for p in net.parameters())
    assert 50_000 < n < 2_000_000, f"Unexpected parameter count: {n}"
