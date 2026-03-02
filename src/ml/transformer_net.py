"""TransformerNet — feed-forward style transfer network.

Architecture (Johnson 2016 + Instance Normalisation):
  Encoder  (3 conv layers, stride 2 downsampling)
  Residual (5 residual blocks with instance norm)
  Decoder  (2 fractional-strided conv layers, final tanh)
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvLayer(nn.Module):
    """Conv2d with reflection padding to avoid border artefacts."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        pad: int = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.reflection_pad(x))


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and instance normalisation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer(channels, channels, 3, 1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(channels, channels, 3, 1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleConvLayer(nn.Module):
    """Upsample then conv (avoids checkerboard artefacts from transposed conv)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        upsample: int | None = None,
    ) -> None:
        super().__init__()
        self.upsample = upsample
        pad: int = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = nn.functional.interpolate(
                x,
                scale_factor=float(self.upsample),
                mode="nearest",
            )
        return self.conv(self.reflection_pad(x))


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class TransformerNet(nn.Module):
    """Full style-transfer network (encoder → residual → decoder)."""

    def __init__(self) -> None:
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(32, 64, 3, 2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(64, 128, 3, 2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.residuals = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # Decoder
        self.decoder = nn.Sequential(
            UpsampleConvLayer(128, 64, 3, 1, upsample=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            UpsampleConvLayer(64, 32, 3, 1, upsample=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(32, 3, 9, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape [B, 3, H, W] in range [0, 255].

        Returns:
            Styled tensor of shape [B, 3, H, W] in range [0, 255].
        """
        out = self.encoder(x)
        out = self.residuals(out)
        out = self.decoder(out)
        # Clamp to valid pixel range
        return torch.clamp(out, 0.0, 255.0)
