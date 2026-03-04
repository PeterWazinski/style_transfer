"""VGGPerceptualLoss — content + style loss using frozen VGG-16 features.

Content loss: MSE of relu3_3 feature maps between output and content image.
Style loss  : MSE of Gram matrices at relu1_2, relu2_2, relu3_3, relu4_3
              between output and style image (averaged across layers).
"""
from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


# ---------------------------------------------------------------------------
# Gram matrix helper
# ---------------------------------------------------------------------------

def gram_matrix(feature: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for a feature map.

    Args:
        feature: Tensor of shape [B, C, H, W].

    Returns:
        Gram matrix of shape [B, C, C] (normalised by H*W).
    """
    b, c, h, w = feature.shape
    f = feature.view(b, c, h * w)
    gram = torch.bmm(f, f.transpose(1, 2))
    return gram / (c * h * w)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

# VGG-16 layer indices for the relu activations we care about.
# (0-indexed from the features Sequential)
#   relu1_2 → index 3
#   relu2_2 → index 8
#   relu3_3 → index 15   ← used for content loss too
#   relu4_3 → index 22
_STYLE_LAYERS: tuple[int, ...] = (3, 8, 15, 22)
_CONTENT_LAYER: int = 15


class VGGFeatureExtractor(nn.Module):
    """Extracts intermediate feature maps from frozen VGG-16."""

    def __init__(self) -> None:
        super().__init__()
        vgg: nn.Sequential = cast(nn.Sequential, models.vgg16(weights=VGG16_Weights.DEFAULT).features)
        # Freeze all parameters
        for param in vgg.parameters():
            param.requires_grad_(False)

        # Keep only up to the last layer we need
        max_idx: int = max(_STYLE_LAYERS) + 1
        self.slices: nn.ModuleList = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:_STYLE_LAYERS[0] + 1]),
            nn.Sequential(*list(vgg.children())[_STYLE_LAYERS[0] + 1:_STYLE_LAYERS[1] + 1]),
            nn.Sequential(*list(vgg.children())[_STYLE_LAYERS[1] + 1:_STYLE_LAYERS[2] + 1]),
            nn.Sequential(*list(vgg.children())[_STYLE_LAYERS[2] + 1:max_idx]),
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return feature maps at relu1_2, relu2_2, relu3_3, relu4_3."""
        features: list[torch.Tensor] = []
        h: torch.Tensor = x
        for s in self.slices:
            h = s(h)
            features.append(h)
        return features


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """Combined content + style perceptual loss using VGG-16."""

    def __init__(self) -> None:
        super().__init__()
        self.extractor = VGGFeatureExtractor()
        # VGG-16 ImageNet normalisation
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0,
        )

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise pixel tensor [0,255] for VGG-16 input."""
        return (x - self.mean) / self.std  # type: ignore[operator]

    def forward(
        self,
        output: torch.Tensor,
        content: torch.Tensor,
        style_grams: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute content and style losses.

        Args:
            output:       Styled image tensor [B, 3, H, W] in [0, 255].
            content:      Original content image [B, 3, H, W] in [0, 255].
            style_grams:  Precomputed Gram matrices of style image (4 layers).

        Returns:
            Tuple of (content_loss, style_loss) — scalar tensors.
        """
        out_norm = self._normalise(output)
        con_norm = self._normalise(content)

        out_features = self.extractor(out_norm)
        con_features = self.extractor(con_norm)

        # Content loss: MSE at relu3_3 (index 2 in our slice list)
        content_loss: torch.Tensor = F.mse_loss(out_features[2], con_features[2].detach())

        # Style loss: Gram MSE across all 4 layers
        # Expand style_gram from [1, C, C] to [B, C, C] to avoid broadcasting warning.
        style_loss: torch.Tensor = torch.tensor(0.0, device=output.device)
        batch_size: int = output.size(0)
        for out_feat, style_gram in zip(out_features, style_grams):
            out_gram = gram_matrix(out_feat)
            style_gram_b = style_gram.detach().expand(batch_size, -1, -1)
            style_loss = style_loss + F.mse_loss(out_gram, style_gram_b)

        return content_loss, style_loss

    @torch.no_grad()
    def compute_style_grams(self, style_image: torch.Tensor) -> list[torch.Tensor]:
        """Precompute Gram matrices for a style image (call once before training).

        Args:
            style_image: Style image tensor [1, 3, H, W] in [0, 255].

        Returns:
            List of 4 Gram matrices (one per VGG layer).
        """
        norm = self._normalise(style_image)
        features = self.extractor(norm)
        return [gram_matrix(f) for f in features]
