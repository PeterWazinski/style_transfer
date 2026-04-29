# Fast Neural Style Transfer (Goal 1 Only)
## Canonical Single‑Artist Style Using Multiple Images

**Target environment**  
- Training: **Kaggle Notebook** with **NVIDIA Tesla T4 (16 GB)**  
- Framework: **PyTorch**  
- Perceptual network: **VGG16 (frozen)**

This guideline focuses **exclusively on Goal 1**:

> **Learn one stable, canonical “artist style” model by training on multiple artworks of the same artist.**

The goal is *consistency and robustness*, not diversity.

---

## 1. Concept Overview

Classic Fast Neural Style Transfer (Johnson et al.) trains a feed‑forward generator using **perceptual losses** computed in a fixed CNN (VGG) feature space. Style is represented via **Gram matrices** (feature‑correlation statistics) introduced by Gatys et al.

Instead of matching the style of **one** painting, we:

- Collect **many artworks** from the same artist
- Compute **average style statistics** (mean Gram matrices)
- Train a single generator to match these averages

This yields:
- A *canonical* artist look
- Less overfitting to individual motifs
- Stable inference (no style image needed at run time)

---

## 2. Mathematical Formulation (Goal 1)

### 2.1 Notation

- Content image: `c`  
- Style image set (same artist): `S = {s₁, …, sₙ}`  
- Generator: `x = T_θ(c)`  
- Fixed VGG16 feature maps at layer `l`: `F_l(·)`

### 2.2 Gram Matrix (Style Representation)

For feature maps `F_l(y) ∈ ℝ^{C×H×W}`:

\[
G_l(y) = \frac{1}{CHW} \; F_l(y)_{(C×HW)} · F_l(y)_{(C×HW)}^T
\]

Gram matrices encode texture/style information while discarding spatial layout.

### 2.3 Mean Style Target (Key Idea)

Precompute **mean Gram matrices** over all artist images:

\[
\bar{G}_l = \frac{1}{N} \sum_{i=1}^N G_l(s_i)
\]

These `\bar{G}_l` are fixed targets during training.

### 2.4 Loss Functions

**Content Loss** (layer `l_c`):

\[
\mathcal{L}_{content} = \lVert F_{l_c}(x) - F_{l_c}(c) \rVert_2^2
\]

**Style Loss (mean style)**:

\[
\mathcal{L}_{style}^{mean} = \sum_{l∈L_s} w_l \; \lVert G_l(x) - \bar{G}_l \rVert_F^2
\]

**Total Variation Loss (optional)**:

\[
\mathcal{L}_{tv} = \sum_{i,j}[(x_{i,j}-x_{i+1,j})^2 + (x_{i,j}-x_{i,j+1})^2]
\]

**Total Objective**:

\[
\min_θ \; α·\mathcal{L}_{content} + β·\mathcal{L}_{style}^{mean} + γ·\mathcal{L}_{tv}
\]

---

## 3. Recommended Layer Selection (VGG16)

| Purpose | VGG16 Layer |
|------|------------|
| Content | `relu3_3` |
| Style | `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3` |

(Indices depend on how layers are enumerated in code.)

---

## 4. PyTorch Implementation (Goal 1)

### 4.1 Utilities: Gram Matrix & VGG16 Feature Extractor

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def gram_matrix(feat):
    B, C, H, W = feat.shape
    f = feat.view(B, C, H * W)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (C * H * W)


class VGG16Features(nn.Module):
    def __init__(self, layer_ids):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.layer_ids = set(layer_ids)

    def forward(self, x):
        feats = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_ids:
                feats[i] = x
        return feats
```

---

### 4.2 Precompute Mean Style Targets

This step is done **once** before training.

```python
@torch.no_grad()
def compute_mean_grams(style_loader, vgg, style_layer_ids, device):
    sums = {lid: None for lid in style_layer_ids}
    count = 0

    for s in style_loader:
        s = s.to(device)
        feats = vgg(s)
        for lid in style_layer_ids:
            g = gram_matrix(feats[lid])      # [B,C,C]
            g_sum = g.sum(dim=0)             # [C,C]
            sums[lid] = g_sum if sums[lid] is None else sums[lid] + g_sum
        count += s.shape[0]

    return {lid: sums[lid] / count for lid in style_layer_ids}
```

---

### 4.3 Training Loop (AMP, Tesla T4‑friendly)

```python
from torch.amp import autocast, GradScaler


def train_mean_style(
    generator, content_loader, style_loader,
    vgg, content_layer_id, style_layer_ids,
    alpha=1.0, beta=1e5, gamma=1e-6,
    lr=1e-3, epochs=2, device='cuda'
):
    generator.train().to(device)
    vgg.to(device)

    mean_grams = compute_mean_grams(style_loader, vgg, style_layer_ids, device)

    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    scaler = GradScaler('cuda')

    for ep in range(epochs):
        for c in content_loader:
            c = c.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', dtype=torch.float16):
                x = generator(c)
                fx = vgg(x)
                fc = vgg(c)

                Lc = F.mse_loss(fx[content_layer_id], fc[content_layer_id])

                Ls = 0.0
                for lid in style_layer_ids:
                    gx = gram_matrix(fx[lid]).mean(dim=0)
                    Ls = Ls + F.mse_loss(gx, mean_grams[lid])

                tv = ((x[:, :, 1:, :] - x[:, :, :-1, :])**2).mean() + \
                     ((x[:, :, :, 1:] - x[:, :, :, :-1])**2).mean()

                loss = alpha*Lc + beta*Ls + gamma*tv

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"epoch {ep}: loss={loss.item():.4f}")
```

---

## 5. Generator Architecture Recommendations

- Use a **Johnson‑style residual generator**
- Replace BatchNorm with **InstanceNorm2d(affine=True)**
- Use nearest‑neighbor upsampling + conv (avoid transposed‑conv artifacts)

This setup is well‑established for high‑quality fast style transfer.

---

## 6. Dataset Curation (Critical for Goal 1)

Since the model learns *average style statistics*, data quality matters more than quantity.

**Recommended process**:

1. Select works from a **single period / medium** if possible  
2. Start with **20–50 images**, scale up if available
3. Remove frames, text, museum labels, signatures
4. Normalize color space and resolution
5. Remove near‑duplicate scans

Avoid mixing fundamentally different styles (e.g. oil paintings + sketches), or the average style will become muddy.

---

## 7. Practical Defaults (Tesla T4)

- Image size: **256 → 384 px** (then optionally 512)
- Batch size: **8–16** (with AMP)
- Optimizer: Adam, lr = 1e‑3
- Loss weights:  
  `α=1`, `β≈1e5`, `γ≈1e‑6`

---

## 8. What This Gives You

✅ One **stable feed‑forward model per artist**  
✅ No style image needed at inference time  
✅ Robust, recognizable artist signature  
✅ Ideal for production or batch stylization

---

*Scope note*: This document intentionally **excludes style sampling, multi‑style interpolation, or diversity‑focused methods** to stay strictly aligned with **Goal 1**.
