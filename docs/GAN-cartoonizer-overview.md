# GAN-Based Cartoonizers — Technical Overview

## 1. How GAN-Based Cartoonization Works (Theory)

### 1.1 The core GAN setup

A **Generative Adversarial Network** consists of two neural networks trained in opposition:

```
Real photo ──► Generator (G) ──► Cartoon-like image
                                        │
                                        ▼
                              Discriminator (D) ──► Real cartoon / Fake?
                                        │
                               Adversarial loss
                                        │
                              Back-propagated to G
```

- **Generator G** transforms a photo into a cartoonized version
- **Discriminator D** tries to distinguish real cartoon images from G's outputs
- G learns to fool D; D learns to catch G — they improve each other

This produces a model where G, after training, can transform any photo into a cartoon *without* having paired training data (photo A → cartoon A) — the network learns the *distribution* of cartoons, not individual mappings.

### 1.2 What makes cartoonization different from plain NST

| | Neural Style Transfer (this project) | GAN Cartoonizer |
|---|---|---|
| Architecture | Feed-forward TransformerNet | Generator + Discriminator |
| Training data | Style images only | Large cartoon dataset (~10k images) |
| Output geometry | Unchanged (texture only) | Can reshape face proportions |
| Speed | Fast (one forward pass) | One forward pass after training |
| Training cost | Hours on CPU | Days on GPU |
| Control | Strength slider (0–100%) | Binary (on/off per model) |
| Artefacts | Tile seam risk | GAN mode collapse, checkerboard noise |

---

## 2. The Three Main Architectures

### 2.1 CartoonGAN (Chen et al., 2018)

**Paper:** "CartoonGAN: Generative Adversarial Networks for Photo Cartoonization"

**Key idea:** Two losses drive the generator:
1. **Content loss** — VGG feature comparison to preserve photo structure
2. **Adversarial loss** — discriminator trained on real cartoon edge maps

**Edge-promoting loss:** Cartoon images have flat colour regions separated by crisp edges. CartoonGAN pre-processes training cartoons with an edge detector and blurs the edges slightly, forcing the discriminator to especially reward sharp edge production.

```
Training cartoons ──► Edge detector ──► Blurred-edge negative samples
                                               │
Real cartoons ────────────────────────────────►├─► Discriminator
Generated images ─────────────────────────────►│
```

**Result:** Photos → flat colour areas + strong ink-line edges. Geometry unchanged.
**Weakness:** Style is tied to the training cartoon dataset; one model = one style.

---

### 2.2 AnimeGAN / AnimeGANv2 / AnimeGANv3 (2020–2023)

**Papers:** "AnimeGAN: A Novel Lightweight GAN for Photo Animation" + v2/v3 improvements

**Key improvements over CartoonGAN:**
- **Gram matrix style loss** (same as NST) added to the adversarial loss — better texture
- **Grayscale style loss** — penalises colour bleeding into B&W areas (important for anime)
- **Color reconstruction loss** — prevents the generator from shifting colours too aggressively
- Much **lighter generator** (MobileNet backbone variant) — runs on CPU at reasonable speed

**Training datasets:** Hayao Miyazaki (Studio Ghibli), Shinkai Makoto (Your Name), Kon Satoshi — one model per artist.

**AnimeGANv2 improvements:**
- Reduced generator parameters (~8M vs ~11M)
- Better colour fidelity
- Less "whitewashing" of skin tones

**AnimeGANv3 improvements:**
- Portrait mode (face-specific training variant)
- Better hair detail
- Reduced checkerboard artefacts

**Geometry:** Still no face shape modification.

---

### 2.3 U-GAT-IT (Kim et al., 2019)

**Paper:** "U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation"

**Key idea:** Adds an **attention mechanism** to focus on semantically important regions:

```
Input photo
     │
     ▼
CAM (Class Activation Map) ──► Attention weights
     │                               │
     ▼                               ▼
Encoder ──────────────────► Weighted feature maps
                                     │
                                     ▼
                              AdaLIN (Adaptive Layer-Instance Normalization)
                                     │
                                     ▼
                               Decoder ──► Selfie2Anime output
```

**AdaLIN (Adaptive Layer-Instance Normalization):**
- Dynamically blends between Layer Norm and Instance Norm per-layer
- Learned during training; allows the model to decide how much global vs. local style to apply
- This is what enables **face geometry modification** — the attention map focuses on eyes, chin, face outline and can reshape them

**Training data:** Selfie2Anime dataset (~3,500 selfies + ~3,500 anime portraits)

**Result:** The only architecture in this list that genuinely reshapes face proportions — larger eyes, smaller chin, smoother skin.
**Weakness:** Very expensive to train (days on a single GPU), only works well on portraits.

---

## 3. Open Source Python Projects (Local Execution)

### 3.1 AnimeGANv2 — TachibanaYoshino/AnimeGANv2

| | |
|---|---|
| **Repo** | https://github.com/TachibanaYoshino/AnimeGANv2 |
| **Framework** | TensorFlow 2.x |
| **Pre-trained models** | Hayao, Shinkai, Paprika, Face-Portrait |
| **Input** | Any photo or video |
| **Local CPU** | Yes (slow, ~5–15s per 512px image) |
| **Local GPU** | CUDA / TF-DirectML |
| **Python** | 3.7–3.10 |
| **Licence** | MIT |
| **Quality** | Very good for anime texture; best maintained repo |

**Minimal inference:**
```python
import tensorflow as tf
from AnimeGANv2 import generator  # simplified

model = tf.saved_model.load("weights/Hayao")
img = load_image("photo.jpg")          # [1, H, W, 3], float32 [-1, 1]
result = model(img, training=False)
save_image(result, "out.jpg")
```

---

### 3.2 AnimeGANv3 — TachibanaYoshino/AnimeGANv3

| | |
|---|---|
| **Repo** | https://github.com/TachibanaYoshino/AnimeGANv3 |
| **Framework** | PyTorch |
| **Pre-trained models** | Hayao_v3, PortraitSketch, etc. |
| **ONNX export** | Yes — can be run with onnxruntime (fits your existing engine!) |
| **Licence** | MIT |
| **Note** | Some weights gated behind Patreon; core model free |

This is the most interesting option for this project because it can be exported to ONNX and dropped directly into the existing `StyleTransferEngine` pipeline with zero architecture changes.

---

### 3.3 CartoonGAN — Yijunmaverick/CartoonGAN-Test-Pytorch-Torch

| | |
|---|---|
| **Repo** | https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch |
| **Framework** | PyTorch |
| **Pre-trained models** | Hayao, Hosoda, Paprika, Shinkai |
| **ONNX export** | Straightforward via `torch.onnx.export` |
| **Licence** | MIT |
| **Quality** | Good but slightly dated; AnimeGANv2 usually better |

---

### 3.4 U-GAT-IT — znxlwm/UGATIT-pytorch

| | |
|---|---|
| **Repo** | https://github.com/znxlwm/UGATIT-pytorch |
| **Framework** | PyTorch |
| **Pre-trained models** | Selfie2Anime (official weights available) |
| **ONNX export** | Possible but non-trivial (attention module needs careful tracing) |
| **Licence** | MIT |
| **GPU required?** | Inference works on CPU; slow (~30–60s per portrait) |
| **Best for** | Portrait selfie → anime face with geometry change |

---

### 3.5 Lightweight / web-friendly alternatives

| Project | Framework | Note |
|---|---|---|
| **whitebox-cartoonize** (SystemErrorWang) | TF 1.x | White-box approach; good for landscape photos |
| **photo_cartoonization** (SystemErrorWang) | TF 2.x | Updated version of above |
| **pytorch-CartoonGAN** (znxlwm) | PyTorch | Clean re-implementation, easy to read |

---

## 4. Integration Effort for a Local Desktop App

### Option A — AnimeGANv3 ONNX (recommended)

**Effort:** Low. Export the PyTorch model to ONNX once, then drop the `.onnx` file into the existing `styles/` catalog. The `StyleTransferEngine` tiling + blending pipeline works unchanged.

**Steps:**
1. Clone AnimeGANv3, install PyTorch (CPU)
2. Run `torch.onnx.export(model, dummy, "animegan_hayao.onnx", opset_version=11)`
3. Add catalog entry in `styles/catalog.json`
4. Done — works in the existing UI with progress bar, strength slider, batch scripts etc.

**Caveat:** GAN models often use `InstanceNorm` which has dynamic shapes — you may need `dynamic_axes` in the ONNX export and a fixed-size padding pre-step.

### Option B — U-GAT-IT as separate pipeline

**Effort:** High. The attention + AdaLIN architecture doesn't export to ONNX cleanly without custom operator handling. Would need a separate inference path, likely TorchScript or direct PyTorch.

**Recommendation:** Only worth it if face geometry reshaping (big eyes / small chin) is a hard requirement.

### Option C — Separate feature / plugin

Add a "Cartoonize" button alongside the existing style gallery that routes to a separate GAN inference pipeline while reusing the same UI shell (photo canvas, progress bar, save logic).

---

## 5. Summary Comparison

| | CartoonGAN | AnimeGANv2 | AnimeGANv3 | U-GAT-IT |
|---|---|---|---|---|
| Geometry change | No | No | No | Yes (portrait) |
| ONNX-ready | Easy | No (TF) | Yes | Difficult |
| CPU speed | ~10s | ~10s | ~5s | ~40s |
| Texture quality | Good | Very good | Best | Good |
| Portrait specialisation | No | Partial | Yes | Yes |
| Integration effort | Medium | High (TF) | **Low** | High |
| Best use case | Landscapes / scenes | Anime scenes | Everything | Selfie → anime face |

**Recommendation for this project:** Start with **AnimeGANv3 ONNX export** — lowest integration effort, best quality, fits the existing engine architecture perfectly.
