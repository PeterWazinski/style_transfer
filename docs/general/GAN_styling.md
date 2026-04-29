# GAN-Based Artistic Styling — Analysis for This Project

> Companion to `GAN-cartoonizer-overview.md` (architecture reference) and
> `fast_nst_canonical_artist_style.md` (CNN artist styles).
> This document focuses on the *practical differences* between the CNN-based
> fast NST already deployed and what a GAN-based pipeline would offer.

---

## 1. What You Currently Have — CNN Fast NST

The existing pipeline uses **feed-forward CNNs** (Johnson/TransformerNet architecture):

```
Content photo ──► TransformerNet (trained on one style) ──► Stylised photo
```

- One `.onnx` model per style, typically 6–8 MB
- Training: ~4 hours on CPU against a single style-reference image (or small set)
- Inference: single forward pass, tiles blended with Gaussian feathering
- Output: texture and colour of the style image applied to the content geometry
- **Strength slider** linearly blends output ↔ original

### What the CNN style transfer actually does

The TransformerNet minimises a weighted combination of:
- **Content loss** — VGG feature map distance (preserves photo structure)
- **Style loss** — Gram matrix distance from style reference (transfers texture/colour)
- **Total variation loss** — reduces high-frequency noise

The result: **colour palette and brushstroke texture** of the style painting are woven
into the photo while preserving the original geometry and sharpness of edges.

**Hundertwasser example behaviour:**
- Saturated, warm/earthy colour palette imposed across the whole image
- Organic, irregular-contour texture from Hundertwasser's painted surfaces
- Detail in the content photo is *preserved but textured over* — not blurred or
  redrawn; what looks like "blurred contours" is the style loss washing fine
  detail into coarser brushstroke patterns from the training paintings

---

## 2. What a GAN Would Change

### 2.1 The key architectural difference

```
CNN fast NST:
  Content photo ──► Fixed-weight TransformerNet ──► Output
  (style baked into weights during training)

Image-to-Image GAN (e.g. AnimeGANv3, CycleGAN):
  Content photo ──► Generator G ──► Output
                                        │
                              Discriminator D ──► "Does this look like domain B?"
                                        │
                              Adversarial feedback
```

The discriminator enforces that the output matches the **statistical distribution** of an
entire target domain (e.g. Ghibli films, Hundertwasser paintings), not just Gram matrix
proximity to a single reference image.

### 2.2 Practical differences for artistic styling

| Aspect | CNN Fast NST | GAN (Image-to-Image) |
|---|---|---|
| **Colour fidelity** | Colours shift toward style palette; tunable via style_weight | Stronger domain shift; discriminator enforces full palette replacement |
| **Edge / contour handling** | Edges survive, slightly textured | Can be drawn/redrawn as stylised lines (e.g. ink outlines in anime) |
| **Fine detail** | Detail softened by Gram matrix averaging | Detail actively reshaped to match domain convention (e.g. flat colour regions) |
| **Geometry** | Completely preserved | Preserved by most (CartoonGAN, AnimeGAN); modified only by U-GAT-IT |
| **Consistency across images** | High (same weights = same look) | High for same-domain images; may vary for out-of-distribution photos |
| **Style strength control** | Continuous 0–100 % strength slider | Typically binary (fully on/off); partial blending possible via `lerp` of latent features |
| **Training requirement** | One style image (or small set), fast training | Large image dataset (thousands) for the target domain, long training |

### 2.3 What a Hundertwasser GAN would look like vs the CNN

A CNN TransformerNet trained on Hundertwasser gives you:
> Original photograph with Hundertwasser *texture and palette* applied as a filter

A GAN trained on a large set of Hundertwasser paintings would give you:
> Photograph whose colour regions are **redrawn** to match Hundertwasser's characteristic
> flat-filled mosaic segments, bold black outlines, organic curves, and gold/turquoise
> palette — the content is recognisable but the rendering *mode* is fundamentally different

The GAN doesn't ask "what colour is the grass in the style?" — it asks "how would a
Hundertwasser painting handle a scene like this?" which allows it to, for example, add
mosaic tile boundaries where the original photo had a smooth gradient.

---

## 3. Training Your Own Artist-Specific GAN

### 3.1 Required training data

| GAN architecture | Minimum images | Recommended | Source |
|---|---|---|---|
| CycleGAN | 200–500 style images | 1,000+ | Scrape museum/Wikipedia (CC licence) |
| CartoonGAN-style | 2,000+ | 5,000+ | Scanned prints, Wikiart |
| AnimeGANv2/v3 | 2,000+ | 5,000+ | Fan art databases |
| U-GAT-IT | 3,500+ pairs | 10,000+ | Paired selfie/portrait datasets |

For a **Hundertwasser GAN** specifically:
- Wikiart has ~300 Hundertwasser images (publicly scrapeable)
- Kunsthaus Wien digital archive has ~150 more
- Total ~450–500 — marginal for a full GAN but usable for CycleGAN with heavy augmentation
- Better outcome: combine with related artists (Klimt, Schiele) to bulk the style domain

### 3.2 CycleGAN — the most practical choice for custom artists

CycleGAN does **unpaired** image-to-image translation — it does not need `(photo, painting)`
pairs, only two unpaired sets:

```
Domain A: 5,000 photographs (any subject)
Domain B: 450 Hundertwasser paintings

CycleGAN learns: A → B and B → A
Cycle-consistency loss: A → G_AB(A) → G_BA(G_AB(A)) ≈ A
```

This makes it the most realistic option for artist-specific training because paintings
and photos are structurally incomparable — you can't create ground-truth pairs.

**Training time:**
- 200 epochs on a single GPU (RTX 3080): ~12–24 hours
- On CPU only: not practical for full training (weeks)
- On Kaggle free GPU (T4): ~18–27 hours total — 2–3 sessions of up to 9 h each
  (Kaggle caps each notebook run at 9 h; the free tier gives 30 h/week, so this
  fits within a single week's quota)

**Output model size:** Generator is typically ~11 MB (ResNet-9 blocks backbone).
Exportable to ONNX.

### 3.3 Artist-style GAN training pipeline (proposed)

```
1. Collect paintings
   scripts/scrape_wikiart.py --artist hundertwasser --out data/hundertwasser/

2. Pre-process
   scripts/prepare_gan_dataset.py --style data/hundertwasser/ --photo data/coco_sample/
   → resizes to 256×256, augments (flip, jitter, crop)

3. Train CycleGAN
   python main_cyclegan_trainer.py --style_dir data/hundertwasser/ \
                                   --photo_dir data/coco_sample/ \
                                   --epochs 200 \
                                   --output runs/hundertwasser_gan/

4. Export to ONNX
   python scripts/export_gan_onnx.py --checkpoint runs/hundertwasser_gan/latest_G_A.pth \
                                     --output styles/hundertwasser_gan/model.onnx

5. Add to catalog
   scripts/add_style.ipynb  (existing workflow, just set tensor_layout accordingly)
```

None of steps 1–5 require changes to the inference engine — only the training side is new.

---

## 4. Available Open Source Frameworks

### 4.1 CycleGAN (official PyTorch implementation)

| | |
|---|---|
| **Repo** | https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix |
| **Paper** | Zhu et al., 2017 — "Unpaired Image-to-Image Translation" |
| **Framework** | PyTorch |
| **What it does** | Unpaired domain transfer — ideal for artist-style training |
| **Pre-trained** | Monet, Van Gogh, Ukiyo-e, Cézanne, horse↔zebra, etc. |
| **ONNX export** | Straightforward (`torch.onnx.export`) |
| **Training data** | Your own painting collection |
| **Local inference** | Yes, CPU usable (~3–8 s per 512 px image) |
| **Licence** | BSD |

Best fit for custom artist training. The pre-trained Van Gogh / Monet models
demonstrate exactly the kind of artistic domain transfer applicable to Hundertwasser.

---

### 4.2 AnimeGANv3

| | |
|---|---|
| **Repo** | https://github.com/TachibanaYoshino/AnimeGANv3 |
| **Framework** | PyTorch |
| **What it does** | Photo → anime/sketch style (Hayao Miyazaki, portrait sketch) |
| **ONNX models** | Available — already integrated into this project (`nhwc_tanh` layout) |
| **Custom training** | Possible; needs large anime image dataset |
| **Local inference** | Yes (already working in this project) |
| **Licence** | MIT (some weights Patreon-gated) |

---

### 4.3 StyleGAN3 / StyleGAN2-ADA (NVidia)

| | |
|---|---|
| **Repo** | https://github.com/NVlabs/stylegan3 |
| **Framework** | PyTorch |
| **What it does** | Generates new images *in the style of* a training set (not style transfer of photos) |
| **Key difference** | Unconditional generation — creates novel images, does not take a content photo |
| **ONNX export** | Not practical |
| **Training data** | 1,000+ images minimum; ADA variant works with ~100 |
| **Local GPU** | Needs CUDA — not suitable for laptop CPU inference |
| **Best for** | Generating novel paintings in an artist's style, not photo modification |

*Not directly applicable to the photo-modification use case.*

---

### 4.4 DiffusionGAN / DALL-E fine-tuning

Modern **diffusion models** (Stable Diffusion, DALL-E 3) with fine-tuning via:
- **DreamBooth** — fine-tune on 20–30 images of a style; guided by text prompt
- **LoRA** — lightweight adapter trained on style images, plugged into base SD model

| | |
|---|---|
| **What it does** | Text-guided image generation OR photo → style via img2img |
| **Photo modification** | Yes via img2img with style LoRA (`strength` parameter = denoising ratio) |
| **Local** | Stable Diffusion runs on 8 GB VRAM; CPU-only is very slow (minutes per image) |
| **Custom artist** | Train LoRA on 20–50 Hundertwasser paintings in ~1 hour on a GPU |
| **Quality** | Excellent — state of the art for artistic style |
| **Controllability** | High (prompt + img2img strength) |

*Most powerful option but heaviest infrastructure; not suitable for your current laptop-CPU-only setup without GPU VRAM.*

---

### 4.5 Feature summary

| Framework | Task | Custom artist? | ONNX? | CPU-feasible? | Effort |
|---|---|---|---|---|---|
| **CycleGAN** | Artist style transfer | ✅ (unpaired) | ✅ | ✅ (slow) | Medium |
| **AnimeGANv3** | Anime/sketch | ❌ (fixed domains) | ✅ (already done) | ✅ | Done |
| **CartoonGAN** | Cartoon line art | ❌ (fixed) | ✅ | ✅ | Low |
| **U-GAT-IT** | Portrait anime + face reshape | ❌ | Difficult | Slow | High |
| **StyleGAN3** | Generate new art | ✅ | ❌ | ❌ (GPU) | High |
| **SD + LoRA** | Text-guided style | ✅ (20–50 images) | ❌ | ❌ (GPU needed) | Medium |

---

## 5. Laptop Feasibility Assessment

### Current hardware context
- Intel Arc 140V integrated GPU (DirectML via `onnxruntime-directml`)
- CPU: Intel Core Ultra 7 155H (16 cores)
- RAM: 32 GB (assumed from Endress+Hauser laptop spec)

### Inference feasibility (onedir build on this laptop)

| Model | Size | CPU time / 1 MP image | DirectML time |
|---|---|---|---|
| Fast NST TransformerNet | 6–8 MB | ~2–5 s | ~0.5–1 s |
| AnimeGANv3 ONNX | 4–6 MB | ~3–8 s | ~1–2 s |
| CycleGAN ResNet-9 ONNX | ~11 MB | ~5–15 s | ~2–5 s |
| U-GAT-IT ONNX (if exported) | ~25 MB | ~30–60 s | ~10–20 s |
| Stable Diffusion (img2img) | ~2 GB | ~10 min | ~2–5 min (8 GB VRAM needed) |

**Conclusion:** CycleGAN ONNX is feasible on this laptop via DirectML. Stable Diffusion
is not — the Arc 140V has 8 GB shared VRAM which is technically enough but the driver
support for SD on Arc is still maturing and inference would be slow.

### Training feasibility

| Task | Hardware | Estimated time |
|---|---|---|
| Fine-tune TransformerNet (current CNN) | CPU (this laptop) | 4–8 hours |
| CycleGAN on custom artist | Kaggle T4 GPU | ~27–36 h total — 3–4 sessions × 9 h (fits in one week's 30 h free quota) |
| CycleGAN on custom artist | This laptop CPU | ~2–4 weeks — not practical |
| SD LoRA fine-tune | Kaggle T4 GPU | 1–2 hours |
| SD LoRA fine-tune | This laptop | Not practical |

**Training recommendation:** Keep using Kaggle for training (existing workflow). Local
laptop used only for inference.

---

## 6. Extending the Existing Project to GANs

The current architecture is well-positioned for a phased GAN extension.

### Phase 1 — Zero code change: drop in CycleGAN ONNX models (Low effort)

1. Download the pre-trained CycleGAN Van Gogh / Monet / Cézanne generators from the
   official repo (PyTorch `.pth` weights)
2. Export each generator to ONNX:
   ```python
   import torch
   from models import ResnetGenerator  # from CycleGAN repo

   G = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
   G.load_state_dict(torch.load("latest_net_G_A.pth"))
   G.eval()
   dummy = torch.randn(1, 3, 256, 256)
   torch.onnx.export(G, dummy, "monet.onnx", opset_version=11,
                     input_names=["input"], output_names=["output"],
                     dynamic_axes={"input": {0:1,2:"H",3:"W"},
                                   "output":{0:1,2:"H",3:"W"}})
   ```
3. Add `tensor_layout = "nchw"` to catalog entry — same layout as existing NST models
4. Works immediately in the UI, gallery, batch_styler, etc.

**Note:** CycleGAN output range is `[-1, 1]` NCHW tanh — the engine already handles
`nhwc_tanh`; you would need to add a `nchw_tanh` branch or rescale the output in a
post-processing step. This is a small change (one extra branch in `_infer_tile`).

### Phase 2 — Train a Hundertwasser CycleGAN on Kaggle (Medium effort)

1. Scrape ~450 Hundertwasser images from Wikiart + Kunsthaus Wien
2. Adapt the existing `kaggle_multi_pic_trainer.ipynb` to CycleGAN training loop
   (or use the official CycleGAN repo directly on Kaggle with a custom dataset)
3. Train 200 epochs (~12–20 h on T4)
4. Export generator to ONNX via the script above
5. Add to catalog

The existing `scripts/add_style.ipynb` workflow handles everything after step 4
with no changes.

### Phase 3 — New tensor_layout `nchw_tanh` (Small code change)

CycleGAN and many GAN generators output `[-1, 1]` in NCHW layout (not NHWC like
AnimeGANv3). Add one branch to `src/core/engine.py`:

```python
elif tensor_layout == "nchw_tanh":
    arr = np.array(tile, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 127.5 - 1.0
    out_raw = session.run(None, {session.get_inputs()[0].name: arr})[0]
    out = np.clip((out_raw[0].transpose(1, 2, 0) + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
```

This one addition unlocks all NCHW-output GAN models (CycleGAN, CartoonGAN, etc.).

### Phase 4 — Optional: Stable Diffusion img2img (Future, GPU required)

If you gain access to a GPU workstation or cloud instance with 8+ GB VRAM:
- Use `diffusers` library with a Hundertwasser LoRA adapter
- Add a separate "AI Remix" tab to the UI that routes to the diffusion pipeline
- Reuses the photo canvas, progress bar, save logic from the existing app

---

## 7. Summary Recommendations

### For immediate wins (no new training)
1. **Export CycleGAN pre-trained models** (Monet, Van Gogh, Cézanne, Ukiyo-e) to ONNX
   → 4 new high-quality artistic styles with ~1 hour effort
2. **Add `nchw_tanh` tensor layout** to the engine → unlocks all standard GAN generators

### For custom Hundertwasser GAN
3. Scrape Wikiart (~450 images) + augment heavily (flips, colour jitter, rotation)
4. Train CycleGAN on Kaggle (3–4 sessions over 1–2 weeks)
5. Export ONNX → drop into existing catalog — no app changes

### What the GAN will do better than the current CNN for Hundertwasser
- **Mosaic tile boundaries**: GAN discriminator learns to enforce the hard segment
  edges that define Hundertwasser's mosaic style; CNN Gram matrices only approximate this
- **Colour region uniformity**: GAN produces flat-filled regions; CNN produces texture
- **Characteristic outlines**: CartoonGAN-style discriminator can reward bold outlines
  that CNN NST cannot produce (NST only redistributes texture, never draws new edges)

### What the CNN still does better
- **Strength slider**: continuous blend; GAN is effectively on/off
- **Speed**: NST TransformerNet is faster (smaller model, simpler architecture)
- **Training time**: ~4 h on CPU vs ~12–20 h on Kaggle GPU
- **Stability**: No GAN training pathologies (mode collapse, checkerboard artefacts)

### Bottom line
The existing CNN pipeline is excellent for **texture and colour transfer**. GANs are
worth adding for **domain-level rendering style** (mosaic segments, ink outlines,
flat colour regions) that Gram matrix matching cannot produce. The two approaches are
complementary — both can coexist in the same catalog and UI.
