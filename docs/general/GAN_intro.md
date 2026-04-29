# A Practical Introduction to GANs

---

## 1. What is a GAN?

A **Generative Adversarial Network** (GAN) is a machine-learning architecture
introduced by Ian Goodfellow in 2014.  The key idea is elegantly simple: train
two neural networks against each other.

```
Real data ──►┐
             │   Discriminator  ──► "Real" / "Fake"
Generator ──►┘         │
    ▲                  │
    └──── loss signal ─┘
```

| Network | Role | Goal |
|---------|------|------|
| **Generator** G | Creates synthetic data (images, audio, text…) from random noise | Fool the Discriminator |
| **Discriminator** D | Classifies inputs as "real" (from training data) or "fake" (from G) | Catch the Generator |

Training alternates: D gets better at spotting fakes → G is forced to produce
more convincing output → D has to improve again → and so on.  At equilibrium
(the Nash equilibrium of this two-player game) G produces data
indistinguishable from real.

### Key characteristics

- **No explicit loss function** for image quality.  Quality emerges from the
  adversarial competition rather than from a handcrafted metric like MSE.
- **Mode collapse risk** — G may learn to produce only a narrow variety of
  outputs that reliably fool D (e.g. always painting sunflowers when asked for
  "Van Gogh").
- **Training instability** — the two networks must improve at roughly the same
  rate; if D becomes too strong, G gets no useful gradient; if D is too weak,
  G gets no pressure to improve.
- **Implicit distribution matching** — G learns the statistical distribution
  of the training data without ever seeing a loss that says "this pixel should
  be 128".

---

## 2. History

| Year | Milestone |
|------|-----------|
| **2014** | Ian Goodfellow et al. publish the original GAN paper. Proof-of-concept on MNIST digits. |
| **2015** | **DCGAN** — Deep Convolutional GAN. First stable training recipe; generates recognisable bedroom photos at 64×64. |
| **2016** | **pix2pix** — Conditional GAN for paired image-to-image translation (e.g. sketch → photo). |
| **2017** | **CycleGAN** — unpaired image translation (no matched pairs needed). Monet ↔ photo, horse ↔ zebra. Also **ProGAN** (progressive growing) reaching 1024×1024 faces. |
| **2018** | **BigGAN** — large-scale class-conditional generation; **StyleGAN** — disentangled latent space, photorealistic faces. |
| **2019** | **StyleGAN2** — removes artefacts. **GauGAN** (SPADE) — semantic layout → photorealistic scene. |
| **2020** | **VQGAN** — discrete latent codes; later backbone of DALL-E 1. |
| **2021** | DALL-E 1 (OpenAI) uses VQGAN + transformer. Diffusion models start outperforming GANs on benchmarks. |
| **2022–present** | **Diffusion models** (DALL-E 2, Stable Diffusion, Midjourney) largely replace GANs for general image synthesis. GANs remain dominant for **real-time / video** tasks (face swapping, super-resolution, style transfer). |

---

## 3. What are GANs used for?

GANs are **not** limited to image manipulation, though that is where they are
most visible.

### Image domain
- **Style transfer** — CycleGAN, AnimeGAN
- **Super-resolution** — ESRGAN (upscale low-res photos 4–8×)
- **Inpainting** — fill missing regions (e.g. remove objects)
- **Face editing** — age progression, expression transfer, deepfakes
- **Text-to-image** (early) — DALL-E 1 via VQGAN

### Video and 3D
- **Video prediction** — predict next frames
- **Talking heads** — animate a still portrait from audio
- **3D shape generation** — 3D-GAN, NeRF hybrids

### Audio
- **WaveGAN / MelGAN** — generate raw audio waveforms or spectrograms
- **Voice conversion** — change speaker identity while preserving content

### Scientific and engineering applications
- **Medical imaging** — generate synthetic MRI/CT scans to augment scarce
  training datasets; anonymise patient data
- **Drug discovery** — MolGAN generates valid molecular graphs
- **Data augmentation** — generate rare failure cases for industrial inspection
- **Anomaly detection** — train on normal data only; flag anything the
  discriminator finds "unusual"

---

## 4. Are DALL-E, Stable Diffusion, Midjourney GAN-based?

**No** — the major modern text-to-image systems are based on **diffusion
models**, not GANs.  The distinction matters:

| Architecture | How it works | Examples |
|---|---|---|
| **GAN** | Generator + Discriminator trained adversarially | StyleGAN, CycleGAN, AnimeGAN |
| **Diffusion model** | Gradually adds noise to real images, then learns to reverse the process | Stable Diffusion, DALL-E 2/3, Midjourney, Imagen |
| **VAE** (Variational Autoencoder) | Encodes to a latent distribution, decodes back | Early face generators |
| **Transformer (autoregressive)** | Predicts next token (pixel or patch) sequentially | DALL-E 1 (uses VQGAN tokens) |

**DALL-E 1 (2021)** is a hybrid: it uses a VQGAN (GAN-adjacent) as a
tokeniser, then a GPT-style transformer to sequence those tokens.

**DALL-E 2 / 3, Stable Diffusion, Midjourney** — pure diffusion.  They won
out over GANs for text-to-image because diffusion models:
- Are more stable to train (no adversarial game)
- Cover the full distribution better (no mode collapse)
- Accept text conditioning more naturally via CLIP embeddings

**GANs still dominate** where inference speed is critical (real-time video,
mobile super-resolution) or where the input is structured (sketch, semantic
map, reference image) rather than free-form text.

---

## 5. Advantages of GANs over CNNs (for style transfer)

"CNN" in the context of this project means the **Neural Style Transfer (NST)
TransformerNet** — a feed-forward convolutional network trained with a
perceptual loss.  The comparison is meaningful because both produce stylised
images.

| Criterion | CNN TransformerNet (this app) | CycleGAN |
|---|---|---|
| **Training data** | One style-reference image | Hundreds of paintings by the artist |
| **Training time** | 2–4 hours on Kaggle T4 | 18–36 hours on Kaggle T4 |
| **Model size** | ~6–7 MB | ~11 MB (generator only) |
| **Inference speed** | Fast; deterministic | Fast; deterministic |
| **Texture vs. content** | Strong on texture patterns | Strong on brushstroke feel and colour palette |
| **Generalisation** | May miss artist-specific colour palette | Learns artist's actual visual language |
| **Artefacts** | Tiling seams at high resolution (mitigated by overlap blending) | Can produce checkerboard artefacts at tile boundaries |
| **Custom artist** | Needs only 1 painting | Needs 50–200 paintings |

**In plain terms:** NST/CNN is fast to train and cheap on data — great for
geometric or texture-heavy styles (mosaic, candy, Escher).  CycleGAN captures
"how an artist actually sees the world" more convincingly for painters like
Monet or Van Gogh, at the cost of more training data and compute.

---

## 6. How are GANs trained?

Training a GAN is a **minimax game** — G minimises, D maximises the same
objective.

### Standard training loop (one iteration)

```
Step 1 — Update Discriminator:
  a. Sample a batch of real images from the training set.
  b. Sample random noise z, generate fake images:  fake = G(z)
  c. Compute D loss:
       L_D = −[ log D(real) + log(1 − D(fake)) ]
  d. Backprop through D only; update D weights.

Step 2 — Update Generator:
  a. Sample new random noise z, generate fakes:  fake = G(z)
  b. Compute G loss:
       L_G = −log D(fake)       ← G wants D to call fakes "real"
  c. Backprop through D (frozen) into G; update G weights only.
```

### CycleGAN training (unpaired)

CycleGAN adds a **cycle-consistency loss** to avoid mode collapse when there
are no paired examples:

```
Photo → G_AB → Fake_Painting → G_BA → Reconstructed_Photo
                                         ↕ must match original (L1 loss)
```

Two generators (G_AB, G_BA) and two discriminators (D_A, D_B) are trained
simultaneously.  The cycle loss forces G_AB to produce something meaningful
that G_BA can invert, preventing G_AB from ignoring the input content.

### Practical training tips
- **Spectral normalisation** on D stabilises gradients.
- **Learning rate** G and D are typically trained at the same rate (2e-4).
- **Replay buffer** — keep a pool of previously generated fakes to prevent D
  from forgetting early patterns ("catastrophic forgetting").
- **Label smoothing** — feed D "real = 0.9" instead of "real = 1.0" to
  prevent overconfidence.
- **Instance normalisation** (not batch normalisation) in the generators for
  image translation tasks.

---

## 7. Training data and computing resources

### Data requirements

| Model | Minimum | Recommended |
|-------|---------|-------------|
| NST TransformerNet | 1 style image + ~80 000 content images (COCO) | Same |
| CycleGAN (artist style) | 50 paintings + 1 000 photos | 200+ paintings + 5 000 photos |
| CycleGAN (object domain, e.g. horse↔zebra) | 1 000 images per domain | 5 000+ per domain |
| StyleGAN2 (faces) | 10 000 images | 70 000+ (FFHQ) |

The paintings do **not** need to be paired with photos.  Any collection of the
artist's work scraped from WikiArt or a museum API is sufficient.

### Hardware

| Setup | Typical epoch time (CycleGAN 256×256) | Full training (~200 epochs) |
|-------|--------------------------------------|-----------------------------|
| NVIDIA RTX 3080 (10 GB) | ~4 min | ~13 hours |
| Kaggle T4 (free tier, 30 h/week) | ~6 min | ~18–27 hours (2–3 sessions × 9 h) |
| Intel Arc 140V (this laptop, DirectML) | Not practical — no PyTorch DML training support | — |
| CPU only | ~60–90 min/epoch | Weeks |

### Storage
- COCO 2017 training images: ~18 GB
- WikiArt artist subset (~200 paintings): ~50–200 MB
- CycleGAN checkpoint: ~45 MB (both generators + both discriminators)
- Exported ONNX generator: ~11 MB

---

## 8. Why do `nchw`, `nhwc_tanh`, and `nchw_tanh` exist?

This is a purely practical engineering question that arises because the three
model families were implemented in different frameworks with different
conventions.

### The two independent axes

**Axis 1: Channel order (how pixel channels are laid out in memory)**

A colour image has 3 channels (R, G, B).  A 4-D tensor adds batch size N and
the spatial dimensions H×W.  Two conventions exist:

| Tag prefix | Layout | What it means | Origin |
|---|---|---|---|
| `nchw` | [N, C, H, W] | Batch, Channels, Height, Width | PyTorch default |
| `nhwc` | [N, H, W, C] | Batch, Height, Width, Channels | TensorFlow default |

A model trained in PyTorch saves weights that expect `nchw` tensors.  A model
trained in TensorFlow expects `nhwc`.  Feeding the wrong layout gives
physically incorrect results (the network "sees" the image sideways / as a
noise pattern) — or an explicit shape error like the one below:

```
[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input:
  index: 3  Got: 1024  Expected: 3
```
That error means the model expected 3 in position 3 (channels, NHWC format)
but received 1024 (width, NCHW format).

**Axis 2: Pixel normalisation (what numeric range the model expects)**

| Tag suffix | Range | Formula (uint8 → float) | Used by |
|---|---|---|---|
| *(none)* | [0, 255] | `x = pixel` | NST TransformerNet |
| `_tanh` | [−1, 1] | `x = pixel / 127.5 − 1.0` | GAN generators with tanh output |

GAN generators almost universally use a **tanh** activation on the final layer,
which saturates at ±1.  If you feed [0, 255] data into a model expecting [−1,
1], the network is wildly out of distribution; outputs will be uniformly grey
or completely saturated.

### The three combinations in this project

| Tag | Layout | Range | Framework | Model |
|-----|--------|-------|-----------|-------|
| `nchw` | [1, 3, H, W] | [0, 255] | PyTorch | NST TransformerNet |
| `nhwc_tanh` | [1, H, W, 3] | [−1, 1] | TensorFlow | AnimeGANv3 |
| `nchw_tanh` | [1, 3, H, W] | [−1, 1] | PyTorch | CycleGAN generators |

CycleGAN is PyTorch (→ nchw order) but uses tanh (→ _tanh normalisation),
giving `nchw_tanh`.  AnimeGANv3 is TensorFlow (→ nhwc order) and also uses
tanh, giving `nhwc_tanh`.

### What the engine does with this information

In `src/core/engine.py`, the `tensor_layout` string routes each tile to the
correct pre/post-processing path:

```python
def _infer_tile(self, session, tile, *, use_float16, tensor_layout):
    if tensor_layout == "nhwc_tanh":
        return self._infer_tile_nhwc_tanh(session, tile)
    if tensor_layout == "nchw_tanh":
        return self._infer_tile_nchw_tanh(session, tile, use_float16=use_float16)
    # default: nchw  [0, 255]
    ...
```

The value is stored in `styles/catalog.json` per style and is completely
transparent to the user — they just see the style name in the gallery.

---

## Further reading

- Goodfellow et al. (2014) — *Generative Adversarial Networks*
  https://arxiv.org/abs/1406.2661
- Zhu et al. (2017) — *Unpaired Image-to-Image Translation using
  Cycle-Consistent Adversarial Networks*
  https://arxiv.org/abs/1703.10593
- Karras et al. (2019) — *A Style-Based Generator Architecture for GANs*
  https://arxiv.org/abs/1812.04948
- Ho et al. (2020) — *Denoising Diffusion Probabilistic Models* (why
  diffusion won)
  https://arxiv.org/abs/2006.11239
