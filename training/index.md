# Training — Tools & Workflow

This folder contains everything needed to train new style-transfer models and
add them to the PetersPictureStyler gallery.

---

## Files at a Glance

| File | Purpose |
|------|---------|
| `kaggle_trainer.ipynb` | Train a **single-image** NST (TransformerNet / CNN) model on Kaggle |
| `kaggle_multi_pic_trainer.ipynb` | Train a **multi-image** NST model (mean Gram matrix across N images) on Kaggle |
| `export_cyclegan_to_onnx.ipynb` | Export the four official pretrained **CycleGAN** generators to ONNX |
| `add_CNN_style.ipynb` | Install a trained CNN (TransformerNet) `.onnx` into the local gallery |
| `add_GAN_style.ipynb` | Install a trained GAN (CycleGAN / AnimeGAN) `.onnx` into the local gallery |
| `kaggle_training_helper.py` | Backend logic for the Kaggle trainer notebooks (CLI also available) |
| `add_style_helper.py` | Backend logic for the add-style notebooks |

---

## Concept: Model Types

| Type | Tensor layout | Notebooks to use |
|------|--------------|-----------------|
| CNN / TransformerNet | `nchw` | `kaggle_trainer.ipynb` or `kaggle_multi_pic_trainer.ipynb` → `add_CNN_style.ipynb` |
| CycleGAN (PyTorch) | `nchw_tanh` | `export_cyclegan_to_onnx.ipynb` → `add_GAN_style.ipynb` |
| AnimeGAN / TF NHWC | `nhwc_tanh` | (export externally) → `add_GAN_style.ipynb` |

---

## Workflow A — Train a New CNN Style on Kaggle

Use this when you have one or more reference style images and want to train a
custom TransformerNet model.

### Prerequisites (Kaggle notebook settings)
1. **Add COCO 2017 dataset** — Notebook sidebar → *+ Add Data* →
   search `awsaf49/coco-2017-dataset` → Add.  
   Mounts at `/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017/`.
2. **Enable GPU** — Sidebar → Settings → Accelerator → **GPU T4 x1**.
3. **Enable internet** — Sidebar → Settings → Internet → On.  
   Required to download VGG16 weights (~550 MB) on the first run.

### Single-image training
Open `kaggle_trainer.ipynb` on Kaggle and run cells top-to-bottom:

| Step | What happens |
|------|-------------|
| 1 | GPU + COCO availability check |
| 1b | Internet access check |
| 2 | `git clone` repo + `pip install -e .[trainer]` |
| 3 | Upload your style image; set `STYLE_PATH`, `STYLE_ID`, `STYLE_NAME` |
| 4 | Smoke-test (2 000 batches, ~5 min) — validates the setup |
| 5 | Full training (~2 epochs × 118 k images, ~2–3 h on T4) |
| 6 | Package & download `.onnx` (+ optional `.pth` backup) |

### Multi-image training
Open `kaggle_multi_pic_trainer.ipynb`.  
Extra prerequisite: upload all style images into one folder under
`/kaggle/working/my_style/` before running.  
Remaining steps are identical to single-image training.

### Key hyperparameters (TrainingConfig)
| Parameter | Default | Notes |
|-----------|---------|-------|
| `style_weight` | `1e10` | Higher → stronger style transfer |
| `content_weight` | `1e5` | Higher → more content preserved |
| `tv_weight` | `1e-6` | Total Variation; set `0.0` to disable |
| `epochs` | `2` | 1 epoch ≈ 118 k COCO images |
| `batch_size` | `4` | Keep at 4 for T4 (16 GB VRAM) |
| `image_size` | `256` | Training crop size |

### CLI alternative (runs locally with a GPU)
`kaggle_training_helper.py` exposes the same pipeline as CLI sub-commands:

```bash
python training/kaggle_training_helper.py verify
python training/kaggle_training_helper.py analyse  --style path/to/style.jpg
python training/kaggle_training_helper.py smoke    --style path/to/style.jpg --id my_style --coco path/to/coco/train2017
python training/kaggle_training_helper.py train    --style path/to/style.jpg --id my_style --coco path/to/coco/train2017
python training/kaggle_training_helper.py resume   --id my_style
python training/kaggle_training_helper.py package  --id my_style
```

---

## Workflow B — Export Pretrained CycleGAN Models

Open `export_cyclegan_to_onnx.ipynb` on Kaggle (GPU recommended).

The notebook downloads the four official CycleGAN `.pth` weight files,
instantiates the ResNet-9 generator architecture, and exports each to ONNX
with dynamic H/W axes.

Output files (download after the run):

| ONNX file | Style |
|-----------|-------|
| `style_monet.onnx` | Monet impressionist paintings |
| `style_vangogh.onnx` | Van Gogh post-impressionism |
| `style_cezanne.onnx` | Cézanne post-impressionism |
| `style_ukiyoe.onnx` | Japanese Ukiyo-e woodblock |

Install via `add_GAN_style.ipynb` with layout **`nchw_tanh`**.

---

## Workflow C — Install a Trained Model Locally

After downloading an `.onnx` from Kaggle (or elsewhere), install it with the
appropriate notebook — run with Jupyter inside the repo root, or with VS Code's
Jupyter extension.

### CNN / TransformerNet model → `add_CNN_style.ipynb`

1. Open `training/add_CNN_style.ipynb` (Jupyter kernel pointed at `.venv`).
2. Run **Cell 1** — verifies repo root and loads `catalog.json`.
3. Run **Cell 2** — opens a file-picker dialog; select your `.onnx` file.
4. Run **Cell 3** — reports file sizes; asserts `.onnx` exists.
5. Run **Cell 4** — enter *Style name*, *Description*, and *Author* in the
   widget; the style ID is auto-derived (`"Rain Princess"` → `rain_princess`).
6. Run **Cell 5** — copies model files into `styles/<style_id>/`, generates
   a preview thumbnail, and appends the entry to `catalog.json`.

Tensor layout is always `nchw` for TransformerNet — no selection needed.

### GAN model (CycleGAN / AnimeGAN) → `add_GAN_style.ipynb`

Same 6-cell flow as above, with one extra step in Cell 4: select the
**tensor layout** from the dropdown.

| Layout | Use for |
|--------|---------|
| `nchw_tanh` | CycleGAN (PyTorch) |
| `nhwc_tanh` | AnimeGAN / TensorFlow NHWC export |

---

## Model File Formats

| Extension | Content | Required at runtime |
|-----------|---------|-------------------|
| `.onnx` | Inference graph + embedded weights | ✅ Yes |
| `.onnx.data` | External weight blob (only for models > 2 GB) — must stay alongside `.onnx` | ✅ Yes (if present) |
| `.pth` | PyTorch state-dict backup; used for re-training or re-export only | ❌ No |

---

## After Installing a New Style

1. Rebuild the app: run `compile.ps1` from the project root.
2. The new style appears automatically in both the GUI and BatchStyler — no
   code changes required.
