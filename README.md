# Peter's Picture Stylist

A desktop app that applies the visual style of famous paintings to your photos using fast neural style transfer (Johnson et al., 2016).

Inference runs entirely through ONNX Runtime — no GPU or Python knowledge required to use the compiled app.

---

## Features

- **18 built-in styles** — CNN TransformerNet models (Candy, Mosaic, Rain Princess, …) and CycleGAN models (Monet, Van Gogh, Cézanne, Ukiyo-e) plus Anime, Manga, Cubism and more
- **Tiled inference** — handles large photos without running out of memory
- **Strength slider** — blend between original and styled result
- **Style chains** — record and replay a sequence of styles as a YAML file
- **Batch Styler CLI** — headless style-overview PDFs and style-chain overviews from the command line
- **Portable exe** — copy `dist\PetersPictureStyler\` and double-click, no install needed
- **Extensible** — train new styles on Kaggle (free GPU) and drop them in without recompiling

---

## Quick Start (compiled app)

1. Copy the entire `dist\PetersPictureStyler\` folder to your machine
2. Double-click `PetersPictureStyler.exe` — no Python or dependencies needed
3. Open a photo · pick a style · click Apply · save the result

**Add a new style without recompiling:**
1. Drop the style folder (containing `model.onnx`) into `PetersPictureStyler\styles\`
2. Append the entry to `PetersPictureStyler\styles\catalog.json`
3. Restart the app — the new style appears in the gallery

---

## Run from Source

```powershell
# 1. Create venv and install dependencies
python -m venv .venv
.venv\Scripts\pip install -e ".[stylist]"

# 2. Launch
.venv\Scripts\python.exe main_image_styler.py
```

---

## Build the App

```powershell
.\compile.ps1        # produces dist\PetersPictureStyler\
```

Requires PyInstaller in the venv (`pip install pyinstaller`). torch/torchvision are excluded — inference uses ONNX Runtime only.

The build produces a **directory** (not a single exe). The `styles\` folder is copied in after compilation so styles remain editable without recompiling. Intermediate stub EXEs at `dist\` root are removed automatically by the spec file.

---

## Batch Styler CLI

`BatchStyler.exe` (headless, no GUI) produces overview PDFs from the command line.

```powershell
cd dist\PetersPictureStyler

# PDF contact sheet — all styles at multiple strengths
.\BatchStyler.exe --style-overview --outdir C:\out photos\photo.jpg

# PDF overview — all style chains in a directory
.\BatchStyler.exe --style-chain-overview sample_images\style-chains --outdir C:\out photos\photo.jpg

# Apply a single style chain to one photo
.\BatchStyler.exe --apply-style-chain sample_images\style-chains\my_chain.yml photos\photo.jpg
```

Progress is printed per-style with timing and a running count:
```
(1/18) Processing style 'Abstract' in 33 seconds.
(2/18) Processing style 'Anime' in 28 seconds.
...
PDF written: C:\out\photo_style_overview.pdf
```

Full option reference: `.\BatchStyler.exe --help`

### Batch script

`scripts\create_sample_overview.bat` runs both overview modes for every photo in
`sample_images\sample_pics\` and writes PDFs to `sample_images\style-overviews\`
and `sample_images\style-chain-overviews\`.

---

## Train a New Style

Training requires a GPU. The easiest free option is Kaggle (30 h/week GPU T4).
See **[training/index.md](training/index.md)** for the full step-by-step guide.

Short summary:

1. Open `training/kaggle_trainer.ipynb` (or `kaggle_multi_pic_trainer.ipynb`) on Kaggle
2. Add the `awsaf49/coco-2017-dataset` input dataset and enable GPU T4
3. Run all cells — training takes ~3 h for 2 epochs (~118 k images)
4. Download the `.onnx` from the Output tab
5. Run `training/add_CNN_style.ipynb` locally to register the new style in the gallery
6. Rebuild with `.\compile.ps1`

> **Analyse before training:** `scripts/style_analysis.ipynb` scores your style image,
> recommends weights, and runs a local CPU smoke test before committing to the full Kaggle run.

---

## Project Layout

```
src/
  stylist/      Qt/PySide6 GUI app (no torch)
    apply_controller.py       ← style-apply mixin
    style_chain_controller.py ← chain-management mixin
    help_dialogs.py           ← standalone help dialogs
  batch_styler/ Headless CLI for overview PDFs
  trainer/      Training pipeline (torch, dev only)
  core/         Shared ONNX inference, registry, settings, style-chain schema
styles/         Pretrained ONNX models + catalog.json
training/       Training notebooks + helpers (see training/index.md)
scripts/        Analysis tools, helper scripts (see scripts/index.md)
bin/            Subprocess entry points for trainer (memory isolation from GUI)
docs/           Architecture notes, refactoring plan
sample_images/  Sample photos, style chains, overview output
assets/         App icon
main_image_styler.py    → thin stub: launches src.stylist.app:main
main_style_trainer.py   → thin stub: launches src.trainer.app:main
compile.ps1             → build the portable app directory (dist\PetersPictureStyler\)
style_transfer.spec     → PyInstaller spec (PetersPictureStyler + BatchStyler)
```

---

## Tests

```powershell
# Fast suite (~8 s, 341 tests)
.venv\Scripts\python.exe -m pytest tests/ -q --tb=short --ignore=tests/trainer/test_multi_pic_gram.py
```

---

## Credits

Pretrained CNN models from [yakhyo/fast-neural-style-transfer](https://github.com/yakhyo/fast-neural-style-transfer) and [igreat/fast-style-transfer](https://github.com/igreat/fast-style-transfer) (both MIT).  
CycleGAN models from [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) (BSD).  
Architecture based on Johnson et al., *Perceptual Losses for Real-Time Style Transfer*, 2016.

