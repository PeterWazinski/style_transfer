# Peter's Picture Stylist

A desktop app that applies the visual style of famous paintings to your photos using fast neural style transfer (Johnson et al., 2016).

Inference runs entirely through ONNX Runtime — no GPU or Python knowledge required to use the compiled app.

---

## Features

- **5 built-in styles** — Candy, Mosaic, Rain Princess, Abstract, Starry Night
- **Tiled inference** — handles large photos without running out of memory
- **Strength slider** — blend between original and styled result
- **Portable exe** — single `PetersPictureStylist.exe`, no install needed
- **Extensible** — train new styles on Kaggle (free GPU) and drop them in

---

## Quick Start (compiled exe)

1. Download `dist\PetersPictureStylist.exe`
2. Double-click — no Python or dependencies needed
3. Open a photo · pick a style · click Apply · save the result

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

## Build the Exe

```powershell
.\compile.ps1        # produces dist\PetersPictureStylist.exe (~157 MB)
```

Requires PyInstaller in the venv (`pip install pyinstaller`). torch/torchvision are excluded — inference uses ONNX Runtime only.

---

## Batch Styler

Apply every installed style to a single photo in one command.

```powershell
# PDF contact sheet — one page per style, low-res previews
.\scripts\batch_styler.ps1 -pdfoverview photos\photo.jpg

# Full-resolution per-style JPEGs — one file per style in a dated output folder
.\scripts\batch_styler.ps1 -fullimage photos\photo.jpg

# From cmd.exe
powershell -ExecutionPolicy RemoteSigned -File scripts\batch_styler.ps1 -pdfoverview photos\photo.jpg

# Optional flags — work with both modes
.\scripts\batch_styler.ps1 -fullimage photos\photo.jpg -Strength 0.85 -TileSize 512
```

| Flag | Default | Description |
|---|---|---|
| `-pdfoverview` | — | Generate a PDF contact sheet |
| `-fullimage` | — | Generate full-resolution JPEGs |
| `-Strength` | `1.0` | Style blend (0.0 = original, 1.0 = full style) |
| `-TileSize` | `256` | Tile size in pixels for tiled inference |

---

## Train a New Style

Training requires a GPU. The easiest free option is Kaggle (30 h/week GPU T4):

1. Open `scripts/kaggle_style_training.ipynb` in a Kaggle notebook
2. Add the `awsaf49/coco-2017-dataset` input dataset and enable GPU T4
3. Run all cells — training takes ~3 h for 2 epochs (~166 k images)
4. Download `model.onnx` + `preview.jpg` from the Output tab
5. Run `scripts/add_style.ipynb` locally to register the new style in the gallery

> **Analyse before training:** open `scripts/style_analysis.ipynb` to score your style image
> and run a local CPU smoke test before committing to the full Kaggle run.

---

## Project Layout

```
src/
  stylist/   Qt/PySide6 GUI app (no torch)
  trainer/   Training pipeline (torch, dev only)
  core/      Shared ONNX inference, registry, settings
styles/      Pretrained ONNX models + catalog.json
scripts/     Training notebooks, analysis tools, helper scripts (see scripts/index.md)
docs/        Architecture notes, roadmap
main_image_styler.py    → launch the app
main_style_trainer.py   → train a new style (CLI)
compile.ps1             → build the portable exe
```

---

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -k "not takes_long"   # ~6 s, 142 tests
```

---

## Credits

Pretrained models from [yakhyo/fast-neural-style-transfer](https://github.com/yakhyo/fast-neural-style-transfer) and [igreat/fast-style-transfer](https://github.com/igreat/fast-style-transfer) (both MIT).  
Architecture based on Johnson et al., *Perceptual Losses for Real-Time Style Transfer*, 2016.
