# scripts/ — index

| File | Purpose |
|---|---|
| `add_style.ipynb` | Interactive wizard to register a newly trained style into the gallery. Run cells 1–5 in order: picks the `.onnx` + `.pth` files via a dialog, lets you enter the display name and description, renders a preview thumbnail, then writes everything to `styles/catalog.json`. |
| `benchmark.py` | Measures ONNX inference latency for every style model in `styles/`. Runs a warmup pass then times N inferences per model and prints a summary table. |
| `code_metrics.ipynb` | Static analysis dashboard. Counts files, classes, methods, and lines of code for every Python source file, groups them into categories (`core`, `trainer`, `ui`, `tests`, …), and renders bar charts. |
| `download_pretrained.py` | Downloads pre-trained `.onnx` models from GitHub Releases and places them in `styles/`. Run once after cloning if you want bundled styles without training from scratch. |
| `kaggle_style_training.ipynb` | **Cockpit notebook for Kaggle GPU training.** Thin wrapper — each step calls one method on `KaggleStyleRunner`. Run on Kaggle (GPU T4) to smoke-test, train, package, and resume style models. See `kaggle_training_helper.py` for the underlying logic. |
| `kaggle_training_helper.py` | Backend for the Kaggle cockpit. Provides `TrainingConfig` (typed dataclass with JSON save/load) and `KaggleStyleRunner` (verify env, analyse style, smoke test, full train, resume, package). Also usable as a CLI: `python scripts/kaggle_training_helper.py <command>`. |
| `setup_models.py` | End-to-end local model setup: downloads pre-trained `.pth` weights, exports them to ONNX, and generates `preview.jpg` thumbnails for all styles. Run after cloning or after adding a new style. |
| `style_analysis.ipynb` | Interactive style-image analyser. Computes texture/geometry metrics for every JPEG in a folder, recommends `STYLE_WEIGHT` / `CONTENT_WEIGHT`, shows a thumbnail grid, and provides a local smoke-test widget (CPU) with a signal-strength check. Use before a Kaggle run to screen images. |
