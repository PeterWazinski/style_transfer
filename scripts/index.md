# scripts/ — index

| File | Purpose |
|---|---|
| `add_style.ipynb` | Interactive wizard to register a newly trained style into the gallery. Run cells 1–5 in order: picks the `.onnx` + `.pth` files via a dialog, lets you enter the display name and description, renders a preview thumbnail, then writes everything to `styles/catalog.json`. |
| `benchmark.py` | Measures ONNX inference latency for every style model in `styles/`. Runs a warmup pass then times N inferences per model and prints a summary table. |
| `download_art.ipynb` | Downloads training images from Wikimedia Commons for a given painter/style set. Edit the **Configuration** cell to specify `OUTPUT_SUBDIR` and the list of artworks, then run all cells. Handles SSL-inspection proxies and API/CDN rate-limit retries automatically. |
| `kaggle_style_training.ipynb` | **Cockpit notebook for Kaggle GPU training.** Thin wrapper — each step calls one method on `KaggleStyleRunner`. Run on Kaggle (GPU T4) to smoke-test, train, package, and resume style models. See `kaggle_training_helper.py` for the underlying logic. |
| `kaggle_trainer.ipynb` | Single-image Kaggle training notebook (renamed from `kaggle_style_training.ipynb`). Accepts one style image via `TrainingConfig(style_images=[...])`. |
| `kaggle_multi_pic_trainer.ipynb` | **Multi-image Kaggle training notebook.** Configure `STYLE_IMAGES_DIR` pointing to a folder of style JPEGs; the notebook auto-collects all images, runs per-image analysis with thumbnail strip, smoke-tests with mean Gram matrices across all N images, then trains a single model that captures the generalised style. Adding N images only adds seconds to precomputation. |
| `kaggle_training_helper.py` | Backend for the Kaggle cockpit. Provides `TrainingConfig` (typed dataclass with JSON save/load) and `KaggleStyleRunner` (verify env, analyse style, smoke test, full train, resume, package). Also usable as a CLI: `python scripts/kaggle_training_helper.py <command>`. |
| `style_analysis.ipynb` | Interactive style-image analyser. Computes texture/geometry metrics for every JPEG in a folder, recommends `STYLE_WEIGHT` / `CONTENT_WEIGHT`, shows a thumbnail grid, and provides a local smoke-test widget (CPU) with a signal-strength check. Use before a Kaggle run to screen images. |
