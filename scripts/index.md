# scripts/ — index

| File | Purpose |
|---|---|
| `benchmark.py` | Measures ONNX inference latency for every style model in `styles/`. Runs a warmup pass then times N inferences per model and prints a summary table. |
| `download_art.ipynb` | Downloads training images from Wikimedia Commons for a given painter/style set. Edit the **Configuration** cell to specify `OUTPUT_SUBDIR` and the list of artworks, then run all cells. Handles SSL-inspection proxies and API/CDN rate-limit retries automatically. |
| `gen_palette_ico_temp.py` | Generates palette and icon assets for the application. |
| `style_analysis.ipynb` | Interactive style-image analyser. Computes texture/geometry metrics for every JPEG in a folder, recommends `STYLE_WEIGHT` / `CONTENT_WEIGHT`, shows a thumbnail grid, and provides a local smoke-test widget (CPU) with a signal-strength check. Use before a Kaggle run to screen images. |

