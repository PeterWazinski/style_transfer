# CycleGAN Pre-trained Model Implementation Plan

**Status:** Complete — all phases implemented and tested (Phase 5 recompile pending)  
**Scope:** Monet, Van Gogh, Cézanne, Ukiyo-e (4 styles, ~44 MB total)

---

## 1. Background and feasibility

The official CycleGAN repository (junyanz/pytorch-CycleGAN-and-pix2pix) ships
four "photo-to-painting" generators trained on art datasets:

| Model ID        | Style description            | Download size |
|-----------------|------------------------------|---------------|
| `style_monet`   | Monet impressionist paintings | ~11 MB .pth   |
| `style_vangogh` | Van Gogh post-impressionism   | ~11 MB .pth   |
| `style_cezanne` | Cézanne post-impressionism    | ~11 MB .pth   |
| `style_ukiyoe`  | Japanese Ukiyo-e woodblock    | ~11 MB .pth   |

Download URL pattern:
```
http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/<model_id>.pth
```

**These are PyTorch state-dict files, not ONNX.**  A one-time export step
(~10 min on Kaggle, no GPU training) converts each `.pth` to `.onnx`.
After that, the files are dropped into `styles/<id>/model.onnx` exactly like
every other style — no further changes to the app workflow.

---

## 2. Tensor layout: what changes in the engine

The three model families used in this app differ only in how they
pre/post-process pixel values:

| Layout tag     | Tensor shape     | Pixel range | Used by              |
|----------------|------------------|-------------|----------------------|
| `nchw`         | [1, 3, H, W]     | [0, 255]    | NST TransformerNet   |
| `nhwc_tanh`    | [1, H, W, 3]     | [-1, 1]     | AnimeGANv3 (TF)      |
| `nchw_tanh` ⬅ new | [1, 3, H, W] | [-1, 1]     | CycleGAN (PyTorch)   |

CycleGAN uses standard NCHW order (same as NST) but expects pixels normalised
to [-1, 1] and outputs the same range (tanh final activation).

**The only code change required is a new `_infer_tile_nchw_tanh()` method in
`engine.py` — ~20 lines — plus a one-line dispatch in `_infer_tile()`.**
Everything else (gallery, catalog, batch styler, settings, tests) requires zero
changes because `tensor_layout` is already a first-class field in `StyleModel`
and the catalog JSON.

---

## 3. Implementation phases

### Phase 1 — ONNX export (Kaggle, ~30 min one-time effort) ✅ DONE

**Notebook:** `scripts/export_cyclegan_to_onnx.ipynb` (run on Kaggle T4 GPU)

The notebook:

1. Downloads each `.pth` from the Berkeley server (with HuggingFace fallback)
2. Loads the `ResnetGenerator` (self-contained, no CycleGAN repo clone needed)
3. Exports to ONNX with dynamic H/W axes, **opset 13** (opset 17 was changed to 13 to avoid
   a `Pad` version-conversion bug in `onnxscript`, and `dynamo=False` was set to force
   the TorchScript exporter which correctly embeds weights; opset 17 triggered the dynamo
   exporter and produced a 0.2 MB skeleton instead of the correct ~44 MB file)
4. Validates output shape and `[-1, 1]` range with ONNX Runtime
5. Runs a visual spot-check on a synthetic image

All four `.onnx` files (43.5 MB each) downloaded and installed in `styles/`.

> **Bugs fixed during export (all committed):**
> - `ModuleNotFoundError: onnxruntime` — added install cell before imports
> - `ModuleNotFoundError: onnxscript` — added to same install cell
> - `strict=True` rejected InstanceNorm running-stats buffers — changed to `strict=False`
> - `dynamo` exporter produced 0.2 MB skeleton — switched to `opset 13` + `dynamo=False`

---

### Phase 2 — Engine: add `nchw_tanh` layout ✅ DONE (commit `0803996`)

Add a new private method alongside the existing `_infer_tile_nhwc_tanh()`:

```python
def _infer_tile_nchw_tanh(
    self,
    session: "ort.InferenceSession",
    tile: Image.Image,
    *,
    use_float16: bool = False,
) -> Image.Image:
    """Inference for NCHW models with tanh-normalised I/O (e.g. CycleGAN).

    Input : [1, 3, H, W]  float32  range [-1, 1]
    Output: [1, 3, H, W]  float32  range [-1, 1]
    """
    try:
        arr = np.array(tile.convert("RGB"), dtype=np.float32)
        arr = arr / 127.5 - 1.0                        # [0,255] → [-1,1]
        tensor = arr.transpose(2, 0, 1)[np.newaxis, ...]  # → [1, 3, H, W]
        if use_float16:
            tensor = tensor.astype(np.float16)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: tensor})
    except MemoryError as exc:
        raise OOMError(...) from exc
    except Exception as exc:
        _msg = str(exc).lower()
        if any(k in _msg for k in ("out of memory", ...)):
            raise OOMError(...) from exc
        raise
    result_arr = np.clip((output[0][0].transpose(1, 2, 0) + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    result = Image.fromarray(result_arr)
    if result.size != tile.size:
        result = result.crop((0, 0, tile.width, tile.height))
    return result
```

Wire it in `_infer_tile()`:
```python
if tensor_layout == "nhwc_tanh":
    return self._infer_tile_nhwc_tanh(session, tile)
if tensor_layout == "nchw_tanh":                  # ← new dispatch
    return self._infer_tile_nchw_tanh(session, tile, use_float16=use_float16)
```

---

### Phase 3 — Install the four styles ✅ DONE (commits `a21bc82`, `0b956ff`)

Use `add_GAN_style.ipynb` (the new GAN-specific notebook) for each model, or drop files
manually:

```
styles/
  style_monet/
    model.onnx         # exported in Phase 1
    preview.jpg        # generate with the app's own Apply button
  style_vangogh/ …
  style_cezanne/ …
  style_ukiyoe/  …
```

Catalog entries (`styles/catalog.json`):
```json
{
  "id": "style_monet",
  "name": "Monet",
  "description": "Impressionist painting style — Claude Monet",
  "author": "junyanz/CycleGAN (MIT License)",
  "model_path": "styles/style_monet/model.onnx",
  "preview_path": "styles/style_monet/preview.jpg",
  "tensor_layout": "nchw_tanh",
  "is_builtin": true
}
```

Repeat for `style_vangogh`, `style_cezanne`, `style_ukiyoe`.

---

### Phase 4 — Tests ✅ DONE (commits `0803996`, `3adead1`)

Implemented in `tests/core/test_engine_nchw_tanh.py` (7 tests) and
`tests/scripts/test_batch_styler.py` (`TestApplyAllStylesTensorLayout`).

**Additional fix:** `generate_preview()` in `src/trainer/preview.py` was missing a
`nchw_tanh` branch — it fell through to the `nchw` branch, which skipped
normalisation and produced all-black thumbnails.  Fixed with an explicit
`elif tensor_layout == "nchw_tanh"` branch and 4 new tests in
`tests/trainer/test_preview.py`.

Add unit tests in `tests/core/test_engine_nchw_tanh.py`:

- `test_infer_tile_nchw_tanh_returns_correct_size` — mock session returning
  all-zero output, check result shape.
- `test_infer_tile_nchw_tanh_value_range` — verify pixel values are in [0, 255].
- `test_load_model_with_nchw_tanh_layout` — confirm the layout is stored in
  `_model_meta`.
- `test_apply_dispatches_to_nchw_tanh` — mock `_infer_tile_nchw_tanh`, call
  `engine.apply()`, assert it is called.

Add integration entries in `tests/scripts/test_batch_styler.py`:

- `test_nchw_tanh_layout_forwarded_from_catalog` — same pattern as the
  existing `TestApplyAllStylesTensorLayout` tests.

---

### Phase 5 — Recompile ⏳ PENDING

Run `.\compile.ps1` to produce updated `dist\PetersPictureStyler\` with the
four new style folders included.  No changes to `style_transfer.spec` needed —
the COLLECT step already picks up everything under `styles\`.

---

## 3b. Unplanned changes implemented during this work

| Change | Commit | Reason |
|--------|--------|--------|
| `generate_preview()` `nchw_tanh` branch | `3adead1` | Missing branch produced black preview thumbnails |
| Strength slider extended to 300% with extrapolation | `5c02358` | CycleGAN styles were too subtle at 100%; `strength > 1.0` now extrapolates the style delta beyond the model's native output |

---

## 4. What does NOT need to change

| Component             | Change needed? | Reason                                      |
|-----------------------|---------------|---------------------------------------------|
| `StyleModel` dataclass | No            | `tensor_layout` field already exists        |
| `StyleStore` / catalog JSON | No      | Reads arbitrary `tensor_layout` values      |
| Style gallery UI      | No            | Displays all catalog entries generically    |
| Main window / apply   | No            | Passes `tensor_layout` through unchanged    |
| `BatchStyler`         | No            | Fixed in commit `25be21a`                   |
| Settings              | No            | No new settings required                   |
| `add_CNN_style.ipynb` | No            | CNN-only notebook; `nchw` hardcoded — no changes needed |
| `add_GAN_style.ipynb` | No            | New GAN notebook; already has `nchw_tanh` and `nhwc_tanh` options |

---

## 5. Risk assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Download URL goes stale (efrosgans.eecs.berkeley.edu) | Low | Models are also mirrored on HuggingFace (`huggan`, `tmabraham` orgs) |
| ONNX export fails for a specific model | Low | All four use identical ResnetGenerator-9 architecture |
| Dynamic-shape inference slower than fixed-shape | Medium | CycleGAN tile-processes same as other styles; no worse |
| Colour shift vs. original paper | Low | Tiling introduces minor seams for all styles; already mitigated by overlap blending |
| GPU OOM on large images | Same as existing | Same tile-size / overlap settings apply |

---

## 6. Summary: effort and timeline

| Phase | Where | Effort |
|-------|-------|--------|
| 1 — ONNX export (4 models) | Kaggle | ~30 min (no training) |
| 2 — Engine `nchw_tanh` dispatch | Local | ~1 h (code + tests) |
| 3 — Install 4 styles + previews | Local app | ~20 min |
| 4 — Tests | Local | ~1 h |
| 5 — Recompile | Local | ~10 min |
| **Total** | | **~3 hours** |
