# Multi-Image Style Trainer — Implementation Plan & Tracking

**Goal:** Train a canonical artist-style model from N style images (mean Gram matrices)
instead of a single image, for a more robust and motif-free style transfer.

**Reference:** `docs/fast_nst_canonical_artist_style.md`  
**New notebook:** `scripts/kaggle_multi_pic_trainer.ipynb`

---

## Phase Checklist

### Phase 1 — Core library changes (no UI yet)

- [ ] **P1-1** `src/trainer/vgg_loss.py`  
  Add `compute_mean_style_grams(style_images: list[Tensor]) -> list[Tensor]`  
  Loops over N style images, accumulates Gram sums, divides by N.  
  `forward()` and `compute_style_grams()` stay unchanged (backward compatible).

- [ ] **P1-2** `src/trainer/style_trainer.py`  
  Change single-image Gram precomputation (line ~129):  
  `style_tensor = load_style_tensor(style_images[0], ...)` + `compute_style_grams()`  
  → `compute_mean_style_grams([load_style_tensor(p, ...) for p in style_images])`  
  Works for N=1 (identical to current behaviour) and N>1.

- [ ] **P1-3** `scripts/kaggle_training_helper.py`  
  - `TrainingConfig.style_image: Path` → `style_images: list[Path]`  
  - Add `style_images_dir: Path | None = None` — if set, auto-expands `*.jpg/*.png` to list  
  - Update `analyse_style()`, `run_smoke_test()`, `run_full_training()`, `resume_training()`  
  - Smoke test: build mean Gram from all N images (not just first)
  - `run_smoke_test()` return dict: add `"n_style_images": int`

- [ ] **P1-4** Unit tests: `tests/trainer/test_multi_pic_gram.py`  
  - `test_mean_grams_single_image_matches_compute_style_grams` — N=1 must produce same result  
  - `test_mean_grams_averages_correctly` — 2 known images → known average  
  - `test_style_trainer_accepts_multiple_images` — smoke train 1 batch, N=3

---

### Phase 2 — New Kaggle notebook `scripts/kaggle_multi_pic_trainer.ipynb`

- [ ] **P2-1** Header markdown: title, instructions, structure overview

- [ ] **P2-2** Step 1: GPU + COCO check (copy from `kaggle_trainer`, unchanged)

- [ ] **P2-3** Step 1b: Internet access check (copy, unchanged)

- [ ] **P2-4** Step 2: Clone + install (copy, unchanged)

- [ ] **P2-5** Step 3: Configure style images directory  
  - Set `STYLE_IMAGES_DIR` path (uploaded Kaggle dataset or `/kaggle/working/`)  
  - Auto-glob `*.jpg`, `*.jpeg`, `*.png`  
  - Print list of found images + count  
  - Assert ≥ 2 images found

- [ ] **P2-6** Step 3a: Style analysis grid  
  - Run `analyse_style()` on each image  
  - Print table: filename | flat% | patch_std | edge | verdict  
  - Warn if any image looks like a bad candidate (very flat or very different from others)  
  - Show thumbnail grid using `IPython.display`

- [ ] **P2-7** Step 3b: Smoke test (~10 min on T4)  
  - Uses mean Gram from all N images  
  - Shows 3-way comparison: content | styled | style ref (first image)  
  - Same verdict table as `kaggle_trainer`

- [ ] **P2-8** Step 4: Full training (calls `runner.run_full_training()`, unchanged)

- [ ] **P2-9** Step 5: Preview + download (copy from `kaggle_trainer`, unchanged)

- [ ] **P2-10** Step 6: Resume from checkpoint (copy, unchanged)

---

### Phase 3 — Style analyser enhancements

- [ ] **P3-1** `src/trainer/style_analyser.py`  
  Add `analyse_style_set(paths: list[Path]) -> dict` that:  
  - Calls `analyse_style()` on each image  
  - Returns per-image metrics + aggregate stats (mean, std per metric)  
  - Warns if any image is an outlier (e.g. flat_pct > 2× mean)

- [ ] **P3-2** Add `hist_overlap_matrix(paths) -> np.ndarray`  
  N×N pairwise colour-histogram similarity.  
  High similarity → consistent artist palette.  
  Low average similarity → warning "images may be too diverse".

---

### Phase 4 — Integration & Polish

- [ ] **P4-1** `scripts/add_style.ipynb`  
  No changes needed — it accepts any model.onnx/model.pth regardless of how it was trained.

- [ ] **P4-2** `scripts/index.md`  
  Add row for `kaggle_multi_pic_trainer.ipynb`.

- [ ] **P4-3** Run full test suite — all existing tests must still pass (P1-2 is backward compatible).

- [ ] **P4-4** Test on Kaggle with 5 Hundertwasser images — compare smoke test result vs single-image baseline.

- [ ] **P4-5** Commit + push all changes.

---

## Design Decisions

| # | Question | Decision |
|---|---|---|
| D1 | TV loss (γ≈1e-6 per guide)? | Add as optional param `tv_weight: float = 0.0` — off by default to not change existing behaviour |
| D2 | Smoke test uses N images or just first? | All N — otherwise smoke test does not validate the actual training objective |
| D3 | Kaggle dataset vs working dir upload? | Support both — `style_images_dir` accepts any path |
| D4 | `add_style.ipynb` changes? | None — accepts any trained model |
| D5 | Min N images enforced? | Warn if N=1 (use `kaggle_trainer` instead), but do not hard-fail |
| D6 | Backward compat for N=1? | Yes — `compute_mean_style_grams([single])` ≡ current behaviour |

---

## Files Changed / Created

| File | Change |
|---|---|
| `src/trainer/vgg_loss.py` | Add `compute_mean_style_grams()` |
| `src/trainer/style_trainer.py` | Use mean Gram precomputation |
| `scripts/kaggle_training_helper.py` | `style_images: list[Path]`, `style_images_dir` |
| `scripts/kaggle_multi_pic_trainer.ipynb` | **New** |
| `src/trainer/style_analyser.py` | Add `analyse_style_set()`, `hist_overlap_matrix()` |
| `tests/trainer/test_multi_pic_gram.py` | **New** |
| `scripts/index.md` | Add row for new notebook |
