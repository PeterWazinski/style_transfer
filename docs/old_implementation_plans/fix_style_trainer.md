# Fix Style Trainer — Implementation Roadmap

**Status:** Phase 1 ✅ — Phase 2 ✅ — Phase 3 ✅ — Phase 4 mostly ✅ — Phase 5 ✅  
**Created:** 2026-04-22  
**Problem:** Newly trained styles show no visible effect on photos. Analysis notebook gives false-positive verdicts for good style images.

---

## Root-cause summary

| # | Severity | Issue | Yakhyo reference | Our code |
|---|----------|-------|-----------------|----------|
| A | **Critical** | Default `style_weight` | `1e10` | `1e8` — 100× too low |
| B | Moderate | Content-loss VGG layer | `relu2_2` | `relu3_3` (more semantic, loses texture signal) |
| C | Moderate | Signal test calibrated at wrong SW | — | `_compute_signal_frac` runs at SW ≤ 1e9; recommends weights still 100× too small |
| D | Moderate | TransformerNet output clamp during training | No clamp | `torch.clamp(0,255)` zeroes gradients for saturated neurons |
| E | Minor | Smoke-test batch count | N/A | 100 CPU batches too few to detect invisible style |
| F | Minor | Kaggle epoch count | 1 epoch = 83k images | If < 40k images trained, model has not converged |

---

## Phase 1 — Match the reference implementation

- [x] **P1-1** `src/trainer/transformer_net.py` — Removed `torch.clamp(out, 0.0, 255.0)` from `TransformerNet.forward()`. Returns raw `out`. Clamp at inference/ONNX-export only.
- [x] **P1-2** `src/trainer/vgg_loss.py` — Content layer changed from `relu3_3` to `relu2_2`: `_CONTENT_LAYER = 8`; `forward()` now uses `out_features[1]` / `con_features[1]`.
- [x] **P1-3** `src/trainer/style_trainer.py` — Default `style_weight` changed `1e8` → `1e10`.
- [x] **P1-4** `docs/kaggle_style_training.ipynb` — `STYLE_WEIGHT` hardcoded to `1e10` (overrides texture-analysis recommendation). CLI example updated to `--style-weight 1e10`. Epoch count = 2 × 83k = 166k images ≥ 40k ✓.

---

## Phase 2 — Fix the analysis and smoke-test notebook

- [x] **P2-1** `_smoke_sync()` — Added `SIGNAL_TEST_SW: float = 1e10` constant; `_compute_signal_frac()` now called with `SIGNAL_TEST_SW` instead of `sw_base` (which was capped at 1e9).
- [x] **P2-2** Auto-calibration cap removed — `sw_final` starts at `SIGNAL_TEST_SW`; scale-up expressions use `SIGNAL_TEST_SW` as base so result is never capped below 1e10.
- [x] **P2-3** `IntSlider` default changed `100` → `500`; max raised to 2000; step 20 → 50. Go/no-go text updated to reference 500 batches.
- [x] **P2-4** CPU diagnostic note added after smoke-test verdict: “CPU smoke test is diagnostic only — for definitive results run 2000 batches on Kaggle GPU (≈10 min on T4).”

---

## Phase 3 — Validate before full 6-hour Kaggle run  *(manual)*

- [x] **P3-1** Run 2000-batch Kaggle smoke test: SW=1e10, CW=1e5, size=256, batch=4 on `candy.jpg`. **mean_diff=57.0** — ✓ GOOD (threshold 20). Colour shift confirmed.
- [x] **P3-2** Colour shift confirmed → launched full training (2 epochs ≈ 166k images). *(training running on Kaggle)*
- [n/a] **P3-3** ~~Still < 0.05 → switch content layer~~ — N/A, smoke test passed with mean_diff=57 (threshold 20).
- [x] **P3-4** ✅ Trained **Hundertwasser** style on Kaggle (2 epochs, SW=1e10, CW=1e5). ONNX added to gallery via `scripts/add_style.ipynb`. Style confirmed visible in the Stylist app.

---

## Phase 4 — Commit, regenerate models, close out

- [x] **P4-1** Fix commits done: `322452a` (P1), `acc97e0` (P2), `82ae50e` (P5).
- [x] **P4-2** ✅ Hundertwasser style added: `model.onnx`, `model.pth`, `preview.jpg` committed in `fbeed1d`.
- [ ] **P4-3** ⏳ Remove diagnostic tag `style-trainer-issues` once all phases pass.
- [x] **P4-4** ✅ New style working end-to-end in Stylist app — visual validation complete.

---

## Verification checklist

- [x] Signature check: `style_weight=1e10` in `StyleTrainer.train` ✅ (P1-3)
- [x] Kaggle 2000-batch smoke test → mean_diff=57.0 ✅ (P3-1, threshold 20)
- [x] ✅ Hundertwasser ONNX visibly stylises photos in the Stylist app (P3-4)
- [ ] ⏳ `python -m pytest tests/ -k "not takes_long"` all pass (run after any further changes)

---

---

## Phase 5 — Refactor `docs/kaggle_style_training.ipynb` into cockpit + helper

### Problem

The Kaggle notebook has grown to 8 code cells (~12,600 chars) embedding substantial backend logic:
- `_analyse_style()` + `_recommend()` — texture analysis (21 lines)
- `_smoke_cb()` + full smoke-test pipeline — train, ONNX-export, inference, scoring, verdict (50+ lines)
- Full-training subprocess builder (30 lines)
- Preview display + zip packaging (25 lines)
- Resume-from-checkpoint logic (35 lines)

This makes the notebook hard to test locally and hard to maintain. None of it can be run from the command line without copy-pasting.

### Design decisions

| # | Decision |
|---|----------|
| Location | `scripts/kaggle_training_helper.py` — runs on Kaggle after repo clone, no install step needed |
| Shared analyser | Move `_analyse_style` / `_recommend` to `src/trainer/style_analyser.py` (importable module). Both notebooks import from there. Zero duplication. |
| Config persistence | `TrainingConfig` saved as `config.json` next to `model.pth` after training. Resume reads config automatically — no need to re-enter weights. |

### Proposed target structure

```
src/trainer/
  style_analyser.py           ← new: _analyse_style, _recommend (shared by all callers)
scripts/
  kaggle_training_helper.py   ← new: TrainingConfig, KaggleStyleRunner, argparse CLI
docs/
  kaggle_style_training.ipynb ← cockpit only (~30 lines of code total)
```

### `scripts/kaggle_training_helper.py` — class + CLI design

```python
from src.trainer.style_analyser import analyse_style, recommend_weights  # shared module

@dataclass
class TrainingConfig:
    style_image:    pathlib.Path
    style_id:       str
    style_name:     str
    coco_path:      pathlib.Path
    style_weight:   float = 1e10
    content_weight: float = 1e5
    epochs:         int   = 2
    batch_size:     int   = 4
    image_size:     int   = 256
    smoke_batches:  int   = 200
    device:         str   = "cuda"

    def save(self, out_dir: pathlib.Path) -> None:   # writes config.json
    @classmethod
    def load(cls, out_dir: pathlib.Path) -> "TrainingConfig":  # reads config.json

class KaggleStyleRunner:
    def __init__(self, cfg: TrainingConfig): ...
    def verify_environment(self) -> None      # GPU, COCO, internet checks
    def analyse_style(self) -> dict           # delegates to src.trainer.style_analyser
    def run_smoke_test(self) -> dict          # returns {mean_diff, color_shift, verdict}
    def run_full_training(self) -> None       # spawns main_style_trainer.py subprocess; saves config.json
    def resume_training(self) -> None         # loads config.json, finds latest .ckpt_*.pth
    def package_output(self) -> pathlib.Path  # copy to /kaggle/output/, create zip
```

CLI entry point (each command calls one method):
```bash
python scripts/kaggle_training_helper.py verify
python scripts/kaggle_training_helper.py analyse  --style /kaggle/working/my_style.jpg
python scripts/kaggle_training_helper.py smoke    --style ... --id my_style --coco ...
python scripts/kaggle_training_helper.py train    --style ... --id my_style --coco ...
python scripts/kaggle_training_helper.py resume   --id my_style
python scripts/kaggle_training_helper.py package  --id my_style
```

### Refactored notebook — cockpit only

Each step becomes 2–4 lines. Example:
```python
# Step 3 — configure job
from scripts.kaggle_training_helper import TrainingConfig, KaggleStyleRunner
cfg = TrainingConfig(
    style_image=pathlib.Path("/kaggle/working/my_style.jpg"),
    style_id="my_style",
    style_name="My Style",
    coco_path=COCO_TRAIN,
)
runner = KaggleStyleRunner(cfg)
runner.analyse_style()
```

### Estimated impact

| Metric | Before | After |
|---|---|---||
| Notebook code lines | ~130 | ~30 |
| Testable without notebook | No | Yes (`pytest` + CLI) |
| Duplicate `_analyse_style` logic | Yes (2 notebooks) | No (`src/trainer/style_analyser.py`) |
| Resume requires re-entering weights | Yes | No (`config.json` auto-loaded) |
| CLI smoke test for local dev | No | `python scripts/kaggle_training_helper.py smoke --style ...` |

### Files to create/modify

- [x] **P5-1** Create `src/trainer/style_analyser.py` — `analyse_style(path) -> dict` and `recommend_weights(metrics) -> tuple[float, float, str]`. Extracted from both notebooks.
- [x] **P5-2** Create `scripts/kaggle_training_helper.py` — `TrainingConfig` dataclass (with `save()`/`load()` JSON methods) + `KaggleStyleRunner` class + argparse `main()` CLI.
- [x] **P5-3** Created `scripts/kaggle_style_training.ipynb` (cockpit) — Cell 8: `_analyse_style`/`_recommend` replaced with `runner.analyse_style()`. Reference copy kept as `docs/kaggle_style_training.OLD.ipynb`.
- [x] **P5-4** Cockpit Cell 10 — smoke test collapsed to `runner.run_smoke_test()`.
- [x] **P5-5** Cockpit Cell 12 — full training collapsed to `runner.run_full_training()` (saves `config.json`).
- [x] **P5-6** Cockpit Cell 14 — preview+package collapsed to `runner.package_output()`.
- [x] **P5-7** Cockpit Cell 16 — resume collapsed to `runner.resume_training()` (loads `config.json` automatically).
- [x] **P5-8** Moved `docs/style_analysis.ipynb` → `scripts/style_analysis.ipynb`. Imports `analyse_style` / `recommend_weights` from `src.trainer.style_analyser` (no inline duplicates).
- [x] **P5-9** Add unit tests: `tests/trainer/test_style_analyser.py` — 16 tests covering all branches of `analyse_style` + `recommend_weights` + `KaggleStyleRunner.analyse_style()` + `TrainingConfig` round-trip. All pass.

### Decisions taken

| Decision | Outcome |
|---|---|
| Helper location | `scripts/kaggle_training_helper.py` ✅ |
| Shared analyser | `src/trainer/style_analyser.py` — imported by both notebooks ✅ |
| Config persistence | `TrainingConfig.save()/load()` writes `config.json` next to `model.pth`; resume is automatic ✅ |

---

## Next actions

1. **P4-3** — remove git tag `style-trainer-issues`: `git tag -d style-trainer-issues`.
2. **Run full test suite** — `python -m pytest tests/ -k "not takes_long"` (confirm all pass).
3. **Train more styles** — use `scripts/kaggle_style_training.ipynb` for new style images; add to gallery via `scripts/add_style.ipynb`.

---

## Out of scope (v1 fix)

- Gram matrix normalization (already correct: `/ c*h*w`)
- Adding extra VGG layers (e.g. relu5_3)
- Replacing the COCO dataset pipeline
- Multi-style blending