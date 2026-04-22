# Fix Style Trainer — Implementation Roadmap

**Status:** Phase 1 (P1-1, P1-2, P1-3) complete — P1-4 and Phase 2 in progress  
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
- [ ] **P1-4** `docs/kaggle_style_training.ipynb` — Update `STYLE_WEIGHT` constant to `1e10`. Confirm epoch count gives ≥ 40,000 images.

---

## Phase 2 — Fix the analysis and smoke-test notebook

- [ ] **P2-1** `_smoke_sync()` — Replace `_compute_signal_frac(sw_base, ...)` with a fixed `SIGNAL_TEST_SW = 1e10` as baseline.
- [ ] **P2-2** Auto-calibration cap — Remove implicit upper cap on `sw_final` so it can scale to `1e10`.
- [ ] **P2-3** `IntSlider` default — Change from `100` → `500` batches. Add UI note that < 300 CPU batches is unreliable.
- [ ] **P2-4** Smoke-test output — Add note that definitive validation needs a 2000-batch Kaggle GPU run (~10 min).

---

## Phase 3 — Validate before full 6-hour Kaggle run  *(manual)*

- [ ] **P3-1** Run 2000-batch Kaggle smoke test: SW=1e10, CW=1e5, size=256, batch=4 on `candy.jpg`. Expected: colour_shift > 0.05.
- [ ] **P3-2** Colour shift confirmed → launch full training (2 epochs ≈ 166k images).
- [ ] **P3-3** Still < 0.05 → switch content layer to `relu2_2` (P1-2) and re-test.
- [ ] **P3-4** Apply trained ONNX to 3 real photos — verify style visible to naked eye.

---

## Phase 4 — Commit, regenerate models, close out

- [ ] **P4-1** Commit: `fix: match yakhyo style_weight 1e10, relu2_2 content layer, remove training clamp`
- [ ] **P4-2** Run `python scripts/setup_models.py` for each new style to regenerate `model.onnx` and `preview.jpg`.
- [ ] **P4-3** Remove diagnostic tag `style-trainer-issues` once all phases pass.
- [ ] **P4-4** Add `fix-complete` git tag after successful visual validation.

---

## Verification checklist

- [ ] Signature check: `style_weight=1e10` in `StyleTrainer.train`
- [ ] Cell 9 on `candy.jpg` at SW=1e10 → signal fraction ≥ 35%
- [ ] Kaggle 2000-batch smoke test → colour_shift > 0.05
- [ ] Full run ONNX visibly stylises 3 real photos
- [ ] `python -m pytest tests/ -k "not takes_long"` all pass

---

---

## Phase 5 — Refactor `docs/kaggle_style_training.ipynb` into cockpit + helper  *(proposal for review)*

### Problem

The Kaggle notebook has grown to 8 code cells (~12,600 chars) embedding substantial backend logic:
- `_analyse_style()` + `_recommend()` — texture analysis (21 lines)
- `_smoke_cb()` + full smoke-test pipeline — train, ONNX-export, inference, scoring, verdict (50+ lines)
- Full-training subprocess builder (30 lines)
- Preview display + zip packaging (25 lines)
- Resume-from-checkpoint logic (35 lines)

This makes the notebook hard to test locally and hard to maintain. None of it can be run from the command line without copy-pasting.

### Proposed target structure

```
scripts/
  kaggle_training_helper.py   ← new: all backend logic + argparse CLI
docs/
  kaggle_style_training.ipynb ← renamed cockpit (thin cells, imports only)
```

### `scripts/kaggle_training_helper.py` — class + CLI design

```python
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

class KaggleStyleRunner:
    def __init__(self, cfg: TrainingConfig): ...
    def verify_environment(self) -> None     # GPU, COCO, internet checks
    def analyse_style(self) -> dict          # texture metrics + weight recommendation
    def run_smoke_test(self) -> dict         # returns {mean_diff, color_shift, verdict}
    def run_full_training(self) -> None      # spawns main_style_trainer.py subprocess
    def resume_training(self) -> None        # finds latest .ckpt_*.pth, resumes
    def package_output(self) -> pathlib.Path # copy to /kaggle/output/, create zip
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
| Duplicate logic with style_analysis.ipynb | Yes (`_analyse_style`, `_recommend`) | No (shared import) |
| CLI smoke test for local dev | No | `python scripts/kaggle_training_helper.py smoke --style ...` |

### Files to create/modify

- [ ] **P5-1** Create `scripts/kaggle_training_helper.py` — `TrainingConfig` dataclass + `KaggleStyleRunner` class + argparse `main()` CLI
- [ ] **P5-2** Delete `_analyse_style()` / `_recommend()` from `docs/kaggle_style_training.ipynb` Cell 4 — replace with `runner.analyse_style()`
- [ ] **P5-3** Collapse Cell 5 (smoke test) to `runner.run_smoke_test()` call
- [ ] **P5-4** Collapse Cell 6 (full train) to `runner.run_full_training()` call
- [ ] **P5-5** Collapse Cell 7 (preview + package) to `runner.package_output()` call
- [ ] **P5-6** Collapse Cell 8 (resume) to `runner.resume_training()` call
- [ ] **P5-7** Move duplicate `_analyse_style` / `_recommend` from `docs/style_analysis.ipynb` to import from `kaggle_training_helper` (or a shared `src/trainer/style_analyser.py`)
- [ ] **P5-8** Add unit tests for `KaggleStyleRunner.analyse_style()` and `run_smoke_test()` (mock `StyleTrainer`)

### Open questions for review

1. **Location of helper**: `scripts/` (runs on Kaggle too) vs `src/kaggle/` (installable) — recommend `scripts/` since Kaggle clones the repo anyway.
2. **Shared analyser**: duplicate `_analyse_style` logic exists in both notebooks. Should it move to `src/trainer/style_analyser.py` so both notebooks import it? Recommend yes.
3. **Config persistence**: should `TrainingConfig` be saved as `config.json` next to the output model so resume is automatic? Recommend yes.

---

## Out of scope (v1 fix)

- Gram matrix normalization (already correct: `/ c*h*w`)
- Adding extra VGG layers (e.g. relu5_3)
- Replacing the COCO dataset pipeline
- Multi-style blending