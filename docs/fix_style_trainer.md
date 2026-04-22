# Fix Style Trainer — Implementation Roadmap

**Status:** Phase 1 (P1-1, P1-2, P1-3) complete — P1-4 and Phase 2 in progress — Phase 5 design decisions recorded  
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

- [ ] **P5-1** Create `src/trainer/style_analyser.py` — `analyse_style(path) -> dict` and `recommend_weights(metrics) -> tuple[float, float, str]`. Extract from both notebooks.
- [ ] **P5-2** Create `scripts/kaggle_training_helper.py` — `TrainingConfig` dataclass (with `save()`/`load()` JSON methods) + `KaggleStyleRunner` class + argparse `main()` CLI
- [ ] **P5-3** Refactor `docs/kaggle_style_training.ipynb` Cell 4 — remove `_analyse_style` / `_recommend`, replace with `runner.analyse_style()`
- [ ] **P5-4** Collapse Cell 5 (smoke test) to `runner.run_smoke_test()` call
- [ ] **P5-5** Collapse Cell 6 (full train) to `runner.run_full_training()` call (saves `config.json`)
- [ ] **P5-6** Collapse Cell 7 (preview + package) to `runner.package_output()` call
- [ ] **P5-7** Collapse Cell 8 (resume) to `runner.resume_training()` call (reads `config.json` automatically)
- [ ] **P5-8** Update `docs/style_analysis.ipynb` to import `analyse_style` / `recommend_weights` from `src.trainer.style_analyser` instead of defining them inline
- [ ] **P5-9** Add unit tests for `style_analyser.py` functions and `KaggleStyleRunner.analyse_style()` (mock `StyleTrainer` for smoke-test test)

### Open questions for review

1. **Location of helper**: `scripts/` (runs on Kaggle too) vs `src/kaggle/` (installable) — recommend `scripts/` since Kaggle clones the repo anyway.
2. **Shared analyser**: duplicate `_analyse_style` logic exists in both notebooks. Should it move to `src/trainer/style_analyser.py` so both notebooks import it? Recommend yes.
3. **Config persistence**: should `TrainingConfig` be saved as `config.json` next to the output model so resume is automatic? Recommend yes.

---

## Recommended phase sequence

```
P1-4  →  P2  →  P3 (Kaggle validation)  →  P5 (refactor)  →  P4 (close out)
```

**Rationale:**

| Step | Why before P5 / not after |
|---|---|
| P1-4 first | Tiny fix (1 line in Kaggle notebook). Gets the notebook correct before any validation. |
| P2 first | Fixes `style_analysis.ipynb` signal test so local pre-checks are trustworthy. |
| P3 before P5 | Validates the bug fixes on real Kaggle GPU. Refactoring on *proven* working code is much safer. If P3 fails after P5 it is harder to tell whether the refactor introduced a regression. |
| P5 before P4 | Refactor produces the clean codebase worth tagging. P4 (git tag, model regeneration) is the final ceremony on polished code. |

**Do not do P5 before P3.** A structural refactor moves ~130 lines into a new class. If the Kaggle smoke test fails after that it becomes ambiguous whether the training fix or the refactor broke it.

---

## Out of scope (v1 fix)

- Gram matrix normalization (already correct: `/ c*h*w`)
- Adding extra VGG layers (e.g. relu5_3)
- Replacing the COCO dataset pipeline
- Multi-style blending