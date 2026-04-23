"""Build scripts/kaggle_style_training.ipynb (cockpit) from the OLD reference notebook.

Applies P5-3 through P5-7:
  P5-3: Cell index 8  — replace _analyse_style/_recommend with runner.analyse_style()
  P5-4: Cell index 10 — collapse smoke test to runner.run_smoke_test()
  P5-5: Cell index 12 — collapse full train to runner.run_full_training()
  P5-6: Cell index 14 — collapse preview+package to runner.package_output()
  P5-7: Cell index 16 — collapse resume to runner.resume_training()
"""
import json
import pathlib

SRC  = pathlib.Path("docs/kaggle_style_training.OLD.ipynb")
DEST = pathlib.Path("scripts/kaggle_style_training.ipynb")

nb = json.loads(SRC.read_text(encoding="utf-8"))


def _src(code: str) -> list[str]:
    """Convert a multi-line code string to the notebook source list format."""
    lines = code.split("\n")
    result = []
    for i, line in enumerate(lines):
        result.append(line + ("\n" if i < len(lines) - 1 else ""))
    return result


# ---------------------------------------------------------------------------
# P5-3 — Cell 8: configure + analyse (replaces _analyse_style + _recommend)
# ---------------------------------------------------------------------------

CELL_8 = """\
import sys as _sys
_sys.path.insert(0, str(REPO_DIR))
from scripts.kaggle_training_helper import TrainingConfig, KaggleStyleRunner

# ── Edit these three lines to match your job ─────────────────────────────────
STYLE_IMAGE = pathlib.Path("/kaggle/working/style_transfer/sample_images/style-pics/hundertwasser2.jpg")   # ← your uploaded file
STYLE_ID    = "hundertwasser"     # slug used as folder name and catalog ID
STYLE_NAME  = "Hundertwasser"     # display name shown in the gallery
# ─────────────────────────────────────────────────────────────────────────────

assert STYLE_IMAGE.exists(), f"Style image not found: {STYLE_IMAGE}\\nUpload it via the file browser first."

cfg = TrainingConfig(
    style_image=STYLE_IMAGE,
    style_id=STYLE_ID,
    style_name=STYLE_NAME,
    coco_path=COCO_TRAIN,
    style_weight=1e10,   # fixed: matches yakhyo reference (do not lower)
    content_weight=1e5,  # override if desired, e.g. 5e4 for more style influence
)
runner = KaggleStyleRunner(cfg)
runner.analyse_style()\
"""

# ---------------------------------------------------------------------------
# P5-4 — Cell 10: smoke test
# ---------------------------------------------------------------------------

CELL_10 = """\
# Optionally override before running:
# cfg.smoke_batches = 500   # default 200; raise for a stronger early signal
# cfg.content_weight = 5e4  # reduce content weight for stronger style influence

results = runner.run_smoke_test()\
"""

# ---------------------------------------------------------------------------
# P5-5 — Cell 12: full training
# ---------------------------------------------------------------------------

CELL_12 = """\
# Launches full training (~3 h on T4 × 2 for 2 epochs = 166 k images).
# Saves config.json next to model.pth — resume_training() loads it automatically.
runner.run_full_training()\
"""

# ---------------------------------------------------------------------------
# P5-6 — Cell 14: preview + package
# ---------------------------------------------------------------------------

CELL_14 = """\
# Copies output files to /kaggle/output/<style_id>/ and creates a zip.
# Download from the Output tab (right panel) in the Kaggle notebook UI.
runner.package_output()\
"""

# ---------------------------------------------------------------------------
# P5-7 — Cell 16: resume
# ---------------------------------------------------------------------------

CELL_16 = """\
# Resume training from the last checkpoint.
# Reads config.json automatically — no need to re-enter weights.
# STYLE_ID and runner must be configured (run Step 3 first if they are not).
runner.resume_training()\
"""

# ---------------------------------------------------------------------------
# Apply replacements
# ---------------------------------------------------------------------------

replacements = {
    8:  CELL_8,
    10: CELL_10,
    12: CELL_12,
    14: CELL_14,
    16: CELL_16,
}

for idx, code in replacements.items():
    c = nb["cells"][idx]
    assert c["cell_type"] == "code", f"Cell {idx} is not a code cell — check notebook structure"
    c["source"] = _src(code)
    # Clear any stale outputs
    c["outputs"] = []
    c["execution_count"] = None

# ---------------------------------------------------------------------------
# Write cockpit notebook
# ---------------------------------------------------------------------------

DEST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {DEST}")

# Sanity check
parsed = json.loads(DEST.read_text(encoding="utf-8"))
code_cells = [c for c in parsed["cells"] if c["cell_type"] == "code"]
print(f"  {len(parsed['cells'])} cells total, {len(code_cells)} code cells")
print("  JSON: OK")
