"""Patch scripts/kaggle_style_training.ipynb:
- Cell 9 markdown: 200 batches → 2000 batches, ~4 min → ~10 min
- Cell 10 code: activate cfg.smoke_batches = 2000 (was commented-out 500)
"""
import json
import pathlib

NB = pathlib.Path("scripts/kaggle_style_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# ── Cell 9: markdown ──────────────────────────────────────────────────────────
src9 = "".join(nb["cells"][9]["source"])
assert "200 batches" in src9, "Expected '200 batches' in cell 9"
assert "~4 min" in src9, "Expected '~4 min' in cell 9"
src9 = src9.replace("200 batches", "2000 batches")
src9 = src9.replace("(~4 min on T4)", "(~10 min on T4)")

lines9 = src9.split("\n")
nb["cells"][9]["source"] = [
    l + ("\n" if i < len(lines9) - 1 else "")
    for i, l in enumerate(lines9)
]

# ── Cell 10: code ─────────────────────────────────────────────────────────────
NEW_CELL10 = (
    "cfg.smoke_batches = 2000  "
    "# 2000 validated on T4 (mean_diff=57 on candy); lower to 500 for a quicker check\n"
    "# cfg.content_weight = 5e4  # reduce for stronger style influence\n"
    "\n"
    "results = runner.run_smoke_test()"
)
lines10 = NEW_CELL10.split("\n")
nb["cells"][10]["source"] = [
    l + ("\n" if i < len(lines10) - 1 else "")
    for i, l in enumerate(lines10)
]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
json.loads(NB.read_text(encoding="utf-8"))
print("Patched and validated OK.")
