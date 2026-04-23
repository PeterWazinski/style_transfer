"""Patch docs/style_analysis.ipynb:
- Cell 2: replace inline _analyse_style_image + _recommend definitions with
  imports from src.trainer.style_analyser
- Cell 8: remove the duplicate SIGNAL_TEST_SW line
"""
import json
import pathlib
import sys

NB_PATH = pathlib.Path("docs/style_analysis.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))


# ── Cell 2: replace inline function definitions ───────────────────────────────
NEW_CELL2_SOURCE = (
    "import sys as _sys\n"
    "_sys.path.insert(0, str(pathlib.Path(\"..\").resolve()))\n"
    "from src.trainer.style_analyser import (\n"
    "    analyse_style as _analyse_style_image,\n"
    "    recommend_weights as _recommend,\n"
    ")\n"
    "\n"
    "results = [_analyse_style_image(p) for p in images]\n"
    "print(f\"Analysed {len(results)} image(s).\")\n"
)

# Find Cell 2 by looking for the cell that defines _analyse_style_image
cell2_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "def _analyse_style_image" in src:
        cell2_idx = i
        break

if cell2_idx is None:
    print("ERROR: could not find Cell 2 (def _analyse_style_image not found)", file=sys.stderr)
    sys.exit(1)

# Split into lines (notebook source is stored as a list of line strings)
nb["cells"][cell2_idx]["source"] = [
    line + ("\n" if not line.endswith("\n") else "")
    for line in NEW_CELL2_SOURCE.splitlines()
]
# Fix: last line should not have trailing newline in notebook convention
if nb["cells"][cell2_idx]["source"]:
    nb["cells"][cell2_idx]["source"][-1] = nb["cells"][cell2_idx]["source"][-1].rstrip("\n")

print(f"  Patched cell {cell2_idx}: replaced inline function definitions with imports")


# ── Cell 8: remove duplicate SIGNAL_TEST_SW line ─────────────────────────────
SIGNAL_LINE = "SIGNAL_TEST_SW: float = 1e10  # always test at training weight — not texture-analysis cap (max 1e9)"

for i, c in enumerate(nb["cells"]):
    src_lines = c["source"]
    # Count how many times the line appears
    matching = [j for j, s in enumerate(src_lines) if SIGNAL_LINE in s]
    if len(matching) > 1:
        # Remove all but the first occurrence
        for j in reversed(matching[1:]):
            del src_lines[j]
        print(f"  Patched cell {i}: removed {len(matching) - 1} duplicate SIGNAL_TEST_SW line(s)")


# ── Write back ────────────────────────────────────────────────────────────────
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("  Wrote patched notebook.")

# Verify it parses cleanly
json.loads(NB_PATH.read_text(encoding="utf-8"))
print("  JSON validation: OK")
