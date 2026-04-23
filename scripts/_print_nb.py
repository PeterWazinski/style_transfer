"""Utility: print all code cells from a notebook."""
import json, sys, pathlib

path = pathlib.Path(sys.argv[1])
nb = json.loads(path.read_text(encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    ct = c["cell_type"]
    src = "".join(c["source"])
    print(f"=== CELL {i} ({ct}) ===")
    print(src)
    print()
