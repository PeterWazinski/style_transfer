# Implementation Plan — Style Chain Rename & BatchStyler Enhancements

**Status**: Approved — ready for implementation  
**Date**: 2026-05-02

---

## Scope Overview

| # | Area | Summary |
|---|------|---------|
| 1 | Stylist app | Rename "Replay Log" → "Style Chain"; app icon in taskbar; upfront chain validation; update Help text |
| 2 | BatchStyler | Rename CLI options; improved `-h` help with style listing + examples |
| 3 | BatchStyler (new) | Add `--style-chain-overview` command: batch apply all chains in a dir, produce A4 portrait PDF |

---

## 1. Stylist App Changes

### 1.1 Terminology rename — "Replay Log" → "Style Chain"

| Location | Current text | New text |
|----------|-------------|----------|
| `main_window.py` — menu action | `"Replay Log to Clipboard"` | `"Style Chain to Clipboard"` |
| `main_window.py` — menu action | `"Load Replay Log…"` | `"Apply Style Chain…"` |
| `main_window.py` — `QMessageBox` titles (×4) | `"Load Replay Log"` | `"Apply Style Chain"` |
| `main_window.py` — status bar message | `"Replay log applied: …"` | `"Style chain applied: …"` |
| `main_window.py` — `QFileDialog` title | `"Load Replay Log"` | `"Apply Style Chain"` |
| `main_window.py` — action variable names | `_replay_copy_action`, `_replay_load_action` | `_chain_copy_action`, `_chain_apply_action` |
| `main_window.py` — method name | `_load_and_apply_replay_log()` | `_apply_style_chain()` |
| `main_window.py` — `_format_replay_log()` / `_copy_replay_log_to_clipboard()` | keep internal names unless confusing | rename to `_format_style_chain()` / `_copy_style_chain_to_clipboard()` |

> **Note**: The YAML file format (`version: 1`, field names `steps`, `style`, `strength`, `tile_size`, `tile_overlap`) is **not** changed — it is a serialisation format, not user-visible text. The module `src/core/replay_schema.py` **is** renamed to `src/core/style_chain_schema.py` (see §4 — Rename `replay_schema.py`).

---

### 1.2 App icon shown in the Windows taskbar

**Current behaviour**: `app.setWindowIcon(_make_palette_icon())` sets the icon on the `QApplication` object in `app.py`, which should propagate to the taskbar. If the Python disk icon appears instead, the issue is that PyInstaller needs an `.ico` file embedded in the EXE.

**Plan**:
1. Generate `assets/palette.ico` once (multi-size: 16, 32, 48, 64, 256 px) using a one-off helper script and **commit it as a static asset**. The icon is unlikely to change, and a static file keeps `compile.ps1` simple.
2. Add `icon='assets/palette.ico'` to **both** `EXE()` blocks in `style_transfer.spec`.
3. Re-call `self.setWindowIcon(_make_palette_icon())` in `MainWindow.__init__` — this ensures the window-level icon is also set for the taskbar button.
4. Rebuild with `compile.ps1`.

> The programmatic icon generation code in `app.py` is kept as-is. `assets/palette.ico` is the only new committed binary.

---

### 1.3 Upfront style-chain validation before execution

**Current behaviour**: Styles are checked step-by-step during execution. An unknown style name or missing model causes an error mid-chain, leaving the canvas in a partially-applied state.

**New behaviour**: Before starting to apply any step, perform a **pre-flight check** on the loaded `ReplayLog`:
- For each `step.style`, call `_resolve_style_id_by_name(step.style)`.
- If *any* name is not found → collect **all** unknown names → show a single `QMessageBox.critical()` listing them → abort (do **not** start applying).
- Strength range is already guaranteed by Pydantic (`ge=1, le=300`) — no additional check needed in the UI layer.

**Implementation location**: first block of `_apply_style_chain()` (renamed from `_load_and_apply_replay_log`), right after `load_replay_log()` succeeds and before any canvas mutation.

**BatchStyler** — same logic in `cmd_apply_style_chain()` (renamed from `cmd_replay()`):
- Load and validate YAML with `load_style_chain()` (renamed from `load_replay_log()`).
- Iterate all steps up front; resolve each style name against the catalog.
- Collect all unknown names; if any → `sys.exit(...)` with a list of unknown names before any inference starts.

---

### 1.4 Update Help → "How to Use" text

Add a new section to the existing HTML in `_show_how_to_use()`:

```
Style Chains
Copy the current style chain to the clipboard via File → Style Chain to Clipboard.
The YAML can be saved as a .yml file and later re-applied via File → Apply Style Chain…,
or processed in batch via BatchStyler.exe --apply-style-chain.
```

Also update any occurrence of "Replay Log" in the dialog text to "Style Chain".

---

### 1.5 Tests to update / add

| Test file | Changes needed |
|-----------|---------------|
| `tests/stylist/test_main_window_replay.py` | Rename test class/methods; update menu action names, dialog titles, status messages to new strings; add test for pre-flight validation (unknown style shows error, no steps applied) |
| `tests/core/test_replay_schema.py` | No changes needed (schema unchanged) |

Commit message: `feat(stylist): rename Replay Log→Style Chain; upfront chain validation; taskbar icon`

---

## 2. BatchStyler CLI Renames

### 2.1 Option renames

| Old option | New option | Notes |
|-----------|-----------|-------|
| `--pdfoverview` | `--style-overview` | Mutually exclusive group |
| `--replay FILE` | `--apply-style-chain FILE` | Mutually exclusive group |
| `--style NAME` | `--apply-style NAME` | Only valid with `--style-overview` |
| `--strength-override PCT` | `--strength-scale PCT` | Renamed for clarity; same range 1–300; applies to all modes |

### 2.2 Output filename change

| Mode | Old filename | New filename |
|------|-------------|-------------|
| `--style-overview` | `<stem>_thumbnails.pdf` | `<stem>_style_overview.pdf` |
| `--apply-style-chain` | unchanged | unchanged |

### 2.3 Internal renames

| Old name | New name |
|---------|---------|
| `cmd_pdfoverview()` | `cmd_style_overview()` |
| `cmd_replay()` | `cmd_apply_style_chain()` |
| argparse `--pdfoverview` dest | `args.style_overview` |
| argparse `--replay` dest | `args.apply_style_chain` |
| argparse `--strength-override` dest | `args.strength_scale` |

### 2.4 Improved `-h` help output

When `-h` is passed (or no mode specified), the help message should:

1. **List all available styles** by reading `styles/catalog.json` and printing their display names — inserted after the mode descriptions.
2. **Extended examples section** replacing the current short list:

```
Examples:
  BatchStyler.exe --style-overview portrait.jpg
      → portrait_style_overview.pdf  (all styles, 100/150/200 %)

  BatchStyler.exe --style-overview portrait.jpg --apply-style "Candy"
      → portrait_style_overview.pdf  (only Candy)

  BatchStyler.exe --apply-style-chain my_chain.yml portrait.jpg
      → portrait_my_chain.jpg

  BatchStyler.exe --apply-style-chain my_chain.yml portrait.jpg --strength-override 80
      → portrait_my_chain_80.jpg

  BatchStyler.exe --apply-style-chain my_chain.yml portrait.jpg --outdir C:\output
      → C:\output\portrait_my_chain.jpg

  BatchStyler.exe --style-chain-overview C:\chains portrait.jpg
      → portrait_chains_overview.pdf
```

**Implementation**: extend `_USAGE` string and add a helper `_list_styles_for_help()` that reads the catalog at runtime (gracefully returns `"(catalog not found)"` if missing).

### 2.5 Tests to update / add

| Test file | Changes needed |
|-----------|---------------|
| `tests/scripts/test_batch_styler.py` | Update all references to old option/function names; update expected output filenames (`_thumbnails.pdf` → `_style_overview.pdf`); rename `strength_override` → `strength_scale` in all call sites; add test verifying `--apply-style` is rejected with `--apply-style-chain`; add test for help style listing |

Commit message: `feat(BatchStyler): rename CLI options; add style list to -h; update output filenames`

---

## 3. New BatchStyler Command — `--style-chain-overview`

### 3.1 Purpose

Apply every `.yml` / `.yaml` file found in a given directory to a single input image, collect all results, and produce a single **A4 portrait** PDF summary.

### 3.2 CLI syntax

```
BatchStyler.exe --style-chain-overview <chain-dir> <image> [options]
```

- `<chain-dir>`: path to a directory containing `.yml`/`.yaml` style-chain files.
- `<image>`: source image (JPEG or PNG).
- Common options (`--tile-size`, `--overlap`, `--float16`, `--outdir`) all apply.
- `--strength-scale N` scales every individual step strength proportionally: `effective_strength = min(300, round(step.strength * N / 100))`. The result is **capped at 300 %** to stay within the valid range. Example: `--strength-scale 50` with steps at 100 % and 200 % → effective strengths 50 % and 100 % respectively. Example of capping: `--strength-scale 200` with a step at 200 % → capped to 300 % (not 400 %).

Error conditions:
- `<chain-dir>` does not exist → `sys.exit()` with message.
- `<chain-dir>` exists but contains no `.yml`/`.yaml` files → `sys.exit()` with message.
- A chain file fails schema validation → skip that file, print a warning, continue.
- A style referenced in a chain is not found in the catalog → skip that chain entirely, **print a warning message** (do not insert an empty/placeholder page in the PDF), continue.

### 3.3 PDF layout

- Format: A4 portrait (210 × 297 mm), 150 DPI → 1240 × 1754 px.
- **2 pictures per page** (1 column × 2 rows).
- **Page 1, slot 1**: original photo with caption `"Original"`.
- **Page 1, slot 2** onwards: each chain result in alphabetical order of chain filename.
- Caption below each result: chain filename stem (e.g. `"my_chain"`).
- Images are aspect-ratio-preserved, centred in their cell.
- Caption height: `LABEL_H` (same constant as existing layout).

### 3.4 Output filename

```
<image-stem>_<chain-dir-name>_overview.pdf
```

Stored in `--outdir DIR` if specified, otherwise next to the source image.

Example: image `portrait.jpg`, chain dir `C:\chains` → `portrait_chains_overview.pdf`.

### 3.5 New constants (add alongside existing layout constants)

```python
A4P_W: int = int(210 / 25.4 * DPI)   # ≈ 1240 px
A4P_H: int = int(297 / 25.4 * DPI)   # ≈ 1754 px
CHAIN_ROWS: int = 2
CHAIN_COLS: int = 1
```

### 3.6 New function `cmd_style_chain_overview()`

```
cmd_style_chain_overview(
    image_path: Path,
    chain_dir: Path,
    tile_size: int | None,
    overlap: int | None,
    use_float16: bool,
    strength_scale: int | None,
    out_dir: Path | None,
) -> None
```

Steps:
1. Collect and sort `.yml`/`.yaml` files from `chain_dir`.
2. Load catalog.
3. For each chain file:
   a. `load_replay_log()` — skip on `ValueError`, print warning.
   b. Pre-flight: check all step styles exist in catalog — skip chain on failure, print warning.
   c. Apply all steps in sequence using `StyleTransferEngine` (same logic as `cmd_apply_style_chain()`).
   d. Append `(chain_stem, result_image)` to results list.
4. Build cell list: `[("Original", source)] + results`.
5. Lay out pages (2 cells per page, portrait A4).
6. Save PDF.

### 3.7 Helper to extract shared step-application logic

Currently `cmd_apply_style_chain()` contains the inference loop inline. To avoid duplication in `cmd_style_chain_overview()`, extract a private helper:

```python
def _apply_chain_to_image(
    source: Image.Image,
    chain: StyleChain,          # renamed from ReplayLog
    styles: list[dict],
    engine: StyleTransferEngine,
    tile_size: int,
    overlap: int,
    use_float16: bool,
    strength_scale: int | None,  # renamed from strength_override; scales proportionally
) -> Image.Image:
    ...
```

**Strength-scale formula** (applied per step):
```python
effective_pct = min(300, round(step.strength * strength_scale / 100)) if strength_scale is not None else step.strength
strength = effective_pct / 100.0
```

> **Cap rule**: the result is clamped to 300 before converting to a float factor. This prevents `strength_scale=200` applied to a step already at `200 %` from producing an illegal `400 %` value.

This helper:
- Takes an already-validated `ReplayLog` and a pre-built catalog styles list.
- Assumes style names have already been pre-flight checked.
- Returns the final styled PIL image.

Both `cmd_apply_style_chain()` and `cmd_style_chain_overview()` call this helper.

### 3.8 Main argument parser changes

Add to the mutually exclusive group:

```python
mode_group.add_argument(
    "--style-chain-overview", type=Path, metavar="CHAIN_DIR",
    help="Apply all .yml chains in CHAIN_DIR to the image and produce a portrait A4 PDF.",
)
```

Replace `--strength-override` with `--strength-scale`:

```python
parser.add_argument(
    "--strength-scale", type=int, default=None, metavar="PCT",
    help="Scale each step's strength by this percentage (1–300). "
         "E.g. --strength-scale 50 turns 100%%→50%%, 200%%→100%%. "
         "Result is capped at 300%% (e.g. --strength-scale 200 on a 200%% step → 300%%).",
)
```

### 3.9 Tests to add (`tests/scripts/test_batch_styler.py`)

| Test | Verifies |
|------|---------|
| `test_chain_overview_applies_all_chains` | All `.yml` files in dir are applied; one result per chain |
| `test_chain_overview_pdf_written` | Output PDF created with correct name |
| `test_chain_overview_outdir` | Respects `--outdir` |
| `test_chain_overview_empty_dir_exits` | `sys.exit` when no `.yml` files found |
| `test_chain_overview_invalid_chain_skipped` | Chain with bad schema → warning printed, rest continue, no placeholder in PDF |
| `test_chain_overview_unknown_style_skipped` | Chain referencing unknown style → warning printed, chain skipped, no placeholder in PDF |
| `test_chain_overview_strength_scale` | `--strength-scale 50` halves each step's strength proportionally |

Commit message: `feat(BatchStyler): add --style-chain-overview command with portrait A4 PDF output`

---

## Cross-cutting items

### File changes summary

| File | Type of change |
|------|---------------|
| `src/core/replay_schema.py` → `src/core/style_chain_schema.py` | **Rename** — update all imports in `main_window.py`, `batch_styler.py`, and all test files |
| `src/stylist/main_window.py` | Rename actions, methods, dialog titles; add pre-flight validation block |
| `src/stylist/app.py` | Add `setWindowIcon` call in `MainWindow.__init__` |
| `assets/palette.ico` | **New static asset** — multi-size ICO for taskbar + EXE embedding |
| `style_transfer.spec` | Add `icon='assets/palette.ico'` to both `EXE()` blocks |
| `scripts/batch_styler.py` | Rename options/functions incl. `--strength-override`→`--strength-scale`; new output filename; new `--style-chain-overview` command; extract `_apply_chain_to_image()` helper; update `_USAGE` |
| `tests/stylist/test_main_window_replay.py` | Update names and add pre-flight test |
| `tests/scripts/test_batch_styler.py` | Update names/filenames; replace `strength_override` with `strength_scale`; add 7 new tests for chain overview |

### Commit sequence (suggested)

1. `chore: rename replay_schema.py → style_chain_schema.py; update all imports`
2. `feat(stylist): rename Replay Log → Style Chain; upfront chain validation; taskbar icon`
3. `feat(BatchStyler): rename CLI options incl. --strength-override→--strength-scale; style list in -h; update output filenames`
4. `feat(BatchStyler): add --style-chain-overview command`

### Decisions recorded

| # | Question | Decision |
|---|----------|----------|
| 1 | Icon asset workflow | **Static asset** — commit `assets/palette.ico` to the repo; simplifies `compile.ps1`. |
| 2 | Strength-override semantics | **Proportional scaling** — `--strength-scale 50` with steps at 100 % and 200 % → 50 % and 100 %. Option renamed to `--strength-scale PCT` (integer 1–300) for clarity. |
| 3 | Chain overview partial failures | **Skip + warning message** — no placeholder page in the PDF; a clearly visible warning is printed to stdout so the user knows which chains were skipped. |
| 4 | `replay_schema.py` rename | **Rename to `style_chain_schema.py`** — done as a dedicated first commit to keep the diff clean. |
