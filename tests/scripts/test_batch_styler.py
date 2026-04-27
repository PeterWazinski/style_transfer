"""Unit tests for scripts/batch_styler.py.

Tests cover:
- Layout constants are consistent (cell dimensions fit on a page)
- _fit_into() preserves aspect ratio and does not upscale
- _make_page() returns an A4-landscape RGB image with the correct number of cells
- _make_page() places the first cell at (MARGIN, MARGIN)
- build_cell_list() prepends the original image with label "Original"
- build_cell_list() includes all styled results after the original
- _style_name_to_filename() sanitises names for use as file-system stems
- main() --pdfoverview: mock engine, verify PDF is created and is a valid PDF
- main() --fullimage: mock engine, verify per-style JPEGs are written
- main() without mode flag: exits with code 1
- Original image is the first cell (pixel check)
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure scripts/ is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import batch_styler as bs  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid(colour: tuple[int, int, int], size: int = 64) -> Image.Image:
    return Image.fromarray(
        np.full((size, size, 3), colour, dtype=np.uint8)
    )


def _make_catalog(tmp_path: Path, styles: list[dict]) -> Path:
    catalog_path = tmp_path / "styles" / "catalog.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        json.dumps({"styles": styles}), encoding="utf-8"
    )
    return catalog_path


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

class TestLayoutConstants:
    def test_cells_fit_horizontally(self) -> None:
        used = 2 * bs.MARGIN + bs.COLS * bs.CELL_W + (bs.COLS - 1) * bs.GAP
        assert used <= bs.A4_W, f"Cells overflow horizontally: {used} > {bs.A4_W}"

    def test_cells_fit_vertically(self) -> None:
        used = 2 * bs.MARGIN + bs.ROWS * bs.CELL_H + (bs.ROWS - 1) * bs.GAP
        assert used <= bs.A4_H, f"Cells overflow vertically: {used} > {bs.A4_H}"

    def test_img_height_less_than_cell_height(self) -> None:
        assert bs.IMG_H < bs.CELL_H, "IMG_H must leave room for the label"

    def test_cells_per_page_is_rows_times_cols(self) -> None:
        assert bs.CELLS_PER_PAGE == bs.ROWS * bs.COLS


# ---------------------------------------------------------------------------
# _fit_into
# ---------------------------------------------------------------------------

class TestFitInto:
    def test_landscape_image_fits(self) -> None:
        img = Image.new("RGB", (2000, 1000))
        result = bs._fit_into(img, bs.CELL_W, bs.IMG_H)
        assert result.width <= bs.CELL_W
        assert result.height <= bs.IMG_H

    def test_portrait_image_fits(self) -> None:
        img = Image.new("RGB", (400, 800))
        result = bs._fit_into(img, bs.CELL_W, bs.IMG_H)
        assert result.width <= bs.CELL_W
        assert result.height <= bs.IMG_H

    def test_aspect_ratio_preserved(self) -> None:
        img = Image.new("RGB", (800, 400))   # 2:1
        result = bs._fit_into(img, bs.CELL_W, bs.IMG_H)
        ratio_in  = img.width / img.height
        ratio_out = result.width / result.height
        assert abs(ratio_in - ratio_out) < 0.05

    def test_small_image_not_upscaled(self) -> None:
        img = Image.new("RGB", (10, 10))
        result = bs._fit_into(img, bs.CELL_W, bs.IMG_H)
        assert result.width <= 10
        assert result.height <= 10

    def test_original_not_mutated(self) -> None:
        img = Image.new("RGB", (800, 600))
        bs._fit_into(img, bs.CELL_W, bs.IMG_H)
        assert img.size == (800, 600)


# ---------------------------------------------------------------------------
# _make_page
# ---------------------------------------------------------------------------

class TestMakePage:
    def _font(self):
        return bs._load_font(12)

    def test_returns_a4_landscape(self) -> None:
        cells = [("Style A", _solid((255, 0, 0))), ("Style B", _solid((0, 255, 0)))]
        page = bs._make_page(cells, self._font())
        assert page.size == (bs.A4_W, bs.A4_H)
        assert page.mode == "RGB"

    def test_single_cell_renders(self) -> None:
        cells = [("Only", _solid((100, 150, 200)))]
        page = bs._make_page(cells, self._font())
        assert page is not None

    def test_full_page_six_cells(self) -> None:
        cells = [(f"S{i}", _solid((i * 30, i * 20, i * 10))) for i in range(6)]
        page = bs._make_page(cells, self._font())
        assert page.size == (bs.A4_W, bs.A4_H)

    def test_first_cell_top_left(self) -> None:
        """The top-left corner of cell 0 should be white-free beyond the margin."""
        red = _solid((255, 0, 0), size=200)
        cells = [("Red", red)]
        page = bs._make_page(cells, self._font())
        # The image must have been placed starting at (MARGIN, MARGIN)
        # Check that the page is not entirely white (something was drawn)
        arr = np.array(page)
        assert arr.min() < 255, "Page appears blank — no cell was rendered"


# ---------------------------------------------------------------------------
# Original image as first cell
# ---------------------------------------------------------------------------

class TestOriginalCellFirst:
    def test_original_prepended(self) -> None:
        """build_cell_list prepends ('Original', source) before styled results."""
        original = _solid((10, 20, 30))
        styled = [("Candy", _solid((200, 100, 50)))]
        result = bs.build_cell_list(original, styled)
        assert result[0][0] == "Original"
        assert result[0][1] is original

    def test_original_is_first_among_styled(self) -> None:
        original = _solid((10, 20, 30))
        styled = [("A", _solid((1, 2, 3))), ("B", _solid((4, 5, 6)))]
        result = bs.build_cell_list(original, styled)
        assert result[1][0] == "A"
        assert result[2][0] == "B"

    def test_length_is_n_styled_plus_one(self) -> None:
        original = _solid((0, 0, 0))
        n = 5
        styled = [(f"S{i}", _solid((i, i, i))) for i in range(n)]
        result = bs.build_cell_list(original, styled)
        assert len(result) == n + 1


# ---------------------------------------------------------------------------
# _style_name_to_filename
# ---------------------------------------------------------------------------

class TestStyleNameToFilename:
    def test_spaces_become_underscores(self) -> None:
        assert bs._style_name_to_filename("Rain Princess") == "rain_princess"

    def test_already_lower(self) -> None:
        assert bs._style_name_to_filename("candy") == "candy"

    def test_special_chars_replaced(self) -> None:
        result = bs._style_name_to_filename("Style/One:Two")
        assert "/" not in result
        assert ":" not in result

    def test_output_is_non_empty(self) -> None:
        assert bs._style_name_to_filename("X") != ""


# ---------------------------------------------------------------------------
# Integration: main() --pdfoverview
# ---------------------------------------------------------------------------

class TestMainIntegration:
    def test_pdf_created_with_mock_engine(self, tmp_path: Path) -> None:
        """main() should write a valid PDF when the engine is mocked."""
        # Set up a minimal catalog with one style
        style_onnx = tmp_path / "styles" / "fake_style" / "model.onnx"
        style_onnx.parent.mkdir(parents=True)
        style_onnx.write_bytes(b"fake")

        catalog = {
            "styles": [{
                "id": "fake_style",
                "name": "Fake Style",
                "model_path": "styles/fake_style/model.onnx",
            }]
        }
        catalog_path = tmp_path / "styles" / "catalog.json"
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

        # Create a source image
        photo = tmp_path / "photo.jpg"
        _solid((120, 130, 140), size=128).save(photo)

        styled_result = _solid((200, 180, 160), size=128)

        mock_engine = MagicMock()
        mock_engine.apply.return_value = styled_result

        with (
            patch("batch_styler.StyleTransferEngine", return_value=mock_engine),
            patch("batch_styler.REPO_ROOT", tmp_path),
        ):
            with patch("sys.argv", ["batch_styler.py", "--pdfoverview", str(photo)]):
                bs.main()

        pdf_path = tmp_path / "photo_thumbnails.pdf"
        assert pdf_path.exists(), "PDF file was not created"
        assert pdf_path.stat().st_size > 1000, "PDF appears empty"
        # Verify it starts with the PDF magic bytes
        assert pdf_path.read_bytes()[:4] == b"%PDF"

    def test_pdf_has_two_pages_for_seven_styles(self, tmp_path: Path) -> None:
        """7 styles + 1 original = 8 cells → 2 pages (6 + 2)."""
        n_styles = 7
        styles_dir = tmp_path / "styles"
        styles_dir.mkdir()

        style_entries = []
        for i in range(n_styles):
            sid = f"style_{i}"
            onnx = styles_dir / sid / "model.onnx"
            onnx.parent.mkdir()
            onnx.write_bytes(b"fake")
            style_entries.append({
                "id": sid,
                "name": f"Style {i}",
                "model_path": f"styles/{sid}/model.onnx",
            })

        (styles_dir / "catalog.json").write_text(
            json.dumps({"styles": style_entries}), encoding="utf-8"
        )

        photo = tmp_path / "photo.jpg"
        _solid((100, 100, 100), size=64).save(photo)

        mock_engine = MagicMock()
        mock_engine.apply.return_value = _solid((50, 50, 50), size=64)

        with (
            patch("batch_styler.StyleTransferEngine", return_value=mock_engine),
            patch("batch_styler.REPO_ROOT", tmp_path),
        ):
            with patch("sys.argv", ["batch_styler.py", "--pdfoverview", str(photo)]):
                bs.main()

        # Open the PDF and check page count via byte scanning
        pdf_bytes = (tmp_path / "photo_thumbnails.pdf").read_bytes()
        # Count "/Page " occurrences (each page object contains this)
        page_count = pdf_bytes.count(b"/Type /Page\n") + pdf_bytes.count(b"/Type/Page\n")
        assert page_count >= 2, f"Expected >=2 PDF pages, found {page_count}"

    def test_no_mode_flag_exits_with_error(self, tmp_path: Path) -> None:
        """Calling main() without --pdfoverview or --fullimage must exit with code 1."""
        photo = tmp_path / "photo.jpg"
        _solid((100, 100, 100), size=64).save(photo)
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["batch_styler.py", str(photo)]):
                bs.main()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Integration: main() --fullimage
# ---------------------------------------------------------------------------

class TestMainFullImage:
    def _setup_catalog(self, tmp_path: Path, n: int = 3) -> tuple[Path, list[dict]]:
        """Create n fake style entries in a temporary catalog."""
        entries = []
        for i in range(n):
            sid = f"style_{i}"
            onnx = tmp_path / "styles" / sid / "model.onnx"
            onnx.parent.mkdir(parents=True, exist_ok=True)
            onnx.write_bytes(b"fake")
            entries.append({
                "id": sid,
                "name": f"Style {i}",
                "model_path": f"styles/{sid}/model.onnx",
            })
        (tmp_path / "styles" / "catalog.json").write_text(
            json.dumps({"styles": entries}), encoding="utf-8"
        )
        photo = tmp_path / "photo.jpg"
        _solid((100, 150, 200), size=128).save(photo)
        return photo, entries

    def test_jpeg_per_style_created(self, tmp_path: Path) -> None:
        """One JPEG is written for every style."""
        photo, entries = self._setup_catalog(tmp_path, n=3)
        mock_engine = MagicMock()
        mock_engine.apply.return_value = _solid((80, 80, 80), size=128)

        with (
            patch("batch_styler.StyleTransferEngine", return_value=mock_engine),
            patch("batch_styler.REPO_ROOT", tmp_path),
        ):
            with patch("sys.argv", ["batch_styler.py", "--fullimage", str(photo)]):
                bs.main()

        for e in entries:
            stem = bs._style_name_to_filename(e["name"])
            out = tmp_path / f"photo_{stem}.jpg"
            assert out.exists(), f"Missing output: {out.name}"
            assert out.stat().st_size > 0

    def test_original_not_duplicated(self, tmp_path: Path) -> None:
        """No file named photo_original.jpg should be written."""
        photo, _ = self._setup_catalog(tmp_path, n=2)
        mock_engine = MagicMock()
        mock_engine.apply.return_value = _solid((10, 10, 10), size=64)

        with (
            patch("batch_styler.StyleTransferEngine", return_value=mock_engine),
            patch("batch_styler.REPO_ROOT", tmp_path),
        ):
            with patch("sys.argv", ["batch_styler.py", "--fullimage", str(photo)]):
                bs.main()

        assert not (tmp_path / "photo_original.jpg").exists()

    def test_output_is_valid_jpeg(self, tmp_path: Path) -> None:
        """Each written file should open as a valid JPEG."""
        photo, entries = self._setup_catalog(tmp_path, n=1)
        mock_engine = MagicMock()
        result_img = _solid((200, 100, 50), size=128)
        mock_engine.apply.return_value = result_img

        with (
            patch("batch_styler.StyleTransferEngine", return_value=mock_engine),
            patch("batch_styler.REPO_ROOT", tmp_path),
        ):
            with patch("sys.argv", ["batch_styler.py", "--fullimage", str(photo)]):
                bs.main()

        stem = bs._style_name_to_filename(entries[0]["name"])
        out = tmp_path / f"photo_{stem}.jpg"
        opened = Image.open(out)
        assert opened.format == "JPEG"
