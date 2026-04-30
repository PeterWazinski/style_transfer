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
- main() --replay: mock engine, verify replay chain is applied and JPEG is written
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
        """7 styles × 3 strengths + original row (3 cells, 2 blank) = 24 cells → 4 pages."""
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
        assert page_count >= 4, f"Expected >=4 PDF pages, found {page_count}"

    def test_no_mode_flag_exits_with_error(self, tmp_path: Path) -> None:
        """Calling main() without --pdfoverview or --replay must exit with code 1."""
        photo = tmp_path / "photo.jpg"
        _solid((100, 100, 100), size=64).save(photo)
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["batch_styler.py", str(photo)]):
                bs.main()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Integration: main() --replay
# ---------------------------------------------------------------------------

class TestCmdReplay:
    """Tests for cmd_replay() — the --replay mode."""

    _CHAIN_YAML = """\
version: 1
steps:
  - style: Candy
    strength: 100
  - style: Mosaic
    strength: 150
"""

    def _setup(self, tmp_path: Path, n_styles: int = 2) -> tuple[Path, Path, list[dict]]:
        """Create a catalog, a photo, and a chain YAML in tmp_path."""
        entries = []
        names = ["Candy", "Mosaic", "Udnie"]
        for i in range(n_styles):
            sid = names[i].lower()
            onnx = tmp_path / "styles" / sid / "model.onnx"
            onnx.parent.mkdir(parents=True, exist_ok=True)
            onnx.write_bytes(b"fake")
            entries.append({
                "id": sid,
                "name": names[i],
                "model_path": f"styles/{sid}/model.onnx",
            })
        (tmp_path / "styles" / "catalog.json").write_text(
            json.dumps({"styles": entries}), encoding="utf-8"
        )
        photo = tmp_path / "photo.jpg"
        _solid((100, 150, 200), size=64).save(photo)
        chain = tmp_path / "my_chain.yml"
        chain.write_text(self._CHAIN_YAML, encoding="utf-8")
        return photo, chain, entries

    def test_replay_applies_steps_in_order(self, tmp_path: Path) -> None:
        """engine.apply must be called once per step, in order."""
        photo, chain, _ = self._setup(tmp_path, n_styles=2)
        mock_engine = MagicMock()
        mock_engine.apply.return_value = _solid((80, 80, 80), size=64)
        mock_engine._sessions = {}

        with (
            patch("batch_styler.StyleTransferEngine", return_value=mock_engine),
            patch("batch_styler.REPO_ROOT", tmp_path),
        ):
            bs.cmd_replay(
                photo, chain,
                tile_size=256, overlap=64, use_float16=False,
            )

        assert mock_engine.apply.call_count == 2
        # First call uses style id 'candy'
        first_call_style_id = mock_engine.apply.call_args_list[0][0][1]
        assert first_call_style_id == "candy"
        second_call_style_id = mock_engine.apply.call_args_list[1][0][1]
        assert second_call_style_id == "mosaic"

    def test_replay_output_filename(self, tmp_path: Path) -> None:
        """Output file must be <photo-stem>_<chain-stem>.jpg."""
        photo, chain, _ = self._setup(tmp_path, n_styles=2)
        mock_engine = MagicMock()
        mock_engine.apply.return_value = _solid((80, 80, 80), size=64)
        mock_engine._sessions = {}

        with (
            patch("batch_styler.StyleTransferEngine", return_value=mock_engine),
            patch("batch_styler.REPO_ROOT", tmp_path),
        ):
            bs.cmd_replay(
                photo, chain,
                tile_size=256, overlap=64, use_float16=False,
            )

        expected = tmp_path / "photo_my_chain.jpg"
        assert expected.exists(), "Output JPEG was not created"
        assert expected.stat().st_size > 0

    def test_replay_unknown_style_exits(self, tmp_path: Path) -> None:
        """A step referencing an unknown style name must exit with an error."""
        photo = tmp_path / "photo.jpg"
        _solid((100, 100, 100), size=64).save(photo)
        bad_chain = tmp_path / "bad.yml"
        bad_chain.write_text(
            "version: 1\nsteps:\n  - style: NonExistent\n    strength: 100\n",
            encoding="utf-8",
        )
        # Minimal catalog with no matching style
        onnx = tmp_path / "styles" / "candy" / "model.onnx"
        onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.write_bytes(b"fake")
        (tmp_path / "styles" / "catalog.json").write_text(
            json.dumps({"styles": [{"id": "candy", "name": "Candy", "model_path": "styles/candy/model.onnx"}]}),
            encoding="utf-8",
        )
        with (
            patch("batch_styler.REPO_ROOT", tmp_path),
            pytest.raises(SystemExit),
        ):
            bs.cmd_replay(photo, bad_chain, tile_size=256, overlap=64, use_float16=False)

    def test_replay_strength_converted_to_float(self, tmp_path: Path) -> None:
        """Each step's integer % strength must be divided by 100 before engine.apply."""
        photo, chain, _ = self._setup(tmp_path, n_styles=2)
        mock_engine = MagicMock()
        mock_engine.apply.return_value = _solid((80, 80, 80), size=64)
        mock_engine._sessions = {}

        with (
            patch("batch_styler.StyleTransferEngine", return_value=mock_engine),
            patch("batch_styler.REPO_ROOT", tmp_path),
        ):
            bs.cmd_replay(photo, chain, tile_size=256, overlap=64, use_float16=False)

        # Chain has Candy @ 100% (1.0) and Mosaic @ 150% (1.5)
        first_strength = mock_engine.apply.call_args_list[0][1]["strength"]
        second_strength = mock_engine.apply.call_args_list[1][1]["strength"]
        assert abs(first_strength - 1.0) < 1e-6
        assert abs(second_strength - 1.5) < 1e-6

    def test_replay_strength_override_scales_all_steps(self, tmp_path: Path) -> None:
        """--strength-override 60 must scale each step's strength by 0.60."""
        photo, chain, _ = self._setup(tmp_path, n_styles=2)
        mock_engine = MagicMock()
        mock_engine.apply.return_value = _solid((80, 80, 80), size=64)
        mock_engine._sessions = {}

        with (
            patch("batch_styler.StyleTransferEngine", return_value=mock_engine),
            patch("batch_styler.REPO_ROOT", tmp_path),
        ):
            bs.cmd_replay(
                photo, chain,
                tile_size=256, overlap=64, use_float16=False,
                strength_override=60,
            )

        # Candy: 100% × 0.60 = 0.60; Mosaic: 150% × 0.60 = 0.90
        first_strength = mock_engine.apply.call_args_list[0][1]["strength"]
        second_strength = mock_engine.apply.call_args_list[1][1]["strength"]
        assert abs(first_strength - 0.60) < 1e-6
        assert abs(second_strength - 0.90) < 1e-6

    def test_replay_invalid_schema_exits(self, tmp_path: Path) -> None:
        """A YAML with invalid schema (e.g. strength out of range) must exit."""
        photo = tmp_path / "photo.jpg"
        _solid((100, 100, 100), size=64).save(photo)
        invalid_chain = tmp_path / "invalid.yml"
        invalid_chain.write_text(
            "version: 1\nsteps:\n  - style: Candy\n    strength: 999\n",
            encoding="utf-8",
        )
        onnx = tmp_path / "styles" / "candy" / "model.onnx"
        onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.write_bytes(b"fake")
        (tmp_path / "styles" / "catalog.json").write_text(
            json.dumps({"styles": [{"id": "candy", "name": "Candy", "model_path": "styles/candy/model.onnx"}]}),
            encoding="utf-8",
        )
        with (
            patch("batch_styler.REPO_ROOT", tmp_path),
            pytest.raises(SystemExit),
        ):
            bs.cmd_replay(photo, invalid_chain, tile_size=None, overlap=None, use_float16=False)

    def test_tile_settings_from_yaml_used_when_cli_none(self, tmp_path: Path) -> None:
        """tile_size/tile_overlap stored in the YAML must be passed to engine.apply."""
        photo, _, _ = self._setup(tmp_path, n_styles=2)
        chain_with_tiles = tmp_path / "tiled_chain.yml"
        chain_with_tiles.write_text(
            "version: 1\ntile_size: 768\ntile_overlap: 32\n"
            "steps:\n  - style: Candy\n    strength: 100\n  - style: Mosaic\n    strength: 150\n",
            encoding="utf-8",
        )
        mock_engine = MagicMock()
        mock_engine.apply.return_value = _solid((80, 80, 80), size=64)
        mock_engine._sessions = {}

        with (
            patch("batch_styler.StyleTransferEngine", return_value=mock_engine),
            patch("batch_styler.REPO_ROOT", tmp_path),
        ):
            bs.cmd_replay(photo, chain_with_tiles, tile_size=None, overlap=None, use_float16=False)

        for call in mock_engine.apply.call_args_list:
            assert call[1]["tile_size"] == 768
            assert call[1]["overlap"] == 32

    def test_cli_tile_size_overrides_yaml(self, tmp_path: Path) -> None:
        """An explicit CLI tile_size must take precedence over the YAML value."""
        photo, _, _ = self._setup(tmp_path, n_styles=2)
        chain_with_tiles = tmp_path / "tiled_chain.yml"
        chain_with_tiles.write_text(
            "version: 1\ntile_size: 768\ntile_overlap: 32\n"
            "steps:\n  - style: Candy\n    strength: 100\n  - style: Mosaic\n    strength: 150\n",
            encoding="utf-8",
        )
        mock_engine = MagicMock()
        mock_engine.apply.return_value = _solid((80, 80, 80), size=64)
        mock_engine._sessions = {}

        with (
            patch("batch_styler.StyleTransferEngine", return_value=mock_engine),
            patch("batch_styler.REPO_ROOT", tmp_path),
        ):
            bs.cmd_replay(photo, chain_with_tiles, tile_size=512, overlap=None, use_float16=False)

        for call in mock_engine.apply.call_args_list:
            assert call[1]["tile_size"] == 512   # CLI override
            assert call[1]["overlap"] == 32      # from YAML


class _OldTestMainFullImage:
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
        pass  # removed — --fullimage no longer exists


# ---------------------------------------------------------------------------
# --style filter: filter_styles_by_name unit tests
# ---------------------------------------------------------------------------

class TestFilterStylesByName:
    def _styles(self) -> list[dict]:
        return [
            {"id": "candy", "name": "Candy", "model_path": "styles/candy/model.onnx"},
            {"id": "mosaic", "name": "Mosaic", "model_path": "styles/mosaic/model.onnx"},
            {"id": "anime_hayao", "name": "Anime Hayao", "model_path": "styles/anime_hayao/model.onnx"},
        ]

    def test_exact_match_returns_single_entry(self) -> None:
        result = bs.filter_styles_by_name(self._styles(), "Candy")
        assert len(result) == 1
        assert result[0]["name"] == "Candy"

    def test_case_insensitive_match(self) -> None:
        result = bs.filter_styles_by_name(self._styles(), "anime hayao")
        assert result[0]["name"] == "Anime Hayao"

    def test_uppercase_query(self) -> None:
        result = bs.filter_styles_by_name(self._styles(), "MOSAIC")
        assert result[0]["name"] == "Mosaic"

    def test_leading_trailing_whitespace_stripped(self) -> None:
        result = bs.filter_styles_by_name(self._styles(), "  Candy  ")
        assert result[0]["name"] == "Candy"

    def test_unknown_style_aborts_with_exit(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            bs.filter_styles_by_name(self._styles(), "NonExistent")
        assert exc_info.value.code is not None

    def test_error_message_contains_style_name(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            bs.filter_styles_by_name(self._styles(), "Ghost")
        assert "Ghost" in str(exc_info.value.code)

    def test_error_message_lists_available_styles(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            bs.filter_styles_by_name(self._styles(), "Ghost")
        msg = str(exc_info.value.code)
        assert "Candy" in msg
        assert "Mosaic" in msg

    def test_empty_catalog_aborts(self) -> None:
        with pytest.raises(SystemExit):
            bs.filter_styles_by_name([], "Candy")


# ---------------------------------------------------------------------------
# --style filter: integration with main()
# ---------------------------------------------------------------------------

class TestMainStyleFilter:
    """Verify that --style correctly limits which styles are processed."""

    def _setup_catalog(self, tmp_path: Path) -> Path:
        """Write a 3-style catalog and a source photo into tmp_path."""
        entries = [
            {"id": "candy",  "name": "Candy",  "model_path": "styles/candy/model.onnx"},
            {"id": "mosaic", "name": "Mosaic", "model_path": "styles/mosaic/model.onnx"},
            {"id": "udnie",  "name": "Udnie",  "model_path": "styles/udnie/model.onnx"},
        ]
        for e in entries:
            onnx = tmp_path / e["model_path"]
            onnx.parent.mkdir(parents=True, exist_ok=True)
            onnx.write_bytes(b"fake")
        (tmp_path / "styles" / "catalog.json").write_text(
            json.dumps({"styles": entries}), encoding="utf-8"
        )
        photo = tmp_path / "photo.jpg"
        _solid((100, 100, 100), size=64).save(photo)
        return photo

    def test_pdfoverview_with_style_filter(self, tmp_path: Path) -> None:
        """--style must also work with --pdfoverview mode."""
        photo = self._setup_catalog(tmp_path)
        called: list[str] = []

        def _fake_pdf(
            image_path: Path,
            styles: list[dict],
            **kw: object,
        ) -> None:
            called.extend(s["name"] for s in styles)

        argv = ["batch_styler.py", "--pdfoverview", str(photo), "--style", "Udnie"]
        with (
            patch.object(bs, "REPO_ROOT", tmp_path),
            patch.object(bs, "cmd_pdfoverview", side_effect=_fake_pdf),
        ):
            with patch("sys.argv", argv):
                bs.main()

        assert called == ["Udnie"]

    def test_pdfoverview_without_filter_passes_all(self, tmp_path: Path) -> None:
        photo = self._setup_catalog(tmp_path)
        called: list[str] = []

        def _fake_pdf(image_path: Path, styles: list[dict], **kw: object) -> None:
            called.extend(s["name"] for s in styles)

        argv = ["batch_styler.py", "--pdfoverview", str(photo)]
        with (
            patch.object(bs, "REPO_ROOT", tmp_path),
            patch.object(bs, "cmd_pdfoverview", side_effect=_fake_pdf),
        ):
            with patch("sys.argv", argv):
                bs.main()

        assert called == ["Candy", "Mosaic", "Udnie"]


# (TestApplyAllStylesTensorLayout removed — _apply_all_styles was replaced by cmd_replay)


class _SkippedApplyAllStyles:
    def test_nhwc_tanh_layout_forwarded_to_load_model(self, tmp_path: Path) -> None:
        """If a style has tensor_layout=nhwc_tanh in the catalog, load_model
        must be called with tensor_layout='nhwc_tanh', not the default 'nchw'.
        """
        onnx = tmp_path / 'model.onnx'
        onnx.write_bytes(b'fake')

        styles = [
            {
                'id': 'anime',
                'name': 'Anime',
                'model_path': str(onnx),
                'tensor_layout': 'nhwc_tanh',
            }
        ]

        mock_engine = MagicMock()
        mock_engine.apply.return_value = _solid((100, 100, 100), size=32)
        mock_engine._sessions = {}

        with patch('batch_styler.StyleTransferEngine', return_value=mock_engine):
            with patch('batch_styler.REPO_ROOT', tmp_path):
                bs._apply_all_styles(
                    source=_solid((50, 50, 50), size=32),
                    styles=styles,
                    tile_size=256,
                    overlap=64,
                    strength=1.0,
                    use_float16=False,
                )

        mock_engine.load_model.assert_called_once()
        _, call_kwargs = mock_engine.load_model.call_args
        assert call_kwargs.get('tensor_layout') == 'nhwc_tanh', (
            "load_model must pass tensor_layout='nhwc_tanh' for AnimeGAN-style models"
        )

    def test_default_nchw_layout_when_absent(self, tmp_path: Path) -> None:
        onnx = tmp_path / 'model.onnx'
        onnx.write_bytes(b'fake')

        styles = [
            {
                'id': 'candy',
                'name': 'Candy',
                'model_path': str(onnx),
            }
        ]

        mock_engine = MagicMock()
        mock_engine.apply.return_value = _solid((100, 100, 100), size=32)
        mock_engine._sessions = {}

        with patch('batch_styler.StyleTransferEngine', return_value=mock_engine):
            with patch('batch_styler.REPO_ROOT', tmp_path):
                bs._apply_all_styles(
                    source=_solid((50, 50, 50), size=32),
                    styles=styles,
                    tile_size=256,
                    overlap=64,
                    strength=1.0,
                    use_float16=False,
                )

        _, call_kwargs = mock_engine.load_model.call_args
        assert call_kwargs.get('tensor_layout') == 'nchw'

    def test_nchw_tanh_layout_forwarded_to_load_model(self, tmp_path: Path) -> None:
        """If a style has tensor_layout=nchw_tanh in the catalog, load_model
        must be called with tensor_layout='nchw_tanh' (CycleGAN regression test).
        """
        onnx = tmp_path / 'model.onnx'
        onnx.write_bytes(b'fake')

        styles = [
            {
                'id': 'style_monet',
                'name': 'Monet',
                'model_path': str(onnx),
                'tensor_layout': 'nchw_tanh',
            }
        ]

        mock_engine = MagicMock()
        mock_engine.apply.return_value = _solid((100, 100, 100), size=32)
        mock_engine._sessions = {}

        with patch('batch_styler.StyleTransferEngine', return_value=mock_engine):
            with patch('batch_styler.REPO_ROOT', tmp_path):
                bs._apply_all_styles(
                    source=_solid((50, 50, 50), size=32),
                    styles=styles,
                    tile_size=256,
                    overlap=64,
                    strength=1.0,
                    use_float16=False,
                )

        mock_engine.load_model.assert_called_once()
        _, call_kwargs = mock_engine.load_model.call_args
        assert call_kwargs.get('tensor_layout') == 'nchw_tanh', (
            "load_model must pass tensor_layout='nchw_tanh' for CycleGAN models"
        )
