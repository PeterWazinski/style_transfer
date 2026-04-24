"""Unit tests for src/trainer/style_analyser.py and KaggleStyleRunner.analyse_style()."""
from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.trainer.style_analyser import analyse_style, analyse_style_set, hist_overlap, hist_overlap_matrix, recommend_weights, snap_sw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_style_image(tmp_path: pathlib.Path) -> pathlib.Path:
    """Save a 64×64 random RGB image to disk and return its path."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    path = tmp_path / "style.jpg"
    Image.fromarray(arr).save(str(path))
    return path


@pytest.fixture()
def flat_style_image(tmp_path: pathlib.Path) -> pathlib.Path:
    """Completely flat (single-colour) 64×64 image — exercises the ⚠ Weak branch."""
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    path = tmp_path / "flat_style.jpg"
    Image.fromarray(arr).save(str(path))
    return path


@pytest.fixture()
def high_texture_style_image(tmp_path: pathlib.Path) -> pathlib.Path:
    """Checkerboard-like image with strong local variance — exercises ✓ Excellent branch."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            arr[i, j] = 255 if (i // 4 + j // 4) % 2 == 0 else 0
    path = tmp_path / "high_texture_style.jpg"
    Image.fromarray(arr).save(str(path))
    return path


# ---------------------------------------------------------------------------
# analyse_style — return-value contract
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "name", "flat_pct", "mean_patch_std", "edge_density",
    "color_std", "local_var", "white", "black",
    "pure_r", "pure_g", "pure_b", "pure_y",
}


def test_analyse_style_returns_all_keys(synthetic_style_image: pathlib.Path) -> None:
    m = analyse_style(synthetic_style_image)
    assert set(m.keys()) == EXPECTED_KEYS


def test_analyse_style_name_matches_filename(synthetic_style_image: pathlib.Path) -> None:
    m = analyse_style(synthetic_style_image)
    assert m["name"] == synthetic_style_image.name


def test_analyse_style_values_are_floats(synthetic_style_image: pathlib.Path) -> None:
    m = analyse_style(synthetic_style_image)
    for key in EXPECTED_KEYS - {"name"}:
        assert isinstance(m[key], float), f"{key} should be float, got {type(m[key])}"


def test_analyse_style_flat_pct_range(synthetic_style_image: pathlib.Path) -> None:
    m = analyse_style(synthetic_style_image)
    assert 0.0 <= m["flat_pct"] <= 100.0


def test_analyse_style_flat_image_has_high_flat_pct(flat_style_image: pathlib.Path) -> None:
    """A uniform-colour image should have near-100% flat_pct."""
    m = analyse_style(flat_style_image)
    assert m["flat_pct"] > 90.0, f"Expected flat_pct > 90, got {m['flat_pct']:.1f}"


def test_analyse_style_local_var_nonnegative(synthetic_style_image: pathlib.Path) -> None:
    m = analyse_style(synthetic_style_image)
    assert m["local_var"] >= 0.0


def test_analyse_style_edge_density_nonnegative(synthetic_style_image: pathlib.Path) -> None:
    m = analyse_style(synthetic_style_image)
    assert m["edge_density"] >= 0.0


# ---------------------------------------------------------------------------
# recommend_weights — decision-rule coverage
# ---------------------------------------------------------------------------

def test_recommend_returns_three_tuple(synthetic_style_image: pathlib.Path) -> None:
    m = analyse_style(synthetic_style_image)
    result = recommend_weights(m)
    assert len(result) == 3


def test_recommend_excellent_branch() -> None:
    """local_var ≥ 900, flat_pct < 20 → Excellent."""
    m: dict = {"local_var": 950.0, "flat_pct": 10.0}
    sw, cw, verdict = recommend_weights(m)
    assert verdict == "✓ Excellent"
    assert sw == 1e8
    assert cw == 1e5


def test_recommend_good_branch() -> None:
    """local_var ≥ 700, flat_pct not < 20 → Good."""
    m: dict = {"local_var": 750.0, "flat_pct": 25.0}
    sw, cw, verdict = recommend_weights(m)
    assert verdict == "✓ Good"
    assert sw == 3e8
    assert cw == 1e5


def test_recommend_weak_ceiling_high_flat_pct() -> None:
    """flat_pct ≥ 55 → Weak/ceiling with sw=1e9."""
    m: dict = {"local_var": 300.0, "flat_pct": 60.0}
    sw, cw, verdict = recommend_weights(m)
    assert verdict == "⚠ Weak / ceiling"
    assert sw == 1e9


def test_recommend_weak_ceiling_moderate_flat_pct() -> None:
    """40 ≤ flat_pct < 55 → Weak/ceiling with sw=5e8."""
    m: dict = {"local_var": 300.0, "flat_pct": 45.0}
    sw, cw, verdict = recommend_weights(m)
    assert verdict == "⚠ Weak / ceiling"
    assert sw == 5e8


def test_recommend_moderate_branch() -> None:
    """Falls through all other conditions → Moderate."""
    m: dict = {"local_var": 400.0, "flat_pct": 15.0}
    sw, cw, verdict = recommend_weights(m)
    assert verdict == "~ Moderate"
    assert sw == 5e8
    assert cw == 5e4


def test_recommend_verdicts_are_known_strings(synthetic_style_image: pathlib.Path) -> None:
    m = analyse_style(synthetic_style_image)
    _, _, verdict = recommend_weights(m)
    assert verdict in {"✓ Excellent", "✓ Good", "~ Moderate", "⚠ Weak / ceiling"}


# ---------------------------------------------------------------------------
# KaggleStyleRunner.analyse_style() — integration with analyse_style module
# ---------------------------------------------------------------------------

def test_runner_analyse_style_calls_module(synthetic_style_image: pathlib.Path) -> None:
    """KaggleStyleRunner.analyse_style() should delegate to src.trainer.style_analyser."""
    from scripts.kaggle_training_helper import KaggleStyleRunner, TrainingConfig

    cfg = TrainingConfig(
        style_images=[synthetic_style_image],
        style_id="test",
        style_name="Test",
        coco_path=pathlib.Path("."),
    )
    runner = KaggleStyleRunner(cfg)

    with patch("scripts.kaggle_training_helper.analyse_style", wraps=analyse_style) as mock_fn:
        result = runner.analyse_style()

    mock_fn.assert_called_once_with(synthetic_style_image)
    assert isinstance(result, list)
    assert len(result) == 1
    assert set(result[0]["metrics"].keys()) == EXPECTED_KEYS


# ---------------------------------------------------------------------------
# TrainingConfig.save() / load() round-trip
# ---------------------------------------------------------------------------

def test_training_config_save_load_roundtrip(
    synthetic_style_image: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    from scripts.kaggle_training_helper import TrainingConfig

    cfg = TrainingConfig(
        style_images=[synthetic_style_image],
        style_id="roundtrip",
        style_name="Round Trip",
        coco_path=tmp_path,
        style_weight=1e10,
        content_weight=5e4,
        epochs=2,
        batch_size=4,
        image_size=256,
        smoke_batches=2000,
        device="cpu",
    )
    cfg.save(tmp_path)
    loaded = TrainingConfig.load(tmp_path)

    assert loaded.style_id == cfg.style_id
    assert loaded.style_name == cfg.style_name
    assert loaded.style_weight == cfg.style_weight
    assert loaded.content_weight == cfg.content_weight
    assert loaded.epochs == cfg.epochs
    assert loaded.device == cfg.device
    assert loaded.style_images == cfg.style_images


# ---------------------------------------------------------------------------
# snap_sw — rounding behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (1.1e8,  1e8),   # 1.1 ≤ 1×1.42 → snaps to 1
    (1.9e8,  2e8),   # 1.9 ≤ 2×1.42 → snaps to 2
    (2.8e8,  2e8),   # 2.8 ≤ 2×1.42=2.84 → snaps to 2
    (4.2e8,  3e8),   # 4.2 ≤ 3×1.42=4.26 → snaps to 3
    (6.5e8,  5e8),   # 6.5 ≤ 5×1.42=7.10 → snaps to 5
    (9.5e8,  7e8),   # 9.5 ≤ 7×1.42=9.94 → snaps to 7
    (1.0,    1.0),   # minimum / edge-case
])
def test_snap_sw_rounds_correctly(raw: float, expected: float) -> None:
    assert snap_sw(raw) == pytest.approx(expected)


def test_snap_sw_zero_guarded() -> None:
    """snap_sw(0) must not raise (max(0,1)=1 guard)."""
    result = snap_sw(0.0)
    assert result >= 1.0


# ---------------------------------------------------------------------------
# hist_overlap — contract tests
# ---------------------------------------------------------------------------

def test_hist_overlap_identical_images_returns_one() -> None:
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8).astype(np.float32)
    assert hist_overlap(arr, arr) == pytest.approx(1.0, abs=1e-6)


def test_hist_overlap_range_zero_to_one() -> None:
    a = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8).astype(np.float32)
    b = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8).astype(np.float32)
    score = hist_overlap(a, b)
    assert 0.0 <= score <= 1.0


def test_hist_overlap_different_palettes_lower_than_identical() -> None:
    """All-red vs all-blue should score lower than all-red vs all-red."""
    red  = np.zeros((64, 64, 3), dtype=np.float32); red[..., 0]  = 200.0
    blue = np.zeros((64, 64, 3), dtype=np.float32); blue[..., 2] = 200.0
    assert hist_overlap(red, blue) < hist_overlap(red, red)


def test_hist_overlap_symmetric() -> None:
    a = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8).astype(np.float32)
    b = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8).astype(np.float32)
    assert hist_overlap(a, b) == pytest.approx(hist_overlap(b, a), abs=1e-6)


# ---------------------------------------------------------------------------
# analyse_style_set
# ---------------------------------------------------------------------------

def test_analyse_style_set_single_image(synthetic_style_image: pathlib.Path) -> None:
    result = analyse_style_set([synthetic_style_image])
    assert len(result["images"]) == 1
    assert set(result["means"].keys()) == {"flat_pct", "mean_patch_std", "edge_density", "color_std", "local_var"}
    assert set(result["stds"].keys())  == {"flat_pct", "mean_patch_std", "edge_density", "color_std", "local_var"}
    assert "warnings" in result
    assert any("1 style image" in w for w in result["warnings"]), "Expected N=1 warning"


def test_analyse_style_set_multi_image(
    synthetic_style_image: pathlib.Path,
    flat_style_image: pathlib.Path,
    high_texture_style_image: pathlib.Path,
) -> None:
    paths = [synthetic_style_image, flat_style_image, high_texture_style_image]
    result = analyse_style_set(paths)
    assert len(result["images"]) == 3
    for k in ("flat_pct", "mean_patch_std", "edge_density", "color_std", "local_var"):
        assert isinstance(result["means"][k], float)
        assert isinstance(result["stds"][k], float)


def test_analyse_style_set_means_match_manual(
    synthetic_style_image: pathlib.Path,
    flat_style_image: pathlib.Path,
) -> None:
    paths = [synthetic_style_image, flat_style_image]
    result = analyse_style_set(paths)
    m0 = analyse_style(paths[0])
    m1 = analyse_style(paths[1])
    expected_mean_flat = (m0["flat_pct"] + m1["flat_pct"]) / 2.0
    assert result["means"]["flat_pct"] == pytest.approx(expected_mean_flat, abs=1e-6)


def test_analyse_style_set_flat_image_flagged_as_outlier(
    flat_style_image: pathlib.Path,
    high_texture_style_image: pathlib.Path,
) -> None:
    """A flat image alongside a high-texture image should be an outlier."""
    # The flat image has very high flat_pct; high_texture has low flat_pct.
    # flat_pct of flat > 2× mean only if the flat image is extreme enough.
    # Use 3 high-texture + 1 flat to make the mean clearly low.
    paths = [high_texture_style_image, high_texture_style_image, flat_style_image]
    result = analyse_style_set(paths)
    outlier_paths = [o["path"] for o in result["outliers"]]
    assert flat_style_image in outlier_paths, (
        f"Flat image was not flagged as outlier. "
        f"flat_pcts={[m['flat_pct'] for m in result['images']]}"
    )


def test_analyse_style_set_empty_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        analyse_style_set([])


def test_analyse_style_set_no_false_outliers(
    synthetic_style_image: pathlib.Path,
    high_texture_style_image: pathlib.Path,
) -> None:
    """Two similar-texture images should not flag either as an outlier."""
    result = analyse_style_set([synthetic_style_image, high_texture_style_image])
    # Neither random nor checkerboard should flag as flat outlier given both have
    # varied texture — allow at most 0 outliers here.
    assert len(result["outliers"]) == 0


# ---------------------------------------------------------------------------
# hist_overlap_matrix
# ---------------------------------------------------------------------------

def test_hist_overlap_matrix_diagonal_is_one(synthetic_style_image: pathlib.Path) -> None:
    mat = hist_overlap_matrix([synthetic_style_image, synthetic_style_image])
    assert mat[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert mat[1, 1] == pytest.approx(1.0, abs=1e-5)


def test_hist_overlap_matrix_shape(
    synthetic_style_image: pathlib.Path,
    flat_style_image: pathlib.Path,
    high_texture_style_image: pathlib.Path,
) -> None:
    paths = [synthetic_style_image, flat_style_image, high_texture_style_image]
    mat = hist_overlap_matrix(paths)
    assert mat.shape == (3, 3)


def test_hist_overlap_matrix_symmetric(
    synthetic_style_image: pathlib.Path,
    flat_style_image: pathlib.Path,
) -> None:
    mat = hist_overlap_matrix([synthetic_style_image, flat_style_image])
    assert mat[0, 1] == pytest.approx(mat[1, 0], abs=1e-6)


def test_hist_overlap_matrix_values_in_range(
    synthetic_style_image: pathlib.Path,
    flat_style_image: pathlib.Path,
) -> None:
    mat = hist_overlap_matrix([synthetic_style_image, flat_style_image])
    assert float(mat.min()) >= 0.0
    assert float(mat.max()) <= 1.0 + 1e-6


def test_hist_overlap_matrix_single_image(synthetic_style_image: pathlib.Path) -> None:
    mat = hist_overlap_matrix([synthetic_style_image])
    assert mat.shape == (1, 1)
    assert mat[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_hist_overlap_matrix_empty_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        hist_overlap_matrix([])


def test_hist_overlap_matrix_diverse_warns(
    synthetic_style_image: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """A red and a blue image pair should trigger a UserWarning about diversity."""
    red_arr  = np.zeros((64, 64, 3), dtype=np.uint8); red_arr[..., 0]  = 220
    blue_arr = np.zeros((64, 64, 3), dtype=np.uint8); blue_arr[..., 2] = 220
    red_path  = tmp_path / "red.jpg";  Image.fromarray(red_arr).save(str(red_path))
    blue_path = tmp_path / "blue.jpg"; Image.fromarray(blue_arr).save(str(blue_path))
    with pytest.warns(UserWarning, match="diverse"):
        hist_overlap_matrix([red_path, blue_path])

