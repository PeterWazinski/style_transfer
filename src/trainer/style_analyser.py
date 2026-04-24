"""Style image analysis utilities.

Shared by ``scripts/style_analysis.ipynb`` and ``scripts/kaggle_training_helper.py``.
Provides six public functions:

* :func:`analyse_style` — compute texture/geometry metrics for one style image.
* :func:`analyse_style_set` — analyse N images; returns per-image metrics + aggregates + outliers.
* :func:`hist_overlap_matrix` — N×N pairwise colour-histogram similarity matrix.
* :func:`recommend_weights` — map metrics to ``(style_weight, content_weight, verdict)``.
* :func:`snap_sw` — round a raw style-weight to the nearest human-friendly value.
* :func:`hist_overlap` — per-channel histogram overlap between two image arrays.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


def analyse_style(path: pathlib.Path) -> dict:
    """Return texture/geometry metrics for a style image.

    Resizes the image to 512×512 before analysis so results are
    resolution-independent.

    Args:
        path: Path to a JPEG or PNG style image.

    Returns:
        A dict with the following keys:

        * ``name``            – ``path.name``
        * ``flat_pct``        – % of 16×16 blocks with std < 10 (uniform areas)
        * ``mean_patch_std``  – mean local patch std-dev (richness indicator)
        * ``edge_density``    – mean finite-difference edge density
        * ``color_std``       – overall per-channel colour std
        * ``local_var``       – local texture variance in 8×8 blocks (primary predictor)
        * ``white``, ``black``, ``pure_r``, ``pure_g``, ``pure_b``, ``pure_y``
                              – dominant-colour fractions (%)
    """
    img = Image.open(path).convert("RGB").resize((512, 512))
    arr = np.array(img, dtype=float)
    gray = arr.mean(axis=2)

    # Flatness: fraction of 16×16 blocks with std < 10 (uniform colour areas)
    patches16 = sliding_window_view(gray, (16, 16))[::16, ::16]
    patch_stds = patches16.std(axis=(-1, -2))
    flat_pct = float((patch_stds < 10).mean() * 100)
    mean_patch_std = float(patch_stds.mean())

    # Edge density (Sobel approximation via finite differences)
    dx = float(np.abs(np.diff(gray, axis=1)).mean())
    dy = float(np.abs(np.diff(gray, axis=0)).mean())
    edge_density = (dx + dy) / 2

    # Overall colour variability
    color_std = float(arr.std(axis=(0, 1)).mean())

    # Local texture variance in 8×8 blocks — strongest predictor of training success
    patches8 = sliding_window_view(gray, (8, 8))[::8, ::8]
    local_var = float(patches8.var(axis=(-1, -2)).mean())

    # Dominant-colour fractions (sampled at 64×64)
    pixels = np.array(img.resize((64, 64)), dtype=float).reshape(-1, 3)
    white  = float((pixels > 200).all(axis=1).mean() * 100)
    black  = float((pixels < 50).all(axis=1).mean() * 100)
    pure_r = float(((pixels[:, 0] > 150) & (pixels[:, 1] < 100) & (pixels[:, 2] < 100)).mean() * 100)
    pure_g = float(((pixels[:, 1] > 150) & (pixels[:, 0] < 100) & (pixels[:, 2] < 100)).mean() * 100)
    pure_b = float(((pixels[:, 0] < 100) & (pixels[:, 1] < 100) & (pixels[:, 2] > 100)).mean() * 100)
    pure_y = float(((pixels[:, 0] > 150) & (pixels[:, 1] > 150) & (pixels[:, 2] < 100)).mean() * 100)

    return {
        "name": path.name,
        "flat_pct": flat_pct,
        "mean_patch_std": mean_patch_std,
        "edge_density": edge_density,
        "color_std": color_std,
        "local_var": local_var,
        "white": white,
        "black": black,
        "pure_r": pure_r,
        "pure_g": pure_g,
        "pure_b": pure_b,
        "pure_y": pure_y,
    }


def recommend_weights(m: dict) -> tuple[float, float, str]:
    """Map style-image metrics to recommended training hyperparameters.

    Args:
        m: Metrics dict as returned by :func:`analyse_style`.

    Returns:
        ``(style_weight, content_weight, verdict)`` where *verdict* is one of:
        ``"✓ Excellent"``, ``"✓ Good"``, ``"~ Moderate"``, ``"⚠ Weak / ceiling"``.

    Note:
        These are *texture-analysis* recommendations only.  The training
        notebook always overrides ``style_weight`` to ``1e10`` (the value
        validated against the yakhyo reference implementation).  Use the
        returned verdict to decide whether the image is worth training at all.
    """
    lv = m["local_var"]
    fp = m["flat_pct"]
    if lv >= 900 and fp < 20:
        return 1e8, 1e5, "✓ Excellent"
    if lv >= 700:
        return 3e8, 1e5, "✓ Good"
    if fp >= 40:
        sw = 1e9 if fp >= 55 else 5e8
        return sw, 5e4, "⚠ Weak / ceiling"
    return 5e8, 5e4, "~ Moderate"


def snap_sw(raw: float) -> float:
    """Round a raw style-weight to the nearest human-friendly value.

    Snaps to the nearest value in the set {1, 2, 3, 5, 7} × 10^n so that
    suggested weights are easy to read and remember.

    Args:
        raw: Any positive float.

    Returns:
        Nearest snapped value, e.g. ``snap_sw(2.8e8)`` → ``3e8``.
    """
    exp = math.floor(math.log10(max(raw, 1.0)))
    m = raw / 10 ** exp
    for snap in (1.0, 2.0, 3.0, 5.0, 7.0, 10.0):
        if m <= snap * 1.42:
            return snap * 10 ** exp
    return 10 ** (exp + 1)


def analyse_style_set(paths: list[pathlib.Path]) -> dict:
    """Analyse a set of style images and return per-image metrics + aggregate stats.

    Args:
        paths: List of paths to style images.

    Returns:
        A dict with keys:

        * ``"images"``   – list of per-image metric dicts (one per path)
        * ``"means"``    – dict of mean value per numeric metric key
        * ``"stds"``     – dict of std dev per numeric metric key
        * ``"outliers"`` – list of ``{"path": Path, "reason": str}`` for
          images whose ``flat_pct`` exceeds 2× the set mean flat_pct.
        * ``"warnings"`` – list of human-readable warning strings
    """
    if not paths:
        raise ValueError("paths must not be empty")

    metrics_keys = (
        "flat_pct", "mean_patch_std", "edge_density",
        "color_std", "local_var",
    )
    images = [analyse_style(p) for p in paths]

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for k in metrics_keys:
        vals = np.array([m[k] for m in images], dtype=float)
        means[k] = float(vals.mean())
        stds[k] = float(vals.std())

    outliers: list[dict] = []
    mean_flat = means["flat_pct"]
    for img_m, path in zip(images, paths):
        if img_m["flat_pct"] > max(2.0 * mean_flat, 60.0):
            outliers.append({"path": path, "reason": f"flat_pct={img_m['flat_pct']:.1f} > 2× mean"})

    warnings: list[str] = []
    if len(paths) == 1:
        warnings.append("Only 1 style image — multi-image training has no benefit over kaggle_trainer.")
    if outliers:
        for o in outliers:
            warnings.append(f"⚠ {o['path'].name}: {o['reason']} — consider removing.")

    return {
        "images":   images,
        "means":    means,
        "stds":     stds,
        "outliers": outliers,
        "warnings": warnings,
    }


def hist_overlap_matrix(paths: list[pathlib.Path], bins: int = 32) -> np.ndarray:
    """Compute an N×N pairwise colour-histogram similarity matrix.

    Entry [i, j] is the ``hist_overlap`` score between images i and j.
    The diagonal is always 1.0.

    Args:
        paths: List of paths to style images (N ≥ 1).
        bins:  Number of histogram bins per channel (default 32).

    Returns:
        Float32 ndarray of shape (N, N) with values in [0, 1].
        Prints a warning if the mean off-diagonal similarity < 0.4
        (images may be too stylistically diverse for a coherent mean model).
    """
    if not paths:
        raise ValueError("paths must not be empty")

    n = len(paths)
    arrays: list[np.ndarray] = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB").resize((64, 64)), dtype=np.float32)
        arrays.append(arr)

    mat = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            score = hist_overlap(arrays[i], arrays[j], bins=bins)
            mat[i, j] = score
            mat[j, i] = score

    if n > 1:
        off_diag = mat[~np.eye(n, dtype=bool)]
        mean_sim = float(off_diag.mean())
        if mean_sim < 0.4:
            import warnings as _w
            _w.warn(
                f"Style images may be too diverse (mean pairwise colour similarity = {mean_sim:.2f} < 0.4). "
                "Consider using images from the same artist / colour palette.",
                UserWarning,
                stacklevel=2,
            )

    return mat


def hist_overlap(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Compute per-channel histogram overlap between two image arrays.

    Both arrays must be float32 with pixel values in ``[0, 255]`` and shape
    ``(..., 3)`` (last axis = RGB channels).

    Args:
        a: First image array.
        b: Second image array.
        bins: Number of histogram bins (default 32).

    Returns:
        Overlap score in ``[0, 1]``.  ``1.0`` means identical colour distributions.
    """
    bin_w = 256.0 / bins
    total = 0.0
    for c in range(3):
        ha, _ = np.histogram(a[..., c], bins=bins, range=(0.0, 256.0), density=True)
        hb, _ = np.histogram(b[..., c], bins=bins, range=(0.0, 256.0), density=True)
        total += float(np.minimum(ha, hb).sum() * bin_w)
    return total / 3.0
