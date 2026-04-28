# Painter Style Creation Guide

A practical guide for collecting and selecting style images to train new NST styles.

---

## Key constraint: colour consistency

The style analyser warns when mean pairwise colour similarity < 0.4.
**Colour consistency across your style images is more important than artistic quality.**
Run the analyser before committing to a full training run.

---

## Cubism

### Option A — Analytic Cubism (1908–1912) ⭐ recommended first attempt

Monochromatic palette: ochres, grey-browns, muted greens. Very fragmented planes.
The narrow palette maps cleanly to NST and transfers predictably.

| Artist | Title | Year |
|---|---|---|
| Picasso | *Portrait of Ambroise Vollard* | 1910 |
| Picasso | *Girl with a Mandolin* | 1910 |
| Picasso | *The Reservoir, Horta de Ebro* | 1909 |
| Braque | *Houses at l'Estaque* | 1908 |
| Braque | *Violin and Candlestick* | 1910 |
| Braque | *The Portuguese* | 1911 |

> **Why Braque?** His Analytic Cubist works share the same palette as Picasso's
> and are slightly more textural — NST responds well to texture variation.
> Mixing both artists is fine; they were working side by side.

> **Avoid** *Les Demoiselles d'Avignon* — it is Proto-Cubist with skin tones and
> African mask figures that pull the style in a different direction.

### Option B — Synthetic Cubism (1913–1921)

Flat, bolder shapes, sometimes collage. More colourful output but harder to keep
the style analyser happy. Increase `style_weight` if colours look washed out.

| Artist | Title | Year |
|---|---|---|
| Picasso | *Harlequin* | 1915 |
| Picasso | *Three Musicians* | 1921 |
| Léger | *The City* | 1919 |
| Gris | *Guitar and Flowers* | 1912 |

---

## General tips (any painter style)

| Topic | Recommendation |
|---|---|
| **Image count** | 5–8 from one sub-period; 4 is the tested minimum |
| **Resolution** | ≥ 600 × 600 px — Wikimedia Commons has museum-quality scans |
| **Sub-period discipline** | Pick one sub-period per style; mixing eras hurts colour consistency |
| **Analyser check** | Run `run_smoke_test` (cell 2 of `kaggle_multi_pic_trainer.ipynb`) before full training |
| **Colour similarity < 0.4** | Drop the outlier image or switch to a tighter sub-period |
| **style_weight** | Analytic styles (narrow palette) → default `1e10`; Synthetic/colourful → try `5e10` |

---

## Recommended workflow

1. Collect 6–8 candidate images from Wikimedia Commons.
2. Run the style analyser; drop images that drag mean overlap below 0.4.
3. Run a smoke test (2 000 batches) and inspect the preview.
4. If the preview looks good, launch full training on Kaggle.
5. Add the exported ONNX to the gallery with `scripts/add_style.ipynb`.
