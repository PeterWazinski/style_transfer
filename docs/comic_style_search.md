# Training a Comic / Asterix-Style Model

## What makes Asterix's art style distinctive

The Uderzo style (Asterix) has very specific visual characteristics that NST picks up well:

- **Bold black outlines** — thick, clean ink contours around every shape
- **Flat colour fills** — solid colours with no photographic gradients
- **Hatching for shadow** — parallel lines instead of smooth shading
- **High saturation, warm palette** — reds, yellows, sky blues, skin tones
- **No texture** — paper-flat, no brushwork noise

This combination is called **Franco-Belgian ligne claire** ("clear line") style.

---

## What to Google

| Search term | Why |
|---|---|
| `Asterix comic panel scan high resolution` | Actual source pages, best for style |
| `Hergé Tintin ligne claire art` | The purest ligne claire — even cleaner outlines than Asterix |
| `Lucky Luke comic page scan` | Similar palette, very flat fills |
| `Franco-Belgian comics art style` | Overview of the genre |
| `Spirou Fantasio comic page` | High saturation, bold outlines |
| `Moebius comic art` | More detailed but same ink-line tradition |

**Tintin is actually a better training target than Asterix** — the ligne claire style is more geometrically consistent, which NST captures more reliably than Uderzo's slightly rounder, busier panels.

---

## Image selection tips for training

1. **Use interior comic panels, not covers** — covers often have non-representative rendering
2. **Avoid speech bubbles and text** — crop them out; the model will learn to replicate the lettering noise
3. **Pick panels with backgrounds** — sky, buildings, landscapes give the model the full colour vocabulary
4. **Avoid action blur panels** — motion lines confuse the texture model
5. **6–12 diverse panels** from the same artist works well; use `kaggle_multi_pic_trainer.ipynb` to train on them all at once
6. **Resolution**: 512×512 or larger per panel before feeding to the trainer

---

## Expected result

NST with ligne claire images produces: flat colour regions with strong edge contrast — your photos will look ink-outlined and colour-filled, which reads convincingly as "comic book". The bold outlines are especially strong because VGG16 Gram matrices respond well to high-frequency edge patterns.
