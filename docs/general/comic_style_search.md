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

---

# Training a Manga Black-and-White Style Model

## What makes manga art style distinctive

- **High-contrast ink lines** — thick/thin brush or pen strokes on pure white
- **Screentone dot/hatch patterns** — printed halftone dots used for grey shading
- **Flat black silhouettes** — large solid-black shadow areas
- **Speed/motion lines** — radial or parallel ruling lines
- **No colour** — pure B&W; model learns ink density, not hue

Keep source images as **RGB** (3 channels) even though they are greyscale — the model architecture expects 3-channel input.

---

## What to Google

### Line art (clean ink)
| Search term | Why |
|---|---|
| `manga panel line art black white` | Generic high-contrast panels |
| `manga screentone pattern scan` | Classic halftone dot texture |
| `shounen manga page high contrast scan` | Action-heavy, bold lines |

### Series with strong, consistent styles
| Search term | Style characteristics |
|---|---|
| `Dragon Ball manga page Toriyama` | Bold thick lines, clean silhouettes — good starter |
| `Akira manga page Otomo` | Ultra-detailed crosshatching, cinematic — **best all-round choice** |
| `Berserk manga page Miura` | Dense hatching, extreme contrast |
| `One Piece manga page Oda` | Energetic line-weight variation |
| `Vagabond manga Inoue` | Brushstroke ink, high artistic quality |

### Texture-specific
| Search term | Why |
|---|---|
| `manga screentone dot pattern` | Classic halftone dot texture |
| `manga crosshatch shading panel` | Hatched grey shading |
| `manga speed lines panel` | Radial motion-line effect |

---

## Image selection tips for manga training

1. **Prefer pages with mostly black ink on white** — avoid dialogue-heavy text-only pages
2. **Pick action or landscape panels** — more visual texture for the model to learn
3. **Stay with one artist** — 20–30 pages from the same series; style consistency is critical
4. **Scan resolution 300+ DPI** if possible; JPEG compression artefacts on screentones hurt quality
5. **Crop speech bubbles** — same advice as for ligne claire

---

## Expected result

Manga-style transfer produces photos with **ink-line outlines and screentone-like halftone shading**. Colours are desaturated toward grey/black tones. The effect is strongest on portrait and figure shots where the model maps facial shadows to solid screentone regions.

**Best starting point:** Akira — extreme contrast and detailed crosshatching give the most recognisable result on photographs.
