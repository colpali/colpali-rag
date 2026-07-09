"""Render a ColPali similarity grid as a heatmap overlaid on the page image.

Pure NumPy + Pillow (no matplotlib) so it stays light. The similarity grid itself
is produced by the model in `embedder.py`; this module only turns a small
(rows x cols) score grid into a pretty, geometrically-aligned overlay.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

# An inferno-like colormap (control points), interpolated — perceptually uniform,
# reads well on both light and dark document backgrounds.
_CPS = np.array(
    [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
     [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]],
    dtype="float32",
)
_TS = np.linspace(0.0, 1.0, len(_CPS))


def normalize(grid: np.ndarray) -> np.ndarray:
    """Scale a raw score grid to [0, 1]."""
    g = np.asarray(grid, dtype="float32")
    lo, hi = float(g.min()), float(g.max())
    return (g - lo) / (hi - lo + 1e-6)


def colorize(x01: np.ndarray) -> np.ndarray:
    """Map values in [0,1] to inferno RGB (uint8, shape (...,3))."""
    x = np.clip(x01, 0.0, 1.0)
    rgb = np.stack([np.interp(x, _TS, _CPS[:, k]) for k in range(3)], axis=-1)
    return rgb.astype("uint8")


def overlay(
    page: Image.Image,
    grid: np.ndarray,
    *,
    alpha: float = 0.62,
    gamma: float = 0.85,
    dim_base: float = 0.25,
) -> Image.Image:
    """Overlay a (rows x cols) similarity grid onto the page image.

    grid rows correspond to the page's vertical (height) axis, cols to horizontal.
    Values are normalized, gamma-shaped, upsampled to the page size, then alpha-
    blended. Cold regions are dimmed a touch (dim_base) so hot regions pop while
    the page text stays readable.
    """
    page = page.convert("RGB")
    W, H = page.size
    g = normalize(grid)
    # upsample the coarse grid to full page resolution (bilinear)
    hp = Image.fromarray((g * 255).astype("uint8"), mode="L").resize((W, H), Image.BILINEAR)
    heat = (np.asarray(hp).astype("float32") / 255.0) ** gamma          # (H, W) in [0,1]
    heat_rgb = colorize(heat).astype("float32")                          # (H, W, 3)

    # dim the page proportionally to how *cold* each pixel is, then blend heat on top
    base = np.asarray(page).astype("float32")
    base = base * (1.0 - dim_base * (1.0 - heat[..., None]))
    a = (heat * alpha)[..., None]
    out = base * (1.0 - a) + heat_rgb * a
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))


def to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"
