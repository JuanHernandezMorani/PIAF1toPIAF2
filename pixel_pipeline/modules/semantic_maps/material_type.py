"""Semantic material classification based on dominant color."""
from __future__ import annotations

import numpy as np
from PIL import Image

from ...core import utils_color

__all__ = ["generate", "classify_color"]

_MATERIAL_COLORS = {
    "metal": (192, 192, 210, 255),
    "wood": (140, 90, 60, 255),
    "stone": (130, 130, 130, 255),
    "organic": (90, 140, 80, 255),
    "liquid": (70, 120, 200, 255),
}


def classify_color(rgb: tuple[float, float, float]) -> str:
    h, s, l = utils_color.rgb_to_hsl(tuple(int(c) for c in rgb))

    if s < 0.25 and l > 0.6:
        return "metal"

    if 0.02 < h < 0.13 and 0.3 < s < 0.8:
        return "wood"

    if (0.25 < h < 0.45 and s > 0.3) or (h < 0.08 and l < 0.4):
        return "organic"

    if (0.45 < h < 0.75 and s > 0.25 and 0.3 < l < 0.8):
        return "liquid"

    return "stone"


def generate(image: Image.Image) -> Image.Image:
    """Generate a semantic material map for *image*."""

    rgba = image.convert("RGBA")
    rgb_array = np.asarray(rgba.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    mean_color = rgb_array[alpha > 0].mean(axis=0) if np.any(alpha > 0) else (0, 0, 0)
    material = classify_color(tuple(mean_color))
    color = _MATERIAL_COLORS[material]
    semantic = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    semantic[..., 0] = color[0]
    semantic[..., 1] = color[1]
    semantic[..., 2] = color[2]
    semantic[..., 3] = alpha
    return Image.fromarray(semantic, mode="RGBA")


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (180, 160, 60, 255))
    generate(sample).show()
