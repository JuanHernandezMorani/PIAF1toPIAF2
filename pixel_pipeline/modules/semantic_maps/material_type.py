"""Semantic material classification based on dominant color and texture cues."""
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


def detect_texture_context(rgb_array: np.ndarray) -> str:
    """Infer the most plausible neutral material class via grayscale texture statistics."""

    if rgb_array.size == 0:
        return "stone"

    gray = np.mean(rgb_array, axis=-1)
    texture_var = float(np.var(gray))
    histogram, _ = np.histogram(gray, bins=32, range=(0, 255), density=True)
    entropy = float(-np.sum(histogram * np.log2(histogram + 1e-9)))

    if entropy > 3.5 and texture_var < 1000:
        return "organic"  # piel, madera, vegetación
    if entropy < 2.0:
        return "metal"  # superficies uniformes reflectantes
    return "stone"


def generate(image: Image.Image) -> Image.Image:
    """Generate a semantic material map enriched with neutral-tone texture inference."""

    rgba = image.convert("RGBA")
    rgb_array = np.asarray(rgba.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    valid_mask = alpha > 0
    if np.any(valid_mask):
        valid_rgb = rgb_array[valid_mask]
    else:
        valid_rgb = rgb_array.reshape(-1, 3)

    mean_color = valid_rgb.mean(axis=0) if valid_rgb.size else np.array((0.0, 0.0, 0.0))
    if np.allclose(mean_color, (128, 128, 128), atol=15):
        material = detect_texture_context(valid_rgb)
    else:
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
