"""Approximate index of refraction based on material classification."""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..semantic_maps import material_type

_IOR_VALUES = {
    "metal": 2.5,
    "wood": 1.4,
    "stone": 1.54,
    "organic": 1.38,
    "liquid": 1.33,
}


def generate(image: Image.Image) -> Image.Image:
    """Generate an IOR map encoded in grayscale."""

    rgba = image.convert("RGBA")
    rgb_array = np.asarray(rgba.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    mean_color = rgb_array[alpha > 0].mean(axis=0) if np.any(alpha > 0) else (0, 0, 0)
    material = material_type.classify_color(tuple(mean_color))
    ior = _IOR_VALUES.get(material, 1.4)
    normalized = (ior - 1.0) / 1.5
    channel = np.full(alpha.shape, int(np.clip(normalized, 0.0, 1.0) * 255), dtype=np.uint8)
    ior_rgba = np.dstack([channel, channel, channel, alpha])
    return Image.fromarray(ior_rgba, mode="RGBA")


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (120, 160, 200, 255))
    generate(sample).show()
