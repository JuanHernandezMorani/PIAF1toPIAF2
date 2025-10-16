"""Metalness estimation for pixel art assets."""
from __future__ import annotations

import numpy as np
from PIL import Image


def _saturation(image: Image.Image) -> np.ndarray:
    hsv = image.convert("HSV")
    return np.asarray(hsv.split()[1], dtype=np.float32) / 255.0


def generate(image: Image.Image) -> Image.Image:
    """Estimate metallic regions using color saturation."""

    rgba = image.convert("RGBA")
    saturation = _saturation(rgba)
    threshold = np.where(saturation > 0.45, 255, (saturation * 255).astype(np.uint8))
    metallic_rgb = np.stack([threshold, threshold, threshold], axis=-1).astype(np.uint8)
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    metallic_rgba = np.dstack([metallic_rgb, alpha])
    return Image.fromarray(metallic_rgba, mode="RGBA")


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (160, 160, 200, 255))
    generate(sample).show()
