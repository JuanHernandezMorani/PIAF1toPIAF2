"""Opacity map derived from the alpha channel."""
from __future__ import annotations

import numpy as np
from PIL import Image


def generate(image: Image.Image) -> Image.Image:
    """Binarize the alpha channel to create an opacity mask."""

    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    threshold = np.where(alpha > 128, 255, 64).astype(np.uint8)
    opacity_rgba = np.dstack([threshold, threshold, threshold, alpha])
    return Image.fromarray(opacity_rgba, mode="RGBA")


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (200, 40, 120, 200))
    generate(sample).show()
