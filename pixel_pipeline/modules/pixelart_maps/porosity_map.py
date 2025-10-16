"""Porosity approximation using ordered dithering."""
from __future__ import annotations

import numpy as np
from PIL import Image

_BAYER_MATRIX = (
    np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float32,
    )
    / 16.0
)


def generate(image: Image.Image) -> Image.Image:
    """Create a porosity map by applying Bayer dithering."""

    rgba = image.convert("RGBA")
    luminance = np.asarray(rgba.convert("L"), dtype=np.float32) / 255.0
    tiled = np.tile(_BAYER_MATRIX, (luminance.shape[0] // 4 + 1, luminance.shape[1] // 4 + 1))
    tiled = tiled[: luminance.shape[0], : luminance.shape[1]]
    porous = (luminance + tiled * 0.25).clip(0, 1)
    porous = (porous * 255.0).astype(np.uint8)
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    porous_rgba = np.dstack([porous, porous, porous, alpha])
    return Image.fromarray(porous_rgba, mode="RGBA")


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (100, 120, 140, 255))
    generate(sample).show()
