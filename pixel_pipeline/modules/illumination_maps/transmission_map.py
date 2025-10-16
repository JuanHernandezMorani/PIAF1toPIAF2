"""Transmission map approximating translucent regions."""
from __future__ import annotations

import numpy as np
from PIL import Image


def generate(image: Image.Image) -> Image.Image:
    """Infer transmission from semi-transparent pixels."""

    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.split()[-1], dtype=np.float32) / 255.0
    transmission = np.clip(1.0 - np.abs(alpha - 0.5) * 2.0, 0.0, 1.0)
    channel = (transmission * 255.0).astype(np.uint8)
    transmission_rgba = np.dstack([channel, channel, channel, np.asarray(rgba.split()[-1], dtype=np.uint8)])
    return Image.fromarray(transmission_rgba, mode="RGBA")


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (120, 120, 200, 128))
    generate(sample).show()
