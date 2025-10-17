"""Generate height maps from sprite luminance."""
from __future__ import annotations

import numpy as np
from PIL import Image


def generate(image: Image.Image) -> Image.Image:
    """Convert luminance to a height map."""

    rgba = image.convert("RGBA")
    luminance = np.asarray(rgba.convert("L"), dtype=np.float32)
    normalized = (luminance - luminance.min()) / max(np.ptp(luminance), 1.0)
    height = (normalized * 255.0).astype(np.uint8)
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    height_rgba = np.dstack([height, height, height, alpha])
    return Image.fromarray(height_rgba, mode="RGBA")


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (40, 80, 120, 255))
    generate(sample).show()
