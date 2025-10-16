"""Fuzz map adds controlled noise to sprite edges."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def generate(image: Image.Image) -> Image.Image:
    """Generate a fuzz map using edge masks and random noise."""

    rgba = image.convert("RGBA")
    edges = rgba.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_array = np.asarray(edges, dtype=np.float32) / 255.0
    noise = np.random.default_rng().uniform(-0.2, 0.2, edge_array.shape)
    fuzz = np.clip(edge_array + noise, 0.0, 1.0)
    fuzz_bytes = (fuzz * 255.0).astype(np.uint8)
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    fuzz_rgba = np.dstack([fuzz_bytes, fuzz_bytes, fuzz_bytes, alpha])
    return Image.fromarray(fuzz_rgba, mode="RGBA")


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (180, 40, 120, 255))
    generate(sample).show()
