"""Approximate ambient occlusion from blurred height maps."""
from __future__ import annotations

from PIL import Image, ImageFilter, ImageOps

from .height_map import generate as generate_height


def generate(image: Image.Image) -> Image.Image:
    """Create a soft ambient occlusion approximation."""

    height = generate_height(image)
    blurred = height.filter(ImageFilter.GaussianBlur(radius=2))
    inverted = ImageOps.invert(blurred.convert("RGB"))
    alpha = height.split()[-1]
    return Image.merge("RGBA", (*inverted.split(), alpha))


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (40, 80, 120, 255))
    generate(sample).show()
