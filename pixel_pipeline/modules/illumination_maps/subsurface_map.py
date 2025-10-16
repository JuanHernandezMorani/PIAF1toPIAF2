"""Subsurface scattering approximation."""
from __future__ import annotations

from PIL import Image, ImageFilter


def generate(image: Image.Image) -> Image.Image:
    """Simulate subsurface scattering by blurring the red channel."""

    rgba = image.convert("RGBA")
    red, green, blue, alpha = rgba.split()
    blurred_red = red.filter(ImageFilter.GaussianBlur(radius=1.5))
    blend = Image.merge("RGB", (blurred_red, green, blue))
    subsurface = Image.merge("RGBA", (*blend.split(), alpha))
    return subsurface


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (200, 40, 60, 255))
    generate(sample).show()
