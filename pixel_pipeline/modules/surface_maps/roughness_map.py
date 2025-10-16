"""Derive roughness maps from the inverse of specular intensity."""
from __future__ import annotations

from PIL import Image, ImageOps

from .specular_map import generate as generate_specular


def generate(image: Image.Image) -> Image.Image:
    """Create a roughness map by inverting the specular map."""

    specular = generate_specular(image)
    inverted_rgb = ImageOps.invert(specular.convert("RGB"))
    alpha = specular.split()[-1]
    return Image.merge("RGBA", (*inverted_rgb.split(), alpha))


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (200, 180, 120, 255))
    generate(sample).show()
