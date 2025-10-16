"""Generate specular maps based on luminance intensity."""
from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter


def generate(image: Image.Image) -> Image.Image:
    """Create a specular map emphasising bright regions."""

    rgba = image.convert("RGBA")
    luminance = rgba.convert("L")
    boosted = ImageEnhance.Brightness(luminance).enhance(1.2)
    blurred = boosted.filter(ImageFilter.GaussianBlur(radius=1))
    alpha = rgba.split()[-1]
    specular = Image.merge("RGBA", (blurred, blurred, blurred, alpha))
    return specular


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (200, 180, 120, 255))
    generate(sample).show()
