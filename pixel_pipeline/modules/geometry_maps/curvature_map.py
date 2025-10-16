"""Curvature estimation using edge detection."""
from __future__ import annotations

from PIL import Image, ImageFilter, ImageOps

from .height_map import generate as generate_height


def generate(image: Image.Image) -> Image.Image:
    """Highlight curvature regions from the height map."""

    height = generate_height(image)
    edges = height.convert("L").filter(ImageFilter.FIND_EDGES)
    enhanced = ImageOps.autocontrast(edges)
    alpha = height.split()[-1]
    curvature_rgb = Image.merge("RGB", (enhanced, enhanced, enhanced))
    return Image.merge("RGBA", (*curvature_rgb.split(), alpha))


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (80, 40, 120, 255))
    generate(sample).show()
