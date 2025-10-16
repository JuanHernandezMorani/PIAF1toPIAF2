"""Structural segmentation map focusing on large shapes."""
from __future__ import annotations

from PIL import Image, ImageFilter


def generate(image: Image.Image) -> Image.Image:
    """Highlight structural regions using morphological filters."""

    rgba = image.convert("RGBA")
    blurred = rgba.filter(ImageFilter.BoxBlur(radius=1))
    downsampled = blurred.resize((max(1, rgba.width // 4), max(1, rgba.height // 4)), Image.NEAREST)
    upsampled = downsampled.resize(rgba.size, Image.NEAREST)
    structure = Image.blend(rgba, upsampled, alpha=0.6).convert("RGB")
    alpha = rgba.split()[-1]
    return Image.merge("RGBA", (*structure.split(), alpha))


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (20, 200, 60, 255))
    generate(sample).show()
