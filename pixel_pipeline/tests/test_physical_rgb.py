from __future__ import annotations

import numpy as np
from PIL import Image

from pixel_pipeline.modules.pbr.physical_rgb import sanitize_rgba_image


def _sprite_with_transparent_void() -> Image.Image:
    sprite = Image.new("RGBA", (6, 6), (0, 0, 0, 0))
    for x in range(6):
        sprite.putpixel((x, 0), (255, 0, 0, 255))
        sprite.putpixel((x, 5), (0, 255, 0, 255))
    for y in range(6):
        sprite.putpixel((0, y), (0, 0, 255, 255))
        sprite.putpixel((5, y), (255, 255, 0, 255))
    return sprite


def test_sanitize_rgba_preserves_alpha() -> None:
    sprite = _sprite_with_transparent_void()
    sanitized = sanitize_rgba_image(sprite)
    original_alpha = np.asarray(sprite.split()[-1], dtype=np.uint8)
    sanitized_alpha = np.asarray(sanitized.split()[-1], dtype=np.uint8)
    np.testing.assert_array_equal(sanitized_alpha, original_alpha)


def test_sanitize_rgba_fills_transparent_rgb() -> None:
    sprite = _sprite_with_transparent_void()
    sanitized = sanitize_rgba_image(sprite)
    rgb = np.asarray(sanitized.convert("RGB"), dtype=np.uint8)
    centre_colour = rgb[3, 3]
    assert centre_colour.mean() > 0, "Transparent void retained black RGB values"


def test_sanitize_rgba_idempotent_on_opaque() -> None:
    opaque = Image.new("RGBA", (4, 4), (12, 34, 56, 255))
    sanitized = sanitize_rgba_image(opaque)
    rgb = np.asarray(sanitized.convert("RGB"), dtype=np.uint8)
    np.testing.assert_array_equal(rgb, np.full((4, 4, 3), (12, 34, 56), dtype=np.uint8))
