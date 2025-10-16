"""Tests for pixel art specific map generators."""
from __future__ import annotations

import pytest

pytest.importorskip("PIL")
from PIL import Image

from pixel_pipeline.modules.pixelart_maps import fuzz_map, porosity_map


def _sprite() -> Image.Image:
    sprite = Image.new("RGBA", (8, 8), (100, 120, 140, 255))
    sprite.putpixel((2, 2), (180, 40, 90, 255))
    sprite.putpixel((5, 5), (40, 200, 120, 200))
    return sprite


def test_porosity_map_output() -> None:
    result = porosity_map.generate(_sprite())
    assert result.mode == "RGBA"
    assert result.size == _sprite().size


def test_fuzz_map_noise_variation() -> None:
    first = fuzz_map.generate(_sprite())
    second = fuzz_map.generate(_sprite())
    assert first.tobytes() != second.tobytes()
