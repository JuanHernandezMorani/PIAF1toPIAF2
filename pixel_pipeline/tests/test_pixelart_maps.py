"""Tests for additional unified PBR maps."""
from __future__ import annotations

import pytest

pytest.importorskip("PIL")
from PIL import Image

from pixel_pipeline.modules.pbr.analysis import analyze_image
from pixel_pipeline.modules.pbr.generation import (
    generate_alpha_accurate,
    generate_opacity_accurate,
    generate_material_accurate,
)


def _sprite() -> tuple[Image.Image, object]:
    sprite = Image.new("RGBA", (8, 8), (100, 120, 140, 255))
    sprite.putpixel((2, 2), (180, 40, 90, 255))
    sprite.putpixel((5, 5), (40, 200, 120, 200))
    analysis = analyze_image(sprite, None, {})
    return sprite, analysis


def test_opacity_map_preserves_size() -> None:
    sprite, analysis = _sprite()
    opacity = generate_opacity_accurate(analysis)
    assert opacity.size == sprite.size
    assert opacity.mode == "RGBA"


def test_alpha_generation_returns_image() -> None:
    sprite, analysis = _sprite()
    alpha_img = generate_alpha_accurate(analysis, None)
    assert alpha_img.mode == "RGBA"
    assert alpha_img.size == sprite.size


def test_material_map_contains_color_information() -> None:
    sprite, analysis = _sprite()
    material_map = generate_material_accurate(analysis)
    assert material_map.mode == "RGBA"
    pixels = list(material_map.getdata())
    assert len({pixel[:3] for pixel in pixels}) >= 1
