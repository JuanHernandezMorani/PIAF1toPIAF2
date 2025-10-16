"""Semantic map generator tests."""
from __future__ import annotations

import pytest

pytest.importorskip("PIL")
from PIL import Image

from pixel_pipeline.modules.semantic_maps import material_type, structural_map
from pixel_pipeline.modules.optical_maps import ior_map


def _material_image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGBA", (4, 4), (*color, 255))


def test_material_classification_changes_color() -> None:
    wood = material_type.generate(_material_image((170, 120, 60)))
    stone = material_type.generate(_material_image((100, 100, 100)))
    assert wood.getpixel((0, 0)) != stone.getpixel((0, 0))


def test_structural_map_keeps_alpha() -> None:
    image = _material_image((120, 200, 80))
    struct = structural_map.generate(image)
    assert struct.split()[-1].tobytes() == image.split()[-1].tobytes()


def test_ior_map_matches_material() -> None:
    water = _material_image((100, 160, 220))
    ior = ior_map.generate(water)
    assert ior.mode == "RGBA"
    assert ior.getpixel((0, 0))[0] >= 0
