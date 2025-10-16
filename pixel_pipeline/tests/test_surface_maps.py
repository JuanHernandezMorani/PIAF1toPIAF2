"""Unit tests for surface map generators."""
from __future__ import annotations

import pytest

pytest.importorskip("PIL")
from PIL import Image

from pixel_pipeline.modules.surface_maps import metallic_map, normal_map, roughness_map, specular_map


def _sample_image() -> Image.Image:
    return Image.new("RGBA", (8, 8), (120, 80, 200, 255))


def test_normal_map_dimensions() -> None:
    result = normal_map.generate(_sample_image())
    assert result.size == (8, 8)
    assert result.mode == "RGBA"


def test_specular_map_is_rgba() -> None:
    result = specular_map.generate(_sample_image())
    assert result.mode == "RGBA"


def test_roughness_inversion() -> None:
    specular = specular_map.generate(_sample_image())
    roughness = roughness_map.generate(_sample_image())
    assert roughness.size == specular.size


def test_metallic_map_alpha_preserved() -> None:
    image = _sample_image()
    metallic = metallic_map.generate(image)
    assert metallic.split()[-1].getextrema() == image.split()[-1].getextrema()
