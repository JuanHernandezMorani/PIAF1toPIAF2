"""Unit tests for unified PBR surface map generators."""
from __future__ import annotations

import pytest

pytest.importorskip("PIL")
from PIL import Image

from pixel_pipeline.modules.pbr.analysis import analyze_image
from pixel_pipeline.modules.pbr.generation import (
    generate_curvature_enhanced,
    generate_height_from_normal,
    generate_normal_enhanced,
    generate_roughness_physically_accurate,
    generate_specular_coherent,
    generate_metallic_physically_accurate,
)


def _sample_analysis() -> tuple[Image.Image, object]:
    image = Image.new("RGBA", (8, 8), (120, 80, 200, 255))
    analysis = analyze_image(image, None, {})
    return image, analysis


def test_normal_map_dimensions() -> None:
    image, analysis = _sample_analysis()
    normal = generate_normal_enhanced(image, analysis)
    assert normal.size == image.size
    assert normal.mode == "RGBA"


def test_height_generation_from_normal() -> None:
    image, analysis = _sample_analysis()
    normal = generate_normal_enhanced(image, analysis)
    height = generate_height_from_normal(normal, analysis)
    assert height.size == image.size
    assert height.mode == "RGBA"


def test_roughness_and_specular_consistency() -> None:
    image, analysis = _sample_analysis()
    metallic = generate_metallic_physically_accurate(image, analysis, None)
    roughness = generate_roughness_physically_accurate(image, analysis, metallic)
    specular = generate_specular_coherent(roughness, analysis)
    assert roughness.size == specular.size == image.size
    assert roughness.mode == specular.mode == "RGBA"


def test_curvature_matches_image_size() -> None:
    image, analysis = _sample_analysis()
    curvature = generate_curvature_enhanced(analysis)
    assert curvature.size == image.size
    assert curvature.mode == "RGBA"
