import pytest

pytest.importorskip("PIL")
from PIL import Image

from pixel_pipeline.modules.pbr import pipeline
from pixel_pipeline.modules.pbr.analysis import analyze_image
from pixel_pipeline.modules.pbr.generation import (
    generate_fuzz_accurate,
    generate_material_accurate,
    generate_porosity_accurate,
    generate_opacity_accurate,
)

EXPECTED_MAPS = {
    "metallic",
    "roughness",
    "normal",
    "height",
    "ao",
    "curvature",
    "transmission",
    "subsurface",
    "specular",
    "ior",
    "emissive",
    "structural",
    "porosity",
    "opacity",
    "fuzz",
    "material",
}


def _base_sprite() -> Image.Image:
    sprite = Image.new("RGBA", (16, 16), (90, 130, 180, 255))
    sprite.putpixel((4, 4), (200, 80, 40, 255))
    sprite.putpixel((10, 12), (60, 200, 120, 180))
    return sprite


def test_generation_functions_return_rgba() -> None:
    sprite = _base_sprite()
    analysis = analyze_image(sprite, None, {})
    fuzz = generate_fuzz_accurate(sprite, analysis)
    porosity = generate_porosity_accurate(analysis)
    material = generate_material_accurate(analysis)
    opacity = generate_opacity_accurate(analysis)
    for generated in (fuzz, porosity, material, opacity):
        assert generated.mode == "RGBA"
        assert generated.size == sprite.size


def test_pipeline_outputs_unified_maps() -> None:
    result = pipeline.generate_physically_accurate_pbr_maps(_base_sprite(), None, {})
    maps = result["maps"]
    assert set(maps) == EXPECTED_MAPS
    for name in EXPECTED_MAPS:
        assert maps[name].mode == "RGBA"
        assert maps[name].size == _base_sprite().size
