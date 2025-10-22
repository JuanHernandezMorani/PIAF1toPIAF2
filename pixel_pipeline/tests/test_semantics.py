import pytest

pytest.importorskip("PIL")
from PIL import Image

from pixel_pipeline.modules.pbr.analysis import analyze_image
from pixel_pipeline.modules.pbr.generation import (
    generate_material_accurate,
    generate_structural_distinct,
    generate_ior_physically_accurate,
    generate_transmission_physically_accurate,
)


def _material_image(color: tuple[int, int, int]) -> tuple[Image.Image, object]:
    image = Image.new("RGBA", (4, 4), (*color, 255))
    analysis = analyze_image(image, None, {})
    return image, analysis


def test_material_map_changes_with_color() -> None:
    _, analysis_wood = _material_image((170, 120, 60))
    _, analysis_stone = _material_image((100, 100, 100))
    wood_map = generate_material_accurate(analysis_wood)
    stone_map = generate_material_accurate(analysis_stone)
    assert wood_map.getpixel((0, 0)) != stone_map.getpixel((0, 0))


def test_structural_map_keeps_dimensions() -> None:
    image, analysis = _material_image((120, 200, 80))
    struct = generate_structural_distinct(analysis, None)
    assert struct.size == image.size
    assert struct.mode == "RGBA"


def test_ior_depends_on_transmission() -> None:
    image, analysis = _material_image((100, 160, 220))
    transmission = generate_transmission_physically_accurate(analysis, None)
    ior_map = generate_ior_physically_accurate(transmission, analysis.material_analysis, analysis)
    assert ior_map.mode == "RGBA"
    assert ior_map.size == image.size
