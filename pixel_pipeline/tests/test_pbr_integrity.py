"""Regression tests ensuring PBR integrity improvements."""

from __future__ import annotations

import pytest
pytest.importorskip("numpy")
import numpy as np
from PIL import Image

from pixel_pipeline.modules.pbr import pipeline


@pytest.fixture(scope="module")
def _sample_sprite() -> Image.Image:
    sprite = Image.new("RGBA", (32, 32), (140, 160, 190, 0))
    for x in range(8, 24):
        for y in range(8, 24):
            alpha = 255 if (x + y) % 2 == 0 else 200
            sprite.putpixel((x, y), (180, 120, 90, alpha))
    return sprite


@pytest.fixture(scope="module")
def pipeline_result(_sample_sprite: Image.Image):
    return pipeline.generate_physically_accurate_pbr_maps(_sample_sprite, None, {})


@pytest.fixture(scope="module")
def out_maps(pipeline_result):
    return pipeline_result["maps"]


@pytest.fixture(scope="module")
def analysis(pipeline_result):
    return pipeline_result["analysis"]


def test_no_black_halo_on_alpha_edge(out_maps):
    rgba = np.array(out_maps["structural"])
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    edge = (alpha > 0) & (alpha < 255)
    if not np.any(edge):  # pragma: no cover - safety for degenerate masks
        pytest.skip("No partial alpha edges to evaluate")
    assert (rgb[edge].mean(axis=1) > 5).all(), "Detected black halos around alpha edges"


def test_metallic_for_organic_materials(analysis, out_maps):
    if analysis.material_class == "organic":
        metallic = np.array(out_maps["metallic"].convert("L")) / 255.0
        assert float(np.mean(metallic)) < 0.1, "Organic materials should not be metallic"
