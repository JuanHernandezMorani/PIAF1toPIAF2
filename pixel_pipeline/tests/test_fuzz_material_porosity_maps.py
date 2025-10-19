"""Integration tests for newly reintegrated optional maps in the PBR pipeline."""
from __future__ import annotations

import pytest

pytest.importorskip("PIL")
from PIL import Image

from pixel_pipeline.modules.pbr import pipeline


def _base_sprite() -> Image.Image:
    sprite = Image.new("RGBA", (16, 16), (90, 130, 180, 255))
    sprite.putpixel((4, 4), (200, 80, 40, 255))
    sprite.putpixel((10, 12), (60, 200, 120, 180))
    return sprite


def test_pipeline_outputs_optional_maps() -> None:
    result = pipeline.generate_physically_accurate_pbr_maps(_base_sprite(), None, {})
    maps = result["maps"]
    for key in ("fuzz", "material", "porosity"):
        assert key in maps
        assert maps[key].mode == "RGBA"
        assert maps[key].size == _base_sprite().size


def test_pipeline_fallback_when_generators_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "generate_fuzz_enhanced", None, raising=False)

    def _broken_material(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "generate_material_semantic", _broken_material, raising=False)
    monkeypatch.setattr(pipeline, "generate_porosity_physically_accurate", None, raising=False)

    result = pipeline.generate_physically_accurate_pbr_maps(_base_sprite(), None, {})
    maps = result["maps"]
    assert maps["fuzz"].mode == "RGBA"
    assert maps["material"].mode == "RGBA"
    assert maps["porosity"].mode == "RGBA"
    expected_size = _base_sprite().size
    assert maps["fuzz"].size == expected_size
    assert maps["material"].size == expected_size
    assert maps["porosity"].size == expected_size
