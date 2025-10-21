import types

import pytest
np = pytest.importorskip("numpy")
from PIL import Image

from pixel_pipeline.modules.pbr.alpha_utils import apply_alpha, _enhanced_alpha_bleed_rgba_v3_improved
from pixel_pipeline.modules.pbr.pipeline import _preserve_foreground_texture_transmission
from pixel_pipeline.modules.pbr.validation import _evaluate_quality_checks_improved


def _analysis_stub(shape):
    height, width = shape
    dummy_false_metal = np.zeros(shape, dtype=np.float32)
    material = types.SimpleNamespace(likelihoods={}, false_metal_risks=dummy_false_metal)
    return types.SimpleNamespace(
        alpha=np.ones(shape, dtype=np.float32),
        mask=np.ones(shape, dtype=np.float32),
        geometric_features={},
        material_analysis=material,
        base_image=Image.new("RGBA", (width, height)),
        background_image=None,
    )


def test_anti_halo_simple():
    size = 32
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., 3] = 0

    core = slice(8, size - 8)
    rgba[core, core, :3] = 255
    rgba[core, core, 3] = 255

    ring_mask = np.zeros((size, size), dtype=bool)
    ring_mask[7 : size - 7, 7 : size - 7] = True
    ring_mask[8 : size - 8, 8 : size - 8] = False
    rgba[ring_mask, :3] = 0
    rgba[ring_mask, 3] = 150

    image = Image.fromarray(rgba, mode="RGBA")
    result, changed = _enhanced_alpha_bleed_rgba_v3_improved(image, iterations=6, radius=2)

    assert changed is True
    arr = np.asarray(result, dtype=np.uint8)
    alpha = arr[..., 3] / 255.0
    edge_mask = (alpha > 0.05) & (alpha < 0.95)
    assert np.count_nonzero(edge_mask) > 0
    rgb = arr[..., :3] / 255.0
    dark_ratio = np.count_nonzero(np.linalg.norm(rgb[edge_mask], axis=1) < 0.05) / np.count_nonzero(edge_mask)
    assert dark_ratio < 0.01


def test_transmission_preservation_simple():
    size = 8
    gradient = np.linspace(0.0, 1.0, size * size, dtype=np.float32).reshape(size, size)

    fg_rgba = np.zeros((size, size, 4), dtype=np.uint8)
    fg_rgba[..., :3] = (gradient[..., None] * 255).astype(np.uint8)
    fg_rgba[..., 3] = 255
    foreground = Image.fromarray(fg_rgba, mode="RGBA")

    background = Image.new("RGBA", (size, size), (64, 64, 64, 255))

    candidate_gray = np.clip(0.3 + 0.2 * gradient, 0.0, 1.0)
    candidate = Image.fromarray((candidate_gray * 255).astype(np.uint8), mode="L").convert("RGBA")

    analysis = _analysis_stub((size, size))
    analysis.base_image = foreground

    preserved, issues = _preserve_foreground_texture_transmission(foreground, background, analysis, candidate)

    preserved_arr = np.asarray(preserved.convert("L"), dtype=np.float32) / 255.0
    mask = analysis.alpha > 0.05
    corr = np.corrcoef(gradient[mask], preserved_arr[mask])[0, 1]
    assert corr >= 0.7
    assert "foreground_texture_lost" not in issues


def test_chromatic_consistency_simple():
    size = 10
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    base_rgb = np.zeros((size, size, 3), dtype=np.uint8)
    base_rgb[..., 0] = (80 + 120 * x).astype(np.uint8)
    base_rgb[..., 1] = (100 + 60 * x[::-1]).astype(np.uint8)
    base_rgb[..., 2] = (90 + 100 * x).astype(np.uint8)
    base_image = Image.fromarray(base_rgb, mode="RGB")

    alpha_map = np.ones((size, size), dtype=np.float32)
    composite = apply_alpha(base_image, alpha_map)

    transmission_gray = np.linspace(0.2, 0.8, size * size, dtype=np.float32).reshape(size, size)
    transmission = Image.fromarray((transmission_gray * 255).astype(np.uint8), mode="L").convert("RGBA")

    maps = {"base_color": base_image, "transmission": transmission}
    foreground_texture = np.asarray(base_image.convert("L"), dtype=np.float32) / 255.0

    checks = _evaluate_quality_checks_improved(
        maps,
        base_image=base_image,
        composite=composite,
        foreground_texture=foreground_texture,
        fuzz_ok=True,
        transmission_ok=True,
    )

    assert checks["halos_eliminated"] is True
    assert checks["transmission_preserved"] is True
    assert checks["chromatic_consistency"] is True
    assert checks["chromatic_stable"] is True
