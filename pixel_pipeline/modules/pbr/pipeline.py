"""Primary orchestration for the physically accurate PBR pipeline."""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Tuple

import inspect
import logging
import numpy as np
from PIL import Image

from .alpha_utils import apply_alpha_to_maps, derive_alpha_map
from .analysis import AnalysisResult, analyze_image_comprehensive, calculate_entropy
from .generation import (
    generate_alpha_accurate,
    generate_ao_physically_accurate,
    generate_curvature_enhanced,
    generate_emissive_accurate,
    generate_height_from_normal,
    generate_ior_physically_accurate,
    generate_metallic_physically_accurate,
    generate_normal_enhanced,
    generate_roughness_physically_accurate,
    generate_specular_coherent,
    generate_structural_distinct,
    generate_subsurface_accurate,
    generate_transmission_physically_accurate,
)
from .validation import auto_correct_failed_maps, log_corrections_applied, validate_all_maps

LOGGER = logging.getLogger("pixel_pipeline.pbr.pipeline")


def _resolve_generator(
    primary_import: Callable[[], Callable[[Image.Image, AnalysisResult], Image.Image] | None],
    fallback_module_path: str | None = None,
    fallback_attr: str = "generate",
) -> Callable[[Image.Image, AnalysisResult], Image.Image] | None:
    try:
        generator = primary_import()
        if generator is not None:
            return generator
    except ImportError:
        generator = None

    if fallback_module_path is None:
        return generator

    try:
        module = __import__(fallback_module_path, fromlist=[fallback_attr])
    except ImportError:
        return generator

    return getattr(module, fallback_attr, None)


try:
    from pixel_pipeline.modules.pixelart_maps.fuzz_map import generate_fuzz_enhanced as _fuzz_generator
except ImportError:
    _fuzz_generator = None

if _fuzz_generator is None:
    _fuzz_generator = _resolve_generator(
        lambda: None,
        "pixel_pipeline.modules.pixelart_maps.fuzz_map",
        "generate",
    )

try:
    from pixel_pipeline.modules.semantic_maps.material_type import (
        generate_material_semantic as _material_generator,
    )
except ImportError:
    _material_generator = None

if _material_generator is None:
    _material_generator = _resolve_generator(
        lambda: None,
        "pixel_pipeline.modules.semantic_maps.material_type",
        "generate",
    )


def _import_porosity_generator() -> Callable[[Image.Image, AnalysisResult], Image.Image] | None:
    try:
        from pixel_pipeline.modules.surface_maps.porosity_map import (
            generate_porosity_physically_accurate,
        )

        return generate_porosity_physically_accurate
    except ImportError:
        try:
            from pixel_pipeline.modules.pixelart_maps.porosity_map import (
                generate_porosity_physically_accurate,
            )

            return generate_porosity_physically_accurate
        except ImportError:
            return None


_porosity_generator = _import_porosity_generator()

if _porosity_generator is None:
    _porosity_generator = _resolve_generator(
        lambda: None,
        "pixel_pipeline.modules.pixelart_maps.porosity_map",
        "generate",
    )


generate_fuzz_enhanced = _fuzz_generator
generate_material_semantic = _material_generator
generate_porosity_physically_accurate = _porosity_generator


def _fallback_map(base_img: Image.Image, name: str) -> Image.Image:
    size = base_img.size
    alpha = base_img.split()[-1] if "A" in base_img.getbands() else Image.new("L", size, 255)
    channel = Image.new("L", size, 0 if name in {"opacity", "transmission", "metallic"} else 128)
    return Image.merge("RGBA", (channel, channel, channel, alpha))


def _call_generator(
    generator: Callable[..., Image.Image],
    base_img: Image.Image,
    analysis: AnalysisResult,
) -> Image.Image:
    try:
        signature = inspect.signature(generator)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        has_var_positional = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        if has_var_positional or len(positional) >= 2:
            return generator(base_img, analysis)
        if len(positional) == 1:
            return generator(base_img)

    try:
        return generator(base_img, analysis)
    except TypeError:
        return generator(base_img)


def _generate_optional_map(
    name: str,
    generator: Callable[..., Image.Image] | None,
    base_img: Image.Image,
    analysis: AnalysisResult,
) -> Image.Image:
    if generator is None:
        LOGGER.warning("Generator for %s map unavailable; using fallback", name)
        return _fallback_map(base_img, name)
    try:
        result = _call_generator(generator, base_img, analysis)
    except Exception as exc:  # pragma: no cover - defensive log branch
        LOGGER.warning("Failed to generate %s map (%s); using fallback", name, exc)
        return _fallback_map(base_img, name)
    if not isinstance(result, Image.Image):  # pragma: no cover - safety
        LOGGER.warning("Generator for %s returned non-image result; using fallback", name)
        return _fallback_map(base_img, name)
    return result


def _ensure_current_maps(base_img: Image.Image, current_maps: Mapping[str, Image.Image]) -> Dict[str, Image.Image]:
    required = {
        "opacity",
        "transmission",
        "metallic",
        "ior",
        "roughness",
        "specular",
        "subsurface",
        "structural",
        "normal",
        "height",
        "ao",
        "curvature",
        "emissive",
    }
    prepared: Dict[str, Image.Image] = {}
    for key in required:
        if key in current_maps and current_maps[key] is not None:
            prepared[key] = current_maps[key]
        else:
            LOGGER.debug("Falling back to placeholder for missing %s map", key)
            prepared[key] = _fallback_map(base_img, key)
    # include any extra maps unchanged
    for key, value in current_maps.items():
        if key not in prepared:
            prepared[key] = value
    return prepared


def _update_final_maps(
    base_img: Image.Image,
    analysis: AnalysisResult,
    current_maps: Mapping[str, Image.Image],
) -> Dict[str, Image.Image]:
    final_maps: Dict[str, Image.Image] = dict(current_maps)

    final_maps["opacity"] = generate_alpha_accurate(analysis, current_maps.get("opacity"))
    final_maps["transmission"] = generate_transmission_physically_accurate(analysis, current_maps.get("transmission"))
    final_maps["metallic"] = generate_metallic_physically_accurate(base_img, analysis, current_maps.get("metallic"))
    final_maps["ior"] = generate_ior_physically_accurate(
        final_maps["transmission"],
        analysis.material_analysis,
        analysis,
    )
    final_maps["roughness"] = generate_roughness_physically_accurate(base_img, analysis, final_maps["metallic"])
    final_maps["specular"] = generate_specular_coherent(final_maps["roughness"], analysis)
    final_maps["subsurface"] = generate_subsurface_accurate(analysis, final_maps["transmission"])
    final_maps["structural"] = generate_structural_distinct(analysis, current_maps.get("structural"))

    final_maps["normal"] = generate_normal_enhanced(base_img, analysis)
    final_maps["height"] = generate_height_from_normal(final_maps["normal"], analysis)
    final_maps["ao"] = generate_ao_physically_accurate(analysis)
    final_maps["curvature"] = generate_curvature_enhanced(analysis)
    final_maps["emissive"] = generate_emissive_accurate(base_img, analysis)

    final_maps["fuzz"] = _generate_optional_map("fuzz", generate_fuzz_enhanced, base_img, analysis)
    final_maps["material"] = _generate_optional_map(
        "material",
        generate_material_semantic,
        base_img,
        analysis,
    )
    final_maps["porosity"] = _generate_optional_map(
        "porosity",
        generate_porosity_physically_accurate,
        base_img,
        analysis,
    )

    return final_maps


def _enforce_rgba_alpha(
    base_img: Image.Image,
    maps: Mapping[str, Image.Image],
    analysis: AnalysisResult,
) -> tuple[Dict[str, Image.Image], np.ndarray]:
    alpha_map = derive_alpha_map(base_img, maps, analysis)
    updated_maps = apply_alpha_to_maps(maps, alpha_map)
    return updated_maps, alpha_map


def generate_physically_accurate_pbr_maps(
    base_img: Image.Image,
    bg_img: Image.Image | None,
    current_maps: Mapping[str, Image.Image],
) -> Dict[str, object]:
    prepared_maps = _ensure_current_maps(base_img, current_maps)
    analysis = analyze_image_comprehensive(base_img, bg_img, prepared_maps)
    final_maps = _update_final_maps(base_img, analysis, prepared_maps)
    final_maps, alpha_map = _enforce_rgba_alpha(base_img, final_maps, analysis)

    validation_report = validate_all_maps(final_maps, analysis)
    corrections: Tuple[str, ...] = ()
    if validation_report.has_critical_issues():
        final_maps, applied = auto_correct_failed_maps(final_maps, analysis, validation_report)
        corrections = tuple(applied)
        log_corrections_applied(corrections)
        final_maps, alpha_map = _enforce_rgba_alpha(base_img, final_maps, analysis)
        validation_report = validate_all_maps(final_maps, analysis)
    final_validation = validate_all_maps(final_maps, analysis)
    if not final_validation.passes_all_critical():
        LOGGER.warning("Final maps still report issues: %s", final_validation.issues)
    quality_report = generate_quality_report(final_maps, analysis)
    return {
        "maps": final_maps,
        "analysis": analysis,
        "validation": final_validation,
        "corrections_applied": corrections,
        "quality_report": quality_report,
        "alpha": alpha_map,
    }


def generate_quality_report(maps: Mapping[str, Image.Image], analysis: AnalysisResult | None = None) -> Dict[str, object]:
    report = {
        "flatness_analysis": {},
        "entropy_metrics": {},
        "recommendations": [],
    }
    for name, image in maps.items():
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        _, counts = np.unique((array * 255).astype(np.uint8), return_counts=True)
        uniform_ratio = float(counts.max() / counts.sum()) if counts.size else 1.0
        report["flatness_analysis"][name] = {
            "min": float(array.min()),
            "max": float(array.max()),
            "ptp": float(array.max() - array.min()),
            "uniform_ratio": uniform_ratio,
        }
        report["entropy_metrics"][name] = calculate_entropy(array)
    if analysis is not None:
        metallic = maps.get("metallic")
        if metallic is not None:
            array = np.asarray(metallic.convert("L"), dtype=np.float32) / 255.0
            organic = analysis.material_analysis.likelihoods.get("organic")
            if organic is not None and np.any(organic > 0.4) and np.any(array > 0.1):
                report["recommendations"].append("Inspect metallic map for organic overlaps")
    return report


__all__ = [
    "generate_physically_accurate_pbr_maps",
    "generate_quality_report",
]
