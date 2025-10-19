"""Primary orchestration for the physically accurate PBR pipeline."""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import logging
import numpy as np
from PIL import Image

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


def _fallback_map(base_img: Image.Image, name: str) -> Image.Image:
    size = base_img.size
    alpha = base_img.split()[-1] if "A" in base_img.getbands() else Image.new("L", size, 255)
    channel = Image.new("L", size, 0 if name in {"opacity", "transmission", "metallic"} else 128)
    return Image.merge("RGBA", (channel, channel, channel, alpha))


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

    return final_maps


def generate_physically_accurate_pbr_maps(
    base_img: Image.Image,
    bg_img: Image.Image | None,
    current_maps: Mapping[str, Image.Image],
) -> Dict[str, object]:
    prepared_maps = _ensure_current_maps(base_img, current_maps)
    analysis = analyze_image_comprehensive(base_img, bg_img, prepared_maps)
    final_maps = _update_final_maps(base_img, analysis, prepared_maps)

    validation_report = validate_all_maps(final_maps, analysis)
    corrections: Tuple[str, ...] = ()
    if validation_report.has_critical_issues():
        final_maps, applied = auto_correct_failed_maps(final_maps, analysis, validation_report)
        corrections = tuple(applied)
        log_corrections_applied(corrections)
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
