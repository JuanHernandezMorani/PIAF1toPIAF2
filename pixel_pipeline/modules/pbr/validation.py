"""Validation and correction routines for the PBR pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import logging
import numpy as np
from PIL import Image

from .analysis import calculate_entropy
from .parameters import CRITICAL_PARAMETERS

LOGGER = logging.getLogger("pixel_pipeline.pbr.validation")

MAX_IOR = max(CRITICAL_PARAMETERS["IOR_TRANSLUCENT_RANGES"].values())
IOR_RANGE = max(MAX_IOR - CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"], 1e-3)


@dataclass
class ValidationReport:
    issues: Dict[str, List[str]]

    def has_critical_issues(self) -> bool:
        return any(self.issues.values())

    def passes_all_critical(self) -> bool:
        return not self.has_critical_issues()

    def items(self):  # pragma: no cover - convenience
        return self.issues.items()


VALIDATION_RULES = {
    "metallic": [
        "no_organic_false_positives",
        "binary_or_three_levels",
        "entropy_min",
    ],
    "ior": [
        "opaque_areas_have_1.0",
        "translucent_areas_have_physical_values",
        "not_completely_gray",
    ],
    "opacity": [
        "background_transparent",
        "object_opaque",
        "smooth_transitions",
        "not_completely_white",
    ],
    "transmission": [
        "only_in_translucent_materials",
        "zero_in_metals",
        "not_completely_black",
    ],
    "structural": [
        "different_from_base",
        "high_frequency_features",
        "not_flat",
    ],
    "subsurface": [
        "thin_sections_highlighted",
        "organic_materials_focused",
        "different_from_structural",
    ],
}


def _to_array(image_or_array) -> np.ndarray:
    if isinstance(image_or_array, np.ndarray):
        arr = image_or_array.astype(np.float32)
        if arr.max() > 1.0 + 1e-6:
            arr = arr / 255.0
        return arr
    if isinstance(image_or_array, Image.Image):
        return np.asarray(image_or_array.convert("L"), dtype=np.float32) / 255.0
    arr = np.asarray(image_or_array, dtype=np.float32)
    if arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    return arr


def _np_from_image(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _entropy(image: Image.Image) -> float:
    return calculate_entropy(_np_from_image(image))


def _uniform_ratio(array: np.ndarray) -> float:
    values, counts = np.unique((array * 255).astype(np.uint8), return_counts=True)
    if len(counts) == 0:
        return 1.0
    return float(counts.max() / counts.sum())


def _smoothness(array: np.ndarray) -> float:
    grad_y, grad_x = np.gradient(array)
    return float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))


def _binary_levels(array: np.ndarray) -> bool:
    unique = np.unique(np.round(array * 10) / 10.0)
    return len(unique) <= 3


def validate_all_maps(maps: Mapping[str, Image.Image], analysis) -> ValidationReport:
    """Run all validation rules and return a report."""

    issues: Dict[str, List[str]] = {name: [] for name in maps.keys()}
    mask = analysis.mask
    material = analysis.material_analysis

    metallic = _np_from_image(maps.get("metallic", Image.new("L", (1, 1))))
    organic = material.likelihoods.get("organic", np.zeros_like(metallic))
    if np.any(metallic > 0.05) and np.any(organic > 0.6):
        overlap = metallic * organic
        if float(overlap.max()) > CRITICAL_PARAMETERS["METALLIC_ORGANIC_TOLERANCE"]:
            issues["metallic"].append("no_organic_false_positives")
    if not _binary_levels(metallic):
        issues["metallic"].append("binary_or_three_levels")
    if _entropy(maps["metallic"]) < CRITICAL_PARAMETERS["MIN_ENTROPY"]:
        issues["metallic"].append("entropy_min")

    # IOR checks
    ior_array = _np_from_image(maps.get("ior", Image.new("L", (1, 1)))) * IOR_RANGE + CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"]
    transmission = _np_from_image(maps.get("transmission", Image.new("L", (1, 1))))
    opaque_mask = transmission <= CRITICAL_PARAMETERS["TRANSMISSION_OPAQUE_MAX"]
    if np.any(np.abs(ior_array[opaque_mask] - CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"]) > 0.01):
        issues["ior"].append("opaque_areas_have_1.0")
    translucent_mask = transmission > 0.1
    if np.any(translucent_mask):
        physical_values = []
        for mat, value in CRITICAL_PARAMETERS["IOR_TRANSLUCENT_RANGES"].items():
            likelihood = material.likelihoods.get(mat, np.zeros_like(transmission))
            physical_values.append(likelihood * np.abs(ior_array - value))
        if physical_values:
            stack = np.stack(physical_values, axis=0)
            if float(stack.min()) > 0.2:
                issues["ior"].append("translucent_areas_have_physical_values")
    if _uniform_ratio(_np_from_image(maps["ior"])) > CRITICAL_PARAMETERS["MAX_UNIFORM_RATIO"]:
        issues["ior"].append("not_completely_gray")

    # Opacity checks
    opacity = _np_from_image(maps.get("opacity", Image.new("L", (1, 1))))
    background_transparent = opacity[mask < 0.1].mean() <= CRITICAL_PARAMETERS["ALPHA_BACKGROUND_MAX"]
    object_opaque = opacity[mask > 0.5].mean() >= CRITICAL_PARAMETERS["ALPHA_OBJECT_MIN"]
    if not background_transparent:
        issues["opacity"].append("background_transparent")
    if not object_opaque:
        issues["opacity"].append("object_opaque")
    if _smoothness(opacity) > 0.2:
        issues["opacity"].append("smooth_transitions")
    if opacity.max() < 0.5 or _uniform_ratio(opacity) > CRITICAL_PARAMETERS["MAX_UNIFORM_RATIO"]:
        issues["opacity"].append("not_completely_white")

    # Transmission checks
    if np.all(transmission < 1e-3):
        issues["transmission"].append("not_completely_black")
    metal_like = material.likelihoods.get("metal", np.zeros_like(transmission)) > 0.5
    if np.any(transmission[metal_like] > CRITICAL_PARAMETERS["TRANSMISSION_METAL_MAX"] + 1e-3):
        issues["transmission"].append("zero_in_metals")
    translucent_candidates = sum(
        material.likelihoods.get(mat, np.zeros_like(transmission))
        for mat in ("glass", "water", "crystal", "ice", "thin_fabric", "wings", "fins", "leaves")
    )
    if np.any(transmission > 0.05) and not np.any(translucent_candidates > 0.2):
        issues["transmission"].append("only_in_translucent_materials")

    # Structural checks
    structural = maps.get("structural")
    if structural is not None:
        struct_array = _np_from_image(structural)
        base = _to_array(analysis.geometric_features.get("luminance"))
        corr = float(np.corrcoef(struct_array.flatten(), base.flatten())[0, 1]) if struct_array.size > 1 else 1.0
        if corr > 0.3:
            issues["structural"].append("different_from_base")
        if np.mean(np.abs(np.gradient(struct_array)[0])) < 0.03:
            issues["structural"].append("high_frequency_features")
        if struct_array.max() - struct_array.min() < 0.05:
            issues["structural"].append("not_flat")

    # Subsurface checks
    subsurface = maps.get("subsurface")
    if subsurface is not None and structural is not None:
        sub_array = _np_from_image(subsurface)
        struct_array = _np_from_image(structural)
        thin_sections = analysis.geometric_features.get("thickness")
        organic_like = material.likelihoods.get("organic", np.zeros_like(sub_array))
        if np.mean(sub_array[thin_sections > 0.6]) < 0.2:
            issues["subsurface"].append("thin_sections_highlighted")
        if np.mean(sub_array[organic_like > 0.4]) < 0.2:
            issues["subsurface"].append("organic_materials_focused")
        corr = float(np.corrcoef(sub_array.flatten(), struct_array.flatten())[0, 1]) if sub_array.size > 1 else 1.0
        if abs(corr) > 0.85:
            issues["subsurface"].append("different_from_structural")

    return ValidationReport(issues=issues)


def add_procedural_variation(array: np.ndarray, detail_reference: np.ndarray) -> np.ndarray:
    noise = np.random.default_rng().normal(scale=0.02, size=array.shape)
    detail = detail_reference / (detail_reference.max() + 1e-6)
    variation = array + noise + detail * 0.05
    return np.clip(variation, 0.0, 1.0)


def enforce_organic_metallic_ban(metallic: Image.Image, material_analysis) -> Image.Image:
    metallic_np = _np_from_image(metallic)
    organic = material_analysis.likelihoods.get("organic", np.zeros_like(metallic_np))
    metallic_np[organic > 0.2] = 0.0
    array = (metallic_np * 255).astype(np.uint8)
    alpha = metallic.split()[-1]
    rgba = Image.merge("RGBA", (Image.fromarray(array, mode="L"),) * 3 + (alpha,))
    return rgba


def auto_correct_failed_maps(maps: Dict[str, Image.Image], analysis, validation_report: ValidationReport):
    corrections_applied: List[str] = []
    for map_name, issues in validation_report.issues.items():
        if not issues:
            continue
        if map_name in {"metallic", "roughness", "specular", "opacity", "transmission"}:
            base_array = _np_from_image(maps[map_name])
            detail = analysis.geometric_features.get("edge_map")
            corrected = add_procedural_variation(base_array, detail)
            alpha = maps[map_name].split()[-1]
            corrected_img = Image.fromarray((corrected * 255).astype(np.uint8), mode="L")
            maps[map_name] = Image.merge("RGBA", (corrected_img, corrected_img, corrected_img, alpha))
            corrections_applied.append(f"{map_name}_flat_fixed")
        if "no_organic_false_positives" in issues and map_name == "metallic":
            maps["metallic"] = enforce_organic_metallic_ban(maps["metallic"], analysis.material_analysis)
            corrections_applied.append("metallic_false_positives_fixed")
    return maps, corrections_applied


def log_corrections_applied(corrections: Iterable[str]) -> None:
    if not corrections:
        return
    LOGGER.info("Applied corrections: %s", ", ".join(corrections))


__all__ = [
    "VALIDATION_RULES",
    "ValidationReport",
    "add_procedural_variation",
    "auto_correct_failed_maps",
    "enforce_organic_metallic_ban",
    "log_corrections_applied",
    "validate_all_maps",
]
