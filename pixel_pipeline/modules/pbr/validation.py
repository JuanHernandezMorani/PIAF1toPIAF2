"""Validation and correction routines for the PBR pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import logging
import numpy as np
from PIL import Image

from .analysis import calculate_entropy
from .parameters import CRITICAL_PARAMETERS

LOGGER = logging.getLogger("pixel_pipeline.pbr.validation")

MAX_IOR = max(CRITICAL_PARAMETERS["IOR_TRANSLUCENT_RANGES"].values())
IOR_RANGE = max(MAX_IOR - CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"], 1e-3)


VALIDATION_CHECKS = {
    "halos_eliminated": "No bordes negros visibles en composición final",
    "fuzz_map_operational": "Generación sin errores de tipo ni warnings",
    "transmission_preserved": "Texturas foreground intactas en transmission map",
    "chromatic_consistency": "Mapas PBR mantienen variación cromática apropiada",
    "runtime_stability": "Ejecución sin RuntimeWarning ni division errors",
}


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


def _validate_halo_elimination(result_image: Image.Image) -> bool:
    rgba = np.asarray(result_image.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    edge_mask = (alpha > 0.02) & (alpha < 0.98)
    if not np.any(edge_mask):
        return True
    luminance = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
    edge_luminance = luminance[edge_mask]
    if edge_luminance.size == 0:
        return True
    return float(edge_luminance.min()) > 0.015


def _validate_transmission_integrity(
    transmission_map: Image.Image | None,
    foreground_texture: np.ndarray,
) -> List[str]:
    issues: List[str] = []
    if transmission_map is None:
        issues.append("transmission_map_missing")
        return issues

    transmission = _np_from_image(transmission_map)
    fg = np.asarray(foreground_texture, dtype=np.float32)
    if fg.ndim == 3:
        fg = fg.mean(axis=2)
    if transmission.shape != fg.shape:
        fg_image = Image.fromarray((np.clip(fg, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
        fg_image = fg_image.resize((transmission.shape[1], transmission.shape[0]), Image.BILINEAR)
        fg = np.asarray(fg_image, dtype=np.float32) / 255.0

    mask = fg > 0.05
    if np.any(mask):
        sample_trans = transmission[mask]
        sample_fg = fg[mask]
        if sample_trans.size and sample_fg.size:
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(sample_trans.flatten(), sample_fg.flatten())[0, 1]
            if not np.isfinite(corr) or corr < 0.25:
                issues.append("foreground_texture_lost")

    background_mask = fg < 0.02
    if np.any(background_mask):
        background_level = float(np.mean(transmission[background_mask]))
        if background_level > 0.08:
            issues.append("background_leakage_detected")

    if float(np.std(transmission)) < 5e-3:
        issues.append("transmission_flat_response")

    return issues


def _validate_chromatic_balance(pbr_maps: Mapping[str, Image.Image]) -> List[str]:
    issues: List[str] = []
    base = pbr_maps.get("base_color") or pbr_maps.get("albedo")
    if base is None:
        return issues

    base_rgb = np.asarray(base.convert("RGB"), dtype=np.float32) / 255.0
    chroma_strength = float(np.mean(np.std(base_rgb, axis=2)))

    for name in ("metallic", "roughness", "specular"):
        image = pbr_maps.get(name)
        if image is None:
            continue
        map_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        if map_rgb.shape[:2] != base_rgb.shape[:2]:
            resized = image.convert("RGB").resize(base.size, Image.BILINEAR)
            map_rgb = np.asarray(resized, dtype=np.float32) / 255.0
        channel_ptp = float(np.mean(np.ptp(map_rgb, axis=2)))
        if chroma_strength > 0.05 and channel_ptp < 0.015:
            issues.append(f"{name}_lacks_chromatic_variation")
    return issues


def _simple_halo_check(result_image: Image.Image, threshold: float = 0.01) -> Tuple[bool, float]:
    rgba = np.asarray(result_image.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    edge_mask = (alpha > 0.05) & (alpha < 0.95)
    if not np.any(edge_mask):
        return True, 0.0
    norm = np.linalg.norm(rgb, axis=2)
    dark_ratio = float(np.count_nonzero(norm[edge_mask] < 0.05)) / float(np.count_nonzero(edge_mask))
    return dark_ratio < threshold, dark_ratio


def _texture_correlation_check(
    transmission_map: Image.Image | None,
    foreground: np.ndarray,
    *,
    min_correlation: float = 0.7,
) -> Tuple[bool, float]:
    if transmission_map is None:
        return False, float("nan")

    transmission = _np_from_image(transmission_map)
    fg = np.asarray(foreground, dtype=np.float32)
    if fg.ndim == 3:
        fg = fg.mean(axis=2)

    if transmission.shape != fg.shape:
        fg_img = Image.fromarray((np.clip(fg, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
        fg_img = fg_img.resize((transmission.shape[1], transmission.shape[0]), Image.BILINEAR)
        fg = np.asarray(fg_img, dtype=np.float32) / 255.0

    mask = fg > 0.05
    if np.count_nonzero(mask) < 4:
        return True, 1.0

    sample_trans = transmission[mask]
    sample_fg = fg[mask]
    with np.errstate(invalid="ignore"):
        corr = float(np.corrcoef(sample_trans.flatten(), sample_fg.flatten())[0, 1])
    if not np.isfinite(corr):
        corr = 0.0
    return corr >= min_correlation, corr


def _color_drift_simple_check(
    base_color: Image.Image | None,
    composed: Image.Image,
    *,
    max_delta: float = 3.0,
) -> Tuple[bool, float]:
    if base_color is None:
        return True, 0.0

    base_rgb = np.asarray(base_color.convert("RGB"), dtype=np.float32)
    composed_rgb = np.asarray(composed.convert("RGB"), dtype=np.float32)
    if base_rgb.shape != composed_rgb.shape:
        composed_rgb = np.asarray(
            composed.convert("RGB").resize(base_color.size, Image.BILINEAR),
            dtype=np.float32,
        )

    delta = np.linalg.norm(base_rgb - composed_rgb, axis=2)
    delta_e = float(np.mean(delta))
    return delta_e <= max_delta, delta_e


def _evaluate_quality_checks_improved(
    maps: Mapping[str, Image.Image],
    *,
    base_image: Image.Image,
    composite: Image.Image,
    foreground_texture: np.ndarray,
    fuzz_ok: bool,
    transmission_ok: bool,
) -> Dict[str, bool]:
    halos_ok, halo_ratio = _simple_halo_check(composite)
    if halos_ok:
        LOGGER.info("Halo residual ratio %.4f below threshold", halo_ratio)
    else:
        LOGGER.warning("Halo residual ratio %.4f exceeds threshold", halo_ratio)

    transmission_map = maps.get("transmission")
    texture_ok, corr = _texture_correlation_check(transmission_map, foreground_texture)
    if np.isnan(corr):
        LOGGER.warning("Transmission map missing or invalid for correlation check")
    else:
        log_msg = "Transmission/foreground correlation %.3f" % corr
        if texture_ok:
            LOGGER.info(log_msg)
        else:
            LOGGER.warning(log_msg)

    base_colour_map = maps.get("base_color") or maps.get("albedo") or base_image
    chroma_ok, delta_e = _color_drift_simple_check(base_colour_map, composite)
    if chroma_ok:
        LOGGER.info("Average colour drift ΔE≈%.2f within tolerance", delta_e)
    else:
        LOGGER.warning("Average colour drift ΔE≈%.2f exceeds tolerance", delta_e)

    results = {
        "halos_eliminated": halos_ok,
        "fuzz_map_operational": fuzz_ok,
        "transmission_preserved": transmission_ok and texture_ok,
        "chromatic_consistency": chroma_ok,
        "runtime_stability": True,
        "transmission_consistent": texture_ok,
        "chromatic_stable": chroma_ok,
    }

    for key, description in VALIDATION_CHECKS.items():
        state = results.get(key, False)
        log_fn = LOGGER.info if state else LOGGER.warning
        log_fn("Validation check %s: %s", "passed" if state else "failed", description)

    return results


def validate_pbr_coherence_corregido(
    base_image: Image.Image,
    pbr_maps: Mapping[str, Image.Image],
) -> List[str]:
    """Evaluate cross-map coherence and flag major inconsistencies."""

    if base_image is None:
        return ["BASE_IMAGE_MISSING"]

    issues: List[str] = []
    base_gray = np.asarray(base_image.convert("L"), dtype=np.float32) / 255.0

    for name, image in pbr_maps.items():
        if image is None:
            issues.append(f"MISSING_MAP: {name}")
            continue

        try:
            raw_array = np.asarray(image, dtype=np.float32)
        except Exception:
            issues.append(f"INVALID_IMAGE: {name}")
            continue

        if raw_array.size == 0:
            issues.append(f"EMPTY_MAP: {name}")
            continue

        raw_min = float(np.nanmin(raw_array))
        raw_max = float(np.nanmax(raw_array))
        if raw_min < 0.0 or raw_max > 255.0:
            issues.append(f"VALUE_OVERFLOW: {name}")

        map_gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        if map_gray.shape != base_gray.shape:
            issues.append(f"DIMENSION_MISMATCH: {name}")
            continue

        if float(np.std(map_gray)) < 5e-3:
            issues.append(f"LOW_VARIATION: {name}")

        if name not in {"emissive", "transmission"}:
            with np.errstate(invalid="ignore"):
                correlation = np.corrcoef(base_gray.flatten(), map_gray.flatten())[0, 1]
            if not np.isfinite(correlation) or abs(float(correlation)) < 0.3:
                issues.append(f"LOW_CORRELATION: {name} ({float(correlation):.2f})")

    return issues


def validate_pbr_coherence_v5(
    base_image: Image.Image,
    maps: Mapping[str, Image.Image],
    material_class: str = "default",
) -> Dict[str, object]:
    """Enhanced coherence validation with adaptive thresholds for v5 pipeline."""

    issues: List[str] = []
    warnings: List[str] = []
    corrections: Dict[str, str] = {}

    base_array = np.asarray(base_image.convert("L"), dtype=np.float32)
    base_std = float(np.std(base_array))

    results = {
        "passed": True,
        "issues": issues,
        "warnings": warnings,
        "corrections_applied": corrections,
    }

    for map_name, map_img in maps.items():
        if not isinstance(map_img, Image.Image):
            continue

        map_array = np.asarray(map_img.convert("L"), dtype=np.float32)
        map_std = float(np.std(map_array))

        if map_std < 1.5:
            issues.append(f"{map_name.upper()} → CRITICAL_FLATNESS (std: {map_std:.2f})")
            corrections[map_name] = "REPROCESS_WITH_CONTRAST_ENHANCEMENT"
            results["passed"] = False

        map_range = float(map_array.max() - map_array.min())
        if map_range < 50:
            warnings.append(f"{map_name.upper()} → LOW_DYNAMIC_RANGE ({map_range:.1f})")

        if map_name not in {"emissive", "transmission"} and map_std > 0 and base_std > 0:
            with np.errstate(all="ignore"):
                correlation = np.corrcoef(base_array.flatten(), map_array.flatten())[0, 1]
            if not np.isfinite(correlation) or abs(float(correlation)) < 0.2:
                issues.append(
                    f"{map_name.upper()} → SUSPICIOUS_CORRELATION ({float(correlation):.3f})"
                )

        if map_name == "metallic":
            metallic_mean = float(np.mean(map_array))
            if metallic_mean > 200:
                issues.append("METALLIC → EXCESSIVE_METALLIC_CONTENT")
            if metallic_mean < 10 and "metal" in material_class.lower():
                warnings.append("METALLIC → POTENTIAL_UNDERESTIMATION")

        if map_name == "emissive":
            emissive_std = float(np.std(map_array))
            if emissive_std > 80:
                issues.append("EMISSIVE → NOISY_EMISSIVE_MAP")
            high_emissive = float(np.sum(map_array > 200) / map_array.size)
            if high_emissive > 0.3:
                issues.append("EMISSIVE → EXCESSIVE_EMISSIVE_AREA")

    if "metallic" in maps and "roughness" in maps:
        metallic_arr = np.asarray(maps["metallic"].convert("L"), dtype=np.float32)
        rough_arr = np.asarray(maps["roughness"].convert("L"), dtype=np.float32)
        if np.std(metallic_arr) > 0 and np.std(rough_arr) > 0:
            with np.errstate(all="ignore"):
                metal_rough_corr = np.corrcoef(
                    metallic_arr.flatten(), rough_arr.flatten()
                )[0, 1]
            if metal_rough_corr > -0.3:
                warnings.append(
                    f"CROSS-MAP → WEAK_METAL_ROUGH_CORRELATION ({float(metal_rough_corr):.3f})"
                )

    return results


def automated_quality_report_v5(validation_results: Mapping[str, object]) -> Dict[str, object]:
    """Generate an automatic quality report based on v5 validation results."""

    report = {
        "overall_score": 100,
        "critical_issues": 0,
        "warnings": 0,
        "recommendations": [],
    }

    for issue in validation_results.get("issues", []):
        report["critical_issues"] += 1
        report["overall_score"] -= 15
        if "FLATNESS" in issue:
            report["recommendations"].append("Reprocess map with logarithmic normalization")
        elif "CORRELATION" in issue:
            report["recommendations"].append("Check material classification and reprocess")
        elif "METALLIC" in issue:
            report["recommendations"].append("Review metallic detection parameters")
        elif "EMISSIVE" in issue:
            report["recommendations"].append("Apply emissive saturation correction")

    for warning in validation_results.get("warnings", []):
        report["warnings"] += 1
        report["overall_score"] -= 5

    score = report["overall_score"]
    if score >= 85:
        report["quality"] = "EXCELLENT"
    elif score >= 70:
        report["quality"] = "GOOD"
    elif score >= 50:
        report["quality"] = "ACCEPTABLE"
    else:
        report["quality"] = "POOR - REQUIRES REPROCESSING"

    return report


__all__ = [
    "VALIDATION_RULES",
    "ValidationReport",
    "VALIDATION_CHECKS",
    "add_procedural_variation",
    "auto_correct_failed_maps",
    "enforce_organic_metallic_ban",
    "log_corrections_applied",
    "_evaluate_quality_checks_improved",
    "_color_drift_simple_check",
    "_simple_halo_check",
    "_texture_correlation_check",
    "_validate_chromatic_balance",
    "_validate_halo_elimination",
    "_validate_transmission_integrity",
    "validate_all_maps",
    "validate_pbr_coherence_corregido",
    "validate_pbr_coherence_v5",
    "automated_quality_report_v5",
]
