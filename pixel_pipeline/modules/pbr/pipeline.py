"""Primary orchestration for the physically accurate PBR pipeline."""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Tuple

import logging
import numpy as np
from PIL import Image, ImageOps

try:  # pragma: no cover - optional dependency
    from skimage import exposure as _skimage_exposure
except ImportError:  # pragma: no cover - graceful fallback
    _skimage_exposure = None

try:  # pragma: no cover - optional dependency
    from skimage.restoration import denoise_bilateral as _skimage_denoise_bilateral
except ImportError:  # pragma: no cover - graceful fallback
    _skimage_denoise_bilateral = None

try:  # pragma: no cover - optional dependency
    from scipy import ndimage as _scipy_ndimage
except ImportError:  # pragma: no cover - graceful fallback
    _scipy_ndimage = None

try:  # pragma: no cover - optional dependency
    import cv2 as _cv2
except ImportError:  # pragma: no cover - graceful fallback
    _cv2 = None

from .alpha_utils import apply_alpha, apply_alpha_to_maps, derive_alpha_map
from .background_edge_correction import correct_edges_with_background
from .analysis import AnalysisResult, analyze_image_comprehensive, calculate_entropy
from .generation import (
    generate_alpha_accurate,
    generate_ao_physically_accurate,
    generate_curvature_enhanced,
    generate_emissive_accurate,
    generate_height_from_normal,
    generate_ior_physically_accurate,
    generate_material_accurate,
    generate_metallic_physically_accurate,
    generate_normal_enhanced,
    generate_opacity_accurate,
    generate_porosity_accurate,
    generate_roughness_physically_accurate,
    generate_specular_coherent,
    generate_structural_distinct,
    generate_subsurface_accurate,
    generate_fuzz_accurate,
    generate_transmission_physically_accurate,
)
from .physical_rgb import sanitize_rgba_image
from .validation import (
    _evaluate_quality_checks_improved,
    auto_correct_failed_maps,
    log_corrections_applied,
    validate_all_maps,
    validate_pbr_coherence_corregido,
    automated_quality_report_v5,
    validate_pbr_coherence_v5,
    detect_seams_validation,
)

LOGGER = logging.getLogger("pixel_pipeline.pbr.pipeline")

GENERATION_ORDER = (
    "metallic",
    "roughness",
    "specular",
    "transmission",
    "subsurface",
    "ior",
    "normal",
    "height",
    "ao",
    "curvature",
    "emissive",
    "structural",
    "porosity",
    "opacity",
    "fuzz",
    "material",
)

EXPECTED_TOTAL_MAPS = 16


def _fallback_map(base_img: Image.Image, name: str) -> Image.Image:
    size = base_img.size
    rgba = base_img.convert("RGBA")
    rgba_np = np.asarray(rgba, dtype=np.uint8)
    alpha = Image.fromarray(rgba_np[..., 3], mode="L")
    opaque = rgba_np[..., 3] >= 200
    if np.any(opaque):
        matte = np.median(rgba_np[..., :3][opaque], axis=0)
    else:
        matte = np.median(rgba_np[..., :3].reshape(-1, 3), axis=0)
    matte = np.clip(matte, 0, 255).astype(np.uint8)

    base_value = 0 if name in {"opacity", "transmission", "metallic"} else 128
    mix_factor = 0.2 + 0.8 * (base_value / 255.0)
    fill_color = tuple(int(round(component * mix_factor)) for component in matte)

    rgb_channel = Image.new("RGB", size, fill_color)
    return Image.merge("RGBA", (*rgb_channel.split(), alpha))


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.zeros_like(numerator, dtype=np.float32)
        mask = denominator > 1e-8
        result[mask] = numerator[mask] / denominator[mask]
        result[~mask] = 0.5
    return result


def _analysis_shape(analysis: AnalysisResult | None) -> Tuple[int, int]:
    if analysis is not None:
        alpha = getattr(analysis, "alpha", None)
        if isinstance(alpha, np.ndarray) and alpha.size:
            return alpha.shape
        base_image = getattr(analysis, "base_image", None)
        if isinstance(base_image, Image.Image):
            width, height = base_image.size
            return height, width
    return 1, 1


def _safe_map_generation(
    base_array: Image.Image | np.ndarray | None,
    analysis: AnalysisResult,
    *,
    neutral: float = 0.5,
) -> np.ndarray:
    target_shape = _analysis_shape(analysis)
    if base_array is None:
        return np.full(target_shape, neutral, dtype=np.float32)

    if isinstance(base_array, Image.Image):
        array = np.asarray(base_array.convert("L"), dtype=np.float32)
    elif isinstance(base_array, np.ndarray):
        array = base_array.astype(np.float32, copy=False)
    else:
        try:
            array = np.asarray(base_array, dtype=np.float32)
        except Exception:
            return np.full(target_shape, neutral, dtype=np.float32)

    if array.size == 0:
        return np.full(target_shape, neutral, dtype=np.float32)

    if array.ndim == 3:
        array = array.mean(axis=2)

    if array.max() > 1.0 + 1e-6 or array.min() < -1e-6:
        array = np.clip(array, 0.0, 255.0)
    if array.max() > 1.0 + 1e-6:
        array = array / 255.0

    array = np.nan_to_num(array, nan=neutral, posinf=neutral, neginf=neutral).astype(np.float32)

    if array.shape != target_shape:
        image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
        resized = image.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0

    return np.clip(array, 0.0, 1.0)


def _apply_gaussian_filter(array: np.ndarray, sigma: float) -> np.ndarray:
    if _scipy_ndimage is not None:
        return _scipy_ndimage.gaussian_filter(array, sigma=sigma, mode="reflect")

    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    return _convolve2d(array, kernel, pad_mode="reflect")


def _apply_laplacian_filter(array: np.ndarray) -> np.ndarray:
    if _scipy_ndimage is not None:
        return _scipy_ndimage.laplace(array, mode="reflect")

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    return _convolve2d(array, kernel, pad_mode="reflect")


def _convolve2d(array: np.ndarray, kernel: np.ndarray, *, pad_mode: str = "reflect") -> np.ndarray:
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x)), mode=pad_mode)
    LOGGER.debug("Edge processing: padding=%s, filter_mode=%s", (pad_y, pad_x), pad_mode)
    result = np.zeros_like(array, dtype=np.float32)
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            result += kernel[y, x] * padded[y : y + array.shape[0], x : x + array.shape[1]]
    return result.astype(np.float32, copy=False)


def _normalize_01(array: np.ndarray) -> np.ndarray:
    min_val = float(array.min()) if array.size else 0.0
    max_val = float(array.max()) if array.size else 0.0
    if max_val - min_val < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    normalized = (array - min_val) / (max_val - min_val)
    return normalized.astype(np.float32, copy=False)


def _bilateral_filter(array: np.ndarray, sigma_color: float = 0.1, sigma_spatial: float = 3.0) -> np.ndarray:
    if _skimage_denoise_bilateral is not None:
        return _skimage_denoise_bilateral(
            array.astype(np.float32, copy=False),
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
            channel_axis=None,
        ).astype(np.float32, copy=False)
    return _apply_gaussian_filter(array, sigma=max(sigma_spatial * 0.5, 0.5))


def _equalize_histogram(array: np.ndarray) -> np.ndarray:
    array = np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)
    if _skimage_exposure is not None:
        try:  # pragma: no cover - optional dependency branch
            return _skimage_exposure.equalize_hist(array).astype(np.float32, copy=False)
        except Exception:  # pragma: no cover - robust fallback
            pass

    hist, bin_edges = np.histogram(array, bins=256, range=(0.0, 1.0), density=False)
    cdf = np.cumsum(hist).astype(np.float32)
    if cdf[-1] <= 0:
        return array
    cdf /= cdf[-1]
    values = np.interp(array.flatten(), bin_edges[:-1], cdf)
    return values.reshape(array.shape).astype(np.float32, copy=False)


def _adaptive_equalize(array: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    array = np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)
    if _skimage_exposure is not None and hasattr(_skimage_exposure, "equalize_adapthist"):
        try:  # pragma: no cover - optional dependency branch
            return _skimage_exposure.equalize_adapthist(array, clip_limit=clip_limit).astype(
                np.float32,
                copy=False,
            )
        except Exception:  # pragma: no cover - robust fallback
            pass
    return _equalize_histogram(array)


def _sobel_gradients(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if _scipy_ndimage is not None:
        grad_x = _scipy_ndimage.sobel(array, axis=1, mode="reflect")
        grad_y = _scipy_ndimage.sobel(array, axis=0, mode="reflect")
    else:
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        grad_x = _convolve2d(array, sobel_x, pad_mode="reflect")
        grad_y = _convolve2d(array, sobel_y, pad_mode="reflect")
    return grad_x.astype(np.float32, copy=False), grad_y.astype(np.float32, copy=False)


def _enhance_transmission_correlation(
    transmission_map: Image.Image,
    base_image: Image.Image,
    foreground_mask: Image.Image | np.ndarray | None,
) -> Image.Image:
    try:
        width, height = transmission_map.size
        transmission_array = np.asarray(transmission_map.convert("L"), dtype=np.float32) / 255.0

        resized_base = base_image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        base_array = np.asarray(resized_base, dtype=np.float32) / 255.0

        if foreground_mask is None:
            mask_array = np.ones((height, width), dtype=np.float32)
        elif isinstance(foreground_mask, Image.Image):
            mask_resized = foreground_mask.resize((width, height), Image.Resampling.BILINEAR)
            mask_array = np.asarray(mask_resized, dtype=np.float32) / 255.0
        else:
            mask_array = np.asarray(foreground_mask, dtype=np.float32)
            if mask_array.ndim == 3:
                mask_array = mask_array[..., 0]
            if mask_array.shape != (height, width):
                mask_img = Image.fromarray((np.clip(mask_array, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
                mask_img = mask_img.resize((width, height), Image.Resampling.BILINEAR)
                mask_array = np.asarray(mask_img, dtype=np.float32) / 255.0
            elif mask_array.max() > 1.0 + 1e-6 or mask_array.min() < -1e-6:
                mask_array = np.clip(mask_array, 0.0, 255.0)
                if mask_array.max() > 1.0 + 1e-6:
                    mask_array = mask_array / 255.0

        mask_array = np.clip(mask_array, 0.0, 1.0).astype(np.float32, copy=False)

        gray_base = base_array.mean(axis=2)
        texture_strength = _bilateral_filter(gray_base, sigma_color=0.15, sigma_spatial=3.0)
        texture_strength = _normalize_01(texture_strength)

        laplacian = _apply_laplacian_filter(texture_strength)
        edge_potential = _normalize_01(np.abs(laplacian))

        transmission = np.clip(texture_strength * 0.3 + edge_potential * 0.7, 0.0, 1.0)
        transmission = np.clip(transmission * mask_array, 0.0, 1.0)

        if np.any(mask_array > 0.05):
            original = np.clip(transmission_array, 0.0, 1.0)
            transmission = np.clip(transmission * 0.6 + original * 0.4, 0.0, 1.0)

        transmission = _normalize_01(transmission)
        transmission = np.power(np.clip(transmission, 0.0, 1.0), 0.8)
        return Image.fromarray((transmission * 255.0).astype(np.uint8), mode="L")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Transmission enhancement failed: %s", exc)
        return transmission_map


def _enhance_height_map(height_map: Image.Image) -> Image.Image:
    height_array = np.asarray(height_map.convert("L"), dtype=np.float32) / 255.0
    enhanced = _adaptive_equalize(height_array, clip_limit=0.025)
    enhanced = np.clip(enhanced, 0.0, 1.0)
    return Image.fromarray((enhanced * 255.0).astype(np.uint8), mode="L")


def _generate_normal_map_corrected(height_map: Image.Image, base_image: Image.Image) -> Image.Image:
    height_array = np.asarray(height_map.convert("L"), dtype=np.float32) / 255.0
    if height_array.size == 0:
        return Image.new("RGB", base_image.size, (128, 128, 255))

    height_enhanced = _adaptive_equalize(height_array, clip_limit=0.03)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    if _scipy_ndimage is not None:
        grad_x = _scipy_ndimage.convolve(height_enhanced, sobel_x, mode="reflect") * 2.0
        grad_y = _scipy_ndimage.convolve(height_enhanced, sobel_y, mode="reflect") * 2.0
    else:
        grad_x = _convolve2d(height_enhanced, sobel_x, pad_mode="reflect") * 2.0
        grad_y = _convolve2d(height_enhanced, sobel_y, pad_mode="reflect") * 2.0

    z_component = 0.5
    normal = np.dstack((-grad_x, -grad_y, np.ones_like(height_array) * z_component))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / (norm + 1e-8)
    normal_rgb = np.clip((normal + 1.0) / 2.0, 0.0, 1.0)

    return Image.fromarray((normal_rgb * 255.0).astype(np.uint8), mode="RGB")


def _preserve_foreground_texture_transmission(
    foreground: Image.Image,
    background: Image.Image | None,
    analysis: AnalysisResult,
    candidate: Image.Image,
) -> Tuple[Image.Image, List[str]]:
    issues: List[str] = []

    fg_texture = _safe_map_generation(foreground.convert("L"), analysis, neutral=0.0)
    bg_texture = (
        _safe_map_generation(background.convert("L"), analysis, neutral=0.0)
        if background is not None
        else np.zeros_like(fg_texture)
    )

    alpha = getattr(analysis, "alpha", None)
    if isinstance(alpha, np.ndarray) and alpha.size:
        alpha_map = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    else:
        alpha_map = _safe_map_generation(candidate.split()[-1] if "A" in candidate.getbands() else None, analysis, neutral=0.0)

    mask_image = Image.fromarray((np.clip(alpha_map, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    enhanced_candidate = _enhance_transmission_correlation(candidate, foreground, mask_image)
    transmission = _safe_map_generation(enhanced_candidate, analysis, neutral=0.0)

    padded = np.pad(fg_texture, ((1, 1), (1, 1)), mode="reflect")
    blurred = np.zeros_like(fg_texture)
    height, width = fg_texture.shape
    for dy in range(3):
        for dx in range(3):
            blurred += padded[dy : dy + height, dx : dx + width]
    blurred /= 9.0
    detail = np.clip(fg_texture - blurred + 0.5, 0.0, 1.0)

    preserved = np.clip(transmission * (0.6 + 0.4 * alpha_map) + detail * 0.3, 0.0, 1.0)
    preserved = np.where(alpha_map > 0.05, preserved, np.clip(bg_texture, 0.0, 1.0))

    mask = alpha_map > 0.05
    if np.any(mask):
        fg_sample = fg_texture[mask]
        preserved_sample = preserved[mask]
        if fg_sample.size > 0 and preserved_sample.size > 0:
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(fg_sample.flatten(), preserved_sample.flatten())[0, 1]
            if not np.isfinite(corr) or corr < 0.2:
                issues.append("low_texture_correlation")

    alpha_channel = np.clip(alpha_map, 0.0, 1.0)
    preserved_rgb = np.repeat(preserved[..., None], 3, axis=2)
    combined = np.dstack((np.clip(preserved_rgb, 0.0, 1.0), alpha_channel[..., None]))
    image = Image.fromarray((combined * 255.0).astype(np.uint8), mode="RGBA")

    if not issues:
        LOGGER.info("Transmission map corrected: foreground texture preservation active")
    else:
        LOGGER.warning("Transmission map adjustment triggered: %s", ", ".join(issues))

    return image, issues

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
        "porosity",
        "fuzz",
        "material",
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


def _detect_and_correct_flat_maps(
    maps_dict: Mapping[str, Image.Image],
    base_image: Image.Image,
    analysis: AnalysisResult,
) -> Dict[str, Image.Image]:
    corrected: Dict[str, Image.Image] = {}
    for map_name, map_image in maps_dict.items():
        if not isinstance(map_image, Image.Image):
            corrected[map_name] = map_image
            continue

        map_array = np.asarray(map_image.convert("L"), dtype=np.float32) / 255.0
        if map_array.size == 0:
            corrected[map_name] = map_image
            continue

        std_dev = float(np.std(map_array))
        if std_dev < 5e-3:
            LOGGER.warning(
                "Mapa %s detectado como uniforme (std: %.5f); aplicando realce adaptativo",
                map_name,
                std_dev,
            )
            if map_name == "metallic":
                regenerated = generate_metallic_physically_accurate(base_image, analysis, map_image)
            elif map_name == "roughness":
                metallic_ref = maps_dict.get("metallic", map_image)
                regenerated = generate_roughness_physically_accurate(base_image, analysis, metallic_ref)
            elif map_name == "opacity":
                regenerated = generate_opacity_accurate(analysis)
            else:
                enhanced = ImageOps.autocontrast(map_image.convert("L"))
                if map_image.mode == "RGBA":
                    alpha = map_image.split()[-1]
                    rgb = Image.merge("RGB", (enhanced, enhanced, enhanced))
                    regenerated = Image.merge("RGBA", (*rgb.split(), alpha))
                elif map_image.mode == "RGB":
                    regenerated = Image.merge("RGB", (enhanced, enhanced, enhanced))
                else:
                    regenerated = enhanced.convert(map_image.mode)
            corrected[map_name] = regenerated
        else:
            corrected[map_name] = map_image
    return corrected


def _update_final_maps(
    base_img: Image.Image,
    analysis: AnalysisResult,
    current_maps: Mapping[str, Image.Image],
) -> Tuple[Dict[str, Image.Image], Dict[str, bool]]:
    diagnostics: Dict[str, bool | List[str]] = {}
    final_maps: Dict[str, Image.Image] = {}

    opacity_map = generate_opacity_accurate(analysis)
    final_maps["opacity"] = opacity_map

    raw_transmission = generate_transmission_physically_accurate(
        analysis, current_maps.get("transmission")
    )
    preserved_transmission, transmission_issues = _preserve_foreground_texture_transmission(
        base_img,
        analysis.background_image,
        analysis,
        raw_transmission,
    )
    final_maps["transmission"] = preserved_transmission
    diagnostics["transmission_preserved"] = not transmission_issues

    metallic_map = generate_metallic_physically_accurate(
        base_img, analysis, current_maps.get("metallic")
    )
    final_maps["metallic"] = metallic_map

    roughness_map = generate_roughness_physically_accurate(base_img, analysis, metallic_map)
    final_maps["roughness"] = roughness_map

    final_maps["specular"] = generate_specular_coherent(roughness_map, analysis)
    final_maps["subsurface"] = generate_subsurface_accurate(analysis, preserved_transmission)
    final_maps["ior"] = generate_ior_physically_accurate(
        preserved_transmission,
        analysis.material_analysis,
        analysis,
    )

    structural = generate_structural_distinct(analysis, current_maps.get("structural"))
    final_maps["structural"] = structural

    normal_candidate = generate_normal_enhanced(base_img, analysis)
    height_candidate = generate_height_from_normal(normal_candidate, analysis)
    enhanced_height = _enhance_height_map(height_candidate)
    refined_normal = _generate_normal_map_corrected(enhanced_height, base_img)
    final_maps["normal"] = refined_normal
    final_maps["height"] = enhanced_height

    final_maps["ao"] = generate_ao_physically_accurate(analysis)
    final_maps["curvature"] = generate_curvature_enhanced(analysis)
    final_maps["emissive"] = generate_emissive_accurate(base_img, analysis)

    porosity = generate_porosity_accurate(analysis)
    final_maps["porosity"] = porosity

    fuzz_map = generate_fuzz_accurate(base_img, analysis)
    final_maps["fuzz"] = fuzz_map
    fuzz_array = np.asarray(fuzz_map.convert("L"), dtype=np.float32) / 255.0
    diagnostics["fuzz_map_operational"] = float(np.std(fuzz_array)) > 5e-3

    final_maps["material"] = generate_material_accurate(analysis)

    final_maps = _detect_and_correct_flat_maps(
        final_maps,
        base_img,
        analysis,
    )

    missing_maps = [name for name in GENERATION_ORDER if name not in final_maps]
    if missing_maps:
        LOGGER.warning("PBR maps missing from unified generation: %s", ", ".join(missing_maps))
    elif len(final_maps) != EXPECTED_TOTAL_MAPS:
        LOGGER.warning(
            "Unificado generó %d mapas; se esperaban %d",
            len(final_maps),
            EXPECTED_TOTAL_MAPS,
        )

    coherence_issues = validate_pbr_coherence_corregido(base_img, final_maps)
    if coherence_issues:
        LOGGER.warning("Validación de coherencia PBR detectó inconsistencias: %s", ", ".join(coherence_issues))
    diagnostics["coherence_issues"] = coherence_issues

    return final_maps, diagnostics


def _enforce_rgba_alpha(
    base_img: Image.Image,
    maps: Mapping[str, Image.Image],
    analysis: AnalysisResult,
) -> tuple[Dict[str, Image.Image], np.ndarray]:
    alpha_candidate_img = generate_alpha_accurate(analysis, maps.get("opacity"))
    candidate_alpha = np.asarray(alpha_candidate_img.split()[-1], dtype=np.float32) / 255.0
    if candidate_alpha.shape != (base_img.height, base_img.width):
        candidate_alpha = np.asarray(
            alpha_candidate_img.split()[-1].resize(base_img.size, Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0

    alpha_map = derive_alpha_map(base_img, maps, analysis)
    if alpha_map.shape == candidate_alpha.shape:
        alpha_map = np.clip(alpha_map * 0.6 + candidate_alpha * 0.4, 0.0, 1.0)
    updated_maps = apply_alpha_to_maps(maps, alpha_map)
    return updated_maps, alpha_map


def generate_physically_accurate_pbr_maps(
    base_img: Image.Image,
    bg_img: Image.Image | None,
    current_maps: Mapping[str, Image.Image],
) -> Dict[str, object]:
    sanitized_base = sanitize_rgba_image(base_img)
    sanitized_bg = sanitize_rgba_image(bg_img) if bg_img is not None else None

    prepared_maps = _ensure_current_maps(sanitized_base, current_maps)
    analysis = analyze_image_comprehensive(sanitized_base, None, prepared_maps)
    final_maps, diagnostics = _update_final_maps(sanitized_base, analysis, prepared_maps)
    final_maps, alpha_map = _enforce_rgba_alpha(sanitized_base, final_maps, analysis)

    validation_report = validate_all_maps(final_maps, analysis)
    corrections: Tuple[str, ...] = ()
    if validation_report.has_critical_issues():
        final_maps, applied = auto_correct_failed_maps(final_maps, analysis, validation_report)
        corrections = tuple(applied)
        log_corrections_applied(corrections)
        final_maps, alpha_map = _enforce_rgba_alpha(sanitized_base, final_maps, analysis)
        validation_report = validate_all_maps(final_maps, analysis)

    final_validation = validate_all_maps(final_maps, analysis)
    seam_issues = detect_seams_validation(final_maps)
    if seam_issues:
        LOGGER.warning("Edge seam validation triggered: %s", ", ".join(seam_issues))
        diagnostics.setdefault("seam_issues", []).extend(seam_issues)
    if final_validation.has_critical_issues():
        LOGGER.info("Remaining issues after improvements: %s", final_validation.issues)
    else:
        LOGGER.info("All PBR maps validated successfully - background and metal issues resolved")

    if sanitized_bg is not None:
        alpha_binary = (alpha_map > 0.1).astype(np.uint8) * 255
        alpha_mask = Image.fromarray(alpha_binary, mode="L")
        composite = sanitized_bg.copy()
        foreground_rgb = sanitized_base.convert("RGB")
        composite.paste(foreground_rgb, (0, 0), alpha_mask)
    else:
        composite = apply_alpha(sanitized_base, alpha_map)
    foreground_texture = np.asarray(sanitized_base.convert("L"), dtype=np.float32) / 255.0
    quality_checks = _evaluate_quality_checks_improved(
        final_maps,
        base_image=sanitized_base,
        composite=composite,
        foreground_texture=foreground_texture,
        fuzz_ok=diagnostics.get("fuzz_map_operational", True),
        transmission_ok=diagnostics.get("transmission_preserved", True),
    )
    quality_report = generate_quality_report(final_maps, analysis)
    material_class = getattr(analysis, "material_class", "default")
    coherence_v5 = validate_pbr_coherence_v5(sanitized_base, final_maps, material_class)
    quality_report_v5 = automated_quality_report_v5(coherence_v5)
    diagnostics_payload = {
        **diagnostics,
        "edge_correction_applied": sanitized_bg is not None,
        "correction_method": "background_extrapolation" if sanitized_bg is not None else "none",
    }

    return {
        "maps": final_maps,
        "analysis": analysis,
        "validation": final_validation,
        "corrections_applied": corrections,
        "quality_report": quality_report,
        "coherence_v5": coherence_v5,
        "quality_report_v5": quality_report_v5,
        "alpha": alpha_map,
        "quality_checks": quality_checks,
        "coherence_issues": diagnostics.get("coherence_issues", []),
        "composite": composite,
        "diagnostics": diagnostics_payload,
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


def safe_composition_with_background_correction(
    foreground: Image.Image,
    background: Image.Image,
) -> Image.Image:
    """Compose ``foreground`` over ``background`` after correcting edge bleed."""

    corrected = correct_edges_with_background(foreground, background)
    fg_arr = np.asarray(corrected.convert("RGBA"), dtype=np.float32) / 255.0
    bg_arr = np.asarray(background.convert("RGB"), dtype=np.float32) / 255.0

    fg_rgb = fg_arr[..., :3]
    fg_alpha = fg_arr[..., 3][..., None]
    composite_rgb = fg_rgb * fg_alpha + bg_arr * (1.0 - fg_alpha)
    composite_rgb = np.clip(composite_rgb, 0.0, 1.0)
    return Image.fromarray((composite_rgb * 255.0).astype(np.uint8), mode="RGB")


__all__ = [
    "generate_physically_accurate_pbr_maps",
    "generate_quality_report",
    "_detect_and_correct_flat_maps",
    "safe_composition_with_background_correction",
]
