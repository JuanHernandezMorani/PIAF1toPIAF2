"""Primary orchestration for the physically accurate PBR pipeline."""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Tuple

import inspect
import logging
import numpy as np
from PIL import Image

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

from .alpha_utils import apply_alpha, apply_alpha_to_maps, derive_alpha_map
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
from .validation import (
    _evaluate_quality_checks_improved,
    auto_correct_failed_maps,
    log_corrections_applied,
    validate_all_maps,
    validate_pbr_coherence_corregido,
)

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
        return _scipy_ndimage.gaussian_filter(array, sigma=sigma)

    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    padded = np.pad(array, 1, mode="edge")
    result = (
        kernel[0, 0] * padded[:-2, :-2]
        + kernel[0, 1] * padded[:-2, 1:-1]
        + kernel[0, 2] * padded[:-2, 2:]
        + kernel[1, 0] * padded[1:-1, :-2]
        + kernel[1, 1] * padded[1:-1, 1:-1]
        + kernel[1, 2] * padded[1:-1, 2:]
        + kernel[2, 0] * padded[2:, :-2]
        + kernel[2, 1] * padded[2:, 1:-1]
        + kernel[2, 2] * padded[2:, 2:]
    )
    return result.astype(np.float32, copy=False)


def _apply_laplacian_filter(array: np.ndarray) -> np.ndarray:
    if _scipy_ndimage is not None:
        return _scipy_ndimage.laplace(array)

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    padded = np.pad(array, 1, mode="edge")
    result = (
        kernel[1, 1] * padded[1:-1, 1:-1]
        + kernel[0, 1] * padded[:-2, 1:-1]
        + kernel[2, 1] * padded[2:, 1:-1]
        + kernel[1, 0] * padded[1:-1, :-2]
        + kernel[1, 2] * padded[1:-1, 2:]
    )
    return result.astype(np.float32, copy=False)


def _convolve2d(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
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
        grad_x = _scipy_ndimage.sobel(array, axis=1)
        grad_y = _scipy_ndimage.sobel(array, axis=0)
    else:
        grad_y, grad_x = np.gradient(array.astype(np.float32, copy=False))
    return grad_x.astype(np.float32, copy=False), grad_y.astype(np.float32, copy=False)


def _generate_fuzz_map(analysis_result: AnalysisResult, base_image: Image.Image) -> Tuple[np.ndarray, bool]:
    width, height = base_image.size
    target_shape = (height, width)

    try:
        if not isinstance(analysis_result, AnalysisResult):
            raise ValueError("analysis_result inválido")

        microsurface = getattr(analysis_result, "microsurface_data", None)
        fuzz_potential = getattr(analysis_result, "fuzz_potential", None)

        if microsurface is None or fuzz_potential is None:
            raise ValueError("Datos de microsuperficie ausentes en AnalysisResult")

        microsurface = np.asarray(microsurface, dtype=np.float32)
        fuzz_potential = np.asarray(fuzz_potential, dtype=np.float32)

        if microsurface.ndim >= 3:
            microsurface = microsurface[..., 0]
        if fuzz_potential.ndim >= 3:
            fuzz_potential = fuzz_potential[..., 0]

        if microsurface.shape != target_shape or fuzz_potential.shape != target_shape:
            raise ValueError("Dimensiones de microsuperficie o fuzz_potential no coinciden")

        if (not np.isfinite(microsurface).all()) or (not np.isfinite(fuzz_potential).all()):
            raise ValueError("microsurface_data o fuzz_potential contienen valores no finitos")

        microsurface = np.clip(microsurface, 0.0, 1.0)
        fuzz_potential = np.clip(fuzz_potential, 0.0, 1.0)

        if np.std(microsurface) < 1e-4 or np.std(fuzz_potential) < 1e-4:
            raise ValueError("microsurface_data o fuzz_potential carecen de variación")

    except Exception as exc:
        LOGGER.error("ERROR CRÍTICO en fuzz map: %s", exc)
        base_array = np.asarray(base_image.convert("L"), dtype=np.float32) / 255.0
        microsurface = _normalize_01(_bilateral_filter(base_array, sigma_color=0.2, sigma_spatial=2.0))
        fuzz_potential = np.clip(1.0 - microsurface, 0.0, 1.0)
        success = False
    else:
        success = True

    fuzz_map = np.clip(microsurface * fuzz_potential, 0.0, 1.0).astype(np.float32, copy=False)

    fuzz_low = _apply_gaussian_filter(fuzz_map, sigma=1.0)
    fuzz_high = _apply_gaussian_filter(fuzz_map, sigma=0.3)
    combined = np.clip(fuzz_low * 0.7 + fuzz_high * 0.3, 0.0, 1.0)
    fuzz_map = _equalize_histogram(combined)
    fuzz_map = np.clip(fuzz_map, 0.0, 1.0)

    if not success and np.std(fuzz_map) < 5e-3:
        height, width = target_shape
        x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        noise = np.sin(x * 20.0) * np.sin(y * 20.0) * 0.3 + 0.5
        fuzz_map = np.clip(_apply_gaussian_filter(noise.astype(np.float32), sigma=1.0), 0.0, 1.0)

    return fuzz_map.astype(np.float32, copy=False), success


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


def _robust_fuzz_generation(base_image: Image.Image, analysis: AnalysisResult) -> Tuple[Image.Image, bool]:
    candidate = _generate_optional_map("fuzz", generate_fuzz_enhanced, base_image, analysis)
    fuzz_array, fuzz_operational = _generate_fuzz_map(analysis, base_image)

    detail_source = analysis.geometric_features.get("edge_map") if analysis.geometric_features else None
    detail_map = _safe_map_generation(detail_source, analysis, neutral=0.5)

    candidate_array = np.asarray(candidate.convert("L"), dtype=np.float32) / 255.0
    if candidate_array.shape == fuzz_array.shape:
        fuzz_array = np.clip(fuzz_array * 0.7 + candidate_array * 0.3, 0.0, 1.0)

    if np.std(fuzz_array) < 5e-3:
        LOGGER.warning("Fuzz map generation: variación insuficiente, reforzando con detalle geométrico")
        fuzz_array = np.clip(fuzz_array + (detail_map - 0.5) * 0.3, 0.0, 1.0)
        fuzz_operational = False

    candidate_rgba = candidate.convert("RGBA")
    if candidate_rgba.size != base_image.size:
        candidate_rgba = candidate_rgba.resize(base_image.size, Image.BILINEAR)
    alpha_channel = np.asarray(candidate_rgba.split()[-1], dtype=np.float32) / 255.0
    fuzz_rgb = np.repeat(fuzz_array[..., None], 3, axis=2)
    combined = np.dstack((fuzz_rgb, alpha_channel[..., None]))
    result = Image.fromarray((combined * 255.0).astype(np.uint8), mode="RGBA")
    return result, fuzz_operational


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
        grad_x = _scipy_ndimage.convolve(height_enhanced, sobel_x, mode="nearest") * 2.0
        grad_y = _scipy_ndimage.convolve(height_enhanced, sobel_y, mode="nearest") * 2.0
    else:
        grad_x = _convolve2d(height_enhanced, sobel_x) * 2.0
        grad_y = _convolve2d(height_enhanced, sobel_y) * 2.0

    z_component = 0.5
    normal = np.dstack((-grad_x, -grad_y, np.ones_like(height_array) * z_component))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / (norm + 1e-8)
    normal_rgb = np.clip((normal + 1.0) / 2.0, 0.0, 1.0)

    return Image.fromarray((normal_rgb * 255.0).astype(np.uint8), mode="RGB")


def _generate_metallic_map_corrected(
    base_image: Image.Image,
    analysis: AnalysisResult,
    roughness_map: Image.Image | None,
    candidate: Image.Image | None,
) -> Image.Image:
    base_rgb = np.asarray(base_image.convert("RGB"), dtype=np.float32) / 255.0
    brightness = base_rgb.mean(axis=2)
    saturation = np.std(base_rgb, axis=2)
    metallic_likelihood = np.clip(brightness * 0.6 + (1.0 - saturation) * 0.4, 0.0, 1.0)

    if roughness_map is not None:
        rough_img = roughness_map.convert("L") if isinstance(roughness_map, Image.Image) else roughness_map
        if isinstance(rough_img, Image.Image) and rough_img.size != base_image.size:
            rough_img = rough_img.resize(base_image.size, Image.BILINEAR)
        rough_array = np.asarray(rough_img, dtype=np.float32) / 255.0
        if rough_array.ndim == 3:
            rough_array = rough_array[..., 0]
        metallic_likelihood *= np.clip(1.0 - rough_array, 0.0, 1.0)

    mask = getattr(analysis, "mask", None)
    if isinstance(mask, np.ndarray) and mask.shape == metallic_likelihood.shape:
        metallic_likelihood *= np.clip(mask.astype(np.float32), 0.0, 1.0)

    material = getattr(analysis, "material_analysis", None)
    if material is not None:
        metal_like = material.likelihoods.get("metal") if material.likelihoods else None
        if metal_like is not None and metal_like.shape == metallic_likelihood.shape:
            metallic_likelihood *= np.clip(metal_like.astype(np.float32) + 0.35, 0.0, 1.0)
        organic_like = material.likelihoods.get("organic") if material.likelihoods else None
        if organic_like is not None and organic_like.shape == metallic_likelihood.shape:
            metallic_likelihood *= np.clip(1.0 - organic_like.astype(np.float32) * 0.8, 0.0, 1.0)
        false_risk = getattr(material, "false_metal_risks", None)
        if false_risk is not None and false_risk.shape == metallic_likelihood.shape:
            metallic_likelihood *= np.clip(1.0 - false_risk.astype(np.float32) * 0.7, 0.0, 1.0)

    metallic_likelihood = _normalize_01(metallic_likelihood)
    active_pixels = metallic_likelihood[metallic_likelihood > 0.01]
    if active_pixels.size:
        threshold = float(np.percentile(active_pixels, 70))
    else:
        threshold = 0.0
    refined = np.where(metallic_likelihood > threshold, metallic_likelihood, 0.0)
    refined = np.clip(_apply_gaussian_filter(refined, sigma=1.0), 0.0, 1.0)
    refined = _normalize_01(refined)

    if candidate is not None:
        candidate_gray = np.asarray(candidate.convert("L"), dtype=np.float32) / 255.0
        if candidate_gray.shape != refined.shape:
            candidate_gray_image = candidate.convert("L").resize(base_image.size, Image.BILINEAR)
            candidate_gray = np.asarray(candidate_gray_image, dtype=np.float32) / 255.0
        refined = np.clip(refined * 0.7 + candidate_gray * 0.3, 0.0, 1.0)

    refined = np.clip(refined, 0.0, 1.0)
    gray_image = Image.fromarray((refined * 255.0).astype(np.uint8), mode="L")

    if candidate is not None and "A" in candidate.getbands():
        alpha = candidate.convert("RGBA").split()[-1]
    else:
        alpha = Image.new("L", base_image.size, 255)

    rgb = Image.merge("RGB", (gray_image, gray_image, gray_image))
    return Image.merge("RGBA", (*rgb.split(), alpha))
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

    padded = np.pad(fg_texture, ((1, 1), (1, 1)), mode="edge")
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
) -> Tuple[Dict[str, Image.Image], Dict[str, bool]]:
    final_maps: Dict[str, Image.Image] = dict(current_maps)
    diagnostics = {"fuzz_map_operational": True, "transmission_preserved": True}

    final_maps["opacity"] = generate_alpha_accurate(analysis, current_maps.get("opacity"))
    raw_transmission = generate_transmission_physically_accurate(analysis, current_maps.get("transmission"))
    preserved_transmission, transmission_issues = _preserve_foreground_texture_transmission(
        base_img,
        analysis.background_image,
        analysis,
        raw_transmission,
    )
    final_maps["transmission"] = preserved_transmission
    diagnostics["transmission_preserved"] = not transmission_issues

    metallic_candidate = generate_metallic_physically_accurate(
        base_img, analysis, current_maps.get("metallic")
    )
    final_maps["metallic"] = _generate_metallic_map_corrected(
        base_img,
        analysis,
        current_maps.get("roughness"),
        metallic_candidate,
    )
    final_maps["ior"] = generate_ior_physically_accurate(
        final_maps["transmission"],
        analysis.material_analysis,
        analysis,
    )
    final_maps["roughness"] = generate_roughness_physically_accurate(
        base_img, analysis, final_maps["metallic"]
    )
    final_maps["specular"] = generate_specular_coherent(final_maps["roughness"], analysis)
    final_maps["subsurface"] = generate_subsurface_accurate(analysis, final_maps["transmission"])
    final_maps["structural"] = generate_structural_distinct(analysis, current_maps.get("structural"))

    normal_candidate = generate_normal_enhanced(base_img, analysis)
    height_candidate = generate_height_from_normal(normal_candidate, analysis)
    enhanced_height = _enhance_height_map(height_candidate)
    refined_normal = _generate_normal_map_corrected(enhanced_height, base_img)
    final_maps["normal"] = refined_normal
    final_maps["height"] = enhanced_height
    final_maps["ao"] = generate_ao_physically_accurate(analysis)
    final_maps["curvature"] = generate_curvature_enhanced(analysis)
    final_maps["emissive"] = generate_emissive_accurate(base_img, analysis)

    fuzz_map, fuzz_ok = _robust_fuzz_generation(base_img, analysis)
    final_maps["fuzz"] = fuzz_map
    diagnostics["fuzz_map_operational"] = fuzz_ok

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
    final_maps, diagnostics = _update_final_maps(base_img, analysis, prepared_maps)
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
    else:
        LOGGER.info("All PBR maps validated successfully")
    composite = apply_alpha(base_img, alpha_map)
    foreground_texture = np.asarray(base_img.convert("L"), dtype=np.float32) / 255.0
    quality_checks = _evaluate_quality_checks_improved(
        final_maps,
        base_image=base_img,
        composite=composite,
        foreground_texture=foreground_texture,
        fuzz_ok=diagnostics.get("fuzz_map_operational", True),
        transmission_ok=diagnostics.get("transmission_preserved", True),
    )
    quality_report = generate_quality_report(final_maps, analysis)
    return {
        "maps": final_maps,
        "analysis": analysis,
        "validation": final_validation,
        "corrections_applied": corrections,
        "quality_report": quality_report,
        "alpha": alpha_map,
        "quality_checks": quality_checks,
        "coherence_issues": diagnostics.get("coherence_issues", []),
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
