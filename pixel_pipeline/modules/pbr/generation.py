"""Map generation helpers for the physically accurate PBR pipeline."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image, ImageFilter

from ._image_features import ensure_variation, gaussian_blur, normalize01
from .analysis import AnalysisResult
from .parameters import CRITICAL_PARAMETERS

MAX_IOR = max(CRITICAL_PARAMETERS["IOR_TRANSLUCENT_RANGES"].values())
IOR_RANGE = max(MAX_IOR - CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"], 1e-3)


def _to_float_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _from_single_channel(array: np.ndarray, alpha: Image.Image | None = None) -> Image.Image:
    arr = np.clip(array, 0.0, 1.0)
    channel = (arr * 255).astype(np.uint8)
    gray = Image.fromarray(channel, mode="L")
    if alpha is None:
        alpha = Image.new("L", gray.size, 255)
    return Image.merge("RGBA", (gray, gray, gray, alpha))


def _is_pixel_art(analysis: AnalysisResult) -> bool:
    width, height = analysis.base_image.size
    if min(width, height) <= 64:
        return True
    if width * height <= 128 * 128:
        rgba = np.asarray(analysis.base_image.convert("RGBA"), dtype=np.uint8)
        unique_colors = np.unique(rgba[..., :3].reshape(-1, 3), axis=0)
        if unique_colors.shape[0] <= 512:
            return True
    return False


def _compute_height_field(analysis: AnalysisResult) -> np.ndarray:
    luminance = analysis.luminance_map
    edge_map = analysis.geometric_features.get("edge_map", np.zeros_like(luminance))
    thickness = analysis.geometric_features.get("thickness", np.ones_like(luminance))

    base_field = np.clip(0.55 * luminance + 0.35 * edge_map + 0.15 * (1.0 - thickness), 0.0, 1.0)

    likelihoods = analysis.material_analysis.likelihoods
    organic = likelihoods.get("organic", np.zeros_like(base_field))
    skin = likelihoods.get("skin", np.zeros_like(base_field))
    metal = likelihoods.get("metal", np.zeros_like(base_field))
    stone = likelihoods.get("stone", np.zeros_like(base_field))

    amplitude = 0.45 + 0.35 * metal + 0.25 * stone - 0.25 * organic - 0.35 * skin
    amplitude = np.clip(amplitude, 0.18, 1.0)

    smoothed = gaussian_blur(base_field, radius=0.9)
    height_field = ensure_variation(np.clip(smoothed * amplitude, 0.0, 1.0))
    height_field *= analysis.mask
    height_field[analysis.background_mask] = 0.0
    return height_field


def _height_to_normal(height_field: np.ndarray) -> np.ndarray:
    blurred = gaussian_blur(height_field, radius=1.2)
    grad_y, grad_x = np.gradient(blurred)
    normal = np.dstack((-grad_x, -grad_y, np.ones_like(blurred)))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / np.clip(norm, 1e-5, None)
    normal = np.clip((normal * 0.5) + 0.5, 0.0, 1.0)
    return normal


def strategic_gaussian_blur(array: np.ndarray, mask: np.ndarray, sigma: float, preserve_organic: np.ndarray | None = None) -> np.ndarray:
    image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    blurred_np = np.asarray(blurred, dtype=np.float32) / 255.0
    if preserve_organic is not None:
        preserve = preserve_organic.astype(bool)
        blurred_np[preserve] = array[preserve]
    return np.clip(blurred_np * mask, 0.0, 1.0)


def enforce_metallic_coherence(metallic_map: np.ndarray, analysis: AnalysisResult) -> np.ndarray:
    mask = analysis.mask > 0.05
    data = np.where(mask, metallic_map, 0.0)
    data[analysis.material_analysis.false_metal_risks > 0.2] = 0.0
    image = Image.fromarray((np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    opened = image.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    coherent = (np.asarray(opened, dtype=np.float32) / 255.0) > 0.5
    return coherent.astype(np.float32)


def generate_metallic_physically_accurate(base_img: Image.Image, analysis: AnalysisResult, current_metallic: Image.Image | None) -> Image.Image:
    mask = analysis.mask
    likelihoods = analysis.material_analysis.likelihoods
    metal_like = likelihoods.get("metal", np.zeros_like(mask))
    organic_like = likelihoods.get("organic", np.zeros_like(mask))
    skin_like = likelihoods.get("skin", np.zeros_like(mask))
    fabric_like = likelihoods.get("thin_fabric", np.zeros_like(mask))
    wood_like = likelihoods.get("wood", np.zeros_like(mask))
    false_risk = analysis.material_analysis.false_metal_risks

    specular = analysis.specular_achromaticity
    saturation = analysis.hsv[1]
    roughness_hint = np.clip(1.0 - specular, 0.0, 1.0)

    candidate = (
        (metal_like > 0.55)
        & (false_risk < 0.25)
        & (specular > max(0.7, CRITICAL_PARAMETERS["METALLIC_SPECULAR_MIN"]))
        & (roughness_hint < 0.35)
        & (saturation < 0.2)
        & (mask > 0.05)
    )

    metallic_map = np.zeros_like(mask, dtype=np.float32)
    metallic_map[candidate] = np.clip(metal_like[candidate], 0.0, 1.0)

    if current_metallic is not None:
        prev = _to_float_array(current_metallic)
        metallic_map = np.maximum(metallic_map, np.clip(prev, 0.0, 1.0) * 0.6)

    organic_suppression = (
        (organic_like > 0.3)
        | (skin_like > 0.3)
        | (fabric_like > 0.3)
        | (wood_like > 0.3)
    )

    metallic_map[organic_suppression] = np.minimum(metallic_map[organic_suppression], CRITICAL_PARAMETERS["METALLIC_ORGANIC_TOLERANCE"])
    metallic_map[false_risk > 0.25] = 0.0

    metallic_image = Image.fromarray((np.clip(metallic_map, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    opened = metallic_image.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    closed = opened.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    metallic_map = np.asarray(closed, dtype=np.float32) / 255.0

    metallic_map = strategic_gaussian_blur(metallic_map, mask, sigma=0.6, preserve_organic=organic_suppression)
    metallic_map = enforce_metallic_coherence(metallic_map, analysis)
    metallic_map = np.clip(metallic_map * mask, 0.0, 1.0)

    if metallic_map.sum() < 3:
        metallic_map[:] = 0.0

    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(metallic_map, alpha)


def _prepare_analysis_channel(channel: np.ndarray | Iterable[float], target_shape: Tuple[int, int]) -> np.ndarray:
    """Normalize and resize analysis channels to match ``target_shape``."""

    array = np.asarray(channel, dtype=np.float32)
    if array.ndim >= 3:
        array = array[..., 0]
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.size == 0:
        return np.zeros(target_shape, dtype=np.float32)
    if array.shape != target_shape:
        image = Image.fromarray((normalize01(array) * 255.0).astype(np.uint8), mode="L")
        image = image.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
    max_value = float(array.max())
    scale = max(max_value, 1e-5)
    return np.clip(array / scale, 0.0, 1.0).astype(np.float32, copy=False)


def generate_porosity_accurate(analysis: AnalysisResult) -> Image.Image:
    """Generate a porosity map derived from unified analysis features."""

    mask = analysis.mask
    thickness = analysis.geometric_features.get("thickness", np.ones_like(mask))
    edge_density = analysis.geometric_features.get("edge_map", np.zeros_like(mask))
    cavities = 1.0 - np.clip(thickness, 0.0, 1.0)
    rough_hint = np.clip(1.0 - analysis.specular_achromaticity, 0.0, 1.0)

    porosity = np.clip(cavities * 0.55 + rough_hint * 0.25 + edge_density * 0.2, 0.0, 1.0)
    porosity *= mask
    porosity = gaussian_blur(porosity, radius=0.6)
    porosity = ensure_variation(porosity)

    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(porosity, alpha)


def generate_opacity_accurate(analysis: AnalysisResult) -> Image.Image:
    """Generate an opacity map coherent with transmission cues."""

    mask = analysis.mask
    base_alpha = np.clip(analysis.alpha, 0.0, 1.0)
    translucency = np.clip(analysis.transmission_seed, 0.0, 1.0)
    thin_regions = analysis.geometric_features.get("thin_regions", np.zeros_like(mask))

    opacity = np.clip(base_alpha * 0.7 + (1.0 - translucency) * 0.2 + (1.0 - thin_regions) * 0.1, 0.0, 1.0)
    opacity *= mask
    opacity = gaussian_blur(opacity, radius=0.8)
    opacity = ensure_variation(opacity)

    alpha = Image.fromarray((base_alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(opacity, alpha)


def generate_fuzz_accurate(base_img: Image.Image, analysis: AnalysisResult) -> Image.Image:
    """Generate fuzz map using microsurface descriptors with graceful fallback."""

    width, height = base_img.size
    target_shape = (height, width)

    microsurface = getattr(analysis, "microsurface_data", None)
    fuzz_potential = getattr(analysis, "fuzz_potential", None)

    if microsurface is not None and fuzz_potential is not None:
        micro_channel = _prepare_analysis_channel(microsurface, target_shape)
        fuzz_channel = _prepare_analysis_channel(fuzz_potential, target_shape)
    else:
        gray = np.asarray(base_img.convert("L"), dtype=np.float32) / 255.0
        micro_channel = gaussian_blur(gray, radius=0.9)
        fuzz_channel = np.clip(1.0 - micro_channel, 0.0, 1.0)

    combined = np.clip(micro_channel * 0.6 + fuzz_channel * 0.4, 0.0, 1.0)
    combined *= analysis.mask
    combined = gaussian_blur(combined, radius=0.5)
    combined = ensure_variation(combined)

    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(combined, alpha)


def generate_material_accurate(analysis: AnalysisResult) -> Image.Image:
    """Generate a semantic material map from dominant material likelihoods."""

    mask = analysis.mask
    height, width = mask.shape
    material = analysis.material_analysis
    likelihoods = material.likelihoods if material is not None else {}

    palette: Dict[str, Tuple[float, float, float]] = {
        "metal": (0.75, 0.75, 0.85),
        "stone": (0.55, 0.55, 0.6),
        "organic": (0.45, 0.7, 0.4),
        "skin": (0.85, 0.6, 0.55),
        "fabric": (0.6, 0.35, 0.65),
        "wood": (0.65, 0.45, 0.3),
        "glass": (0.6, 0.8, 0.95),
        "water": (0.4, 0.6, 0.85),
    }

    color = np.zeros((height, width, 3), dtype=np.float32)
    total = np.zeros((height, width, 1), dtype=np.float32)

    for name, rgb in palette.items():
        likelihood = likelihoods.get(name)
        if likelihood is None:
            continue
        channel = _prepare_analysis_channel(likelihood, (height, width))
        rgb_vec = np.array(rgb, dtype=np.float32).reshape(1, 1, 3)
        color += channel[..., None] * rgb_vec
        total += channel[..., None]

    if np.all(total <= 1e-6):
        base_rgb = np.asarray(analysis.rgb, dtype=np.float32)
        base_rgb = normalize01(base_rgb)
        color = base_rgb
    else:
        color = np.divide(color, np.maximum(total, 1e-6))

    color = np.clip(color * mask[..., None], 0.0, 1.0)
    color = ensure_variation(color)

    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    rgb_image = Image.fromarray((color * 255.0).astype(np.uint8), mode="RGB")
    return Image.merge("RGBA", (*rgb_image.split(), alpha))


def generate_ior_physically_accurate(
    transmission: Image.Image,
    material_analysis,
    analysis: AnalysisResult | None = None,
) -> Image.Image:
    transmission_np = _to_float_array(transmission)
    likelihoods = material_analysis.likelihoods

    ior_map = np.ones_like(transmission_np) * 1.45
    material_ior = {
        "glass": 1.50,
        "water": 1.33,
        "skin": 1.40,
        "stone": 1.54,
        "crystal": 1.65,
    }

    mask = np.ones_like(transmission_np, dtype=bool)
    if analysis is not None:
        mask = analysis.mask > 0.05

    for material, value in material_ior.items():
        likelihood = likelihoods.get(material, np.zeros_like(transmission_np))
        zone = material_analysis.zones.get(material, likelihood > 0.6)
        material_mask = mask & (likelihood > 0.5) & zone
        if analysis is not None and material in {"skin"}:
            organic = analysis.material_analysis.likelihoods.get("organic", np.zeros_like(transmission_np))
            material_mask &= organic > 0.3
        if np.any(material_mask):
            ior_map[material_mask] = value

    if analysis is not None:
        translucent_seed = np.maximum(transmission_np, analysis.transmission_seed)
    else:
        translucent_seed = transmission_np

    ior_map = ior_map * (1.0 - translucent_seed) + 1.33 * translucent_seed

    if analysis is not None:
        roughness_hint = np.clip(1.0 - analysis.specular_achromaticity, 0.0, 1.0)
    else:
        roughness_hint = np.clip(1.0 - transmission_np, 0.0, 1.0)

    blend = np.clip(roughness_hint * 0.3, 0.0, 1.0)
    ior_map = ior_map * (1.0 - blend) + 1.48 * blend

    if analysis is not None:
        ior_map[analysis.material_analysis.false_metal_risks > 0.2] = CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"]
        ior_map[analysis.mask < 0.05] = CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"]
        alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    else:
        alpha = transmission.split()[-1]

    ior_map = np.clip(ior_map, CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"], MAX_IOR)
    normalized = np.clip((ior_map - CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"]) / IOR_RANGE, 0.0, 1.0)
    return _from_single_channel(normalized, alpha)


def has_meaningful_alpha(alpha_image: Image.Image | None) -> bool:
    if alpha_image is None:
        return False
    alpha = _to_float_array(alpha_image)
    return float(np.std(alpha)) > 0.02


def refine_alpha_from_existing(alpha_image: Image.Image, analysis: AnalysisResult) -> np.ndarray:
    alpha = _to_float_array(alpha_image)
    refined = np.clip(alpha * 0.85 + analysis.mask * 0.15, 0.0, 1.0)
    refined[analysis.background_mask] = 0.0
    return refined


def generate_alpha_from_scratch(analysis: AnalysisResult) -> np.ndarray:
    rgb = analysis.rgb
    background_mask = analysis.background_mask
    if np.any(background_mask):
        bg_color = np.median(rgb[background_mask], axis=0)
    else:
        bg_color = np.median(rgb.reshape(-1, 3), axis=0)
    color_distance = np.linalg.norm(rgb - bg_color, axis=-1)
    color_distance = gaussian_blur(color_distance, radius=1.0)
    color_distance = ensure_variation(np.clip(color_distance, 0.0, None))
    normalized = (color_distance - color_distance.min()) / (color_distance.max() - color_distance.min() + 1e-6)
    alpha = np.clip(normalized * analysis.mask, 0.0, 1.0)
    alpha[analysis.background_mask] = 0.0
    return alpha


def apply_edge_falloff(alpha: np.ndarray, mask: np.ndarray, falloff_pixels: int) -> np.ndarray:
    radius = max(falloff_pixels, 1)
    blurred_alpha = gaussian_blur(alpha, radius=radius)
    blurred_mask = gaussian_blur(mask, radius=radius)
    smoothed = np.clip(0.7 * blurred_alpha + 0.3 * blurred_mask, 0.0, 1.0)
    smoothed[mask < 0.05] = 0.0
    return smoothed


def is_flat_alpha(alpha: np.ndarray) -> bool:
    return float(np.std(alpha)) < 0.01


def emergency_alpha_generation(analysis: AnalysisResult) -> np.ndarray:
    gradient = analysis.geometric_features.get("edge_map")
    emergency = np.clip(analysis.mask * 0.9 + gradient * 0.1, 0.0, 1.0)
    emergency[analysis.background_mask] = 0.0
    return emergency


def generate_alpha_accurate(analysis: AnalysisResult, current_alpha: Image.Image | None) -> Image.Image:
    if has_meaningful_alpha(current_alpha):
        alpha = refine_alpha_from_existing(current_alpha, analysis)
    else:
        alpha = generate_alpha_from_scratch(analysis)
    alpha = apply_edge_falloff(alpha, analysis.mask, falloff_pixels=CRITICAL_PARAMETERS["ALPHA_EDGE_FALLOFF"])
    if is_flat_alpha(alpha):
        alpha = emergency_alpha_generation(analysis)
    physical_alpha = np.clip(np.maximum(alpha, analysis.alpha) * 0.6 + analysis.mask * 0.4, 0.0, 1.0)
    physical_alpha = gaussian_blur(physical_alpha, radius=0.75)
    physical_alpha[analysis.background_mask] = 0.0

    if _is_pixel_art(analysis):
        physical_alpha = (physical_alpha >= 0.5).astype(np.float32)

    physical_alpha = ensure_variation(physical_alpha)
    alpha_image = Image.fromarray((physical_alpha * 255).astype(np.uint8), mode="L")
    return _from_single_channel(physical_alpha, alpha_image)


def is_thin_section(geometric_features: Dict[str, np.ndarray]) -> np.ndarray:
    thin = geometric_features.get("thin_regions")
    if thin is None:
        return np.zeros((1, 1), dtype=np.float32)
    return thin


def generate_transmission_physically_accurate(analysis: AnalysisResult, current_transmission: Image.Image | None) -> Image.Image:
    mask = analysis.mask
    likelihoods = analysis.material_analysis.likelihoods
    thin_regions = analysis.geometric_features.get("thin_regions", np.zeros_like(mask))
    thickness = analysis.geometric_features.get("thickness", np.ones_like(mask))
    luminance = analysis.luminance_map
    saturation = analysis.hsv[1]

    base = np.clip(luminance * (1.0 - saturation), 0.0, 1.0)
    seed = np.clip(np.maximum(base, analysis.transmission_seed) * 0.6, 0.0, 1.0)

    transmission_map = seed

    translucent_materials = [
        ("glass", 0.85, False),
        ("crystal", 0.8, False),
        ("water", 0.65, False),
        ("ice", 0.6, False),
        ("thin_fabric", 0.35, True),
        ("wings", 0.35, True),
        ("fins", 0.35, True),
        ("leaves", 0.3, True),
    ]
    for material, value, requires_thin in translucent_materials:
        likelihood = likelihoods.get(material, np.zeros_like(mask))
        zone = analysis.material_analysis.zones.get(material, likelihood > 0.6)
        material_mask = (likelihood > 0.55) & zone & (mask > 0.05)
        if requires_thin:
            material_mask &= thin_regions > 0.1
        if np.any(material_mask):
            transmission_map[material_mask] = np.maximum(transmission_map[material_mask], value)

    bright_low_sat = (luminance > 0.65) & (saturation < 0.25)
    transmission_map[bright_low_sat] = np.maximum(transmission_map[bright_low_sat], 0.45)

    if current_transmission is not None:
        prev = _to_float_array(current_transmission)
        transmission_map = np.maximum(transmission_map, prev * 0.6)

    metallic = likelihoods.get("metal", np.zeros_like(mask))
    stone = likelihoods.get("stone", np.zeros_like(mask))
    opaque_regions = (metallic > 0.4) | (stone > 0.5)
    transmission_map[opaque_regions] = 0.0

    delicate_regions = (thin_regions > 0.2) & (thickness < 0.35)
    if np.any(delicate_regions):
        transmission_map[delicate_regions] = np.clip(transmission_map[delicate_regions], 0.15, 0.35)

    transmission_map[mask < 0.05] = 0.0
    minimum_transmission = 0.02
    transmission_map = np.where(mask > 0.05, np.maximum(transmission_map, minimum_transmission), transmission_map)
    transmission_map[opaque_regions] = 0.0

    transmission_map = strategic_gaussian_blur(transmission_map, mask, sigma=0.9)
    transmission_map = ensure_variation(np.clip(transmission_map, 0.0, 1.0))

    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(transmission_map, alpha)


def generate_roughness_physically_accurate(base_img: Image.Image, analysis: AnalysisResult, metallic_map: Image.Image) -> Image.Image:
    edge_map = analysis.geometric_features.get("edge_map")
    specular = analysis.specular_achromaticity
    metallic_np = _to_float_array(metallic_map)

    roughness = np.clip(1.0 - specular * 0.6 + edge_map * 0.2, 0.05, 1.0)
    roughness = np.where(metallic_np > 0.5, np.clip(roughness * 0.4, 0.05, 0.65), roughness)
    roughness = ensure_variation(roughness * analysis.mask + (1.0 - analysis.mask) * 0.05)

    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(roughness, alpha)


def generate_specular_coherent(roughness_map: Image.Image, analysis: AnalysisResult) -> Image.Image:
    roughness = _to_float_array(roughness_map)
    specular = np.clip(1.0 - roughness * 0.85, 0.05, 1.0)
    specular *= analysis.mask
    specular = ensure_variation(specular)
    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(specular, alpha)


def generate_subsurface_accurate(analysis: AnalysisResult, transmission: Image.Image) -> Image.Image:
    organic = analysis.material_analysis.likelihoods.get("organic", np.zeros_like(analysis.mask))
    skin = analysis.material_analysis.likelihoods.get("skin", np.zeros_like(analysis.mask))
    organic_mask = np.clip(organic + skin * 0.5, 0.0, 1.0)

    if not np.any(organic_mask > 0.2):
        organic_mask = np.zeros_like(analysis.mask)

    transmission_np = _to_float_array(transmission)
    alpha_channel = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    min_alpha = np.asarray(alpha_channel.filter(ImageFilter.MinFilter(3)), dtype=np.float32) / 255.0

    low_freq = gaussian_blur(analysis.luminance_map, radius=3.5)
    subsurface_np = np.clip(0.5 * low_freq + 0.3 * min_alpha + 0.2 * transmission_np, 0.0, 1.0)
    subsurface_np *= organic_mask
    subsurface_np *= analysis.mask
    subsurface_np = ensure_variation(subsurface_np)

    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(subsurface_np, alpha)


def generate_structural_distinct(analysis: AnalysisResult, current_structural: Image.Image | None) -> Image.Image:
    height_field = _compute_height_field(analysis)
    grad_y, grad_x = np.gradient(height_field)
    sobel = np.clip(np.sqrt(grad_x ** 2 + grad_y ** 2), 0.0, 1.0)

    normal_field = _height_to_normal(height_field)
    normal_mag = np.linalg.norm(normal_field - 0.5, axis=2)
    log_normal = np.abs(normal_mag - gaussian_blur(normal_mag, radius=1.2))

    edge_map = analysis.geometric_features.get("edge_map", np.zeros_like(height_field))
    structural = np.clip(0.45 * sobel + 0.35 * log_normal + 0.2 * edge_map, 0.0, 1.0)

    if current_structural is not None:
        previous = _to_float_array(current_structural)
        structural = np.clip(0.6 * structural + 0.4 * previous, 0.0, 1.0)

    structural *= analysis.mask
    structural = ensure_variation(structural)

    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(structural, alpha)


def generate_normal_enhanced(base_img: Image.Image, analysis: AnalysisResult) -> Image.Image:
    height_field = _compute_height_field(analysis)
    normals = _height_to_normal(height_field)
    normal_bytes = (np.clip(normals, 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha_channel = (analysis.alpha * 255.0).astype(np.uint8)
    normal_rgba = np.dstack((normal_bytes, alpha_channel[..., None]))
    return Image.fromarray(normal_rgba, mode="RGBA")


def generate_height_from_normal(normal_map_image: Image.Image, analysis: AnalysisResult) -> Image.Image:
    height_field = _compute_height_field(analysis)
    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(height_field, alpha)


def generate_ao_physically_accurate(analysis: AnalysisResult) -> Image.Image:
    height_field = _compute_height_field(analysis)
    min_dimension = min(height_field.shape)
    kernel = max(3, round(min_dimension / 48))
    radius = max(1, kernel // 2)
    blurred = gaussian_blur(height_field, radius=radius)
    thickness = analysis.geometric_features.get("thickness", np.ones_like(height_field))
    ao = np.clip(1.0 - blurred * 0.7 - (1.0 - thickness) * 0.3, 0.0, 1.0)
    ao *= analysis.mask
    ao = ensure_variation(np.clip(ao, 0.0, 1.0))
    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(ao, alpha)


def generate_curvature_enhanced(analysis: AnalysisResult) -> Image.Image:
    height_field = _compute_height_field(analysis)
    grad_y, grad_x = np.gradient(height_field)
    dxx = np.gradient(grad_x, axis=1)
    dyy = np.gradient(grad_y, axis=0)
    curvature = np.abs(dxx) + np.abs(dyy)
    local_norm = gaussian_blur(curvature, radius=2.0) + 1e-4
    curvature = np.clip(curvature / local_norm, 0.0, 1.0)
    curvature *= analysis.mask
    curvature = ensure_variation(curvature)
    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(curvature, alpha)


def generate_emissive_accurate(base_img: Image.Image, analysis: AnalysisResult) -> Image.Image:
    emissive = np.clip((analysis.luminance_map - 0.6) * 2.0 + analysis.specular_achromaticity * 0.3, 0.0, 1.0)
    emissive *= analysis.mask
    emissive = ensure_variation(emissive)
    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(emissive, alpha)


def generate_secondary_maps(base_img: Image.Image, analysis: AnalysisResult) -> Dict[str, Image.Image]:
    return {
        "normal": generate_normal_enhanced(base_img, analysis),
    }


__all__ = [
    "generate_alpha_accurate",
    "generate_ao_physically_accurate",
    "generate_fuzz_accurate",
    "generate_curvature_enhanced",
    "generate_emissive_accurate",
    "generate_height_from_normal",
    "generate_ior_physically_accurate",
    "generate_material_accurate",
    "generate_metallic_physically_accurate",
    "generate_normal_enhanced",
    "generate_opacity_accurate",
    "generate_porosity_accurate",
    "generate_roughness_physically_accurate",
    "generate_specular_coherent",
    "generate_structural_distinct",
    "generate_subsurface_accurate",
    "generate_transmission_physically_accurate",
]
