"""Map generation helpers for the physically accurate PBR pipeline."""
from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image, ImageFilter

from ..geometry_maps import ambient_occlusion, height_map
from ..illumination_maps import subsurface_map
from ..surface_maps import normal_map
from ..semantic_maps import structural_map

from ._image_features import ensure_variation, gaussian_blur
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
    scales_like = likelihoods.get("scales", np.zeros_like(mask))
    false_risk = analysis.material_analysis.false_metal_risks

    specular = analysis.specular_achromaticity
    diffuse = analysis.diffuse_albedo

    candidate = (
        (metal_like > 0.6)
        & (false_risk < 0.2)
        & (specular > max(0.65, CRITICAL_PARAMETERS["METALLIC_SPECULAR_MIN"]))
        & (diffuse < 0.45)
    )
    organic_zones = (organic_like > 0.5) | (skin_like > 0.4) | (scales_like > 0.4)

    metallic_map = np.zeros_like(mask, dtype=np.float32)
    metallic_map[candidate & ~organic_zones] = 1.0

    if current_metallic is not None:
        prev = _to_float_array(current_metallic)
        metallic_map = np.maximum(metallic_map, (prev > 0.9).astype(np.float32) * 0.5)

    metallic_map = strategic_gaussian_blur(metallic_map, mask, sigma=0.8, preserve_organic=organic_zones)
    metallic_map = enforce_metallic_coherence(metallic_map, analysis)

    if metallic_map.sum() < 3:
        metallic_map[:] = 0.0

    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(metallic_map, alpha)


def generate_ior_physically_accurate(
    transmission: Image.Image,
    material_analysis,
    analysis: AnalysisResult | None = None,
) -> Image.Image:
    transmission_np = _to_float_array(transmission)
    ior_map = np.ones_like(transmission_np) * CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"]

    translucent_seed = transmission_np
    if analysis is not None:
        translucent_seed = np.maximum(translucent_seed, analysis.transmission_seed)
    translucent_mask = translucent_seed > 0.08

    for material, value in CRITICAL_PARAMETERS["IOR_TRANSLUCENT_RANGES"].items():
        likelihood = material_analysis.likelihoods.get(material, np.zeros_like(transmission_np))
        zone = material_analysis.zones.get(material, likelihood > 0.6)
        material_mask = translucent_mask & (likelihood > 0.6) & zone
        if analysis is not None and material in {"thin_fabric", "wings", "fins", "leaves"}:
            thin_regions = analysis.geometric_features.get("thin_regions", np.zeros_like(transmission_np))
            material_mask &= thin_regions > 0.1
        ior_map[material_mask] = value

    residual = translucent_mask & (ior_map <= CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"] + 1e-3)
    if np.any(residual):
        ior_map[residual] = np.clip(1.05 + 0.25 * translucent_seed[residual], 1.0, MAX_IOR)

    if analysis is not None:
        ior_map[analysis.material_analysis.false_metal_risks > 0.2] = CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"]
        ior_map[analysis.mask < 0.05] = CRITICAL_PARAMETERS["IOR_OPAQUE_DEFAULT"]
        alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    else:
        alpha = transmission.split()[-1]

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
    alpha = ensure_variation(alpha)
    alpha_image = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    return _from_single_channel(alpha, alpha_image)


def is_thin_section(geometric_features: Dict[str, np.ndarray]) -> np.ndarray:
    thin = geometric_features.get("thin_regions")
    if thin is None:
        return np.zeros((1, 1), dtype=np.float32)
    return thin


def generate_transmission_physically_accurate(analysis: AnalysisResult, current_transmission: Image.Image | None) -> Image.Image:
    mask = analysis.mask
    likelihoods = analysis.material_analysis.likelihoods
    thin_regions = analysis.geometric_features.get("thin_regions", np.zeros_like(mask))
    seed = analysis.transmission_seed

    transmission_map = np.clip(seed * 0.5, 0.0, 1.0)

    translucent_materials = [
        ("glass", 0.8, False),
        ("crystal", 0.8, False),
        ("water", 0.6, False),
        ("ice", 0.6, False),
        ("thin_fabric", 0.3, True),
        ("wings", 0.35, True),
        ("fins", 0.4, True),
        ("leaves", 0.25, True),
    ]
    for material, value, requires_thin in translucent_materials:
        likelihood = likelihoods.get(material, np.zeros_like(mask))
        zone = analysis.material_analysis.zones.get(material, likelihood > 0.6)
        material_mask = (likelihood > 0.6) & zone & (mask > 0.05)
        if requires_thin:
            material_mask &= thin_regions > 0.1
        transmission_map[material_mask] = np.maximum(transmission_map[material_mask], value)

    if current_transmission is not None:
        prev = _to_float_array(current_transmission)
        transmission_map = np.maximum(transmission_map, prev * 0.5)

    metallic = likelihoods.get("metal", np.zeros_like(mask))
    transmission_map[metallic > 0.4] = 0.0
    transmission_map[mask < 0.05] = 0.0

    transmission_map = strategic_gaussian_blur(transmission_map, mask, sigma=1.0)
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
    base_subsurface = subsurface_map.generate(analysis.base_image)
    subsurface_np = _to_float_array(base_subsurface)
    organic = analysis.material_analysis.likelihoods.get("organic", np.zeros_like(subsurface_np))
    thin_sections = analysis.geometric_features.get("thin_regions", np.zeros_like(subsurface_np))
    transmission_np = _to_float_array(transmission)
    subsurface_np = np.clip(subsurface_np * 0.4 + organic * 0.35 + thin_sections * 0.2 + transmission_np * 0.3, 0.0, 1.0)
    subsurface_np *= analysis.mask
    subsurface_np = ensure_variation(subsurface_np)
    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(subsurface_np, alpha)


def generate_structural_distinct(analysis: AnalysisResult, current_structural: Image.Image | None) -> Image.Image:
    base = structural_map.generate(analysis.base_image)
    base_array = _to_float_array(base)
    edge_map = analysis.geometric_features.get("edge_map")
    luminance = analysis.luminance_map
    high_pass = np.clip(luminance - gaussian_blur(luminance, radius=2.5) + 0.5, 0.0, 1.0)

    structural = np.clip(0.5 * base_array + 0.3 * edge_map + 0.2 * high_pass, 0.0, 1.0)
    if current_structural is not None:
        previous = _to_float_array(current_structural)
        structural = np.clip(0.6 * structural + 0.4 * previous, 0.0, 1.0)
    structural *= analysis.mask
    structural = ensure_variation(structural)

    alpha = Image.fromarray((analysis.alpha * 255.0).astype(np.uint8), mode="L")
    return _from_single_channel(structural, alpha)


def generate_normal_enhanced(base_img: Image.Image, analysis: AnalysisResult) -> Image.Image:
    return normal_map.generate(base_img)


def generate_height_from_normal(normal_map_image: Image.Image, analysis: AnalysisResult) -> Image.Image:
    return height_map.generate(analysis.base_image)


def generate_ao_physically_accurate(analysis: AnalysisResult) -> Image.Image:
    return ambient_occlusion.generate(analysis.base_image)


def generate_curvature_enhanced(analysis: AnalysisResult) -> Image.Image:
    curvature = analysis.geometric_features.get("curvature")
    curvature = np.clip(curvature * analysis.mask, 0.0, 1.0)
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
    "generate_curvature_enhanced",
    "generate_emissive_accurate",
    "generate_height_from_normal",
    "generate_ior_physically_accurate",
    "generate_metallic_physically_accurate",
    "generate_normal_enhanced",
    "generate_roughness_physically_accurate",
    "generate_specular_coherent",
    "generate_structural_distinct",
    "generate_subsurface_accurate",
    "generate_transmission_physically_accurate",
]
