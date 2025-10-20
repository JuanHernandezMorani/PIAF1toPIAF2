"""Comprehensive analysis utilities for physically accurate PBR generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
from PIL import Image, ImageFilter

from ._image_features import (
    edge_strength as edge_strength_feature,
    gaussian_blur,
    hue as compute_hue,
    luminance as compute_luminance,
    normalize01,
    saturation as compute_saturation,
    to_rgb_alpha,
)

@dataclass
class MaterialAnalysis:
    """Container for material likelihood information."""

    likelihoods: Dict[str, np.ndarray]
    zones: Dict[str, np.ndarray]
    false_metal_risks: np.ndarray


@dataclass
class AnalysisResult:
    """Aggregate structure returned by :func:`analyze_image_comprehensive`."""

    mask: np.ndarray
    alpha: np.ndarray
    background_mask: np.ndarray
    rgb: np.ndarray
    hsv: Tuple[np.ndarray, np.ndarray, np.ndarray]
    luminance_map: np.ndarray
    specular_achromaticity: np.ndarray
    diffuse_albedo: np.ndarray
    transmission_seed: np.ndarray
    material_analysis: MaterialAnalysis
    geometric_features: Dict[str, np.ndarray]
    current_issues: Dict[str, Iterable[str]]
    base_image: Image.Image
    background_image: Image.Image | None

    @property
    def material_class(self) -> str:
        likelihoods = getattr(self.material_analysis, "likelihoods", {}) or {}
        if not likelihoods:
            return "unknown"
        scores = {name: float(np.mean(field)) for name, field in likelihoods.items() if field is not None}
        if not scores:
            return "unknown"
        return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _pil_to_np(image: Image.Image, *, mode: str = "RGB") -> np.ndarray:
    converted = image.convert(mode)
    array = np.asarray(converted, dtype=np.float32)
    if mode in {"RGB", "L"}:
        return array / 255.0
    return array


def extract_alpha_if_exists(image: Image.Image | None) -> np.ndarray | None:
    """Extract a normalized alpha channel when present."""

    if image is None:
        return None
    if "A" not in image.getbands():
        return None
    alpha = np.asarray(image.split()[-1], dtype=np.float32) / 255.0
    return alpha


def estimate_foreground_mask_v2(
    base_img: Image.Image,
    bg_img: Image.Image | None,
    *,
    alpha: np.ndarray | None = None,
    min_foreground_ratio: float = 0.02,
) -> np.ndarray:
    """Estimate a soft foreground mask using alpha and color differences."""

    base_rgb = _pil_to_np(base_img, mode="RGB")
    if alpha is not None and np.any(alpha > 0.05):
        mask = alpha > 0.02
    else:
        if bg_img is not None:
            bg_rgb = _pil_to_np(bg_img, mode="RGB")
            diff = np.sqrt(np.sum((base_rgb - bg_rgb) ** 2, axis=2))
        else:
            luminance = base_rgb.mean(axis=2)
            diff = np.abs(luminance - np.mean(luminance))
        threshold = np.median(diff[diff > 0]) if np.any(diff > 0) else 0.0
        mask = diff > max(threshold, 0.05)

    mask = mask.astype(np.float32)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(size=3))
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=1.2))
    refined = np.asarray(mask_image, dtype=np.float32) / 255.0
    if refined.mean() < min_foreground_ratio:
        # Promote the highest contrast pixels to ensure we have a usable mask
        flat = np.abs(base_rgb - base_rgb.mean(axis=(0, 1), keepdims=True)).mean(axis=2)
        quantile = np.quantile(flat[flat > 0], 0.8) if np.any(flat > 0) else 0.0
        refined = (flat > quantile).astype(np.float32)
    return np.clip(refined, 0.0, 1.0)


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB→HSV conversion for arrays in range [0, 1]."""

    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc

    mask = delta > 1e-5
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    red = mask & (maxc == r)
    green = mask & (maxc == g)
    blue = mask & (maxc == b)

    h[red] = (g[red] - b[red]) / delta[red]
    h[green] = (b[green] - r[green]) / delta[green] + 2.0
    h[blue] = (r[blue] - g[blue]) / delta[blue] + 4.0

    h = (h / 6.0) % 1.0
    s[maxc > 1e-5] = delta[maxc > 1e-5] / maxc[maxc > 1e-5]
    return h, s, v


def _estimate_background(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Estimate background pixels from border colours and transparency."""

    height, width, _ = rgb.shape
    edge_mask = np.zeros((height, width), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True

    candidate_pixels = rgb[edge_mask]
    if candidate_pixels.size == 0:
        candidate_pixels = rgb.reshape(-1, 3)
    bg_color = np.median(candidate_pixels, axis=0)
    color_distance = np.linalg.norm(rgb - bg_color, axis=-1)
    if np.any(edge_mask):
        threshold = max(0.04, np.percentile(color_distance[edge_mask], 70))
    else:
        threshold = 0.04
    background_mask = color_distance < threshold
    background_mask |= alpha < 0.03
    return background_mask


def _local_contrast(luminance: np.ndarray) -> np.ndarray:
    smooth = gaussian_blur(luminance, radius=1.2)
    contrast = np.abs(luminance - smooth)
    if contrast.max() > 1e-6:
        contrast /= contrast.max()
    return contrast


def compute_material_likelihoods_v2(
    base_img: Image.Image,
    mask: np.ndarray,
    *,
    hue: np.ndarray | None = None,
    saturation: np.ndarray | None = None,
    value: np.ndarray | None = None,
    specular: np.ndarray | None = None,
    translucency: np.ndarray | None = None,
    thin_regions: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Return heuristic likelihood fields for key material types."""

    if hue is None or saturation is None or value is None:
        rgb = _pil_to_np(base_img, mode="RGB")
        hue, saturation, value = _rgb_to_hsv(rgb)
    if specular is None:
        specular = (1.0 - saturation) * 0.5
    if translucency is None:
        translucency = np.zeros_like(value)
    if thin_regions is None:
        thin_regions = np.zeros_like(value, dtype=bool)

    thin_regions = thin_regions.astype(np.float32)
    achromaticity = 1.0 - saturation

    likelihoods: Dict[str, np.ndarray] = {}
    likelihoods["metal"] = np.clip(specular * (0.4 + 0.6 * (value > 0.45)), 0.0, 1.0)
    likelihoods["organic"] = np.clip((0.6 * saturation + 0.3 * np.exp(-((hue - 0.05) ** 2) / 0.004)) * (value > 0.2), 0.0, 1.0)
    likelihoods["skin"] = np.clip(np.exp(-((hue - 0.08) ** 2) / 0.003) * (saturation < 0.55) * (value > 0.25), 0.0, 1.0)
    likelihoods["scales"] = np.clip(np.exp(-((hue - 0.45) ** 2) / 0.01) * (saturation > 0.55), 0.0, 1.0)

    glass = np.clip(translucency * specular, 0.0, 1.0)
    water = np.clip(translucency * np.exp(-((hue - 0.55) ** 2) / 0.008) * (saturation > 0.2), 0.0, 1.0)
    crystal = np.clip(translucency * specular * np.exp(-((hue - 0.58) ** 2) / 0.01), 0.0, 1.0)
    ice = np.clip(translucency * np.exp(-((hue - 0.55) ** 2) / 0.015) * (value > 0.5), 0.0, 1.0)

    likelihoods["glass"] = glass
    likelihoods["water"] = water
    likelihoods["crystal"] = crystal
    likelihoods["ice"] = ice
    likelihoods["thin_fabric"] = np.clip(translucency * (saturation > 0.2) * thin_regions, 0.0, 1.0)
    likelihoods["wings"] = np.clip(translucency * thin_regions * np.exp(-((hue - 0.16) ** 2) / 0.01), 0.0, 1.0)
    likelihoods["fins"] = np.clip(translucency * thin_regions * np.exp(-((hue - 0.55) ** 2) / 0.01), 0.0, 1.0)
    likelihoods["leaves"] = np.clip(translucency * thin_regions * np.exp(-((hue - 0.33) ** 2) / 0.008), 0.0, 1.0)

    for key in likelihoods:
        likelihoods[key] = np.clip(likelihoods[key], 0.0, 1.0) * mask
    return likelihoods


def detect_material_zones(base_img: Image.Image, mask: np.ndarray, likelihoods: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Create boolean zones from material likelihoods."""

    zones: Dict[str, np.ndarray] = {}
    for key, field in likelihoods.items():
        zones[key] = field > 0.55
    return zones


def identify_metallic_false_positive_areas(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Estimate areas prone to metallic false positives."""

    warm = np.exp(-((hue - 0.08) ** 2) / 0.003)
    leafy = np.exp(-((hue - 0.33) ** 2) / 0.01)
    highlight = np.clip(value * (1.0 - saturation), 0.0, 1.0)
    risk = np.clip((warm + leafy) * saturation * highlight * mask, 0.0, 1.0)
    return risk


def extract_geometric_features_enhanced(
    luminance: np.ndarray,
    mask: np.ndarray,
    alpha: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return enhanced geometric descriptors (edges, thickness, curvature cues)."""

    grad_y, grad_x = np.gradient(luminance)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_map = np.clip(gradient_magnitude / (gradient_magnitude.max() + 1e-6), 0.0, 1.0)

    blurred_mask = Image.fromarray((mask * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=2.5))
    blurred = np.asarray(blurred_mask, dtype=np.float32) / 255.0
    thickness = np.clip(mask - blurred + 0.5, 0.0, 1.0)

    curvature = np.abs(np.gradient(grad_x)[0]) + np.abs(np.gradient(grad_y)[1])
    curvature = np.clip(curvature / (curvature.max() + 1e-6), 0.0, 1.0)

    thin_regions = edge_strength_feature(alpha) > 0.18

    return {
        "edge_map": edge_map,
        "thickness": thickness,
        "curvature": curvature,
        "luminance": luminance,
        "thin_regions": thin_regions.astype(np.float32),
    }


def _map_issues(image: Image.Image) -> Iterable[str]:
    array = _pil_to_np(image, mode="L")
    flat = array.max() - array.min() < 0.02
    entropy = calculate_entropy(array)
    issues = []
    if flat:
        issues.append("flat")
    if entropy < 0.02:
        issues.append("low_entropy")
    return issues


def calculate_entropy(array: np.ndarray) -> float:
    values = np.clip((array * 255).astype(np.uint8), 0, 255)
    hist, _ = np.histogram(values, bins=256, range=(0, 255), density=False)
    prob = hist / max(hist.sum(), 1)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def diagnose_current_map_issues(current_maps: Mapping[str, Image.Image], mask: np.ndarray) -> Dict[str, Iterable[str]]:
    """Analyse provided maps and flag flat or low-information outputs."""

    diagnostics: Dict[str, Iterable[str]] = {}
    for name, image in current_maps.items():
        diagnostics[name] = _map_issues(image)
    return diagnostics


def analyze_image_comprehensive(
    base_img: Image.Image,
    bg_img: Image.Image | None,
    current_maps: Mapping[str, Image.Image],
) -> AnalysisResult:
    """Run comprehensive analysis before map generation."""

    rgb, alpha = to_rgb_alpha(base_img)
    alpha = np.clip(alpha, 0.0, 1.0)
    mask = estimate_foreground_mask_v2(base_img, bg_img, alpha=alpha)
    background_mask = _estimate_background(rgb, alpha)

    luminance = compute_luminance(rgb)
    saturation = compute_saturation(rgb)
    hue = compute_hue(rgb)
    value = rgb.max(axis=-1)

    specular = np.clip((1.0 - saturation) * (0.5 + 0.5 * _local_contrast(luminance)), 0.0, 1.0)
    diffuse = np.clip(luminance * (0.6 + 0.4 * saturation), 0.0, 1.0)

    thin_regions = edge_strength_feature(alpha) > 0.18
    translucency = np.clip(
        (alpha > 0.05).astype(np.float32) * 0.6
        + (edge_strength_feature(luminance) * (1.0 - saturation)) * 0.4
        + (1.0 - alpha) * 0.3,
        0.0,
        1.0,
    )
    translucency *= mask
    translucency = normalize01(translucency)

    material_likelihoods = compute_material_likelihoods_v2(
        base_img,
        mask,
        hue=hue,
        saturation=saturation,
        value=value,
        specular=specular,
        translucency=translucency,
        thin_regions=thin_regions,
    )
    material_zones = detect_material_zones(base_img, mask, material_likelihoods)
    false_metal = identify_metallic_false_positive_areas(hue, saturation, value, mask)
    material_analysis = MaterialAnalysis(
        likelihoods=material_likelihoods,
        zones=material_zones,
        false_metal_risks=false_metal,
    )

    geometric_features = extract_geometric_features_enhanced(luminance, mask, alpha)
    geometric_features.setdefault("thin_regions", thin_regions.astype(np.float32))
    current_issues = diagnose_current_map_issues(current_maps, mask)

    return AnalysisResult(
        mask=mask,
        alpha=alpha,
        background_mask=background_mask,
        rgb=rgb,
        hsv=(hue, saturation, value),
        luminance_map=luminance,
        specular_achromaticity=specular,
        diffuse_albedo=diffuse,
        transmission_seed=translucency,
        material_analysis=material_analysis,
        geometric_features=geometric_features,
        current_issues=current_issues,
        base_image=base_img,
        background_image=bg_img,
    )


__all__ = [
    "AnalysisResult",
    "MaterialAnalysis",
    "analyze_image_comprehensive",
    "calculate_entropy",
    "diagnose_current_map_issues",
    "estimate_foreground_mask_v2",
    "extract_alpha_if_exists",
]
