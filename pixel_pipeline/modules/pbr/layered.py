"""Layer-aware orchestration for generating and compositing PBR maps."""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import hashlib
import logging

import numpy as np
from PIL import Image

from .pipeline import (
    generate_physically_accurate_pbr_maps,
    _detect_and_correct_flat_maps_v5,
    _generate_emissive_map_v5,
    _generate_metallic_map_v5,
    _generate_normal_map_v5,
    _generate_roughness_map_v5,
    automated_quality_report_v5,
    validate_pbr_coherence_v5,
)
from .validation import _validate_halo_elimination

LOGGER = logging.getLogger("pixel_pipeline.pbr.layered")


LAYERED_PBR_CONFIG: Mapping[str, object] = {
    "generate_separate_pbr": True,
    "composite_before_color_variant": True,
    "rotation_after_composition": True,
    "cache_layers_separately": True,
    "background_tint_strength": 0.18,
    "save_component_images": False,
    "composition_methods": {
        "albedo": "alpha_blending",
        "base_color": "alpha_blending",
        "normal": "geometric_replacement",
        "height": "geometric_replacement",
        "metallic": "alpha_blending",
        "roughness": "alpha_blending",
        "specular": "alpha_blending",
        "opacity": "optical_composition",
        "transmission": "optical_composition",
    },
}


def _smart_alpha_composition_improved(foreground: Image.Image, background: Image.Image) -> Image.Image:
    """Composite two images while robustly handling dimensional mismatches."""

    def _normalize(image: Image.Image | np.ndarray) -> np.ndarray:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = array[..., None]
        if array.size == 0:
            return np.zeros((1, 1, 4), dtype=np.float32)
        if array.max() > 1.0 + 1e-6 or array.min() < -1e-6:
            array = np.clip(array, 0.0, 255.0)
            if array.max() > 1.0 + 1e-6:
                array = array / 255.0
        return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)

    def _split_rgba(array: np.ndarray, default_alpha: float) -> tuple[np.ndarray, np.ndarray]:
        if array.ndim != 3:
            raise ValueError("Input array must have three dimensions after preprocessing")
        channels = array.shape[2]
        if channels >= 4:
            rgb = array[..., :3]
            alpha = array[..., 3]
        elif channels == 3:
            rgb = array
            alpha = np.full(array.shape[:2], default_alpha, dtype=array.dtype)
        elif channels == 1:
            rgb = np.repeat(array, 3, axis=2)
            alpha = np.full(array.shape[:2], default_alpha, dtype=array.dtype)
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
        return rgb, np.clip(alpha, 0.0, 1.0)

    fg_array = _normalize(foreground)
    bg_array = _normalize(background)

    fg_rgb, fg_alpha = _split_rgba(fg_array, 1.0)
    bg_rgb, bg_alpha = _split_rgba(bg_array, 0.0)

    if fg_rgb.shape[:2] != bg_rgb.shape[:2]:
        bg_image = Image.fromarray((bg_rgb * 255.0).astype(np.uint8), mode="RGB")
        bg_alpha_img = Image.fromarray((bg_alpha * 255.0).astype(np.uint8), mode="L")
        resized_rgb = bg_image.resize((fg_rgb.shape[1], fg_rgb.shape[0]), Image.Resampling.LANCZOS)
        resized_alpha = bg_alpha_img.resize((fg_rgb.shape[1], fg_rgb.shape[0]), Image.Resampling.LANCZOS)
        bg_rgb = np.asarray(resized_rgb, dtype=np.float32) / 255.0
        bg_alpha = np.asarray(resized_alpha, dtype=np.float32) / 255.0

    fg_rgb = np.clip(fg_rgb, 0.0, 1.0)
    bg_rgb = np.clip(bg_rgb, 0.0, 1.0)
    fg_alpha = np.clip(fg_alpha, 0.0, 1.0)
    bg_alpha = np.clip(bg_alpha, 0.0, 1.0)

    alpha_out = fg_alpha + bg_alpha * (1.0 - fg_alpha)
    alpha_out = np.clip(alpha_out, 0.0, 1.0)

    safe_alpha = np.where(alpha_out < 1e-8, 1.0, alpha_out)
    rgb_out = (
        fg_rgb * fg_alpha[..., None]
        + bg_rgb * bg_alpha[..., None] * (1.0 - fg_alpha[..., None])
    ) / safe_alpha[..., None]

    rgb_out = np.clip(rgb_out, 0.0, 1.0)

    return Image.fromarray(
        (np.dstack((rgb_out, alpha_out)) * 255.0).astype(np.uint8),
        mode="RGBA",
    )


def _smart_alpha_composition(foreground: Image.Image, background: Image.Image) -> Image.Image:
    """Backward compatible wrapper that uses the improved composition."""

    try:
        composed = _smart_alpha_composition_v5(foreground, background)
        cleaned = remove_alpha_halo_v5(composed)
        return Image.fromarray((cleaned * 255.0).astype(np.uint8), mode="RGBA")
    except Exception:  # pragma: no cover - fallback for environments without cv2/scipy
        return _smart_alpha_composition_improved(foreground, background)


def _smart_alpha_composition_v5(
    foreground: Image.Image | np.ndarray, background: Image.Image | np.ndarray
) -> np.ndarray:
    """Advanced anti-halo alpha composition returning a normalised RGBA array."""

    import numpy as _np

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency safeguard
        raise RuntimeError("OpenCV is required for v5 alpha composition") from exc

    fg = _np.asarray(foreground, dtype=_np.float32)
    bg = _np.asarray(background, dtype=_np.float32)

    if fg.ndim == 2:
        fg = _np.repeat(fg[..., None], 4, axis=2)
    if bg.ndim == 2:
        bg = _np.repeat(bg[..., None], 4, axis=2)

    if fg.max() > 1.0:
        fg /= 255.0
    if bg.max() > 1.0:
        bg /= 255.0

    if fg.shape[-1] < 4:
        alpha_channel = _np.ones(fg.shape[:2], dtype=_np.float32)
        fg = _np.dstack((fg[..., :3], alpha_channel))
    if bg.shape[-1] < 4:
        alpha_channel = _np.zeros(bg.shape[:2], dtype=_np.float32)
        bg = _np.dstack((bg[..., :3], alpha_channel))

    fg_rgb = _np.clip(fg[..., :3], 0.0, 1.0)
    fg_alpha = _np.clip(fg[..., 3], 0.0, 1.0)
    bg_rgb = _np.clip(bg[..., :3], 0.0, 1.0)
    bg_alpha = _np.clip(bg[..., 3], 0.0, 1.0)

    alpha_out = fg_alpha + bg_alpha * (1.0 - fg_alpha)
    alpha_safe = _np.where(alpha_out < 1e-8, 1.0, alpha_out)

    rgb_out = (
        fg_rgb * fg_alpha[..., None]
        + bg_rgb * bg_alpha[..., None] * (1.0 - fg_alpha[..., None])
    ) / alpha_safe[..., None]

    alpha_8bit = (_np.clip(alpha_out, 0.0, 1.0) * 255).astype(_np.uint8)
    edges_fine = cv2.Canny(alpha_8bit, 20, 60).astype(_np.float32) / 255.0
    edges_wide = cv2.Canny(alpha_8bit, 10, 30).astype(_np.float32) / 255.0
    edge_mask = _np.clip(edges_fine + edges_wide * 0.5, 0.0, 1.0)

    kernel = _np.ones((2, 2), _np.uint8)
    edge_mask_dilated = cv2.dilate(edge_mask, kernel, iterations=1)

    alpha_guided = cv2.GaussianBlur(alpha_out.astype(_np.float32), (3, 3), 0.8)
    blend_weights = edge_mask_dilated[..., None] * alpha_guided[..., None]

    rgb_blurred = cv2.bilateralFilter(rgb_out.astype(_np.float32), 5, 15, 15)
    rgb_corrected = rgb_out * (1.0 - blend_weights) + rgb_blurred * blend_weights

    near_zero_alpha = alpha_out < 0.01
    rgb_corrected[near_zero_alpha] = bg_rgb[near_zero_alpha]

    return _np.clip(_np.dstack((rgb_corrected, alpha_out)), 0.0, 1.0)


def remove_alpha_halo_v5(
    rgba_image: np.ndarray | Image.Image, dilation_iterations: int = 2
) -> np.ndarray:
    """Post-processing routine that removes residual halo artefacts."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency safeguard
        raise RuntimeError("OpenCV is required for halo removal") from exc

    if isinstance(rgba_image, Image.Image):
        img = np.asarray(rgba_image.convert("RGBA"), dtype=np.float32)
    else:
        img = np.asarray(rgba_image, dtype=np.float32)

    if img.max() > 1.0:
        img /= 255.0

    rgb, alpha = img[..., :3].copy(), np.clip(img[..., 3], 0.0, 1.0)
    alpha_8bit = (alpha * 255).astype(np.uint8)
    edges = cv2.Canny(alpha_8bit, 15, 45)

    kernel = np.ones((3, 3), np.uint8)
    halo_region = cv2.dilate(edges, kernel, iterations=dilation_iterations)
    halo_mask = halo_region.astype(np.float32) / 255.0

    if halo_mask.max() > 0.0:
        for channel in range(3):
            channel_blur = cv2.medianBlur((rgb[..., channel] * 255).astype(np.uint8), 3)
            rgb[..., channel] = np.where(
                halo_mask > 0,
                channel_blur.astype(np.float32) / 255.0,
                rgb[..., channel],
            )

    return np.clip(np.dstack((rgb, alpha)), 0.0, 1.0)


def _chromatic_pbr_integration(
    pbr_maps: Mapping[str, Image.Image],
    base_color: Image.Image | None,
) -> Tuple[Dict[str, Image.Image], bool]:
    """Re-inject base colour variation into grayscale PBR maps."""

    if base_color is None:
        return {name: img.copy() for name, img in pbr_maps.items()}, False

    base_rgb = np.asarray(base_color.convert("RGB"), dtype=np.float32) / 255.0
    if base_rgb.size == 0:
        return {name: img.copy() for name, img in pbr_maps.items()}, False

    colour_norm = np.linalg.norm(base_rgb, axis=2, keepdims=True)
    colour_direction = np.where(colour_norm > 1e-4, base_rgb / colour_norm, 1.0 / np.sqrt(3.0))
    chroma_strength = np.std(base_rgb, axis=2, keepdims=True)
    adjusted: Dict[str, Image.Image] = {}
    changed = False

    for name in ("metallic", "roughness", "specular"):
        image = pbr_maps.get(name)
        if image is None:
            continue
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4]
        channel_ptp = float(np.mean(np.ptp(rgb, axis=2)))
        if channel_ptp > 0.02:
            adjusted[name] = image.copy()
            continue

        luminance = rgb.mean(axis=2, keepdims=True)
        tint = 0.8 + 0.2 * chroma_strength
        tinted = np.clip(luminance * tint * colour_direction, 0.0, 1.0)
        # Preserve the average intensity to avoid shifting the physical response.
        original_mean = float(np.mean(luminance)) if luminance.size else 0.0
        tinted_mean = float(np.mean(tinted)) if tinted.size else 0.0
        if tinted_mean > 1e-6:
            scale = np.clip(original_mean / tinted_mean, 0.6, 1.4)
            tinted *= scale
        combined = np.dstack((np.clip(tinted, 0.0, 1.0), alpha))
        adjusted[name] = Image.fromarray((combined * 255.0).astype(np.uint8), mode="RGBA")
        changed = True

    updated = {name: img.copy() for name, img in pbr_maps.items()}
    updated.update(adjusted)
    return updated, changed


def _merge_layered_config(base: Mapping[str, object], overrides: Mapping[str, object] | None) -> Dict[str, object]:
    if not overrides:
        return dict(base)
    merged = dict(base)
    for key, value in overrides.items():
        if key == "composition_methods" and isinstance(value, Mapping):
            comp = dict(base.get("composition_methods", {}))
            comp.update(value)
            merged[key] = comp
        else:
            merged[key] = value
    if "composition_methods" not in merged:
        merged["composition_methods"] = dict(base.get("composition_methods", {}))
    return merged


class LayeredPBRCache:
    """Cache optimized for layered foreground/background PBR maps."""

    def __init__(self) -> None:
        self.foreground_cache: Dict[str, Dict[str, Image.Image]] = {}
        self.background_cache: Dict[str, Dict[str, Image.Image]] = {}
        self.composition_cache: Dict[tuple[str, str], Dict[str, Image.Image]] = {}

    def clear(self) -> None:
        self.foreground_cache.clear()
        self.background_cache.clear()
        self.composition_cache.clear()

    def get_foreground_pbr(self, image: Image.Image) -> Dict[str, Image.Image]:
        key = self._compute_hash(image)
        if key not in self.foreground_cache:
            self.foreground_cache[key] = self._generate_foreground_pbr(image)
        return {name: img.copy() for name, img in self.foreground_cache[key].items()}

    def get_background_pbr(self, image: Image.Image) -> Dict[str, Image.Image]:
        key = self._compute_hash(image)
        if key not in self.background_cache:
            self.background_cache[key] = self._generate_background_pbr(image)
        return {name: img.copy() for name, img in self.background_cache[key].items()}

    def set_composition(self, foreground: Image.Image, background: Image.Image, maps: Mapping[str, Image.Image]) -> None:
        key = (self._compute_hash(foreground), self._compute_hash(background))
        self.composition_cache[key] = {name: img.copy() for name, img in maps.items()}

    @staticmethod
    def _generate_foreground_pbr(image: Image.Image) -> Dict[str, Image.Image]:
        LOGGER.debug("Generating uncached foreground PBR maps")
        result = generate_physically_accurate_pbr_maps(image, None, {})
        maps = result.get("maps", {})
        return {name: img.copy() for name, img in maps.items()}

    @staticmethod
    def _generate_background_pbr(image: Image.Image) -> Dict[str, Image.Image]:
        LOGGER.debug("Generating uncached background PBR maps")
        result = generate_physically_accurate_pbr_maps(image, None, {})
        maps = result.get("maps", {})
        return {name: img.copy() for name, img in maps.items()}

    @staticmethod
    def _compute_hash(image: Image.Image) -> str:
        rgba = image.convert("RGBA")
        digest = hashlib.sha256()
        digest.update(str(rgba.size).encode("utf-8"))
        digest.update(rgba.tobytes())
        return digest.hexdigest()


class PBRBaseManager:
    """Helper that retrieves base PBR maps with optional caching."""

    def __init__(self, layer: str, *, cache: LayeredPBRCache | None = None) -> None:
        self.layer = layer
        self.cache = cache

    def get_base_pbr_maps(self, base_image: Image.Image, background: Image.Image | None) -> Dict[str, Image.Image]:
        if background is not None:
            result = generate_physically_accurate_pbr_maps(base_image, background, {})
            return {name: img.copy() for name, img in result.get("maps", {}).items()}
        if self.cache is not None:
            if self.layer == "foreground":
                return self.cache.get_foreground_pbr(base_image)
            if self.layer == "background":
                return self.cache.get_background_pbr(base_image)
        result = generate_physically_accurate_pbr_maps(base_image, background, {})
        return {name: img.copy() for name, img in result.get("maps", {}).items()}


class PBRLayerManager:
    """Generate and composite PBR maps for foreground and background layers."""

    def __init__(
        self,
        config: Mapping[str, object] | None = None,
        *,
        cache: LayeredPBRCache | None = None,
    ) -> None:
        self.config = _merge_layered_config(LAYERED_PBR_CONFIG, config)
        self.cache = cache if self.config.get("cache_layers_separately", True) else None
        self.foreground_manager = PBRBaseManager("foreground", cache=self.cache)
        self.background_manager = PBRBaseManager("background", cache=self.cache)

    def generate_layered_pbr_maps(
        self,
        foreground_image: Image.Image,
        background_image: Image.Image,
    ) -> Dict[str, Image.Image]:
        foreground_maps = self.foreground_manager.get_base_pbr_maps(foreground_image, None)
        background_maps = self.background_manager.get_base_pbr_maps(background_image, None)
        composited = self._composite_pbr_layers(
            background_image,
            foreground_image,
            background_maps,
            foreground_maps,
        )
        if self.cache is not None:
            self.cache.set_composition(foreground_image, background_image, composited)
        return composited

    def _composite_pbr_layers(
        self,
        background_image: Image.Image,
        foreground_image: Image.Image,
        background_maps: Mapping[str, Image.Image],
        foreground_maps: Mapping[str, Image.Image],
    ) -> Dict[str, Image.Image]:
        composited: Dict[str, Image.Image] = {}
        alpha = self._extract_alpha(foreground_image, foreground_maps)
        method_map = self.config.get("composition_methods", {})
        keys = set(background_maps.keys()) | set(foreground_maps.keys())
        for name in keys:
            bg_map = background_maps.get(name)
            fg_map = foreground_maps.get(name)
            if fg_map is None and bg_map is None:
                continue
            if fg_map is None:
                if bg_map is not None:
                    composited[name] = bg_map.copy()
                continue
            if bg_map is None:
                composited[name] = fg_map.copy()
                continue
            method = method_map.get(name, "alpha_blending")
            if method == "geometric_replacement":
                composited[name] = self._composite_geometric(bg_map, fg_map, alpha)
            elif method == "optical_composition":
                composited[name] = self._composite_optical(bg_map, fg_map, alpha)
            elif method == "alpha_blending":
                composited[name] = self._composite_surface(bg_map, fg_map, alpha)
            else:
                LOGGER.warning("Unknown composition method %s for map %s, defaulting to alpha blend", method, name)
                composited[name] = self._composite_surface(bg_map, fg_map, alpha)
        base_reference = composited.get("base_color") or foreground_maps.get("base_color")
        composited, chroma_adjusted = _chromatic_pbr_integration(composited, base_reference)
        if chroma_adjusted:
            LOGGER.info("Chromatic integration: PBR maps maintain physical color relationships")
        return composited

    @staticmethod
    def _extract_alpha(foreground_image: Image.Image, foreground_maps: Mapping[str, Image.Image]) -> Image.Image:
        if "opacity" in foreground_maps:
            return foreground_maps["opacity"].convert("L")
        if foreground_image.mode.endswith("A"):
            return foreground_image.split()[-1]
        return Image.new("L", foreground_image.size, 0)

    @staticmethod
    def _composite_geometric(bg_map: Image.Image, fg_map: Image.Image, alpha: Image.Image) -> Image.Image:
        bg = np.asarray(bg_map.convert("RGBA"), dtype=np.float32) / 255.0
        fg = np.asarray(fg_map.convert("RGBA"), dtype=np.float32) / 255.0
        alpha_arr = np.asarray(alpha, dtype=np.float32) / 255.0
        mask = alpha_arr[..., np.newaxis] > 0.5
        result = np.where(mask, fg, bg)
        result[..., 3] = np.clip(np.maximum(bg[..., 3], alpha_arr), 0.0, 1.0)
        return Image.fromarray(np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")

    @staticmethod
    def _composite_surface(bg_map: Image.Image, fg_map: Image.Image, alpha: Image.Image) -> Image.Image:
        composed = _smart_alpha_composition(fg_map, bg_map)
        if not _validate_halo_elimination(composed):
            LOGGER.warning("Detected residual haloing after smart composition; applying fallback blend")
            bg = np.asarray(bg_map.convert("RGBA"), dtype=np.float32) / 255.0
            fg = np.asarray(fg_map.convert("RGBA"), dtype=np.float32) / 255.0
            alpha_arr = np.asarray(alpha, dtype=np.float32) / 255.0
            alpha_arr = np.expand_dims(alpha_arr, axis=-1)
            result = bg * (1.0 - alpha_arr) + fg * alpha_arr
            result[..., 3] = np.clip(np.maximum(bg[..., 3], alpha_arr[..., 0]), 0.0, 1.0)
            composed = Image.fromarray(np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")
        return composed

    @staticmethod
    def _composite_optical(bg_map: Image.Image, fg_map: Image.Image, alpha: Image.Image) -> Image.Image:
        composed = _smart_alpha_composition(fg_map, bg_map)
        return composed


def execute_physically_correct_pbr_v5(
    input_foreground: Image.Image | np.ndarray,
    input_background: Image.Image | np.ndarray,
    material_class: str = "default",
) -> Dict[str, object]:
    """Pipeline orchestration that applies the v5 physical corrections."""

    if isinstance(input_foreground, Image.Image):
        foreground = input_foreground.convert("RGBA")
    else:
        foreground = Image.fromarray(np.asarray(input_foreground, dtype=np.uint8)).convert("RGBA")

    if isinstance(input_background, Image.Image):
        background = input_background.convert("RGBA")
    else:
        background = Image.fromarray(np.asarray(input_background, dtype=np.uint8)).convert("RGBA")

    try:
        composed_array = _smart_alpha_composition_v5(foreground, background)
        cleaned_array = remove_alpha_halo_v5(composed_array)
        composed_image = Image.fromarray((cleaned_array * 255.0).astype(np.uint8), mode="RGBA")
    except Exception:
        composed_image = _smart_alpha_composition_improved(foreground, background)

    base_image = composed_image.convert("RGB")
    pipeline_result = generate_physically_accurate_pbr_maps(base_image, None, {})
    analysis = pipeline_result.get("analysis")

    maps = dict(pipeline_result.get("maps", {}))
    metallic_candidate = maps.get("metallic")
    roughness_candidate = maps.get("roughness")

    maps["metallic"] = _generate_metallic_map_v5(
        base_image,
        roughness_candidate,
        analysis,
        material_class=material_class,
        candidate=metallic_candidate,
    )
    maps["roughness"] = _generate_roughness_map_v5(
        base_image,
        analysis,
        maps.get("metallic"),
        material_class,
    )
    maps["metallic"] = _generate_metallic_map_v5(
        base_image,
        maps.get("roughness"),
        analysis,
        material_class=material_class,
        candidate=maps.get("metallic"),
    )
    maps["emissive"] = _generate_emissive_map_v5(
        base_image,
        analysis,
        material_class=material_class,
    )
    maps["normal"] = _generate_normal_map_v5(base_image, analysis)

    maps = _detect_and_correct_flat_maps_v5(maps, base_image, analysis, material_class)

    validation = validate_pbr_coherence_v5(base_image, maps, material_class)
    quality_report = automated_quality_report_v5(validation)

    return {
        "final_image": base_image,
        "pbr_maps": maps,
        "validation": validation,
        "quality_report": quality_report,
        "pipeline_result": pipeline_result,
    }


__all__ = [
    "LAYERED_PBR_CONFIG",
    "LayeredPBRCache",
    "PBRBaseManager",
    "PBRLayerManager",
    "execute_physically_correct_pbr_v5",
]
