"""Layer-aware orchestration for generating and compositing PBR maps."""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import hashlib
import logging

import numpy as np
from PIL import Image

from .pipeline import generate_physically_accurate_pbr_maps
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

    def _to_unit_array(image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, Image.Image):
            array = np.asarray(image, dtype=np.float32)
        else:
            array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = array[..., None]
        if array.size == 0:
            return np.zeros((1, 1, 4), dtype=np.float32)
        if array.max() > 1.0 + 1e-6 or array.min() < -1e-6:
            array = np.clip(array, 0.0, 255.0)
            if array.max() > 1.0 + 1e-6:
                array = array / 255.0
        else:
            array = np.clip(array, 0.0, 1.0)
        return array.astype(np.float32, copy=False)

    def _ensure_rgba(array: np.ndarray) -> np.ndarray:
        if array.ndim != 3:
            raise ValueError("Input array must have three dimensions after preprocessing")
        channels = array.shape[2]
        if channels == 4:
            return array
        if channels == 3:
            alpha = np.ones(array.shape[:2], dtype=array.dtype)
            return np.dstack((array, alpha))
        if channels == 1:
            rgb = np.repeat(array, 3, axis=2)
            alpha = np.ones(array.shape[:2], dtype=array.dtype)
            return np.dstack((rgb, alpha))
        if channels > 4:
            return np.dstack((array[..., :3], np.ones(array.shape[:2], dtype=array.dtype)))
        raise ValueError(f"Unsupported channel count: {channels}")

    fg = _ensure_rgba(_to_unit_array(foreground))
    bg = _ensure_rgba(_to_unit_array(background))

    if fg.shape[:2] != bg.shape[:2]:
        bg_image = Image.fromarray((np.clip(bg, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")
        resized = bg_image.resize((fg.shape[1], fg.shape[0]), Image.Resampling.LANCZOS)
        bg = np.asarray(resized, dtype=np.float32) / 255.0

    fg_rgb = fg[..., :3]
    fg_alpha = fg[..., 3:4]
    bg_rgb = bg[..., :3]
    bg_alpha = bg[..., 3:4]

    premult_fg = fg_rgb * fg_alpha
    premult_bg = bg_rgb * bg_alpha
    combined_premult = premult_fg + premult_bg * (1.0 - fg_alpha)
    combined_alpha = fg_alpha + bg_alpha * (1.0 - fg_alpha)

    epsilon = 1e-8
    safe_alpha = np.maximum(combined_alpha, epsilon)
    result_rgb = combined_premult / safe_alpha

    low_alpha = fg_alpha < 0.05
    result_rgb = np.where(low_alpha, bg_rgb, result_rgb)
    result_alpha = np.where(low_alpha, bg_alpha, combined_alpha)

    halo_mask = (result_alpha > 0.05) & (np.linalg.norm(result_rgb, axis=2) < 1e-3)
    if np.any(halo_mask):
        result_rgb = np.where(halo_mask[..., None], bg_rgb, result_rgb)

    residual_mask = (result_alpha > 0.4) & (np.linalg.norm(result_rgb, axis=2) < 0.02)
    if np.any(residual_mask):
        padded = np.pad(result_rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
        local_avg = np.zeros_like(result_rgb)
        height, width = result_alpha.shape[:2]
        for dy in range(3):
            for dx in range(3):
                local_avg += padded[dy : dy + height, dx : dx + width]
        local_avg /= 9.0
        result_rgb = np.where(residual_mask[..., None], local_avg, result_rgb)

    composed = np.dstack((np.clip(result_rgb, 0.0, 1.0), np.clip(result_alpha, 0.0, 1.0)))
    return Image.fromarray((composed * 255.0).astype(np.uint8), mode="RGBA")


def _smart_alpha_composition(foreground: Image.Image, background: Image.Image) -> Image.Image:
    """Backward compatible wrapper that uses the improved composition."""

    return _smart_alpha_composition_improved(foreground, background)


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


__all__ = [
    "LAYERED_PBR_CONFIG",
    "LayeredPBRCache",
    "PBRBaseManager",
    "PBRLayerManager",
]
