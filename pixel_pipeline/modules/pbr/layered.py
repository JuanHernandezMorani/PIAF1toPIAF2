"""Layer-aware orchestration for generating and compositing PBR maps."""
from __future__ import annotations

from typing import Dict, Mapping

import hashlib
import logging

import numpy as np
from PIL import Image

from .pipeline import generate_physically_accurate_pbr_maps

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
        bg = np.asarray(bg_map.convert("RGBA"), dtype=np.float32) / 255.0
        fg = np.asarray(fg_map.convert("RGBA"), dtype=np.float32) / 255.0
        alpha_arr = np.asarray(alpha, dtype=np.float32) / 255.0
        alpha_arr = np.expand_dims(alpha_arr, axis=-1)
        result = bg * (1.0 - alpha_arr) + fg * alpha_arr
        result[..., 3] = np.clip(np.maximum(bg[..., 3], alpha_arr[..., 0]), 0.0, 1.0)
        return Image.fromarray(np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")

    @staticmethod
    def _composite_optical(bg_map: Image.Image, fg_map: Image.Image, alpha: Image.Image) -> Image.Image:
        bg = np.asarray(bg_map.convert("RGBA"), dtype=np.float32) / 255.0
        fg = np.asarray(fg_map.convert("RGBA"), dtype=np.float32) / 255.0
        result_rgb = np.minimum(bg[..., :3], fg[..., :3])
        alpha_arr = np.asarray(alpha, dtype=np.float32) / 255.0
        result_alpha = np.clip(np.minimum(bg[..., 3], alpha_arr), 0.0, 1.0)
        result = np.dstack((result_rgb, result_alpha))
        return Image.fromarray(np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")


__all__ = [
    "LAYERED_PBR_CONFIG",
    "LayeredPBRCache",
    "PBRBaseManager",
    "PBRLayerManager",
]
