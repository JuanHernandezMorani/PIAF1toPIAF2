"""Helpers for deriving consistent alpha channels across generated assets."""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
from PIL import Image

from .analysis import AnalysisResult


def _as_float_array(image: Image.Image | None) -> np.ndarray | None:
    if image is None:
        return None
    array = np.asarray(image.convert("L"), dtype=np.float32)
    if array.size == 0:
        return None
    return np.clip(array / 255.0, 0.0, 1.0)


def _alpha_from_image(image: Image.Image | None) -> np.ndarray | None:
    if image is None or "A" not in image.getbands():
        return None
    alpha = np.asarray(image.split()[-1], dtype=np.float32) / 255.0
    if alpha.size == 0:
        return None
    return np.clip(alpha, 0.0, 1.0)


def _has_variation(array: np.ndarray | None, threshold: float = 1e-3) -> bool:
    if array is None:
        return False
    return float(np.std(array)) > threshold


def _compute_coherence(analysis: AnalysisResult | None) -> np.ndarray | None:
    if analysis is None:
        return None
    coherence = np.array(analysis.mask, dtype=np.float32)
    false_metal = getattr(analysis.material_analysis, "false_metal_risks", None)
    if isinstance(false_metal, np.ndarray) and false_metal.shape == coherence.shape:
        coherence *= np.clip(1.0 - false_metal, 0.0, 1.0)
    edge_map = analysis.geometric_features.get("edge_map") if analysis.geometric_features else None
    if isinstance(edge_map, np.ndarray) and edge_map.shape == coherence.shape:
        coherence = np.clip(coherence * (0.85 + 0.15 * edge_map), 0.0, 1.0)
    return np.clip(coherence, 0.0, 1.0)


def derive_alpha_map(
    base_image: Image.Image,
    maps: Mapping[str, Image.Image],
    analysis: AnalysisResult | None = None,
) -> np.ndarray:
    width, height = base_image.size
    preferred: list[np.ndarray | None] = []

    if analysis is not None and getattr(analysis.alpha, "size", 0):
        preferred.append(np.clip(analysis.alpha.astype(np.float32), 0.0, 1.0))
    else:
        preferred.append(_alpha_from_image(base_image))

    preferred.append(_as_float_array(maps.get("opacity")))
    preferred.append(_as_float_array(maps.get("transmission")))
    preferred.append(_as_float_array(maps.get("porosity")))

    chosen: np.ndarray | None = None
    for candidate in preferred:
        if candidate is None:
            continue
        if chosen is None:
            chosen = candidate
            continue
        if not _has_variation(chosen) and _has_variation(candidate):
            chosen = candidate

    if chosen is None:
        chosen = np.ones((height, width), dtype=np.float32)
    else:
        if chosen.shape != (height, width):
            chosen = np.asarray(
                Image.fromarray((chosen * 255).astype(np.uint8), mode="L").resize((width, height), Image.BILINEAR),
                dtype=np.float32,
            )
            chosen /= 255.0
        chosen = np.clip(chosen, 0.0, 1.0)

    coherence = _compute_coherence(analysis)
    if coherence is not None:
        if coherence.shape != chosen.shape:
            coherence = np.asarray(
                Image.fromarray((coherence * 255).astype(np.uint8), mode="L").resize((width, height), Image.BILINEAR),
                dtype=np.float32,
            )
            coherence /= 255.0
        chosen = np.clip(chosen * coherence, 0.0, 1.0)

    return chosen


def apply_alpha(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    rgba = image.convert("RGBA")
    if alpha.shape != (rgba.height, rgba.width):
        alpha_img = Image.fromarray((np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8), mode="L").resize(
            rgba.size, Image.BILINEAR
        )
    else:
        alpha_img = Image.fromarray((np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    result = rgba.copy()
    result.putalpha(alpha_img)
    return result


def apply_alpha_to_maps(maps: Mapping[str, Image.Image], alpha: np.ndarray) -> Dict[str, Image.Image]:
    updated: Dict[str, Image.Image] = {}
    for name, image in maps.items():
        updated[name] = apply_alpha(image, alpha)
    return updated


__all__ = ["apply_alpha", "apply_alpha_to_maps", "derive_alpha_map"]
