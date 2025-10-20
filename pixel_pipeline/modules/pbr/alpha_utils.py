"""Helpers for deriving consistent alpha channels across generated assets."""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
from PIL import Image

from .analysis import AnalysisResult


def _alpha_bleed_rgba(image: Image.Image, iterations: int = 8, radius: int = 1) -> Image.Image:
    rgba = image.convert("RGBA")
    array = np.asarray(rgba, dtype=np.uint8).astype(np.float32)
    rgb = array[..., :3]
    alpha = array[..., 3]

    if iterations <= 0 or radius <= 0:
        return rgba

    color_valid = alpha > 1
    height, width = alpha.shape

    for _ in range(iterations):
        to_fill = (~color_valid) & (alpha <= 1)
        if not np.any(to_fill):
            break

        accum = np.zeros_like(rgb, dtype=np.float32)
        weight = np.zeros_like(alpha, dtype=np.float32)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue

                src_y_start = max(0, -dy)
                src_y_end = min(height, height - dy)
                src_x_start = max(0, -dx)
                src_x_end = min(width, width - dx)
                dst_y_start = max(0, dy)
                dst_y_end = min(height, height + dy)
                dst_x_start = max(0, dx)
                dst_x_end = min(width, width + dx)

                if src_y_start >= src_y_end or src_x_start >= src_x_end:
                    continue

                src_slice = (slice(src_y_start, src_y_end), slice(src_x_start, src_x_end))
                dst_slice = (slice(dst_y_start, dst_y_end), slice(dst_x_start, dst_x_end))

                src_valid = color_valid[src_slice]
                if not np.any(src_valid):
                    continue

                dst_fillable = to_fill[dst_slice]
                mask = src_valid & dst_fillable
                if not np.any(mask):
                    continue

                src_rgb = rgb[src_slice]
                accum_dst = accum[dst_slice]

                mask3 = mask[..., None]
                accum_dst[mask3] += src_rgb[mask3]
                weight_dst = weight[dst_slice]
                weight_dst[mask] += 1.0

        if not np.any(weight):
            break

        average = np.zeros_like(rgb, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            average = np.divide(accum, weight[..., None], out=average, where=weight[..., None] > 0)

        update_mask = (weight > 0) & to_fill
        if not np.any(update_mask):
            break

        rgb = np.where(update_mask[..., None], average, rgb)
        color_valid |= update_mask

    combined = np.dstack((np.clip(rgb, 0.0, 255.0), alpha[..., None]))
    return Image.fromarray(combined.astype(np.uint8), mode="RGBA")


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
        bleeded = _alpha_bleed_rgba(image)
        updated[name] = apply_alpha(bleeded, alpha)
    return updated


__all__ = ["apply_alpha", "apply_alpha_to_maps", "derive_alpha_map"]
