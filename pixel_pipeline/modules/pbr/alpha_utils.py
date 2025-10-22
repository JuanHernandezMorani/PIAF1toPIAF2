"""Helpers for deriving consistent alpha channels across generated assets."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import logging

import numpy as np
from PIL import Image

from .analysis import AnalysisResult

LOGGER = logging.getLogger("pixel_pipeline.pbr.alpha_utils")

try:  # pragma: no cover - Pillow compatibility shim
    _RESAMPLING = Image.Resampling
except AttributeError:  # pragma: no cover - older Pillow
    _RESAMPLING = Image

LANCZOS = getattr(_RESAMPLING, "LANCZOS", Image.BICUBIC)


def _build_simple_distance_kernel(radius: int) -> np.ndarray:
    """Return a simple inverse-distance kernel normalised to 1."""

    radius = max(int(radius), 1)
    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            if x == radius and y == radius:
                continue
            distance = np.hypot(float(x - radius), float(y - radius))
            kernel[y, x] = 1.0 / (1.0 + distance)
    total = float(kernel.sum())
    if total > 0.0:
        kernel /= total
    return kernel


def _enhanced_alpha_bleed_rgba_v3_improved(
    image: Image.Image,
    iterations: int = 12,
    radius: int = 2,
) -> Tuple[Image.Image, bool]:
    """Combine V3 robustness with V1 precision for alpha bleeding."""

    if iterations <= 0 or radius <= 0:
        rgba = image.convert("RGBA")
        return rgba, False

    rgba = image.convert("RGBA")
    array = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb = array[..., :3]
    alpha = array[..., 3]

    colour_energy = np.linalg.norm(rgb, axis=2)
    valid_colour = (alpha > 1e-3) & (colour_energy > 1e-4)
    if not np.any((alpha > 1e-3) & (~valid_colour)):
        return rgba, False

    kernel = _build_simple_distance_kernel(radius)
    offsets: Sequence[Tuple[int, int, float]] = []
    size = kernel.shape[0]
    centre = size // 2
    for y in range(size):
        for x in range(size):
            weight = float(kernel[y, x])
            if weight <= 0.0:
                continue
            dy = y - centre
            dx = x - centre
            if dy == 0 and dx == 0:
                continue
            offsets.append((dy, dx, weight))

    height, width = alpha.shape
    changed = False

    for _ in range(iterations):
        pending = (alpha > 1e-3) & (~valid_colour)
        if not np.any(pending):
            break

        accum = np.zeros_like(rgb)
        weight_accum = np.zeros_like(alpha)

        for dy, dx, weight in offsets:
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

            src_mask = valid_colour[src_slice]
            if not np.any(src_mask):
                continue

            dst_mask = pending[dst_slice]
            if not np.any(dst_mask):
                continue

            influence = alpha[src_slice] * weight
            colour = rgb[src_slice] * influence[..., None]

            accum[dst_slice] += np.where(src_mask[..., None], colour, 0.0)
            weight_accum[dst_slice] += np.where(src_mask, influence, 0.0)

        if not np.any(weight_accum):
            break

        with np.errstate(divide="ignore", invalid="ignore"):
            averaged = np.divide(
                accum,
                weight_accum[..., None],
                out=np.zeros_like(rgb),
                where=weight_accum[..., None] > 0,
            )

        update_mask = (weight_accum > 0) & pending
        if not np.any(update_mask):
            break

        rgb[update_mask] = averaged[update_mask]
        valid_colour |= update_mask
        changed = True

    if changed:
        luminance = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
        dark_edges = (alpha > 0.1) & (alpha < 0.98) & (luminance < 0.03)
        if np.any(dark_edges):
            padded = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
            local = np.zeros_like(rgb)
            for dy in range(3):
                for dx in range(3):
                    local += padded[dy : dy + height, dx : dx + width]
            local /= 9.0
            rgb = np.where(dark_edges[..., None], local, rgb)

    combined = np.dstack((np.clip(rgb, 0.0, 1.0), alpha[..., None]))
    result = Image.fromarray(np.clip(combined * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")
    return result, changed


def _enhanced_alpha_bleed_rgba(
    image: Image.Image,
    iterations: int = 12,
    radius: int = 2,
) -> Tuple[Image.Image, bool]:
    """Backward-compatible wrapper around the improved bleeding routine."""

    return _enhanced_alpha_bleed_rgba_v3_improved(image, iterations=iterations, radius=radius)


def _alpha_bleed_rgba(image: Image.Image, iterations: int = 8, radius: int = 1) -> Image.Image:
    """Backward-compatible wrapper around :func:`_enhanced_alpha_bleed_rgba`."""

    result, _ = _enhanced_alpha_bleed_rgba(image, iterations=max(iterations, 1), radius=max(radius, 1))
    return result


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
                Image.fromarray((chosen * 255).astype(np.uint8), mode="L").resize((width, height), LANCZOS),
                dtype=np.float32,
            )
            chosen /= 255.0
        chosen = np.clip(chosen, 0.0, 1.0)

    coherence = _compute_coherence(analysis)
    if coherence is not None:
        if coherence.shape != chosen.shape:
            coherence = np.asarray(
                Image.fromarray((coherence * 255).astype(np.uint8), mode="L").resize((width, height), LANCZOS),
                dtype=np.float32,
            )
            coherence /= 255.0
        chosen = np.clip(chosen * coherence, 0.0, 1.0)

    return chosen


def apply_alpha(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha_array = np.asarray(alpha, dtype=np.float32)
    if alpha_array.shape != (rgba.height, rgba.width):
        alpha_img = Image.fromarray((np.clip(alpha_array, 0.0, 1.0) * 255).astype(np.uint8), mode="L").resize(
            rgba.size, LANCZOS
        )
        alpha_array = np.asarray(alpha_img, dtype=np.float32) / 255.0
    else:
        alpha_array = np.clip(alpha_array, 0.0, 1.0)
        alpha_img = Image.fromarray((alpha_array * 255).astype(np.uint8), mode="L")

    alpha_binary = (alpha_array > 0.1).astype(np.uint8) * 255
    alpha_mask = Image.fromarray(alpha_binary, mode="L")
    transparent_bg = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    transparent_bg.paste(rgba.convert("RGB"), (0, 0), alpha_mask)
    return transparent_bg


def apply_alpha_to_maps(maps: Mapping[str, Image.Image], alpha: np.ndarray) -> Dict[str, Image.Image]:
    updated: Dict[str, Image.Image] = {}
    bleed_applied = False
    for name, image in maps.items():
        bleeded, changed = _enhanced_alpha_bleed_rgba(image)
        bleed_applied = bleed_applied or changed
        updated[name] = apply_alpha(bleeded, alpha)
    if bleed_applied:
        LOGGER.info("Enhanced alpha bleeding applied: eliminated black borders")
    return updated


__all__ = ["apply_alpha", "apply_alpha_to_maps", "derive_alpha_map"]
