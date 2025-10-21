"""Utilities for correcting dark halo artifacts using the known background."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as _scipy_ndimage
except ImportError:  # pragma: no cover - exercised when SciPy is absent
    _scipy_ndimage = None


def _ensure_float_array(array: np.ndarray) -> np.ndarray:
    """Return array as float32 in the range [0, 1]."""

    if array.dtype != np.float32 and array.dtype != np.float64:
        array = array.astype(np.float32)
    array = np.clip(array, 0.0, 1.0)
    return array


def _binary_dilation(image: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Perform binary dilation using SciPy when available, otherwise NumPy."""

    if _scipy_ndimage is not None:
        return _scipy_ndimage.binary_dilation(image, structure=structure)

    image = np.asarray(image, dtype=bool)
    structure = np.asarray(structure, dtype=bool)

    result = np.zeros_like(image, dtype=bool)

    structure_coords = np.argwhere(structure)
    if structure_coords.size == 0:
        return result

    origin_y = structure.shape[0] // 2
    origin_x = structure.shape[1] // 2

    height, width = image.shape

    for sy, sx in structure_coords:
        shift_y = sy - origin_y
        shift_x = sx - origin_x

        if shift_y >= 0:
            src_y_start = 0
            src_y_end = height - shift_y
            dst_y_start = shift_y
            dst_y_end = height
        else:
            src_y_start = -shift_y
            src_y_end = height
            dst_y_start = 0
            dst_y_end = height + shift_y

        if shift_x >= 0:
            src_x_start = 0
            src_x_end = width - shift_x
            dst_x_start = shift_x
            dst_x_end = width
        else:
            src_x_start = -shift_x
            src_x_end = width
            dst_x_start = 0
            dst_x_end = width + shift_x

        if src_y_end <= src_y_start or src_x_end <= src_x_start:
            continue

        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] |= image[
            src_y_start:src_y_end, src_x_start:src_x_end
        ]

    return result


def detect_background_edge_pixels(
    alpha: np.ndarray,
    rgb: np.ndarray,
    *,
    alpha_threshold: float = 0.05,
    black_threshold: float = 0.1,
) -> np.ndarray:
    """Return mask of semi-transparent, dark pixels adjacent to the background."""

    alpha = _ensure_float_array(alpha)
    rgb = _ensure_float_array(rgb)

    solid_foreground = alpha > 0.95
    real_background = alpha < alpha_threshold
    edge_alpha = (alpha >= alpha_threshold) & (alpha <= 0.9) & ~solid_foreground

    structure = np.ones((3, 3), dtype=bool)
    dilated_background = _binary_dilation(real_background, structure)
    edge_adjacent_to_bg = edge_alpha & dilated_background

    luminance = rgb.mean(axis=2)
    is_black = luminance < black_threshold

    return edge_adjacent_to_bg & is_black


def _collect_background_colour(
    rgb: np.ndarray,
    alpha: np.ndarray,
    y: int,
    x: int,
    radius: int,
) -> Optional[np.ndarray]:
    """Collect median colour from transparent neighbours around ``(y, x)``."""

    height, width = alpha.shape
    y_min = max(0, y - radius)
    y_max = min(height, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(width, x + radius + 1)

    local_alpha = alpha[y_min:y_max, x_min:x_max]
    local_rgb = rgb[y_min:y_max, x_min:x_max]

    mask = local_alpha < 0.01
    if not np.any(mask):
        return None

    colours = local_rgb[mask]
    if colours.size == 0:
        return None

    return np.median(colours, axis=0)


def extrapolate_from_background(
    rgb: np.ndarray,
    alpha: np.ndarray,
    edge_mask: np.ndarray,
    background_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Fill edge pixels with colours sourced from the background."""

    if not np.any(edge_mask):
        return rgb

    alpha = _ensure_float_array(alpha)
    rgb = _ensure_float_array(rgb)
    corrected = rgb.copy()

    height, width = alpha.shape
    has_external_bg = (
        background_rgb is not None and background_rgb.shape[:2] == (height, width)
    )

    if has_external_bg:
        background_rgb = _ensure_float_array(background_rgb)

    edge_positions = np.argwhere(edge_mask)
    for y, x in edge_positions:
        bg_color: Optional[np.ndarray]
        bg_color = None

        if has_external_bg:
            bg_color = background_rgb[y, x]
        if bg_color is None:
            bg_color = _collect_background_colour(rgb, alpha, y, x, radius=4)
        if bg_color is None:
            bg_color = _collect_background_colour(rgb, alpha, y, x, radius=8)
        if bg_color is None:
            continue

        corrected[y, x] = bg_color

    return corrected


def correct_edges_with_background(
    foreground: Image.Image,
    background: Image.Image | None = None,
) -> Image.Image:
    """Return a new foreground image with dark edge pixels recoloured."""

    if foreground.mode != "RGBA":
        return foreground

    fg_array = np.asarray(foreground, dtype=np.float32) / 255.0
    fg_rgb = fg_array[..., :3]
    fg_alpha = fg_array[..., 3]

    bg_rgb = None
    if background is not None:
        resized_bg = background.resize(foreground.size, Image.Resampling.LANCZOS)
        bg_rgb = np.asarray(resized_bg.convert("RGB"), dtype=np.float32) / 255.0

    edge_mask = detect_background_edge_pixels(fg_alpha, fg_rgb)
    if not np.any(edge_mask):
        return foreground

    corrected_rgb = extrapolate_from_background(fg_rgb, fg_alpha, edge_mask, bg_rgb)
    corrected_rgba = np.dstack((np.clip(corrected_rgb, 0.0, 1.0), fg_alpha[..., None]))
    return Image.fromarray((corrected_rgba * 255.0).astype(np.uint8), mode="RGBA")


__all__ = [
    "correct_edges_with_background",
    "detect_background_edge_pixels",
    "extrapolate_from_background",
]
