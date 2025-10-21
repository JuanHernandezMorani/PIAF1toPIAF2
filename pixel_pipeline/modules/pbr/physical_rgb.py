"""Utilities to convert arbitrary sprites into physically valid RGB data."""
from __future__ import annotations

from collections import deque
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


_EIGHT_CONNECTED_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def _nearest_colour_propagation(rgb: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Fill transparent pixels with the colour of the nearest opaque neighbour."""

    height, width, _ = rgb.shape

    if not np.any(~valid_mask):
        return rgb

    filled = rgb.copy()
    distances = np.full(valid_mask.shape, np.iinfo(np.int32).max, dtype=np.int32)

    coords = np.argwhere(valid_mask)
    if coords.size == 0:
        fallback = np.median(rgb.reshape(-1, 3), axis=0) if rgb.size else np.zeros(3, dtype=np.float32)
        return np.broadcast_to(fallback, rgb.shape).copy()

    queue: deque[Tuple[int, int]] = deque()
    for y, x in coords:
        y_i = int(y)
        x_i = int(x)
        distances[y_i, x_i] = 0
        queue.append((y_i, x_i))

    offsets: Iterable[Tuple[int, int]]
    offsets = _EIGHT_CONNECTED_OFFSETS

    while queue:
        y, x = queue.popleft()
        next_dist = distances[y, x] + 1
        colour = filled[y, x]
        for dy, dx in offsets:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if distances[ny, nx] <= next_dist:
                continue
            distances[ny, nx] = next_dist
            filled[ny, nx] = colour
            queue.append((ny, nx))

    padded = np.pad(filled, ((1, 1), (1, 1), (0, 0)), mode="edge")
    local_avg = np.zeros_like(filled)
    for dy in range(3):
        for dx in range(3):
            local_avg += padded[dy : dy + height, dx : dx + width]
    local_avg /= 9.0
    filled = np.where(valid_mask[..., None], filled, local_avg)

    return filled


def sanitize_rgba_image(
    image: Image.Image,
    *,
    return_arrays: bool = False,
    alpha_threshold: float = 1e-3,
) -> tuple[Image.Image, np.ndarray, np.ndarray] | Image.Image:
    """Return a new RGBA image with RGB data propagated into transparent regions."""

    rgba = image.convert("RGBA")
    array = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb = array[..., :3]
    alpha = array[..., 3]

    valid_mask = alpha > alpha_threshold
    if np.any(~valid_mask):
        inactive = rgb[~valid_mask]
        if inactive.size and np.all(np.linalg.norm(inactive, axis=1) > 1e-3):
            propagated_rgb = rgb
        else:
            propagated_rgb = _nearest_colour_propagation(rgb, valid_mask)
    else:
        propagated_rgb = rgb

    combined = np.dstack((np.clip(propagated_rgb, 0.0, 1.0), alpha[..., None]))
    sanitized = Image.fromarray(np.clip(combined * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")

    if return_arrays:
        return sanitized, propagated_rgb, alpha
    return sanitized


__all__ = ["sanitize_rgba_image"]

