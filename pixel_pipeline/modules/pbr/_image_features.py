"""Utility helpers for rich image feature extraction used by the PBR pipeline."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


def to_rgb_alpha(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized RGB array and alpha mask from any PIL image."""

    rgba = image.convert("RGBA")
    array = np.asarray(rgba, dtype=np.float32)
    rgb = array[..., :3] / 255.0
    alpha = array[..., 3] / 255.0
    return rgb, alpha


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute linear luminance from normalized RGB values."""

    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def saturation(rgb: np.ndarray) -> np.ndarray:
    """Compute HSV-like saturation from normalized RGB values."""

    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc
    sat = np.zeros_like(maxc)
    valid = maxc > 1e-5
    sat[valid] = delta[valid] / maxc[valid]
    return sat


def hue(rgb: np.ndarray) -> np.ndarray:
    """Compute hue in the [0, 1] range for normalized RGB values."""

    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc
    hue = np.zeros_like(maxc)
    mask = delta > 1e-5
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    denom = np.where(mask, delta, 1.0)
    hr = ((g - b) / denom) % 6.0
    hg = ((b - r) / denom) + 2.0
    hb = ((r - g) / denom) + 4.0

    hue = np.where((maxc == r) & mask, hr, hue)
    hue = np.where((maxc == g) & mask, hg, hue)
    hue = np.where((maxc == b) & mask, hb, hue)
    hue = (hue / 6.0) % 1.0
    return hue


def edge_strength(channel: np.ndarray) -> np.ndarray:
    """Approximate edge strength from a single channel image in [0, 1]."""

    image = Image.fromarray(np.clip(channel * 255.0, 0, 255).astype(np.uint8), mode="L")
    edges = image.filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.MaxFilter(size=3))
    return np.asarray(edges, dtype=np.float32) / 255.0


def gaussian_blur(channel: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur for a single channel image in [0, 1]."""

    image = Image.fromarray(np.clip(channel * 255.0, 0, 255).astype(np.uint8), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def ensure_variation(channel: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Inject subtle variation if a map becomes perfectly uniform."""

    if np.std(channel) < epsilon:
        height, width = channel.shape
        yy, xx = np.mgrid[0:height, 0:width]
        gradient = (xx / max(1, width - 1) + yy / max(1, height - 1)) / 2.0
        channel = np.clip(channel + 0.02 * (gradient - 0.5), 0.0, 1.0)
    return channel


def normalize01(channel: np.ndarray) -> np.ndarray:
    """Normalize arbitrary data to the [0, 1] range."""

    min_val = float(channel.min())
    max_val = float(channel.max())
    if max_val - min_val < 1e-5:
        return np.zeros_like(channel)
    return (channel - min_val) / (max_val - min_val)


__all__ = [
    "edge_strength",
    "ensure_variation",
    "gaussian_blur",
    "hue",
    "luminance",
    "normalize01",
    "saturation",
    "to_rgb_alpha",
]
