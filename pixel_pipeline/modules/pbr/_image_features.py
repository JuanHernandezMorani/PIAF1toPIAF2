"""Utility helpers for rich image feature extraction used by the PBR pipeline."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from PIL import Image, ImageFilter

try:  # pragma: no cover - optional dependency
    from scipy import ndimage as _scipy_ndimage
except ImportError:  # pragma: no cover - graceful fallback
    _scipy_ndimage = None

from .physical_rgb import sanitize_rgba_image


def to_rgb_alpha(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized RGB array and alpha mask from any PIL image."""

    _, rgb, alpha = sanitize_rgba_image(image, return_arrays=True)
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


def _filter_with_edge_handling(
    array: np.ndarray,
    filter_func: Callable[[np.ndarray], np.ndarray],
    *,
    pad_width: int = 2,
    mode: str = "reflect",
    **kwargs,
) -> np.ndarray:
    """Apply ``filter_func`` on a reflect-padded copy of ``array``.

    The helper enforces consistent edge handling so convolution-like operations do
    not introduce dark seams around the borders of generated maps.
    """

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3:
        pad_spec = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
    else:
        pad_spec = tuple((pad_width, pad_width) for _ in range(arr.ndim))
    padded = np.pad(arr, pad_spec, mode=mode)
    filtered = filter_func(padded, **kwargs)
    slices = []
    for axis in range(arr.ndim):
        width = pad_spec[axis][0]
        if width == 0:
            slices.append(slice(None))
        else:
            slices.append(slice(width, -width))
    return filtered[tuple(slices)]


def edge_strength(channel: np.ndarray) -> np.ndarray:
    """Approximate edge strength from a single channel image in [0, 1]."""

    image = Image.fromarray(np.clip(channel * 255.0, 0, 255).astype(np.uint8), mode="L")
    edges = image.filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.MaxFilter(size=3))
    return np.asarray(edges, dtype=np.float32) / 255.0


def gaussian_blur(channel: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur for a single channel image in [0, 1] with robust edges."""

    array = np.asarray(channel, dtype=np.float32)
    if _scipy_ndimage is not None:
        return _scipy_ndimage.gaussian_filter(array, sigma=radius, mode="reflect")

    radius = max(float(radius), 0.0)
    pad = max(int(np.ceil(radius * 2.0)), 1)

    def _pil_blur(padded: np.ndarray, *, blur_radius: float) -> np.ndarray:
        image = Image.fromarray(
            np.clip(padded * 255.0, 0, 255).astype(np.uint8), mode="L"
        )
        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return np.asarray(blurred, dtype=np.float32) / 255.0

    return _filter_with_edge_handling(array, _pil_blur, pad_width=pad, blur_radius=radius)


def ensure_variation(channel: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Inject subtle variation if a map becomes perfectly uniform."""

    array = np.asarray(channel, dtype=np.float32)
    if array.size == 0:
        return array

    if float(np.std(array)) >= epsilon or array.ndim < 2:
        return array

    height, width = array.shape[:2]
    if height == 0 or width == 0:
        return array

    yy, xx = np.mgrid[0:height, 0:width]
    gradient = (xx / max(1, width - 1) + yy / max(1, height - 1)) / 2.0
    gradient = gradient.astype(np.float32)
    gradient -= float(np.mean(gradient))

    if array.ndim == 2:
        array = np.clip(array + 0.02 * gradient, 0.0, 1.0)
    else:
        array = np.clip(array + 0.02 * gradient[..., None], 0.0, 1.0)

    return array


def normalize01(channel: np.ndarray) -> np.ndarray:
    """Normalize arbitrary data to the [0, 1] range."""

    min_val = float(channel.min())
    max_val = float(channel.max())
    if max_val - min_val < 1e-5:
        return np.zeros_like(channel)
    return (channel - min_val) / (max_val - min_val)


__all__ = [
    "_filter_with_edge_handling",
    "edge_strength",
    "ensure_variation",
    "gaussian_blur",
    "hue",
    "luminance",
    "normalize01",
    "saturation",
    "to_rgb_alpha",
]
