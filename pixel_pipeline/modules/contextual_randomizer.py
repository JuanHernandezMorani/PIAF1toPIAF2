"""Contextual randomization utilities for recolor pipeline."""
from __future__ import annotations

import contextlib
import random
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore


_PIXEL_VARIATION_CALLBACK: Optional[Callable[[Image.Image], Image.Image]] = None


@contextlib.contextmanager
def pixel_variation_callback(callback: Callable[[Image.Image], Image.Image]) -> Iterator[None]:
    """Temporarily set the pixel-variation callback used by contextual blending."""

    global _PIXEL_VARIATION_CALLBACK
    previous = _PIXEL_VARIATION_CALLBACK
    _PIXEL_VARIATION_CALLBACK = callback
    try:
        yield
    finally:
        _PIXEL_VARIATION_CALLBACK = previous


def _ensure_rng(rng: random.Random | np.random.Generator) -> np.random.Generator:
    """Normalise RNG input to a numpy Generator."""

    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, random.Random):
        seed = rng.randint(0, 2**32 - 1)
        return np.random.default_rng(seed)
    raise TypeError(f"Unsupported RNG type: {type(rng)!r}")


def _normalise_noise(noise: np.ndarray) -> np.ndarray:
    """Normalise noise array to [0, 1]."""

    min_val = float(noise.min())
    max_val = float(noise.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(noise, dtype=np.float32)
    return ((noise - min_val) / (max_val - min_val)).astype(np.float32)


def generate_multiscale_noise(
    width: int,
    height: int,
    rng: random.Random | np.random.Generator,
    scales: Sequence[int] = (4, 8, 16),
    weights: Sequence[float] = (0.6, 0.3, 0.1),
) -> np.ndarray:
    """Create a fractal noise map in range [0, 1] using gaussian components."""

    if len(scales) != len(weights):
        raise ValueError("`scales` and `weights` must have the same length")
    if width <= 0 or height <= 0:
        return np.zeros((height, width), dtype=np.float32)

    np_rng = _ensure_rng(rng)
    total_weight = float(sum(weights))
    if total_weight <= 0:
        return np.zeros((height, width), dtype=np.float32)

    accumulator = np.zeros((height, width), dtype=np.float32)
    for scale, weight in zip(scales, weights):
        if weight <= 0:
            continue
        coarse_h = max(1, height // max(scale, 1))
        coarse_w = max(1, width // max(scale, 1))
        base = np_rng.normal(0.0, 1.0, (coarse_h, coarse_w)).astype(np.float32)
        if cv2 is not None:
            resized = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
            sigma = max(scale / 2.0, 0.1)
            resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:
            pil = Image.fromarray(base, mode="F")
            pil = pil.resize((width, height), resample=Image.BICUBIC)
            pil = pil.filter(ImageFilter.GaussianBlur(radius=max(scale / 2.0, 0.1)))
            resized = np.asarray(pil, dtype=np.float32)
        accumulator += resized * (weight / total_weight)

    return _normalise_noise(accumulator)


def apply_structural_variation(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply contextual structural noise and subtle warping while preserving alpha."""

    rgba = image.convert("RGBA")
    width, height = rgba.size
    if width == 0 or height == 0:
        return rgba

    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    rgb = np.asarray(rgba.convert("RGB"), dtype=np.float32) / 255.0
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    if cv2 is not None:
        sobelx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    else:
        sobelx = np.gradient(luminance, axis=1)
        sobely = np.gradient(luminance, axis=0)
    edge_map = np.sqrt(sobelx**2 + sobely**2)
    if edge_map.size:
        edge_map /= max(edge_map.max(), 1e-6)
    edge_map = edge_map.astype(np.float32)

    # Generate smooth displacement fields influenced by edges
    displacement_base = generate_multiscale_noise(width, height, rng, scales=(4, 8, 16), weights=(0.5, 0.3, 0.2))
    displacement_detail = generate_multiscale_noise(width, height, rng, scales=(2, 4, 8), weights=(0.4, 0.4, 0.2))
    strength = 1.5
    multiplier = 0.35 + edge_map * 0.65
    offset_x = (displacement_base - 0.5) * strength * multiplier
    offset_y = (displacement_detail - 0.5) * strength * multiplier

    grid_y, grid_x = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    map_x = np.clip(grid_x + offset_x, 0, width - 1).astype(np.float32)
    map_y = np.clip(grid_y + offset_y, 0, height - 1).astype(np.float32)

    src_rgb = (rgb * 255.0).astype(np.uint8)
    if cv2 is not None:
        bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
        warped_bgr = cv2.remap(bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    else:
        sample_y = np.rint(map_y).astype(int)
        sample_x = np.rint(map_x).astype(int)
        warped_rgb = src_rgb[sample_y, sample_x]

    warped_rgb = warped_rgb.astype(np.float32) / 255.0

    shading_noise = generate_multiscale_noise(width, height, rng, scales=(6, 12, 24), weights=(0.5, 0.3, 0.2))
    shading = 1.0 + (shading_noise - 0.5) * 0.25
    edge_enhance = 1.0 + edge_map * 0.12
    shading = np.clip(shading * edge_enhance, 0.7, 1.3)

    coloured = warped_rgb * shading[..., None]
    coloured = np.clip(coloured, 0.0, 1.0)

    final_rgba = np.dstack((coloured * 255.0, alpha.astype(np.float32)))
    return Image.fromarray(final_rgba.astype(np.uint8), mode="RGBA")


def integrate_contextual_variation(image: Image.Image, rng: random.Random) -> Image.Image:
    """Run structural and pixel-level variation steps sequentially."""

    if _PIXEL_VARIATION_CALLBACK is None:
        raise RuntimeError("Pixel variation callback is not configured")
    structured = apply_structural_variation(image, rng)
    return _PIXEL_VARIATION_CALLBACK(structured)
