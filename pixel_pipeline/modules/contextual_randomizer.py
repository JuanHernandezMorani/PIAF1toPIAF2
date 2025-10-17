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


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 1] to linear RGB."""

    threshold = 0.04045
    low = rgb <= threshold
    high = ~low
    result = np.empty_like(rgb, dtype=np.float32)
    result[low] = rgb[low] / 12.92
    result[high] = np.power((rgb[high] + 0.055) / 1.055, 2.4)
    return result


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB values to sRGB in [0, 1]."""

    threshold = 0.0031308
    low = rgb <= threshold
    high = ~low
    result = np.empty_like(rgb, dtype=np.float32)
    result[low] = rgb[low] * 12.92
    result[high] = 1.055 * np.power(rgb[high], 1.0 / 2.4) - 0.055
    return result


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB array in [0, 1] to CIE LAB."""

    linear = _srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    matrix = np.array(
        (
            (0.4124564, 0.3575761, 0.1804375),
            (0.2126729, 0.7151522, 0.0721750),
            (0.0193339, 0.1191920, 0.9503041),
        ),
        dtype=np.float32,
    )
    xyz = linear @ matrix.T

    # Reference white for D65
    ref_white = np.array((0.95047, 1.0, 1.08883), dtype=np.float32)
    xyz_scaled = xyz / ref_white

    delta = 6.0 / 29.0

    def _f(t: np.ndarray) -> np.ndarray:
        cube = np.cbrt(t)
        linear_mask = t <= delta**3
        result = cube
        if np.any(linear_mask):
            result = result.astype(np.float32)
            result[linear_mask] = (t[linear_mask] / (3.0 * delta**2)) + (4.0 / 29.0)
        return result.astype(np.float32)

    fx = _f(xyz_scaled[..., 0])
    fy = _f(xyz_scaled[..., 1])
    fz = _f(xyz_scaled[..., 2])

    l = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    lab = np.stack((l, a, b), axis=2)
    return lab.astype(np.float32)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE LAB array back to sRGB in [0, 1]."""

    lab = lab.astype(np.float32)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0

    delta = 6.0 / 29.0

    def _f_inv(t: np.ndarray) -> np.ndarray:
        result = np.power(t, 3)
        linear_mask = t <= delta
        if np.any(linear_mask):
            result = result.astype(np.float32)
            result[linear_mask] = 3.0 * delta**2 * (t[linear_mask] - 4.0 / 29.0)
        return result.astype(np.float32)

    x = _f_inv(fx)
    y = _f_inv(fy)
    z = _f_inv(fz)

    ref_white = np.array((0.95047, 1.0, 1.08883), dtype=np.float32)
    xyz = np.stack((x, y, z), axis=2) * ref_white

    matrix = np.array(
        (
            (3.2404542, -1.5371385, -0.4985314),
            (-0.9692660, 1.8760108, 0.0415560),
            (0.0556434, -0.2040259, 1.0572252),
        ),
        dtype=np.float32,
    )
    linear_rgb = xyz @ matrix.T
    return np.clip(_linear_to_srgb(linear_rgb), 0.0, 1.0)


def _rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB array in [0, 1] to YUV."""

    matrix = np.array(
        (
            (0.299, 0.587, 0.114),
            (-0.14713, -0.28886, 0.436),
            (0.615, -0.51499, -0.10001),
        ),
        dtype=np.float32,
    )
    return rgb @ matrix.T


def _yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    """Convert YUV array back to sRGB in [0, 1]."""

    matrix = np.array(
        (
            (1.0, 0.0, 1.13983),
            (1.0, -0.39465, -0.58060),
            (1.0, 2.03211, 0.0),
        ),
        dtype=np.float32,
    )
    rgb = yuv @ matrix.T
    return np.clip(rgb, 0.0, 1.0)


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


def apply_global_recolor(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply an adaptive recolor step to break dominant color dependency."""

    rgba = image.convert("RGBA")
    width, height = rgba.size
    if width == 0 or height == 0:
        return rgba

    alpha_channel = rgba.getchannel("A")
    rgb_image = rgba.convert("RGB")
    arr = np.asarray(rgb_image, dtype=np.float32) / 255.0
    if arr.size == 0:
        recolored = rgba.copy()
        recolored.putalpha(alpha_channel)
        return recolored

    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    mode = rng.choice(["HSL", "LAB", "YUV"])

    if mode == "HSL":
        import colorsys

        hls = np.empty_like(arr)
        for y in range(height):
            for x in range(width):
                r, g, b = arr[y, x]
                h, l, s = colorsys.rgb_to_hls(float(r), float(g), float(b))
                hls[y, x] = (h, l, s)
        hls[..., 0] = (hls[..., 0] + rng.uniform(-0.6, 0.6)) % 1.0
        hls[..., 1] = np.clip(hls[..., 1] * rng.uniform(0.5, 1.8), 0.0, 1.0)
        hls[..., 2] = np.clip(hls[..., 2] * rng.uniform(0.4, 1.6), 0.0, 1.0)
        for y in range(height):
            for x in range(width):
                arr[y, x] = colorsys.hls_to_rgb(*hls[y, x])
    elif mode == "LAB":
        lab = _rgb_to_lab(arr)
        lab[..., 1:] += np_rng.normal(0.0, 20.0, lab[..., 1:].shape).astype(np.float32)
        lab[..., 0] *= rng.uniform(0.7, 1.4)
        arr = np.clip(_lab_to_rgb(lab), 0.0, 1.0)
    else:  # YUV
        yuv = _rgb_to_yuv(arr)
        yuv[..., 1:] += np_rng.normal(0.0, 0.1, yuv[..., 1:].shape).astype(np.float32)
        yuv[..., 0] = np.clip(yuv[..., 0] * rng.uniform(0.6, 1.6), 0.0, 1.0)
        arr = np.clip(_yuv_to_rgb(yuv), 0.0, 1.0)

    tint = np.array([rng.uniform(0.6, 1.4) for _ in range(3)], dtype=np.float32)
    arr = np.clip(arr * tint, 0.0, 1.0)

    # Apply smooth fractal noise per channel
    noise_strength = rng.uniform(0.08, 0.22)
    noise_fields = []
    for _ in range(3):
        channel_rng = random.Random(rng.randint(0, 2**32 - 1))
        noise_map = generate_multiscale_noise(
            width,
            height,
            channel_rng,
            scales=(4, 8, 16, 32),
            weights=(0.4, 0.3, 0.2, 0.1),
        )
        noise_fields.append(noise_map)
    fractal_noise = np.stack(noise_fields, axis=2).astype(np.float32)
    arr = np.clip(arr + (fractal_noise - 0.5) * noise_strength, 0.0, 1.0)

    micro_noise_strength = rng.uniform(0.01, 0.05)
    micro_noise = np_rng.normal(0.0, micro_noise_strength, arr.shape).astype(np.float32)
    arr = np.clip(arr + micro_noise, 0.0, 1.0)

    recolored_rgb = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
    recolored = recolored_rgb.convert("RGBA")
    recolored.putalpha(alpha_channel)
    return recolored
