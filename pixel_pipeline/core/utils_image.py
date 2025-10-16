"""Image utility helpers for color space conversions and transforms."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

_D65 = (95.047, 100.0, 108.883)
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)


def _srgb_channel_to_linear(channel: np.ndarray) -> np.ndarray:
    threshold = 0.04045
    return np.where(
        channel <= threshold,
        channel / 12.92,
        ((channel + 0.055) / 1.055) ** 2.4,
    )


def srgb_to_linear(image: Image.Image) -> Image.Image:
    """Convert an sRGB image to linear space."""

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    linear = _srgb_channel_to_linear(array)
    result = np.clip(linear * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


def _linear_channel_to_srgb(channel: np.ndarray) -> np.ndarray:
    return np.where(
        channel <= 0.0031308,
        channel * 12.92,
        1.055 * np.power(channel, 1 / 2.4) - 0.055,
    )


def linear_to_srgb(image: Image.Image) -> Image.Image:
    """Convert a linear RGB image back to sRGB."""

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    srgb = _linear_channel_to_srgb(array)
    result = np.clip(srgb * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


def _rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb, _RGB_TO_XYZ.T)


def rgb_to_lab(image: Image.Image) -> Image.Image:
    """Convert an sRGB :class:`~PIL.Image.Image` to CIE Lab."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    rgb_linear = _srgb_channel_to_linear(rgb)
    xyz = _rgb_to_xyz(rgb_linear)
    xyz /= np.array(_D65, dtype=np.float32)

    def f(t: np.ndarray) -> np.ndarray:
        delta = 6 / 29
        return np.where(
            t > delta**3,
            np.cbrt(t),
            t / (3 * delta**2) + 4 / 29,
        )

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    l = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    lab = np.stack([l, a, b], axis=-1)
    lab_normalized = np.clip(lab, [-16, -128, -128], [116, 127, 127])
    lab_image = ((lab_normalized - [-16, -128, -128]) / [132, 255, 255] * 255).astype(np.uint8)
    return Image.fromarray(lab_image, mode="RGB")


def _lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    fy = (l + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    def f_inv(t: np.ndarray) -> np.ndarray:
        delta = 6 / 29
        return np.where(
            t > delta,
            t**3,
            3 * delta**2 * (t - 4 / 29),
        )

    x = f_inv(fx) * _D65[0]
    y = f_inv(fy) * _D65[1]
    z = f_inv(fz) * _D65[2]
    return np.stack([x, y, z], axis=-1)


def lab_to_rgb(image: Image.Image) -> Image.Image:
    """Convert a Lab image produced by :func:`rgb_to_lab` back to sRGB."""

    lab = np.asarray(image.convert("RGB"), dtype=np.float32)
    lab_scaled = (lab / 255.0) * [132, 255, 255] + [-16, -128, -128]
    xyz = _lab_to_xyz(lab_scaled)
    rgb_linear = np.dot(xyz, _XYZ_TO_RGB.T)
    rgb = _linear_channel_to_srgb(rgb_linear)
    result = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


def normalize(image: Image.Image) -> Image.Image:
    """Normalize the luminance of *image* into the full range."""

    array = np.asarray(image.convert("L"), dtype=np.float32)
    min_val = float(array.min())
    max_val = float(array.max())
    if math.isclose(max_val, min_val):
        return image.convert("L").convert("RGBA")
    normalized = (array - min_val) / (max_val - min_val) * 255.0
    normalized_image = Image.fromarray(normalized.astype(np.uint8), mode="L")
    return normalized_image.convert("RGBA")


def resize_power_of_two(image: Image.Image) -> Image.Image:
    """Resize *image* to the next power-of-two dimensions without filtering."""

    width, height = image.size
    pow2_width = 1 << (width - 1).bit_length()
    pow2_height = 1 << (height - 1).bit_length()
    if pow2_width == width and pow2_height == height:
        return image.copy()
    return image.resize((pow2_width, pow2_height), resample=Image.NEAREST)
