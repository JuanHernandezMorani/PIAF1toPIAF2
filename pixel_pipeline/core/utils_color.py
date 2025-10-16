"""Color utility helpers used across the pipeline."""
from __future__ import annotations

import colorsys
import math
import random
from typing import List, Sequence, Tuple

from PIL import ImageColor

ColorTuple = Tuple[int, int, int]


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp *value* between *min_value* and *max_value*."""

    return max(min_value, min(max_value, value))


def color_distance(color_a: Sequence[int], color_b: Sequence[int]) -> float:
    """Return the Euclidean distance between two RGB colors."""

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(color_a[:3], color_b[:3])))


def mix_colors(color_a: Sequence[int], color_b: Sequence[int], ratio: float) -> ColorTuple:
    """Linearly interpolate between *color_a* and *color_b*."""

    ratio = clamp(ratio, 0.0, 1.0)
    return tuple(int(a + (b - a) * ratio) for a, b in zip(color_a[:3], color_b[:3]))  # type: ignore[return-value]


def add_color_noise(color: Sequence[int], intensity: float) -> ColorTuple:
    """Add random noise to *color* with the given *intensity*."""

    intensity = clamp(intensity, 0.0, 1.0)
    noisy = []
    for channel in color[:3]:
        delta = random.uniform(-255 * intensity, 255 * intensity)
        noisy.append(int(clamp(channel + delta, 0, 255)))
    return tuple(noisy)  # type: ignore[return-value]


def generate_high_variation_colors(base_color: Sequence[int], count: int, intensity: float) -> List[ColorTuple]:
    """Generate high variation colors based on *base_color*."""

    palette = []
    for idx in range(max(1, count)):
        amount = (idx + 1) / max(1, count)
        ratio = clamp(amount * intensity, 0.0, 1.0)
        noise = add_color_noise(base_color, ratio)
        palette.append(noise)
    return palette


def rgb_to_hsl(color: Sequence[int]) -> Tuple[float, float, float]:
    """Convert an RGB color to HSL values."""

    r, g, b = [c / 255.0 for c in color[:3]]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, s, l


def hsl_to_rgb(hsl: Sequence[float]) -> ColorTuple:
    """Convert an HSL tuple to an RGB color."""

    h, s, l = hsl
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return tuple(int(round(clamp(channel * 255.0, 0.0, 255.0))) for channel in (r, g, b))  # type: ignore[return-value]


def random_palette(seed: int | None = None, size: int = 8) -> List[ColorTuple]:
    """Generate a reproducible random color palette."""

    rnd = random.Random(seed)
    palette: List[ColorTuple] = []
    for _ in range(size):
        hue = rnd.random()
        sat = rnd.uniform(0.4, 0.9)
        light = rnd.uniform(0.3, 0.8)
        palette.append(hsl_to_rgb((hue, sat, light)))
    return palette


def parse_color(value: str | Sequence[int]) -> ColorTuple:
    """Parse arbitrary color input into an RGB tuple."""

    if isinstance(value, str):
        rgb = ImageColor.getrgb(value)
        return rgb[:3]  # type: ignore[return-value]
    return tuple(value[:3])  # type: ignore[return-value]
