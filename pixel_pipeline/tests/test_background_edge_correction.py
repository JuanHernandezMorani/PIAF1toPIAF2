"""Tests for background-aware edge correction utilities."""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np
from PIL import Image

from pixel_pipeline.modules.pbr.background_edge_correction import (
    correct_edges_with_background,
)


def _create_test_foreground(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    centre = size // 2
    radius = size // 4

    for y in range(size):
        for x in range(size):
            distance = np.hypot(x - centre, y - centre)
            if distance <= radius:
                image.putpixel((x, y), (255, 120, 80, 255))
            elif distance <= radius + 2:
                alpha = int(255 * max(0.0, 1.0 - (distance - radius) / 2.0) * 0.5)
                image.putpixel((x, y), (0, 0, 0, alpha))
    return image


def test_preserves_foreground_shape() -> None:
    size = 32
    foreground = _create_test_foreground(size)

    background = Image.new("RGB", (size, size), (80, 140, 200))
    for y in range(size):
        for x in range(size):
            if (x + y) % 4 == 0:
                background.putpixel((x, y), (96, 160, 224))

    corrected = correct_edges_with_background(foreground, background)

    original_alpha = np.asarray(foreground.split()[-1])
    corrected_alpha = np.asarray(corrected.split()[-1])
    np.testing.assert_array_equal(original_alpha, corrected_alpha)

    original_rgb = np.asarray(foreground.convert("RGB"))
    corrected_rgb = np.asarray(corrected.convert("RGB"))
    edge_mask = original_alpha < 255

    assert np.any(edge_mask)
    assert not np.array_equal(original_rgb[edge_mask], corrected_rgb[edge_mask])

    background_rgb = np.asarray(background)
    diff = np.mean(np.abs(corrected_rgb[edge_mask] - background_rgb[edge_mask]))
    assert diff < 50


def test_no_contamination_from_foreground() -> None:
    size = 16
    foreground = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for y in range(5, 11):
        for x in range(5, 11):
            foreground.putpixel((x, y), (255, 0, 0, 255))
    for y in range(4, 12):
        for x in range(4, 12):
            if foreground.getpixel((x, y))[3] == 0:
                foreground.putpixel((x, y), (0, 0, 0, 128))

    background = Image.new("RGB", (size, size), (0, 0, 255))

    corrected = correct_edges_with_background(foreground, background)
    corrected_rgb = np.asarray(corrected.convert("RGB"))
    alpha = np.asarray(foreground.split()[-1])
    edge_mask = alpha == 128

    edge_pixels = corrected_rgb[edge_mask]
    assert edge_pixels.size
    assert np.mean(edge_pixels[:, 2]) > np.mean(edge_pixels[:, 0])
    assert not np.any(edge_pixels[:, 0] > 200)
