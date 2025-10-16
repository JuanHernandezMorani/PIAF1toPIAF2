"""Generate tangent-space normal maps from pixel art sprites."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _sobel_edges(gray: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    gx_kernel = ImageFilter.Kernel(
        size=(3, 3),
        kernel=(-1, 0, 1, -2, 0, 2, -1, 0, 1),
        scale=1,
    )
    gy_kernel = ImageFilter.Kernel(
        size=(3, 3),
        kernel=(-1, -2, -1, 0, 0, 0, 1, 2, 1),
        scale=1,
    )
    grad_x = gray.filter(gx_kernel)
    grad_y = gray.filter(gy_kernel)
    return np.asarray(grad_x, dtype=np.float32), np.asarray(grad_y, dtype=np.float32)


def generate(image: Image.Image) -> Image.Image:
    """Create a normal map from *image* using Sobel gradients."""

    rgba = image.convert("RGBA")
    gray = rgba.convert("L")
    grad_x, grad_y = _sobel_edges(gray)
    depth = np.full_like(grad_x, 255.0)
    normals = np.stack([grad_x, grad_y, depth], axis=-1)
    normals -= normals.min(axis=(0, 1), keepdims=True)
    normals /= np.maximum(normals.max(axis=(0, 1), keepdims=True), 1.0)
    normals = (normals * 255.0).astype(np.uint8)
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    normal_rgba = np.dstack([normals, alpha])
    return Image.fromarray(normal_rgba, mode="RGBA")


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    sample = Image.new("RGBA", (16, 16), (120, 100, 90, 255))
    normal_map = generate(sample)
    normal_map.show()
