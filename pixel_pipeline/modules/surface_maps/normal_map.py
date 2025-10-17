"""Generate tangent-space normal maps from pixel art sprites (Sobel + normalization)."""
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


def generate(image: Image.Image, strength: float = 2.0) -> Image.Image:
    """Create a physically plausible normal map (tangent space, bluish) from *image* using Sobel gradients."""
    rgba = image.convert("RGBA")
    gray = rgba.convert("L")

    gx, gy = _sobel_edges(gray)
    gx /= 255.0
    gy /= 255.0

    # compute tangent-space normal
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(gray, dtype=np.float32)

    # normalize vectors
    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx /= length
    ny /= length
    nz /= length

    # remap from [-1,1] → [0,1]
    normal = np.stack([(nx + 1) * 0.5, (ny + 1) * 0.5, (nz + 1) * 0.5], axis=-1)
    normal = np.clip(normal * 255.0, 0, 255).astype(np.uint8)

    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    normal_rgba = np.dstack([normal, alpha])
    return Image.fromarray(normal_rgba, mode="RGBA")


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    sample = Image.new("RGBA", (16, 16), (120, 100, 90, 255))
    generate(sample).show()
