"""Generate tangent-space normal maps with perceptually smooth gradients."""
from __future__ import annotations

import numpy as np
from PIL import Image

try:  # OpenCV offers an efficient bilateral filter when available
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

def generate(image: Image.Image, strength: float = 2.0) -> Image.Image:
    """Create a tangent-space normal map with bilateral pre-filtering.

    The pipeline smooths the luminance field using a bilateral (or Gaussian)
    filter before estimating gradients. This step suppresses salt-and-pepper
    noise and yields normals that behave coherently under shading, especially
    for sprites with hard alpha edges or low-light palettes.
    """
    rgba = image.convert("RGBA")
    gray = rgba.convert("L")

    img_gray = np.asarray(gray, dtype=np.float32)
    if cv2 is not None:
        smooth = cv2.bilateralFilter(img_gray, d=5, sigmaColor=30, sigmaSpace=15)
    else:  # pragma: no cover - SciPy fallback for environments without OpenCV
        from scipy.ndimage import gaussian_filter

        smooth = gaussian_filter(img_gray, sigma=1.0)

    grad_y, grad_x = np.gradient(smooth)
    gx = grad_x.astype(np.float32) / 255.0
    gy = grad_y.astype(np.float32) / 255.0

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
