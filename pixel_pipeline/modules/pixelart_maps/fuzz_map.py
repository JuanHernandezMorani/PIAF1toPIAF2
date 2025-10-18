"""Fuzz map generation tuned for perceptual continuity under extreme lighting."""

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage import gaussian_filter

def generate(image: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Generate a micro-edge fuzz map with adaptive noise for dark sprites.

    The procedure modulates uniform noise by the average brightness of the
    detected edge field so that extremely dark sprites (e.g., all-black inputs)
    do not accumulate bright speckles that break temporal coherence. A light
    Gaussian filter ensures spatial continuity before simulating secondary
    scattering.
    """
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0

    # Detección de bordes suaves en luminancia
    gray = np.asarray(ImageOps.grayscale(rgba), dtype=np.float32) / 255.0
    sobel_x = np.gradient(gray, axis=1)
    sobel_y = np.gradient(gray, axis=0)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = np.clip(edges, 0.0, 1.0)

    # Ruido adaptativo a la luminancia media
    brightness = float(np.mean(edges))
    noise = np.random.default_rng().uniform(-0.1, 0.1, edges.shape)
    fuzz = np.clip(edges + noise * (0.5 + brightness) * intensity, 0.0, 1.0)

    # Suavizado espacial leve para reforzar coherencia local
    fuzz = gaussian_filter(fuzz, sigma=0.6)

    # Simular scattering (alas, pelaje, etc.)
    blur = Image.fromarray((fuzz * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(2))
    scatter = np.asarray(blur, dtype=np.float32) / 255.0
    final = np.clip((0.7 * fuzz + 0.3 * scatter), 0.0, 1.0)

    # Componer RGBA
    fuzz_bytes = (final * 255).astype(np.uint8)
    fuzz_rgba = np.dstack([fuzz_bytes, fuzz_bytes, fuzz_bytes, (alpha * 255).astype(np.uint8)])
    return Image.fromarray(fuzz_rgba, mode="RGBA")


if __name__ == "__main__":  # pragma: no cover
    sample = Image.new("RGBA", (16, 16), (180, 40, 120, 255))
    generate(sample).show()
