import numpy as np
from PIL import Image, ImageFilter, ImageOps

def generate(image: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Generate an enhanced fuzz map emphasizing micro edges and soft translucency."""
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0

    # Detección de bordes suaves en luminancia
    gray = np.asarray(ImageOps.grayscale(rgba), dtype=np.float32) / 255.0
    sobel_x = np.gradient(gray, axis=1)
    sobel_y = np.gradient(gray, axis=0)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = np.clip(edges, 0.0, 1.0)

    # Ruido multiescala fractal
    rng = np.random.default_rng()
    base = rng.normal(0.0, 0.1, edges.shape)
    detail = rng.normal(0.0, 0.05, edges.shape)
    noise = base + detail
    fuzz = np.clip(edges + noise * intensity, 0.0, 1.0)

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
