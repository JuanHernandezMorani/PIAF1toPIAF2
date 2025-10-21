"""Legacy entry point for the recolor pipeline with defensive patches applied."""
from __future__ import annotations

import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _convert_to_image(source: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(source, Image.Image):
        return source

    array = np.asarray(source)
    if array.ndim == 2:
        array = array[..., None]
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0 if array.max() <= 1.0 + 1e-6 else 255.0)
        if array.max() <= 1.0 + 1e-6:
            array = (array * 255.0).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    return Image.fromarray(array)


def _basic_alpha_composite(foreground: Image.Image, background: Image.Image) -> Image.Image:
    fg_img = _convert_to_image(foreground).convert("RGBA")
    bg_img = _convert_to_image(background).convert("RGBA")

    if fg_img.size != bg_img.size:
        bg_img = bg_img.resize(fg_img.size, Image.Resampling.LANCZOS)

    fg = np.asarray(fg_img, dtype=np.float32) / 255.0
    bg = np.asarray(bg_img, dtype=np.float32) / 255.0

    fg_rgb = fg[..., :3]
    fg_alpha = fg[..., 3:4]
    bg_rgb = bg[..., :3]
    bg_alpha = bg[..., 3:4]

    result_alpha = fg_alpha + bg_alpha * (1.0 - fg_alpha)
    epsilon = 1e-8
    safe_alpha = np.maximum(result_alpha, epsilon)
    result_rgb = (fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1.0 - fg_alpha)) / safe_alpha

    combined = np.dstack((np.clip(result_rgb, 0.0, 1.0), np.clip(result_alpha, 0.0, 1.0)))
    return Image.fromarray((combined * 255.0).astype(np.uint8), mode="RGBA")


try:
    from pixel_pipeline.modules.pbr import layered as _layered_module
except ImportError:  # pragma: no cover - safety fallback
    _layered_module = None
else:
    _ORIGINAL_SMART_ALPHA = getattr(_layered_module, "_smart_alpha_composition_improved", None)

    if _ORIGINAL_SMART_ALPHA is not None:

        def _safe_alpha_composite(foreground, background):  # type: ignore[override]
            try:
                return _ORIGINAL_SMART_ALPHA(foreground, background)
            except Exception:
                return _basic_alpha_composite(foreground, background)

        _layered_module._smart_alpha_composition_improved = _safe_alpha_composite


from pixel_pipeline.main_recolor import main


if __name__ == "__main__":
    main()
