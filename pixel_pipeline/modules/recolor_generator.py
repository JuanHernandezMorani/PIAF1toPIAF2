"""Core recoloring pipeline that orchestrates image variants and map generation."""
from __future__ import annotations

import importlib
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

from ..core import config
from ..core.utils_color import (
    add_color_noise,
    color_distance,
    mix_colors,
    random_palette,
    rgb_to_hsl,
    hsl_to_rgb,
)
from ..core.utils_io import SafeFileManager, ensure_dir
from ..core.utils_parallel import limited_threads, run_parallel
from . import contextual_randomizer
from .pbr import generate_physically_accurate_pbr_maps, generate_quality_report
from .pbr.alpha_utils import apply_alpha, apply_alpha_to_maps, derive_alpha_map
from .pbr.layered import LAYERED_PBR_CONFIG, LayeredPBRCache, PBRLayerManager

try:
    from pixel_pipeline.modules.perceptual_vision import (
        apply_human_vision_simulation,
        validate_photopic_parameters,
    )

    PERCEPTUAL_VISION_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PERCEPTUAL_VISION_AVAILABLE = False

LOGGER = logging.getLogger("pixel_pipeline.recolor")

GeneratorFn = Callable[[Image.Image], Image.Image]

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

DEFAULT_COLOR_CONFIG = {
    "MIN_COLOR": 100,
    "MAX_DOMINANT_COLORS": 30,
    "MIN_VARIATIONS_PER_COLOR": 15,
    "MAX_VARIATIONS_PER_COLOR": 40,
    "MIN_COLOR_DIFFERENCE": 8.0,
    "HUE_VARIATION_RANGE": (-0.3, 0.3),
    "SATURATION_VARIATION_RANGE": (-0.7, 0.7),
    "LIGHTNESS_VARIATION_RANGE": (-0.5, 0.5),
    "INTENSITY_VARIATION_RANGE": (0.5, 1.8),
    "PRESERVE_BRIGHTNESS": False,
    "RANDOM_COLOR_PROBABILITY": 0.3,
    "CROSS_COLOR_MIXING": True,
    "MAX_COLOR_MIXES": 256,
    "NOISE_INTENSITY": 0.08,
    "ENSURE_MIN_VARIATIONS": True,
    "MAX_GENERATION_ATTEMPTS": 800,
}


@dataclass
class Variant:
    """Represent a generated variant with metadata."""

    name: str
    image: Image.Image
    background: Image.Image
    foreground_base: Optional[Image.Image] = None
    background_base: Optional[Image.Image] = None
    foreground_variant: Optional[Image.Image] = None
    background_variant: Optional[Image.Image] = None
    color_variant: Optional[Tuple[int, int, int]] = None
    rotation: float = 0.0


class BackgroundLibrary:
    """Load and adapt background imagery for compositing."""

    def __init__(self, background_dir: Path, *, rng: random.Random, allow_generated: bool = True) -> None:
        self.background_dir = background_dir
        self.rng = rng
        self.allow_generated = allow_generated
        self.backgrounds: List[Path] = []
        if background_dir.exists():
            self.backgrounds = [p for p in sorted(background_dir.iterdir()) if p.suffix.lower() in VALID_EXTENSIONS]
        if not self.backgrounds:
            LOGGER.warning("No background images found in %s", background_dir)

    def _apply_variations(self, image: Image.Image) -> Image.Image:
        brightness = self.rng.uniform(0.7, 1.3)
        contrast = self.rng.uniform(0.6, 1.4)
        color = self.rng.uniform(0.8, 1.2)
        varied = ImageEnhance.Brightness(image).enhance(brightness)
        varied = ImageEnhance.Contrast(varied).enhance(contrast)
        varied = ImageEnhance.Color(varied).enhance(color)
        hue_shift = self.rng.uniform(-15, 15)
        if abs(hue_shift) > 1e-3:
            varied = shift_hue(varied, hue_shift)
        noise = self.rng.uniform(0.0, 0.05)
        if noise > 0:
            varied = add_texture_noise(varied, noise)
        return varied

    def random_background(self, size: Tuple[int, int]) -> Image.Image:
        if not self.backgrounds:
            if not self.allow_generated:
                return Image.new("RGBA", size, (0, 0, 0, 0))
            return self._generated_background(size)
        selected = self.rng.choice(self.backgrounds)
        with Image.open(selected) as img:
            rgba = ImageOps.fit(img.convert("RGBA"), size, method=Image.NEAREST)
        return self._apply_variations(rgba)

    def _generated_background(self, size: Tuple[int, int]) -> Image.Image:
        width, height = size
        top_color = random_palette(self.rng.randint(0, 1000000), size=1)[0]
        bottom_color = add_color_noise(top_color, 0.3)
        gradient = Image.new("RGBA", size)
        for y in range(height):
            ratio = y / max(height - 1, 1)
            color = mix_colors(top_color, bottom_color, ratio)
            row = Image.new("RGBA", (width, 1), (*color, 255))
            gradient.paste(row, (0, y))
        return gradient


def shift_hue(image: Image.Image, degrees: float) -> Image.Image:
    """Shift hue by given degrees (safe for negative values)."""
    if degrees == 0:
        return image
    hsv = image.convert("HSV")
    h, s, v = hsv.split()
    # Convert to int32 para permitir valores negativos sin overflow
    np_h = np.asarray(h, dtype=np.int32)
    # Convertir grados a desplazamiento en rango [0,255)
    shift = int(degrees / 360.0 * 255)
    np_h = (np_h + shift) % 255
    np_h = np.clip(np_h, 0, 255).astype(np.uint8)
    h = Image.fromarray(np_h, "L")
    merged = Image.merge("HSV", (h, s, v))
    return merged.convert("RGBA")



def add_texture_noise(image: Image.Image, amount: float) -> Image.Image:
    """Overlay monochromatic noise to add subtle variation."""

    rgba = image.convert("RGBA")
    width, height = rgba.size
    rng = np.random.default_rng()
    noise = (rng.random((height, width, 1)) * amount * 255).astype(np.uint8)
    noise_rgba = np.repeat(noise, 3, axis=2)
    noise_rgba = np.concatenate([noise_rgba, np.full((height, width, 1), 255, dtype=np.uint8)], axis=2)
    noise_image = Image.fromarray(noise_rgba, mode="RGBA")
    blended = ImageChops.add(rgba, noise_image, scale=1.0, offset=0)
    return blended


def _rgb_to_hsv_array(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB→HSV conversion for arrays in range [0, 1]."""

    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    mask = delta > 1e-6
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    red_mask = (maxc == r) & mask
    green_mask = (maxc == g) & mask
    blue_mask = (maxc == b) & mask

    if np.any(red_mask):
        h_val = (g[red_mask] - b[red_mask]) / delta[red_mask]
        h[red_mask] = np.mod(h_val, 6.0)
    if np.any(green_mask):
        h[green_mask] = (b[green_mask] - r[green_mask]) / delta[green_mask] + 2.0
    if np.any(blue_mask):
        h[blue_mask] = (r[blue_mask] - g[blue_mask]) / delta[blue_mask] + 4.0

    h = np.mod(h / 6.0, 1.0)
    s = np.zeros_like(maxc)
    nonzero = maxc > 1e-6
    s[nonzero] = delta[nonzero] / maxc[nonzero]
    v = maxc
    return h, s, v


def _hsv_to_rgb_array(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV→RGB conversion for arrays in range [0, 1]."""

    h = np.mod(h, 1.0)
    s = np.clip(s, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    h_prime = h * 6.0
    sector = np.floor(h_prime).astype(int)
    fraction = h_prime - sector

    p = v * (1.0 - s)
    q = v * (1.0 - s * fraction)
    t = v * (1.0 - s * (1.0 - fraction))

    sector_mod = np.mod(sector, 6)
    conditions = [sector_mod == idx for idx in range(6)]

    r = np.select(conditions, [v, q, p, p, t, v], default=v)
    g = np.select(conditions, [t, v, v, q, p, p], default=v)
    b = np.select(conditions, [p, p, t, v, v, q], default=q)
    return np.stack((r, g, b), axis=2)


def _generate_zone_noise(rng: np.random.Generator, height: int, width: int, scale: float) -> np.ndarray:
    """Create smooth noise fields to perturb pixels by regions."""

    coarse_h = max(1, height // 4)
    coarse_w = max(1, width // 4)
    coarse = rng.normal(0.0, scale, (coarse_h, coarse_w)).astype(np.float32)
    repeat_y = int(np.ceil(height / coarse_h))
    repeat_x = int(np.ceil(width / coarse_w))
    tiled = np.kron(coarse, np.ones((repeat_y, repeat_x), dtype=np.float32))
    return tiled[:height, :width]


def _rgb_to_hls_array(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB→HLS conversion for arrays in range [0, 1]."""

    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    delta = maxc - minc
    sumc = maxc + minc

    l = 0.5 * sumc

    s = np.zeros_like(maxc)
    mask = delta > 1e-6
    denom1 = sumc
    mask1 = mask & (denom1 > 1e-6) & (l <= 0.5)
    s[mask1] = delta[mask1] / denom1[mask1]

    denom2 = 2.0 - sumc
    mask2 = mask & (denom2 > 1e-6) & (l > 0.5)
    s[mask2] = delta[mask2] / denom2[mask2]

    h = np.zeros_like(maxc)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    denom = np.where(mask, delta, 1.0)
    rc = np.zeros_like(maxc)
    gc = np.zeros_like(maxc)
    bc = np.zeros_like(maxc)
    rc[mask] = (maxc[mask] - r[mask]) / denom[mask]
    gc[mask] = (maxc[mask] - g[mask]) / denom[mask]
    bc[mask] = (maxc[mask] - b[mask]) / denom[mask]

    red_mask = (maxc == r) & mask
    green_mask = (maxc == g) & mask
    blue_mask = (maxc == b) & mask

    if np.any(red_mask):
        h[red_mask] = bc[red_mask] - gc[red_mask]
    if np.any(green_mask):
        h[green_mask] = 2.0 + rc[green_mask] - bc[green_mask]
    if np.any(blue_mask):
        h[blue_mask] = 4.0 + gc[blue_mask] - rc[blue_mask]

    h = np.mod(h / 6.0, 1.0)
    return h, s, l

def safe_rotate(img: Image.Image, angle: int) -> Image.Image:
    w, h = img.size
    if w != h:
        size = max(w, h)
        square = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        square.paste(img, ((size - w) // 2, (size - h) // 2))
        img = square
    return img.rotate(angle, expand=True)

def _hls_to_rgb_array(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Vectorized HLS→RGB conversion for arrays in range [0, 1]."""

    h = np.mod(h, 1.0)
    s = np.clip(s, 0.0, 1.0)
    l = np.clip(l, 0.0, 1.0)

    m2 = np.where(l <= 0.5, l * (1.0 + s), l + s - l * s)
    m2 = np.clip(m2, 0.0, 1.0)
    m1 = np.clip(2.0 * l - m2, 0.0, 1.0)

    def _hue_to_rgb(m1_arr: np.ndarray, m2_arr: np.ndarray, hue: np.ndarray) -> np.ndarray:
        hue = np.mod(hue, 1.0)
        result = np.empty_like(hue)

        cond1 = hue < (1.0 / 6.0)
        cond2 = (hue >= (1.0 / 6.0)) & (hue < 0.5)
        cond3 = (hue >= 0.5) & (hue < (2.0 / 3.0))

        if np.any(cond1):
            result[cond1] = m1_arr[cond1] + (m2_arr[cond1] - m1_arr[cond1]) * hue[cond1] * 6.0
        if np.any(cond2):
            result[cond2] = m2_arr[cond2]
        if np.any(cond3):
            result[cond3] = m1_arr[cond3] + (m2_arr[cond3] - m1_arr[cond3]) * ((2.0 / 3.0) - hue[cond3]) * 6.0
        other = ~(cond1 | cond2 | cond3)
        if np.any(other):
            result[other] = m1_arr[other]
        return np.clip(result, 0.0, 1.0)

    r = _hue_to_rgb(m1, m2, h + (1.0 / 3.0))
    g = _hue_to_rgb(m1, m2, h)
    b = _hue_to_rgb(m1, m2, h - (1.0 / 3.0))
    return np.stack((r, g, b), axis=2)


_SOBEL_X = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32)
_SOBEL_Y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=np.float32)
_UNIFORM_KERNEL_5 = np.full((5, 5), 1.0 / 25.0, dtype=np.float32)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Lightweight 2D convolution with reflective padding."""

    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    output = np.zeros_like(image, dtype=np.float32)
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            output += kernel[y, x] * padded[y : y + image.shape[0], x : x + image.shape[1]]
    return output


def _compute_perceptual_silhouette(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Estimate a perceptual silhouette mask from luminance and texture cues."""

    if rgb.size == 0:
        return alpha

    gray = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
    gray = gray.astype(np.float32)
    _, saturation, _ = _rgb_to_hls_array(rgb)

    grad_x = _convolve2d(gray, _SOBEL_X)
    grad_y = _convolve2d(gray, _SOBEL_Y)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    gradient /= gradient.max() + 1e-6

    mean = _convolve2d(gray, _UNIFORM_KERNEL_5)
    mean_sq = _convolve2d(gray * gray, _UNIFORM_KERNEL_5)
    variance = np.clip(mean_sq - mean * mean, 0.0, None)
    if variance.max() > 1e-6:
        variance /= variance.max()

    response = 0.6 * gradient + 0.25 * variance + 0.15 * saturation
    response = np.clip(response, 0.0, 1.0)
    if response.max() > 1e-6:
        response /= response.max()

    threshold = max(float(np.percentile(response, 60.0)), 0.15)
    binary = (response >= threshold).astype(np.float32)

    mask_image = Image.fromarray(np.clip(binary * 255.0, 0, 255).astype(np.uint8), mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(size=3))
    mask_image = mask_image.filter(ImageFilter.MinFilter(size=3))
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=1.5))
    perceptual_mask = np.asarray(mask_image, dtype=np.float32) / 255.0
    perceptual_mask = np.clip(perceptual_mask, 0.0, 1.0)

    alpha_base = np.clip(alpha[..., 0], 0.0, 1.0)
    has_transparency = float(np.mean(alpha_base < 0.98)) > 0.02
    if has_transparency:
        combined = np.maximum(alpha_base, perceptual_mask)
    else:
        combined = np.maximum(perceptual_mask, alpha_base)

    if combined.max() < 1e-6:
        combined = np.clip(alpha_base, 0.0, 1.0)

    return combined[..., None]


class RecolorPipeline:
    """Generate recolored variants and derived maps for input sprites."""

    def __init__(self, cfg: Dict[str, object]) -> None:
        self.config = cfg
        self.input_path = Path(cfg["PATH_INPUT"])
        self.output_path = Path(cfg["PATH_OUTPUT"])
        self.background_path = Path(cfg["PATH_BACKGROUNDS"])
        ensure_dir(self.output_path)
        self.file_manager = SafeFileManager(self.output_path)
        self.rotation_angles: Tuple[int, ...] = tuple(cfg.get("ROTATION_ANGLES", (0,)))  # type: ignore[arg-type]
        self.max_variants: int = int(cfg.get("MAX_VARIANTS", 200))
        seed = cfg.get("RANDOM_SEED")
        self.rng = random.Random(seed)
        self.use_real_backgrounds = bool(cfg.get("USE_REAL_BACKGROUNDS_ONLY", True))
        self.backgrounds = BackgroundLibrary(self.background_path, rng=self.rng, allow_generated=not self.use_real_backgrounds)
        self.map_generators = self._load_map_generators(cfg.get("MAP_TYPES", config.MAP_TYPES))
        self.pixel_resolution = int(cfg.get("PIXEL_RESOLUTION", config.PIXEL_RESOLUTION))
        self.logger = LOGGER
        self.enable_gpu = bool(cfg.get("ENABLE_GPU", False))
        self.gpu_available = self._detect_gpu() if self.enable_gpu else False
        if self.enable_gpu and not self.gpu_available:
            self.logger.warning("GPU acceleration requested but no compatible backend was found")
        self.logger.debug("Pipeline configured with %s", cfg)
        color_cfg = DEFAULT_COLOR_CONFIG.copy()
        color_cfg.update({k: v for k, v in cfg.items() if k in DEFAULT_COLOR_CONFIG})
        self.min_color_pixels = int(color_cfg["MIN_COLOR"])
        self.max_dominant_colors = int(color_cfg["MAX_DOMINANT_COLORS"])
        self.min_variations_per_color = max(1, int(color_cfg["MIN_VARIATIONS_PER_COLOR"]))
        self.max_variations_per_color = max(self.min_variations_per_color, int(color_cfg["MAX_VARIATIONS_PER_COLOR"]))
        self.min_color_difference = float(color_cfg["MIN_COLOR_DIFFERENCE"])
        self.hue_variation_range = tuple(float(v) for v in color_cfg["HUE_VARIATION_RANGE"])
        self.saturation_variation_range = tuple(float(v) for v in color_cfg["SATURATION_VARIATION_RANGE"])
        self.lightness_variation_range = tuple(float(v) for v in color_cfg["LIGHTNESS_VARIATION_RANGE"])
        self.intensity_variation_range = tuple(float(v) for v in color_cfg["INTENSITY_VARIATION_RANGE"])
        self.preserve_brightness = bool(color_cfg["PRESERVE_BRIGHTNESS"])
        self.random_color_probability = float(color_cfg["RANDOM_COLOR_PROBABILITY"])
        self.cross_color_mixing = bool(color_cfg["CROSS_COLOR_MIXING"])
        self.max_color_mixes = int(color_cfg["MAX_COLOR_MIXES"])
        self.noise_intensity = float(color_cfg["NOISE_INTENSITY"])
        self.ensure_min_variations = bool(color_cfg["ENSURE_MIN_VARIATIONS"])
        self.max_generation_attempts = int(color_cfg["MAX_GENERATION_ATTEMPTS"])
        layered_cfg_input = cfg.get("LAYERED_PBR")
        base_layered_cfg = dict(LAYERED_PBR_CONFIG)
        base_layered_cfg["composition_methods"] = dict(LAYERED_PBR_CONFIG.get("composition_methods", {}))
        if isinstance(layered_cfg_input, Mapping):
            for key, value in layered_cfg_input.items():
                if key == "composition_methods" and isinstance(value, Mapping):
                    base_layered_cfg["composition_methods"].update(value)
                else:
                    base_layered_cfg[key] = value
        self.layered_pbr_config = base_layered_cfg
        self.layered_pbr_enabled = bool(self.layered_pbr_config.get("generate_separate_pbr", False))
        self.layered_cache = (
            LayeredPBRCache()
            if self.layered_pbr_enabled and self.layered_pbr_config.get("cache_layers_separately", True)
            else None
        )
        self.pbr_layer_manager: PBRLayerManager | None = None
        if self.layered_pbr_enabled:
            self.pbr_layer_manager = PBRLayerManager(self.layered_pbr_config, cache=self.layered_cache)
        self.background_tint_strength = float(self.layered_pbr_config.get("background_tint_strength", 0.0))
        self._stats_lock = threading.Lock()
        self._images_processed = 0
        self._variants_generated = 0
        self._total_image_time = 0.0
        contextual_randomizer.pixel_variation_callback(
            self._apply_pixel_variation, persistent=True
        )

    def _load_map_generators(self, mapping: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, GeneratorFn]]:
        generators: Dict[str, Dict[str, GeneratorFn]] = {}
        for category, entries in mapping.items():
            category_generators: Dict[str, GeneratorFn] = {}
            for name, module_path in entries.items():
                module = importlib.import_module(module_path)
                generator = getattr(module, "generate", None)
                if generator is None:
                    raise AttributeError(f"Module {module_path} does not expose a generate() function")
                category_generators[name] = generator
            generators[category] = category_generators
        return generators

    def _iter_input_files(self) -> Iterator[Path]:
        for entry in sorted(self.input_path.iterdir()):
            if entry.suffix.lower() in VALID_EXTENSIONS and entry.is_file():
                yield entry

    def _detect_gpu(self) -> bool:
        try:
            import importlib

            importlib.import_module("cupy")
            return True
        except ModuleNotFoundError:
            try:
                importlib.import_module("pyopencl")
                return True
            except ModuleNotFoundError:
                return False

    def run(self, *, threads: int = 4) -> None:
        """Process the input directory and generate variants."""

        files = list(self._iter_input_files())
        if not files:
            self.logger.warning("No input files found in %s", self.input_path)
            return
        self.logger.info("Processing %d input files", len(files))
        start_time = time.perf_counter()
        with limited_threads(threads):
            run_parallel(self._process_file, files, max_workers=threads)
        total_time = time.perf_counter() - start_time
        with self._stats_lock:
            images = self._images_processed
            variants = self._variants_generated
            total_image_time = self._total_image_time
        avg_variants = variants / images if images else 0.0
        avg_time = total_image_time / images if images else 0.0
        self.logger.info(
            "Imagenes transformadas: %d, Transformaciones por imagen: %.2f, Tiempo transcurrido por imagen: %.2fs, Tiempo total transcurrido para la tarea: %.2fs",
            images,
            avg_variants,
            avg_time,
            total_time,
        )

    # Variant generation -------------------------------------------------
    def _process_file(self, path: Path) -> None:
        self.logger.info("Generating variants for %s", path.name)
        process_start = time.perf_counter()
        try:
            with Image.open(path) as img:
                rgba = img.convert("RGBA")
        except Exception as exc:  # pragma: no cover - logging path
            self.logger.exception("Failed to open %s: %s", path, exc)
            return

        variants = list(self._generate_variants(path.stem, rgba))
        persisted = 0
        for variant in variants[: self.max_variants]:
            self._persist_variant(path.stem, variant)
            persisted += 1
        elapsed = time.perf_counter() - process_start
        with self._stats_lock:
            self._images_processed += 1
            self._variants_generated += persisted
            self._total_image_time += elapsed

    def _generate_variants(self, stem: str, image: Image.Image) -> Iterator[Variant]:
        palette = self._extract_palette(image)
        variant_index = 0
        base_foreground_template = image.convert("RGBA") if self.layered_pbr_enabled else None
        for rotation in self.rotation_angles:
            if variant_index >= self.max_variants:
                return
            rotated: Optional[Image.Image] = None
            for tint in palette:
                if variant_index >= self.max_variants:
                    return
                if self.layered_pbr_enabled:
                    base_foreground = base_foreground_template.copy() if base_foreground_template else image.convert("RGBA")
                    foreground_variant = self._apply_tint(base_foreground.copy(), tint)
                    if PERCEPTUAL_VISION_AVAILABLE and self.config.get("enable_perceptual_simulation", True):
                        lighting_condition = str(self.config.get("lighting_condition", "photopic"))
                        adaptation_level = float(self.config.get("adaptation_level", 1.0))
                        luminance = self.config.get("luminance_cd_m2")
                        foreground_variant = apply_human_vision_simulation(
                            foreground_variant,
                            lighting_condition=lighting_condition,
                            adaptation_level=adaptation_level,
                            luminance_cd_m2=float(luminance) if luminance is not None else None,
                        )
                        if self.logger.isEnabledFor(logging.DEBUG):
                            metrics = validate_photopic_parameters(foreground_variant)
                            self.logger.debug("Perceptual metrics for %s: %s", tint, metrics)
                    background_base = self._adapt_background(foreground_variant).convert("RGBA")
                    background_variant = self._apply_background_color_variant(background_base.copy(), tint)
                    name = f"{stem}_r{rotation:03d}_{variant_index:04d}"
                    variant_index += 1
                    yield Variant(
                        name=name,
                        image=foreground_variant,
                        background=background_variant,
                        foreground_base=base_foreground,
                        background_base=background_base,
                        foreground_variant=foreground_variant,
                        background_variant=background_variant,
                        color_variant=tint,
                        rotation=float(rotation),
                    )
                else:
                    if rotated is None:
                        rotated = safe_rotate(image, rotation)
                    variant_image = self._apply_tint(rotated, tint)
                    if PERCEPTUAL_VISION_AVAILABLE and self.config.get("enable_perceptual_simulation", True):
                        lighting_condition = str(self.config.get("lighting_condition", "photopic"))
                        adaptation_level = float(self.config.get("adaptation_level", 1.0))
                        luminance = self.config.get("luminance_cd_m2")
                        variant_image = apply_human_vision_simulation(
                            variant_image,
                            lighting_condition=lighting_condition,
                            adaptation_level=adaptation_level,
                            luminance_cd_m2=float(luminance) if luminance is not None else None,
                        )
                        if self.logger.isEnabledFor(logging.DEBUG):
                            metrics = validate_photopic_parameters(variant_image)
                            self.logger.debug("Perceptual metrics for %s: %s", tint, metrics)
                    adapted = self._adapt_background(variant_image)
                    name = f"{stem}_r{rotation:03d}_{variant_index:04d}"
                    variant_index += 1
                    yield Variant(name=name, image=variant_image, background=adapted, rotation=float(rotation))

    def _extract_palette(self, image: Image.Image) -> List[Tuple[int, int, int]]:
        resized = image.convert("RGBA").resize((self.pixel_resolution, self.pixel_resolution), resample=Image.NEAREST)
        colors = resized.getcolors(maxcolors=self.pixel_resolution * self.pixel_resolution)
        if not colors:
            return random_palette(self.rng.randint(0, 1000000))
        colors_sorted = sorted(colors, key=lambda item: item[0], reverse=True)
        dominant: List[Tuple[int, int, int]] = []
        for count, color in colors_sorted:
            if count < self.min_color_pixels:
                continue
            rgb = color[:3]
            if all(color_distance(rgb, existing) >= self.min_color_difference for existing in dominant):
                dominant.append(rgb)
            if len(dominant) >= self.max_dominant_colors:
                break
        if not dominant:
            for _, color in colors_sorted:
                rgb = color[:3]
                if all(color_distance(rgb, existing) >= self.min_color_difference for existing in dominant):
                    dominant.append(rgb)
                if len(dominant) >= self.max_dominant_colors:
                    break
        if not dominant:
            dominant.extend(random_palette(self.rng.randint(0, 1000000), size=4))

        palette: List[Tuple[int, int, int]] = []
        for base in dominant:
            variations = self._generate_color_variations(base)
            for variant in variations:
                if all(color_distance(variant, existing) >= self.min_color_difference for existing in palette):
                    palette.append(variant)
                if len(palette) >= self.max_variants:
                    break
            if len(palette) >= self.max_variants:
                break
            if self.random_color_probability > 0 and self.rng.random() < self.random_color_probability:
                random_color = random_palette(self.rng.randint(0, 1000000), size=1)[0]
                if all(color_distance(random_color, existing) >= self.min_color_difference for existing in palette):
                    palette.append(random_color)
            if len(palette) >= self.max_variants:
                break

        if self.cross_color_mixing and len(dominant) >= 2 and len(palette) < self.max_variants:
            mixes = 0
            limit = max(0, self.max_color_mixes)
            for idx, color_a in enumerate(dominant):
                for color_b in dominant[idx + 1 :]:
                    if limit and mixes >= limit:
                        break
                    ratio = self.rng.uniform(0.25, 0.75)
                    mixed = mix_colors(color_a, color_b, ratio)
                    if self.noise_intensity > 0:
                        mixed = add_color_noise(mixed, self.noise_intensity)
                    if all(color_distance(mixed, existing) >= self.min_color_difference for existing in palette):
                        palette.append(mixed)
                        mixes += 1
                    if len(palette) >= self.max_variants:
                        break
                if limit and mixes >= limit:
                    break
                if len(palette) >= self.max_variants:
                    break

        if self.ensure_min_variations and len(palette) < self.min_variations_per_color:
            needed = self.min_variations_per_color - len(palette)
            fallback_colors = random_palette(self.rng.randint(0, 1000000), size=needed * 2)
            for color in fallback_colors:
                if all(color_distance(color, existing) >= self.min_color_difference for existing in palette):
                    palette.append(color)
                if len(palette) >= self.min_variations_per_color:
                    break

        if not palette:
            palette = random_palette(self.rng.randint(0, 1000000))

        max_per_rotation = max(1, self.max_variants // max(1, len(self.rotation_angles)))
        return palette[: max_per_rotation]

    def _generate_color_variations(self, base: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        base_h, base_s, base_l = rgb_to_hsl(base)
        variations: List[Tuple[int, int, int]] = []
        np_rng = np.random.default_rng(self.rng.randint(0, 2**32 - 1))
        target_count = np_rng.integers(self.min_variations_per_color, self.max_variations_per_color + 1)
        attempts = 0
        max_attempts = max(self.max_generation_attempts, target_count * 2)
        while len(variations) < target_count and attempts < max_attempts:
            attempts += 1
            hue_shift = np_rng.uniform(self.hue_variation_range[0], self.hue_variation_range[1])
            sat_shift = np_rng.uniform(self.saturation_variation_range[0], self.saturation_variation_range[1])
            light_shift = np_rng.uniform(self.lightness_variation_range[0], self.lightness_variation_range[1])
            intensity = np_rng.uniform(self.intensity_variation_range[0], self.intensity_variation_range[1])
            new_h = (base_h + hue_shift) % 1.0
            new_s = np.clip(base_s + sat_shift, 0.0, 1.0)
            target_l = np.clip(base_l * intensity + light_shift, 0.0, 1.0)
            if self.preserve_brightness:
                new_l = np.clip(0.5 * base_l + 0.5 * target_l, 0.0, 1.0)
            else:
                new_l = target_l
            candidate = hsl_to_rgb((new_h, new_s, new_l))
            if all(color_distance(candidate, existing) >= self.min_color_difference for existing in variations):
                variations.append(candidate)
        if not variations:
            variations.append(base)
        return variations

    def _apply_tint(self, image: Image.Image, tint: Tuple[int, int, int]) -> Image.Image:
        rgba = image.convert("RGBA")
        arr = np.asarray(rgba, dtype=np.float32) / 255.0
        alpha = arr[..., 3:4]
        rgb = arr[..., :3]
        if rgb.size == 0:
            return image

        np_rng = np.random.default_rng(self.rng.randint(0, 2**32 - 1))
        h, s, l = _rgb_to_hls_array(rgb)

        base_h, base_s, base_l = rgb_to_hsl(tint)
        hue_noise = np_rng.uniform(self.hue_variation_range[0], self.hue_variation_range[1], h.shape)
        sat_noise = np_rng.uniform(self.saturation_variation_range[0], self.saturation_variation_range[1], s.shape)
        light_noise = np_rng.uniform(self.lightness_variation_range[0], self.lightness_variation_range[1], l.shape)
        intensity = np_rng.uniform(self.intensity_variation_range[0], self.intensity_variation_range[1], l.shape)

        base_h_array = np.full_like(h, base_h)
        base_s_array = np.full_like(s, base_s)
        base_l_array = np.full_like(l, base_l)

        new_h = np.mod(base_h_array + hue_noise, 1.0)
        new_s = np.clip(base_s_array + sat_noise, 0.0, 1.0)
        target_l = np.clip(base_l_array * intensity + light_noise, 0.0, 1.0)
        if self.preserve_brightness:
            new_l = np.clip(0.5 * l + 0.5 * target_l, 0.0, 1.0)
        else:
            new_l = target_l

        varied_rgb = _hls_to_rgb_array(new_h, new_s, new_l)
        silhouette = _compute_perceptual_silhouette(rgb, alpha)
        combined_rgb = rgb * (1.0 - silhouette) + varied_rgb * silhouette
        combined_alpha = np.clip(np.maximum(alpha, silhouette), 0.0, 1.0)
        combined = np.dstack((combined_rgb, combined_alpha))
        tinted = Image.fromarray(np.clip(combined * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")
        tinted = contextual_randomizer.apply_global_recolor(tinted, self.rng)
        tinted = contextual_randomizer.integrate_contextual_variation(tinted, self.rng)
        brightness = self.rng.uniform(0.8, 1.2)
        contrast = self.rng.uniform(0.8, 1.3)
        tinted = ImageEnhance.Brightness(tinted).enhance(brightness)
        tinted = ImageEnhance.Contrast(tinted).enhance(contrast)
        return tinted

    def _apply_background_color_variant(
        self, background: Image.Image, tint: Tuple[int, int, int]
    ) -> Image.Image:
        if not self.layered_pbr_enabled:
            return background
        strength = float(np.clip(self.background_tint_strength, 0.0, 1.0))
        if strength <= 0.0:
            return background
        rgba = background.convert("RGBA")
        arr = np.asarray(rgba, dtype=np.float32) / 255.0
        alpha = arr[..., 3:4]
        rgb = arr[..., :3]
        tint_rgb = np.array(tint, dtype=np.float32) / 255.0
        tinted_rgb = np.clip(rgb * (1.0 - strength) + tint_rgb * strength, 0.0, 1.0)
        combined = np.concatenate((tinted_rgb, alpha), axis=2)
        return Image.fromarray(np.clip(combined * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")

    def _apply_pixel_variation(self, image: Image.Image) -> Image.Image:
        """Apply pixel-wise hue, saturation, and luminance perturbations."""

        rgba = image.convert("RGBA")
        arr = np.asarray(rgba, dtype=np.float32)
        alpha = np.clip(arr[..., 3:4], 0.0, 255.0)
        rgb = arr[..., :3] / 255.0
        height, width = rgb.shape[:2]
        if height == 0 or width == 0:
            return image

        rng = np.random.default_rng(self.rng.randint(0, 2**32 - 1))
        hue_noise = rng.uniform(-0.04, 0.04, (height, width)).astype(np.float32)
        sat_noise = rng.normal(0.0, 0.06, (height, width)).astype(np.float32)
        val_noise = rng.normal(0.0, 0.06, (height, width)).astype(np.float32)
        h, s, v = _rgb_to_hsv_array(rgb)
        zone_noise = _generate_zone_noise(rng, height, width, scale=0.08)

        # Simulate human perception (non-linear luminance)
        gamma = 2.2
        v = np.power(v, 1.0 / gamma)
        h = np.mod(h + hue_noise, 1.0)
        s = np.clip(s + sat_noise + zone_noise * 0.5, 0.0, 1.0)
        v = np.clip(v + val_noise + zone_noise * 0.5, 0.0, 1.0)

        varied_rgb = _hsv_to_rgb_array(h, s, v)
        varied_rgb = np.clip(varied_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        alpha_channel = alpha.astype(np.uint8)
        combined = np.concatenate((varied_rgb, alpha_channel), axis=2)
        return Image.fromarray(combined, mode="RGBA")

    def _adapt_background(self, variant: Image.Image) -> Image.Image:
        size = variant.size
        background = self.backgrounds.random_background(size)
        variant_rgb = np.asarray(variant.convert("RGB"), dtype=np.float32)
        alpha = np.asarray(variant.split()[-1], dtype=np.float32) / 255.0
        if np.any(alpha > 0):
            foreground_mean = variant_rgb[alpha > 0].mean(axis=0)
        else:
            foreground_mean = np.array([128, 128, 128], dtype=np.float32)
        bg_rgb = np.asarray(background.convert("RGB"), dtype=np.float32)
        h, s, l = rgb_to_hsl(tuple(int(c) for c in foreground_mean))
        adjusted_color = hsl_to_rgb((h, min(1.0, s * 0.9), min(1.0, l * 1.05)))
        adapted_base = np.array(adjusted_color, dtype=np.float32)
        factor = 0.25
        adapted_rgb = bg_rgb * (1 - factor) + adapted_base * factor
        adapted_rgb = np.clip(adapted_rgb, 0, 255).astype(np.uint8)
        adapted = Image.fromarray(adapted_rgb, mode="RGB").convert("RGBA")
        adapted.putalpha(background.split()[-1])
        return adapted

    # Persistence ---------------------------------------------------------
    def _persist_variant(self, stem: str, variant: Variant) -> None:
        if (
            self.layered_pbr_enabled
            and variant.foreground_base is not None
            and variant.background_base is not None
        ):
            self._persist_layered_variant(stem, variant)
            return
        base_name = variant.name
        composited = Image.alpha_composite(variant.background, variant.image)
        target_path = self.output_path / f"{base_name}.png"
        maps, alpha_map = self._generate_maps(composited, variant.background)
        composited = apply_alpha(composited, alpha_map)
        self.file_manager.atomic_save(composited, target_path)
        for map_name, map_image in maps.items():
            map_path = self.output_path / f"{base_name}_{map_name}.png"
            self.file_manager.atomic_save(map_image, map_path)

    def _persist_layered_variant(self, stem: str, variant: Variant) -> None:
        if self.pbr_layer_manager is None:
            raise RuntimeError("Layered PBR manager is not configured")
        base_name = variant.name
        layered_maps, alpha_map = self._generate_maps_layered(
            variant.foreground_base,  # type: ignore[arg-type]
            variant.background_base,  # type: ignore[arg-type]
            rotation=variant.rotation,
        )
        foreground_variant = variant.foreground_variant or variant.image
        background_variant = variant.background_variant or variant.background
        composited = Image.alpha_composite(background_variant, foreground_variant)
        composited = apply_alpha(composited, alpha_map)
        if variant.rotation and self.layered_pbr_config.get("rotation_after_composition", True):
            composited = self._rotate_layer_image(composited, variant.rotation, Image.BILINEAR)
            rotated_foreground = self._rotate_layer_image(foreground_variant, variant.rotation, Image.BILINEAR)
            rotated_background = self._rotate_layer_image(background_variant, variant.rotation, Image.BILINEAR)
        else:
            rotated_foreground = foreground_variant
            rotated_background = background_variant
        target_path = self.output_path / f"{base_name}.png"
        self.file_manager.atomic_save(composited, target_path)
        for map_name, map_image in layered_maps.items():
            map_path = self.output_path / f"{base_name}_{map_name}.png"
            self.file_manager.atomic_save(map_image, map_path)
        if self.layered_pbr_config.get("save_component_images", False):
            fg_path = self.output_path / f"{base_name}_foreground.png"
            bg_path = self.output_path / f"{base_name}_background.png"
            self.file_manager.atomic_save(rotated_foreground, fg_path)
            self.file_manager.atomic_save(rotated_background, bg_path)

    def _generate_maps(
        self, base_image: Image.Image, background: Image.Image
    ) -> tuple[Dict[str, Image.Image], np.ndarray]:
        baseline: Dict[str, Image.Image] = {}
        for category, generators in self.map_generators.items():
            for map_name, generator in generators.items():
                try:
                    baseline[map_name] = generator(base_image)
                except Exception as exc:  # pragma: no cover - logging path
                    self.logger.exception("Failed to generate %s map: %s", map_name, exc)
        pbr_result = generate_physically_accurate_pbr_maps(base_image, background, baseline)
        final_maps: Dict[str, Image.Image] = pbr_result["maps"]
        analysis = pbr_result.get("analysis")
        analysis_obj = analysis if hasattr(analysis, "mask") else None
        alpha_map = pbr_result.get("alpha")
        if isinstance(alpha_map, np.ndarray):
            derived_alpha = alpha_map
        else:
            derived_alpha = derive_alpha_map(base_image, final_maps, analysis_obj)
        quality = pbr_result.get("quality_report")
        if quality is None:
            quality = generate_quality_report(final_maps, analysis_obj)
        self.logger.debug("PBR quality report: %s", quality)
        return final_maps, derived_alpha

    def _generate_maps_layered(
        self,
        base_image: Image.Image,
        background: Image.Image,
        rotation: float = 0.0,
    ) -> tuple[Dict[str, Image.Image], np.ndarray]:
        if self.pbr_layer_manager is None:
            raise RuntimeError("Layered PBR manager is not configured")
        maps = self.pbr_layer_manager.generate_layered_pbr_maps(base_image, background)
        alpha_map = derive_alpha_map(base_image, maps, analysis=None)
        maps = apply_alpha_to_maps(maps, alpha_map)
        if rotation and self.layered_pbr_config.get("rotation_after_composition", True):
            maps = self._rotate_pbr_maps_layered(maps, rotation)
        return maps, alpha_map

    def _rotate_pbr_maps_layered(
        self, pbr_maps: Mapping[str, Image.Image], rotation: float
    ) -> Dict[str, Image.Image]:
        rotated: Dict[str, Image.Image] = {}
        for map_name, map_image in pbr_maps.items():
            resample = self._select_pbr_rotation_resample(map_name)
            rotated_map = map_image.rotate(
                rotation,
                resample=resample,
                expand=True,
                fillcolor=(0, 0, 0, 0),
            )
            rotated[map_name] = rotated_map
        return rotated

    @staticmethod
    def _select_pbr_rotation_resample(map_name: str) -> int:
        map_name_lower = map_name.lower()
        if map_name_lower in {"normal", "height", "curvature"}:
            return Image.BICUBIC
        if map_name_lower in {"metallic", "roughness", "specular"}:
            return Image.NEAREST
        if map_name_lower in {"albedo", "base_color"}:
            return Image.BILINEAR
        return Image.BILINEAR

    @staticmethod
    def _rotate_layer_image(image: Image.Image, rotation: float, resample: int) -> Image.Image:
        return image.rotate(rotation, resample=resample, expand=True, fillcolor=(0, 0, 0, 0))


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    cfg = config.build_config()
    pipeline = RecolorPipeline(cfg)
    pipeline.run(threads=cfg.get("THREADS", 4))
