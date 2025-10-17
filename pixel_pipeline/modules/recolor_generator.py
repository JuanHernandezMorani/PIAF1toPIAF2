"""Core recoloring pipeline that orchestrates image variants and map generation."""
from __future__ import annotations

import importlib
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps

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
        self._stats_lock = threading.Lock()
        self._images_processed = 0
        self._variants_generated = 0
        self._total_image_time = 0.0

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
        for rotation in self.rotation_angles:
            if variant_index >= self.max_variants:
                return
            rotated = image.rotate(rotation, resample=Image.NEAREST, expand=False)
            for tint in palette:
                if variant_index >= self.max_variants:
                    return
                variant_image = self._apply_tint(rotated, tint)
                adapted = self._adapt_background(variant_image)
                name = f"{stem}_r{rotation:03d}_{variant_index:04d}"
                variant_index += 1
                yield Variant(name=name, image=variant_image, background=adapted)

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
        foreground = alpha > 1e-6
        combined_rgb = np.where(foreground, varied_rgb, rgb)
        combined = np.dstack((combined_rgb, alpha))
        tinted = Image.fromarray(np.clip(combined * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGBA")
        tinted = contextual_randomizer.apply_global_recolor(tinted, self.rng)
        with contextual_randomizer.pixel_variation_callback(self._apply_pixel_variation):
            tinted = contextual_randomizer.integrate_contextual_variation(tinted, self.rng)
        brightness = self.rng.uniform(0.8, 1.2)
        contrast = self.rng.uniform(0.8, 1.3)
        tinted = ImageEnhance.Brightness(tinted).enhance(brightness)
        tinted = ImageEnhance.Contrast(tinted).enhance(contrast)
        return tinted

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
        base_name = variant.name
        composited = Image.alpha_composite(variant.background, variant.image)
        target_path = self.output_path / f"{base_name}.png"
        self.file_manager.atomic_save(composited, target_path)
        maps = self._generate_maps(composited, variant.background)
        for map_name, map_image in maps.items():
            map_path = self.output_path / f"{base_name}_{map_name}.png"
            self.file_manager.atomic_save(map_image, map_path)

    def _generate_maps(self, base_image: Image.Image, background: Image.Image) -> Dict[str, Image.Image]:
        results: Dict[str, Image.Image] = {}
        for category, generators in self.map_generators.items():
            for map_name, generator in generators.items():
                try:
                    generated = generator(base_image)
                except Exception as exc:  # pragma: no cover - logging path
                    self.logger.exception("Failed to generate %s map: %s", map_name, exc)
                    continue
                composite = Image.alpha_composite(background, generated) if background else generated
                results[map_name] = composite
        return results


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    cfg = config.build_config()
    pipeline = RecolorPipeline(cfg)
    pipeline.run(threads=cfg.get("THREADS", 4))
