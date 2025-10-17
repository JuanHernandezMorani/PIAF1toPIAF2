"""Core recoloring pipeline that orchestrates image variants and map generation."""
from __future__ import annotations

import importlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps

from ..core import config
from ..core.utils_color import (
    add_color_noise,
    color_distance,
    generate_high_variation_colors,
    mix_colors,
    random_palette,
    rgb_to_hsl,
    hsl_to_rgb,
)
from ..core.utils_io import SafeFileManager, ensure_dir
from ..core.utils_parallel import limited_threads, run_parallel

LOGGER = logging.getLogger("pixel_pipeline.recolor")

GeneratorFn = Callable[[Image.Image], Image.Image]

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


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


class RecolorPipeline:
    """Generate recolored variants and derived maps for input sprites."""

    def __init__(self, cfg: Dict[str, object]) -> None:
        self.config = cfg
        self.input_path = Path(cfg["PATH_INPUT"])
        self.output_path = Path(cfg["PATH_OUTPUT"])
        self.background_path = Path(cfg["PATH_BACKGROUNDS"])
        ensure_dir(self.output_path)
        self.file_manager = SafeFileManager(self.output_path)
        self.rotation_angles: Iterable[int] = cfg.get("ROTATION_ANGLES", (0,))  # type: ignore[assignment]
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
        with limited_threads(threads):
            run_parallel(self._process_file, files, max_workers=threads)

    # Variant generation -------------------------------------------------
    def _process_file(self, path: Path) -> None:
        self.logger.info("Generating variants for %s", path.name)
        try:
            with Image.open(path) as img:
                rgba = img.convert("RGBA")
        except Exception as exc:  # pragma: no cover - logging path
            self.logger.exception("Failed to open %s: %s", path, exc)
            return

        variants = list(self._generate_variants(path.stem, rgba))
        for variant in variants[: self.max_variants]:
            self._persist_variant(path.stem, variant)

    def _generate_variants(self, stem: str, image: Image.Image) -> Iterator[Variant]:
        palette = self._extract_palette(image)
        variant_index = 0
        for rotation in self.rotation_angles:
            rotated = image.rotate(rotation, resample=Image.NEAREST, expand=False)
            for tint in palette:
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
        for _, color in colors_sorted:
            rgb = color[:3]
            if all(color_distance(rgb, existing) > 12 for existing in dominant):
                dominant.append(rgb)
            if len(dominant) >= 8:
                break
        if len(dominant) < 4:
            dominant.extend(random_palette(self.rng.randint(0, 1000000), size=4 - len(dominant)))
        variants: List[Tuple[int, int, int]] = []
        for base in dominant:
            mixed = mix_colors(base, add_color_noise(base, 0.2), 0.5)
            variants.append(mixed)
            variants.extend(generate_high_variation_colors(base, 2, 0.7))
        return variants

    def _apply_tint(self, image: Image.Image, tint: Tuple[int, int, int]) -> Image.Image:
        rgba = image.convert("RGBA")
        arr = np.asarray(rgba, dtype=np.float32)
        alpha = arr[..., 3:4] / 255.0
        tinted_color = add_color_noise(tint, 0.12)
        tint_arr = np.array(tinted_color, dtype=np.float32)
        arr[..., :3] = arr[..., :3] * (1 - 0.35) + tint_arr * 0.35
        arr[..., :3] = np.clip(arr[..., :3], 0, 255)
        arr[..., 3:4] = alpha * 255.0
        tinted = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        brightness = self.rng.uniform(0.8, 1.2)
        contrast = self.rng.uniform(0.8, 1.3)
        tinted = ImageEnhance.Brightness(tinted).enhance(brightness)
        tinted = ImageEnhance.Contrast(tinted).enhance(contrast)
        return tinted

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
