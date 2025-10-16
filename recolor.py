"""Utilities for generating color variants from input images.

This module focuses on reliable and Windows friendly file handling while
providing a simple recoloring pipeline for image assets.  The original script
had grown organically and pulled many optional dependencies that were not
strictly necessary.  The refactor keeps the public entry point as a script
(`python recolor.py`) but trims the implementation down to a maintainable,
well-tested subset with explicit imports and guarded filesystem operations.

Highlights of the refactor
--------------------------
* All imports are explicit and organised at the top of the file.
* The configuration for the pipeline is stored in a dataclass for clarity.
* File saving uses temporary files and ``os.replace`` to avoid permission
  errors on Windows while guaranteeing atomic writes on POSIX platforms.
* Saving retries with exponential backoff mitigate transient ``PermissionError``
  exceptions that are common on Windows due to virus scanners or indexing
  services keeping file handles open momentarily.
* The code base no longer depends on heavyweight libraries that were not used
  by the project.
* Logging has been simplified, and the command line interface gives quick
  feedback about what the pipeline is doing.

The module can still be imported by other Python files.  The main orchestrator
is :func:`run_pipeline`, which receives a :class:`PipelineConfig`.  When executed
as a script the arguments can be customised through command line flags.  The
recoloring approach is intentionally conservative: each variant applies a colour
overlay combined with brightness and saturation adjustments.  This covers the
requirements of generating synthetic colour variants for datasets without the
complexity of the original tool.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from PIL import Image, ImageColor, ImageEnhance

# Pillow does not enable loading truncated images by default.  Allow it so the
# pipeline is robust against partially downloaded assets.  The import is local
# to prevent polluting the module namespace when Pillow is not available during
# static analysis tools.
try:  # pragma: no cover - guarded import for optional attribute
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:  # pragma: no cover - we can continue without this tweak
    pass


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------


def _default_cpu_workers() -> int:
    """Return the number of workers to use by default."""

    detected = os.cpu_count() or 1
    return max(1, min(8, detected))  # avoid spawning hundreds of threads


@dataclass(slots=True)
class PipelineConfig:
    """Runtime configuration for the recoloring pipeline."""

    input_dir: Path = Path("input")
    output_dir: Path = Path("variants")
    palette_file: Path = Path("colors.json")
    max_variants_per_image: int = 12
    min_colors_in_palette: int = 1
    brightness_range: Tuple[float, float] = (0.85, 1.25)
    saturation_range: Tuple[float, float] = (0.8, 1.4)
    alpha_range: Tuple[float, float] = (0.25, 0.6)
    valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    random_seed: Optional[int] = None
    workers: int = field(default_factory=_default_cpu_workers)

    def normalise(self) -> None:
        """Normalise paths and validate numeric ranges."""

        self.input_dir = self.input_dir.resolve()
        self.output_dir = self.output_dir.resolve()
        self.palette_file = self.palette_file.resolve()

        if self.max_variants_per_image <= 0:
            raise ValueError("max_variants_per_image must be positive")
        if self.min_colors_in_palette <= 0:
            raise ValueError("min_colors_in_palette must be positive")
        if self.alpha_range[0] <= 0 or self.alpha_range[1] > 1:
            raise ValueError("alpha_range values must be in (0, 1]")
        if self.alpha_range[0] >= self.alpha_range[1]:
            raise ValueError("alpha_range lower bound must be < upper bound")
        if self.brightness_range[0] <= 0:
            raise ValueError("brightness lower bound must be > 0")
        if self.brightness_range[0] >= self.brightness_range[1]:
            raise ValueError("brightness_range lower bound must be < upper bound")
        if self.saturation_range[0] <= 0:
            raise ValueError("saturation lower bound must be > 0")
        if self.saturation_range[0] >= self.saturation_range[1]:
            raise ValueError("saturation_range lower bound must be < upper bound")
        if self.workers <= 0:
            raise ValueError("workers must be a positive integer")

    @classmethod
    def from_args(cls, args: Sequence[str]) -> "PipelineConfig":
        """Create a configuration from command line arguments."""

        parser = argparse.ArgumentParser(description="Generate recoloured variants")
        parser.add_argument("--input", dest="input_dir", default="input", help="Directory with the source images")
        parser.add_argument(
            "--output",
            dest="output_dir",
            default="variants",
            help="Directory where variants will be stored",
        )
        parser.add_argument(
            "--palette",
            dest="palette_file",
            default="colors.json",
            help="JSON file describing the colour palette",
        )
        parser.add_argument(
            "--max-variants",
            dest="max_variants_per_image",
            type=int,
            default=12,
            help="Maximum number of variants to produce per image",
        )
        parser.add_argument(
            "--brightness",
            dest="brightness_range",
            nargs=2,
            type=float,
            metavar=("MIN", "MAX"),
            default=(0.85, 1.25),
            help="Brightness multiplier range",
        )
        parser.add_argument(
            "--saturation",
            dest="saturation_range",
            nargs=2,
            type=float,
            metavar=("MIN", "MAX"),
            default=(0.8, 1.4),
            help="Saturation multiplier range",
        )
        parser.add_argument(
            "--alpha",
            dest="alpha_range",
            nargs=2,
            type=float,
            metavar=("MIN", "MAX"),
            default=(0.25, 0.6),
            help="Opacity of the colour overlay (0-1)",
        )
        parser.add_argument(
            "--seed",
            dest="random_seed",
            type=int,
            default=None,
            help="Optional random seed for reproducible variants",
        )
        parser.add_argument(
            "--workers",
            dest="workers",
            type=int,
            default=_default_cpu_workers(),
            help="Number of worker threads to use",
        )

        namespace = parser.parse_args(args)
        config = cls(
            input_dir=Path(namespace.input_dir),
            output_dir=Path(namespace.output_dir),
            palette_file=Path(namespace.palette_file),
            max_variants_per_image=namespace.max_variants_per_image,
            brightness_range=tuple(namespace.brightness_range),
            saturation_range=tuple(namespace.saturation_range),
            alpha_range=tuple(namespace.alpha_range),
            random_seed=namespace.random_seed,
            workers=namespace.workers,
        )
        config.normalise()
        return config


# ---------------------------------------------------------------------------
# Palette handling
# ---------------------------------------------------------------------------

DEFAULT_PALETTE: Tuple[str, ...] = (
    "#EC407A",
    "#AB47BC",
    "#42A5F5",
    "#26C6DA",
    "#66BB6A",
    "#FFCA28",
    "#FF7043",
)


def _flatten_palette(data: object) -> Iterable[str]:
    """Yield colour strings from diverse JSON representations."""

    if isinstance(data, str):
        yield data
    elif isinstance(data, dict):
        for value in data.values():
            yield from _flatten_palette(value)
    elif isinstance(data, (list, tuple, set)):
        for entry in data:
            yield from _flatten_palette(entry)


def load_palette(palette_file: Path, min_colors: int) -> List[Tuple[int, int, int]]:
    """Load a palette from a JSON file or use the default one.

    The JSON file can contain a list of hex strings, dictionaries with lists or
    any nested structure.  Invalid values are ignored with a warning.  When the
    file is missing or empty, :data:`DEFAULT_PALETTE` is used.
    """

    if not palette_file.exists():
        logging.warning("Palette file %s does not exist. Using default palette.", palette_file)
        colour_strings = list(DEFAULT_PALETTE)
    else:
        try:
            raw = json.loads(palette_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logging.error("Could not parse palette file %s: %s", palette_file, exc)
            colour_strings = list(DEFAULT_PALETTE)
        else:
            colour_strings = [colour for colour in _flatten_palette(raw)]
            if not colour_strings:
                colour_strings = list(DEFAULT_PALETTE)

    palette: List[Tuple[int, int, int]] = []
    for colour in colour_strings:
        try:
            palette.append(ImageColor.getrgb(str(colour)))
        except ValueError:
            logging.debug("Ignoring unknown colour entry: %s", colour)

    if len(palette) < min_colors:
        logging.warning(
            "Palette contains %d colours but %d are required. Extending with defaults.",
            len(palette),
            min_colors,
        )
        for fallback in DEFAULT_PALETTE:
            rgb = ImageColor.getrgb(fallback)
            palette.append(rgb)
            if len(palette) >= min_colors:
                break

    return palette[: max(len(palette), min_colors)]


# ---------------------------------------------------------------------------
# Image loading and transformation utilities
# ---------------------------------------------------------------------------


def load_image(path: Path) -> Image.Image:
    """Load an image from disk ensuring the file handle is closed on Windows."""

    try:
        with Image.open(path) as img:
            converted = img.convert("RGBA")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input image not found: {path}") from exc
    except PermissionError as exc:
        raise PermissionError(f"Permission denied when reading {path}") from exc
    except OSError as exc:
        raise OSError(f"Could not read image {path}: {exc}") from exc

    return converted


class VariantBuilder:
    """Create image variants using colour overlays and adjustments."""

    def __init__(
        self,
        palette: Sequence[Tuple[int, int, int]],
        brightness_range: Tuple[float, float],
        saturation_range: Tuple[float, float],
        alpha_range: Tuple[float, float],
        rng: random.Random,
    ) -> None:
        self.palette = palette
        self.brightness_range = brightness_range
        self.saturation_range = saturation_range
        self.alpha_range = alpha_range
        self.rng = rng

    def _random_colour(self) -> Tuple[int, int, int]:
        return self.rng.choice(self.palette)

    def _random_brightness(self) -> float:
        return self.rng.uniform(*self.brightness_range)

    def _random_saturation(self) -> float:
        return self.rng.uniform(*self.saturation_range)

    def _random_alpha(self) -> float:
        return self.rng.uniform(*self.alpha_range)

    def _apply_overlay(
        self,
        base_image: Image.Image,
        colour: Tuple[int, int, int],
        alpha: float,
    ) -> Image.Image:
        base = base_image.convert("RGBA")
        overlay_alpha = max(0, min(255, int(alpha * 255)))
        overlay = Image.new("RGBA", base.size, (*colour, overlay_alpha))
        blended = Image.alpha_composite(base, overlay)
        return blended

    def _enhance(self, image: Image.Image, brightness: float, saturation: float) -> Image.Image:
        result = ImageEnhance.Brightness(image).enhance(brightness)
        result = ImageEnhance.Color(result).enhance(saturation)
        return result

    def build_variants(self, image: Image.Image, count: int) -> Iterator[Tuple[Image.Image, Tuple[int, int, int]]]:
        for _ in range(count):
            colour = self._random_colour()
            alpha = self._random_alpha()
            brightness = self._random_brightness()
            saturation = self._random_saturation()
            tinted = self._apply_overlay(image, colour, alpha)
            enhanced = self._enhance(tinted, brightness, saturation)
            yield enhanced.convert("RGB"), colour


# ---------------------------------------------------------------------------
# Safe file writing utilities
# ---------------------------------------------------------------------------


class WindowsSafeImageWriter:
    """Save images atomically with retries to avoid Windows permission errors."""

    def __init__(self, output_dir: Path, max_retries: int = 5, base_delay: float = 0.2) -> None:
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._lock = threading.Lock()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_destination(self, relative_path: Path) -> Path:
        relative_path = Path(relative_path)
        if relative_path.is_absolute():
            raise ValueError("relative_path must be relative to the output directory")
        final_path = (self.output_dir / relative_path).resolve()
        if self.output_dir not in final_path.parents and final_path != self.output_dir:
            raise ValueError("Attempted to escape the output directory")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            final_path = final_path.with_suffix(".png")
        return final_path

    def save(self, image: Image.Image, relative_path: Path) -> Path:
        final_path = self._resolve_destination(relative_path)
        format_hint = final_path.suffix.lstrip(".").upper()
        if format_hint == "JPG":
            format_hint = "JPEG"

        with self._lock:
            for attempt in range(self.max_retries):
                temp_file: Optional[Path] = None
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        dir=str(final_path.parent),
                        prefix=f"{final_path.stem}_",
                        suffix=".tmp",
                    ) as handle:
                        temp_file = Path(handle.name)
                    image.save(temp_file, format=format_hint)
                    os.replace(temp_file, final_path)
                    logging.debug("Saved variant to %s", final_path)
                    return final_path
                except PermissionError as exc:
                    delay = self.base_delay * (2**attempt)
                    logging.warning(
                        "PermissionError while saving %s (%s). Retrying in %.2fs.",
                        final_path,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                except OSError as exc:
                    logging.error("Failed to save %s: %s", final_path, exc)
                    raise
                finally:
                    if temp_file is not None:
                        with suppress(FileNotFoundError):
                            temp_file.unlink()

        raise PermissionError(f"Could not save {final_path} after {self.max_retries} attempts")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VariantTask:
    image_path: Path
    relative_directory: Path


def discover_images(config: PipelineConfig) -> List[VariantTask]:
    tasks: List[VariantTask] = []
    for file_path in config.input_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in config.valid_extensions:
            continue
        try:
            relative = file_path.relative_to(config.input_dir)
        except ValueError:
            relative = Path(file_path.name)
        tasks.append(VariantTask(image_path=file_path, relative_directory=relative.parent))
    return tasks


def _colour_to_suffix(colour: Tuple[int, int, int]) -> str:
    return f"{colour[0]:02X}{colour[1]:02X}{colour[2]:02X}"


def process_task(
    task: VariantTask,
    builder: VariantBuilder,
    writer: WindowsSafeImageWriter,
    config: PipelineConfig,
) -> None:
    try:
        image = load_image(task.image_path)
    except Exception as exc:
        logging.error("Skipping %s: %s", task.image_path, exc)
        return

    variants = builder.build_variants(image, config.max_variants_per_image)
    base_name = task.image_path.stem
    for index, (variant, colour) in enumerate(variants, start=1):
        suffix = _colour_to_suffix(colour)
        filename = f"{base_name}_{suffix}_{index:02d}.png"
        relative_path = task.relative_directory / filename
        try:
            writer.save(variant, relative_path)
        except PermissionError as exc:
            logging.error("Permission error saving %s: %s", relative_path, exc)
        except OSError as exc:
            logging.error("Could not save %s: %s", relative_path, exc)


def run_pipeline(config: PipelineConfig) -> None:
    config.normalise()
    if not config.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {config.input_dir}")

    palette = load_palette(config.palette_file, config.min_colors_in_palette)
    rng = random.Random(config.random_seed)
    builder = VariantBuilder(
        palette=palette,
        brightness_range=config.brightness_range,
        saturation_range=config.saturation_range,
        alpha_range=config.alpha_range,
        rng=rng,
    )

    writer = WindowsSafeImageWriter(config.output_dir)
    tasks = discover_images(config)
    if not tasks:
        logging.info("No images were found in %s", config.input_dir)
        return

    logging.info("Processing %d images using %d workers", len(tasks), config.workers)

    if config.workers == 1:
        for task in tasks:
            process_task(task, builder, writer, config)
    else:
        from concurrent.futures import ThreadPoolExecutor, wait

        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures = [
                executor.submit(process_task, task, builder, writer, config)
                for task in tasks
            ]
            wait(futures)

    logging.info("Finished generating variants in %s", config.output_dir)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    argv = list(argv or sys.argv[1:])
    try:
        config = PipelineConfig.from_args(argv)
        run_pipeline(config)
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
        return 1
    except Exception as exc:
        logging.error("Fatal error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
