from __future__ import annotations

import json
import logging
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageColor, ImageEnhance, ImageFilter, ImageOps


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("recolor")


CONFIG: Dict[str, object] = {
    "MIN_COLOR": 100,
    "INPUT_DIR": "input",
    "OUTPUT_DIR": "variants",
    "COLORS_JSON": "colors.json",
    "MAX_DOMINANT_COLORS": 30,
    "MAX_VARIATIONS_PER_COLOR": 40,
    "MIN_VARIATIONS_PER_COLOR": 15,
    "COLOR_RANGE": {"min": 5, "max": 253},
    "ALPHA_RANGE": {"min": 0.08, "max": 0.98},
    "INTENSITY_VARIATION_RANGE": {"min": 0.5, "max": 1.8},
    "SAMPLING_THRESHOLD": 200000,
    "BUCKET_SIZE": 28,
    "VALID_EXTENSIONS": {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"},
    "USE_REAL_BACKGROUNDS_ONLY": True,
    "BACKGROUNDS_DIR": "backgrounds",
    "PRESERVE_BRIGHTNESS": False,
    "USE_HLS_METHOD": True,
    "GENERATE_ROTATIONS": True,
    "ROTATION_ANGLES": [0, 90, 180, 270],
    "MIN_COLOR_DIFFERENCE": 8.0,
    "HUE_VARIATION_RANGE": [-0.3, 0.3],
    "SATURATION_VARIATION_RANGE": [-0.7, 0.7],
    "LIGHTNESS_VARIATION_RANGE": [-0.5, 0.5],
    "MAX_GENERATION_ATTEMPTS": 800,
    "SIMILARITY_THRESHOLD": 0.85,
    "ENSURE_MIN_VARIATIONS": True,
    "RANDOM_COLOR_PROBABILITY": 0.3,
    "CROSS_COLOR_MIXING": True,
    "MAX_COLOR_MIXES": 256,
    "NOISE_INTENSITY": 0.08,
    "MAX_WORKERS": 12,
    "CHUNK_SIZE": 2000,
    "MEMORY_MONITORING": False,
    "AUTO_ADJUST_CHUNKS": False,
    "TARGET_MEMORY_GB": 13.5,
    "MAX_MEMORY_PERCENT": 94,
    "MEMORY_PER_IMAGE_MB": 3.2,
    "PARALLEL_ROTATIONS": True,
    "PARALLEL_COLORS": True,
    "MAX_ROTATION_WORKERS": 4,
    "MAX_COLOR_WORKERS": 8,
    "PERCEPTUAL_VARIANT_SYSTEM": {
        "TOTAL_VARIANTS": 500,
        "VARIANTS_PER_GROUP": 25,
        "VARIANT_GROUPS": {
            "base": {"enabled": True, "description": "Línea base estándar"},
            "low_contrast": {"enabled": True, "contrast_range": [0.3, 0.5]},
            "high_contrast": {"enabled": True, "contrast_range": [1.8, 2.2]},
            "illumination": {
                "enabled": True,
                "brightness_range": [0.4, 1.8],
                "contrast_range": [0.6, 1.4],
            },
            "concise_colors": {
                "enabled": True,
                "color_tolerance": 12,
                "palette_size": 8,
            },
        },
        "HUMAN_VISION_PARAMS": {
            "WEBER_FRACTION": 0.02,
            "CONTRAST_SENSITIVITY": {"low": 0.01, "high": 0.1},
            "COLOR_ADAPTATION_SPEED": 0.3,
            "LUMINANCE_PRESERVATION": True,
        },
        "DYNAMIC_BACKGROUNDS": {
            "ENABLED": True,
            "BRIGHTNESS_RANGE": [0.7, 1.3],
            "CONTRAST_RANGE": [0.6, 1.6],
            "ALPHA_RANGE": [0.65, 1.0],
            "HUE_SHIFT_RANGE": [-15, 15],
            "NOISE_LEVEL": 0.03,
            "TEXTURE_VARIATION": True,
        },
    },
}


@dataclass
class VariantResult:
    image: Image.Image
    normal: Image.Image
    specular: Image.Image
    roughness: Image.Image
    emissive: Image.Image
    suffix: str


class BackgroundLibrary:
    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        self.backgrounds: List[Path] = []
        self._load_backgrounds()

    def _load_backgrounds(self) -> None:
        if not self.config.get("USE_REAL_BACKGROUNDS_ONLY", True):
            return
        bg_dir = Path(self.config["BACKGROUNDS_DIR"]).expanduser().resolve()
        if not bg_dir.exists():
            LOGGER.warning("Background directory %s does not exist", bg_dir)
            return
        for entry in sorted(bg_dir.iterdir()):
            if entry.suffix.lower() in self.config["VALID_EXTENSIONS"]:
                self.backgrounds.append(entry)
        if not self.backgrounds:
            LOGGER.warning("No background images found in %s", bg_dir)

    def get_random_background(self, size: Tuple[int, int]) -> Image.Image:
        if not self.backgrounds:
            raise RuntimeError("No real backgrounds available")
        path = random.choice(self.backgrounds)
        with Image.open(path) as img:
            img = img.convert("RGBA")
            resized = ImageOps.fit(img, size, method=Image.BICUBIC)
        return self._apply_dynamic_background_effects(resized)

    def _apply_dynamic_background_effects(self, background: Image.Image) -> Image.Image:
        dynamic_cfg = self.config["PERCEPTUAL_VARIANT_SYSTEM"]["DYNAMIC_BACKGROUNDS"]
        if not dynamic_cfg.get("ENABLED", False):
            return background
        bg = background.copy()
        brightness = random.uniform(*dynamic_cfg["BRIGHTNESS_RANGE"])
        contrast = random.uniform(*dynamic_cfg["CONTRAST_RANGE"])
        alpha_range = dynamic_cfg["ALPHA_RANGE"]
        hue_shift = random.uniform(*dynamic_cfg["HUE_SHIFT_RANGE"])
        noise_level = dynamic_cfg["NOISE_LEVEL"]

        bg = ImageEnhance.Brightness(bg).enhance(brightness)
        bg = ImageEnhance.Contrast(bg).enhance(contrast)

        if hue_shift != 0:
            bg = shift_hue(bg, hue_shift)

        alpha = bg.split()[-1].point(lambda a: int(a * random.uniform(*alpha_range)))
        bg.putalpha(alpha)

        if dynamic_cfg.get("TEXTURE_VARIATION", False):
            texture = bg.filter(ImageFilter.DETAIL)
            bg = ImageChops.add(bg, texture, scale=1.0, offset=0)

        if noise_level > 0:
            bg = add_noise(bg, noise_level)

        return bg


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_palette(path: Path) -> List[Tuple[int, int, int]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Could not parse palette JSON %s: %s", path, exc)
        return []
    palette: List[Tuple[int, int, int]] = []
    if isinstance(data, dict):
        for value in data.values():
            palette.extend(_parse_palette_value(value))
    elif isinstance(data, list):
        palette.extend(_parse_palette_value(data))
    return palette


def _parse_palette_value(value: object) -> List[Tuple[int, int, int]]:
    colors: List[Tuple[int, int, int]] = []
    if isinstance(value, str):
        colors.append(ImageColor.getrgb(value))
    elif isinstance(value, Sequence):
        for item in value:
            colors.extend(_parse_palette_value(item))
    elif isinstance(value, dict):
        for item in value.values():
            colors.extend(_parse_palette_value(item))
    return colors


def image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def numpy_to_image(array: np.ndarray) -> Image.Image:
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGBA")


def apply_alpha_overlay(base: Image.Image, color: Tuple[int, int, int], alpha: float) -> Image.Image:
    overlay = Image.new("RGBA", base.size, (*color, int(alpha * 255)))
    return Image.alpha_composite(base, overlay)


def shift_hue(image: Image.Image, degrees: float) -> Image.Image:
    if degrees == 0:
        return image
    hsv = image.convert("HSV")
    h, s, v = hsv.split()
    np_h = (np.array(h, dtype=np.uint16) + int(degrees / 360.0 * 255)) % 255
    h = Image.fromarray(np_h.astype(np.uint8), "L")
    merged = Image.merge("HSV", (h, s, v))
    return merged.convert("RGBA")


def add_noise(image: Image.Image, intensity: float) -> Image.Image:
    if intensity <= 0:
        return image
    arr = image_to_numpy(image)
    rgb = arr[..., :3]
    noise = np.random.normal(0, 255 * intensity, rgb.shape)
    rgb = np.clip(rgb + noise, 0, 255)
    arr[..., :3] = rgb
    return numpy_to_image(arr)


def enhance_with_alpha(image: Image.Image, enhancer, factor: float) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.split()[-1]
    base_rgb = rgba.convert("RGB")
    enhanced_rgb = enhancer(base_rgb).enhance(factor)
    enhanced_rgba = enhanced_rgb.convert("RGBA")
    enhanced_rgba.putalpha(alpha)
    return enhanced_rgba


def apply_group_adjustments(image: Image.Image, group_name: str, settings: Dict[str, object]) -> Image.Image:
    result = image
    if "contrast_range" in settings:
        low, high = (float(x) for x in settings["contrast_range"])
        factor = random.uniform(low, high)
        result = enhance_with_alpha(result, ImageEnhance.Contrast, factor)
    if "brightness_range" in settings:
        low, high = (float(x) for x in settings["brightness_range"])
        factor = random.uniform(low, high)
        result = enhance_with_alpha(result, ImageEnhance.Brightness, factor)
    if group_name == "concise_colors":
        palette_size = int(settings.get("palette_size", 8))
        tolerance = float(settings.get("color_tolerance", 12))
        rgba = result.convert("RGBA")
        alpha = rgba.split()[-1]
        base_rgb = rgba.convert("RGB")
        reduced_rgb = base_rgb.convert("P", palette=Image.ADAPTIVE, colors=int(palette_size)).convert("RGB")
        reduced = reduced_rgb.convert("RGBA")
        reduced.putalpha(alpha)
        if tolerance > 0:
            blur = result.filter(ImageFilter.MedianFilter(size=3))
            mix = ImageChops.blend(reduced, blur, 0.5)
            result = mix
        else:
            result = reduced
    return result


def match_luminance(reference: Image.Image, candidate: Image.Image) -> Image.Image:
    ref_np = image_to_numpy(reference)[..., :3]
    cand_np = image_to_numpy(candidate)[..., :3]
    ref_luminance = np.mean(ref_np)
    cand_luminance = np.mean(cand_np) + 1e-6
    scale = ref_luminance / cand_luminance
    arr = image_to_numpy(candidate)
    arr[..., :3] = np.clip(arr[..., :3] * scale, 0, 255)
    return numpy_to_image(arr)


def enforce_weber_threshold(reference: Image.Image, candidate: Image.Image, fraction: float) -> Image.Image:
    if fraction <= 0:
        return candidate
    ref = image_to_numpy(reference)
    cand = image_to_numpy(candidate)
    threshold = fraction * 255.0
    diff = cand[..., :3] - ref[..., :3]
    mask = np.abs(diff) < threshold
    adjustment = np.clip(ref[..., :3] + np.sign(diff) * threshold, 0, 255)
    cand[..., :3] = np.where(mask, adjustment, cand[..., :3])
    return numpy_to_image(cand)


def apply_contrast_sensitivity(image: Image.Image, sensitivity: Dict[str, object]) -> Image.Image:
    low = float(sensitivity.get("low", 0.0))
    high = float(sensitivity.get("high", 0.0))
    if low == 0 and high == 0:
        return image
    factor = 1.0 + random.uniform(min(low, high), max(low, high))
    return enhance_with_alpha(image, ImageEnhance.Contrast, factor)


def rgb_to_hls(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    l = (maxc + minc) / 2.0
    s = np.zeros_like(l)
    h = np.zeros_like(l)
    diff = maxc - minc
    nonzero = diff != 0
    s[nonzero] = np.where(
        l[nonzero] <= 0.5,
        diff[nonzero] / (maxc[nonzero] + minc[nonzero] + 1e-6),
        diff[nonzero] / (2.0 - maxc[nonzero] - minc[nonzero] + 1e-6),
    )
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[nonzero] = ((maxc[nonzero] - r[nonzero]) / (diff[nonzero] + 1e-6))
    gc[nonzero] = ((maxc[nonzero] - g[nonzero]) / (diff[nonzero] + 1e-6))
    bc[nonzero] = ((maxc[nonzero] - b[nonzero]) / (diff[nonzero] + 1e-6))
    h_indices = np.argmax(rgb, axis=-1)
    h[nonzero & (h_indices == 0)] = (bc[nonzero & (h_indices == 0)] - gc[nonzero & (h_indices == 0)])
    h[nonzero & (h_indices == 1)] = 2.0 + (rc[nonzero & (h_indices == 1)] - bc[nonzero & (h_indices == 1)])
    h[nonzero & (h_indices == 2)] = 4.0 + (gc[nonzero & (h_indices == 2)] - rc[nonzero & (h_indices == 2)])
    h = (h / 6.0) % 1.0
    return h, l, s


def hls_to_rgb(h: np.ndarray, l: np.ndarray, s: np.ndarray) -> np.ndarray:
    def hue_to_rgb(p: np.ndarray, q: np.ndarray, t: np.ndarray) -> np.ndarray:
        t = t % 1.0
        res = np.empty_like(t)
        res[(t < 1 / 6)] = p[(t < 1 / 6)] + (q[(t < 1 / 6)] - p[(t < 1 / 6)]) * 6 * t[(t < 1 / 6)]
        res[((t >= 1 / 6) & (t < 1 / 2))] = q[((t >= 1 / 6) & (t < 1 / 2))]
        res[((t >= 1 / 2) & (t < 2 / 3))] = p[((t >= 1 / 2) & (t < 2 / 3))] + (q[((t >= 1 / 2) & (t < 2 / 3))] - p[((t >= 1 / 2) & (t < 2 / 3))]) * (2 / 3 - t[((t >= 1 / 2) & (t < 2 / 3))]) * 6
        res[(t >= 2 / 3)] = p[(t >= 2 / 3)]
        return res

    q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q
    r = hue_to_rgb(p, q, h + 1 / 3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1 / 3)
    return np.stack([r, g, b], axis=-1)


def apply_color_model(
    image: Image.Image,
    color: Tuple[int, int, int],
    alpha: float,
    hue_shift: float,
    saturation_shift: float,
    lightness_shift: float,
    intensity_scale: float,
    preserve_brightness: bool,
    use_hls: bool,
) -> Image.Image:
    base = image.convert("RGBA")
    rgba = image_to_numpy(base)
    rgb = rgba[..., :3] / 255.0
    alpha_layer = rgba[..., 3:4] / 255.0
    overlay = np.array(color, dtype=np.float32) / 255.0

    blended = rgb * (1 - alpha) + overlay * alpha
    if use_hls:
        h, l, s = rgb_to_hls(blended)
        h = (h + hue_shift) % 1.0
        s = np.clip(s + saturation_shift, 0.0, 1.0)
        l = np.clip(l + lightness_shift, 0.0, 1.0)
        adjusted = hls_to_rgb(h, l, s)
    else:
        adjusted = blended
        adjusted = np.clip(adjusted * (1 + saturation_shift), 0.0, 1.0)
        adjusted = np.clip(adjusted + lightness_shift, 0.0, 1.0)

    if not preserve_brightness:
        adjusted = np.clip(adjusted * intensity_scale, 0.0, 1.0)

    out_rgb = adjusted * 255.0
    result = np.concatenate([out_rgb, alpha_layer * 255.0], axis=-1)
    return numpy_to_image(result)


def generate_height_map(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    if gray.width * gray.height > CONFIG["SAMPLING_THRESHOLD"]:
        gray = gray.resize(
            (
                max(1, gray.width // 2),
                max(1, gray.height // 2),
            ),
            resample=Image.BICUBIC,
        )
    return np.asarray(gray, dtype=np.float32) / 255.0


def generate_normal_map(image: Image.Image) -> Image.Image:
    height = generate_height_map(image)
    gy, gx = np.gradient(height)
    strength = 2.0
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height)
    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-6
    normal = np.stack(((nx / length + 1) * 0.5, (ny / length + 1) * 0.5, nz / length), axis=-1)
    normal = (normal * 255).astype(np.uint8)
    alpha = image.split()[-1]
    normal_img = Image.merge("RGBA", (
        Image.fromarray(normal[..., 0], "L"),
        Image.fromarray(normal[..., 1], "L"),
        Image.fromarray(normal[..., 2], "L"),
        alpha,
    ))
    return normal_img


def generate_specular_map(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    spec = ImageEnhance.Contrast(grayscale).enhance(1.5)
    spec = ImageEnhance.Brightness(spec).enhance(1.2)
    spec_rgba = Image.merge("RGBA", (spec, spec, spec, image.split()[-1]))
    return spec_rgba


def generate_roughness_map(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    blurred = grayscale.filter(ImageFilter.GaussianBlur(radius=2))
    inverted = ImageOps.invert(blurred)
    rough = Image.merge("RGBA", (inverted, inverted, inverted, image.split()[-1]))
    return rough


def generate_emissive_map(image: Image.Image) -> Image.Image:
    hsv = image.convert("HSV")
    h, s, v = hsv.split()
    boosted = ImageEnhance.Brightness(v).enhance(1.5)
    boosted = ImageEnhance.Contrast(boosted).enhance(1.2)
    emissive = Image.merge("RGBA", (boosted, boosted, boosted, image.split()[-1]))
    return emissive


def generate_pbr_maps(image: Image.Image) -> Dict[str, Image.Image]:
    return {
        "normal": generate_normal_map(image),
        "specular": generate_specular_map(image),
        "roughness": generate_roughness_map(image),
        "emissive": generate_emissive_map(image),
    }


def merge_pbr_maps(
    fg_maps: Dict[str, Image.Image],
    bg_maps: Dict[str, Image.Image],
    alpha_mask: Image.Image,
) -> Dict[str, Image.Image]:
    merged: Dict[str, Image.Image] = {}
    alpha_np = np.asarray(alpha_mask, dtype=np.float32) / 255.0
    alpha_np = np.expand_dims(alpha_np, axis=-1)
    for key in ("normal", "specular", "roughness", "emissive"):
        fg = image_to_numpy(fg_maps[key])
        bg = image_to_numpy(bg_maps[key])
        if fg.shape != bg.shape:
            bg_image = Image.fromarray(bg.astype(np.uint8), "RGBA")
            bg_image = ImageOps.fit(bg_image, fg_maps[key].size)
            bg = image_to_numpy(bg_image)
        combined = fg * alpha_np + bg * (1.0 - alpha_np)
        merged[key] = numpy_to_image(combined)
    return merged


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def clamp_color(color: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return tuple(int(max(0, min(255, component))) for component in color)


def extract_dominant_colors(image: Image.Image, config: Dict[str, object]) -> List[Tuple[int, int, int]]:
    max_colors = config["MAX_DOMINANT_COLORS"]
    img = image.convert("RGBA")
    if img.width * img.height > config["SAMPLING_THRESHOLD"]:
        factor = math.sqrt(config["SAMPLING_THRESHOLD"] / (img.width * img.height))
        new_size = (
            max(1, int(img.width * factor)),
            max(1, int(img.height * factor)),
        )
        img = img.resize(new_size, Image.BILINEAR)
    quantized = img.convert("RGB").quantize(colors=max_colors, method=Image.FASTOCTREE)
    palette = quantized.getpalette()
    color_counts = sorted(quantized.getcolors() or [], reverse=True)
    valid_colors: List[Tuple[int, int, int]] = []
    color_range = config["COLOR_RANGE"]
    min_count = config["MIN_COLOR"]
    for count, index in color_counts:
        if count < min_count:
            continue
        base_index = index * 3
        color = tuple(int(palette[base_index + offset]) for offset in range(3))
        if len(color) != 3:
            continue
        if any(channel < color_range["min"] or channel > color_range["max"] for channel in color):
            continue
        if not valid_colors or all(color_distance(color, existing) >= config["MIN_COLOR_DIFFERENCE"] for existing in valid_colors):
            valid_colors.append(color)
        if len(valid_colors) >= max_colors:
            break
    return valid_colors


def compose_variant(
    base_image: Image.Image,
    background: Image.Image,
    color: Tuple[int, int, int],
    config: Dict[str, object],
    group_name: str,
    group_settings: Dict[str, object],
    variation_index: int,
    attempt: int,
) -> VariantResult:
    hue_range = config["HUE_VARIATION_RANGE"]
    sat_range = config["SATURATION_VARIATION_RANGE"]
    light_range = config["LIGHTNESS_VARIATION_RANGE"]
    alpha_range = config["ALPHA_RANGE"]
    intensity_range = config["INTENSITY_VARIATION_RANGE"]

    alpha = random.uniform(alpha_range["min"], alpha_range["max"])
    hue_shift = random.uniform(*hue_range)
    sat_shift = random.uniform(*sat_range)
    light_shift = random.uniform(*light_range)
    intensity = random.uniform(intensity_range["min"], intensity_range["max"])

    variant = apply_color_model(
        base_image,
        color,
        alpha,
        hue_shift,
        sat_shift,
        light_shift,
        intensity,
        preserve_brightness=config["PRESERVE_BRIGHTNESS"],
        use_hls=config["USE_HLS_METHOD"],
    )
    variant = apply_group_adjustments(variant, group_name, group_settings)
    human_params = config["PERCEPTUAL_VARIANT_SYSTEM"]["HUMAN_VISION_PARAMS"]
    adaptation_speed = float(human_params.get("COLOR_ADAPTATION_SPEED", 0.0))
    if adaptation_speed:
        variant = ImageChops.blend(base_image, variant, adaptation_speed)
    contrast_sensitivity = human_params.get("CONTRAST_SENSITIVITY", {})
    if isinstance(contrast_sensitivity, dict) and contrast_sensitivity:
        variant = apply_contrast_sensitivity(variant, contrast_sensitivity)
    weber_fraction = float(human_params.get("WEBER_FRACTION", 0.0))
    if weber_fraction:
        variant = enforce_weber_threshold(base_image, variant, weber_fraction)
    if human_params.get("LUMINANCE_PRESERVATION", False):
        variant = match_luminance(base_image, variant)
    noise_level = config["NOISE_INTENSITY"]
    if noise_level:
        variant = add_noise(variant, noise_level)

    fg_maps = generate_pbr_maps(variant)
    bg_maps = generate_pbr_maps(background)

    composed = Image.alpha_composite(background, variant)
    alpha_mask = composed.split()[-1]

    merged_maps = merge_pbr_maps(fg_maps, bg_maps, alpha_mask)

    suffix = f"{group_name}_{variation_index:03d}_attempt{attempt:02d}"
    return VariantResult(image=composed, normal=merged_maps["normal"], specular=merged_maps["specular"], roughness=merged_maps["roughness"], emissive=merged_maps["emissive"], suffix=suffix)


def image_similarity(a: Image.Image, b: Image.Image) -> float:
    size = (64, 64)
    arr_a = np.asarray(a.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    arr_b = np.asarray(b.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    arr_a = arr_a.flatten()
    arr_b = arr_b.flatten()
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    similarity = np.dot(arr_a, arr_b) / (norm_a * norm_b)
    return similarity


def rotation_worker(result: VariantResult, angles: Sequence[int]) -> List[VariantResult]:
    outputs: List[VariantResult] = []
    for angle in angles:
        if angle % 360 == 0:
            continue
        rotated_image = result.image.rotate(angle, expand=True)
        rotated_normal = result.normal.rotate(angle, expand=True)
        rotated_specular = result.specular.rotate(angle, expand=True)
        rotated_roughness = result.roughness.rotate(angle, expand=True)
        rotated_emissive = result.emissive.rotate(angle, expand=True)
        outputs.append(
            VariantResult(
                image=rotated_image,
                normal=rotated_normal,
                specular=rotated_specular,
                roughness=rotated_roughness,
                emissive=rotated_emissive,
                suffix=f"{result.suffix}_rot{angle:03d}",
            )
        )
    return outputs


class VariantAccumulator:
    def __init__(self, similarity_threshold: float) -> None:
        self.similarity_threshold = similarity_threshold
        self.references: List[Image.Image] = []
        self._lock = threading.Lock()

    def accept(self, candidate: Image.Image) -> bool:
        with self._lock:
            for ref in self.references:
                if image_similarity(ref, candidate) >= self.similarity_threshold:
                    return False
            self.references.append(candidate)
            return True


class RecolorPipeline:
    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        self.input_dir = Path(config["INPUT_DIR"]).expanduser().resolve()
        self.output_dir = Path(config["OUTPUT_DIR"]).expanduser().resolve()
        ensure_output_directory(self.output_dir)
        palette_path = Path(config["COLORS_JSON"]).expanduser().resolve()
        self.palette = load_palette(palette_path)
        self.backgrounds = BackgroundLibrary(config)

    def iter_input_images(self) -> Iterable[Path]:
        for file in sorted(self.input_dir.iterdir()):
            if file.suffix.lower() in self.config["VALID_EXTENSIONS"]:
                yield file

    def run(self) -> None:
        for image_path in self.iter_input_images():
            LOGGER.info("Processing %s", image_path.name)
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGBA")
                    self.process_image(image_path.stem, image)
            except Exception as exc:
                LOGGER.error("Failed to process %s: %s", image_path, exc)

    def process_image(self, name: str, image: Image.Image) -> None:
        dominant_colors = extract_dominant_colors(image, self.config)
        if self.palette:
            dominant_colors.extend(self.palette)
        unique_colors: List[Tuple[int, int, int]] = []
        seen_colors = set()
        for col in dominant_colors:
            if col not in seen_colors:
                unique_colors.append(col)
                seen_colors.add(col)
        dominant_colors = unique_colors
        if not dominant_colors:
            LOGGER.warning("No dominant colors found for %s", name)
            return
        variation_groups = self._collect_enabled_groups()
        total_groups = len(variation_groups)
        if total_groups == 0:
            LOGGER.warning("No variation groups enabled; skipping %s", name)
            return
        accumulator = VariantAccumulator(self.config["SIMILARITY_THRESHOLD"])

        color_worker_count = self.config["MAX_COLOR_WORKERS"] if self.config["PARALLEL_COLORS"] else 1
        with ThreadPoolExecutor(max_workers=color_worker_count) as executor:
            futures = []
            for index, color in enumerate(dominant_colors):
                futures.append(
                    executor.submit(
                        self._process_color,
                        name,
                        image,
                        color,
                        variation_groups,
                        index,
                        accumulator,
                        dominant_colors,
                    )
                )
            for future in as_completed(futures):
                future.result()

    def _collect_enabled_groups(self) -> Dict[str, Dict[str, object]]:
        groups = self.config["PERCEPTUAL_VARIANT_SYSTEM"]["VARIANT_GROUPS"]
        enabled = {name: settings for name, settings in groups.items() if settings.get("enabled", False)}
        return enabled

    def _process_color(
        self,
        base_name: str,
        image: Image.Image,
        color: Tuple[int, int, int],
        groups: Dict[str, Dict[str, object]],
        color_index: int,
        accumulator: VariantAccumulator,
        all_colors: Sequence[Tuple[int, int, int]],
    ) -> None:
        target_variants = self._plan_variants_per_color(len(groups))
        mix_count = 0
        attempts = 0
        generated = 0
        used_colors: List[Tuple[int, int, int]] = []
        palette_pool = list(all_colors) if all_colors else [color]
        while generated < target_variants and attempts < self.config["MAX_GENERATION_ATTEMPTS"]:
            attempts += 1
            group_name = random.choice(list(groups.keys()))
            group_settings = groups[group_name]
            if self.config["CROSS_COLOR_MIXING"] and mix_count < self.config["MAX_COLOR_MIXES"] and random.random() < 0.5:
                mix_ratio = random.uniform(0.25, 0.75)
                mix_source = random.choice(palette_pool)
                color_variant = clamp_color(
                    tuple((1 - mix_ratio) * component + mix_ratio * mix_source[i] for i, component in enumerate(color))
                )
                mix_count += 1
            elif random.random() < self.config["RANDOM_COLOR_PROBABILITY"]:
                color_variant = clamp_color(tuple(
                    random.randint(self.config["COLOR_RANGE"]["min"], self.config["COLOR_RANGE"]["max"])
                    for _ in range(3)
                ))
            else:
                color_variant = color

            if any(color_distance(color_variant, existing) < self.config["MIN_COLOR_DIFFERENCE"] for existing in used_colors):
                continue

            background = self.backgrounds.get_random_background(image.size)
            variation_index = generated
            try:
                result = compose_variant(
                    image,
                    background,
                    color_variant,
                    self.config,
                    group_name,
                    group_settings,
                    variation_index,
                    attempts,
                )
            except Exception as exc:
                LOGGER.debug("Variant generation failed: %s", exc)
                continue

            if not accumulator.accept(result.image):
                continue

            rotations = [result]
            if self.config["GENERATE_ROTATIONS"]:
                rotation_angles = self.config["ROTATION_ANGLES"]
                if self.config["PARALLEL_ROTATIONS"]:
                    with ThreadPoolExecutor(max_workers=self.config["MAX_ROTATION_WORKERS"]) as rotation_executor:
                        future = rotation_executor.submit(rotation_worker, result, rotation_angles)
                        rotations.extend(future.result())
                else:
                    rotations.extend(rotation_worker(result, rotation_angles))
            for rotated in rotations:
                self._save_variant(base_name, color_index, rotated)
            used_colors.append(color_variant)
            generated += 1

        if generated < self.config["MIN_VARIATIONS_PER_COLOR"] and self.config["ENSURE_MIN_VARIATIONS"]:
            LOGGER.warning(
                "Color %s for %s only generated %s/%s variants", color, base_name, generated, self.config["MIN_VARIATIONS_PER_COLOR"]
            )

    def _plan_variants_per_color(self, group_count: int) -> int:
        target = self.config["PERCEPTUAL_VARIANT_SYSTEM"]["VARIANTS_PER_GROUP"] * group_count
        target = max(target, self.config["MIN_VARIATIONS_PER_COLOR"])
        target = min(target, self.config["MAX_VARIATIONS_PER_COLOR"])
        return target

    def _save_variant(self, base_name: str, color_index: int, variant: VariantResult) -> None:
        base_dir = self.output_dir / base_name
        ensure_output_directory(base_dir)
        suffix = f"c{color_index:02d}_{variant.suffix}"
        filename = base_dir / f"{base_name}_{suffix}.png"
        normal_name = base_dir / f"{base_name}_{suffix}_normal.png"
        specular_name = base_dir / f"{base_name}_{suffix}_specular.png"
        roughness_name = base_dir / f"{base_name}_{suffix}_roughness.png"
        emissive_name = base_dir / f"{base_name}_{suffix}_emissive.png"
        variant.image.save(filename, format="PNG")
        variant.normal.save(normal_name, format="PNG")
        variant.specular.save(specular_name, format="PNG")
        variant.roughness.save(roughness_name, format="PNG")
        variant.emissive.save(emissive_name, format="PNG")


def main(args: Optional[Sequence[str]] = None) -> None:
    random.seed()
    pipeline = RecolorPipeline(CONFIG)
    pipeline.run()


if __name__ == "__main__":
    main()
