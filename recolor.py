import os
import json
import logging
import random
from pathlib import Path
from PIL import Image
import numpy as np
from colorsys import rgb_to_hls, hls_to_rgb
from collections import defaultdict
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing as mp
import gc
import cv2
from scipy import ndimage

try:
    from colormath.color_objects import sRGBColor, LabColor
    from colormath.color_conversions import convert_color
    from colormath.color_diff import delta_e_cie2000
    COLOR_MATH_AVAILABLE = True
except ImportError:
    COLOR_MATH_AVAILABLE = False

CONFIG = {
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

    # 🔥 mayor sampling por lote, mejora throughput con NVMe
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
    "MAX_GENERATION_ATTEMPTS": 800,   # baja latencia

    "SIMILARITY_THRESHOLD": 0.85,
    "ENSURE_MIN_VARIATIONS": True,
    "RANDOM_COLOR_PROBABILITY": 0.3,
    "CROSS_COLOR_MIXING": True,
    "MAX_COLOR_MIXES": 256,  # doblado para explotar RAM sin swapping
    "NOISE_INTENSITY": 0.08,

    # ⚡ Nuevo: Paralelismo real y mayor control
    "MAX_WORKERS": 12,
    "CHUNK_SIZE": 2000,
    "MEMORY_MONITORING": False,
    "AUTO_ADJUST_CHUNKS": False,
    "TARGET_MEMORY_GB": 13.5,
    "MAX_MEMORY_PERCENT": 94,
    "MEMORY_PER_IMAGE_MB": 3.2,

    # ⚡ Nuevo: Paralelismo agresivo por rotaciones y colores
    "PARALLEL_ROTATIONS": True,
    "PARALLEL_COLORS": True,
    "MAX_ROTATION_WORKERS": 4,
    "MAX_COLOR_WORKERS": 8,
}


GENERATE_PBR_MAPS = True
USE_BACKGROUND_MATERIALS = True
USE_REAL_BACKGROUNDS_ONLY = CONFIG["USE_REAL_BACKGROUNDS_ONLY"]
BACKGROUNDS_DIR = CONFIG["BACKGROUNDS_DIR"]

FOREGROUND_NORMAL_INTENSITY = 2.0
FOREGROUND_SPECULAR_SHARPNESS = 1.8
FOREGROUND_SPECULAR_INTENSITY = 0.8

BACKGROUND_NORMAL_INTENSITY = 0.8
BACKGROUND_SPECULAR_SHARPNESS = 1.2
BACKGROUND_SPECULAR_INTENSITY = 0.3

NORMAL_BLEND_RATIO = 0.8
SPECULAR_BLEND_RATIO = 0.7

SIGMA_FOREGROUND = 1.0
SIGMA_BACKGROUND = 2.0
NORMAL_CLAMP_PERCENTILE = 99

MATERIAL_KEYWORDS = {
    "stone": ["stone", "rock", "piedra", "brick", "cobblestone"],
    "wood": ["wood", "madera", "plank", "log", "oak"],
    "metal": ["metal", "iron", "steel", "gold", "copper"],
    "fabric": ["fabric", "cloth", "tela", "wool", "carpet"],
    "crystal": ["crystal", "glass", "gem", "diamond", "ice"],
}

MATERIAL_SPECULAR_PROPERTIES = {
    "metal": {"intensity": 1.2, "roughness": 0.1, "anisotropy": 0.8},
    "fabric": {"intensity": 0.3, "roughness": 0.8, "anisotropy": 0.1},
    "wood": {"intensity": 0.4, "roughness": 0.6, "anisotropy": 0.3},
    "stone": {"intensity": 0.5, "roughness": 0.7, "anisotropy": 0.2},
    "crystal": {"intensity": 0.9, "roughness": 0.3, "anisotropy": 0.6},
    "default": {"intensity": 0.7, "roughness": 0.5, "anisotropy": 0.4},
}

PBR_INTENSITY = FOREGROUND_NORMAL_INTENSITY
SPECULAR_SHARPNESS = FOREGROUND_SPECULAR_SHARPNESS

PBR_POLYGON_JSONL = Path("data/trainDataMinecraft.jsonl")
ALPHA_CUTOFF_SPEC = 24
EMISSIVE_ALPHA_CORE = 255
EMISSIVE_ALPHA_RING_RATIO = 0.5
EMISSIVE_ALPHA_NON_TARGET = 12
EMISSIVE_ALPHA_BACKGROUND = 3
EMISSIVE_COLOR_EYES = (255, 100, 50, 255)
EMISSIVE_COLOR_WINGS = (100, 200, 255, 255)
EMISSIVE_MIN_COMPONENT_PCT = 0.001
ADD_NORMAL_ALPHA = False

try:
    PNG_PARAMS = [
        cv2.IMWRITE_PNG_COMPRESSION,
        9,
        cv2.IMWRITE_PNG_STRATEGY,
        cv2.IMWRITE_PNG_STRATEGY_RLE,
    ]
except AttributeError:  # pragma: no cover - OpenCV sin estrategia RLE
    PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 9]


def rotate_point_90(x, y):
    return y, 1 - x


def rotate_point_180(x, y):
    return 1 - x, 1 - y


def rotate_point_270(x, y):
    return 1 - y, x


ROT_POINT_FUNCS = {
    "rot90": rotate_point_90,
    "rot180": rotate_point_180,
    "rot270": rotate_point_270,
}


_POLYGON_CACHE = None


def _load_polygon_annotations(jsonl_path=None):
    global _POLYGON_CACHE
    if _POLYGON_CACHE is not None:
        return _POLYGON_CACHE

    if jsonl_path is None:
        jsonl_path = PBR_POLYGON_JSONL

    annotations = {}
    try:
        path = Path(jsonl_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    image_name = record.get("image")
                    if not image_name:
                        continue

                    key = Path(image_name).stem.lower()
                    objects = record.get("objects") or []
                    if not isinstance(objects, list):
                        continue

                    if key not in annotations:
                        annotations[key] = []
                    annotations[key].extend(objects)
    except Exception:
        annotations = {}

    _POLYGON_CACHE = annotations
    return _POLYGON_CACHE


def _get_polygon_objects(base_name, jsonl_path=None):
    annotations = _load_polygon_annotations(jsonl_path)
    if not annotations:
        return None

    if not base_name:
        return None

    return annotations.get(base_name.lower())


def _parse_variant_metadata(output_name):
    stem = Path(output_name).stem
    rotation_suffix = "rot0"
    base_key = stem

    for rot_tag in ("_rot90", "_rot180", "_rot270"):
        if rot_tag in base_key:
            rotation_suffix = rot_tag.lstrip("_")
            base_key = base_key.split(rot_tag)[0]
            break

    if "_color" in base_key:
        base_key = base_key.split("_color")[0]

    if not base_key:
        base_key = stem

    return base_key, rotation_suffix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("processing.log", mode='w'), logging.StreamHandler()],
)

def calculate_color_difference(color1, color2):
    if not COLOR_MATH_AVAILABLE or len(color1) < 3 or len(color2) < 3:
        return np.sqrt(np.sum((np.array(color1[:3]) - np.array(color2[:3])) ** 2))
    
    try:
        rgb1 = sRGBColor(color1[0], color1[1], color1[2], is_upscaled=True)
        rgb2 = sRGBColor(color2[0], color2[1], color2[2], is_upscaled=True)
        lab1 = convert_color(rgb1, LabColor)
        lab2 = convert_color(rgb2, LabColor)
        return delta_e_cie2000(lab1, lab2)
    except Exception:
        return np.sqrt(np.sum((np.array(color1[:3]) - np.array(color2[:3])) ** 2))

def is_color_significantly_different(new_color, existing_colors, threshold=None):
    if threshold is None:
        threshold = CONFIG["MIN_COLOR_DIFFERENCE"]
    
    if not existing_colors:
        return True
        
    new_arr = np.array(new_color[:3])
    existing_arr = np.array([c[:3] for c in existing_colors])
    differences = np.sqrt(np.sum((existing_arr - new_arr) ** 2, axis=1))
    return np.all(differences >= threshold)

def load_colors(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        colors = []
        min_color = CONFIG["COLOR_RANGE"]["min"]
        max_color = CONFIG["COLOR_RANGE"]["max"]
        min_alpha = CONFIG["ALPHA_RANGE"]["min"]
        max_alpha = CONFIG["ALPHA_RANGE"]["max"]
        
        for name, color in data.items():
            if (len(color) == 4 and
                min_color <= color[0] <= max_color and
                min_color <= color[1] <= max_color and
                min_color <= color[2] <= max_color and
                min_alpha <= color[3] <= max_alpha):
                colors.append(color)

        return colors
    except Exception as e:
        logging.error(f"Error cargando {json_path}: {str(e)}")
        return []

def generate_random_colors(existing_colors, count):
    new_colors = []
    existing_tuples = [tuple(c[:3]) for c in existing_colors]
    attempts = 0
    min_color = CONFIG["COLOR_RANGE"]["min"]
    max_color = CONFIG["COLOR_RANGE"]["max"]
    
    while len(new_colors) < count and attempts < CONFIG["MAX_GENERATION_ATTEMPTS"]:
        if random.random() < 0.7:
            r = random.choice([random.randint(50, 100), random.randint(150, 255)])
            g = random.choice([random.randint(50, 100), random.randint(150, 255)])
            b = random.choice([random.randint(50, 100), random.randint(150, 255)])
        else:
            r, g, b = [random.randint(min_color, max_color) for _ in range(3)]
        
        if abs(r - g) < 30 and abs(g - b) < 30:
            max_channel = max(r, g, b)
            if max_channel == r:
                r = min(255, r + 50)
            elif max_channel == g:
                g = min(255, g + 50)
            else:
                b = min(255, b + 50)
        
        a = round(random.uniform(CONFIG["ALPHA_RANGE"]["min"], CONFIG["ALPHA_RANGE"]["max"]), 2)
        new_color = [r, g, b, a]

        if is_color_significantly_different(new_color, existing_colors + new_colors, 5.0):
            new_colors.append(new_color)
            existing_tuples.append((r, g, b))

        attempts += 1

    return new_colors

def mix_colors(color1, color2, ratio=0.5):
    r = int(color1[0] * ratio + color2[0] * (1 - ratio))
    g = int(color1[1] * ratio + color2[1] * (1 - ratio))
    b = int(color1[2] * ratio + color2[2] * (1 - ratio))
    return [r, g, b, color1[3]]

def add_color_noise(color, intensity=0.1):
    noise_range = int(255 * intensity)
    r = max(0, min(255, color[0] + random.randint(-noise_range, noise_range)))
    g = max(0, min(255, color[1] + random.randint(-noise_range, noise_range)))
    b = max(0, min(255, color[2] + random.randint(-noise_range, noise_range)))
    return [r, g, b, color[3]]

def generate_high_variation_colors(base_color, num_variations, all_colors=None):
    variations = []
    r, g, b, a = base_color
    h_base, l_base, s_base = rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)

    attempts = 0
    generated_count = 0
    min_color = CONFIG["COLOR_RANGE"]["min"]
    max_color = CONFIG["COLOR_RANGE"]["max"]

    while generated_count < num_variations and attempts < CONFIG["MAX_GENERATION_ATTEMPTS"]:
        attempts += 1
        
        strategy = random.choice(["hue_shift", "saturation_boost", "lightness_variation", "random_component", "color_mix", "completely_random"])
        
        if strategy == "hue_shift" or strategy == "completely_random":
            h_variation = random.uniform(-0.4, 0.4)
            s_variation = random.uniform(-0.3, 0.6)
            l_variation = random.uniform(-0.4, 0.4)
            
            h_new = (h_base + h_variation) % 1.0
            s_new = max(0.1, min(1.0, s_base + s_variation))
            l_new = max(0.1, min(0.9, l_base + l_variation))
            
            r_new, g_new, b_new = hls_to_rgb(h_new, l_new, s_new)
            variation = [int(r_new * 255), int(g_new * 255), int(b_new * 255), a]
            
        elif strategy == "saturation_boost":
            h_new = h_base
            s_new = random.uniform(0.6, 1.0)
            l_new = random.uniform(0.3, 0.7)
            r_new, g_new, b_new = hls_to_rgb(h_new, s_new, l_new)
            variation = [int(r_new * 255), int(g_new * 255), int(b_new * 255), a]
            
        elif strategy == "lightness_variation":
            h_new = h_base
            s_new = random.uniform(0.4, 1.0)
            l_new = random.choice([random.uniform(0.1, 0.3), random.uniform(0.7, 0.9)])
            r_new, g_new, b_new = hls_to_rgb(h_new, s_new, l_new)
            variation = [int(r_new * 255), int(g_new * 255), int(b_new * 255), a]
            
        elif strategy == "random_component":
            variation = [r, g, b, a]
            component = random.randint(0, 2)
            variation[component] = random.randint(min_color, max_color)
            
        elif strategy == "color_mix" and all_colors and len(all_colors) > 1:
            other_color = random.choice(all_colors)
            ratio = random.uniform(0.2, 0.8)
            variation = mix_colors(base_color, other_color, ratio)
            
        else:
            variation = [
                random.randint(min_color, max_color),
                random.randint(min_color, max_color),
                random.randint(min_color, max_color),
                a
            ]

        if random.random() < 0.6:
            variation = add_color_noise(variation, CONFIG["NOISE_INTENSITY"])

        variation[:3] = np.clip(variation[:3], min_color, max_color)

        if is_color_significantly_different(variation, variations, 5.0):
            variations.append(variation)
            generated_count += 1

    if len(variations) < num_variations:
        needed = num_variations - len(variations)
        for i in range(needed):
            intensity = random.uniform(0.3, 2.0)
            variation = [
                max(min_color, min(max_color, int(r * intensity))),
                max(min_color, min(max_color, int(g * intensity))),
                max(min_color, min(max_color, int(b * intensity))),
                a
            ]
            variations.append(variation)

    return variations[:num_variations]

def apply_color_preserving_brightness(original_pixel, target_color):
    r_orig, g_orig, b_orig, a_orig = original_pixel
    r_target, g_target, b_target, a_target = target_color

    try:
        h_orig, l_orig, s_orig = rgb_to_hls(r_orig / 255.0, g_orig / 255.0, b_orig / 255.0)
        h_target, l_target, s_target = rgb_to_hls(r_target / 255.0, g_target / 255.0, b_target / 255.0)

        if random.random() < 0.7:
            r_new, g_new, b_new = hls_to_rgb(h_target, l_orig, s_target)
        else:
            l_variation = random.uniform(-0.2, 0.2)
            l_new = max(0.1, min(0.9, l_orig + l_variation))
            r_new, g_new, b_new = hls_to_rgb(h_target, l_new, s_target)

        r_new, g_new, b_new = int(r_new * 255), int(g_new * 255), int(b_new * 255)
    except Exception:
        orig_luminance = 0.299 * r_orig + 0.587 * g_orig + 0.114 * b_orig
        target_luminance = 0.299 * r_target + 0.587 * g_target + 0.114 * b_target

        if target_luminance > 0:
            scale_factor = orig_luminance / target_luminance
            r_new, g_new, b_new = int(r_target * scale_factor), int(g_target * scale_factor), int(b_target * scale_factor)
        else:
            r_new, g_new, b_new = r_target, g_target, b_target

    r_new, g_new, b_new = np.clip([r_new, g_new, b_new], 0, 255)
    return [r_new, g_new, b_new, a_orig]

def get_dominant_colors(image, max_colors=None):
    if max_colors is None:
        max_colors = CONFIG["MAX_DOMINANT_COLORS"]

    img_array = np.array(image)
    
    if img_array.shape[0] * img_array.shape[1] > CONFIG["SAMPLING_THRESHOLD"]:
        step = max(1, int(np.sqrt(img_array.shape[0] * img_array.shape[1] / (CONFIG["SAMPLING_THRESHOLD"] // 2))))
        pixels = img_array[::step, ::step].reshape(-1, img_array.shape[2])
    else:
        pixels = img_array.reshape(-1, img_array.shape[2])

    if pixels.shape[1] == 4:
        pixels = pixels[pixels[:, 3] > 0]

    if len(pixels) == 0:
        return []

    if len(pixels) > max_colors * 5:
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=max_colors, random_state=42, batch_size=1000)
        labels = kmeans.fit_predict(pixels[:, :3])
        dominant_colors = kmeans.cluster_centers_.astype(int)
        return [tuple(color) for color in dominant_colors]
    else:
        def color_hash(color, bucket_size=CONFIG["BUCKET_SIZE"]):
            return tuple((c // bucket_size) * bucket_size for c in color[:3])

        color_buckets = defaultdict(list)
        for pixel in pixels:
            if len(pixel) >= 3:
                color_buckets[color_hash(pixel)].append(pixel)

        dominant_colors = []
        for bucket_colors in color_buckets.values():
            if bucket_colors:
                avg_color = np.mean(bucket_colors, axis=0).astype(int)
                dominant_colors.append(tuple(avg_color))

        return dominant_colors[:max_colors]

def generate_rotations(img):
    rotations = {}

    for angle in CONFIG["ROTATION_ANGLES"]:
        if angle == 0:
            rotations[""] = img
        else:
            rotated_img = img.rotate(angle, expand=True)
            rotations[f"_rot{angle}"] = rotated_img

    return rotations

def adjust_background_contrast(bg, fg, target_lum_diff=50):
    """Ajusta brillo del fondo según contraste con el objeto (fg)."""
    bg_np = np.array(bg.convert("L"))
    fg_np = np.array(fg.convert("RGBA"))[..., :3]
    fg_gray = np.mean(fg_np, axis=2)

    bg_mean = np.mean(bg_np)
    fg_mean = np.mean(fg_gray)

    diff = abs(fg_mean - bg_mean)
    if diff < target_lum_diff:
        # Necesita más contraste → sube o baja brillo del fondo
        factor = 1.25 if bg_mean < fg_mean else 0.75
        bg = Image.fromarray(np.clip(bg_np * factor, 0, 255).astype("uint8"), "L").convert("RGB")

    return bg


def extract_luminance_linear(img_array):
    """Extrae luminancia física en espacio lineal a partir de un arreglo de imagen."""
    if img_array.ndim == 2:
        rgb = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] >= 3:
        rgb = img_array[:, :, :3]
    else:
        raise ValueError("Imagen inválida para extracción de luminancia")

    rgb_linear = np.power(rgb.astype(np.float32) / 255.0, 2.2)
    luminance = (
        0.2126 * rgb_linear[:, :, 0]
        + 0.7152 * rgb_linear[:, :, 1]
        + 0.0722 * rgb_linear[:, :, 2]
    )

    return luminance


def detect_material_type(filename):
    """Detecta el tipo de material en función del nombre del archivo."""
    if not filename:
        return "default"

    filename_lower = Path(filename).stem.lower()

    for material, keywords in MATERIAL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return material

    return "default"


def get_random_background(size):
    """Selecciona aleatoriamente un fondo real y lo redimensiona al tamaño solicitado."""
    if not size or len(size) != 2:
        raise ValueError("El tamaño del fondo debe ser una tupla (ancho, alto)")

    try:
        width, height = map(int, size)
    except Exception as exc:  # pragma: no cover - validación defensiva
        raise ValueError("El tamaño del fondo debe contener valores numéricos") from exc

    if width <= 0 or height <= 0:
        raise ValueError("El tamaño del fondo debe ser mayor que cero")

    backgrounds_path = Path(CONFIG["BACKGROUNDS_DIR"])
    if not backgrounds_path.exists() or not backgrounds_path.is_dir():
        raise FileNotFoundError(
            f"Directorio de fondos no encontrado: {backgrounds_path}"
        )

    valid_ext = {ext.lower() for ext in CONFIG["VALID_EXTENSIONS"]}
    background_files = [
        path
        for path in backgrounds_path.iterdir()
        if path.is_file() and path.suffix.lower() in valid_ext
    ]

    if not background_files:
        raise FileNotFoundError(
            f"No se encontraron imágenes de fondo válidas en {backgrounds_path}"
        )

    random.shuffle(background_files)
    last_error = None

    for path in background_files:
        try:
            with Image.open(path) as bg_image:
                bg_rgba = bg_image.convert("RGBA")
                resized = bg_rgba.resize((width, height), Image.LANCZOS)
                result = resized.copy()
                result.info["background_path"] = str(path)
                return result
        except Exception as exc:
            logging.warning(f"No se pudo usar el background '{path}': {exc}")
            last_error = exc
            continue

    raise RuntimeError(
        "No se pudo cargar ninguna imagen válida de la carpeta de fondos"
    ) from last_error


def load_background_layers(base_name, size, backgrounds_dir=BACKGROUNDS_DIR):
    """Carga un fondo real aleatorio y genera sus mapas de normales y especular."""
    backgrounds_path = Path(backgrounds_dir)
    if not backgrounds_path.exists() or not backgrounds_path.is_dir():
        raise FileNotFoundError(
            f"Directorio de fondos no encontrado: {backgrounds_dir}"
        )

    try:
        width, height = map(int, size)
    except Exception as exc:  # pragma: no cover - validación defensiva
        raise ValueError("El tamaño debe tener formato (ancho, alto)") from exc

    background_img = get_random_background((width, height))
    background_path_str = background_img.info.get("background_path")
    background_path = Path(background_path_str) if background_path_str else None

    bg_rgba = np.array(background_img, dtype=np.uint8)
    if bg_rgba.ndim == 2:
        bg_rgba = np.stack([bg_rgba] * 3, axis=-1)
        alpha_channel = np.full((height, width, 1), 255, dtype=np.uint8)
        bg_rgba = np.concatenate([bg_rgba, alpha_channel], axis=2)
    elif bg_rgba.shape[2] == 3:
        alpha_channel = np.full((height, width, 1), 255, dtype=np.uint8)
        bg_rgba = np.concatenate([bg_rgba, alpha_channel], axis=2)

    background_material = "default"
    if background_path is not None:
        background_material = detect_material_type(background_path.stem)
    if background_material not in MATERIAL_SPECULAR_PROPERTIES:
        background_material = "default"

    bg_normal = generate_background_normal_map(
        bg_rgba, intensity=BACKGROUND_NORMAL_INTENSITY
    )
    bg_specular = generate_background_specular_map(
        bg_rgba,
        material_type=background_material,
        sharpness=BACKGROUND_SPECULAR_SHARPNESS,
    )

    return bg_rgba, bg_normal, bg_specular, background_material


def generate_foreground_normal_map(img_array, intensity=None):
    """Genera normales de alta frecuencia para el objeto en primer plano."""
    if intensity is None:
        intensity = FOREGROUND_NORMAL_INTENSITY

    luminance = extract_luminance_linear(img_array)
    height, width = luminance.shape

    sigma = 0.8 if max(height, width) > 256 else 0.5
    sigma = min(SIGMA_FOREGROUND, sigma) if SIGMA_FOREGROUND else sigma
    blurred = ndimage.gaussian_filter(luminance, sigma=sigma)

    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx * gx + gy * gy)
    clip_value = (
        np.percentile(magnitude, NORMAL_CLAMP_PERCENTILE)
        if magnitude.size and NORMAL_CLAMP_PERCENTILE
        else 0
    )
    if clip_value > 0:
        gx = np.clip(gx, -clip_value, clip_value)
        gy = np.clip(gy, -clip_value, clip_value)

    gx *= intensity
    gy *= intensity

    gz = np.ones_like(luminance, dtype=np.float32)
    normal = np.dstack((gx, gy, gz)).astype(np.float32)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    norm[norm == 0] = 1.0
    normal = normal / norm

    normal = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return normal


def generate_background_normal_map(background_img, intensity=None):
    """Genera normales suaves para materiales del fondo."""
    if intensity is None:
        intensity = BACKGROUND_NORMAL_INTENSITY

    luminance = extract_luminance_linear(background_img)
    blurred = ndimage.gaussian_filter(luminance, sigma=SIGMA_BACKGROUND)

    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=5)

    gx *= intensity
    gy *= intensity

    gz = np.ones_like(luminance, dtype=np.float32)
    normal = np.dstack((gx, gy, gz)).astype(np.float32)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    norm[norm == 0] = 1.0
    normal = normal / norm

    normal = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return normal


def generate_physically_correct_specular(
    img_array,
    material_type="default",
    sharpness=None,
    alpha_cutoff=ALPHA_CUTOFF_SPEC,
    energy_scale=1.0,
):
    """Genera un mapa specular físicamente correcto en espacio lineal."""
    if sharpness is None:
        sharpness = FOREGROUND_SPECULAR_SHARPNESS

    props = MATERIAL_SPECULAR_PROPERTIES.get(material_type, MATERIAL_SPECULAR_PROPERTIES["default"])

    if img_array.shape[2] == 4:
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3]
    else:
        rgb = img_array
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)

    if alpha_cutoff is None:
        alpha_cutoff = ALPHA_CUTOFF_SPEC

    rgb_linear = np.power(rgb.astype(np.float32) / 255.0, 2.2)
    luminance = (
        0.2126 * rgb_linear[:, :, 0]
        + 0.7152 * rgb_linear[:, :, 1]
        + 0.0722 * rgb_linear[:, :, 2]
    )

    laplacian = cv2.Laplacian(luminance, cv2.CV_32F, ksize=3)
    sobelx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.abs(laplacian) + 0.5 * (np.abs(sobelx) + np.abs(sobely))
    edges = cv2.bilateralFilter(edges.astype(np.float32), 9, 75, 75)
    edges = np.clip(edges, 0.0, None)
    if np.any(edges):
        edges = cv2.normalize(edges, None, 0.0, 1.0, cv2.NORM_MINMAX)

    specular_intensity = edges * luminance * sharpness * props["intensity"] * energy_scale
    specular_intensity = np.clip(specular_intensity, 0.0, 1.0)
    specular_intensity = np.power(specular_intensity, 1.0 / (props["roughness"] + 0.5))

    specular_values = np.power(specular_intensity, 1.0 / 2.2) * 255.0
    specular_values = np.clip(specular_values, 0, 255).astype(np.uint8)

    alpha_mask = alpha > alpha_cutoff
    base_alpha = alpha.astype(np.float32) / 255.0
    anisotropy_factor = 0.5 + 0.5 * props.get("anisotropy", 0.5)
    specular_alpha = (
        (specular_values.astype(np.float32) / 255.0)
        * base_alpha
        * anisotropy_factor
    )
    specular_alpha = np.clip(specular_alpha * 255.0, 0, 255).astype(np.uint8)

    specular_rgb = np.dstack([specular_values] * 3)
    specular_rgb[~alpha_mask] = 0
    specular_alpha[~alpha_mask] = 0

    return np.dstack([specular_rgb, specular_alpha])


def generate_foreground_specular_map(img_array, material_type="default", sharpness=None):
    """Genera un mapa specular para el objeto en primer plano."""
    if sharpness is None:
        sharpness = FOREGROUND_SPECULAR_SHARPNESS

    material = material_type or "default"
    if material not in MATERIAL_SPECULAR_PROPERTIES:
        material = detect_material_type(material)
    if material not in MATERIAL_SPECULAR_PROPERTIES:
        material = "default"

    return generate_physically_correct_specular(
        img_array,
        material_type=material,
        sharpness=sharpness,
        energy_scale=FOREGROUND_SPECULAR_INTENSITY,
    )


def generate_background_specular_map(background_img, material_type="default", sharpness=None):
    """Genera un mapa specular para el material de fondo."""
    if sharpness is None:
        sharpness = BACKGROUND_SPECULAR_SHARPNESS

    material = material_type or "default"
    if material not in MATERIAL_SPECULAR_PROPERTIES:
        material = detect_material_type(material)
    if material not in MATERIAL_SPECULAR_PROPERTIES:
        material = "default"

    if background_img.ndim == 2:
        background_img = np.stack([background_img] * 3, axis=-1)
    if background_img.shape[2] == 3:
        alpha = np.full(background_img.shape[:2] + (1,), 255, dtype=np.uint8)
        bg_rgba = np.concatenate([background_img, alpha], axis=2)
    else:
        bg_rgba = background_img

    return generate_physically_correct_specular(
        bg_rgba,
        material_type=material,
        sharpness=sharpness,
        alpha_cutoff=0,
        energy_scale=BACKGROUND_SPECULAR_INTENSITY,
    )


def blend_normals_physically(normal_fg, normal_bg, alpha_mask, blend_ratio=NORMAL_BLEND_RATIO):
    """Combina normales de forma física respetando el alpha del objeto."""
    if normal_fg.shape[2] > 3:
        normal_fg = normal_fg[:, :, :3]
    if normal_bg.shape[2] > 3:
        normal_bg = normal_bg[:, :, :3]

    normal_fg_vec = normal_fg.astype(np.float32) / 127.5 - 1.0
    normal_bg_vec = normal_bg.astype(np.float32) / 127.5 - 1.0

    alpha_weights = alpha_mask.astype(np.float32) / 255.0
    fg_weight = alpha_weights * blend_ratio
    bg_weight = (1.0 - alpha_weights) * (1.0 - blend_ratio)

    blended_vec = (
        normal_fg_vec * fg_weight[..., None]
        + normal_bg_vec * bg_weight[..., None]
    )

    norm = np.linalg.norm(blended_vec, axis=2, keepdims=True)
    norm[norm == 0] = 1.0
    blended_vec = blended_vec / norm

    blended = ((blended_vec + 1.0) * 127.5).astype(np.uint8)
    return blended


def blend_specular_physically(specular_fg, specular_bg, alpha_mask, blend_ratio=SPECULAR_BLEND_RATIO):
    """Combina mapas specular preservando energía física."""
    if specular_fg.shape[2] == 4:
        spec_fg_rgb = specular_fg[:, :, :3]
        spec_fg_alpha = specular_fg[:, :, 3]
    else:
        spec_fg_rgb = specular_fg
        spec_fg_alpha = np.full(spec_fg_rgb.shape[:2], 255, dtype=np.uint8)

    if specular_bg.shape[2] == 4:
        spec_bg_rgb = specular_bg[:, :, :3]
        spec_bg_alpha = specular_bg[:, :, 3]
    else:
        spec_bg_rgb = specular_bg
        spec_bg_alpha = np.full(spec_bg_rgb.shape[:2], 255, dtype=np.uint8)

    spec_fg_lin = np.power(spec_fg_rgb.astype(np.float32) / 255.0, 2.2)
    spec_bg_lin = np.power(spec_bg_rgb.astype(np.float32) / 255.0, 2.2)

    alpha_weights = alpha_mask.astype(np.float32) / 255.0
    fg_weight = alpha_weights * blend_ratio
    bg_weight = (1.0 - alpha_weights) * (1.0 - blend_ratio)

    blended_lin = (
        spec_fg_lin * fg_weight[..., None]
        + spec_bg_lin * bg_weight[..., None]
    )

    blended_srgb = np.power(np.clip(blended_lin, 0.0, 1.0), 1.0 / 2.2) * 255.0
    blended_srgb = np.clip(blended_srgb, 0, 255).astype(np.uint8)

    blended_alpha = (
        spec_fg_alpha.astype(np.float32) * fg_weight
        + spec_bg_alpha.astype(np.float32) * bg_weight
    )
    blended_alpha = np.clip(blended_alpha, 0, 255).astype(np.uint8)

    return np.dstack([blended_srgb, blended_alpha])


def generate_enhanced_pbr_maps(
    foreground_img,
    background_layers,
    base_name,
    rotation_suffix="rot0",
    class_data=None,
    background_material=None,
):
    """Genera mapas PBR combinando físicamente objeto y fondo."""
    bg_rgb, bg_normal, bg_specular = background_layers

    fg_material = detect_material_type(base_name)
    if fg_material not in MATERIAL_SPECULAR_PROPERTIES:
        fg_material = "default"

    fg_normal = generate_foreground_normal_map(foreground_img, intensity=FOREGROUND_NORMAL_INTENSITY)
    fg_specular = generate_foreground_specular_map(
        foreground_img,
        material_type=fg_material,
        sharpness=FOREGROUND_SPECULAR_SHARPNESS,
    )

    emissive_map = generate_emissive_map(
        foreground_img,
        base_name,
        class_data=class_data,
        rotation_suffix=rotation_suffix,
    )

    if bg_normal is None:
        bg_normal = generate_background_normal_map(bg_rgb, intensity=BACKGROUND_NORMAL_INTENSITY)
    if bg_specular is None:
        bg_material = background_material or detect_material_type(base_name)
        if bg_material not in MATERIAL_SPECULAR_PROPERTIES:
            bg_material = detect_material_type("default")
        bg_specular = generate_background_specular_map(
            bg_rgb,
            material_type=bg_material,
            sharpness=BACKGROUND_SPECULAR_SHARPNESS,
        )

    alpha_mask = foreground_img[:, :, 3] if foreground_img.shape[2] == 4 else np.full(foreground_img.shape[:2], 255, dtype=np.uint8)
    final_normal = blend_normals_physically(fg_normal, bg_normal, alpha_mask, NORMAL_BLEND_RATIO)
    final_specular = blend_specular_physically(fg_specular, bg_specular, alpha_mask, SPECULAR_BLEND_RATIO)

    return final_normal, final_specular, emissive_map


def process_single_color(args):
    """Procesa un solo color en paralelo"""
    (
        color_idx,
        base_color,
        img_array,
        dominant_colors,
        color_variations,
        base_name,
        rot_suffix,
        output_dir,
    ) = args

    try:
        color_mapping = {}
        for i, original_color in enumerate(dominant_colors):
            if CONFIG["CROSS_COLOR_MIXING"] and len(color_variations) > 1:
                base_variation = color_variations[i % len(color_variations)]
                if random.random() < 0.4:
                    other_variation = color_variations[
                        (i + random.randint(1, len(color_variations) - 1))
                        % len(color_variations)
                    ]
                    mix_ratio = random.uniform(0.2, 0.8)
                    color_mapping[original_color] = mix_colors(
                        base_variation, other_variation, mix_ratio
                    )
                else:
                    color_mapping[original_color] = base_variation
            else:
                color_mapping[original_color] = color_variations[
                    i % len(color_variations)
                ]

        new_array = img_array.copy()

        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                original_pixel = img_array[y, x]

                if len(original_pixel) == 4 and original_pixel[3] == 0:
                    continue

                if len(original_pixel) >= 3:
                    original_rgb = tuple(original_pixel[:3])

                    closest_color = min(
                        dominant_colors,
                        key=lambda c: sum((a - b) ** 2 for a, b in zip(original_rgb, c[:3])),
                    )

                    if closest_color in color_mapping:
                        target_variation = color_mapping[closest_color]
                        new_pixel = apply_color_preserving_brightness(
                            original_pixel, target_variation
                        )
                        new_array[y, x] = new_pixel

        if rot_suffix:
            output_name = f"{base_name}{rot_suffix}_color{color_idx+1:03d}.png"
        else:
            output_name = f"{base_name}_color{color_idx+1:03d}.png"

        result_img = Image.fromarray(new_array.astype("uint8"), "RGBA")
        bg = get_random_background(result_img.size)
        bg = adjust_background_contrast(bg, result_img)
        final_img = Image.alpha_composite(bg.convert("RGBA"), result_img)
        output_path = os.path.join(output_dir, output_name)
        final_img.save(output_path, optimize=True)

        if GENERATE_PBR_MAPS:
            try:
                save_pbr_maps(result_img, output_name, output_dir)
            except Exception as pbr_error:
                logging.error(
                    f"Error generando mapas PBR para {output_name}: {pbr_error}"
                )

        del new_array, result_img, final_img, bg
        gc.collect()

        return output_name

    except Exception as e:
        logging.error(
            f"Error procesando color {color_idx} para {base_name}{rot_suffix}: {str(e)}"
        )
        return None


def process_rotation_parallel(args):
    """Procesa una rotación completa en paralelo con todos sus colores"""
    rot_img, base_name, rot_suffix, colors, output_dir = args

    try:
        rot_img = rot_img.convert("RGBA")
        img_array = np.array(rot_img)

        dominant_colors = get_dominant_colors(
            rot_img, CONFIG["MAX_DOMINANT_COLORS"]
        )
        num_dominant_colors = len(dominant_colors)

        logging.info(
            f"Iniciando procesamiento paralelo de rotación {rot_suffix} - {num_dominant_colors} colores dominantes"
        )

        color_tasks = []
        for color_idx, base_color in enumerate(colors):
            color_variations = generate_high_variation_colors(
                base_color, CONFIG["MAX_VARIATIONS_PER_COLOR"], colors
            )

            color_tasks.append(
                (
                    color_idx,
                    base_color,
                    img_array,
                    dominant_colors,
                    color_variations,
                    base_name,
                    rot_suffix,
                    output_dir,
                )
            )

        generated_count = 0
        if CONFIG["PARALLEL_COLORS"] and len(color_tasks) > 1:
            with ThreadPoolExecutor(
                max_workers=min(CONFIG["MAX_COLOR_WORKERS"], len(color_tasks))
            ) as executor:
                futures = [executor.submit(process_single_color, task) for task in color_tasks]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        generated_count += 1
        else:
            for task in color_tasks:
                result = process_single_color(task)
                if result:
                    generated_count += 1

        logging.info(
            f"Completadas {generated_count} variantes para rotación {rot_suffix}"
        )

        del img_array, color_tasks
        gc.collect()

        return f"Rotación {rot_suffix}: {generated_count} variantes"

    except Exception as e:
        logging.error(
            f"Error procesando rotación {rot_suffix}: {str(e)}", exc_info=True
        )
        return f"Error en rotación {rot_suffix}: {str(e)}"


def process_single_image(args):
    image_path, colors, output_dir = args

    try:
        with Image.open(image_path) as img:
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            if CONFIG["GENERATE_ROTATIONS"]:
                rotations = generate_rotations(img)

                if CONFIG["PARALLEL_ROTATIONS"] and len(rotations) > 1:
                    rotation_tasks = []
                    for rot_suffix, rot_img in rotations.items():
                        rotation_tasks.append(
                            (rot_img, base_name, rot_suffix, colors, output_dir)
                        )

                    with ThreadPoolExecutor(
                        max_workers=min(
                            CONFIG["MAX_ROTATION_WORKERS"], len(rotations)
                        )
                    ) as executor:
                        futures = [
                            executor.submit(process_rotation_parallel, task)
                            for task in rotation_tasks
                        ]
                        results = []
                        for future in as_completed(futures):
                            results.append(future.result())

                        return f"Completada: {base_name} - {' | '.join(results)}"
                else:
                    results = []
                    for rot_suffix, rot_img in rotations.items():
                        result = process_rotation_parallel(
                            (rot_img, base_name, rot_suffix, colors, output_dir)
                        )
                        results.append(result)
                    return f"Completada: {base_name} - {' | '.join(results)}"
            else:
                result = process_rotation_parallel(
                    (img, base_name, "", colors, output_dir)
                )
                return f"Completada: {base_name} - {result}"

    except Exception as e:
        return f"Error en {image_path}: {str(e)}"

def process_image_batch(image_batch, colors, output_dir):
    results = []
    for image_path in image_batch:
        result = process_single_image((image_path, colors, output_dir))
        results.append(result)
        
        # Liberar memoria después de cada imagen
        gc.collect()
        
    return results

def main():
    if not validate_config():
        return

    if not COLOR_MATH_AVAILABLE:
        logging.warning("colormath no disponible. Usando método alternativo")

    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    base_colors = load_colors(CONFIG["COLORS_JSON"])
    if len(base_colors) < CONFIG["MIN_COLOR"]:
        additional_colors = generate_random_colors(
            base_colors, CONFIG["MIN_COLOR"] - len(base_colors)
        )
        base_colors.extend(additional_colors)

    logging.info(f"Total de colores: {len(base_colors)}")
    logging.info(f"Usando {CONFIG['MAX_WORKERS']} procesos paralelos")
    logging.info(f"Paralelismo por rotaciones: {CONFIG['PARALLEL_ROTATIONS']}")
    logging.info(f"Paralelismo por colores: {CONFIG['PARALLEL_COLORS']}")
    logging.info(f"Objetivo de memoria (referencia): {CONFIG['TARGET_MEMORY_GB']}GB")

    image_files = [
        os.path.join(CONFIG["INPUT_DIR"], f)
        for f in os.listdir(CONFIG["INPUT_DIR"])
        if os.path.isfile(os.path.join(CONFIG["INPUT_DIR"], f))
        and os.path.splitext(f)[1].lower() in CONFIG["VALID_EXTENSIONS"]
    ]

    if not image_files:
        logging.warning("No se encontraron imágenes válidas")
        return

    chunk_size = CONFIG["CHUNK_SIZE"]

    image_batches = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]

    logging.info(
        f"🚀 Procesando {len(image_files)} imágenes en {len(image_batches)} lotes (chunk size: {chunk_size})"
    )

    with ProcessPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        futures = []

        for batch in image_batches:
            future = executor.submit(
                process_image_batch, batch, base_colors, CONFIG["OUTPUT_DIR"]
            )
            futures.append(future)

        completed = 0
        for future in as_completed(futures):
            batch_results = future.result()
            for result in batch_results:
                logging.info(result)
                completed += 1
                progress_percent = (completed / len(image_files)) * 100
                logging.info(
                    f"📊 Progreso: {completed}/{len(image_files)} imágenes ({progress_percent:.1f}%)"
                )

    logging.info("✅ Procesamiento completado")

def validate_config():
    errors = []

    if CONFIG["MIN_COLOR"] < 1:
        errors.append("MIN_COLOR debe ser al menos 1")
    
    if CONFIG["COLOR_RANGE"]["min"] >= CONFIG["COLOR_RANGE"]["max"]:
        errors.append("COLOR_RANGE min debe ser menor que max")
    
    if CONFIG["MAX_WORKERS"] > mp.cpu_count():
        CONFIG["MAX_WORKERS"] = max(1, mp.cpu_count() - 1)
        logging.info(f"Ajustando MAX_WORKERS a {CONFIG['MAX_WORKERS']}")

    if CONFIG.get("USE_REAL_BACKGROUNDS_ONLY"):
        backgrounds_dir = Path(CONFIG["BACKGROUNDS_DIR"])
        valid_ext = {ext.lower() for ext in CONFIG["VALID_EXTENSIONS"]}

        if not backgrounds_dir.exists():
            backgrounds_dir.mkdir(parents=True, exist_ok=True)
            logging.warning(
                f"La carpeta de fondos no existía. Se creó en {backgrounds_dir.resolve()}"
            )

        if not backgrounds_dir.is_dir():
            errors.append(
                f"La ruta de fondos '{backgrounds_dir}' no es un directorio válido"
            )
        else:
            background_files = [
                path
                for path in backgrounds_dir.iterdir()
                if path.is_file() and path.suffix.lower() in valid_ext
            ]
            if not background_files:
                errors.append(
                    "La carpeta de fondos no contiene imágenes válidas. "
                    "Agrega archivos PNG/JPG/JPEG/BMP/TIFF a 'backgrounds'"
                )

    if errors:
        for error in errors:
            logging.error(f"Error de configuración: {error}")
        return False
    return True


def generate_normal_map(img_array, intensity=None, add_alpha=ADD_NORMAL_ALPHA):
    """Genera un mapa normal a partir de una imagen RGBA/RGB."""
    if intensity is None:
        intensity = PBR_INTENSITY

    if img_array.shape[2] == 4:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    gray = gray.astype(np.float32) / 255.0
    height, width = gray.shape
    sigma = 1.0 if max(height, width) > 256 else 0.6
    gray = ndimage.gaussian_filter(gray, sigma=sigma)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    gx *= intensity
    gy *= intensity

    magnitude = np.sqrt(gx * gx + gy * gy)
    clip_value = np.percentile(magnitude, 99) if magnitude.size else 0
    if clip_value > 0:
        gx = np.clip(gx, -clip_value, clip_value)
        gy = np.clip(gy, -clip_value, clip_value)

    gz = np.ones_like(gray)
    normal = np.dstack((gx, gy, gz))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    norm[norm == 0] = 1.0
    normal = normal / norm

    normal = (normal + 1.0) * 127.5
    normal = np.clip(normal, 0, 255).astype(np.uint8)

    if add_alpha:
        alpha_channel = np.full((height, width, 1), 255, dtype=np.uint8)
        normal = np.concatenate((normal, alpha_channel), axis=2)

    return normal


def generate_specular_map(img_array, sharpness=None, alpha_cutoff=ALPHA_CUTOFF_SPEC):
    """Genera un mapa specular que respeta transparencias y realza detalles físicos."""
    if sharpness is None:
        sharpness = SPECULAR_SHARPNESS

    if img_array.shape[2] == 4:
        rgb = img_array[:, :, :3]
        alpha_channel = img_array[:, :, 3]
    else:
        rgb = img_array
        alpha_channel = np.full(rgb.shape[:2], 255, dtype=np.uint8)

    if alpha_cutoff is None:
        alpha_cutoff = ALPHA_CUTOFF_SPEC

    alpha_mask = alpha_channel > alpha_cutoff

    rgb_norm = rgb.astype(np.float32) / 255.0
    rgb_lin = np.power(rgb_norm, 2.2)
    luminance = (
        0.2126 * rgb_lin[:, :, 0]
        + 0.7152 * rgb_lin[:, :, 1]
        + 0.0722 * rgb_lin[:, :, 2]
    )

    laplacian = cv2.Laplacian(luminance, cv2.CV_32F, ksize=3)
    sobelx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.abs(laplacian) + 0.5 * (np.abs(sobelx) + np.abs(sobely))
    edges = cv2.bilateralFilter(edges.astype(np.float32), 9, 75, 75)
    edges = np.clip(edges, 0.0, None)
    if np.any(edges):
        edges = cv2.normalize(edges, None, 0.0, 1.0, cv2.NORM_MINMAX)

    specular_intensity = np.clip(edges * luminance * sharpness, 0.0, 1.0)
    specular_values = np.power(specular_intensity, 1 / 2.2)
    specular_values = np.clip(specular_values * 255.0, 0, 255).astype(np.uint8)

    specular_rgb = np.dstack([specular_values] * 3)
    specular_rgb[~alpha_mask] = 0

    base_alpha = alpha_channel.astype(np.float32) / 255.0
    spec_alpha_float = (specular_values.astype(np.float32) / 255.0) * base_alpha
    specular_alpha = np.clip(spec_alpha_float * 255.0, 0, 255).astype(np.uint8)
    specular_alpha[~alpha_mask] = 0

    specular_map = np.dstack([specular_rgb, specular_alpha])
    specular_map[~alpha_mask] = 0

    return specular_map


def _rotate_normalized_points(points, rotation_key):
    if rotation_key == "rot0":
        return points

    rotate_func = ROT_POINT_FUNCS.get(rotation_key)
    if rotate_func is None:
        return points

    return np.array([rotate_func(float(x), float(y)) for x, y in points], dtype=np.float32)


def _normalized_to_pixel(points, width, height):
    points = np.asarray(points, dtype=np.float32)
    points = np.clip(points, 0.0, 1.0)
    x_coords = np.clip(np.round(points[:, 0] * (width - 1)), 0, width - 1).astype(np.int32)
    y_coords = np.clip(np.round(points[:, 1] * (height - 1)), 0, height - 1).astype(np.int32)
    return np.stack((x_coords, y_coords), axis=1)


def _rotate_bbox(bbox, rotation_key):
    if not bbox or len(bbox) != 4:
        return None

    x_min, y_min, x_max, y_max = bbox
    corners = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float32,
    )

    rotated = _rotate_normalized_points(corners, rotation_key)
    if rotated.size == 0:
        return None

    x_vals = rotated[:, 0]
    y_vals = rotated[:, 1]

    return [
        float(np.clip(np.min(x_vals), 0.0, 1.0)),
        float(np.clip(np.min(y_vals), 0.0, 1.0)),
        float(np.clip(np.max(x_vals), 0.0, 1.0)),
        float(np.clip(np.max(y_vals), 0.0, 1.0)),
    ]


def generate_emissive_map(
    img_array,
    base_name,
    class_data=None,
    rotation_suffix="rot0",
    emissive_alpha_core=EMISSIVE_ALPHA_CORE,
    emissive_ring_ratio=EMISSIVE_ALPHA_RING_RATIO,
):
    """Genera un mapa emissive con fondo transparente y detección adaptativa de brillos."""
    if img_array.shape[2] == 4:
        rgb = img_array[:, :, :3]
        alpha_channel = img_array[:, :, 3]
    else:
        rgb = img_array
        alpha_channel = np.full(rgb.shape[:2], 255, dtype=np.uint8)

    height, width = rgb.shape[:2]
    emissive = np.zeros((height, width, 4), dtype=np.uint8)
    emissive_rgb = emissive[:, :, :3]
    emissive_alpha = emissive[:, :, 3]

    valid_pixels = alpha_channel > 0
    emissive_alpha[:] = EMISSIVE_ALPHA_BACKGROUND
    emissive_alpha[valid_pixels] = EMISSIVE_ALPHA_NON_TARGET

    class_masks = {}
    if class_data:
        for obj in class_data:
            obj_class = (obj.get("class") or "").lower()
            if obj_class not in {"eyes", "wings"}:
                continue

            polygon = obj.get("polygon") or []
            if len(polygon) < 3:
                continue

            polygon = np.array(polygon, dtype=np.float32)
            rotated_polygon = _rotate_normalized_points(polygon, rotation_suffix)
            pixel_points = _normalized_to_pixel(rotated_polygon, width, height)
            if pixel_points.size == 0:
                continue

            obj_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(obj_mask, [pixel_points.reshape(-1, 1, 2)], 255)

            bbox = obj.get("bbox")
            rotated_bbox = _rotate_bbox(bbox, rotation_suffix)
            if rotated_bbox:
                x_min = int(np.floor(rotated_bbox[0] * (width - 1)))
                y_min = int(np.floor(rotated_bbox[1] * (height - 1)))
                x_max = int(np.ceil(rotated_bbox[2] * (width - 1)))
                y_max = int(np.ceil(rotated_bbox[3] * (height - 1)))

                x_min = int(np.clip(x_min, 0, width - 1))
                y_min = int(np.clip(y_min, 0, height - 1))
                x_max = int(np.clip(x_max, 0, width - 1))
                y_max = int(np.clip(y_max, 0, height - 1))

                mask_crop = np.zeros_like(obj_mask)
                if x_min <= x_max and y_min <= y_max:
                    mask_crop[y_min : y_max + 1, x_min : x_max + 1] = obj_mask[
                        y_min : y_max + 1, x_min : x_max + 1
                    ]
                    obj_mask = mask_crop

            obj_mask = (obj_mask > 0) & valid_pixels
            if not np.any(obj_mask):
                continue

            if obj_class not in class_masks:
                class_masks[obj_class] = np.zeros((height, width), dtype=bool)

            class_masks[obj_class] |= obj_mask

    ring_alpha_value = int(np.clip(emissive_alpha_core * emissive_ring_ratio, 0, 255))

    if class_masks:
        for obj_class, mask in class_masks.items():
            if not np.any(mask):
                continue

            if obj_class == "eyes":
                color = EMISSIVE_COLOR_EYES
            else:
                color = EMISSIVE_COLOR_WINGS

            emissive_rgb[mask] = np.array(color[:3], dtype=np.uint8)
            emissive_alpha[mask] = emissive_alpha_core

            kernel = np.ones((3, 3), np.uint8)
            core_uint8 = mask.astype(np.uint8)
            ring = cv2.dilate(core_uint8, kernel, iterations=1).astype(bool) & ~mask & valid_pixels
            if np.any(ring):
                emissive_rgb[ring] = np.array(color[:3], dtype=np.uint8)
                emissive_alpha[ring] = np.maximum(emissive_alpha[ring], ring_alpha_value)

        emissive[alpha_channel == 0] = 0
        return emissive

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if np.any(valid_pixels):
        p90 = np.percentile(gray[valid_pixels], 90)
    else:
        p90 = 0

    threshold = 0.6 * p90 + 0.4 * 220
    bright_mask = (gray >= threshold) & valid_pixels

    if np.any(bright_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bright_mask.astype(np.uint8), connectivity=8
        )

        min_component_size = max(4, int(height * width * EMISSIVE_MIN_COMPONENT_PCT))
        final_mask = np.zeros_like(bright_mask, dtype=bool)

        for label_idx in range(1, num_labels):
            if stats[label_idx, cv2.CC_STAT_AREA] >= min_component_size:
                final_mask[labels == label_idx] = True

        final_mask &= valid_pixels
    else:
        final_mask = np.zeros_like(valid_pixels, dtype=bool)

    if not np.any(final_mask):
        emissive[alpha_channel == 0] = 0
        return emissive

    lower_name = (base_name or "").lower()
    if "eyes" in lower_name or "eye" in lower_name:
        emissive_color = EMISSIVE_COLOR_EYES
    elif "wing" in lower_name or "ala" in lower_name:
        emissive_color = EMISSIVE_COLOR_WINGS
    elif "crystal" in lower_name or "cristal" in lower_name or "glow" in lower_name:
        emissive_color = (80, 255, 120, 255)
    else:
        emissive_color = (255, 255, 255, 255)

    emissive_rgb[final_mask] = np.array(emissive_color[:3], dtype=np.uint8)
    emissive_alpha[final_mask] = emissive_alpha_core

    kernel = np.ones((3, 3), np.uint8)
    core_uint8 = final_mask.astype(np.uint8)
    ring = cv2.dilate(core_uint8, kernel, iterations=1).astype(bool) & ~final_mask & valid_pixels
    if np.any(ring):
        emissive_rgb[ring] = np.array(emissive_color[:3], dtype=np.uint8)
        emissive_alpha[ring] = np.maximum(emissive_alpha[ring], ring_alpha_value)

    emissive[alpha_channel == 0] = 0

    return emissive


def save_pbr_maps(result_img, output_name, variants_dir):
    """Genera y guarda mapas PBR asociados a una variante."""
    if result_img.mode == "RGBA":
        img_array = np.array(result_img)
    else:
        img_array = np.array(result_img.convert("RGBA"))

    base_name, rotation_suffix = _parse_variant_metadata(output_name)
    class_data = _get_polygon_objects(base_name)

    if USE_BACKGROUND_MATERIALS:
        try:
            bg_rgba, bg_normal, bg_specular, bg_material = load_background_layers(
                base_name,
                (img_array.shape[1], img_array.shape[0]),
                BACKGROUNDS_DIR,
            )
        except Exception as exc:
            logging.error(
                f"No se pudieron cargar fondos reales para {output_name}: {exc}"
            )
            raise
        else:
            if bg_normal is not None and bg_normal.shape[:2] != img_array.shape[:2]:
                bg_normal = cv2.resize(
                    bg_normal,
                    (img_array.shape[1], img_array.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            if bg_specular is not None and bg_specular.shape[:2] != img_array.shape[:2]:
                bg_specular = cv2.resize(
                    bg_specular,
                    (img_array.shape[1], img_array.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

        normal_map, specular_map, emissive_map = generate_enhanced_pbr_maps(
            img_array,
            (bg_rgba, bg_normal, bg_specular),
            base_name,
            rotation_suffix=rotation_suffix,
            class_data=class_data,
            background_material=bg_material,
        )
    else:
        normal_map = generate_normal_map(img_array, intensity=PBR_INTENSITY)
        specular_map = generate_specular_map(img_array, sharpness=SPECULAR_SHARPNESS)
        emissive_map = generate_emissive_map(
            img_array,
            base_name,
            class_data=class_data,
            rotation_suffix=rotation_suffix,
        )

    variants_path = Path(variants_dir)
    pbr_dir = variants_path / "pbr_maps"
    pbr_dir.mkdir(exist_ok=True)

    stem = Path(output_name).stem
    suffix = Path(output_name).suffix or ".png"

    normal_filename = f"{stem}_normal{suffix}"
    specular_filename = f"{stem}_specular{suffix}"
    emissive_filename = f"{stem}_emissive{suffix}"

    normal_map = np.ascontiguousarray(normal_map)
    if normal_map.shape[2] == 3:
        normal_out = cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR)
    else:
        normal_out = cv2.cvtColor(normal_map, cv2.COLOR_RGBA2BGRA)
    normal_out = np.ascontiguousarray(normal_out)

    cv2.imwrite(
        str(pbr_dir / normal_filename),
        normal_out,
        PNG_PARAMS,
    )

    if specular_map.shape[2] == 3:
        alpha_full = np.full(specular_map.shape[:2], 255, dtype=np.uint8)
        specular_map = np.dstack([specular_map, alpha_full])
    if emissive_map.shape[2] == 3:
        alpha_full = np.full(emissive_map.shape[:2], 255, dtype=np.uint8)
        emissive_map = np.dstack([emissive_map, alpha_full])

    specular_map = np.ascontiguousarray(specular_map)
    emissive_map = np.ascontiguousarray(emissive_map)

    specular_bgra = cv2.cvtColor(specular_map, cv2.COLOR_RGBA2BGRA)
    emissive_bgra = cv2.cvtColor(emissive_map, cv2.COLOR_RGBA2BGRA)

    specular_bgra = np.ascontiguousarray(specular_bgra)
    emissive_bgra = np.ascontiguousarray(emissive_bgra)

    cv2.imwrite(
        str(pbr_dir / specular_filename),
        specular_bgra,
        PNG_PARAMS,
    )
    cv2.imwrite(
        str(pbr_dir / emissive_filename),
        emissive_bgra,
        PNG_PARAMS,
    )

    return [normal_filename, specular_filename, emissive_filename]


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
