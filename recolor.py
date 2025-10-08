import os
from os.path import basename
import json
import logging
import random
from pathlib import Path
from PIL import Image
import numpy as np
from colorsys import rgb_to_hls, hls_to_rgb
from collections import defaultdict
from PIL import ImageFilter
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import psutil
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
    "SAMPLING_THRESHOLD": 80000,
    "BUCKET_SIZE": 14,
    "VALID_EXTENSIONS": {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"},
    "PRESERVE_BRIGHTNESS": False,
    "USE_HLS_METHOD": True,
    "GENERATE_ROTATIONS": True,
    "ROTATION_ANGLES": [0, 90, 180, 270],
    "MIN_COLOR_DIFFERENCE": 8.0,
    "HUE_VARIATION_RANGE": [-0.3, 0.3],
    "SATURATION_VARIATION_RANGE": [-0.7, 0.7],
    "LIGHTNESS_VARIATION_RANGE": [-0.5, 0.5],
    "MAX_GENERATION_ATTEMPTS": 1500,
    "SIMILARITY_THRESHOLD": 0.85,
    "ENSURE_MIN_VARIATIONS": True,
    "RANDOM_COLOR_PROBABILITY": 0.3,
    "CROSS_COLOR_MIXING": True,
    "MAX_COLOR_MIXES": 128,
    "NOISE_INTENSITY": 0.1,
    "MAX_WORKERS": 10,
    "CHUNK_SIZE": 1000,
    "MEMORY_MONITORING": True,
    "MAX_MEMORY_PERCENT": 95,
    "AUTO_ADJUST_CHUNKS": True,
    "TARGET_MEMORY_GB": 13.0,
    "MEMORY_PER_IMAGE_MB": 4.0,
}

GENERATE_PBR_MAPS = True
PBR_INTENSITY = 2.0
SPECULAR_SHARPNESS = 1.5

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


def generate_synthetic_background(size):
    width, height = size

    base_color = tuple(np.random.randint(50, 220, size=3))
    bg = Image.new("RGB", size, base_color)

    if random.random() < 0.5:
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        bg = Image.fromarray(np.clip(np.array(bg) + noise, 0, 255).astype("uint8"), "RGB")

    bg_dir = "backgrounds"
    if os.path.exists(bg_dir) and random.random() < 0.6:
        textures = [f for f in os.listdir(bg_dir) if f.lower().endswith((".jpg", ".png"))]
        if textures:
            chosen = random.choice(textures)
            tex = Image.open(os.path.join(bg_dir, chosen)).convert("RGB").resize(size)
            bg = Image.blend(bg, tex, alpha=random.uniform(0.2, 0.5))
            if basename(chosen).lower() in {"a.png", "b.png", "c.png"}:
                bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3)))

    if random.random() < 0.4:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))

    return bg


def process_rotated_image(rot_img, base_name, rot_suffix, colors, output_dir):
    try:
        rot_img = rot_img.convert("RGBA")
        img_array = np.array(rot_img)

        dominant_colors = get_dominant_colors(rot_img, CONFIG["MAX_DOMINANT_COLORS"])
        num_dominant_colors = len(dominant_colors)

        logging.info(f"Procesando rotación {rot_suffix} - {num_dominant_colors} colores dominantes")

        generated_images = []

        for color_idx, base_color in enumerate(colors):
            color_variations = generate_high_variation_colors(
                base_color,
                CONFIG["MAX_VARIATIONS_PER_COLOR"],
                colors
            )

            color_mapping = {}
            for i, original_color in enumerate(dominant_colors):
                if CONFIG["CROSS_COLOR_MIXING"] and len(color_variations) > 1:
                    base_variation = color_variations[i % len(color_variations)]
                    if random.random() < 0.4:
                        other_variation = color_variations[(i + random.randint(1, len(color_variations)-1)) % len(color_variations)]
                        mix_ratio = random.uniform(0.2, 0.8)
                        color_mapping[original_color] = mix_colors(base_variation, other_variation, mix_ratio)
                    else:
                        color_mapping[original_color] = base_variation
                else:
                    color_mapping[original_color] = color_variations[i % len(color_variations)]

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
                            key=lambda c: sum(
                                (a - b) ** 2 for a, b in zip(original_rgb, c[:3])
                            ),
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

            generated_images.append((new_array, output_name))

        for img_array, output_name in generated_images:
            result_img = Image.fromarray(img_array.astype("uint8"), "RGBA")

            bg = generate_synthetic_background(result_img.size)
            bg = adjust_background_contrast(bg, result_img)

            final_img = Image.alpha_composite(bg.convert("RGBA"), result_img)
            output_path = os.path.join(output_dir, output_name)
            final_img.save(output_path, optimize=True)

            if GENERATE_PBR_MAPS:
                try:
                    save_pbr_maps(result_img, output_name, output_dir)
                except Exception as pbr_error:
                    logging.error(
                        f"Error generando mapas PBR para {output_name}: {pbr_error}",
                        exc_info=True,
                    )

        logging.info(f"Generadas {len(generated_images)} variantes para rotación {rot_suffix}")

        # Liberar memoria explícitamente
        del img_array, new_array, generated_images
        gc.collect()

    except Exception as e:
        logging.error(f"Error procesando rotación {rot_suffix}: {str(e)}", exc_info=True)

def process_single_image(args):
    image_path, colors, output_dir = args
    
    try:
        with Image.open(image_path) as img:
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            if CONFIG["GENERATE_ROTATIONS"]:
                rotations = generate_rotations(img)
                for rot_suffix, rot_img in rotations.items():
                    process_rotated_image(rot_img, base_name, rot_suffix, colors, output_dir)
            else:
                process_rotated_image(img, base_name, "", colors, output_dir)

            return f"Completada: {base_name}"

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

def safe_process_image_batch(image_batch, colors, output_dir):
    if CONFIG["MEMORY_MONITORING"]:
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > CONFIG["MAX_MEMORY_PERCENT"]:
            reduced_batch = image_batch[:len(image_batch)//2]
            logging.warning(f"Memoria alta ({memory_percent}%), reduciendo chunk a {len(reduced_batch)} imágenes")
            return process_image_batch(reduced_batch, colors, output_dir)
    
    return process_image_batch(image_batch, colors, output_dir)

def calculate_dynamic_chunk_size(total_images, max_workers):
    # Usar el mínimo entre memoria disponible y TARGET_MEMORY_GB
    available_memory = min(
        CONFIG["TARGET_MEMORY_GB"], 
        psutil.virtual_memory().available / (1024**3)
    )
    
    memory_per_image = CONFIG["MEMORY_PER_IMAGE_MB"] / 1024  # Convertir MB a GB
    max_chunk_by_memory = int((available_memory * 0.8) / (memory_per_image * max_workers))
    
    # También considerar límite por número de imágenes
    dynamic_chunk = min(CONFIG["CHUNK_SIZE"], max_chunk_by_memory, total_images)
    
    logging.info(f"Chunk size dinámico: {dynamic_chunk} (memoria objetivo: {CONFIG['TARGET_MEMORY_GB']}GB, disponible: {available_memory:.1f}GB)")
    return max(1, dynamic_chunk)

def optimize_memory_usage():
    """Optimiza el uso de memoria del proceso actual"""
    try:
        import resource
        # En sistemas Unix: aumentar límite de memoria
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_limit = int(CONFIG["TARGET_MEMORY_GB"] * 1024**3)  # Convertir a bytes
        resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
    except:
        pass  # No disponible en Windows

def main():
    if not validate_config():
        return

    if not COLOR_MATH_AVAILABLE:
        logging.warning("colormath no disponible. Usando método alternativo")

    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    # Optimizar uso de memoria
    optimize_memory_usage()

    base_colors = load_colors(CONFIG["COLORS_JSON"])
    if len(base_colors) < CONFIG["MIN_COLOR"]:
        additional_colors = generate_random_colors(
            base_colors, CONFIG["MIN_COLOR"] - len(base_colors)
        )
        base_colors.extend(additional_colors)

    logging.info(f"Total de colores: {len(base_colors)}")
    logging.info(f"Usando {CONFIG['MAX_WORKERS']} procesos paralelos")
    logging.info(f"Objetivo de memoria: {CONFIG['TARGET_MEMORY_GB']}GB")

    image_files = [
        os.path.join(CONFIG["INPUT_DIR"], f)
        for f in os.listdir(CONFIG["INPUT_DIR"])
        if os.path.isfile(os.path.join(CONFIG["INPUT_DIR"], f))
        and os.path.splitext(f)[1].lower() in CONFIG["VALID_EXTENSIONS"]
    ]

    if not image_files:
        logging.warning("No se encontraron imágenes válidas")
        return

    if CONFIG["AUTO_ADJUST_CHUNKS"]:
        dynamic_chunk_size = calculate_dynamic_chunk_size(len(image_files), CONFIG["MAX_WORKERS"])
        chunk_size = dynamic_chunk_size
    else:
        chunk_size = CONFIG["CHUNK_SIZE"]

    image_batches = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]

    logging.info(f"Procesando {len(image_files)} imágenes en {len(image_batches)} lotes")

    with ProcessPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        futures = []
        
        for batch in image_batches:
            future = executor.submit(safe_process_image_batch, batch, base_colors, CONFIG["OUTPUT_DIR"])
            futures.append(future)
        
        completed = 0
        for future in as_completed(futures):
            batch_results = future.result()
            for result in batch_results:
                logging.info(result)
                completed += 1
                progress_percent = (completed / len(image_files)) * 100
                logging.info(f"Progreso: {completed}/{len(image_files)} imágenes ({progress_percent:.1f}%)")

    logging.info("Procesamiento completado")

def validate_config():
    errors = []

    if CONFIG["MIN_COLOR"] < 1:
        errors.append("MIN_COLOR debe ser al menos 1")
    
    if CONFIG["COLOR_RANGE"]["min"] >= CONFIG["COLOR_RANGE"]["max"]:
        errors.append("COLOR_RANGE min debe ser menor que max")
    
    if CONFIG["MAX_WORKERS"] > mp.cpu_count():
        CONFIG["MAX_WORKERS"] = max(1, mp.cpu_count() - 1)
        logging.info(f"Ajustando MAX_WORKERS a {CONFIG['MAX_WORKERS']}")

    if errors:
        for error in errors:
            logging.error(f"Error de configuración: {error}")
        return False
    return True


def generate_normal_map(img_array, intensity=None):
    """Genera un mapa normal a partir de una imagen RGBA/RGB."""
    if intensity is None:
        intensity = PBR_INTENSITY

    if img_array.shape[2] == 4:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    gray = gray.astype(np.float32) / 255.0
    gray = ndimage.gaussian_filter(gray, sigma=1)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    gx *= intensity
    gy *= intensity

    gz = np.ones_like(gray)
    normal = np.dstack((gx, gy, gz))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    norm[norm == 0] = 1.0
    normal = normal / norm

    normal = (normal + 1.0) * 127.5
    return normal.astype(np.uint8)


def generate_specular_map(img_array, sharpness=None):
    """Genera un mapa specular resaltando zonas de alta frecuencia."""
    if sharpness is None:
        sharpness = SPECULAR_SHARPNESS

    if img_array.shape[2] == 4:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian = np.abs(laplacian)

    specular = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    specular = np.clip(specular * sharpness, 0, 255).astype(np.uint8)

    return np.dstack([specular, specular, specular])


def generate_emissive_map(img_array, base_name, class_data=None):
    """Genera un mapa emissive basado en zonas brillantes y nombre base."""
    rgb_array = img_array[:, :, :3] if img_array.shape[2] == 4 else img_array

    emissive = np.zeros_like(rgb_array)

    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    brightness = hsv[:, :, 2]
    bright_mask = brightness > 200

    lower_name = base_name.lower()
    if "eyes" in lower_name:
        emissive[bright_mask] = [255, 100, 50]
    elif "wings" in lower_name or "ala" in lower_name:
        emissive[bright_mask] = [50, 100, 255]
    elif "cristal" in lower_name or "crystal" in lower_name:
        emissive[bright_mask] = [50, 255, 100]
    else:
        emissive[bright_mask] = [255, 255, 255]

    return emissive


def save_pbr_maps(result_img, output_name, variants_dir):
    """Genera y guarda mapas PBR asociados a una variante."""
    if result_img.mode == "RGBA":
        img_array = np.array(result_img)
    else:
        img_array = np.array(result_img.convert("RGBA"))

    base_name = Path(output_name).stem
    for suffix in ["_color", "_rot90", "_rot180", "_rot270"]:
        base_name = base_name.replace(suffix, "")

    normal_map = generate_normal_map(img_array, intensity=PBR_INTENSITY)
    specular_map = generate_specular_map(img_array, sharpness=SPECULAR_SHARPNESS)
    emissive_map = generate_emissive_map(img_array, base_name)

    variants_path = Path(variants_dir)
    pbr_dir = variants_path / "pbr_maps"
    pbr_dir.mkdir(exist_ok=True)

    stem = Path(output_name).stem
    suffix = Path(output_name).suffix or ".png"

    normal_filename = f"{stem}_normal{suffix}"
    specular_filename = f"{stem}_specular{suffix}"
    emissive_filename = f"{stem}_emissive{suffix}"

    cv2.imwrite(
        str(pbr_dir / normal_filename),
        cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(pbr_dir / specular_filename),
        cv2.cvtColor(specular_map, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(pbr_dir / emissive_filename),
        cv2.cvtColor(emissive_map, cv2.COLOR_RGB2BGR),
    )

    return [normal_filename, specular_filename, emissive_filename]


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
