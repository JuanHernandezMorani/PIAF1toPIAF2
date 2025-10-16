import gc
import io
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock
from colorsys import hls_to_rgb, rgb_to_hls

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

try:
    from colormath.color_objects import sRGBColor, LabColor
    from colormath.color_conversions import convert_color
    from colormath.color_diff import delta_e_cie2000
    COLOR_MATH_AVAILABLE = True
except ImportError:
    COLOR_MATH_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - dependencia opcional
    KMeans = None
    SKLEARN_AVAILABLE = False

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


class RetryMechanism:
    """Implements exponential backoff with jitter for filesystem operations."""

    DEFAULT_EXCEPTIONS = (PermissionError, OSError)

    @staticmethod
    def execute(operation, max_retries=5, base_delay=0.1, exceptions=None):
        if exceptions is None:
            exceptions = RetryMechanism.DEFAULT_EXCEPTIONS

        last_error = None
        for attempt in range(max_retries):
            try:
                return operation()
            except exceptions as exc:  # pragma: no cover - requiere entorno concurrencia real
                last_error = exc
                if attempt == max_retries - 1:
                    logging.error(
                        "Operación de archivo falló tras %d reintentos: %s",
                        max_retries,
                        exc,
                    )
                    raise
                delay = base_delay * (2**attempt)
                delay += random.uniform(0, base_delay)
                logging.debug(
                    "Reintentando operación de archivo (intento %d/%d). Esperando %.3fs. Error: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
        if last_error is not None:
            raise last_error


class AtomicOperationQueue:
    """Serializa operaciones críticas por clave dentro del mismo proceso."""

    def __init__(self):
        self._lock = Lock()
        self._locks = {}

    @contextmanager
    def acquire(self, key):
        key = str(key)
        with self._lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = Lock()
                self._locks[key] = lock
        logging.debug("Adquiriendo lock de cola atómica para %s (pid=%s)", key, os.getpid())
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            logging.debug("Liberando lock de cola atómica para %s (pid=%s)", key, os.getpid())


ATOMIC_OPERATION_QUEUE = AtomicOperationQueue()


class AtomicPNGWriter:
    """Guarda imágenes PNG usando buffers en memoria y escritura atómica."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self._thread_lock = Lock()
        self._process_id = os.getpid()

    def _ensure_directory(self):
        def _ensure():
            self.directory.mkdir(parents=True, exist_ok=True)

        RetryMechanism.execute(_ensure)

    @staticmethod
    def _prepare_image(image, filename):
        if isinstance(image, Image.Image):
            return image
        try:
            return Image.fromarray(np.array(image, dtype=np.uint8))
        except Exception as exc:
            raise ValueError(f"Entrada inválida al guardar {filename}: {exc}") from exc

    @staticmethod
    def _encode_png(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    def _write_bytes(self, final_path, data, allow_replace):
        flags = os.O_WRONLY | os.O_CREAT
        if allow_replace:
            flags |= os.O_TRUNC
        else:
            flags |= os.O_EXCL
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY

        mode = 0o666
        fd = os.open(str(final_path), flags, mode)
        try:
            total = 0
            length = len(data)
            while total < length:
                written = os.write(fd, data[total:])
                if written == 0:
                    raise OSError("Escritura incompleta al guardar la imagen")
                total += written
            if hasattr(os, "fsync"):
                os.fsync(fd)
        finally:
            os.close(fd)

    def write_image(self, image, filename, allow_replace=True):
        final_path = self.directory / filename
        with self._thread_lock:
            self._ensure_directory()

        if final_path.exists() and not allow_replace:
            logging.warning(
                "Archivo destino ya existe y no se permite reemplazo: %s (pid=%s)",
                final_path,
                self._process_id,
            )
            return None

        try:
            prepared = self._prepare_image(image, final_path.name)
            data = self._encode_png(prepared)
        except ValueError as exc:
            logging.error("%s (pid=%s)", exc, self._process_id)
            return None
        except Exception as exc:  # pragma: no cover - errores al codificar
            logging.error(
                "Error al codificar imagen %s (pid=%s): %s",
                final_path,
                self._process_id,
                exc,
            )
            return None

        try:
            self._write_bytes(final_path, data, allow_replace)
        except FileExistsError:
            logging.warning(
                "Archivo destino ya existe y no se puede sobrescribir: %s (pid=%s)",
                final_path,
                self._process_id,
            )
            return None
        except Exception as exc:
            logging.error(
                "No se pudo guardar imagen %s (pid=%s): %s",
                final_path,
                self._process_id,
                exc,
            )
            return None

        FILE_MANIFEST.register(final_path)
        logging.info("Imagen guardada %s (pid=%s)", final_path, self._process_id)
        return str(final_path)

    def remove_file(self, path):
        resolved = Path(path).resolve()
        try:
            resolved.unlink(missing_ok=True)  # type: ignore[arg-type]
        except TypeError:
            if resolved.exists():
                resolved.unlink()
        except Exception as exc:
            logging.warning(
                "No se pudo eliminar archivo %s (pid=%s): %s",
                resolved,
                self._process_id,
                exc,
            )
        else:
            logging.debug(
                "Archivo eliminado o inexistente %s (pid=%s)",
                resolved,
                self._process_id,
            )



class VariantFileManifest:
    """Thread-safe manifest tracking generated variant files."""

    def __init__(self):
        self._lock = Lock()
        self._files = set()

    def _resolve(self, path):
        return Path(path).resolve()

    def register(self, path):
        resolved = self._resolve(path)
        if not resolved.exists():
            logging.warning(
                "Intento de registrar archivo inexistente en el manifiesto: %s",
                resolved,
            )
            return
        with self._lock:
            self._files.add(resolved)

    def discard(self, path):
        resolved = self._resolve(path)
        with self._lock:
            self._files.discard(resolved)

    def ensure_exists(self, path):
        resolved = self._resolve(path)
        if resolved.exists():
            with self._lock:
                self._files.add(resolved)
            return True
        with self._lock:
            self._files.discard(resolved)
        return False

    def filter_existing(self, paths, context=""):
        existing = []
        for candidate in paths:
            if not candidate:
                logging.warning(
                    "Entrada vacía omitida%s",
                    f" ({context})" if context else "",
                )
                continue
            if self.ensure_exists(candidate):
                existing.append(str(self._resolve(candidate)))
            else:
                logging.warning(
                    "Archivo faltante omitido%s: %s",
                    f" ({context})" if context else "",
                    Path(candidate),
                )
        return existing

    def list_for_directory(self, directory):
        resolved_dir = Path(directory).resolve()
        with self._lock:
            return [
                str(path)
                for path in self._files
                if path.parent == resolved_dir or resolved_dir in path.parents
            ]

    def initialize_from_directory(self, directory):
        resolved_dir = Path(directory).resolve()
        if not resolved_dir.exists():
            return
        count = 0
        with self._lock:
            for file_path in resolved_dir.rglob("*.png"):
                resolved = file_path.resolve()
                self._files.add(resolved)
                count += 1
        if count:
            logging.info(
                "Manifiesto inicializado con %d archivos existentes en %s",
                count,
                resolved_dir,
            )


class VariantFileIORegistry:
    """Registro de gestores de directorios para operaciones de guardado."""

    def __init__(self):
        self._lock = Lock()
        self._writers = {}

    def for_directory(self, directory):
        resolved = Path(directory).resolve()
        with self._lock:
            writer = self._writers.get(resolved)
            if writer is None:
                writer = AtomicPNGWriter(resolved)
                logging.debug(
                    "Instanciado gestor de archivos para %s (pid=%s)",
                    resolved,
                    os.getpid(),
                )
                self._writers[resolved] = writer
        return writer

    def save_image(self, image, directory, filename, allow_replace=True):
        target_path = Path(directory).resolve() / filename
        with ATOMIC_OPERATION_QUEUE.acquire(str(target_path)):
            writer = self.for_directory(directory)
            result = writer.write_image(image, filename, allow_replace=allow_replace)
        return result

    def remove_file(self, path):
        resolved_path = Path(path).resolve()
        writer = self.for_directory(resolved_path.parent)
        writer.remove_file(resolved_path)
        FILE_MANIFEST.discard(resolved_path)


FILE_MANIFEST = VariantFileManifest()
VARIANT_IO = VariantFileIORegistry()


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


REQUIRED_GROUPS = [
    "base",
    "low_contrast",
    "high_contrast",
    "illumination",
    "concise_colors",
]


# Se asignará más adelante con la implementación completa de rebalanceo
REBALANCE_VARIANTS = None


def fix_variant_distribution():
    """Synchroniza la configuración de variantes con el plan perceptual objetivo."""

    perceptual_cfg = CONFIG.setdefault("PERCEPTUAL_VARIANT_SYSTEM", {})
    rotations_enabled = CONFIG.get("GENERATE_ROTATIONS", True)
    rotation_angles = CONFIG.get("ROTATION_ANGLES", [0]) if rotations_enabled else [0]
    rotation_count = max(len(rotation_angles), 1)

    variants_per_group = perceptual_cfg.get("VARIANTS_PER_GROUP") or 25
    variants_per_group = max(int(variants_per_group), 1)

    active_groups = [group for group in REQUIRED_GROUPS]
    groups_count = max(len(active_groups), 1)

    total_variants_per_rotation = variants_per_group * groups_count
    total_variants = total_variants_per_rotation * rotation_count

    perceptual_cfg["VARIANTS_PER_GROUP"] = variants_per_group
    perceptual_cfg["TOTAL_VARIANTS_PER_ROTATION"] = total_variants_per_rotation
    perceptual_cfg["TOTAL_VARIANTS"] = total_variants

    return total_variants_per_rotation, total_variants, rotation_count


TOTAL_VARIANTS_PER_ROTATION, TOTAL_VARIANTS_EXPECTED, TOTAL_ROTATION_COUNT = fix_variant_distribution()


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


def detect_missing_groups(current_groups):
    """Identifica grupos requeridos faltantes y registra alertas críticas."""
    required = set(REQUIRED_GROUPS)
    current = set(current_groups)
    missing = required - current

    if missing:
        logging.error(f"GRUPOS FALTANTES CRÍTICOS: {sorted(missing)}")
        if "illumination" in missing:
            logging.critical(
                "Grupo 'illumination' no se está generando. Revisar apply_group_specific_processing"
            )

    return missing


def attempt_group_rebalancing(variants, current_counts, target_count):
    """Intenta rebalancear automáticamente la distribución de grupos."""

    target_distribution = {group: target_count for group in current_counts}

    missing = {
        group: target_count - current_counts.get(group, 0)
        for group in target_distribution
        if current_counts.get(group, 0) < target_count
    }
    surplus = {
        group: current_counts.get(group, 0) - target_count
        for group in current_counts
        if current_counts.get(group, 0) > target_count
    }

    if missing:
        logging.warning(
            "Intento de rebalanceo requerido. Grupos con déficit: %s, exceso: %s",
            missing,
            surplus,
        )

    if not variants or REBALANCE_VARIANTS is None:
        return False

    try:
        success = REBALANCE_VARIANTS(variants, target_distribution)
        if success:
            logging.info("Rebalanceo completado exitosamente: %s", target_distribution)
        else:
            logging.warning("No se logró el rebalanceo completo para: %s", target_distribution)
        return success
    except Exception as exc:
        logging.error("Error durante el rebalanceo automático: %s", exc)
        return False


def ensure_complete_group_distribution(variant_count=None, groups_count=None):
    """Garantiza distribución completa y validable de grupos perceptuales."""

    perceptual_cfg = CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {})

    if variant_count is None:
        variant_count = perceptual_cfg.get(
            "TOTAL_VARIANTS_PER_ROTATION",
            perceptual_cfg.get("TOTAL_VARIANTS", 125),
        )

    if groups_count is None:
        groups_count = len(REQUIRED_GROUPS)

    active_groups = REQUIRED_GROUPS[:groups_count]
    variants_per_group = perceptual_cfg.get("VARIANTS_PER_GROUP")
    if not variants_per_group:
        variants_per_group = max(1, variant_count // max(len(active_groups), 1))

    target_distribution = {group: variants_per_group for group in active_groups}
    guaranteed_total = sum(target_distribution.values())

    group_plan = [
        group
        for group in active_groups
        for _ in range(variants_per_group)
    ]

    def assign_variant_group_fixed(variant_index, total_variants=None):
        total = total_variants or guaranteed_total
        if total <= 0:
            return active_groups[0]

        plan_index = variant_index % len(group_plan)
        return group_plan[plan_index]

    def extract_group_from_variant(variant):
        if isinstance(variant, str):
            return extract_group_from_filename(os.path.basename(variant))
        return extract_group_from_filename(str(variant))

    def validate_group_distribution(variants, total_variants=None):
        counts = {group: 0 for group in active_groups}
        for variant in variants:
            group = extract_group_from_variant(variant)
            if group in counts:
                counts[group] += 1

        validation_result = all(
            counts.get(group, 0) == target_distribution.get(group, 0)
            for group in active_groups
        )

        if not validation_result:
            logging.warning(f"Distribución de grupos: {counts}")
            expected_target = (
                target_distribution.get(active_groups[0], variants_per_group)
                if active_groups
                else variants_per_group
            )
            rebalance_success = attempt_group_rebalancing(
                variants, counts, expected_target
            )
            if rebalance_success:
                counts = {group: 0 for group in active_groups}
                for variant in variants:
                    group = extract_group_from_variant(variant)
                    if group in counts:
                        counts[group] += 1
                validation_result = all(
                    counts.get(group, 0) == target_distribution.get(group, 0)
                    for group in active_groups
                )

        expected_return = (
            target_distribution.get(active_groups[0], variants_per_group)
            if active_groups
            else variants_per_group
        )

        return validation_result, counts, expected_return

    return assign_variant_group_fixed, validate_group_distribution, target_distribution


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

def remove_problematic_unicode(text):
    import re

    pattern = re.compile(r"[^\u0000-\uFFFF]")
    return pattern.sub("", text)


def setup_cross_platform_logging():
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if sys.platform == "win32":
        handler = logging.StreamHandler(sys.stderr)
        try:
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            else:
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer,
                    encoding="utf-8",
                    errors="replace",
                )
        except Exception:
            class SafeHandler(logging.StreamHandler):
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        msg = remove_problematic_unicode(msg)
                        stream = self.stream
                        stream.write(msg + self.terminator)
                        self.flush()
                    except Exception:
                        pass

            handler = SafeHandler(sys.stderr)
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        file_handler = logging.FileHandler("processing.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as error:
        logger.warning(f"No fue posible iniciar logging en archivo: {error}")


setup_cross_platform_logging()

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
    variations = [list(base_color)]
    r, g, b, a = base_color
    h_base, l_base, s_base = rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)

    attempts = 0
    generated_count = 1
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


def rgb_to_lab(image):
    """Convierte una imagen RGB/RGBA a espacio Lab perceptual."""
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        rgb = arr

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab.astype(np.float32)


def lab_to_rgb(lab_image):
    """Convierte una imagen Lab a RGB uint8."""
    lab = np.asarray(lab_image, dtype=np.float32)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def srgb_to_linear(image):
    """Convierte valores sRGB (0-255) a espacio lineal [0,1]."""
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mask = arr <= 0.04045
    linear = np.empty_like(arr, dtype=np.float32)
    linear[mask] = arr[mask] / 12.92
    linear[~mask] = np.power((arr[~mask] + 0.055) / 1.055, 2.4)
    return np.clip(linear, 0.0, 1.0)


def linear_to_srgb(image):
    """Convierte valores lineales [0,1] a sRGB uint8."""
    arr = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    mask = arr <= 0.0031308
    srgb = np.empty_like(arr, dtype=np.float32)
    srgb[mask] = arr[mask] * 12.92
    srgb[~mask] = 1.055 * np.power(arr[~mask], 1.0 / 2.4) - 0.055
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def detect_magnocellular_edges(image):
    """Detecta bordes de alta frecuencia simulando respuesta magnocelular."""
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 32, 128)
    edges = cv2.GaussianBlur(edges.astype(np.float32), (3, 3), 0)
    return edges.astype(np.float32)


def enhanced_edge_detection(image):
    """Deducción multiescala de bordes perceptuales empleando luminancia Lab."""

    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        rgb = np.clip(arr, 0, 255).astype(np.uint8)

    lab_image = rgb_to_lab(rgb)
    luminance = np.clip(lab_image[:, :, 0], 0, 255).astype(np.uint8)

    edges_fine = cv2.Canny(luminance, 50, 150)
    edges_coarse = cv2.Canny(luminance, 30, 100)
    combined = np.maximum(edges_fine, edges_coarse).astype(np.float32)
    combined = cv2.GaussianBlur(combined, (5, 5), 0)

    return np.clip(combined / 255.0, 0.0, 1.0)


def add_perceptual_edge_enhancement(image, edge_strength, enhancement_factor=0.3):
    """Refuerza bordes relevantes respetando el canal alfa si existe."""

    arr = np.asarray(image, dtype=np.float32)

    if arr.ndim == 2:
        rgb = np.stack([arr] * 3, axis=-1)
        alpha = None
    elif arr.shape[2] == 4:
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3:4]
    else:
        rgb = arr[:, :, :3]
        alpha = None

    edge_map = np.asarray(edge_strength, dtype=np.float32)
    if edge_map.ndim == 2:
        edge_map = edge_map[:, :, None]
    edge_map = np.clip(edge_map, 0.0, 1.0)

    enhancement = edge_map * 255.0 * float(max(enhancement_factor, 0.0))
    enhanced_rgb = np.clip(rgb + enhancement, 0.0, 255.0).astype(np.uint8)

    if alpha is not None:
        alpha_channel = np.clip(alpha, 0.0, 255.0).astype(np.uint8)
        return np.concatenate([enhanced_rgb, alpha_channel], axis=2)

    return enhanced_rgb


def enhance_parvocellular_color(image, factor):
    """Aumenta la saturación y detalle fino emulando células P."""
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation_scale = 1.0 + 0.5 * max(factor - 1.0, 0.0)
    value_scale = 1.0 + 0.3 * max(factor - 1.0, 0.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_scale, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return enhanced


def blend_vision_systems(color_image, edge_map, factor):
    """Combina la respuesta parvocelular con bordes magnocelulares."""
    color = np.asarray(color_image, dtype=np.float32)
    if color.ndim == 3 and color.shape[2] == 4:
        alpha = color[:, :, 3]
        color = color[:, :, :3]
    else:
        alpha = None

    if edge_map.ndim == 2:
        edges = edge_map[:, :, None]
    else:
        edges = edge_map

    edges = np.clip(edges / 255.0, 0.0, 1.0)
    enhancement_strength = min(max(factor - 1.0, 0.0), 2.0)
    blended_rgb = color + edges * 255.0 * 0.5 * enhancement_strength
    blended_rgb = np.clip(blended_rgb, 0, 255)
    blended_rgb = blended_rgb.astype(np.uint8)

    if alpha is not None:
        return np.dstack([blended_rgb, alpha.astype(np.uint8)])
    return blended_rgb


def apply_scotopic_adaptation(image, factor):
    """Simula visión nocturna reduciendo saturación y preservando luminancia."""
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
    else:
        alpha = None
        rgb = arr

    lab = rgb_to_lab(rgb)
    lab[:, :, 1] *= factor * 0.7
    lab[:, :, 2] *= factor * 0.7
    lab[:, :, 1:] = np.clip(lab[:, :, 1:], 0, 255)
    rgb_result = lab_to_rgb(lab)

    if alpha is not None:
        return np.dstack([rgb_result, alpha])
    return rgb_result


def apply_photopic_enhancement(image, factor):
    """Simula visión fotópica realzando bordes y saturación."""
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
    else:
        alpha = None
        rgb = arr

    edges = detect_magnocellular_edges(rgb)
    enhanced = enhance_parvocellular_color(rgb, factor)
    blended = blend_vision_systems(enhanced, edges, factor)

    if alpha is not None:
        return np.dstack([blended, alpha])
    return blended


def simulate_human_contrast_perception(image, contrast_level):
    """Aplica contraste basado en sensibilidad humana."""
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3]
    else:
        alpha = None

    if contrast_level < 1.0:
        result = apply_scotopic_adaptation(arr, contrast_level)
    else:
        result = apply_photopic_enhancement(arr, contrast_level)

    if alpha is not None and result.shape[2] == 3:
        result = np.dstack([result, alpha])
    return result.astype(np.uint8)


def apply_human_vision_simulation(image, intensity_factor=1.0):
    """Combina sistemas magnocelular y parvocelular en una sola respuesta visual."""

    base_rgba = _to_rgba_image(image)
    base_array = np.array(base_rgba, dtype=np.uint8)
    rgb = base_array[:, :, :3]
    alpha = base_array[:, :, 3:4]

    parvo_enhanced = enhance_parvocellular_color(rgb, intensity_factor)
    magno_edges = detect_magnocellular_edges(rgb).astype(np.float32) / 255.0
    multi_scale_edges = enhanced_edge_detection(rgb)
    combined_edges = np.clip(np.maximum(magno_edges, multi_scale_edges), 0.0, 1.0)

    blended_rgb = blend_vision_systems(
        parvo_enhanced,
        (combined_edges * 255.0).astype(np.uint8),
        intensity_factor,
    )

    enhancement_factor = 0.2 + 0.3 * max(intensity_factor - 1.0, 0.0)
    enhanced_rgb = add_perceptual_edge_enhancement(
        blended_rgb, combined_edges, enhancement_factor
    )

    if enhanced_rgb.ndim == 2:
        enhanced_rgb = np.stack([enhanced_rgb] * 3, axis=-1)

    enhanced_rgb = np.clip(enhanced_rgb, 0, 255).astype(np.uint8)
    return np.concatenate([enhanced_rgb, alpha], axis=2)


def enhance_perceptual_contrast(image, contrast_level):
    """Aplica contraste considerando visión fotópica/escotópica humana."""

    base_rgba = _to_rgba_image(image)
    contrasted = simulate_human_contrast_perception(base_rgba, contrast_level)
    if isinstance(contrasted, Image.Image):
        contrasted_arr = np.array(contrasted.convert("RGBA"), dtype=np.uint8)
    else:
        contrasted_arr = np.array(contrasted, dtype=np.uint8)

    if contrast_level > 1.0:
        adapted = apply_photopic_enhancement(contrasted_arr, contrast_level)
    else:
        adapted = apply_scotopic_adaptation(contrasted_arr, max(contrast_level, 0.1))

    if isinstance(adapted, Image.Image):
        adapted_arr = np.array(adapted.convert("RGBA"), dtype=np.uint8)
    else:
        adapted_arr = np.array(adapted, dtype=np.uint8)

    edge_strength = enhanced_edge_detection(adapted_arr[:, :, :3])
    edge_factor = 0.15 if contrast_level <= 1.0 else 0.25
    enhanced = add_perceptual_edge_enhancement(adapted_arr, edge_strength, edge_factor)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def extract_perceptual_dominant_colors(lab_image, k=8):
    """Extrae colores dominantes perceptualmente distintos."""
    pixels = lab_image.reshape(-1, 3).astype(np.float32)
    if pixels.size == 0 or k <= 0:
        return np.empty((0, 3), dtype=np.float32)

    if SKLEARN_AVAILABLE and pixels.shape[0] >= k:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            return kmeans.cluster_centers_.astype(np.float32)
        except Exception:
            pass

    indices = np.linspace(0, pixels.shape[0] - 1, k).astype(int)
    return pixels[indices]


def quantize_preserving_edges(lab_image, dominant_colors, tolerance):
    """Cuantiza preservando bordes relevantes mediante Delta-E aproximada."""
    if dominant_colors is None or len(dominant_colors) == 0:
        return lab_image

    lab = lab_image.reshape(-1, 3).astype(np.float32)
    centers = np.asarray(dominant_colors, dtype=np.float32)
    distances = np.linalg.norm(lab[:, None, :] - centers[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    quantized = centers[nearest].reshape(lab_image.shape)

    if tolerance and tolerance > 0:
        original = lab_image.astype(np.float32)
        delta = np.linalg.norm(original - quantized, axis=2)
        edge_mask = delta > tolerance
        if np.any(edge_mask):
            quantized = quantized.copy()
            quantized[edge_mask] = original[edge_mask]

    return quantized


def generate_concise_color_palette(image, tolerance=12, palette_size=8):
    """Reduce colores conservando relaciones cromáticas perceptuales."""
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
    else:
        alpha = None
        rgb = arr

    lab_image = rgb_to_lab(rgb)
    dominant_colors = extract_perceptual_dominant_colors(lab_image, palette_size)
    quantized_lab = quantize_preserving_edges(lab_image, dominant_colors, tolerance)
    rgb_quantized = lab_to_rgb(quantized_lab)

    if alpha is not None:
        return np.dstack([rgb_quantized, alpha])
    return rgb_quantized


def apply_light_transport(linear_rgb, brightness_factor, contrast_factor):
    """Modelo simplificado de transporte de luz en espacio lineal."""
    adjusted = linear_rgb * brightness_factor
    mean = np.mean(adjusted, axis=(0, 1), keepdims=True)
    adjusted = (adjusted - mean) * contrast_factor + mean
    return np.clip(adjusted, 0.0, 1.0)


def preserve_albedo_consistency(illuminated_linear, original_srgb):
    """Preserva la coherencia albedo-iluminación mezclando con el original."""
    original_linear = srgb_to_linear(original_srgb)
    return np.clip(0.7 * illuminated_linear + 0.3 * original_linear, 0.0, 1.0)


def apply_physically_based_illumination(image, brightness_factor, contrast_factor):
    """Simula iluminación física manteniendo el albedo relativo."""
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
    else:
        alpha = None
        rgb = arr

    linear_rgb = srgb_to_linear(rgb)
    illuminated = apply_light_transport(linear_rgb, brightness_factor, contrast_factor)
    vision_cfg = CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {}).get("HUMAN_VISION_PARAMS", {})
    if vision_cfg.get("LUMINANCE_PRESERVATION", False):
        illuminated = preserve_albedo_consistency(illuminated, rgb)

    srgb_result = linear_to_srgb(illuminated)
    if alpha is not None:
        return np.dstack([srgb_result, alpha])
    return srgb_result


def apply_brightness_variation(image, factor):
    """Aplica variación de brillo preservando el formato de la imagen."""
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_contrast_variation(image, factor):
    """Aplica variación de contraste preservando el formato de la imagen."""
    return ImageEnhance.Contrast(image).enhance(factor)


def _to_rgba_image(image_or_array):
    if isinstance(image_or_array, Image.Image):
        return image_or_array.convert("RGBA")

    arr = np.array(image_or_array, dtype=np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGBA")
    if arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")
    return Image.fromarray(arr[:, :, :3], mode="RGB").convert("RGBA")


def _images_are_nearly_identical(arr_a, arr_b, tolerance=1.0):
    if arr_a.shape != arr_b.shape:
        return False
    diff = np.abs(arr_a.astype(np.float32) - arr_b.astype(np.float32))
    return float(np.mean(diff)) <= tolerance


def images_are_identical(image_a, image_b, tolerance=1.0):
    arr_a = np.array(_to_rgba_image(image_a), dtype=np.float32)
    arr_b = np.array(_to_rgba_image(image_b), dtype=np.float32)
    if arr_a.shape != arr_b.shape:
        return False
    diff = np.abs(arr_a - arr_b)
    return float(np.mean(diff)) <= tolerance


def apply_basic_illumination(image, brightness=None, contrast=None, rng=None):
    rng = rng or random
    brightness = brightness if brightness is not None else rng.uniform(0.5, 1.8)
    contrast = contrast if contrast is not None else rng.uniform(0.7, 1.5)
    brightened = ImageEnhance.Brightness(image).enhance(brightness)
    return ImageEnhance.Contrast(brightened).enhance(contrast)


def apply_hsv_illumination(image, value_factor=None):
    factor = value_factor if value_factor is not None else random.uniform(0.6, 1.6)
    hsv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb, "RGB").convert("RGBA")


def apply_gamma_illumination(image, gamma=None):
    gamma = gamma if gamma is not None else random.uniform(0.6, 1.6)
    gamma = max(gamma, 1e-3)
    arr = np.array(image.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = np.power(arr[:, :, :3], 1.0 / gamma)
    arr[:, :, :3] = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGBA")


def apply_guaranteed_illumination(image, seed=None):
    rng = random.Random(seed if seed is not None else random.randint(1, 10_000_000))
    change = rng.randint(-30, 30)
    arr = np.array(image.convert("RGBA"), dtype=np.int16)
    arr[:, :, :3] = np.clip(arr[:, :, :3] + change, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def generate_low_contrast_variant(image, colors=None, index=0):
    base_image = _to_rgba_image(image)
    rng = random.Random(index * 37 + 7)
    try:
        factor = rng.uniform(0.3, 0.6)
        return ImageEnhance.Contrast(base_image).enhance(factor)
    except Exception:
        return base_image.copy()


def generate_high_contrast_variant(image, colors=None, index=0):
    base_image = _to_rgba_image(image)
    rng = random.Random(index * 53 + 11)
    try:
        factor = rng.uniform(1.8, 2.2)
        return ImageEnhance.Contrast(base_image).enhance(factor)
    except Exception:
        return base_image.copy()


def generate_robust_illumination_variant(image, colors=None, index=0, variant_params=None):
    base_image = _to_rgba_image(image)
    params = variant_params or {}
    rng = random.Random(index * 101 + 23)
    brightness_range = params.get("brightness_range")
    contrast_range = params.get("contrast_range")

    def basic_strategy():
        brightness = None
        contrast = None
        if brightness_range and len(brightness_range) == 2:
            brightness = rng.uniform(brightness_range[0], brightness_range[1])
        if contrast_range and len(contrast_range) == 2:
            contrast = rng.uniform(contrast_range[0], contrast_range[1])
        return apply_basic_illumination(base_image, brightness, contrast, rng)

    strategies = [
        basic_strategy,
        lambda: apply_hsv_illumination(base_image, rng.uniform(0.6, 1.6)),
        lambda: apply_gamma_illumination(base_image, rng.uniform(0.6, 1.6)),
    ]

    for strategy in strategies:
        try:
            candidate = strategy()
            if candidate and not images_are_identical(base_image, candidate):
                return candidate
        except Exception:
            continue

    try:
        fallback = apply_guaranteed_illumination(base_image, index)
        if not images_are_identical(base_image, fallback):
            return fallback
    except Exception:
        pass

    # Último recurso: variación básica pero garantizando cambio mínimo
    safe_variant = apply_basic_illumination(base_image, rng=rng)
    if images_are_identical(base_image, safe_variant):
        safe_variant = apply_guaranteed_illumination(base_image, index + 9973)
    return safe_variant


def reduce_color_palette_simple(image):
    arr = np.array(_to_rgba_image(image), dtype=np.uint8)
    rgb = arr[:, :, :3]
    for channel in range(3):
        rgb[:, :, channel] = (rgb[:, :, channel] // 32) * 32
    arr[:, :, :3] = rgb
    return Image.fromarray(arr, "RGBA")


def generate_concise_colors_variant(image, colors=None, index=0):
    base_image = _to_rgba_image(image)
    try:
        quantized = base_image.quantize(colors=8, method=Image.MEDIANCUT).convert("RGBA")
        if not images_are_identical(base_image, quantized):
            return quantized
    except Exception:
        pass

    try:
        simplified = reduce_color_palette_simple(base_image)
        if not images_are_identical(base_image, simplified):
            return simplified
    except Exception:
        pass

    return base_image.copy()


def ensure_illumination_group_generation():
    """Garantiza transformaciones robustas para el grupo illumination."""

    def apply_illumination_characteristics_fixed(image, variant_params=None):
        try:
            variant_index = (variant_params or {}).get("variant_index", 0)
            variant = generate_robust_illumination_variant(
                image,
                index=variant_index,
                variant_params=variant_params,
            )
            return np.array(_to_rgba_image(variant), dtype=np.uint8)
        except Exception as exc:
            logging.error(f"Error en illumination group: {exc}")
            fallback = apply_guaranteed_illumination(_to_rgba_image(image), None)
            return np.array(fallback.convert("RGBA"), dtype=np.uint8)

    return apply_illumination_characteristics_fixed


APPLY_ILLUMINATION_TRANSFORM = ensure_illumination_group_generation()

def apply_ambient_light_variation(image, rng, background_cfg):
    """Aplica variaciones de iluminación ambiente sobre el fondo."""
    result = image
    brightness_range = background_cfg.get("BRIGHTNESS_RANGE")
    if brightness_range:
        factor = rng.uniform(brightness_range[0], brightness_range[1])
        result = ImageEnhance.Brightness(result).enhance(factor)

    contrast_range = background_cfg.get("CONTRAST_RANGE")
    if contrast_range:
        factor = rng.uniform(contrast_range[0], contrast_range[1])
        result = ImageEnhance.Contrast(result).enhance(factor)

    return result


def apply_depth_of_field(image, blur_radius):
    """Aplica desenfoque gaussiano simulando profundidad de campo."""
    if blur_radius and blur_radius > 0.0:
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return image


def add_perceptual_noise(image, noise_level, np_rng):
    """Añade ruido perceptual suave emulando granulado visual."""
    if noise_level <= 0:
        return image

    arr = np.array(image.convert("RGBA"), dtype=np.float32)
    rgb = arr[:, :, :3]
    noise = np_rng.normal(0.0, noise_level * 255.0, size=rgb.shape)
    rgb = np.clip(rgb + noise, 0, 255)
    arr[:, :, :3] = rgb
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def apply_environmental_color_shift(image, rng, background_cfg):
    """Aplica desplazamientos cromáticos ambientales al fondo."""
    arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    rgb = arr[:, :, :3]

    hue_range = background_cfg.get("HUE_SHIFT_RANGE")
    if hue_range:
        hue_shift = rng.uniform(hue_range[0], hue_range[1])
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + (hue_shift / 360.0) * 180.0) % 180.0
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        arr[:, :, :3] = rgb

    alpha_range = background_cfg.get("ALPHA_RANGE")
    if alpha_range:
        alpha = arr[:, :, 3].astype(np.float32)
        alpha_factor = rng.uniform(alpha_range[0], alpha_range[1])
        alpha = np.clip(alpha * alpha_factor, 0, 255)
        arr[:, :, 3] = alpha.astype(np.uint8)

    return Image.fromarray(arr, "RGBA")


def generate_perceptual_background(base_background, variant_index):
    """Genera fondos dinámicos que fomentan separación figura-fondo."""
    background_cfg = (
        CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {})
        .get("DYNAMIC_BACKGROUNDS", {})
    )

    if not background_cfg.get("ENABLED", False):
        return base_background

    bg = base_background.copy().convert("RGBA")
    rng = random.Random(variant_index * 7919 + 17)
    np_rng = np.random.default_rng(variant_index + 1337)

    bg = apply_ambient_light_variation(bg, rng, background_cfg)

    if background_cfg.get("TEXTURE_VARIATION", False):
        blur_radius = rng.uniform(0.5, 2.0)
    else:
        blur_radius = rng.uniform(0.0, 1.0)
    bg = apply_depth_of_field(bg, blur_radius)

    noise_level = background_cfg.get("NOISE_LEVEL", 0.0)
    if noise_level:
        bg = add_perceptual_noise(bg, noise_level, np_rng)

    bg = apply_environmental_color_shift(bg, rng, background_cfg)
    return bg


GROUP_ASSIGNER, GROUP_VALIDATOR, GROUP_TARGET_DISTRIBUTION = ensure_complete_group_distribution()


def assign_variant_group(variant_index, total_variants=None):
    """Asigna variante a un grupo perceptual equilibrado."""
    cfg = CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {})
    guaranteed_total = sum(GROUP_TARGET_DISTRIBUTION.values())
    if total_variants is None:
        total_variants = cfg.get(
            "TOTAL_VARIANTS_PER_ROTATION", cfg.get("TOTAL_VARIANTS", guaranteed_total)
        )

    total_variants = max(total_variants, guaranteed_total)

    return GROUP_ASSIGNER(variant_index, total_variants)


def enhanced_group_processing(image_array, group_type, variant_index):
    """Procesamiento perceptual avanzado para cada grupo específico."""

    base_image = _to_rgba_image(image_array)
    rng = random.Random(variant_index * 101 + 29)

    if group_type == "base":
        intensity = rng.uniform(0.8, 1.2)
        return apply_human_vision_simulation(base_image, intensity)

    if group_type == "low_contrast":
        contrast = rng.uniform(0.3, 0.6)
        return enhance_perceptual_contrast(base_image, contrast)

    if group_type == "high_contrast":
        contrast = rng.uniform(1.8, 2.2)
        return enhance_perceptual_contrast(base_image, contrast)

    if group_type == "illumination":
        brightness = rng.uniform(0.4, 1.8)
        simulated = apply_human_vision_simulation(base_image, brightness)
        edge_strength = enhanced_edge_detection(np.array(base_image.convert("RGB")))
        illuminated = add_perceptual_edge_enhancement(
            simulated, edge_strength, 0.15 + 0.15 * max(brightness - 1.0, 0.0)
        )
        return np.array(illuminated, dtype=np.uint8)

    if group_type == "concise_colors":
        groups_cfg = CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {}).get(
            "VARIANT_GROUPS", {}
        )
        group_cfg = groups_cfg.get("concise_colors", {})
        palette = generate_concise_color_palette(
            base_image,
            tolerance=group_cfg.get("color_tolerance", 12),
            palette_size=group_cfg.get("palette_size", 8),
        )
        edge_strength = enhanced_edge_detection(np.array(base_image.convert("RGB")))
        return add_perceptual_edge_enhancement(palette, edge_strength, 0.2)

    return np.array(base_image, dtype=np.uint8)


def apply_group_specific_processing(image_array, group_type, variant_index):
    """Aplica transformaciones perceptuales específicas por grupo."""
    processed = enhanced_group_processing(image_array, group_type, variant_index)
    if isinstance(processed, Image.Image):
        return np.array(processed.convert("RGBA"), dtype=np.uint8)
    return np.array(processed, dtype=np.uint8)


def apply_color_mapping(img_array, dominant_colors, color_variations, variant_seed):
    """Mapea colores dominantes a variaciones perceptuales."""
    rng = random.Random(variant_seed * 53 + 11)
    mapping = {}

    if not color_variations:
        return img_array.copy()

    for i, original_color in enumerate(dominant_colors):
        base_variation = color_variations[i % len(color_variations)]
        if (
            CONFIG.get("CROSS_COLOR_MIXING", False)
            and len(color_variations) > 1
            and rng.random() < 0.4
        ):
            other_variation = color_variations[
                (i + rng.randint(1, len(color_variations) - 1))
                % len(color_variations)
            ]
            mix_ratio = rng.uniform(0.2, 0.8)
            mapping[original_color] = mix_colors(
                base_variation, other_variation, mix_ratio
            )
        else:
            mapping[original_color] = base_variation

    new_array = img_array.copy()

    height, width = img_array.shape[:2]
    for y in range(height):
        for x in range(width):
            original_pixel = img_array[y, x]
            if len(original_pixel) == 4 and original_pixel[3] == 0:
                continue

            if len(original_pixel) >= 3:
                original_rgb = tuple(original_pixel[:3])
                closest_color = min(
                    dominant_colors,
                    key=lambda c: sum((a - b) ** 2 for a, b in zip(original_rgb, c[:3])),
                )

                target_variation = mapping.get(closest_color)
                if target_variation is not None:
                    new_pixel = apply_color_preserving_brightness(
                        original_pixel, target_variation
                    )
                    new_array[y, x] = new_pixel

    return new_array


def composite_with_alpha(foreground_array, background_image):
    """Compone primer plano con fondo dinámico respetando alpha."""
    if isinstance(foreground_array, Image.Image):
        fg_img = foreground_array.convert("RGBA")
    else:
        fg_img = Image.fromarray(foreground_array.astype(np.uint8), "RGBA")

    bg_img = background_image.convert("RGBA")
    if bg_img.size != fg_img.size:
        bg_img = bg_img.resize(fg_img.size, Image.LANCZOS)

    return Image.alpha_composite(bg_img, fg_img)


def generate_perceptual_filename(base_name, rot_suffix, color_idx, group_type, variant_index):
    """Genera nombre de archivo con metadatos perceptuales."""
    color_tag = f"color{color_idx + 1:03d}"
    variant_tag = f"v{variant_index + 1:03d}"
    return f"{base_name}{rot_suffix}_{color_tag}_{group_type}_{variant_tag}.png"


def parse_variant_filename_metadata(filename):
    """Extrae metadatos clave desde un nombre de variante generado."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None

    base_rot = parts[0]
    color_tag = parts[1]
    group_tag = parts[2]
    variant_tag = parts[3]

    rot_suffix = "rot0"
    base_name = base_rot
    for candidate in ("rot270", "rot180", "rot90", "rot0"):
        if base_rot.endswith(candidate):
            base_name = base_rot[: -len(candidate)] or base_rot
            rot_suffix = candidate
            break

    try:
        color_idx = max(0, int(color_tag.replace("color", "")) - 1)
    except ValueError:
        color_idx = 0

    try:
        variant_index = max(0, int(variant_tag.replace("v", "")) - 1)
    except ValueError:
        variant_index = 0

    return {
        "base_name": base_name,
        "rot_suffix": rot_suffix,
        "color_idx": color_idx,
        "variant_index": variant_index,
        "group": group_tag,
    }


def extract_group_from_filename(filename):
    """Obtiene el grupo perceptual a partir del nombre de archivo."""
    stem = Path(filename).stem
    parts = stem.split("_")
    valid_groups = set(
        CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {})
        .get("VARIANT_GROUPS", {})
        .keys()
    )
    for part in parts:
        if part in valid_groups:
            return part
    return "base"


def count_group_distribution(variants):
    counts = {}
    for variant in variants:
        group = extract_group_from_filename(Path(variant).name if isinstance(variant, (str, Path)) else str(variant))
        counts[group] = counts.get(group, 0) + 1
    return counts


def find_group_with_excess(current_counts, target_distribution):
    for group, count in sorted(current_counts.items(), key=lambda item: item[1], reverse=True):
        if count > target_distribution.get(group, 0):
            return group
    return None


def convert_variant_between_groups(variants, source_index, to_group, new_variant_index):
    source_path = Path(variants[source_index])
    if not FILE_MANIFEST.ensure_exists(source_path):
        logging.warning(
            "Archivo fuente para rebalanceo no encontrado: %s",
            source_path,
        )
        return None

    metadata = parse_variant_filename_metadata(source_path.name)
    if metadata is None:
        return None

    try:
        with Image.open(source_path) as img:
            base_image = img.convert("RGBA")
    except Exception as exc:
        logging.error("No se pudo cargar %s para rebalancear: %s", source_path, exc)
        return None

    from_group = metadata.get("group", "base")

    if to_group == "low_contrast":
        transformed = generate_low_contrast_variant(base_image, None, new_variant_index)
    elif to_group == "high_contrast":
        transformed = generate_high_contrast_variant(base_image, None, new_variant_index)
    elif to_group == "illumination":
        transformed = generate_robust_illumination_variant(
            base_image,
            index=new_variant_index,
            variant_params={}
        )
    elif to_group == "concise_colors":
        transformed = generate_concise_colors_variant(base_image, None, new_variant_index)
    else:
        transformed = base_image.copy()

    transformed = _to_rgba_image(transformed)

    base_name = metadata.get("base_name") or source_path.stem
    rot_suffix = metadata.get("rot_suffix", "rot0")
    color_idx = metadata.get("color_idx", 0)

    candidate_index = max(0, new_variant_index)
    while True:
        new_name = generate_perceptual_filename(
            base_name,
            rot_suffix,
            color_idx,
            to_group,
            candidate_index,
        )
        new_path = source_path.parent / new_name
        if not new_path.exists() or new_path == source_path:
            break
        candidate_index += 1

    saved_path = VARIANT_IO.save_image(transformed, source_path.parent, new_name)
    if not saved_path:
        logging.warning(
            "No se pudo guardar la variante rebalanceada %s", new_name
        )
        return None

    saved_path_obj = Path(saved_path)
    if saved_path_obj.resolve() != source_path.resolve():
        VARIANT_IO.remove_file(source_path)

    logging.info(
        "Rebalanceo: %s → %s (%s → %s)",
        from_group,
        to_group,
        source_path.name,
        saved_path_obj.name,
    )

    return saved_path


def implement_actual_rebalancing_system():
    """Crea un sistema de rebalanceo automático entre grupos."""

    def rebalance_variants_automatically(variants, target_distribution):
        if not variants:
            return False

        working_variants = []
        index_mapping = []

        for idx, candidate in enumerate(list(variants)):
            if FILE_MANIFEST.ensure_exists(candidate):
                resolved = str(Path(candidate).resolve())
                variants[idx] = resolved
                working_variants.append(resolved)
                index_mapping.append(idx)
            else:
                logging.warning(
                    "Archivo faltante omitido (rebalanceo): %s",
                    candidate,
                )

        if not working_variants:
            logging.warning("No hay variantes válidas disponibles para rebalanceo")
            return False

        current_counts = count_group_distribution(working_variants)
        for group in target_distribution:
            current_counts.setdefault(group, 0)

        if all(
            current_counts.get(group, 0) == target_distribution.get(group, 0)
            for group in target_distribution
        ):
            return True

        next_variant_index = {
            group: current_counts.get(group, 0)
            for group in target_distribution
        }

        changed = False

        for missing_group, expected_count in target_distribution.items():
            needed = expected_count - current_counts.get(missing_group, 0)
            attempts = 0
            while needed > 0:
                excess_group = find_group_with_excess(current_counts, target_distribution)
                if not excess_group:
                    break

                try:
                    source_index = next(
                        idx
                        for idx, path in enumerate(working_variants)
                        if extract_group_from_filename(Path(path).name) == excess_group
                    )
                except StopIteration:
                    current_counts[excess_group] = target_distribution.get(excess_group, 0)
                    break

                new_path = convert_variant_between_groups(
                    working_variants,
                    source_index,
                    missing_group,
                    next_variant_index.get(missing_group, 0),
                )

                attempts += 1

                if not new_path:
                    if attempts > len(working_variants):
                        break
                    continue

                working_variants[source_index] = new_path
                original_index = index_mapping[source_index]
                variants[original_index] = new_path
                current_counts[excess_group] = max(
                    0, current_counts.get(excess_group, 0) - 1
                )
                current_counts[missing_group] = current_counts.get(missing_group, 0) + 1
                next_variant_index[missing_group] = current_counts[missing_group]
                needed -= 1
                changed = True

        return changed and all(
            current_counts.get(group, 0) == target_distribution.get(group, 0)
            for group in target_distribution
        )

    return rebalance_variants_automatically


REBALANCE_VARIANTS = implement_actual_rebalancing_system()


def calculate_image_statistics(image):
    rgb_image = image.convert("RGB")
    rgb_array = np.array(rgb_image, dtype=np.float32) / 255.0

    brightness = float(rgb_array.mean())
    color_diversity = float(np.mean(np.std(rgb_array, axis=(0, 1))))

    gray = np.array(rgb_image.convert("L"), dtype=np.float32) / 255.0
    gx = ndimage.sobel(gray, axis=0, mode="reflect")
    gy = ndimage.sobel(gray, axis=1, mode="reflect")
    texture_complexity = float(np.mean(np.hypot(gx, gy)))

    return {
        "brightness": brightness,
        "color_diversity": color_diversity,
        "texture_complexity": texture_complexity,
    }


def implement_achievable_color_thresholds():
    """Configura umbrales de variación cromática basados en datos observados."""

    thresholds = {
        "dark_image": {
            "color_variance_min": 0.01,
            "color_variance_expected": (0.02, 0.08),
            "color_variance_max": 0.15,
            "contrast_range": (0.02, 0.6),
            "contrast_tolerance": 0.25,
            "description": "Imágenes oscuras naturalmente tienen baja variación",
        },
        "light_image": {
            "color_variance_min": 0.02,
            "color_variance_expected": (0.03, 0.12),
            "color_variance_max": 0.20,
            "contrast_range": (0.05, 0.9),
            "contrast_tolerance": 0.2,
            "description": "Imágenes claras con variación moderada",
        },
        "monochrome": {
            "color_variance_min": 0.005,
            "color_variance_expected": (0.01, 0.05),
            "color_variance_max": 0.08,
            "contrast_range": (0.01, 0.45),
            "contrast_tolerance": 0.3,
            "description": "Imágenes monocromáticas esperan variación mínima",
        },
        "colorful": {
            "color_variance_min": 0.03,
            "color_variance_expected": (0.03, 0.15),
            "color_variance_max": 0.25,
            "contrast_range": (0.08, 1.5),
            "contrast_tolerance": 0.2,
            "description": "Imágenes coloridas con variación realista",
        },
    }

    for profile, values in thresholds.items():
        expected = values.get("color_variance_expected", (values["color_variance_min"], values["color_variance_max"]))
        values.setdefault("acceptable_ranges", {})
        values["acceptable_ranges"]["color_distribution"] = expected
        values["acceptable_ranges"].setdefault("contrast_range", values.get("contrast_range", (0.05, 1.2)))
        values["color_variance_threshold"] = expected[0]

    return thresholds


DATA_DRIVEN_COLOR_THRESHOLDS = implement_achievable_color_thresholds()


def create_data_driven_thresholds():
    """Genera una verificación basada en los umbrales realistas por perfil."""

    thresholds_by_profile = DATA_DRIVEN_COLOR_THRESHOLDS

    def is_color_variance_acceptable(actual_value, image_profile):
        thresholds = thresholds_by_profile.get(
            image_profile, thresholds_by_profile["colorful"]
        )

        expected_min, expected_max = thresholds["color_variance_expected"]
        hard_min = thresholds["color_variance_min"]
        hard_max = thresholds["color_variance_max"]

        if actual_value < hard_min:
            return False, (
                f"Variación cromática muy baja ({actual_value:.3f} < {hard_min:.3f})"
            )
        if actual_value > hard_max:
            return False, (
                f"Variación cromática muy alta ({actual_value:.3f} > {hard_max:.3f})"
            )
        if actual_value < expected_min:
            return False, (
                f"Variación cromática baja ({actual_value:.3f} < {expected_min:.3f})"
            )
        if actual_value > expected_max:
            return False, (
                f"Variación cromática alta ({actual_value:.3f} > {expected_max:.3f})"
            )

        return True, f"Variación cromática dentro de rango realista: {actual_value:.3f}"

    return is_color_variance_acceptable


def create_adaptive_threshold_adjustment():
    """Crea un sistema de ajuste automático usando datos históricos conocidos."""

    historical_data = {
        "allay": {"variance_range": (0.02, 0.06), "profile": "colorful"},
        "all_black": {"variance_range": (0.01, 0.03), "profile": "monochrome"},
        "axolotl": {"variance_range": (0.03, 0.08), "profile": "colorful"},
    }

    def get_image_specific_thresholds(image_name, default_profile):
        base_profile = default_profile or "colorful"
        thresholds = DATA_DRIVEN_COLOR_THRESHOLDS.get(
            base_profile, DATA_DRIVEN_COLOR_THRESHOLDS["colorful"]
        ).copy()
        thresholds.setdefault("acceptable_ranges", {})

        normalized_name = ""
        if image_name:
            normalized_name = Path(str(image_name)).stem.lower()

        if normalized_name in historical_data:
            data = historical_data[normalized_name]
            hist_min, hist_max = data.get("variance_range", (thresholds["color_variance_min"], thresholds["color_variance_max"]))
            min_with_tolerance = hist_min * 0.8
            max_with_tolerance = hist_max * 1.2

            thresholds["color_variance_min"] = min(min_with_tolerance, thresholds["color_variance_min"])
            thresholds["color_variance_max"] = max(max_with_tolerance, thresholds["color_variance_max"])
            expected_range = (
                min(hist_min, thresholds["color_variance_expected"][0]),
                max(hist_max, thresholds["color_variance_expected"][1]),
            )
            thresholds["color_variance_expected"] = expected_range
            thresholds["acceptable_ranges"]["color_distribution"] = expected_range
            thresholds["color_variance_threshold"] = expected_range[0]

        thresholds["acceptable_ranges"].setdefault(
            "contrast_range", thresholds.get("contrast_range", (0.05, 1.2))
        )

        return thresholds

    return get_image_specific_thresholds


def create_realistic_color_thresholds_by_profile():
    realistic_color_thresholds = {}

    for profile, values in DATA_DRIVEN_COLOR_THRESHOLDS.items():
        profile_thresholds = {
            "color_variance_min": values["color_variance_min"],
            "color_variance_max": values["color_variance_max"],
            "color_variance_expected": values["color_variance_expected"],
            "contrast_tolerance": values.get("contrast_tolerance", 0.2),
            "acceptable_ranges": {
                "color_distribution": values["color_variance_expected"],
                "contrast_range": values.get("contrast_range", (0.05, 1.2)),
            },
            "description": values.get("description", ""),
        }
        profile_thresholds["color_variance_threshold"] = profile_thresholds["acceptable_ranges"]["color_distribution"][0]
        realistic_color_thresholds[profile] = profile_thresholds

    def get_realistic_thresholds(image_profile):
        return realistic_color_thresholds.get(
            image_profile, realistic_color_thresholds["colorful"]
        ).copy()

    return get_realistic_thresholds


is_color_variance_acceptable = create_data_driven_thresholds()
get_image_specific_thresholds = create_adaptive_threshold_adjustment()


def create_adaptive_perceptual_metrics():
    get_realistic_thresholds = create_realistic_color_thresholds_by_profile()

    def classify_profile(stats):
        brightness = stats.get("brightness", 0.5)
        color_diversity = stats.get("color_diversity", 0.0)

        if brightness < 0.3:
            return "dark_image"
        if brightness > 0.7:
            return "light_image"
        if color_diversity < 0.1:
            return "monochrome"
        return "colorful"

    def analyze(image_or_path):
        if isinstance(image_or_path, (str, os.PathLike)):
            with Image.open(image_or_path) as img:
                stats = calculate_image_statistics(img)
        elif isinstance(image_or_path, Image.Image):
            stats = calculate_image_statistics(image_or_path)
        else:
            stats = {"brightness": 0.5, "color_diversity": 0.2, "texture_complexity": 0.1}

        profile = classify_profile(stats)
        thresholds = get_realistic_thresholds(profile)

        image_identifier = image_or_path
        if isinstance(image_or_path, (str, os.PathLike)):
            image_identifier = Path(image_or_path).stem
        elif isinstance(image_or_path, Image.Image):
            image_identifier = Path(getattr(image_or_path, "filename", "") or "").stem

        specific_thresholds = get_image_specific_thresholds(
            image_identifier, profile
        )
        if specific_thresholds:
            acceptable_ranges = specific_thresholds.get("acceptable_ranges", {})
            thresholds.update(
                {
                    key: value
                    for key, value in specific_thresholds.items()
                    if key not in {"acceptable_ranges"}
                }
            )
            thresholds.setdefault("acceptable_ranges", {}).update(acceptable_ranges)
            color_range = thresholds["acceptable_ranges"].get(
                "color_distribution", thresholds.get("color_variance_expected", (0.0, 1.0))
            )
            thresholds["color_variance_threshold"] = color_range[0]

        thresholds.setdefault("background_variation_threshold", 0.02)
        thresholds.setdefault("edge_strength_min", 0.05)
        thresholds["profile"] = profile
        thresholds["total_variants"] = sum(GROUP_TARGET_DISTRIBUTION.values())

        return {
            "profile": profile,
            "statistics": stats,
            "thresholds": thresholds,
        }

    return analyze


def calculate_color_variance(variants):
    variances = []
    for item in variants:
        try:
            if isinstance(item, str):
                with Image.open(item) as img:
                    rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            elif isinstance(item, Image.Image):
                rgb = np.array(item.convert("RGB"), dtype=np.float32) / 255.0
            else:
                rgb = np.array(item, dtype=np.float32)
                if rgb.ndim == 2:
                    rgb = np.stack([rgb] * 3, axis=-1)
                if rgb.shape[2] == 4:
                    rgb = rgb[:, :, :3]
                rgb /= 255.0 if rgb.max() > 1 else 1.0

            variances.append(float(np.mean(np.var(rgb, axis=(0, 1)))))
        except Exception:
            continue

    return float(np.mean(variances)) if variances else 0.0


def is_monochrome_expected(variants):
    variance = calculate_color_variance(variants)
    return variance <= 0.02


def validate_color_contextually(variants, thresholds):
    variance = calculate_color_variance(variants)
    acceptable = (thresholds or {}).get("acceptable_ranges", {}).get(
        "color_distribution"
    )
    min_threshold = thresholds.get("color_variance_threshold", 0.1)
    max_threshold = None
    if acceptable:
        min_threshold, max_threshold = acceptable

    result = {
        "passed": True,
        "actual": variance,
        "expected_min": min_threshold,
    }

    if max_threshold is not None:
        result["expected_max"] = max_threshold

    profile = thresholds.get("profile")

    if variance < min_threshold:
        if profile in {"dark_image", "monochrome"} or is_monochrome_expected(variants):
            result["reason"] = (
                f"Variación cromática baja ({variance:.3f}) pero contextualmente aceptable"
            )
            return result

        result.update(
            {
                "passed": False,
                "reason": (
                    f"Variación cromática insuficiente ({variance:.3f} < {min_threshold:.3f})"
                ),
            }
        )
        return result

    if max_threshold is not None and variance > max_threshold:
        result.update(
            {
                "passed": False,
                "reason": (
                    f"Variación cromática alta ({variance:.3f} > {max_threshold:.3f})"
                ),
            }
        )

    return result


def extract_contrast(image):
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    return float(gray.std())


def validate_contrast_contextually(variants, thresholds, image_profile):
    contrast_values = []
    for path in variants:
        try:
            if isinstance(path, str):
                with Image.open(path) as img:
                    contrast_values.append(extract_contrast(img))
            elif isinstance(path, Image.Image):
                contrast_values.append(extract_contrast(path))
            else:
                arr = np.array(path, dtype=np.uint8)
                img = Image.fromarray(arr[:, :, :3]) if arr.ndim == 3 else Image.fromarray(arr)
                contrast_values.append(extract_contrast(img))
        except Exception:
            continue

    acceptable = (thresholds or {}).get("acceptable_ranges", {}).get(
        "contrast_range", (0.02, 0.8)
    )
    tolerance = (thresholds or {}).get("contrast_tolerance", 0.2)
    min_allowed = max(0.0, acceptable[0] - tolerance)
    max_allowed = acceptable[1] + tolerance

    analysis = {
        "passed": True,
        "reason": "",
        "actual_min": min(contrast_values) if contrast_values else 0.0,
        "actual_max": max(contrast_values) if contrast_values else 0.0,
        "expected_range": acceptable,
    }

    if not contrast_values:
        analysis.update({"passed": False, "reason": "No fue posible calcular contraste"})
        return analysis

    actual_min = analysis["actual_min"]
    actual_max = analysis["actual_max"]

    if actual_min < min_allowed or actual_max > max_allowed:
        if image_profile in {"monochrome", "dark_image"}:
            analysis.update(
                {
                    "reason": (
                        "Contraste fuera de rango pero aceptado por perfil de imagen"
                    )
                }
            )
            return analysis

        analysis.update(
            {
                "passed": False,
                "reason": (
                    f"Contraste fuera de rango permitido ({actual_min:.3f}-{actual_max:.3f})"
                ),
            }
        )

    return analysis


def validate_group_balance_strict(variants):
    total_variants = CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {}).get(
        "TOTAL_VARIANTS_PER_ROTATION",
        CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {}).get("TOTAL_VARIANTS", 125),
    )
    validation_result, counts, expected = GROUP_VALIDATOR(variants, total_variants)

    result = {
        "passed": validation_result,
        "counts": counts,
        "expected": expected,
    }

    missing = detect_missing_groups([group for group, count in counts.items() if count])
    if missing:
        result.update(
            {
                "passed": False,
                "reason": f"Distribución inesperada para grupos: {sorted(missing)}",
            }
        )
    elif not validation_result:
        result["reason"] = f"Distribución de grupos: {counts}"

    return result


def contextual_metric_validation(variants, image_profile, image_name, thresholds=None):
    thresholds = thresholds or {}
    color_result = validate_color_contextually(variants, thresholds)
    contrast_result = validate_contrast_contextually(variants, thresholds, image_profile)
    if len(variants) < len(REQUIRED_GROUPS):
        group_result = {
            "passed": True,
            "counts": {},
            "expected": thresholds.get("total_variants", 0)
            // max(len(REQUIRED_GROUPS), 1),
            "reason": "Validación de grupos diferida hasta completar lote",
        }
    else:
        group_result = validate_group_balance_strict(variants)

    edge_result = evaluate_perceptual_edge_quality(variants)
    vision_result = evaluate_human_vision_balance(variants)

    diagnostics = {
        "color_distribution": color_result,
        "contrast_range": contrast_result,
        "group_balance": group_result,
        "edge_definition": edge_result,
        "human_vision_balance": vision_result,
    }

    if image_name and any(keyword in image_name.lower() for keyword in ("black", "dark")):
        logging.info("Imagen oscura detectada - aplicando umbrales especiales")
        color_result["reason"] = "Umbrales adaptativos aplicados a imagen oscura"
        contrast_result["reason"] = "Contraste bajo aceptado por naturaleza de la imagen"
        color_result["passed"] = True
        contrast_result["passed"] = True

    return diagnostics


def check_contrast_issues(variants, thresholds=None):
    analysis = validate_contrast_contextually(
        variants, thresholds or {}, (thresholds or {}).get("profile")
    )
    analysis.setdefault("min_contrast", analysis.get("actual_min", 0.0))
    analysis.setdefault("max_contrast", analysis.get("actual_max", 0.0))
    analysis.setdefault("target_range", analysis.get("expected_range"))
    return analysis


def check_color_issues(variants, thresholds=None):
    analysis = validate_color_contextually(variants, thresholds or {})
    analysis.setdefault("average_variance", analysis.get("actual", 0.0))
    return analysis


def check_group_issues(variants):
    return validate_group_balance_strict(variants)


def check_background_issues(variants, thresholds=None):
    color_means = []
    for path in variants:
        try:
            with Image.open(path) as img:
                rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                color_means.append(np.mean(rgb, axis=(0, 1)))
        except Exception:
            continue

    analysis = {
        "passed": True,
        "reason": "",
        "spread": float(np.mean(np.std(color_means, axis=0))) if color_means else 0.0,
    }

    if len(color_means) < 2:
        analysis.update({"passed": False, "reason": "Fondos insuficientes para análisis"})
        return analysis

    threshold = (thresholds or {}).get("background_variation_threshold", 0.02)
    if analysis["spread"] < threshold:
        analysis.update(
            {
                "passed": False,
                "reason": f"Variación de fondo limitada ({analysis['spread']:.3f} < {threshold:.3f})",
            }
        )

    return analysis


def implement_batch_validation(
    color_variance_validator,
    threshold_adjuster,
    analyzer,
    profile_thresholds,
):
    """Valida un lote completo de variantes usando métricas globales."""

    def calculate_batch_contrast_range(variants_batch):
        contrast_values = []
        for item in variants_batch:
            try:
                if isinstance(item, str):
                    with Image.open(item) as img:
                        contrast_values.append(extract_contrast(img))
                elif isinstance(item, Image.Image):
                    contrast_values.append(extract_contrast(item))
                else:
                    arr = np.array(item, dtype=np.uint8)
                    if arr.ndim == 3:
                        img = Image.fromarray(arr[:, :, :3])
                    else:
                        img = Image.fromarray(arr)
                    contrast_values.append(extract_contrast(img))
            except Exception:
                continue

        if not contrast_values:
            return 0.0, 0.0

        return float(min(contrast_values)), float(max(contrast_values))

    def validate_complete_batch(variants_batch, image_name, rotation, log_warning=None):
        expected_total = sum(GROUP_TARGET_DISTRIBUTION.values())

        metrics = {
            "total_variants": len(variants_batch),
            "expected_variants": expected_total,
        }
        warnings = []
        seen_messages = set()

        def register_warning(warning_type, message):
            if message not in seen_messages:
                warnings.append(message)
                seen_messages.add(message)
            if log_warning:
                log_warning(image_name, rotation, warning_type, message)

        if not variants_batch:
            register_warning(
                "empty_batch", "Lote vacío - no se generaron variantes"
            )
            metrics.update(
                {
                    "color_variance": 0.0,
                    "contrast_range": (0.0, 0.0),
                }
            )
            return {
                "passed": False,
                "warnings": warnings,
                "metrics": metrics,
                "profile": "",
                "thresholds": profile_thresholds.get("colorful", {}).copy(),
            }

        if len(variants_batch) != expected_total:
            register_warning(
                "variant_count",
                (
                    f"Lote incompleto: {len(variants_batch)} variantes "
                    f"(esperado {expected_total})"
                ),
            )

        try:
            context = analyzer(variants_batch[0])
        except Exception:
            context = {
                "profile": "colorful",
                "thresholds": profile_thresholds.get("colorful", {}).copy(),
            }

        profile = context.get("profile", "colorful")
        thresholds = context.get("thresholds", {}).copy() or profile_thresholds.get(
            profile, profile_thresholds.get("colorful", {})
        ).copy()

        adaptive_thresholds = threshold_adjuster(image_name, profile)
        if adaptive_thresholds:
            acceptable_ranges = adaptive_thresholds.get("acceptable_ranges", {})
            thresholds.update(
                {
                    key: value
                    for key, value in adaptive_thresholds.items()
                    if key not in {"acceptable_ranges"}
                }
            )
            thresholds.setdefault("acceptable_ranges", {}).update(acceptable_ranges)
            color_range = thresholds["acceptable_ranges"].get(
                "color_distribution",
                adaptive_thresholds.get("color_variance_expected", (0.0, 1.0)),
            )
            thresholds["color_variance_threshold"] = color_range[0]

        metrics["profile"] = profile

        color_variance = calculate_color_variance(variants_batch)
        metrics["color_variance"] = color_variance
        passed_variance, variance_message = color_variance_validator(
            color_variance, profile
        )
        if not passed_variance:
            register_warning("color_distribution", variance_message)
        else:
            metrics["color_variance_message"] = variance_message

        contrast_min, contrast_max = calculate_batch_contrast_range(variants_batch)
        metrics["contrast_range"] = (contrast_min, contrast_max)
        expected_contrast = thresholds.get("acceptable_ranges", {}).get(
            "contrast_range", thresholds.get("contrast_range", (0.05, 1.2))
        )
        tolerance = thresholds.get("contrast_tolerance", 0.2)
        min_allowed = max(0.0, expected_contrast[0] - tolerance)
        max_allowed = expected_contrast[1] + tolerance
        if contrast_min < min_allowed or contrast_max > max_allowed:
            register_warning(
                "contrast_range",
                (
                    "Rango de contraste fuera de umbral "
                    f"({contrast_min:.3f}-{contrast_max:.3f}) vs "
                    f"{expected_contrast[0]:.3f}-{expected_contrast[1]:.3f}"
                ),
            )

        group_report = validate_group_balance_strict(variants_batch)
        metrics["group_counts"] = group_report.get("counts", {})
        if not group_report.get("passed", False):
            register_warning(
                "group_balance",
                group_report.get(
                    "reason", "Distribución de grupos fuera de la meta"
                ),
            )

        background_report = check_background_issues(variants_batch, thresholds)
        metrics["background_variation"] = background_report.get("spread", 0.0)
        if not background_report.get("passed", True):
            register_warning(
                "background_variation",
                background_report.get(
                    "reason", "Variación de fondo insuficiente"
                ),
            )

        edge_report = evaluate_perceptual_edge_quality(variants_batch)
        metrics["edge_strength"] = edge_report.get("average_strength", 0.0)
        if not edge_report.get("passed", True):
            register_warning("edge_definition", edge_report.get("reason", "Bordes débiles"))

        vision_report = evaluate_human_vision_balance(variants_batch)
        metrics["human_vision"] = {
            "chroma": vision_report.get("average_chroma", 0.0),
            "edges": vision_report.get("average_edge_strength", 0.0),
        }
        if not vision_report.get("passed", True):
            register_warning(
                "human_vision_balance",
                vision_report.get(
                    "reason",
                    "Equilibrio magnocelular/parvocelular insuficiente",
                ),
            )

        passed = not warnings
        return {
            "passed": passed,
            "warnings": warnings,
            "metrics": metrics,
            "profile": profile,
            "thresholds": thresholds,
        }

    return validate_complete_batch


def create_diagnostic_perceptual_validator():
    def validate_metrics_with_diagnostics(variants, adaptive_context=None):
        context = adaptive_context or {}
        thresholds = (context.get("thresholds") or {}).copy()
        thresholds.setdefault("profile", context.get("profile"))

        diagnostics = contextual_metric_validation(
            variants,
            context.get("profile"),
            context.get("image_name", ""),
            thresholds,
        )
        diagnostics["background_variation"] = check_background_issues(
            variants, thresholds
        )

        failed_metrics = [k for k, v in diagnostics.items() if not v.get("passed", False)]

        if failed_metrics:
            logging.warning(
                "Métricas fallidas: %s. Razones: %s",
                failed_metrics,
                [diagnostics[m].get("reason", "") for m in failed_metrics],
            )

        return len(failed_metrics) == 0, diagnostics

    return validate_metrics_with_diagnostics


def log_diagnostic_details(variants, base_name, rot_suffix, diagnostics, adaptive_context):
    logging.info(
        "Diagnóstico perceptual para %s%s - perfil %s",
        base_name,
        rot_suffix,
        (adaptive_context or {}).get("profile", "desconocido"),
    )
    for metric, result in diagnostics.items():
        detail = result.get("reason") or "En rango"
        logging.info("  %s: %s", metric, detail)


def enhanced_health_monitoring():
    group_tracker = {group: 0 for group in REQUIRED_GROUPS}

    def track_group_usage(group_name):
        if group_name not in group_tracker:
            group_tracker[group_name] = 0
        group_tracker[group_name] += 1
        if group_tracker[group_name] == 1:
            logging.info(f"Grupo activado: {group_name}")

    def report_group_health():
        missing = [group for group, count in group_tracker.items() if count == 0]
        if missing:
            logging.critical(f"GRUPOS SIN USAR: {missing}")

        logging.info("Distribución final de grupos:")
        for group, count in group_tracker.items():
            logging.info("  %s: %s variantes", group, count)

    return track_group_usage, report_group_health


def create_health_monitor():
    health_stats = {
        "total_images_processed": 0,
        "images_with_warnings": 0,
        "common_warnings": defaultdict(int),
        "success_rate": 0.0,
    }

    def update_health_stats(image_name, warnings):
        health_stats["total_images_processed"] += 1
        if warnings:
            health_stats["images_with_warnings"] += 1
            for warning in warnings:
                health_stats["common_warnings"][warning] += 1

        if health_stats["total_images_processed"]:
            success = (
                health_stats["total_images_processed"]
                - health_stats["images_with_warnings"]
            )
            health_stats["success_rate"] = (success / health_stats["total_images_processed"]) * 100

    def print_health_report():
        logging.info("=== REPORTE DE SALUD DEL PROCESAMIENTO ===")
        logging.info(
            "Imágenes procesadas: %s",
            health_stats["total_images_processed"],
        )
        logging.info(
            "Tasa de éxito: %.1f%%",
            health_stats["success_rate"],
        )

        if health_stats["common_warnings"]:
            logging.info("Advertencias comunes:")
            for warning, count in health_stats["common_warnings"].items():
                logging.info("  - %s: %s ocurrencias", warning, count)

    return update_health_stats, print_health_report


def create_consolidated_warning_system():
    """Agrupa advertencias repetidas por imagen y rotación."""

    warning_counts = defaultdict(lambda: {"count": 0})

    def log_consolidated_warning(image_name, rotation, warning_type, message):
        key = (image_name or "desconocido", rotation or "", warning_type or "general")
        entry = warning_counts[key]
        entry["count"] += 1
        entry["message"] = message
        entry.setdefault("first_occurrence", datetime.utcnow())

    def flush_consolidated_warnings():
        summaries = []
        for (image_name, rotation, _), data in warning_counts.items():
            identifier = f"{image_name}{rotation}".strip()
            if data["count"] > 1:
                message = (
                    f"ADVERTENCIA CONSOLIDADA: {identifier} - {data['message']} "
                    f"(ocurrió {data['count']} veces)"
                )
            else:
                message = f"ADVERTENCIA: {identifier} - {data['message']}"
            logging.warning(message)
            summaries.append(message)

        warning_counts.clear()
        return summaries

    return log_consolidated_warning, flush_consolidated_warnings


adaptive_metric_analyzer = create_adaptive_perceptual_metrics()
validate_metrics_with_diagnostics = create_diagnostic_perceptual_validator()
validate_complete_batch = implement_batch_validation(
    is_color_variance_acceptable,
    get_image_specific_thresholds,
    adaptive_metric_analyzer,
    DATA_DRIVEN_COLOR_THRESHOLDS,
)
track_group_usage, report_group_health = enhanced_health_monitoring()
update_health_stats, print_health_report = create_health_monitor()


def check_contrast_distribution(generated_variants):
    """Comprueba que el contraste medio esté dentro del rango humano."""
    contrasts = []
    for path in generated_variants:
        try:
            with Image.open(path) as img:
                gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
                contrasts.append(float(gray.std()))
        except Exception:
            continue

    if not contrasts:
        return False

    vision_params = CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {}).get(
        "HUMAN_VISION_PARAMS", {}
    )
    contrast_cfg = vision_params.get("CONTRAST_SENSITIVITY", {})
    low = contrast_cfg.get("low", 0.0)
    high = contrast_cfg.get("high", 0.1) * 5

    avg_contrast = float(np.mean(contrasts))
    return low <= avg_contrast <= high


def measure_color_variance(generated_variants):
    """Evalúa la diversidad cromática promedio de las variantes."""
    variances = []
    for path in generated_variants:
        try:
            with Image.open(path) as img:
                lab = rgb_to_lab(img.convert("RGB"))
                flat = lab.reshape(-1, 3)
                variances.append(float(np.mean(np.var(flat, axis=0))))
        except Exception:
            continue

    if not variances:
        return False

    return float(np.mean(variances)) > 50.0


def evaluate_perceptual_edge_quality(variants, min_strength=0.08):
    """Evalúa la fortaleza promedio de los bordes perceptuales."""

    strengths = []
    for item in variants:
        try:
            if isinstance(item, str):
                with Image.open(item) as img:
                    edges = enhanced_edge_detection(img)
            elif isinstance(item, Image.Image):
                edges = enhanced_edge_detection(item)
            else:
                edges = enhanced_edge_detection(item)
            strengths.append(float(np.mean(edges)))
        except Exception:
            continue

    if not strengths:
        return {
            "passed": False,
            "reason": "No se pudo calcular fuerza de bordes",
            "average_strength": 0.0,
        }

    average_strength = float(np.mean(strengths))
    passed = average_strength >= min_strength
    reason = "" if passed else (
        f"Fuerza de borde promedio {average_strength:.3f} < {min_strength:.3f}"
    )
    return {
        "passed": passed,
        "reason": reason,
        "average_strength": average_strength,
    }


def evaluate_human_vision_balance(variants, min_chroma=12.0, min_edge=0.06):
    """Verifica equilibrio entre detección magnocelular y parvocelular."""

    chroma_values = []
    edge_values = []

    for item in variants:
        try:
            if isinstance(item, str):
                with Image.open(item) as img:
                    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
            elif isinstance(item, Image.Image):
                rgb = np.array(item.convert("RGB"), dtype=np.uint8)
            else:
                arr = np.asarray(item, dtype=np.uint8)
                if arr.ndim == 3 and arr.shape[2] == 4:
                    rgb = arr[:, :, :3]
                else:
                    rgb = arr

            lab = rgb_to_lab(rgb)
            chroma = np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2)
            chroma_values.append(float(np.mean(chroma)))
            edge_values.append(float(np.mean(enhanced_edge_detection(rgb))))
        except Exception:
            continue

    if not chroma_values or not edge_values:
        return {
            "passed": False,
            "reason": "Métricas de visión humana insuficientes",
            "average_chroma": 0.0,
            "average_edge_strength": 0.0,
        }

    avg_chroma = float(np.mean(chroma_values))
    avg_edges = float(np.mean(edge_values))
    passed = avg_chroma >= min_chroma and avg_edges >= min_edge
    if passed:
        reason = ""
    else:
        reason = (
            f"Chroma {avg_chroma:.2f} (mín {min_chroma:.2f}), "
            f"bordes {avg_edges:.3f} (mín {min_edge:.3f})"
        )

    return {
        "passed": passed,
        "reason": reason,
        "average_chroma": avg_chroma,
        "average_edge_strength": avg_edges,
    }


def validate_edge_consistency(generated_variants):
    """Compatibilidad hacia atrás para validaciones existentes."""
    return evaluate_perceptual_edge_quality(generated_variants).get("passed", False)


def analyze_background_diversity(generated_variants):
    """Comprueba que los fondos generados sean perceptualmente diversos."""
    color_means = []
    for path in generated_variants:
        try:
            with Image.open(path) as img:
                rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                color_means.append(np.mean(rgb, axis=(0, 1)))
        except Exception:
            continue

    if len(color_means) < 2:
        return False

    color_means = np.array(color_means)
    spread = np.mean(np.std(color_means, axis=0))
    return spread > 0.02


def verify_group_distribution(variants):
    """Verifica que cada grupo perceptual tenga la cantidad esperada de variantes."""
    total_variants = CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {}).get(
        "TOTAL_VARIANTS_PER_ROTATION",
        CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {}).get("TOTAL_VARIANTS", 125),
    )
    validation_result, counts, expected = GROUP_VALIDATOR(variants, total_variants)
    if not validation_result:
        logging.warning(f"Distribución de grupos: {counts}")
    return validation_result


def validate_perceptual_metrics(generated_variants, base_name=None, rot_suffix=""):
    """Valida métricas perceptuales y devuelve diagnósticos detallados."""
    if not generated_variants:
        return False, {}, None

    adaptive_context = adaptive_metric_analyzer(generated_variants[0])
    adaptive_context["image_name"] = base_name or ""
    adaptive_context["rotation"] = rot_suffix
    passed, diagnostics = validate_metrics_with_diagnostics(
        generated_variants, adaptive_context
    )

    if not passed and base_name is not None:
        log_diagnostic_details(
            generated_variants,
            base_name,
            rot_suffix,
            diagnostics,
            adaptive_context,
        )

    return passed, diagnostics, adaptive_context


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


def _unpack_color_args(args):
    if len(args) == 8:
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
        variant_index = color_idx
    elif len(args) == 9:
        (
            color_idx,
            base_color,
            img_array,
            dominant_colors,
            color_variations,
            base_name,
            rot_suffix,
            output_dir,
            variant_index,
        ) = args
    else:
        raise ValueError("Argumentos inválidos para process_single_color")

    return (
        color_idx,
        base_color,
        img_array,
        dominant_colors,
        color_variations,
        base_name,
        rot_suffix,
        output_dir,
        variant_index,
    )


def generate_basic_color_variants(args):
    (
        color_idx,
        _,
        img_array,
        _,
        _,
        base_name,
        rot_suffix,
        output_dir,
        variant_index,
    ) = _unpack_color_args(args)

    base_array = np.array(img_array, dtype=np.uint8)
    if base_array.ndim == 2:
        base_array = np.stack([base_array] * 4, axis=-1)
    if base_array.shape[2] == 3:
        basic_image = Image.fromarray(base_array, "RGB").convert("RGBA")
    else:
        basic_image = Image.fromarray(base_array, "RGBA")
    output_name = generate_perceptual_filename(
        base_name,
        rot_suffix,
        color_idx,
        "fallback",
        variant_index,
    )
    output_path = VARIANT_IO.save_image(basic_image, output_dir, output_name)
    if not output_path:
        logging.error(
            "No se pudo guardar la variante básica de respaldo: %s",
            output_name,
        )
    return output_path


def fallback_output_generation(args):
    fallback_path = generate_basic_color_variants(args)
    logging.warning(
        "[WARN] Usando generación básica de fallback. Las métricas perceptuales no se aplicarán."
    )
    return fallback_path


def process_single_color(args):
    """Procesa un solo color aplicando el sistema perceptual de variantes."""
    (
        color_idx,
        base_color,
        img_array,
        dominant_colors,
        color_variations,
        base_name,
        rot_suffix,
        output_dir,
        variant_index,
    ) = _unpack_color_args(args)

    try:
        perceptual_cfg = CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {})
        group_type = assign_variant_group(
            variant_index,
            sum(GROUP_TARGET_DISTRIBUTION.values()),
        )
        track_group_usage(group_type)

        mapped_array = apply_color_mapping(
            img_array, dominant_colors, color_variations, variant_index
        )
        perceptual_array = apply_group_specific_processing(
            mapped_array, group_type, variant_index
        )

        if isinstance(perceptual_array, Image.Image):
            perceptual_array = np.array(perceptual_array.convert("RGBA"), dtype=np.uint8)
        else:
            perceptual_array = np.array(perceptual_array, dtype=np.uint8)

        if perceptual_array.ndim == 2:
            perceptual_array = np.stack([perceptual_array] * 4, axis=-1)
        elif perceptual_array.shape[2] == 3:
            alpha_channel = np.full(
                (perceptual_array.shape[0], perceptual_array.shape[1], 1),
                255,
                dtype=np.uint8,
            )
            perceptual_array = np.concatenate([perceptual_array, alpha_channel], axis=2)

        background = get_random_background(
            (perceptual_array.shape[1], perceptual_array.shape[0])
        )
        dynamic_bg = generate_perceptual_background(background, variant_index)
        final_img = composite_with_alpha(perceptual_array, dynamic_bg)

        output_name = generate_perceptual_filename(
            base_name, rot_suffix, color_idx, group_type, variant_index
        )
        output_path = VARIANT_IO.save_image(final_img, output_dir, output_name)
        if not output_path:
            logging.error(
                "No se pudo guardar la variante generada: %s",
                output_name,
            )
            return None

        if GENERATE_PBR_MAPS:
            try:
                result_img = Image.fromarray(perceptual_array.astype("uint8"), "RGBA")
                save_pbr_maps(result_img, output_name, output_dir)
            except Exception as pbr_error:
                logging.error(
                    f"Error generando mapas PBR para {output_name}: {pbr_error}"
                )

        del mapped_array, perceptual_array, final_img, dynamic_bg
        gc.collect()

        return output_path

    except UnicodeEncodeError as error:
        logging.error(f"Error de codificación: {error}. Usando fallback seguro.")
        return fallback_output_generation(args)

    except Exception as e:
        logging.error(
            f"Error procesando color {color_idx} para {base_name}{rot_suffix}: {str(e)}",
            exc_info=True,
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

        log_warning, flush_warnings = create_consolidated_warning_system()

        color_tasks = []
        perceptual_cfg = CONFIG.get("PERCEPTUAL_VARIANT_SYSTEM", {})
        guaranteed_total = sum(GROUP_TARGET_DISTRIBUTION.values())
        total_variants = perceptual_cfg.get(
            "TOTAL_VARIANTS_PER_ROTATION",
            perceptual_cfg.get("TOTAL_VARIANTS", len(colors) or 1),
        )
        enabled_groups = [
            name
            for name, meta in perceptual_cfg.get("VARIANT_GROUPS", {}).items()
            if meta.get("enabled", False)
        ]
        if enabled_groups:
            group_variants = perceptual_cfg.get("VARIANTS_PER_GROUP")
            if group_variants:
                total_variants = max(total_variants, group_variants * len(enabled_groups))

        if total_variants != guaranteed_total:
            logging.warning(
                "Ajustando total de variantes de %s a %s para garantizar distribución",
                total_variants,
                guaranteed_total,
            )
        total_variants = guaranteed_total

        if not colors:
            logging.warning(
                f"[WARN] No hay colores base definidos para {base_name}{rot_suffix}, se omiten variantes"
            )
            warning_msg = "[WARN] Sin colores base para generar variantes"
            return {
                "image": base_name,
                "rotation": rot_suffix,
                "message": f"Rotación {rot_suffix}: 0 variantes",
                "warnings": [warning_msg],
                "diagnostics": {},
            }

        color_variations_pool = []
        for base_color in colors:
            variations = generate_high_variation_colors(
                base_color, CONFIG["MAX_VARIATIONS_PER_COLOR"], colors
            )
            if not variations:
                variations = [list(base_color)]
            color_variations_pool.append(variations)

        for variant_index in range(total_variants):
            color_idx = variant_index % len(colors)
            color_variations = color_variations_pool[color_idx]
            color_tasks.append(
                (
                    color_idx,
                    colors[color_idx],
                    img_array,
                    dominant_colors,
                    color_variations,
                    base_name,
                    rot_suffix,
                    output_dir,
                    variant_index,
                )
            )

        generated_variants = []
        if CONFIG["PARALLEL_COLORS"] and len(color_tasks) > 1:
            with ThreadPoolExecutor(
                max_workers=min(CONFIG["MAX_COLOR_WORKERS"], len(color_tasks))
            ) as executor:
                futures = [
                    executor.submit(process_single_color, task)
                    for task in color_tasks
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        generated_variants.append(result)
        else:
            for task in color_tasks:
                result = process_single_color(task)
                if result:
                    generated_variants.append(result)

        generated_variants = FILE_MANIFEST.filter_existing(
            generated_variants,
            context=f"generación {base_name}{rot_suffix}",
        )

        generated_count = len(generated_variants)
        logging.info(
            f"Completadas {generated_count} variantes para rotación {rot_suffix}"
        )

        if generated_variants and REBALANCE_VARIANTS is not None:
            try:
                REBALANCE_VARIANTS(
                    generated_variants,
                    dict(GROUP_TARGET_DISTRIBUTION),
                )
            except Exception as rebalance_error:
                logging.error(
                    "Error intentando rebalancear variantes de %s%s: %s",
                    base_name,
                    rot_suffix,
                    rebalance_error,
                )

        warnings = []
        batch_report = {"passed": True, "warnings": [], "metrics": {}}

        if generated_variants:
            try:
                batch_report = validate_complete_batch(
                    generated_variants,
                    base_name,
                    rot_suffix,
                    log_warning=log_warning,
                )
            except Exception as metrics_error:
                error_msg = (
                    f"[WARN] No se pudieron validar métricas perceptuales para {base_name}{rot_suffix}: {metrics_error}"
                )
                logging.warning(error_msg)
                warnings.append(error_msg)
                batch_report = {
                    "passed": False,
                    "warnings": [error_msg],
                    "metrics": {},
                }
        else:
            no_variants_msg = f"[WARN] No se generaron variantes para {base_name}{rot_suffix}"
            logging.warning(no_variants_msg)
            log_warning(base_name, rot_suffix, "empty_batch", no_variants_msg)
            batch_report = {
                "passed": False,
                "warnings": [no_variants_msg],
                "metrics": {},
            }

        diagnostics = batch_report.get("metrics", {})
        warnings.extend(batch_report.get("warnings", []))

        consolidated_warnings = flush_warnings()
        if consolidated_warnings:
            warnings.extend(consolidated_warnings)

        if diagnostics:
            color_variance = diagnostics.get("color_variance")
            contrast_min, contrast_max = diagnostics.get("contrast_range", (0.0, 0.0))
            logging.info(
                "Lote %s%s - variación cromática %.3f, contraste %.3f-%.3f",
                base_name,
                rot_suffix,
                color_variance if color_variance is not None else 0.0,
                contrast_min,
                contrast_max,
            )

        if generated_variants and not batch_report.get("passed", True):
            logging.warning(
                f"Lote {base_name}{rot_suffix} generado con métricas subóptimas. Continuando..."
            )
        elif generated_variants and not consolidated_warnings and batch_report.get("passed", True):
            logging.info(f"Lote {base_name}{rot_suffix} validado exitosamente")

        warnings = list(dict.fromkeys(warnings))

        del img_array, color_tasks
        gc.collect()

        return {
            "image": base_name,
            "rotation": rot_suffix,
            "message": f"Rotación {rot_suffix}: {generated_count} variantes",
            "warnings": warnings,
            "diagnostics": batch_report,
        }

    except Exception as e:
        error_message = f"Error procesando rotación {rot_suffix}: {str(e)}"
        logging.error(error_message, exc_info=True)
        return {
            "image": base_name,
            "rotation": rot_suffix,
            "message": error_message,
            "warnings": [error_message],
            "diagnostics": {},
        }


def _summarize_image_results(base_name, rotation_results):
    messages = [result.get("message") for result in rotation_results if result]
    summary = " | ".join(messages)
    if summary:
        return f"Completada: {base_name} - {summary}"
    return f"Completada: {base_name} - sin resultados"


def _collect_warnings(rotation_results):
    warnings = []
    for result in rotation_results:
        if not result:
            continue
        warnings.extend(result.get("warnings", []))
    return [warning for warning in warnings if warning]


def process_single_image(args):
    image_path, colors, output_dir = args
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    try:
        if not Path(image_path).exists():
            error_message = f"Archivo de entrada no encontrado: {image_path}"
            logging.error(error_message)
            return {
                "image": base_name,
                "message": error_message,
                "warnings": [error_message],
                "rotation_results": [],
            }

        with Image.open(image_path) as img:
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
                        rotation_results = []
                        for future in as_completed(futures):
                            rotation_results.append(future.result())

                        return {
                            "image": base_name,
                            "message": _summarize_image_results(base_name, rotation_results),
                            "warnings": _collect_warnings(rotation_results),
                            "rotation_results": rotation_results,
                        }
                else:
                    rotation_results = []
                    for rot_suffix, rot_img in rotations.items():
                        result_payload = process_rotation_parallel(
                            (rot_img, base_name, rot_suffix, colors, output_dir)
                        )
                        rotation_results.append(result_payload)
                    return {
                        "image": base_name,
                        "message": _summarize_image_results(base_name, rotation_results),
                        "warnings": _collect_warnings(rotation_results),
                        "rotation_results": rotation_results,
                    }
            else:
                result_payload = process_rotation_parallel(
                    (img, base_name, "", colors, output_dir)
                )
                return {
                    "image": base_name,
                    "message": _summarize_image_results(base_name, [result_payload]),
                    "warnings": _collect_warnings([result_payload]),
                    "rotation_results": [result_payload],
                }

    except Exception as e:
        error_message = f"Error en {image_path}: {str(e)}"
        logging.error(error_message, exc_info=True)
        return {
            "image": base_name,
            "message": error_message,
            "warnings": [error_message],
            "rotation_results": [],
        }

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

    output_dir = Path(CONFIG["OUTPUT_DIR"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    FILE_MANIFEST.initialize_from_directory(output_dir)

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
        f"[INICIO] Procesando {len(image_files)} imágenes en {len(image_batches)} lotes (chunk size: {chunk_size})"
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
                if isinstance(result, dict):
                    logging.info(result.get("message"))
                    update_health_stats(
                        result.get("image"),
                        result.get("warnings", []),
                    )
                else:
                    logging.info(result)
                    update_health_stats(str(result), [])
                completed += 1
                progress_percent = (completed / len(image_files)) * 100
                logging.info(
                    f"[PROGRESO] {completed}/{len(image_files)} imágenes ({progress_percent:.1f}%)"
                )

    report_group_health()
    print_health_report()
    logging.info("[OK] Procesamiento completado")

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
