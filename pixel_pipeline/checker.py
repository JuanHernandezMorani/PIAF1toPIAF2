"""
Enhanced post-process validation tool for CodexPlus-generated variants.

Verifies:
1) Varianza cromática (RGB / LAB)
2) Diferencia perceptual ΔE2000  
3) Preservación estructural (bordes)
4) Consistencia entre rotaciones

Guarda JSON y CSV con resultados detallados.
"""

import os, glob, json, csv, re
import numpy as np
from PIL import Image, ImageFilter
from collections import defaultdict

# skimage es opcional; si no está, desactivamos ΔE2000
try:
    from skimage import color as skcolor, filters
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


CONFIG = {
    "base_dir": "variants/",
    "rotation_patterns": ["_r000_", "_r090_", "_r180_", "_r270_"],  # Todas las rotaciones
    "base_variant": "0000",           # Variante base para comparación
    "min_std_threshold": 75.0,        # ≈ 30% desviación cromática
    "min_deltaE": 20.0,               # diferencia perceptual LAB
    "min_edge_corr": 0.85,            # preservación estructural
    "min_rotation_consistency": 0.80, # consistencia entre rotaciones
    "json_out": "variation_validation_report.json",
    "csv_out": "variation_validation_report.csv",
}


# ----------------------------
# Utilidades numéricas seguras
# ----------------------------
def _pyfloat(x) -> float:
    """Convierte numpy scalars a float nativo para JSON/CSV."""
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return float(x)

def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Correlación con protección contra NaN/constantes."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    # Si ambos son constantes, corr es indefinida; devolvemos 0
    if np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if np.isnan(c):
        return 0.0
    return float(np.clip(c, -1.0, 1.0))


# ----------------------------
# Métricas mejoradas
# ----------------------------
def get_rgb_std(img: Image.Image) -> float:
    """Desviación estándar cromática mejorada."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    # Calcula std por canal y promedia
    std_per_channel = np.std(arr, axis=(0, 1))
    return _pyfloat(np.mean(std_per_channel))

def get_luminance_std(img: Image.Image) -> float:
    """Desviación estándar de luminancia."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return _pyfloat(np.std(gray))

def deltaE_lab(img1: Image.Image, img2: Image.Image) -> float:
    """CIEDE2000 perceptual difference (si hay skimage)."""
    if not _HAS_SKIMAGE:
        return _deltaE_fallback(img1, img2)

    try:
        rgb1 = np.asarray(img1.convert("RGB"), dtype=np.float32) / 255.0
        rgb2 = np.asarray(img2.convert("RGB"), dtype=np.float32) / 255.0
        
        lab1 = skcolor.rgb2lab(rgb1)
        lab2 = skcolor.rgb2lab(rgb2)
        
        delta = skcolor.deltaE_ciede2000(lab1, lab2)
        return _pyfloat(np.nanmean(delta))
    except Exception:
        return _deltaE_fallback(img1, img2)

def _deltaE_fallback(img1: Image.Image, img2: Image.Image) -> float:
    """Fallback para ΔE cuando no hay skimage."""
    arr1 = np.asarray(img1.convert("RGB"), dtype=np.float32) / 255.0
    arr2 = np.asarray(img2.convert("RGB"), dtype=np.float32) / 255.0
    
    # Conversión RGB a LAB simplificada
    def rgb_to_lab_simple(rgb):
        # RGB to XYZ
        mask = rgb <= 0.04045
        linear_rgb = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        
        mat = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750], 
            [0.0193339, 0.1191920, 0.9503041]
        ])
        xyz = linear_rgb @ mat.T
        
        # XYZ to Lab
        xyz_ref = np.array([0.95047, 1.0, 1.08883])
        xyz_normalized = xyz / xyz_ref
        
        epsilon = 0.008856
        kappa = 903.3
        mask = xyz_normalized > epsilon
        f_xyz = np.where(mask, xyz_normalized ** (1/3), (kappa * xyz_normalized + 16) / 116)
        
        L = np.where(xyz_normalized[..., 1] > epsilon, 116 * f_xyz[..., 1] - 16, kappa * xyz_normalized[..., 1])
        a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
        b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
        
        return np.stack([L, a, b], axis=-1)
    
    lab1 = rgb_to_lab_simple(arr1)
    lab2 = rgb_to_lab_simple(arr2)
    
    # ΔE simple (Euclidiana en Lab)
    delta = np.linalg.norm(lab1 - lab2, axis=2)
    return _pyfloat(np.nanmean(delta))

def edge_correlation(img1: Image.Image, img2: Image.Image) -> float:
    """Correlación de bordes mejorada con Sobel."""
    if _HAS_SKIMAGE:
        return _edge_correlation_sobel(img1, img2)
    else:
        return _edge_correlation_basic(img1, img2)

def _edge_correlation_sobel(img1: Image.Image, img2: Image.Image) -> float:
    """Correlación de bordes usando Sobel (más preciso)."""
    gray1 = np.asarray(img1.convert("L"), dtype=np.float32)
    gray2 = np.asarray(img2.convert("L"), dtype=np.float32)
    
    # Usar filtro Sobel de skimage si disponible
    edges1 = filters.sobel(gray1)
    edges2 = filters.sobel(gray2)
    
    return _safe_corrcoef(edges1, edges2)

def _edge_correlation_basic(img1: Image.Image, img2: Image.Image) -> float:
    """Correlación de bordes básica con FIND_EDGES de PIL."""
    gray1 = np.asarray(img1.convert("L"), dtype=np.float32)
    gray2 = np.asarray(img2.convert("L"), dtype=np.float32)
    
    def normalize_edges(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min > 1e-12:
            return (arr - arr_min) / (arr_max - arr_min)
        return arr
    
    e1 = Image.fromarray((normalize_edges(gray1) * 255).astype(np.uint8))
    e2 = Image.fromarray((normalize_edges(gray2) * 255).astype(np.uint8))
    
    edges1 = e1.filter(ImageFilter.FIND_EDGES)
    edges2 = e2.filter(ImageFilter.FIND_EDGES)
    
    arr1 = np.asarray(edges1, dtype=np.float32)
    arr2 = np.asarray(edges2, dtype=np.float32)
    
    return _safe_corrcoef(arr1, arr2)

def structural_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """SSIM simplificado para medida de similitud estructural."""
    gray1 = np.asarray(img1.convert("L"), dtype=np.float32) / 255.0
    gray2 = np.asarray(img2.convert("L"), dtype=np.float32) / 255.0
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = np.mean(gray1)
    mu2 = np.mean(gray2)
    sigma1 = np.std(gray1)
    sigma2 = np.std(gray2)
    sigma12 = np.cov(gray1.ravel(), gray2.ravel())[0, 1]
    
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2)
    
    if denominator == 0:
        return 0.0
    return _pyfloat(numerator / denominator)


# ----------------------------
# Agrupación y análisis de rotaciones
# ----------------------------
def group_images_by_base(base_dir: str):
    """Agrupa imágenes por nombre base y rotación."""
    pattern = os.path.join(base_dir, "*_r*_*.png")
    all_files = glob.glob(pattern)
    
    grouped = defaultdict(lambda: defaultdict(dict))
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        # Patrón corregido: nombre_rXXX_YYYY.png
        match = re.match(r'(.+)_r(\d{3})_(\d{4})\.png', filename)
        if not match:
            continue
            
        base_name, rotation, variant = match.groups()
        grouped[base_name][rotation][variant] = file_path
    
    return grouped

def analyze_rotation_consistency(base_group: dict, base_variant: str) -> float:
    """Analiza consistencia entre diferentes rotaciones."""
    rotations = list(base_group.keys())
    if len(rotations) < 2:
        return 1.0  # Solo una rotación, perfecta consistencia
    
    consistency_scores = []
    
    # Comparar cada par de rotaciones
    for i, rot1 in enumerate(rotations):
        for rot2 in rotations[i+1:]:
            if base_variant in base_group[rot1] and base_variant in base_group[rot2]:
                img1 = Image.open(base_group[rot1][base_variant])
                img2 = Image.open(base_group[rot2][base_variant])
                
                # Usar SSIM como medida de consistencia
                ssim_score = structural_similarity(img1, img2)
                consistency_scores.append(ssim_score)
    
    return _pyfloat(np.mean(consistency_scores)) if consistency_scores else 0.0


# ----------------------------
# Ejecución principal mejorada
# ----------------------------
def analyze_variants():
    base_dir = CONFIG["base_dir"]
    base_variant = CONFIG["base_variant"]
    
    # Agrupar imágenes
    grouped_images = group_images_by_base(base_dir)
    
    if not grouped_images:
        print("❌ No se encontraron imágenes que coincidan con el patrón nombre_rXXX_YYYY.png")
        return
    
    report = []
    
    for base_name, rotations in grouped_images.items():
        print(f"🔍 Analizando: {base_name}")
        
        # Calcular consistencia entre rotaciones
        rotation_consistency = analyze_rotation_consistency(rotations, base_variant)
        
        # Para cada rotación, comparar variantes con la base
        for rotation, variants in rotations.items():
            if base_variant not in variants:
                continue
                
            base_path = variants[base_variant]
            base_image = Image.open(base_path)
            
            # Analizar cada variante en esta rotación
            for variant, variant_path in variants.items():
                if variant == base_variant:
                    continue  # Saltar comparación consigo mismo
                
                try:
                    variant_image = Image.open(variant_path)
                    
                    # Calcular métricas
                    rgb_std = get_rgb_std(variant_image)
                    luminance_std = get_luminance_std(variant_image)
                    deltaE_val = deltaE_lab(base_image, variant_image)
                    edge_corr_val = edge_correlation(base_image, variant_image)
                    ssim_val = structural_similarity(base_image, variant_image)
                    
                    # Validaciones
                    valid_std = rgb_std >= CONFIG["min_std_threshold"]
                    valid_color = deltaE_val >= CONFIG["min_deltaE"]
                    valid_edges = edge_corr_val >= CONFIG["min_edge_corr"]
                    valid_consistency = rotation_consistency >= CONFIG["min_rotation_consistency"]
                    
                    entry = {
                        "base_name": base_name,
                        "rotation": f"r{rotation}",
                        "variant": variant,
                        "base_variant": base_variant,
                        "rgb_std": rgb_std,
                        "luminance_std": luminance_std,
                        "deltaE": deltaE_val,
                        "edge_corr": edge_corr_val,
                        "ssim": ssim_val,
                        "rotation_consistency": rotation_consistency,
                        "valid_std": valid_std,
                        "valid_color": valid_color,
                        "valid_edges": valid_edges,
                        "valid_consistency": valid_consistency,
                        "overall_valid": all([valid_std, valid_color, valid_edges, valid_consistency])
                    }
                    report.append(entry)
                    
                except Exception as e:
                    print(f"⚠️ Error procesando {variant_path}: {e}")
                    continue

    # Generar reporte consolidado
    if not report:
        print("❌ No se generaron reportes válidos")
        return
        
    total_variants = len(report)
    valid_variants = sum(1 for r in report if r["overall_valid"])
    validity_rate = (valid_variants / total_variants * 100.0) if total_variants else 0.0
    
    print(f"\n📊 RESUMEN FINAL:")
    print(f"✅ Variantes válidas: {valid_variants}/{total_variants} ({validity_rate:.1f}%)")
    print(f"📁 Reportes guardados: {CONFIG['json_out']} y {CONFIG['csv_out']}")

    # Guardar JSON
    with open(CONFIG["json_out"], "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_variants": total_variants,
                "valid_variants": valid_variants,
                "validity_rate": validity_rate
            },
            "config": CONFIG,
            "detailed_results": report
        }, f, indent=2, ensure_ascii=False)

    # Guardar CSV
    if report:
        fieldnames = list(report[0].keys())
        with open(CONFIG["csv_out"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in report:
                writer.writerow(row)


if __name__ == "__main__":
    analyze_variants()
