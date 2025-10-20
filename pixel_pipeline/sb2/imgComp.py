import os
import re
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from scipy import ndimage
from scipy.stats import pearsonr

def load_pbr_maps(base_path, base_name, rotation, variant):
    """
    Carga todos los mapas PBR disponibles para una imagen base
    """
    maps = {}
    map_types = [
        'ao', 'curvature', 'emissive', 'fuzz', 'height', 'ior', 
        'material', 'metallic', 'normal', 'opacity', 'porosity', 
        'roughness', 'specular', 'structural', 'subsurface', 'transmission'
    ]
    
    for map_type in map_types:
        map_filename = f"{base_name}_r{rotation}_{variant}_{map_type}.png"
        map_path = os.path.join(base_path, map_filename)
        
        if os.path.exists(map_path):
            try:
                img = Image.open(map_path)
                # Preservar canal alpha si existe
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    img = img.convert('RGBA')
                else:
                    # Convertir a RGB o L según el tipo de mapa
                    if map_type in ['normal', 'emissive', 'material', 'specular', 'structural']:
                        img = img.convert('RGB')
                    else:
                        img = img.convert('L')
                print(f"  Cargado: {map_type} ({img.mode})")
                maps[map_type] = img
            except Exception as e:
                print(f"  Error cargando {map_type}: {e}")
    
    return maps


def extract_alpha_channel(image):
    """
    Extrae el canal alpha de una imagen, si existe
    """
    if image.mode in ('RGBA', 'LA'):
        return image.split()[-1]
    elif image.mode == 'P' and 'transparency' in image.info:
        return image.convert('RGBA').split()[-1]
    else:
        # Crear alpha completo si no existe
        return Image.new('L', image.size, 255)


def apply_alpha_compositing(base_img, alpha_map=None):
    """
    Composición alpha premultiplicada en sRGB/RGBA (sin linearizar).
    Mantiene contornos limpios en semitransparencias.
    """
    if base_img.mode != 'RGBA':
        base_img = base_img.convert('RGBA')

    r, g, b, a = base_img.split()

    if alpha_map is not None:
        if alpha_map.size != base_img.size:
            alpha_map = alpha_map.resize(base_img.size, Image.Resampling.LANCZOS)
        # Combinar alphas en 8-bit como premultiplicado: a' = a_fg * a_map / 255
        a = ImageChops.multiply(a, alpha_map)

    # Premultiplicar RGB por A para almacenado robusto
    # (evita halos cuando otros viewers re-componen)
    ar = ImageChops.multiply(r, a)
    ag = ImageChops.multiply(g, a)
    ab = ImageChops.multiply(b, a)

    # Volver a no-premultiplicado para seguir operando internamente
    # pero con alpha final correcto.
    def unpremul(chan, alpha):
        # evitemos división por cero
        arr = np.array(chan, np.float32)
        aa = np.array(alpha, np.float32)
        out = np.where(aa > 0, np.clip(arr * 255.0 / (aa + 1e-6), 0, 255), 0)
        return Image.fromarray(out.astype(np.uint8))
    r, g, b = map(lambda c: unpremul(c, a), (ar, ag, ab))

    return Image.merge('RGBA', (r, g, b, a))


def normalize_array(arr):
    """
    Normaliza un array numpy a rango [0, 1]
    """
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def biological_light_response(intensity, adaptation_level=0.5):
    """
    Simula la respuesta no lineal del ojo humano a la luz
    Basado en la ley de Stevens y respuesta logarítmica
    """
    # Aproximación de la respuesta visual humana (no lineal)
    return np.power(intensity, 0.45) * adaptation_level + np.power(intensity, 0.6) * (1 - adaptation_level)


def chromatic_adaptation(rgb_array, reference_white=(0.95, 1.0, 1.05)):
    """
    Adaptación cromática (equilibrio de blancos biológico)
    """
    result = rgb_array.copy()
    for i in range(3):
        result[..., i] = np.clip(result[..., i] * reference_white[i], 0, 1)
    return result


def extract_rgb_channels(image_array):
    """
    Extrae los canales RGB de un array de imagen, manejando tanto RGB como RGBA
    """
    if image_array.shape[2] == 4:  # RGBA
        return image_array[..., :3]
    elif image_array.shape[2] == 3:  # RGB
        return image_array
    elif len(image_array.shape) == 2:  # Escala de grises
        return np.stack([image_array] * 3, axis=2)
    else:
        raise ValueError(f"Formato de imagen no soportado: {image_array.shape}")


def microsurface_occlusion(height_map, normal_map, occlusion_strength=0.3):
    """
    Calcula oclusión a nivel micro-superficie basado en height y normal maps
    """
    if height_map is None or normal_map is None:
        return None
    
    height_array = np.array(height_map, dtype=np.float32)
    if len(height_array.shape) == 3:
        height_array = np.mean(height_array, axis=2)
    height_array = height_array / 255.0
    
    # Calcular gradientes para micro-oclusión
    grad_x = ndimage.sobel(height_array, axis=1)
    grad_y = ndimage.sobel(height_array, axis=0)
    
    # Magnitud del gradiente como indicador de micro-oclusión
    micro_occlusion = np.sqrt(grad_x**2 + grad_y**2)
    micro_occlusion = np.clip(micro_occlusion * occlusion_strength, 0, 1)
    
    return micro_occlusion


def subsurface_scattering_effect(base_color, thickness_map, scattering_strength=0.1):
    """
    Simula scattering sub-superficie (efecto biológico en materiales orgánicos)
    """
    if thickness_map is None:
        return base_color
    
    thickness_array = np.array(thickness_map, dtype=np.float32)
    if len(thickness_array.shape) == 3:
        thickness_array = np.mean(thickness_array, axis=2)
    thickness_array = thickness_array / 255.0
    
    # Scattering más pronunciado en canales rojos (efecto biológico)
    scattering_weights = np.array([0.8, 0.6, 0.4])  # R, G, B
    
    scattering_effect = np.zeros_like(base_color)
    for i in range(3):
        # Aplicar blur gaussiano proporcional al grosor
        sigma = thickness_array * scattering_strength * 5.0
        scattered_channel = np.zeros_like(base_color[..., i])
        
        # Aplicar diferentes niveles de blur según el grosor
        unique_sigmas = np.unique(sigma)
        for s in unique_sigmas:
            if s > 0.1:
                mask = (sigma == s)
                blurred = ndimage.gaussian_filter(base_color[..., i], sigma=s)
                scattered_channel[mask] = blurred[mask]
            else:
                mask = (sigma == s)
                scattered_channel[mask] = base_color[..., i][mask]
        
        scattering_effect[..., i] = scattered_channel * scattering_weights[i]
    
    # Mezclar con color original
    blend_factor = thickness_array[..., np.newaxis] * scattering_strength
    result = base_color * (1 - blend_factor) + scattering_effect * blend_factor
    
    return np.clip(result, 0, 1)


def fresnel_effect(normal_map, view_direction=(0, 0, 1), base_reflectivity=0.04):
    """
    Calcula efecto Fresnel para reflexiones más realistas
    """
    if normal_map is None:
        return None
    
    normal_array = np.array(normal_map, dtype=np.float32) / 255.0
    normal_array = normal_array * 2.0 - 1.0
    
    # Extraer solo canales RGB para cálculos de normales
    normal_array = extract_rgb_channels(normal_array)
    
    # Normalizar vectores
    norm = np.linalg.norm(normal_array, axis=2, keepdims=True)
    normal_array = normal_array / (norm + 1e-8)
    
    # Producto punto con dirección de vista
    view_dir = np.array(view_direction)
    view_dir = view_dir / np.linalg.norm(view_dir)
    
    cos_theta = np.abs(np.sum(normal_array * view_dir, axis=2))
    
    # Aproximación de Schlick's Fresnel
    fresnel = base_reflectivity + (1 - base_reflectivity) * np.power(1 - cos_theta, 5)
    
    return fresnel


def fake_shading_2_5d(base_img, height_map, normal_map, ao_map, roughness_map, metallic_map):
    """
    Fake shading 2.5D mejorado con efectos biológicos y físicos
    """
    print("  Aplicando fake shading 2.5D mejorado...")
    
    # Convertir a arrays y normalizar
    base_array = np.array(base_img, dtype=np.float32) / 255.0
    if base_array.shape[2] == 4:  # RGBA
        rgb_array = base_array[..., :3]
        alpha_array = base_array[..., 3]
    else:
        rgb_array = base_array
        alpha_array = np.ones(base_array.shape[:2])
    
    result = rgb_array.copy()
    
    # ===== CONFIGURACIÓN DE ILUMINACIÓN BIOLÓGICAMENTE REALISTA =====
    light_sources = [
        {
            'direction': np.array([0.5, 0.5, 1.0]),
            'color': np.array([1.0, 0.95, 0.9]),  # Luz solar cálida
            'intensity': 1.2,
            'biological_response': 0.6
        },
        {
            'direction': np.array([-0.3, -0.5, 0.7]),
            'color': np.array([0.8, 0.9, 1.0]),  # Luz ambiental fría
            'intensity': 0.4,
            'biological_response': 0.4
        },
        {
            'direction': np.array([0.7, -0.3, 0.5]),
            'color': np.array([1.0, 0.98, 0.95]),  # Luz de acento cálida
            'intensity': 0.3,
            'biological_response': 0.3
        }
    ]
    
    # ===== PROCESAR MAPAS DE GEOMETRÍA =====
    height_array = None
    if height_map is not None:
        height_array = np.array(height_map, dtype=np.float32)
        if len(height_array.shape) == 3:
            height_array = np.mean(height_array, axis=2)
        height_array = height_array / 255.0
    
    normal_array = None
    if normal_map is not None:
        normal_array = np.array(normal_map, dtype=np.float32) / 255.0
        normal_array = normal_array * 2.0 - 1.0
        
        # Extraer solo canales RGB para cálculos de normales
        normal_array = extract_rgb_channels(normal_array)
        
        norm = np.linalg.norm(normal_array, axis=2, keepdims=True)
        normal_array = normal_array / (norm + 1e-8)
    
    # Generar normales desde height map si no hay normal map
    if normal_array is None and height_array is not None:
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        grad_x = ndimage.convolve(height_array, kernel_x) * 3.0
        grad_y = ndimage.convolve(height_array, kernel_y) * 3.0
        
        normal_from_height = np.stack([-grad_x, -grad_y, np.ones_like(grad_x)], axis=2)
        norm = np.linalg.norm(normal_from_height, axis=2, keepdims=True)
        normal_array = normal_from_height / (norm + 1e-8)
    
    # ===== APLICAR ILUMINACIÓN POR CADA FUENTE =====
    for i, light in enumerate(light_sources):
        light_dir = light['direction'] / np.linalg.norm(light['direction'])
        
        if normal_array is not None:
            # Cálculo de intensidad de luz
            light_intensity = np.sum(normal_array * light_dir, axis=2)
            light_intensity = np.clip(light_intensity, 0, 1)
            
            # Aplicar respuesta biológica no lineal
            light_intensity = biological_light_response(light_intensity, light['biological_response'])
            
            # Modificar por roughness si está disponible
            if roughness_map is not None:
                roughness_array = np.array(roughness_map, dtype=np.float32)
                if len(roughness_array.shape) == 3:
                    roughness_array = np.mean(roughness_array, axis=2)
                roughness_array = roughness_array / 255.0
                # Roughness difumina y suaviza los reflejos
                light_intensity = light_intensity * (1.0 - roughness_array * 0.6)
            
            # Aplicar oclusión ambiental
            if ao_map is not None:
                ao_array = np.array(ao_map, dtype=np.float32)
                if len(ao_array.shape) == 3:
                    ao_array = np.mean(ao_array, axis=2)
                ao_array = ao_array / 255.0
                light_intensity = light_intensity * ao_array
            
            # Aplicar efecto Fresnel para metálicos
            if metallic_map is not None:
                metallic_array = np.array(metallic_map, dtype=np.float32)
                if len(metallic_array.shape) == 3:
                    metallic_array = np.mean(metallic_array, axis=2)
                metallic_array = metallic_array / 255.0
                
                fresnel = fresnel_effect(normal_map)
                if fresnel is not None:
                    metallic_fresnel = fresnel * metallic_array
                    light_intensity = light_intensity * (1.0 + metallic_fresnel * 2.0)
            
            # Aplicar micro-oclusión
            micro_occlusion = microsurface_occlusion(height_map, normal_map)
            if micro_occlusion is not None:
                light_intensity = light_intensity * (1.0 - micro_occlusion * 0.3)
            
            # Aplicar luz con color e intensidad
            light_effect = light_intensity[..., np.newaxis] * light['color'] * light['intensity']
            result = result + light_effect
    
    # ===== EFECTOS DE PROFUNDIDAD Y ATMÓSFERA =====
    if height_array is not None:
        # Desenfoque atmosférico (más desenfoque en áreas bajas)
        height_blur = (1.0 - height_array) * 0.4
        
        for channel in range(3):
            channel_data = result[..., channel]
            # Aplicar blur proporcional a la altura
            blurred_channel = ndimage.gaussian_filter(channel_data, sigma=2.0)
            result[..., channel] = channel_data * (1.0 - height_blur) + blurred_channel * height_blur
        
        # Acentuación de bordes en áreas de alta frecuencia
        edge_enhance = height_array * 0.2
        for channel in range(3):
            laplacian = ndimage.laplace(result[..., channel])
            result[..., channel] = np.clip(result[..., channel] + laplacian * edge_enhance, 0, 1)
    
    # ===== SCATTERING SUB-SUPERFICIE =====
    # Usar thickness map o invertir height map para scattering
    thickness_map = height_map
    if thickness_map is not None:
        result = subsurface_scattering_effect(result, thickness_map, scattering_strength=0.15)
    
    # ===== ADAPTACIÓN CROMÁTICA BIOLÓGICA =====
    result = chromatic_adaptation(result)
    
    # ===== AJUSTES FINALES =====
    result = np.clip(result, 0, 1)
    
    # Recombinar con alpha
    if base_array.shape[2] == 4:
        result = np.dstack([result, alpha_array])
    
    return Image.fromarray((result * 255).astype(np.uint8))


def biologically_aware_tone_mapping(image, exposure=1.0, contrast=1.1):
    """
    Mapeo de tono biológicamente consciente
    """
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    if img_array.shape[2] == 4:  # RGBA
        rgb = img_array[..., :3]
        alpha = img_array[..., 3]
    else:
        rgb = img_array
        alpha = None
    
    # Curva de respuesta visual humana (aproximación)
    # Combinación de logarítmico y sigmoide
    exposed = rgb * exposure
    
    # Compresión de rango dinámico
    tone_mapped = exposed / (exposed + 1.0)
    
    # Restaurar contraste
    tone_mapped = np.power(tone_mapped, 1.0/contrast)
    
    # Ajuste final de gamma para pantallas
    tone_mapped = np.power(tone_mapped, 1.0/2.2)
    
    tone_mapped = np.clip(tone_mapped, 0, 1)
    
    if alpha is not None:
        tone_mapped = np.dstack([tone_mapped, alpha])
    
    return Image.fromarray((tone_mapped * 255).astype(np.uint8))


def blend_normal_lighting(base_img, normal_map, height_map=None):
    """
    Aplica efecto de iluminación basado en mapa normal (versión mejorada)
    """
    if normal_map is None:
        return base_img
    
    # Preservar alpha
    alpha_channel = extract_alpha_channel(base_img)
    base_rgb = base_img.convert('RGB')
    
    # Asegurarnos de que el mapa normal tenga 3 canales
    if normal_map.mode != 'RGB':
        normal_map = normal_map.convert('RGB')
    
    base_array = np.array(base_rgb, dtype=np.float32) / 255.0
    normal_array = np.array(normal_map, dtype=np.float32) / 255.0
    
    # Extraer solo canales RGB para cálculos de normales
    normal_array = extract_rgb_channels(normal_array)
    
    # Redimensionar si es necesario
    if base_array.shape[:2] != normal_array.shape[:2]:
        normal_map = normal_map.resize(base_rgb.size, Image.Resampling.LANCZOS)
        normal_array = np.array(normal_map, dtype=np.float32) / 255.0
        normal_array = extract_rgb_channels(normal_array)
    
    # Configuración de luz biológica
    light_dir = np.array([0.3, 0.5, 0.8])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Convertir normal y normalizar
    normal = normal_array * 2.0 - 1.0
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / (norm + 1e-8)
    
    # Cálculo de intensidad con respuesta biológica
    intensity = np.sum(normal * light_dir, axis=2)
    intensity = biological_light_response(intensity, 0.5)
    intensity = np.clip(intensity, 0.3, 1.2)  # Rango biológico
    
    # Aplicar iluminación
    lit_base = base_array * intensity[..., np.newaxis]
    lit_base = np.clip(lit_base, 0, 1)
    
    result = Image.fromarray((lit_base * 255).astype(np.uint8), mode='RGB')
    
    # Restaurar alpha
    if alpha_channel is not None:
        result = result.convert('RGBA')
        result.putalpha(alpha_channel)
    
    return result


def apply_height_displacement(base_img, height_map, intensity=0.02):
    """
    Aplica efecto de desplazamiento basado en mapa de altura (mejorado)
    """
    if height_map is None:
        return base_img
    
    alpha_channel = extract_alpha_channel(base_img)
    base_rgb = base_img.convert('RGB')
    
    base_array = np.array(base_rgb)
    
    # Redimensionar si es necesario
    if height_map.size != base_rgb.size:
        height_map = height_map.resize(base_rgb.size, Image.Resampling.LANCZOS)
    
    height_array = np.array(height_map, dtype=np.float32)
    
    # Convertir a luminancia si es necesario
    if len(height_array.shape) == 3:
        height_array = np.mean(height_array, axis=2)
    
    height_array = height_array / 255.0
    
    # Efecto de relieve mejorado
    height_mask = height_array * intensity * 50
    
    result_array = np.zeros_like(base_array)
    for channel in range(3):
        channel_data = base_array[..., channel]
        
        # Aplicar desplazamiento con muestreo bicúbico para mejor calidad
        displaced = ndimage.map_coordinates(
            channel_data, 
            [
                np.clip(np.indices(base_array.shape[:2])[0] + height_mask, 0, base_array.shape[0]-1),
                np.clip(np.indices(base_array.shape[:2])[1] + height_mask, 0, base_array.shape[1]-1)
            ],
            order=3  # Bicúbico para mejor calidad
        )
        result_array[..., channel] = displaced
    
    result = Image.fromarray(result_array.astype(np.uint8), mode='RGB')
    
    # Restaurar alpha
    if alpha_channel is not None:
        result = result.convert('RGBA')
        result.putalpha(alpha_channel)
    
    return result


def blend_emissive(base_img, emissive_map):
    """
    Mezcla mapa emisivo con respuesta biológica
    """
    if emissive_map is None:
        return base_img
    
    alpha_channel = extract_alpha_channel(base_img)
    base_rgb = base_img.convert('RGB')
    
    base_array = np.array(base_rgb, dtype=np.float32)
    
    # Redimensionar si es necesario
    if emissive_map.size != base_rgb.size:
        emissive_map = emissive_map.resize(base_rgb.size, Image.Resampling.LANCZOS)
    
    emissive_array = np.array(emissive_map, dtype=np.float32)
    
    # Convertir a RGB si es necesario
    if len(emissive_array.shape) == 2:
        emissive_array = np.stack([emissive_array] * 3, axis=2)
    elif len(emissive_array.shape) == 3 and emissive_array.shape[2] == 4:
        # Extraer solo RGB de RGBA
        emissive_array = emissive_array[..., :3]
    
    # Mezcla adativa con respuesta no lineal
    emissive_strength = np.mean(emissive_array, axis=2) / 255.0
    adaptive_response = biological_light_response(emissive_strength, 0.7)
    
    blended = base_array + emissive_array * adaptive_response[..., np.newaxis] * 0.7
    blended = np.clip(blended, 0, 255)
    
    result = Image.fromarray(blended.astype(np.uint8), mode='RGB')
    
    # Restaurar alpha
    if alpha_channel is not None:
        result = result.convert('RGBA')
        result.putalpha(alpha_channel)
    
    return result


def apply_roughness_metallic(base_img, roughness_map, metallic_map):
    """
    Aplica efectos de roughness y metallic (versión mejorada)
    """
    alpha_channel = extract_alpha_channel(base_img)
    result = base_img.convert('RGB')
    base_array = np.array(result, dtype=np.float32)
    
    if roughness_map is not None:
        if roughness_map.size != result.size:
            roughness_map = roughness_map.resize(result.size, Image.Resampling.LANCZOS)
        
        roughness_array = np.array(roughness_map, dtype=np.float32)
        if len(roughness_array.shape) == 3:
            roughness_array = np.mean(roughness_array, axis=2)
        roughness_array = roughness_array / 255.0
        
        # Roughness afecta saturación y contraste de manera no lineal
        hsv = np.array(result.convert('HSV'), dtype=np.float32)
        saturation_reduction = np.power(roughness_array, 0.7) * 0.6
        value_reduction = np.power(roughness_array, 0.5) * 0.4
        
        hsv[..., 1] *= (1.0 - saturation_reduction)
        hsv[..., 2] *= (1.0 - value_reduction)
        result = Image.fromarray(np.clip(hsv, 0, 255).astype(np.uint8), mode='HSV').convert('RGB')
    
    if metallic_map is not None:
        if metallic_map.size != result.size:
            metallic_map = metallic_map.resize(result.size, Image.Resampling.LANCZOS)
            
        metallic_array = np.array(metallic_map, dtype=np.float32)
        if len(metallic_array.shape) == 3:
            metallic_array = np.mean(metallic_array, axis=2)
        metallic_array = metallic_array / 255.0
        
        result_array = np.array(result, dtype=np.float32)
        
        # Efecto metálico más realista
        metallic_boost = metallic_array[..., np.newaxis] * 0.4
        contrast_boost = 1.0 + metallic_array[..., np.newaxis] * 0.5
        
        # Aumentar contraste y saturación en áreas metálicas
        hsv = np.array(result.convert('HSV'), dtype=np.float32)
        hsv[..., 1] += metallic_array * 40  # Aumentar saturación
        hsv[..., 2] = np.clip(hsv[..., 2] * contrast_boost[..., 0], 0, 255)
        
        result = Image.fromarray(np.clip(hsv, 0, 255).astype(np.uint8), mode='HSV').convert('RGB')
        result_array = np.array(result, dtype=np.float32)
        
        # Añadir reflejos especulares
        specular = metallic_array[..., np.newaxis] * 0.3
        result_array = result_array * (1.0 + specular)
        
        result = Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))
    
    # Restaurar alpha
    if alpha_channel is not None:
        result = result.convert('RGBA')
        result.putalpha(alpha_channel)
    
    return result


def apply_ambient_occlusion(base_img, ao_map):
    """
    Aplica oclusión ambiental mejorada
    """
    if ao_map is None:
        return base_img
    
    alpha_channel = extract_alpha_channel(base_img)
    base_rgb = base_img.convert('RGB')
    
    base_array = np.array(base_rgb, dtype=np.float32)
    
    # Redimensionar si es necesario
    if ao_map.size != base_rgb.size:
        ao_map = ao_map.resize(base_rgb.size, Image.Resampling.LANCZOS)
    
    ao_array = np.array(ao_map, dtype=np.float32)
    
    # Convertir a luminancia si es necesario
    if len(ao_array.shape) == 3:
        ao_array = np.mean(ao_array, axis=2)
    
    ao_array = ao_array / 255.0
    
    # AO mejorado con respuesta no lineal
    ao_factor = 1.0 - (1.0 - np.power(ao_array, 0.7)) * 0.8
    
    occluded = base_array * ao_factor[..., np.newaxis]
    
    result = Image.fromarray(occluded.astype(np.uint8), mode='RGB')
    
    # Restaurar alpha
    if alpha_channel is not None:
        result = result.convert('RGBA')
        result.putalpha(alpha_channel)
    
    return result


def apply_specular_effects(base_img, specular_map):
    """
    Aplica efectos especulares mejorados
    """
    if specular_map is None:
        return base_img
    
    alpha_channel = extract_alpha_channel(base_img)
    base_rgb = base_img.convert('RGB')
    
    base_array = np.array(base_rgb, dtype=np.float32)
    
    # Redimensionar si es necesario
    if specular_map.size != base_rgb.size:
        specular_map = specular_map.resize(base_rgb.size, Image.Resampling.LANCZOS)
    
    specular_array = np.array(specular_map, dtype=np.float32)
    
    # Convertir a RGB si es necesario
    if len(specular_array.shape) == 2:
        specular_array = np.stack([specular_array] * 3, axis=2)
    elif len(specular_array.shape) == 3 and specular_array.shape[2] == 4:
        # Extraer solo RGB de RGBA
        specular_array = specular_array[..., :3]
    
    specular_array = specular_array / 255.0
    
    # Efecto especular con respuesta no lineal
    specular_strength = np.power(np.mean(specular_array, axis=2), 0.6) * 0.6
    brightened = base_array * (1.0 + specular_strength[..., np.newaxis] * 0.5)
    
    result = Image.fromarray(np.clip(brightened, 0, 255).astype(np.uint8), mode='RGB')
    
    # Restaurar alpha
    if alpha_channel is not None:
        result = result.convert('RGBA')
        result.putalpha(alpha_channel)
    
    return result


def calculate_pbr_coherence(pbr_maps, target_size):
    """
    Calcula mapa de calor de coherencia PBR mejorado
    CORREGIDO: Maneja arrays constantes sin NaN
    """
    print("  Calculando mapa de coherencia PBR mejorado...")
    
    coherence_maps = []
    map_keys = ['height', 'normal', 'ao', 'roughness', 'metallic', 'curvature', 'specular']
    
    for key in map_keys:
        if key in pbr_maps:
            img = pbr_maps[key]
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            arr = np.array(img, dtype=np.float32)
            
            # Convertir a escala de grises si es necesario
            if len(arr.shape) == 3:
                arr = np.mean(arr, axis=2)
            
            arr = normalize_array(arr)
            coherence_maps.append(arr)
    
    if len(coherence_maps) < 2:
        print("  No hay suficientes mapas para calcular coherencia")
        return None
    
    # Calcular coherencia multivariada
    stacked = np.stack(coherence_maps, axis=0)
    
    # Coherencia basada en correlación local - VERSIÓN CORREGIDA
    window_size = 5
    height, width = target_size[::-1]  # Dimensiones correctas
    coherence = np.zeros((height, width))
    
    # Función segura para calcular correlación que maneja arrays constantes
    def safe_correlation(arr1, arr2):
        # Verificar si alguno de los arrays es constante
        if np.std(arr1) < 1e-8 or np.std(arr2) < 1e-8:
            # Si ambos son constantes, correlación perfecta (1.0)
            if np.std(arr1) < 1e-8 and np.std(arr2) < 1e-8:
                if np.abs(np.mean(arr1) - np.mean(arr2)) < 1e-8:
                    return 1.0  # Mismo valor constante
                else:
                    return 0.0  # Diferentes valores constantes
            else:
                return 0.0  # Solo uno es constante
        else:
            # Ambos tienen variación, calcular correlación normal
            try:
                corr, _ = pearsonr(arr1, arr2)
                return abs(corr) if not np.isnan(corr) else 0.0
            except:
                return 0.0
    
    # Calcular coherencia solo para ventanas válidas
    valid_pixels = 0
    for i in range(window_size//2, height - window_size//2):
        for j in range(window_size//2, width - window_size//2):
            window_data = stacked[:, 
                                i-window_size//2:i+window_size//2+1, 
                                j-window_size//2:j+window_size//2+1]
            
            # Calcular correlación promedio entre mapas de manera segura
            correlations = []
            for k in range(len(coherence_maps)):
                for l in range(k+1, len(coherence_maps)):
                    window_k = window_data[k].flatten()
                    window_l = window_data[l].flatten()
                    
                    corr = safe_correlation(window_k, window_l)
                    correlations.append(corr)
            
            if correlations:
                coherence[i, j] = np.mean(correlations)
                valid_pixels += 1
    
    print(f"  Píxeles válidos calculados: {valid_pixels}/{height * width}")
    
    # Si no hay suficientes píxeles válidos, usar método alternativo
    if valid_pixels < (height * width) * 0.1:  # Menos del 10% de píxeles válidos
        print("  Usando método alternativo de coherencia...")
        return calculate_pbr_coherence_alternative(pbr_maps, target_size)
    
    # Suavizar el resultado
    coherence = ndimage.gaussian_filter(coherence, sigma=1.5)
    
    # Rellenar bordes con valores extrapolados
    coherence = fill_border_coherence(coherence, window_size//2)
    
    return np.clip(coherence, 0, 1)

def calculate_pbr_coherence_alternative(pbr_maps, target_size):
    """
    Método alternativo para calcular coherencia cuando hay muchos arrays constantes
    """
    print("  Calculando coherencia con método alternativo...")
    
    coherence_maps = []
    map_keys = ['height', 'normal', 'ao', 'roughness', 'metallic', 'curvature', 'specular']
    
    for key in map_keys:
        if key in pbr_maps:
            img = pbr_maps[key]
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            arr = np.array(img, dtype=np.float32)
            
            if len(arr.shape) == 3:
                arr = np.mean(arr, axis=2)
            
            arr = normalize_array(arr)
            coherence_maps.append(arr)
    
    if len(coherence_maps) < 2:
        return None
    
    # Método basado en diferencia normalizada (más robusto para arrays constantes)
    stacked = np.stack(coherence_maps, axis=0)
    
    # Calcular la desviación estándar entre mapas
    std_between_maps = np.std(stacked, axis=0)
    
    # Convertir a coherencia (1 - desviación normalizada)
    coherence = 1.0 - std_between_maps / (std_between_maps.max() + 1e-8)
    
    # Suavizar
    coherence = ndimage.gaussian_filter(coherence, sigma=2.0)
    
    return np.clip(coherence, 0, 1)

def fill_border_coherence(coherence, border_size):
    """
    Rellena los bordes del mapa de coherencia mediante extrapolación
    """
    height, width = coherence.shape
    
    # Rellenar bordes superiores e inferiores
    for i in range(border_size):
        # Borde superior
        coherence[i, :] = coherence[border_size, :]
        # Borde inferior  
        coherence[height-1-i, :] = coherence[height-1-border_size, :]
    
    # Rellenar bordes izquierdos y derechos
    for j in range(border_size):
        # Borde izquierdo
        coherence[:, j] = coherence[:, border_size]
        # Borde derecho
        coherence[:, width-1-j] = coherence[:, width-1-border_size]
    
    return coherence

def create_comparison_plot(base_img, fused_img, coherence_map, output_path):
    """
    Crea visualización mejorada con análisis biológico
    """
    print("  Creando visualización comparativa mejorada...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis PBR - Visualización Biológicamente Mejorada', fontsize=16, fontweight='bold')
    
    # Imagen base
    axes[0, 0].imshow(np.array(base_img))
    axes[0, 0].set_title('Imagen Base Original', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Imagen fusionada
    axes[0, 1].imshow(np.array(fused_img))
    axes[0, 1].set_title('Imagen Fusionada 2.5D (Mejorada)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Mapa de calor de coherencia
    if coherence_map is not None:
        im = axes[1, 0].imshow(coherence_map, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 0].set_title('Mapa de Coherencia PBR Mejorado', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Análisis estadístico avanzado
        mean_coherence = np.mean(coherence_map)
        std_coherence = np.std(coherence_map)
        high_coherence_ratio = np.sum(coherence_map > 0.7) / coherence_map.size
        
        axes[1, 0].text(0.02, 0.98, 
                       f'Coherencia Media: {mean_coherence:.3f}\n'
                       f'Desviación: {std_coherence:.3f}\n'
                       f'Alta Coherencia: {high_coherence_ratio:.1%}',
                       transform=axes[1, 0].transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Leyenda de interpretación biológica
    interpretation_text = """
    🧬 ANÁLISIS BIOLÓGICO Y FÍSICO
    
    🔴 Rojo: Baja coherencia física
    - Mapas PBR inconsistentes
    - Efectos visualmente incorrectos
    
    🟡 Amarillo: Coherencia moderada  
    - Alineación parcial de mapas
    - Efectos aceptables
    
    🟢 Verde: Alta coherencia física
    - Mapas perfectamente alineados
    - Efectos biológicamente realistas
    
    Efectos Aplicados:
    • Respuesta visual no lineal
    • Scattering sub-superficie
    • Adaptación cromática
    • Micro-oclusión
    • Efecto Fresnel
    """
    
    axes[1, 1].text(0.05, 0.95, interpretation_text, transform=axes[1, 1].transAxes,
                   verticalalignment='top', fontsize=11, linespacing=1.4,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Guardar la visualización
    plot_filename = output_path.replace('_fusion.png', '_analysis.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Análisis guardado: {plot_filename}")
    
    # Mostrar en ventana
    plt.show()


def fuse_pbr_layers(base_img, pbr_maps):
    """
    Fusiona todas las capas PBR en una imagen final con fake shading 2.5D
    """
    print("  Iniciando fusión de capas PBR...")
    result = base_img
    
    # Aplicar fake shading 2.5D
    result = fake_shading_2_5d(
        result,
        pbr_maps.get('height'),
        pbr_maps.get('normal'),
        pbr_maps.get('ao'),
        pbr_maps.get('roughness'),
        pbr_maps.get('metallic')
    )
    
    # 1. Aplicar oclusión ambiental primero
    print("  Aplicando oclusión ambiental...")
    result = apply_ambient_occlusion(result, pbr_maps.get('ao'))
    
    # 2. Aplicar efectos de material
    print("  Aplicando propiedades de material...")
    result = apply_roughness_metallic(
        result, 
        pbr_maps.get('roughness'), 
        pbr_maps.get('metallic')
    )
    
    # 3. Aplicar efectos especulares
    print("  Aplicando efectos especulares...")
    result = apply_specular_effects(result, pbr_maps.get('specular'))
    
    # 4. Aplicar desplazamiento de altura
    print("  Aplicando mapa de altura...")
    result = apply_height_displacement(result, pbr_maps.get('height'))
    
    # 5. Aplicar iluminación desde mapa normal
    print("  Aplicando iluminación normal...")
    result = blend_normal_lighting(result, pbr_maps.get('normal'), pbr_maps.get('height'))
    
    # 6. Añadir efectos emisivos (último - aditivo)
    print("  Aplicando efectos emisivos...")
    result = blend_emissive(result, pbr_maps.get('emissive'))
    
    # 7. Aplicar opacidad si existe
    if 'opacity' in pbr_maps:
        print("  Aplicando opacidad...")
        opacity_map = pbr_maps['opacity']
        if opacity_map.size != result.size:
            opacity_map = opacity_map.resize(result.size, Image.Resampling.LANCZOS)
            
        opacity_array = np.array(opacity_map, dtype=np.float32)
        if len(opacity_array.shape) == 3:
            opacity_array = np.mean(opacity_array, axis=2)
        opacity_array = opacity_array / 255.0
        
        result_array = np.array(result, dtype=np.float32)
        
        # Combinar con fondo negro basado en opacidad
        background = np.zeros_like(result_array)
        final_array = result_array * opacity_array[..., np.newaxis] + background * (1 - opacity_array[..., np.newaxis])
        result = Image.fromarray(final_array.astype(np.uint8))
    
    print("  Fusión completada!")
    return result


def main():
    test_dir = Path("b")
    
    if not test_dir.exists():
        print(f"Error: Directorio {test_dir} no encontrado")
        return
    
    # Patrón para imágenes base
    base_pattern = re.compile(r'(.+)_r(\d{3})_(\d{4})\.png')
    
    base_images = []
    for file in test_dir.glob("*.png"):
        match = base_pattern.match(file.name)
        if match and not any(map_type in file.name for map_type in [
            'ao', 'curvature', 'emissive', 'fuzz', 'height', 'ior', 
            'material', 'metallic', 'normal', 'opacity', 'porosity', 
            'roughness', 'specular', 'structural', 'subsurface', 'transmission', 'fusion', 'analysis'
        ]):
            base_images.append(file)
    
    print(f"Encontradas {len(base_images)} imágenes base")
    
    for base_image_path in base_images:
        match = base_pattern.match(base_image_path.name)
        if not match:
            continue
            
        base_name = match.group(1)
        rotation = match.group(2)
        variant = match.group(3)
        
        print(f"\nProcesando: {base_image_path.name}")
        
        # Cargar imagen base preservando alpha
        try:
            base_img = Image.open(base_image_path)
            if base_img.mode != 'RGBA':
                base_img = base_img.convert('RGBA')
            print(f"  Tamaño base: {base_img.size} ({base_img.mode})")
        except Exception as e:
            print(f"  Error cargando imagen base: {e}")
            continue
        
        # Cargar mapas PBR
        pbr_maps = load_pbr_maps(test_dir, base_name, rotation, variant)
        print(f"  Mapas PBR encontrados: {list(pbr_maps.keys())}")
        
        if not pbr_maps:
            print("  No se encontraron mapas PBR para esta imagen base")
            continue
        
        # Fusionar capas con enfoque biológico
        try:
            fused_image = fuse_pbr_layers(base_img, pbr_maps)
            
            # Guardar resultado
            output_filename = f"{base_name}_r{rotation}_{variant}_fusion.png"
            output_path = test_dir / output_filename
            fused_image.save(output_path, optimize=True)
            print(f"  Imagen fusionada guardada: {output_path}")
            
            # Calcular mapa de coherencia mejorado
            coherence_map = calculate_pbr_coherence(pbr_maps, base_img.size)
            
            # Crear visualización comparativa
            create_comparison_plot(base_img, fused_image, coherence_map, str(output_path))
            
        except Exception as e:
            print(f"  Error durante la fusión biológica: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Configurar matplotlib para mejor visualización
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    main()
    