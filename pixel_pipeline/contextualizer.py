#!/usr/bin/env python3
"""
YOLOv11 Metadata Generator - Human Perception Optimized
Generates comprehensive metadata .txt files for all PNG images in variants/
"""

import os
import glob
import numpy as np
from PIL import Image, ImageFilter
from collections import defaultdict, Counter
import json
from pathlib import Path

# Configuración
VARIANTS_DIR = "variants"
PROCESS_BATCH_SIZE = 50  # Python puede manejar lotes más grandes

def main():
    print("🚀 Iniciando procesamiento de metadatos para YOLOv11...\n")
    
    # Verificar directorio
    if not os.path.exists(VARIANTS_DIR):
        print(f"❌ Error: No se encuentra la carpeta '{VARIANTS_DIR}/'")
        return
    
    # Encontrar todos los PNG recursivamente
    png_files = list(Path(VARIANTS_DIR).rglob("*.png"))
    print(f"📁 Encontrados {len(png_files)} archivos PNG\n")
    
    if not png_files:
        print("ℹ️  No se encontraron archivos PNG")
        return
    
    # Procesar archivos
    processed_count = 0
    error_count = 0
    
    for i, png_path in enumerate(png_files):
        try:
            result = process_png_file(str(png_path))
            processed_count += 1
            print(f"✅ {result}")
            
            # Mostrar progreso cada 50 archivos
            if (i + 1) % 50 == 0:
                print(f"📊 Procesados {i + 1}/{len(png_files)} archivos...")
                
        except Exception as e:
            error_count += 1
            print(f"❌ Error procesando {png_path}: {str(e)}")
    
    # Resumen final
    print('\n🎉 PROCESAMIENTO COMPLETADO')
    print('════════════════════════════')
    print(f'✅ Archivos procesados: {processed_count}')
    print(f'❌ Errores: {error_count}')
    print(f'📊 Total: {len(png_files)} archivos')

def process_png_file(file_path):
    """Procesa un archivo PNG y genera metadata para YOLOv11"""
    
    # Cargar imagen
    with Image.open(file_path) as img:
        name = os.path.basename(file_path)
        width, height = img.size
        has_alpha = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)
        
        # Convertir a RGB/RGBA para análisis consistente
        if img.mode != 'RGB' and img.mode != 'RGBA':
            img_rgb = img.convert('RGB')
        else:
            img_rgb = img
        
        # Convertir a array numpy
        img_array = np.array(img_rgb)
        
        # Análisis de colores
        unique_colors, color_histogram, dominant_colors = analyze_colors(img_array)
        
        # Análisis de luminancia
        luminance_data = analyze_luminance(img_array)
        
        # Análisis PBR
        pbr_analysis = analyze_pbr_map(name, img_array, luminance_data['average'])
        
        # Análisis de contraste y detalles
        contrast_info = analyze_contrast_details(img_array, luminance_data['average'])
        
        # Análisis de percepción humana
        perceptual_analysis = analyze_human_perception(img_array, pbr_analysis['map_type'])
        
        # Clasificación
        classification = classify_png_enhanced(
            name, width, height, has_alpha, unique_colors, 
            dominant_colors, pbr_analysis, luminance_data['average']
        )
        
        # Generar descripción YOLO
        yolo_description = generate_yolo_description(
            name, width, height, has_alpha, unique_colors,
            dominant_colors, luminance_data, pbr_analysis,
            contrast_info, perceptual_analysis, classification
        )
        
        # Guardar archivo .txt
        txt_path = file_path.replace('.png', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(yolo_description)
        
        return f"Generado: {os.path.basename(txt_path)}"

def analyze_colors(img_array):
    """Analiza colores únicos y dominantes"""
    # Redimensionar para análisis más rápido (opcional)
    if img_array.shape[0] * img_array.shape[1] > 1000000:  # Si más de 1M píxeles
        small_img = Image.fromarray(img_array).resize((500, 500), Image.Resampling.LANCZOS)
        small_array = np.array(small_img)
    else:
        small_array = img_array
    
    # Contar colores únicos
    if len(small_array.shape) == 3:
        # Formatear colores como tuplas RGB
        color_tuples = [tuple(pixel) for pixel in small_array.reshape(-1, small_array.shape[2])]
        unique_colors = len(set(color_tuples))
        
        # Encontrar colores dominantes
        color_counter = Counter(color_tuples)
        dominant_colors = color_counter.most_common(5)
        
        # Formatear para output
        dominant_formatted = [
            {'color': f"#{r:02x}{g:02x}{b:02x}", 'frequency': f"{(count/len(color_tuples)*100):.2f}%"}
            for (r, g, b), count in dominant_colors
        ]
        
        return unique_colors, dict(color_counter), dominant_formatted
    else:
        return 0, {}, []

def analyze_luminance(img_array):
    """Calcula métricas de luminancia"""
    if len(img_array.shape) == 3:
        # Convertir a luminancia (fórmula perceptualmente ponderada)
        luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    else:
        luminance = img_array
    
    avg_luminance = np.mean(luminance)
    std_luminance = np.std(luminance)
    min_luminance = np.min(luminance)
    max_luminance = np.max(luminance)
    
    return {
        'average': avg_luminance,
        'std': std_luminance,
        'min': min_luminance,
        'max': max_luminance,
        'dynamic_range': f"{min_luminance:.0f}-{max_luminance:.0f}"
    }

def analyze_pbr_map(filename, img_array, avg_luminance):
    """Analiza y clasifica mapas PBR"""
    name = filename.lower()
    map_type = 'generic'
    confidence = 0.0
    characteristics = []
    
    # Detección basada en nombre
    if any(keyword in name for keyword in ['albedo', 'diffuse', 'basecolor']):
        map_type = 'albedo'
        confidence = 0.9
        characteristics.extend(['Base color information', 'No lighting data', 'High color variation expected'])
    elif any(keyword in name for keyword in ['normal', 'normals']):
        map_type = 'normal'
        confidence = 0.85
        characteristics.extend(['Surface direction vectors', 'RGB encoded normals', 'Medium frequency details'])
    elif any(keyword in name for keyword in ['roughness', 'gloss']):
        map_type = 'roughness'
        confidence = 0.8
        characteristics.extend(['Surface micro-roughness', 'Grayscale data', 'Fine details important'])
    elif any(keyword in name for keyword in ['metallic', 'metalness']):
        map_type = 'metallic'
        confidence = 0.8
        characteristics.extend(['Metal/non-metal mapping', 'Binary-like data', 'High contrast expected'])
    elif any(keyword in name for keyword in ['ao', 'ambient', 'occlusion']):
        map_type = 'ambient_occlusion'
        confidence = 0.75
        characteristics.extend(['Ambient light accessibility', 'Soft shadow information', 'Low frequency data'])
    elif any(keyword in name for keyword in ['height', 'displacement', 'bump']):
        map_type = 'height'
        confidence = 0.7
        characteristics.extend(['Surface elevation data', 'Grayscale height information', 'Continuous gradients'])
    
    # Análisis estadístico adicional
    color_stats = analyze_color_distribution(img_array)
    
    if map_type == 'generic':
        # Inferir basado en características
        if color_stats['saturation'] < 0.1 and color_stats['contrast'] > 30:
            map_type = 'roughness_or_metallic'
            confidence = 0.6
            characteristics.extend(['Low saturation, high contrast', 'Potential roughness/metallic map'])
        elif color_stats['blue_dominance'] > 0.6:
            map_type = 'normal_map_candidate'
            confidence = 0.5
            characteristics.extend(['Blue channel dominance', 'Possible normal map encoding'])
    
    return {
        'map_type': map_type,
        'confidence': f"{confidence * 100:.1f}%",
        'characteristics': characteristics,
        'color_stats': color_stats
    }

def analyze_color_distribution(img_array):
    """Analiza distribución de color para clasificación PBR"""
    if len(img_array.shape) != 3:
        return {'saturation': 0, 'contrast': 0, 'blue_dominance': 0}
    
    # Muestrear píxeles (cada 10º para velocidad)
    sampled = img_array[::10, ::10]
    
    # Calcular saturación
    max_vals = np.max(sampled, axis=2)
    min_vals = np.min(sampled, axis=2)
    saturation = np.mean(np.where(max_vals > 0, (max_vals - min_vals) / max_vals, 0))
    
    # Calcular contraste
    luminance = 0.299 * sampled[:,:,0] + 0.587 * sampled[:,:,1] + 0.114 * sampled[:,:,2]
    contrast = np.std(luminance)
    
    # Dominancia azul
    total_intensity = np.sum(sampled)
    blue_dominance = np.sum(sampled[:,:,2]) / total_intensity if total_intensity > 0 else 0
    
    return {
        'saturation': float(saturation),
        'contrast': float(contrast),
        'blue_dominance': float(blue_dominance)
    }

def analyze_contrast_details(img_array, avg_luminance):
    """Analiza contraste y detalles estructurales"""
    if len(img_array.shape) == 3:
        # Convertir a escala de grises para análisis
        if img_array.shape[2] == 4:  # RGBA
            gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        else:  # RGB
            gray = np.dot(img_array, [0.299, 0.587, 0.114])
    else:
        gray = img_array
    
    # Métricas básicas
    min_val, max_val = np.min(gray), np.max(gray)
    contrast_ratio = max_val / min_val if min_val > 0 else 1
    
    # Detección de bordes simple (gradiente)
    grad_x = np.abs(np.gradient(gray, axis=1))
    grad_y = np.abs(np.gradient(gray, axis=0))
    edge_strength = np.mean(grad_x + grad_y)
    
    # Detalle de alta frecuencia
    high_freq_detail = np.sum(np.abs(gray - avg_luminance) > 50) / gray.size * 100
    
    return {
        'contrast_ratio': f"{contrast_ratio:.2f}",
        'dynamic_range': f"{min_val:.0f}-{max_val:.0f}",
        'edge_density': f"{edge_strength:.2f}",
        'detail_level': f"{high_freq_detail:.2f}%",
        'perceptual_complexity': calculate_perceptual_complexity(edge_strength, high_freq_detail)
    }

def analyze_human_perception(img_array, map_type):
    """Analiza cómo los humanos perciben la textura"""
    analysis = {
        'visual_saliency': 'medium',
        'attention_areas': [],
        'perceptual_importance': 'medium',
        'human_interpretation': ''
    }
    
    if map_type == 'albedo':
        analysis.update({
            'visual_saliency': 'high',
            'perceptual_importance': 'critical',
            'human_interpretation': 'Humans perceive this as the base color texture, directly affecting material recognition and color consistency under different lighting conditions.',
            'attention_areas': ['Color accuracy', 'Consistency across lighting', 'Material identification']
        })
    elif map_type == 'normal':
        analysis.update({
            'visual_saliency': 'medium', 
            'perceptual_importance': 'high',
            'human_interpretation': 'This map encodes surface details that create the illusion of geometry through lighting interaction. Human vision is sensitive to incorrect normal mapping as it breaks surface continuity.',
            'attention_areas': ['Surface continuity', 'Lighting response', 'Detail preservation']
        })
    elif map_type == 'roughness':
        analysis.update({
            'visual_saliency': 'low',
            'perceptual_importance': 'medium', 
            'human_interpretation': 'Roughness affects how humans perceive material quality and lighting response. Smooth surfaces (low roughness) appear reflective while rough surfaces (high roughness) scatter light diffusely.',
            'attention_areas': ['Micro-surface detail', 'Specular response', 'Material realism']
        })
    else:
        analysis.update({
            'visual_saliency': 'variable',
            'perceptual_importance': 'context_dependent',
            'human_interpretation': 'This texture contributes to overall material perception through indirect lighting and surface properties.',
            'attention_areas': ['Overall consistency', 'Lighting integration', 'Artifact detection']
        })
    
    return analysis

def calculate_perceptual_complexity(edge_density, detail_level):
    """Calcula complejidad perceptual"""
    try:
        detail_num = float(detail_level.strip('%'))
        complexity = edge_density * 0.6 + detail_num * 0.4
        if complexity > 70: return 'high'
        if complexity > 40: return 'medium'
        return 'low'
    except:
        return 'medium'

def classify_png_enhanced(name, width, height, has_alpha, unique_colors, dominant_colors, pbr_analysis, avg_luminance):
    """Clasificación mejorada de texturas"""
    base_classification = classify_png_basic(name, width, height, has_alpha, unique_colors)
    
    if pbr_analysis['map_type'] != 'generic':
        return f"{base_classification} | PBR_{pbr_analysis['map_type'].upper()}_MAP"
    
    if avg_luminance < 50: return f"{base_classification} | LOW_LUMINANCE"
    if avg_luminance > 200: return f"{base_classification} | HIGH_LUMINANCE" 
    if unique_colors < 10: return f"{base_classification} | LIMITED_PALETTE"
    if unique_colors > 1000: return f"{base_classification} | COMPLEX_PALETTE"
    
    return f"{base_classification} | STANDARD_TEXTURE"

def classify_png_basic(name, width, height, has_alpha, unique_colors):
    """Clasificación básica basada en nombre y características"""
    name_lower = name.lower()
    
    if any(keyword in name_lower for keyword in ['albedo', 'diffuse', 'basecolor']):
        return 'Albedo Map'
    elif 'normal' in name_lower:
        return 'Normal Map'
    elif 'roughness' in name_lower:
        return 'Roughness Map'
    elif 'metallic' in name_lower:
        return 'Metallic Map'
    elif any(keyword in name_lower for keyword in ['ao', 'ambientocclusion']):
        return 'Ambient Occlusion Map'
    elif any(keyword in name_lower for keyword in ['height', 'displacement']):
        return 'Height Map'
    elif has_alpha and unique_colors <= 2:
        return 'Binary Mask'
    elif unique_colors <= 16:
        return 'Indexed Texture'
    else:
        return 'Generic Texture'

def generate_yolo_description(name, width, height, has_alpha, unique_colors, dominant_colors, 
                             luminance_data, pbr_analysis, contrast_info, perceptual_analysis, classification):
    """Genera la descripción completa para YOLOv11"""
    
    dominant_colors_str = ", ".join([
        f"{color['color']} ({color['frequency']})" for color in dominant_colors
    ]) if dominant_colors else "N/A"
    
    return f"""# YOLOv11 TRAINING METADATA - HUMAN PERCEPTION OPTIMIZED
File: {name}
Dimensions: {width}x{height}
Alpha Channel: {has_alpha}

## COLOR ANALYSIS
Unique Colors: {unique_colors}
Average Luminance: {luminance_data['average']:.1f}
Luminance Range: {luminance_data['dynamic_range']}
Dominant Colors: {dominant_colors_str}

## TEXTURE CLASSIFICATION  
Primary: {classification}
PBR Map Type: {pbr_analysis['map_type']}
Confidence: {pbr_analysis['confidence']}

## CONTRAST & DETAIL ANALYSIS
Contrast Ratio: {contrast_info['contrast_ratio']}:1
Dynamic Range: {contrast_info['dynamic_range']}
Edge Density: {contrast_info['edge_density']}
Detail Level: {contrast_info['detail_level']}
Perceptual Complexity: {contrast_info['perceptual_complexity']}

## HUMAN PERCEPTION ANALYSIS
Visual Saliency: {perceptual_analysis['visual_saliency']}
Perceptual Importance: {perceptual_analysis['perceptual_importance']}
Attention Areas: {"; ".join(perceptual_analysis['attention_areas'])}

## HUMAN INTERPRETATION
{perceptual_analysis['human_interpretation']}

## PBR MAP CHARACTERISTICS
{chr(10).join(pbr_analysis['characteristics'])}

## TRAINING RECOMMENDATIONS
1. {get_training_recommendation('luminance', luminance_data['average'])}
2. {get_training_recommendation('contrast', float(contrast_info['contrast_ratio']))}
3. {get_training_recommendation('complexity', contrast_info['perceptual_complexity'])}
4. {get_training_recommendation('pbr_type', pbr_analysis['map_type'])}

## YOLOv11 CONFIGURATION NOTES
- Augmentation sensitivity: {get_augmentation_sensitivity(perceptual_analysis['visual_saliency'])}
- Learning rate consideration: {get_learning_rate_note(contrast_info['detail_level'])}
- Attention mechanism: {get_attention_mechanism_note(pbr_analysis['map_type'])}

// Metadata generated for YOLOv11 small multi-object detection
// Human perception optimized training dataset
"""

def get_training_recommendation(recommendation_type, value):
    """Genera recomendaciones de entrenamiento específicas"""
    recommendations = {
        'luminance': (
            'Consider brightness augmentation for low-light conditions' if value < 50 else
            'Apply tone mapping for high-luminance scenarios' if value > 200 else
            'Standard luminance processing appropriate'
        ),
        'contrast': (
            'High contrast - focus on edge preservation' if value > 5 else
            'Low contrast - enhance feature detection' if value < 2 else
            'Balanced contrast - standard processing'
        ),
        'complexity': (
            'Complex textures require careful augmentation to preserve details' if value == 'high' else
            'Simple textures can handle aggressive augmentation' if value == 'low' else
            'Moderate complexity - balanced augmentation'
        ),
        'pbr_type': (
            f'PBR map detected: prioritize {value} channel accuracy in loss function' if value != 'generic' else
            'Generic texture - standard multi-scale feature extraction'
        )
    }
    return recommendations.get(recommendation_type, 'Standard training approach recommended')

def get_augmentation_sensitivity(saliency):
    """Determina sensibilidad a aumentación de datos"""
    sensitivities = {
        'high': 'Low augmentation - preserve visual features',
        'medium': 'Moderate augmentation - balance diversity and preservation', 
        'low': 'High augmentation - can apply aggressive transforms'
    }
    return sensitivities.get(saliency, 'Moderate augmentation recommended')

def get_learning_rate_note(detail_level):
    """Recomendaciones de tasa de aprendizaje"""
    try:
        detail_num = float(detail_level.strip('%'))
        if detail_num > 5: return 'Consider lower learning rate for high-detail preservation'
        if detail_num < 1: return 'Standard learning rate appropriate'
        return 'Moderate learning rate with careful monitoring'
    except:
        return 'Standard learning rate recommended'

def get_attention_mechanism_note(map_type):
    """Recomendaciones de mecanismos de atención"""
    attention_map = {
        'albedo': 'Color-consistent attention crucial',
        'normal': 'Spatial attention for surface details', 
        'roughness': 'Local contrast attention important',
        'metallic': 'Binary pattern attention needed',
        'generic': 'Multi-scale attention recommended'
    }
    return attention_map.get(map_type, 'Adaptive attention mechanism')

if __name__ == "__main__":
    main()