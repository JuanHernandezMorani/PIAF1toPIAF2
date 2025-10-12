import os
import json
import pandas as pd
from pathlib import Path

INPUT_JSONL = "data/trainDataMinecraft.jsonl"
VARIANTS_DIR = Path("variants")
OUT_DIR = Path("trainData")
OUT_DIR.mkdir(exist_ok=True)

CLASS_MAP = {
    "eyes": 0, "wings": 1, "body": 2, "aletas": 3, "extremities": 4,
    "fangs": 5, "claws": 6, "head": 7, "mouth": 8, "heart": 9,
    "cracks": 10, "cristal": 11, "flower": 12, "zombie_zone": 13,
    "armor": 14, "sky": 15, "stars": 16, "extra": 17
}

LAYER_MAP = {
    "base": 0, "emissive": 1, "normal": 2, "specular": 3
}


def detect_layer_from_filename(filename):
    """Detecta el layer asociado a un archivo de variante."""
    name = Path(filename).stem
    if "_normal" in name:
        return 2, "normal"
    if "_specular" in name:
        return 3, "specular"
    if "_emissive" in name:
        return 1, "emissive"
    return 0, "base"

def rotate_point_90(x, y):
    return y, 1 - x 

def rotate_point_180(x, y):
    return 1 - x, 1 - y

def rotate_point_270(x, y):
    return 1 - y, x

ROT_POINT_FUNCS = {
    "rot90": rotate_point_90,
    "rot180": rotate_point_180,
    "rot270": rotate_point_270
}

def normalize_bbox(bbox, width, height):
    """Normaliza las coordenadas del bbox a formato 0-1"""
    return {
        "x": bbox["x"] / width,
        "y": bbox["y"] / height,
        "w": bbox["w"] / width,
        "h": bbox["h"] / height
    }

def extract_base_name(variant_name, base_map):
    """
    Extrae el nombre base de una variante, manejando nombres con múltiples '_'
    Ejemplos:
    - 'warm_chicken_rot90_color1' -> 'warm_chicken'
    - 'allay_rot180_color5' -> 'allay'
    - 'glow_squid_color10' -> 'glow_squid'
    """
    # Lista de todos los posibles sufijos que pueden aparecer después del nombre base
    possible_suffixes = ['rot90', 'rot180', 'rot270'] + [f'color{i}' for i in range(101)]
    
    # Probar combinaciones progresivas de partes
    parts = variant_name.split('_')
    
    for i in range(len(parts), 0, -1):
        candidate_base = '_'.join(parts[:i])
        
        # Verificar si este candidato existe en el base_map
        if candidate_base in base_map:
            return candidate_base
        
        # Verificar si el candidato sin extensión existe
        candidate_stem = Path(candidate_base).stem
        if candidate_stem in base_map:
            return candidate_stem
    
    return None

def convert_gui_to_yolo_segmentation(gui_data):
    """Convierte datos GUI a formato YOLO segmentation con polígonos"""
    rows = []

    for item in gui_data:
        file_name = item["file_name"]
        layer = item.get("layer", "base")
        width = float(item.get("width", 1))
        height = float(item.get("height", 1))

        for obj in item.get("objects", []):
            if not obj.get("enabled", True):
                continue

            class_name = obj.get("class_name", "unknown")
            class_id = obj.get("class_id", CLASS_MAP.get(class_name, -1))
            #layer_id = obj.get("layer_id", LAYER_MAP.get(layer, -1))
            polygon_points = obj.get("polygon", [])
            
            # Validar que tenemos un polígono válido (mínimo 3 puntos)
            if len(polygon_points) < 3:
                continue

            # Normalizar puntos del polígono
            normalized_polygon = []
            for point in polygon_points:
                nx = point["x"] / width
                ny = point["y"] / height
                normalized_polygon.extend([round(nx, 6), round(ny, 6)])

            # Normalizar bbox también
            normalized_bbox = {}
            bbox_data = obj.get("bbox", {})
            if bbox_data and all(k in bbox_data for k in ("x", "y", "w", "h")):
                normalized_bbox = normalize_bbox(bbox_data, width, height)

            rows.append({
                "file_name": file_name,
                #"layer_id": layer_id,
                "layer": layer,
                "class_id": class_id,
                "class_name": class_name,
                "polygon": normalized_polygon,
                "bbox": normalized_bbox
            })

    return rows

def validate_polygon_coordinates(polygon):
    """Valida que todos los puntos del polígono estén dentro de los límites 0-1"""
    for i in range(0, len(polygon), 2):
        x = polygon[i]
        y = polygon[i + 1]
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return False
    return True

def validate_bbox_coordinates(bbox):
    """Valida que las coordenadas del bbox estén dentro de los límites 0-1"""
    if not bbox:
        return True
        
    x = bbox.get("x", 0)
    y = bbox.get("y", 0)
    w = bbox.get("w", 0)
    h = bbox.get("h", 0)
    
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    
    return (x1 >= -0.01 and y1 >= -0.01 and 
            x2 <= 1.01 and y2 <= 1.01 and
            0 < w <= 1 and 0 < h <= 1)

def rotate_bbox(bbox, rot_func):
    """Aplica rotación a un bbox normalizado"""
    if not bbox:
        return {}
    
    x = bbox["x"]
    y = bbox["y"]
    w = bbox["w"]
    h = bbox["h"]
    
    # Calcular centro del bbox
    x_center = x + w/2
    y_center = y + h/2
    
    # Aplicar rotación al centro
    x_center_rot, y_center_rot = rot_func(x_center, y_center)
    
    # Para rotaciones de 90° y 270°, intercambiar width y height
    if rot_func in [rotate_point_90, rotate_point_270]:
        w_rot = h
        h_rot = w
    else:
        w_rot = w
        h_rot = h
    
    # Calcular nueva esquina superior izquierda
    x_rot = x_center_rot - w_rot/2
    y_rot = y_center_rot - h_rot/2
    
    return {
        "x": round(x_rot, 6),
        "y": round(y_rot, 6),
        "w": round(w_rot, 6),
        "h": round(h_rot, 6)
    }

def transform():
    if not os.path.exists(INPUT_JSONL):
        print(f"Error: No se encuentra {INPUT_JSONL}")
        return
    
    # --- Cargar dataset base ---
    gui_data = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            gui_data.append(json.loads(line))

    print(f"Dataset GUI cargado: {len(gui_data)} imágenes")

    rows = convert_gui_to_yolo_segmentation(gui_data)
    
    print(f"Anotaciones convertidas: {len(rows)} polígonos")

    # Mapa base de imágenes - usar gui_data para acceder a los polígonos originales
    base_map = {Path(item["file_name"]).stem: item for item in gui_data}

    # --- Procesar variantes ---
    variant_files = (
        list(VARIANTS_DIR.glob("*.png"))
        + list(VARIANTS_DIR.glob("*.jpg"))
        + list((VARIANTS_DIR / "pbr_maps").glob("*.png"))
        + list((VARIANTS_DIR / "pbr_maps").glob("*.jpg"))
    )
    print(f"Archivos de variantes encontrados: {len(variant_files)}")
    
    if variant_files:
        variant_count = 0
        skipped_files = 0
        found_bases = set()
        
        for file in variant_files:
            name = file.stem

            detected_layer_id, layer_name = detect_layer_from_filename(file.name)

            base_variant_filename = file.name
            if detected_layer_id > 0:
                base_variant_filename = (
                    file.name
                    .replace("_n", "")
                    .replace("_s", "")
                    .replace("_e", "")
                )
                base_lookup_name = Path(base_variant_filename).stem
            else:
                base_lookup_name = name

            base_key = extract_base_name(base_lookup_name, base_map)

            if base_key is None:
                skipped_files += 1
                continue

            if detected_layer_id > 0:
                base_variant_path = VARIANTS_DIR / base_variant_filename
                if not base_variant_path.exists():
                    skipped_files += 1
                    continue

            base_item = base_map[base_key]
            found_bases.add(base_key)
            base_layer = base_item.get("layer", "base")
            current_layer = layer_name if detected_layer_id > 0 else base_layer

            try:
                relative_file_name = str(file.relative_to(VARIANTS_DIR))
            except ValueError:
                relative_file_name = file.name

            # Detectar si es rotación
            rot_func = None
            for rot_tag, func in ROT_POINT_FUNCS.items():
                if rot_tag in name:
                    rot_func = func
                    break

            # Obtener dimensiones de la imagen base
            width = float(base_item.get("width", 1))
            height = float(base_item.get("height", 1))

            # Procesar cada objeto de la imagen base
            for obj in base_item.get("objects", []):
                if not obj.get("enabled", True):
                    continue

                class_name = obj.get("class_name", "unknown")
                class_id = obj.get("class_id", CLASS_MAP.get(class_name, -1))
                #object_layer_id = obj.get("layer_id", LAYER_MAP.get(base_layer, -1))
                polygon_points = obj.get("polygon", [])

                if len(polygon_points) < 3:
                    continue

                # Normalizar puntos del polígono
                normalized_polygon = []
                for point in polygon_points:
                    nx = point["x"] / width
                    ny = point["y"] / height
                    normalized_polygon.extend([round(nx, 6), round(ny, 6)])

                # Normalizar bbox
                normalized_bbox = {}
                bbox_data = obj.get("bbox", {})
                if bbox_data and all(k in bbox_data for k in ("x", "y", "w", "h")):
                    normalized_bbox = normalize_bbox(bbox_data, width, height)

                # Aplicar rotación a cada punto del polígono si corresponde
                if rot_func:
                    rotated_polygon = []
                    for i in range(0, len(normalized_polygon), 2):
                        x = normalized_polygon[i]
                        y = normalized_polygon[i + 1]
                        rx, ry = rot_func(x, y)
                        rotated_polygon.extend([round(rx, 6), round(ry, 6)])
                    normalized_polygon = rotated_polygon

                    # Aplicar rotación al bbox también
                    normalized_bbox = rotate_bbox(normalized_bbox, rot_func)

                rows.append({
                    "file_name": relative_file_name,
                    #"layer_id": detected_layer_id if detected_layer_id > 0 else object_layer_id,
                    "layer": current_layer,
                    "class_id": class_id,
                    "class_name": class_name,
                    "polygon": normalized_polygon,
                    "bbox": normalized_bbox
                })

                variant_count += 1

        print(f"Variantes procesadas: {variant_count} polígonos añadidos")
        print(f"Archivos de variantes omitidos: {skipped_files}")
        print(f"Bases encontradas: {len(found_bases)} - {sorted(found_bases)}")

    # --- Validación final ---
    valid_rows = []
    invalid_polygons = 0
    invalid_bboxes = 0
    
    for row in rows:
        polygon_valid = validate_polygon_coordinates(row["polygon"])
        bbox_valid = validate_bbox_coordinates(row["bbox"])
        
        if polygon_valid and bbox_valid:
            valid_rows.append(row)
        else:
            if not polygon_valid:
                invalid_polygons += 1
            if not bbox_valid:
                invalid_bboxes += 1

    if len(valid_rows) < len(rows):
        invalid_count = len(rows) - len(valid_rows)
        print(f"Se eliminaron {invalid_count} anotaciones inválidas:")
        print(f"  - Polígonos fuera de rango: {invalid_polygons}")
        print(f"  - Bboxes fuera de rango: {invalid_bboxes}")

    df = pd.DataFrame(valid_rows)
    if df.empty:
        print("No se generaron filas válidas")
        return

    # Reorganizar columnas para mejor legibilidad
    column_order = ["file_name", "layer", "class_id", "class_name", "polygon", "bbox"]#["file_name", "layer_id", "layer", "class_id", "class_name", "polygon", "bbox"]
    df = df[column_order]

    # --- Guardar dataset final ---
    csv_path = OUT_DIR / "data.csv"
    parquet_path = OUT_DIR / "data.parquet"
    jsonl_path = OUT_DIR / "data.jsonl"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    df.to_json(jsonl_path, orient="records", lines=True)

    print(f"Dataset final guardado")
    print(f"Imágenes: {df['file_name'].nunique()}")
    print(f"Anotaciones totales: {len(df)} (polígonos para segmentación)")

if __name__ == "__main__":
    transform()
