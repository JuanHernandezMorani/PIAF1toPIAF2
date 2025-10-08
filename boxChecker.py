import json
from pathlib import Path
from PIL import Image, ImageDraw

DATA_JSONL = "trainData/data.jsonl"
ROOT_DIR = Path("input")
VARIANTS_DIR = Path("variants")
OUTPUT_DIR = Path("boxCheck")
OUTPUT_DIR.mkdir(exist_ok=True)

# Colores por clase en formato RGB
COLORS = {
    "eyes": (255, 0, 0),           # red
    "wings": (0, 0, 255),          # blue
    "body": (0, 255, 0),           # green
    "aletas": (255, 255, 0),       # yellow
    "extremities": (255, 165, 0),  # orange
    "fangs": (128, 0, 128),        # purple
    "claws": (0, 255, 255),        # cyan
    "head": (255, 192, 203),       # pink
    "mouth": (139, 0, 0),          # darkred
    "heart": (255, 140, 0),        # darkorange
    "cracks": (211, 211, 211),     # lightgrey
    "cristal": (238, 130, 238),    # violet
    "flower": (144, 238, 144),     # lightgreen
    "zombie_zone": (0, 100, 0),    # darkgreen
    "armor": (0, 0, 139),          # darkblue
    "sky": (173, 216, 230),        # lightblue
    "stars": (255, 255, 255),      # white
    "extra": (255, 0, 255)         # magenta
}

# Configuración de visualización
SHOW_LABELS = False
LINE_WIDTH = 1
FILL_OPACITY = 25

def load_segmentation_data(file_path):
    """Carga datos de segmentación con polígonos"""
    data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    file_name = item.get('file_name', '').strip()
                    if file_name:
                        if file_name not in data:
                            data[file_name] = []
                        data[file_name].append(item)
                except json.JSONDecodeError:
                    continue
        return data
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return {}

def draw_polygon_segmentation(draw, polygon, color_rgb, class_name, img_width, img_height):
    """Dibuja polígonos de segmentación con líneas delgadas"""
    
    # Convertir coordenadas normalizadas a píxeles
    points = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] * img_width
        y = polygon[i + 1] * img_height
        points.append((x, y))
    
    if len(points) < 3:
        return
    
    # Crear color con transparencia para el relleno (RGBA)
    fill_color = color_rgb + (FILL_OPACITY,)  # (R, G, B, A)
    
    # Dibujar polígono relleno con transparencia
    draw.polygon(points, fill=fill_color, outline=color_rgb, width=LINE_WIDTH)
    
    # Dibujar etiqueta si está habilitado
    if SHOW_LABELS and points:
        # Calcular centro del polígono
        x_center = sum(p[0] for p in points) / len(points)
        y_center = sum(p[1] for p in points) / len(points)
        
        label = f"{class_name}"
        # Usar textbbox para obtener dimensiones del texto
        bbox = draw.textbbox((0, 0), label)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Fondo para la etiqueta
        text_bg = [
            x_center - text_width/2 - 2,
            y_center - text_height/2 - 2,
            x_center + text_width/2 + 2,
            y_center + text_height/2 + 2
        ]
        draw.rectangle(text_bg, fill=color_rgb)
        draw.text((x_center - text_width/2, y_center - text_height/2), label, fill="black")

def process_image_segmentation(file_name, annotations):
    """Procesa una imagen y dibuja sus anotaciones de segmentación"""
    # Buscar imagen en directorios
    img_path = ROOT_DIR / file_name
    if not img_path.exists():
        img_path = VARIANTS_DIR / file_name
        if not img_path.exists():
            print(f"❌ Imagen no encontrada: {file_name}")
            return False

    try:
        # Abrir imagen
        img = Image.open(img_path).convert("RGBA")
        W, H = img.size
        
        # Crear capa de dibujo
        draw = ImageDraw.Draw(img, "RGBA")
        
        # Procesar cada anotación
        valid_annotations = 0
        for annotation in annotations:
            class_name = annotation.get('class_name', 'unknown')
            color_rgb = COLORS.get(class_name, (128, 128, 128))  # gray por defecto
            polygon = annotation.get('polygon', [])
            
            if len(polygon) >= 6:  # Mínimo 3 puntos (x,y,x,y,x,y)
                draw_polygon_segmentation(draw, polygon, color_rgb, class_name, W, H)
                valid_annotations += 1
        
        # Guardar imagen resultante
        output_path = OUTPUT_DIR / file_name
        img.save(output_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error procesando {file_name}: {e}")
        return False

def validate_dataset(data):
    """Valida el dataset y muestra estadísticas"""
    total_images = len(data)
    total_annotations = sum(len(annotations) for annotations in data.values())
    total_polygons = 0
    
    for annotations in data.values():
        for ann in annotations:
            polygon = ann.get('polygon', [])
            if len(polygon) >= 6:  # Mínimo 3 puntos
                total_polygons += 1
    
    print(f"📊 Estadísticas del dataset:")
    print(f"   - Imágenes: {total_images}")
    print(f"   - Anotaciones totales: {total_annotations}")
    print(f"   - Polígonos válidos: {total_polygons}")
    print(f"   - Configuración: Línea={LINE_WIDTH}px, Relleno={FILL_OPACITY}/255")

def main():
    print("=== Validador de Segmentación con Polígonos ===")
    
    # Cargar datos
    data = load_segmentation_data(DATA_JSONL)
    if not data:
        print("❌ No se pudieron cargar datos de segmentación")
        return
    
    # Validar dataset
    validate_dataset(data)
    
    # Encontrar archivos existentes
    existing_files = set()
    for pattern in ["*.png", "*.jpg"]:
        existing_files.update(f.name for f in ROOT_DIR.glob(pattern))
        existing_files.update(f.name for f in VARIANTS_DIR.glob(pattern))
    
    # Filtrar datos por archivos existentes
    valid_data = {f: data[f] for f in data.keys() if f in existing_files}
    missing_files = set(data.keys()) - existing_files
    
    if missing_files:
        print(f"⚠️  Se omitirán {len(missing_files)} imágenes no encontradas")
        if len(missing_files) <= 5:
            for f in missing_files:
                print(f"      - {f}")
    
    # Procesar imágenes
    success_count = 0
    total_to_process = len(valid_data)
    
    print(f"\n🔄 Procesando {total_to_process} imágenes...")
    
    for i, (file_name, annotations) in enumerate(valid_data.items()):
        if process_image_segmentation(file_name, annotations):
            success_count += 1
        
        # Mostrar progreso cada 100 imágenes
        if (i + 1) % 100 == 0:
            print(f"   📦 Procesadas {i + 1}/{total_to_process} imágenes...")
    
    # Resultados finales
    print(f"\n✅ Resultados:")
    print(f"   - Imágenes procesadas: {success_count}/{total_to_process}")
    print(f"   - Imágenes validadas guardadas en: {OUTPUT_DIR}")
    
    # Sugerencias de configuración
    print(f"\n💡 Sugerencias:")
    print(f"   - Para líneas MÁS delgadas: Cambia LINE_WIDTH = 1")
    print(f"   - Para sin relleno: Cambia FILL_OPACITY = 0")
    print(f"   - Para más relleno: Cambia FILL_OPACITY = 100")
    print(f"   - Para ver etiquetas: Cambia SHOW_LABELS = True")

if __name__ == "__main__":
    main()
    