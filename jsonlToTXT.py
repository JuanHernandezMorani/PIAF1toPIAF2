import json
from pathlib import Path

def convert_jsonl_to_yolo_txt(jsonl_path, output_dir):
    """Convierte JSONL de segmentación a archivos TXT formato YOLO"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Agrupar anotaciones por imagen
    image_annotations = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            file_name = item['file_name']
            base_name = Path(file_name).stem
            
            if base_name not in image_annotations:
                image_annotations[base_name] = []
            
            layer_id = item['layer_id']
            class_id = item['class_id']
            polygon = item['polygon']
            
            # Formato YOLO: class_id x1 y1 x2 y2 ... xn yn
            annotation_line = [str(layer_id)] + [str(class_id)] + [str(coord) for coord in polygon]
            image_annotations[base_name].append(" ".join(annotation_line))
    
    # Escribir archivos TXT
    for base_name, annotations in image_annotations.items():
        txt_path = output_dir / f"{base_name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for line in annotations:
                f.write(line + '\n')
    
    print(f"✅ Convertidas {len(image_annotations)} imágenes a formato YOLO TXT")
    print(f"📁 Archivos guardados en: {output_dir}")

# Uso
if __name__ == "__main__":
    convert_jsonl_to_yolo_txt(
        "trainData/data.jsonl", 
        "trainData/labels"
    )