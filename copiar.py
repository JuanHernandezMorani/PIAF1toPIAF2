import os
import shutil

def organizar_labels():
    # Rutas de entrada y salida
    input_dir = r'C:\Users\braia\OneDrive\Desktop\AyiAcademy\Parciales_Finales\PF2_IA\generator\trainData\labels'
    train_dir = r'C:\Users\braia\OneDrive\Desktop\AyiAcademy\Parciales_Finales\PF2_IA\generator\trainData\output\train'
    val_dir = r'C:\Users\braia\OneDrive\Desktop\AyiAcademy\Parciales_Finales\PF2_IA\generator\trainData\output\val'

    # Crear directorios de salida si no existen
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Procesar cada archivo en el directorio de entrada
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            # Extraer el valor X del nombre del archivo
            if '_rot' in filename:
                # Caso *_rot*_colorX.png
                parts = filename.split('_color')
            elif '_color' in filename:
                # Caso *_colorX.png
                parts = filename.split('_color')
            else:
                continue
            
            if len(parts) < 2:
                continue  # Saltar si no coincide el patrón

            # Obtener el número X
            x_part = parts[1].split('.')[0]  # Quitar la extensión
            try:
                x = int(x_part)
            except ValueError:
                continue  # Saltar si no es un número válido

            # Determinar directorio destino
            if x < 81:
                dest_dir = train_dir
            else:
                dest_dir = val_dir

            # Copiar archivo
            src_path = os.path.join(input_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy2(src_path, dest_path)
            print(f'Copiado: {filename} -> {dest_dir}')

if __name__ == '__main__':
    organizar_labels()