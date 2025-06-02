import os
import cv2
import tifffile as tiff  # Para manejar imágenes TIFF
import numpy as np

# Configuración de carpetas
INPUT_FOLDER = "/Users/claudiacastrillonalvarez/Desktop/MRC_INNER_EAR/PACIENTE_97"  # Cambia esto según tu dataset
OUTPUT_FOLDER = "/Users/claudiacastrillonalvarez/Desktop/MRC_INNER_EAR/CROPPED/PACIENTE_97"  # Carpeta de salida

# Centroides definidos
CENTROIDS = {
    "right": (94, 160), #para peo: (94, 180)
    "left": (240, 160)  #para pei: (240, 180)
}

CROP_SIZE = 96  # Tamaño de los crops (96x96)

# Crear la carpeta de salida si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Listar todas las imágenes TIFF en la carpeta de entrada
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".tif")]

# Función para hacer los recortes y guardarlos
def crop_and_save_images(image_path, output_folder):
    # Cargar la imagen en escala de grises o en RGB según sea necesario
    image = tiff.imread(image_path)

    # Obtener el nombre del archivo sin extensión
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for side, (cx, cy) in CENTROIDS.items():
        # Calcular los bordes del recorte asegurando que no salimos de la imagen
        x1, x2 = max(cx - CROP_SIZE // 2, 0), min(cx + CROP_SIZE // 2, image.shape[1])
        y1, y2 = max(cy - CROP_SIZE // 2, 0), min(cy + CROP_SIZE // 2, image.shape[0])

        # Realizar el recorte
        cropped_img = image[y1:y2, x1:x2]

        # Guardar el recorte con el nuevo nombre
        output_filename = f"{base_name}_{side}.tif"
        output_path = os.path.join(output_folder, output_filename)
        tiff.imwrite(output_path, cropped_img)

        print(f"Guardado: {output_filename}")

# Procesar todas las imágenes
for img_file in image_files:
    img_path = os.path.join(INPUT_FOLDER, img_file)
    crop_and_save_images(img_path, OUTPUT_FOLDER)

print("Proceso de recorte completado ✅. Las imágenes están en:", OUTPUT_FOLDER)