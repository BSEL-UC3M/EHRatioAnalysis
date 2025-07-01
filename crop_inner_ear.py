import os
import cv2
import tifffile as tiff
import numpy as np

# Carpeta base de entrada y salida
INPUT_BASE = "/Users/claudiacastrillonalvarez/Desktop/inner_ear_slides-104-107"
OUTPUT_BASE = "/Users/claudiacastrillonalvarez/Desktop/CROPPED_INNER_EAR"

# Tamaño del recorte
CROP_SIZE = 96

# Pacientes y modalidades a procesar
pacientes = [f"PACIENTE_{i}" for i in range(104, 108)]
modalidades = ["MRC", "PEI"]

# Centroides por modalidad
CENTROIDS_BY_TYPE = {
    "MRC": {
        "right": (94, 160),
        "left": (240, 160)
    },
    "PEI": {
        "right": (94, 180),
        "left": (240, 180)
    }
}

def crop_and_save(image_path, output_folder, centroids):
    image = tiff.imread(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for side, (cx, cy) in centroids.items():
        x1, x2 = max(cx - CROP_SIZE // 2, 0), min(cx + CROP_SIZE // 2, image.shape[1])
        y1, y2 = max(cy - CROP_SIZE // 2, 0), min(cy + CROP_SIZE // 2, image.shape[0])
        cropped_img = image[y1:y2, x1:x2]

        output_filename = f"{base_name}_{side}.tif"
        output_path = os.path.join(output_folder, output_filename)
        tiff.imwrite(output_path, cropped_img)
        print(f"✅ Guardado: {output_filename}")

# Procesar todos los pacientes y modalidades
for paciente in pacientes:
    for modalidad in modalidades:
        input_folder = os.path.join(INPUT_BASE, paciente, modalidad)
        output_folder = os.path.join(OUTPUT_BASE, paciente, modalidad)

        if not os.path.isdir(input_folder):
            print(f"⚠️ Carpeta no encontrada: {input_folder}")
            continue

        os.makedirs(output_folder, exist_ok=True)
        centroids = CENTROIDS_BY_TYPE[modalidad]

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))]
        print(f"\n🔍 Procesando {len(image_files)} imágenes en {input_folder}")

        for img_file in image_files:
            img_path = os.path.join(input_folder, img_file)
            crop_and_save(img_path, output_folder, centroids)

print("\n🎉 Proceso de recorte completado.")
