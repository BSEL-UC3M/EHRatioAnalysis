import os
import pydicom
import numpy as np
from matplotlib.image import imsave
from pydicom.errors import InvalidDicomError

# Define input and output folders
input_folder = '/Users/claudiacastrillonalvarez/Desktop/PACIENTES_103_A_107/PACIENTE_107/DICOM/25061210/04240000'
output_folder = '/Users/claudiacastrillonalvarez/Desktop/patients_tiff_103_107/PACIENTE_107'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # Saltar carpetas y archivos invisibles
    if not os.path.isfile(file_path) or filename.startswith('.'):
        continue

    try:
        # Leer con force=True por si no tiene cabecera estándar
        dicom = pydicom.dcmread(file_path, force=True)

        if not hasattr(dicom, 'PixelData'):
            print(f"[×] Sin PixelData: {filename}")
            continue

        image = dicom.pixel_array

        # Normalizar a [0, 1]
        image = image.astype(np.float32)
        image -= np.min(image)
        image /= (np.max(image) + 1e-8)

        # Guardar como TIFF
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.tiff")
        imsave(output_path, image, cmap='gray')
        print(f"[✔] Exportado: {output_path}")

    except (InvalidDicomError, AttributeError, KeyError, FileNotFoundError, Exception) as e:
        print(f"[×] Error en {filename}: {e}")
