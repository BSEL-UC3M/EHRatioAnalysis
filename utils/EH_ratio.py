# ======================================================================
# File: EH_ratio.py
# Description: Calculates EH Ratio using predicted segmentation masks.
# Author: @laurarodrmu
# Created: 16/04/2025
# ======================================================================

import os
import re
import csv
import numpy as np
from glob import glob
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import center_of_mass, shift




def extract_patient_and_ear(filename):
    """
    Extracts the patient number and last word from a filename like 'PAC1_left_main_right'.

    Parameters:
        filename (str): Filename without extension or path.

    Returns:
        tuple: (patient_id: int, last_part: str)
    """
    name = os.path.splitext(os.path.basename(filename))[0]  
    parts = name.split("_")

    
    match = re.match(r'PAC(\d+)', parts[0])
    patient_id = int(match.group(1)) if match else None


    ear = parts[-1] if parts else None

    return patient_id, ear



def compute_mask_volume(mask, voxel_volume_mm3):
    print(float(np.sum(mask > 0)))
    print(float(np.sum(mask > 0) * voxel_volume_mm3))
    return float(np.sum(mask > 0) * voxel_volume_mm3)

def collect_volumes_from_folder(folder_path, voxel_volume_mm3):
    """
    Parses all mask images in the folder and computes volumes per patient and ear.
    Returns a dict: {(patient_id, ear): volume_mm3}
    """
    volume_dict = defaultdict(float)
    mask_paths = sorted(glob(os.path.join(folder_path, "*.tif.png")))

    for mask_path in mask_paths:
        patient_id, ear = extract_patient_and_ear(mask_path)
        if patient_id is None or ear is None:
            print(f"⚠️ Skipping unrecognized mask filename: {mask_path}")
            continue
        # TODO: 
        mask = np.array(Image.open(mask_path))
        volume = compute_mask_volume(mask, voxel_volume_mm3)
        volume_dict[(patient_id, ear)] += volume

    return volume_dict

def save_volume_ratios_csv(mrc_volumes, pei_volumes, output_csv):
    """
    Saves the merged MRC/PEI volume table with EH ratio to a CSV.
    """
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["patient_id", "ear", "mrc_volume_mm3", "pei_volume_mm3", "eh_ratio"])

        all_keys = set(mrc_volumes.keys()) | set(pei_volumes.keys())
        for (pid, ear) in sorted(all_keys):
            mrc_vol = mrc_volumes.get((pid, ear), 0.0)
            pei_vol = pei_volumes.get((pid, ear), 0.0)
            ratio = pei_vol / mrc_vol if mrc_vol > 0 else ""
            writer.writerow([pid, ear, f"{mrc_vol:.2f}", f"{pei_vol:.2f}", f"{ratio:.3f}" if ratio != "" else ""])

    print(f"\n📊 EH ratio table saved to: {output_csv}")

def compute_eh_ratios(
    mrc_mask_folder,
    pei_mask_folder,
    output_csv_path,
    mrc_voxel_size=(0.5, 0.5, 0.5),
    pei_voxel_size=(0.5, 0.5, 0.8),
):
    """
    Computes EH ratio using segmentation masks in MRC and PEI folders.
    """
    print("\n📐 Calculating EH volume ratios per patient & ear...")

    mrc_voxel_volume = np.prod(mrc_voxel_size)
    pei_voxel_volume = np.prod(pei_voxel_size)

    mrc_volumes = collect_volumes_from_folder(mrc_mask_folder, voxel_volume_mm3=mrc_voxel_volume)
    pei_volumes = collect_volumes_from_folder(pei_mask_folder, voxel_volume_mm3=pei_voxel_volume)

    save_volume_ratios_csv(mrc_volumes, pei_volumes, output_csv_path)

# ======================================================================
# VISUALIZATIONS 


import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, shift

def cargar_mascara(ruta):
    return cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

def calcular_centroides(mascara):
    return np.array(center_of_mass(mascara))

def calcular_desplazamiento(centro_cavidad, centro_liquido):
    return centro_cavidad - centro_liquido

def alinear_mascara_liquido(mascara_liquido, vector_traslacion):
    return shift(mascara_liquido.astype(float), shift=vector_traslacion, order=0)

def superponer_colores(mask_cav, mask_liq):
    cav = (mask_cav > 0).astype(np.uint8)
    liq = (mask_liq > 0).astype(np.uint8)
    
    h, w = cav.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] += cav * 255  
    rgb[:, :, 1] += liq * 255 
    
    return rgb

def recortar_region_activa(mask1, mask2, margen=5):
    combinada = (mask1 > 0) | (mask2 > 0)
    filas, columnas = np.where(combinada)

    if filas.size == 0 or columnas.size == 0:
        return mask1, mask2, mask1 + mask2

    min_fila = max(np.min(filas) - margen, 0)
    max_fila = min(np.max(filas) + margen, mask1.shape[0])
    min_col = max(np.min(columnas) - margen, 0)
    max_col = min(np.max(columnas) + margen, mask1.shape[1])

    rec1 = mask1[min_fila:max_fila, min_col:max_col]
    rec2 = mask2[min_fila:max_fila, min_col:max_col]
    super_rec = rec1 + rec2

    return rec1, rec2, super_rec

import matplotlib.lines as mlines

def visualizar_resultados(rec_cavidad, rec_liquido, rec_superpuesta):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Máscara Cavidad")
    plt.imshow(rec_cavidad, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Máscara Líquido Alineada")
    plt.imshow(rec_liquido, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Superposición")
    rgb_superpuesta = superponer_colores(rec_cavidad, rec_liquido)
    plt.imshow(rgb_superpuesta)

    red_patch = mlines.Line2D([], [], marker='o', color='r', label="Cavidad", markersize=10)
    green_patch = mlines.Line2D([], [], marker='o', color='g', label="Líquido", markersize=10)
    yellow_patch = mlines.Line2D([], [], marker='o', color='y', label="Superposición", markersize=10)

    plt.legend(handles=[red_patch, green_patch, yellow_patch], loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def procesar_imagenes(mrc_path, pei_path):

    mascara_cavidad = cargar_mascara(mrc_path)
    mascara_liquido = cargar_mascara(pei_path)

    centro_cavidad = calcular_centroides(mascara_cavidad)
    centro_liquido = calcular_centroides(mascara_liquido)

    vector_traslacion = calcular_desplazamiento(centro_cavidad, centro_liquido)

    mascara_liquido_alineada = alinear_mascara_liquido(mascara_liquido, vector_traslacion)

    rec_cavidad, rec_liquido, rec_superpuesta = recortar_region_activa(mascara_cavidad, mascara_liquido_alineada)

    visualizar_resultados(rec_cavidad, rec_liquido, rec_superpuesta)
    
    area_cavidad, area_liquido, ratio = calcular_areas(rec_cavidad, rec_liquido)
    print(f"Área Cavidad: {area_cavidad:.2f} mm²")
    print(f"Área Líquido: {area_liquido:.2f} mm²")
    print(f"Ratio Líquido/Cavidad: {ratio:.3f}")



def calcular_areas(mask1, mask2, tam_pixel_mm=0.5):
    area_pixel = tam_pixel_mm ** 2  

    bin1 = (mask1 > 0).astype(np.uint8)
    bin2 = (mask2 > 0).astype(np.uint8)

    num_pix1 = np.sum(bin1)
    num_pix2 = np.sum(bin2)

    area1 = num_pix1 * area_pixel
    area2 = num_pix2 * area_pixel

    ratio = area2 / area1 if area1 != 0 else float('inf')

    return area1, area2, ratio





