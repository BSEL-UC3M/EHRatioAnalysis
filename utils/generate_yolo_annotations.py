# ==============================================================================
# File: generate_yolo_annotations.py
# Description: 
# This script processes the CSV file generated using the `object_det_annotations_macro.ijm` macro in Fiji.
# The macro allows users to annotate images in Fiji, and this script converts the CSV output into the 
# YOLO annotation format, which is required for training an object detection model.
#
# The script extracts bounding box coordinates from the CSV and normalizes them to YOLO format.
# It then saves annotation files in the correct folders for each patient, following the structure:
# `PACIENTE X MRC TIFF/yolo_annotations/`
#
# Author: @cfusterbarcelo
# Creation Date: 05/01/2025
# Last Update: 26/03/2025
# ==============================================================================

# import os
# import pandas as pd
# import cv2
# import re
# import shutil
# import random

# # Define dataset directories
# base_dataset_folder = "/Users/claudiacastrillonalvarez/Desktop/IMAGES_YOLO_toydataset/PEI_YOLO_toydataset/PEI"
# yolo_annotations_folder="/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis"
# yolo_dataset_folder = os.path.join(yolo_annotations_folder, "YOLO_annotations_toydataset_PEI")  # Static YOLO dataset folder

# # Define split ratios
# TRAIN_RATIO = 0.7
# VAL_RATIO = 0.15
# TEST_RATIO = 0.15

# # ✅ Ensure YOLO directory exists
# os.makedirs(yolo_dataset_folder, exist_ok=True)

# # Load all CSV files
# csv_files = [f for f in os.listdir(base_dataset_folder) if f.endswith('.csv')]

# if not csv_files:
#     print("❌ No CSV files found in the dataset folder!")
#     exit()
# # csv_path="/Users/claudiacastrillonalvarez/Desktop/IMAGES_YOLO_toydataset/MRC_YOLO_toydataset/MRC_coordinates_toydataset.csv"
# csv_path = os.path.join(base_dataset_folder, csv_files[0])  # Assuming only one CSV file
# print(f"🔹 Processing CSV file: {csv_path}")

# # Read the CSV file
# import csv

# df = pd.read_csv(
#     csv_path,
#     encoding="latin1",
#     skiprows=1,
#     header=None,
#     names=["filename", "x_left", "y_left", "x_right", "y_right"],
#     sep=None,              # 🔍 autodetecta separador
#     engine="python",       # 🧠 necesario para usar sep=None
#     on_bad_lines="skip"    # 🧼 ignora líneas problemáticas
# )

# print("CSV loaded correctly ✅")
# print(df.head())
# print("Número total de filas procesadas:", len(df))


# print("First few filenames from CSV:")
# print(df["filename"].head())

# # Extract all patient IDs
# def extract_patient_number(filename):
#     if isinstance(filename, float) or pd.isna(filename):
#         return None
#     filename = str(filename)
    
#     # Detect dataset type based on folder
#     if "MRC" in base_dataset_folder:
#         match = re.search(r"MRC_(\d+)_", filename)  # e.g., MRC_12_12345678.tif → 12
#     elif "PEI" in base_dataset_folder:
#         match = re.search(r"PEI_(\d+)_", filename)  # e.g., PEI_3_59160744.tif → 3
#     else:
#         match = None
    
#     return match.group(1) if match else None


# patients = sorted(set(df["filename"].astype(str).apply(extract_patient_number).dropna()))

# # ✅ Randomly shuffle patients ONCE and store split
# random.shuffle(patients)

# num_train = int(len(patients) * TRAIN_RATIO)
# num_val = int(len(patients) * VAL_RATIO)

# train_patients = set(patients[:num_train])
# val_patients = set(patients[num_train:num_train + num_val])
# test_patients = set(patients[num_train + num_val:])

# # ✅ Ensure no patient is in multiple splits
# assert train_patients.isdisjoint(val_patients), "❌ ERROR: A patient appears in both train and validation!"
# assert train_patients.isdisjoint(test_patients), "❌ ERROR: A patient appears in both train and test!"
# assert val_patients.isdisjoint(test_patients), "❌ ERROR: A patient appears in both validation and test!"

# print(f"✅ Patients successfully split:")
# print(f"   - Train: {len(train_patients)} patients")
# print(f"   - Val: {len(val_patients)} patients")
# print(f"   - Test: {len(test_patients)} patients")

# # ✅ Move images & annotations to YOLO/train, val, test folders
# for _, row in df.iterrows():
#     image_filename = row["filename"]
#     patient_number = extract_patient_number(image_filename)

#     if patient_number is None:
#         print(f"⚠️ Could not determine patient number for {image_filename}, skipping...")
#         continue

#     # Determine the split based on the patient ID
#     if patient_number in train_patients:
#         split_folder = "train"
#     elif patient_number in val_patients:
#         split_folder = "val"
#     else:
#         split_folder = "test"

#     # Build paths
#     if "MRC" in base_dataset_folder:
#         patient_folder = f"PACIENTE {patient_number} MRC TIFF"
#     elif "PEI" in base_dataset_folder:
#         patient_folder = f"PACIENTE {patient_number} PEI TIFF"
#     else:
#         patient_folder = f"PACIENTE {patient_number}"

#     source_image_path = os.path.join(base_dataset_folder, patient_folder, image_filename)
#     dest_image_path = os.path.join(yolo_dataset_folder, split_folder, image_filename)

#     # Copy image
#     if os.path.exists(source_image_path):
#         shutil.copy(source_image_path, dest_image_path)
#     else:
#         print(f"❌ Warning: Image {source_image_path} not found, skipping...")
#         continue  # No point generating annotation if image doesn't exist

#     # Create annotation path (inside split folder directly)
#     annotation_output_folder = os.path.join(yolo_dataset_folder, split_folder)
#     os.makedirs(annotation_output_folder, exist_ok=True)
#     annotation_path = os.path.join(annotation_output_folder, image_filename.replace(".tif", ".txt"))

#     # Get bounding box coordinates
#     x_left, y_left, x_right, y_right = row["x_left"], row["y_left"], row["x_right"], row["y_right"]

#     if pd.isna(x_left) or pd.isna(y_left) or pd.isna(x_right) or pd.isna(y_right):
#         print(f"⚠️ Skipping {image_filename} due to missing bounding box data.")
#         continue

#     image = cv2.imread(source_image_path)
#     if image is None:
#         print(f"❌ Warning: Could not load {source_image_path}, skipping annotation generation.")
#         continue
#     img_height, img_width = image.shape[:2]

#     with open(annotation_path, "w") as f:
#         bbox_size = 96

#         x_min_left = max(0, x_left - bbox_size // 2)
#         y_min_left = max(0, y_left - bbox_size // 2)
#         x_max_left = min(img_width, x_min_left + bbox_size)
#         y_max_left = min(img_height, y_min_left + bbox_size)

#         x_min_right = max(0, x_right - bbox_size // 2)
#         y_min_right = max(0, y_right - bbox_size // 2)
#         x_max_right = min(img_width, x_min_right + bbox_size)
#         y_max_right = min(img_height, y_min_right + bbox_size)

#         x_center_left = (x_min_left + x_max_left) / 2 / img_width
#         y_center_left = (y_min_left + y_max_left) / 2 / img_height
#         width_left = (x_max_left - x_min_left) / img_width
#         height_left = (y_max_left - y_min_left) / img_height

#         x_center_right = (x_min_right + x_max_right) / 2 / img_width
#         y_center_right = (y_min_right + y_max_right) / 2 / img_height
#         width_right = (x_max_right - x_min_right) / img_width
#         height_right = (y_max_right - y_min_right) / img_height

#         f.write(f"0 {x_center_left:.6f} {y_center_left:.6f} {width_left:.6f} {height_left:.6f}\n")
#         f.write(f"1 {x_center_right:.6f} {y_center_right:.6f} {width_right:.6f} {height_right:.6f}\n")

#     print(f"✅ Annotation saved: {annotation_path}")



# # ✅ Generate `dataset.yaml`
# dataset_yaml_path = os.path.join(yolo_dataset_folder, "dataset.yaml")
# with open(dataset_yaml_path, "w") as f:
#     f.write(f"train: {os.path.abspath(os.path.join(yolo_dataset_folder, 'train'))}\n")
#     f.write(f"val: {os.path.abspath(os.path.join(yolo_dataset_folder, 'val'))}\n")
#     f.write(f"test: {os.path.abspath(os.path.join(yolo_dataset_folder, 'test'))}\n")
#     f.write("nc: 2\n")  # Number of classes (left ear, right ear)
#     f.write("names: ['left ear', 'right ear']\n")

# print(f"✅ Dataset split complete! `dataset.yaml` created at: {dataset_yaml_path}")

# ==============================================================================
# File: generate_yolo_annotations.py
# Description: Converts bounding box annotations to YOLO format and organizes
# images and labels in train/val/test folders based on patient ID.
# Author: @cfusterbarcelo (modificado por claudia)
# Last Update: 26/03/2025
# ==============================================================================

import os
import pandas as pd
import cv2
import re
import shutil
import random

# === Configuración de rutas ===
base_dataset_folder = "/Users/claudiacastrillonalvarez/Desktop/IMAGES_YOLO/PEI_YOLO/PEI"
yolo_annotations_folder = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis"
yolo_dataset_folder = os.path.join(yolo_annotations_folder, "YOLO_annotations_PEI")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

os.makedirs(yolo_dataset_folder, exist_ok=True)

# === Leer CSV ===
csv_files = [f for f in os.listdir(base_dataset_folder) if f.endswith('.csv')]
if not csv_files:
    print("❌ No CSV files found in the dataset folder!")
    exit()

csv_path = os.path.join(base_dataset_folder, csv_files[0])
print(f"🔹 Processing CSV file: {csv_path}")

df = pd.read_csv(
    csv_path,
    encoding="latin1",
    skiprows=1,
    header=None,
    names=["filename", "x_left", "y_left", "x_right", "y_right"],
    sep=None,
    engine="python",
    on_bad_lines="skip"
)
df = df.dropna(subset=["filename"])
print("CSV loaded correctly ✅")
print(df.head())

# === Extraer ID de paciente ===
def extract_patient_number(filename):
    if isinstance(filename, float) or pd.isna(filename):
        return None
    filename = str(filename)
    if "MRC" in base_dataset_folder:
        match = re.search(r"MRC_(\d+)_", filename)
    elif "PEI" in base_dataset_folder:
        match = re.search(r"PEI_(\d+)_", filename)
    else:
        match = None
    return match.group(1) if match else None

patients = sorted(set(df["filename"].astype(str).apply(extract_patient_number).dropna()))
random.shuffle(patients)

num_train = int(len(patients) * TRAIN_RATIO)
num_val = int(len(patients) * VAL_RATIO)
train_patients = set(patients[:num_train])
val_patients = set(patients[num_train:num_train + num_val])
test_patients = set(patients[num_train + num_val:])

assert train_patients.isdisjoint(val_patients)
assert train_patients.isdisjoint(test_patients)
assert val_patients.isdisjoint(test_patients)

print(f"✅ Patients successfully split:")
print(f"   - Train: {len(train_patients)} patients")
print(f"   - Val: {len(val_patients)} patients")
print(f"   - Test: {len(test_patients)} patients")

# === Generar anotaciones y copiar imágenes ===
for _, row in df.iterrows():
    image_filename = row["filename"]
    patient_number = extract_patient_number(image_filename)

    if patient_number is None:
        print(f"⚠️ Could not determine patient number for {image_filename}, skipping...")
        continue

    if patient_number in train_patients:
        split_folder = "train"
    elif patient_number in val_patients:
        split_folder = "val"
    else:
        split_folder = "test"

    if "MRC" in base_dataset_folder:
        patient_folder = f"PACIENTE {patient_number} MRC TIFF"
    elif "PEI" in base_dataset_folder:
        patient_folder = f"PACIENTE {patient_number} PEI TIFF"
    else:
        patient_folder = f"PACIENTE {patient_number}"

    source_image_path = os.path.join(base_dataset_folder, patient_folder, image_filename)
    dest_image_path = os.path.join(yolo_dataset_folder, split_folder, image_filename)
    annotation_path = os.path.join(yolo_dataset_folder, split_folder, image_filename.replace(".tif", ".txt"))

    os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)

    if not os.path.exists(source_image_path):
        print(f"❌ Warning: Image {source_image_path} not found, skipping...")
        continue

    shutil.copy(source_image_path, dest_image_path)

    x_left, y_left, x_right, y_right = row["x_left"], row["y_left"], row["x_right"], row["y_right"]

    if pd.isna(x_left) or pd.isna(y_left) or pd.isna(x_right) or pd.isna(y_right):
        print(f"⚠️ Skipping {image_filename} due to missing bounding box data.")
        continue
    import tifffile as tiff
    image = tiff.imread(source_image_path)

    if image is None:
        print(f"❌ Warning: Could not load {source_image_path} with cv2.IMREAD_UNCHANGED")
        continue
    else:
        print(f"✅ Imagen cargada: {source_image_path} - shape: {image.shape}")


    img_height, img_width = image.shape[:2]
    bbox_size = 96

    def convert_to_yolo(x, y):
        x_min = max(0, x - bbox_size // 2)
        y_min = max(0, y - bbox_size // 2)
        x_max = min(img_width, x_min + bbox_size)
        y_max = min(img_height, y_min + bbox_size)
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        return x_center, y_center, width, height

    x1, y1, w1, h1 = convert_to_yolo(x_left, y_left)
    x2, y2, w2, h2 = convert_to_yolo(x_right, y_right)

    with open(annotation_path, "w") as f:
        f.write(f"0 {x1:.6f} {y1:.6f} {w1:.6f} {h1:.6f}\n")  # left ear
        f.write(f"1 {x2:.6f} {y2:.6f} {w2:.6f} {h2:.6f}\n")  # right ear

    print(f"✅ Annotation saved: {annotation_path}")

# === Crear dataset.yaml ===
dataset_yaml_path = os.path.join(yolo_dataset_folder, "dataset.yaml")
with open(dataset_yaml_path, "w") as f:
    f.write(f"train: {os.path.abspath(os.path.join(yolo_dataset_folder, 'train'))}\n")
    f.write(f"val: {os.path.abspath(os.path.join(yolo_dataset_folder, 'val'))}\n")
    f.write(f"test: {os.path.abspath(os.path.join(yolo_dataset_folder, 'test'))}\n")
    f.write("nc: 2\n")
    f.write("names: ['left ear', 'right ear']\n")

print(f"✅ Dataset split complete! `dataset.yaml` created at: {dataset_yaml_path}")
