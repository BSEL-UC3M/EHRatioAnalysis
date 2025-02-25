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
# Last Update: 24/02/2025
# ==============================================================================

import os
import pandas as pd
import cv2
import re
import shutil
import random

# Define dataset directories
base_dataset_folder = "./toydataset/object_detection/"
yolo_dataset_folder = os.path.join(base_dataset_folder, "YOLO")  # Static YOLO dataset folder

# Define split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ✅ Ensure YOLO directory exists
os.makedirs(yolo_dataset_folder, exist_ok=True)

# ✅ Create train/val/test folders inside `YOLO/`
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(yolo_dataset_folder, split, "yolo_annotations"), exist_ok=True)

# Load all CSV files
csv_files = [f for f in os.listdir(base_dataset_folder) if f.endswith('.csv')]

if not csv_files:
    print("❌ No CSV files found in the dataset folder!")
    exit()

csv_path = os.path.join(base_dataset_folder, csv_files[0])  # Assuming only one CSV file
print(f"🔹 Processing CSV file: {csv_path}")

# Read the CSV file
df = pd.read_csv(csv_path, skiprows=1, names=["filename", "x_left", "y_left", "x_right", "y_right"])

# Extract all patient IDs
def extract_patient_number(filename):
    match = re.search(r"MRC_(\d+)_", filename)  # Looks for "MRC_<patient_num>_"
    return match.group(1) if match else None

patients = sorted(set(df["filename"].apply(extract_patient_number).dropna()))

# ✅ Randomly shuffle patients ONCE and store split
random.shuffle(patients)

num_train = int(len(patients) * TRAIN_RATIO)
num_val = int(len(patients) * VAL_RATIO)

train_patients = set(patients[:num_train])
val_patients = set(patients[num_train:num_train + num_val])
test_patients = set(patients[num_train + num_val:])

# ✅ Ensure no patient is in multiple splits
assert train_patients.isdisjoint(val_patients), "❌ ERROR: A patient appears in both train and validation!"
assert train_patients.isdisjoint(test_patients), "❌ ERROR: A patient appears in both train and test!"
assert val_patients.isdisjoint(test_patients), "❌ ERROR: A patient appears in both validation and test!"

print(f"✅ Patients successfully split:")
print(f"   - Train: {len(train_patients)} patients")
print(f"   - Val: {len(val_patients)} patients")
print(f"   - Test: {len(test_patients)} patients")

# ✅ Move images & annotations to YOLO/train, val, test folders
for _, row in df.iterrows():
    image_filename = row["filename"]
    patient_number = extract_patient_number(image_filename)

    if patient_number is None:
        print(f"⚠️ Could not determine patient number for {image_filename}, skipping...")
        continue

    # Determine the split based on the patient ID
    if patient_number in train_patients:
        split_folder = "train"
    elif patient_number in val_patients:
        split_folder = "val"
    else:
        split_folder = "test"

    # Construct source and destination paths
    patient_folder = f"PACIENTE {patient_number} MRC TIFF"
    source_image_path = os.path.join(base_dataset_folder, patient_folder, image_filename)
    dest_image_path = os.path.join(yolo_dataset_folder, split_folder, image_filename)

    # ✅ Copy image to the split folder
    if os.path.exists(source_image_path):
        shutil.copy(source_image_path, dest_image_path)
    else:
        print(f"❌ Warning: Image {source_image_path} not found, skipping...")

    # Save YOLO annotation
    annotations_folder = os.path.join(base_dataset_folder, patient_folder, "yolo_annotations")
    source_annotation_path = os.path.join(annotations_folder, image_filename.replace(".tif", ".txt"))
    dest_annotation_path = os.path.join(yolo_dataset_folder, split_folder, "yolo_annotations", image_filename.replace(".tif", ".txt"))

    if os.path.exists(source_annotation_path):
        shutil.copy(source_annotation_path, dest_annotation_path)
    else:
        print(f"❌ Warning: Annotation {source_annotation_path} not found, skipping...")

# ✅ Generate `dataset.yaml`
dataset_yaml_path = os.path.join(yolo_dataset_folder, "dataset.yaml")
with open(dataset_yaml_path, "w") as f:
    f.write(f"train: {os.path.abspath(os.path.join(yolo_dataset_folder, 'train'))}\n")
    f.write(f"val: {os.path.abspath(os.path.join(yolo_dataset_folder, 'val'))}\n")
    f.write(f"test: {os.path.abspath(os.path.join(yolo_dataset_folder, 'test'))}\n")
    f.write("nc: 2\n")  # Number of classes (left ear, right ear)
    f.write("names: ['left ear', 'right ear']\n")

print(f"✅ Dataset split complete! `dataset.yaml` created at: {dataset_yaml_path}")

