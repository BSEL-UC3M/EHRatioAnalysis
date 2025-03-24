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
base_dataset_folder = "/Users/claudiacastrillonalvarez/Desktop/IMAGES_YOLO/MRC_YOLO/MRC"
yolo_annotations_folder="/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis"
yolo_dataset_folder = os.path.join(yolo_annotations_folder, "YOLO_annotations")  # Static YOLO dataset folder

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
# csv_path="/Users/claudiacastrillonalvarez/Desktop/IMAGES_YOLO_toydataset/MRC_YOLO_toydataset/MRC_coordinates_toydataset.csv"
csv_path = os.path.join(base_dataset_folder, csv_files[0])  # Assuming only one CSV file
print(f"🔹 Processing CSV file: {csv_path}")

# Read the CSV file
df = pd.read_csv(csv_path, skiprows=1, names=["filename", "x_left", "y_left", "x_right", "y_right"])

# Extract all patient IDs
def extract_patient_number(filename):
    if isinstance(filename, float) or pd.isna(filename):  # Handle NaN and non-string cases
        return None
    filename = str(filename)  # Ensure it is a string
    match = re.search(r"MRC_(\d+)_", filename)  # Looks for "MRC_<patient_num>_"
    return match.group(1) if match else None

patients = sorted(set(df["filename"].astype(str).apply(extract_patient_number).dropna()))

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
    #annotations_folder = os.path.join(base_dataset_folder, patient_folder, "yolo_annotations")
    #source_annotation_path = os.path.join(annotations_folder, image_filename.replace(".tif", ".txt"))
    #dest_annotation_path = os.path.join(yolo_dataset_folder, split_folder, "yolo_annotations", image_filename.replace(".tif", ".txt"))

    #if os.path.exists(source_annotation_path):
    #    shutil.copy(source_annotation_path, dest_annotation_path)
    #else:
    #    print(f"❌ Warning: Annotation {source_annotation_path} not found, skipping...")
    # Generate YOLO annotations from CSV data
    # Create a patient-specific folder inside annotations
    annotation_output_folder = os.path.join(yolo_dataset_folder, split_folder)
    os.makedirs(annotation_output_folder, exist_ok=True)

    # Save annotation file inside the patient-specific folder
    annotation_path = os.path.join(annotation_output_folder, image_filename.replace(".tif", ".txt"))


    # Get bounding box coordinates
    x_left, y_left, x_right, y_right = row["x_left"], row["y_left"], row["x_right"], row["y_right"]

    # Ensure bounding boxes are valid
    if pd.isna(x_left) or pd.isna(y_left) or pd.isna(x_right) or pd.isna(y_right):
        print(f"⚠️ Skipping {image_filename} due to missing bounding box data.")
        continue

    # Convert bounding box to YOLO format
    image = cv2.imread(source_image_path)  # Load image to get dimensions
    if image is None:
        print(f"❌ Warning: Could not load {source_image_path}, skipping annotation generation.")
        continue
    img_height, img_width = image.shape[:2]

    with open(annotation_path, "w") as f:
        # Define fixed bounding box size (96x96) in the same scale as the image
        bbox_size = 96

        # Ensure bounding box centers remain relative to the original image
        x_center_left = x_left
        y_center_left = y_left

        x_center_right = x_right
        y_center_right = y_right

        # Ensure bounding boxes fit within the image by adjusting their position
        x_min_left = max(0, x_center_left - bbox_size // 2)
        y_min_left = max(0, y_center_left - bbox_size // 2)
        x_max_left = min(img_width, x_min_left + bbox_size)
        y_max_left = min(img_height, y_min_left + bbox_size)

        x_min_right = max(0, x_center_right - bbox_size // 2)
        y_min_right = max(0, y_center_right - bbox_size // 2)
        x_max_right = min(img_width, x_min_right + bbox_size)
        y_max_right = min(img_height, y_min_right + bbox_size)

        # Recalculate final center points based on adjusted min/max values
        x_center_left = (x_min_left + x_max_left) / 2
        y_center_left = (y_min_left + y_max_left) / 2
        x_center_right = (x_min_right + x_max_right) / 2
        y_center_right = (y_min_right + y_max_right) / 2

        # Compute width and height (should always be 96x96 unless cropped at image edges)
        width_left = x_max_left - x_min_left
        height_left = y_max_left - y_min_left
        width_right = x_max_right - x_min_right
        height_right = y_max_right - y_min_right

        # Normalize the coordinates (YOLO format expects values between 0 and 1)
        x_center_left /= img_width
        y_center_left /= img_height
        width_left /= img_width
        height_left /= img_height

        x_center_right /= img_width
        y_center_right /= img_height
        width_right /= img_width
        height_right /= img_height

        # Write left ear annotation (Class 0)
        f.write(f"0 {x_center_left:.6f} {y_center_left:.6f} {width_left:.6f} {height_left:.6f}\n")

        # Write right ear annotation (Class 1)
        f.write(f"1 {x_center_right:.6f} {y_center_right:.6f} {width_right:.6f} {height_right:.6f}\n")

    print(f"✅ Annotation saved: {annotation_path}")



# ✅ Generate `dataset.yaml`
dataset_yaml_path = os.path.join(yolo_dataset_folder, "dataset.yaml")
with open(dataset_yaml_path, "w") as f:
    f.write(f"train: {os.path.abspath(os.path.join(yolo_dataset_folder, 'train'))}\n")
    f.write(f"val: {os.path.abspath(os.path.join(yolo_dataset_folder, 'val'))}\n")
    f.write(f"test: {os.path.abspath(os.path.join(yolo_dataset_folder, 'test'))}\n")
    f.write("nc: 2\n")  # Number of classes (left ear, right ear)
    f.write("names: ['left ear', 'right ear']\n")

print(f"✅ Dataset split complete! `dataset.yaml` created at: {dataset_yaml_path}")

