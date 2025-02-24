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
# ==============================================================================


import os
import pandas as pd
import cv2
import re

# Define base dataset directory
base_dataset_folder = "./toydataset/object_detection/"

# Define bounding box size
bbox_size = 56  # Fixed bounding box size for U-Net compatibility

# Load all CSV files (only one file is expected now)
csv_files = [f for f in os.listdir(base_dataset_folder) if f.endswith('.csv')]

if not csv_files:
    print("❌ No CSV files found in the dataset folder!")
    exit()

csv_path = os.path.join(base_dataset_folder, csv_files[0])  # Assuming only one CSV file
print(f"🔹 Processing CSV file: {csv_path}")

# Read the CSV file
df = pd.read_csv(csv_path, skiprows=1, names=["filename", "x_left", "y_left", "x_right", "y_right"])

# Function to extract patient number from filename
def extract_patient_number(filename):
    match = re.search(r"MRC_(\d+)_", filename)  # Looks for "MRC_<patient_num>_"
    if match:
        return match.group(1)  # Extract patient number
    return None

# Process each row in the CSV
for _, row in df.iterrows():
    image_filename = row["filename"]
    patient_number = extract_patient_number(image_filename)

    if patient_number is None:
        print(f"⚠️ Could not determine patient number for {image_filename}, skipping...")
        continue

    # Construct the patient folder name
    patient_folder = f"PACIENTE {patient_number} MRC TIFF"
    image_path = os.path.join(base_dataset_folder, patient_folder, image_filename)

    # Ensure correct path format
    image_path = image_path.replace("\\", "/")

    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"❌ Warning: Image {image_path} not found, skipping...")
        continue

    # Load image to get dimensions
    img = cv2.imread(image_path)
    img_height, img_width, _ = img.shape

    # Normalize bounding box parameters
    x_left_norm = row["x_left"] / img_width
    y_left_norm = row["y_left"] / img_height
    x_right_norm = row["x_right"] / img_width
    y_right_norm = row["y_right"] / img_height
    width_norm = bbox_size / img_width
    height_norm = bbox_size / img_height

    # Create YOLO annotation lines
    yolo_lines = [
        f"0 {x_left_norm:.6f} {y_left_norm:.6f} {width_norm:.6f} {height_norm:.6f}",  # Left ear (Class 0)
        f"1 {x_right_norm:.6f} {y_right_norm:.6f} {width_norm:.6f} {height_norm:.6f}"  # Right ear (Class 1)
    ]

    # Save annotation file in the corresponding patient folder
    annotations_folder = os.path.join(base_dataset_folder, patient_folder, "yolo_annotations")
    os.makedirs(annotations_folder, exist_ok=True)

    yolo_filename = os.path.join(annotations_folder, image_filename.replace(".tif", ".txt"))
    with open(yolo_filename, "w") as f:
        f.write("\n".join(yolo_lines))

print("✅ YOLO annotations generated successfully! Run this script only when CSVs change.")
