# ==============================================================================
# STEP 1: Load CSV and Remove NaN Rows
# ==============================================================================

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

csv_file = "/Users/claudiacastrillonalvarez/Desktop/IMAGES_YOLO_toydataset/MRC_YOLO_toydataset/MRC_coordinates_toydataset.csv"

# Load CSV and remove empty rows
annotations = pd.read_csv(csv_file, header=None)
annotations.dropna(inplace=True)  # Remove NaN rows

# Assign column names dynamically
expected_columns = ["filename", "left_x", "left_y", "right_x", "right_y"]
if annotations.shape[1] == len(expected_columns):
    annotations.columns = expected_columns
else:
    print(f"⚠️ Unexpected number of columns: {annotations.shape[1]}")
    print(annotations.head(10))
    raise ValueError("Check the CSV format!")

annotations["filename"] = annotations["filename"].astype(str).str.strip()

print("✅ Step 1: CSV loaded and cleaned.")

# ==============================================================================
# STEP 2: Process Each Patient Folder
# ==============================================================================

# Root directory where patient folders are stored
image_root_dir = Path("/Users/claudiacastrillonalvarez/Desktop/IMAGES_YOLO_toydataset/MRC_YOLO_toydataset/MRC")
output_root_dir = Path("results/object_detector/MRC_object_detector/debugger")
output_root_dir.mkdir(parents=True, exist_ok=True)

# Loop through each patient folder
for patient_folder in image_root_dir.glob("PACIENTE* MRC TIFF"):
    patient_id = patient_folder.stem.replace(" ", "_")
    patient_output_dir = output_root_dir / patient_id
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Processing {patient_id}...")

    # Filter annotations for this patient
    patient_images = list(patient_folder.glob("*.tif"))
    patient_filenames = [img.name for img in patient_images]
    patient_annotations = annotations[annotations["filename"].isin(patient_filenames)]

    if patient_annotations.empty:
        print(f"⚠️ No annotations found for {patient_id}. Skipping...")
        continue

    # Compute the average centroids for left and right ears
    x_center_avg = (patient_annotations["left_x"] + patient_annotations["right_x"]) / 2
    y_center_avg = (patient_annotations["left_y"] + patient_annotations["right_y"]) / 2
    avg_left_centroid = [patient_annotations["left_x"].mean(), patient_annotations["left_y"].mean()]
    avg_right_centroid = [patient_annotations["right_x"].mean(), patient_annotations["right_y"].mean()]

    print(f"📍 Left Ear Avg: {avg_left_centroid}, Right Ear Avg: {avg_right_centroid}")

    # ==============================================================================
    # STEP 3: Load One Image and Crop
    # ==============================================================================

    # Use the first available image in the patient folder
    if not patient_images:
        print(f"⚠️ No images found for {patient_id}. Skipping...")
        continue

    image_path = patient_images[0]  # Take the first image in the folder
    try:
        pil_image = Image.open(image_path).convert("RGB")
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape
        print(f"✅ Loaded image {image_path.name} - Shape: {image.shape}")
    except Exception as e:
        print(f"❌ Failed to load image {image_path}: {e}")
        continue

    # ==============================================================================
    # STEP 4: Crop and Save Left & Right Ear Images
    # ==============================================================================

    crop_size = 96
    centroids = {"left": avg_left_centroid, "right": avg_right_centroid}

    for ear_side, (x_center, y_center) in centroids.items():
        x1 = max(0, int(x_center - crop_size / 2))
        y1 = max(0, int(y_center - crop_size / 2))
        x2 = min(w, int(x_center + crop_size / 2))
        y2 = min(h, int(y_center + crop_size / 2))

        ear_crop = image[y1:y2, x1:x2]

        if ear_crop.size == 0:
            print(f"⚠️ Empty crop for {ear_side} ear in {patient_id}. Skipping save.")
            continue

        ear_crop_resized = cv2.resize(ear_crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        output_filename = patient_output_dir / f"{patient_id}_{ear_side}_ear.jpg"
        cv2.imwrite(str(output_filename), ear_crop_resized)
        print(f"✅ Saved: {output_filename}")

    print(f"✅ Completed processing for {patient_id}.\n")

print("🎉 All patients processed successfully!")
