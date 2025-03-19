# ==============================================================================
# File: models/object_detector.py
# Description: Defines the YOLO object detection model.
# Author: @claudiacastrillon
# Creation Date: 24/02/2025
# ==============================================================================
import torch
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

def compute_average_centroids(annotations):
    """
    Compute the average centroids for left and right ears from the CSV annotations.
    """
    annotations["x_center"] = (annotations["left_x"] + annotations["right_x"]) / 2
    annotations["y_center"] = (annotations["left_y"] + annotations["right_y"]) / 2
    grouped = annotations.groupby("filename")[["x_center", "y_center"]].mean()
    return grouped.reset_index()

def save_cropped_ears(image, centroids, output_dir, filename, crop_size=96):
    """
    Save cropped left and right ear images based on computed centroids.
    """
    output_path = output_dir / filename.stem
    output_path.mkdir(parents=True, exist_ok=True)
    
    h, w, _ = image.shape
    
    for i, (x_center, y_center) in enumerate(centroids):
        x1 = max(0, int(x_center - crop_size / 2))
        y1 = max(0, int(y_center - crop_size / 2))
        x2 = min(w, int(x_center + crop_size / 2))
        y2 = min(h, int(y_center + crop_size / 2))
        
        ear_crop = image[y1:y2, x1:x2]
        
        if ear_crop.size == 0:
            print(f"⚠️ Empty crop for {filename}. Skipping save.")
            continue

        ear_crop_resized = cv2.resize(ear_crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        ear_side = "left" if i == 0 else "right"
        output_filename = output_path / f"{filename.stem}_{ear_side}_ear.jpg"
        cv2.imwrite(str(output_filename), ear_crop_resized)
        print(f"✅ Saved: {output_filename}")

class YOLOv5:
    """
    Wrapper class for YOLOv5 model.
    """
    def __init__(self, model_name="yolov5su", pretrained=True):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = YOLO(f"{model_name}.pt") if pretrained else YOLO()
        self.model.to(self.device)

    def detect_and_save(self, image_root_dir, csv_file, output_dir, save_results=True):
        """
        Perform cropping using computed average centroids and save left and right ear images.
        """
        image_root_dir = Path(image_root_dir)
        output_dir = Path(output_dir)
        if save_results:
            output_dir.mkdir(parents=True, exist_ok=True)

        annotations = pd.read_csv(csv_file, header=None)
        
        # Remove NaN rows before processing
        annotations.dropna(inplace=True)
        
        # Detect number of columns and assign names dynamically
        expected_columns = ["filename", "left_x", "left_y", "right_x", "right_y"]
        if annotations.shape[1] == len(expected_columns):
            annotations.columns = expected_columns
        else:
            print(f"⚠️ Unexpected number of columns: {annotations.shape[1]}")
            print("Detected first row:")
            print(annotations.head(1))
            return
        
        annotations['filename'] = annotations['filename'].astype(str).str.strip()
        
        avg_centroids = compute_average_centroids(annotations)
        
        for patient_folder in image_root_dir.glob("PACIENTE* MRC TIFF"):
            patient_id = patient_folder.stem.replace(" ", "_")
            patient_output_dir = output_dir / patient_id
            patient_output_dir.mkdir(parents=True, exist_ok=True)
            
            for image_path in patient_folder.glob("*.tif"):
                print(f"🖼️ Processing image: {image_path.name}")
                
                # Load image using PIL and convert to OpenCV format
                try:
                    pil_image = Image.open(image_path).convert("RGB")
                    image = np.array(pil_image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"❌ Failed to load image {image_path}: {e}")
                    continue
                
                # Find centroid for the current image
                matching_rows = avg_centroids[avg_centroids['filename'] == image_path.name]
                if matching_rows.empty:
                    print(f"⚠️ No centroid found for {image_path.name}. Skipping.")
                    continue
                
                centroids = matching_rows.iloc[:, 1:].values.tolist()
                save_cropped_ears(image, centroids, patient_output_dir, image_path)