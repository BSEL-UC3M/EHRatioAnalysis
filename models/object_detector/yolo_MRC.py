# ==============================================================================
# File: models/object_detector.py
# Description: Defines the YOLO object detection model.
# Author: @claudiacastrillon
# Creation Date: 24/02/2025
# ==============================================================================

import torch
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

def save_detections(image, detections, output_dir, filename):
    """
    Save detection results as annotated images.
    """
    output_path = output_dir / filename

    if detections is None or len(detections) == 0:
        print(f"⚠️ No detections found for {filename}. Skipping save.")
        return

    for detection in detections:
        if len(detection) < 5:
            print(f"⚠️ Skipping invalid detection: {detection}")
            continue

        x_center, y_center, w, h = detection[1:]
        x1 = int((x_center - w / 2) * image.shape[1])
        y1 = int((y_center - h / 2) * image.shape[0])
        x2 = int((x_center + w / 2) * image.shape[1])
        y2 = int((y_center + h / 2) * image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite(str(output_path), image)


class YOLOv5:
    """
    Wrapper class for YOLOv5 model.
    """
    def __init__(self, model_name="yolov5s", pretrained=True):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = YOLO(f"{model_name}.pt") if pretrained else YOLO()
        self.model.to(self.device)

    def detect_and_save(self, image_root_dir, csv_file, output_dir, save_results=True):
        """
        Perform object detection on all MRC images in patient folders and save results.
        """
        image_root_dir = Path(image_root_dir)
        output_dir = Path(output_dir)
        if save_results:
            output_dir.mkdir(parents=True, exist_ok=True)

        annotations = pd.read_csv(csv_file, header=None)
        annotations.columns = ["filename", "left_x", "left_y", "right_x", "right_y"]
        annotations.dropna(inplace=True)
        annotations['filename'] = annotations['filename'].astype(str).str.strip()

        for patient_folder in image_root_dir.glob("PACIENTE* MRC TIFF"):
            for image_path in patient_folder.glob("*.tif"):
                print(f"🖼️ Processing image: {image_path.name}")
                image = cv2.imread(str(image_path))
                results = self.model.detect(image, save=save_results, conf=0.25)
                if save_results:
                    save_detections(image, results, output_dir, image_path.name)