# ==============================================================================
# File: dataloader_object_detector.py
# Description: DataLoader for MRC TIFF images and YOLO object detection annotations.
# Author: @claudiacastrillon
# Creation Date: 05/01/2025
# Last Update: 25/02/2025
# ==============================================================================

import os
import torch
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class MRCObjectDetectorDataset(Dataset):
    """
    Dataset for loading MRC images and YOLO-compatible bounding box annotations from CSV files.
    """
    def __init__(self, image_root_dir, csv_file, transform=None):
        self.image_root_dir = Path(image_root_dir)
        self.transform = transform
        
        # Load CSV with centroid coordinates
        annotations = pd.read_csv(csv_file, header=None)
        annotations.columns = ["filename", "left_x", "left_y", "right_x", "right_y"]
        annotations.dropna(inplace=True)
        annotations["filename"] = annotations["filename"].astype(str).str.strip()
        self.annotations = annotations
        
        # Collect all image paths from patient subdirectories
        self.image_files = []
        for patient_folder in self.image_root_dir.glob("PACIENTE* MRC TIFF"):
            for image_file in patient_folder.glob("*.tif"):
                self.image_files.append(image_file)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        filename = image_path.name

        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} not found.")

        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Image file {image_path} could not be loaded.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32') / 255.0

        # Ensure filename exists in CSV
        row = self.annotations[self.annotations['filename'].str.lower() == filename.lower()]
        
        if row.empty:
            print(f"⚠️ Warning: No annotation found for {filename}")
            yolo_annotations = torch.zeros((0, 5), dtype=torch.float32)  # Empty annotations
        else:
            row = row.iloc[0]
            height, width = image.shape[:2]
            yolo_annotations = [
                [0, row['left_x'] / width, row['left_y'] / height, 0.1, 0.1],
                [0, row['right_x'] / width, row['right_y'] / height, 0.1, 0.1]
            ]
            yolo_annotations = torch.tensor(yolo_annotations, dtype=torch.float32)

        # Convert image to tensor format
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, yolo_annotations


class MRCObjectDetectionDataLoader:
    @staticmethod
    def load_from_existing_split(image_root_dir, csv_file, batch_size=8, shuffle=True, transform=None):
        dataset = MRCObjectDetectorDataset(image_root_dir, csv_file, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
