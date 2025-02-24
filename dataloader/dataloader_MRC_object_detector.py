# ==============================================================================
# File: dataloader_object_detector.py
# Description: DataLoader for MRC TIFF images and YOLO object detection annotations.
# Author: @cfusterbarcelo
# Creation Date: 05/01/2025
# ==============================================================================

import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import random


class YoloObjectDetectorDataset(Dataset):
    """
    PyTorch Dataset for loading MRC TIFF images and YOLO-compatible bounding box annotations.
    """

    def __init__(self, image_files, image_folder, annotations_folder, transform=None):
        """
        Parameters:
        - image_files: List of image filenames.
        - image_folder: Path to the folder containing patient folders with images.
        - annotations_folder: Path to the folder containing YOLO annotations.
        - transform: Optional transformations.
        """
        self.image_files = image_files
        self.image_folder = image_folder
        self.annotations_folder = annotations_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        patient_id = os.path.dirname(image_file).split(os.sep)[-1]
        filename = os.path.basename(image_file).strip()

        # Load image
        image_path = os.path.join(self.image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file {image_path} not found.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image.astype('float32') / 255.0  # Normalize

        # Load YOLO annotation
        annotation_path = os.path.join(self.annotations_folder, patient_id, "yolo_annotations", filename.replace(".tif", ".txt"))
        yolo_annotations = []

        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    yolo_annotations.append([class_id, x_center, y_center, width, height])
        else:
            print(f"❌ Warning: Annotation file {annotation_path} not found.")

        if self.transform:
            image = self.transform(image)

        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert to (C, H, W)
        yolo_annotations = torch.tensor(yolo_annotations, dtype=torch.float32) if yolo_annotations else torch.zeros((0, 5))

        return image, yolo_annotations


class ObjectDetectionDataLoader:
    """
    Utility class to load object detection datasets.
    """

    @staticmethod
    def load_from_existing_split(images_folder, annotations_folder, batch_size=8, shuffle=True, transform=None):
        """
        Loads pre-split YOLO train, val, and test datasets.

        Parameters:
        - images_folder: Path to the folder containing train/val/test images.
        - annotations_folder: Path to the folder containing YOLO annotations.
        - batch_size: Batch size for the DataLoader.
        - shuffle: Whether to shuffle the dataset.
        - transform: Optional image transformations.

        Returns:
        - train_loader, val_loader, test_loader
        """
        def get_image_files(split):
            split_folder = os.path.join(images_folder, split)
            image_files = [
                os.path.join(split, file)
                for file in os.listdir(split_folder) if file.endswith('.tif')
            ]
            return image_files

        train_files = get_image_files("train")
        val_files = get_image_files("val")
        test_files = get_image_files("test")

        train_dataset = YoloObjectDetectorDataset(train_files, images_folder, annotations_folder, transform=transform)
        val_dataset = YoloObjectDetectorDataset(val_files, images_folder, annotations_folder, transform=transform)
        test_dataset = YoloObjectDetectorDataset(test_files, images_folder, annotations_folder, transform=transform)

        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), \
               DataLoader(val_dataset, batch_size=batch_size, shuffle=False), \
               DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
