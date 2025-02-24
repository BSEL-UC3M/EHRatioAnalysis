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
    Utility class to load and split images and annotations for YOLO object detection.
    """

    @staticmethod
    def train_val_test_split(images_folder, annotations_folder, splits=(0.7, 0.15, 0.15), batch_size=8, shuffle=True, seed=None, transform=None):
        """
        Splits the dataset into training, validation, and test sets.
        """
        assert sum(splits) == 1.0, "Splits must sum to 1.0."

        patient_folders = [f for f in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, f))]
        if seed is not None:
            random.seed(seed)
        if shuffle:
            random.shuffle(patient_folders)

        num_patients = len(patient_folders)
        num_train = int(splits[0] * num_patients)
        num_val = int(splits[1] * num_patients)

        train_patients = patient_folders[:num_train]
        val_patients = patient_folders[num_train:num_train + num_val]
        test_patients = patient_folders[num_train + num_val:]

        def get_image_files(patients):
            image_files = []
            for patient in patients:
                patient_folder = os.path.join(images_folder, patient)
                if os.path.exists(patient_folder):
                    image_files.extend([os.path.join(patient, file) for file in os.listdir(patient_folder) if file.endswith('.tif')])
            return image_files

        return DataLoader(YoloObjectDetectorDataset(get_image_files(train_patients), images_folder, annotations_folder, transform), batch_size=batch_size, shuffle=True), \
               DataLoader(YoloObjectDetectorDataset(get_image_files(val_patients), images_folder, annotations_folder, transform), batch_size=batch_size, shuffle=False), \
               DataLoader(YoloObjectDetectorDataset(get_image_files(test_patients), images_folder, annotations_folder, transform), batch_size=batch_size, shuffle=False)
