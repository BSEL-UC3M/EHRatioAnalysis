# ==============================================================================
# File: dataloader_classificator.py
# Description: DataLoader for MRC TIFF images and annotations for classification tasks.
# Author: @claudiacastrillon
# Creation Date: 25/02/2025
# ==============================================================================

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import cv2
from PIL import Image
import numpy as np


# Add utils folder to path to import preprocessing script
import sys
sys.path.append("/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/utils")
from utils.preprocessing_all_images import preprocess_all_images
from pipeline_scripts.utils import preprocess_pei_image

class ClassificationDataset(Dataset):
    """
    PyTorch Dataset for loading MRC TIFF images and their corresponding labels
    from annotations extracted from an Excel file.
    """

    def __init__(self, image_files, image_folder, labels, transform=None):
        """
        Parameters:
        - image_files: List of image filenames.
        - image_folder: Path to the folder containing patient folders with images.
        - labels: Dictionary mapping image filenames to their class labels.
        - transform: Optional transformations to apply to the images.
        """
        self.image_files = image_files
        self.image_folder = image_folder
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get the file name and patient ID
        image_file = self.image_files[idx]
        patient_id = os.path.dirname(image_file).split(os.sep)[-1]
        filename = os.path.basename(image_file).strip()
        
        # Normalize the 'File Name' column in the DataFrame
        self.labels[patient_id]['File Name'] = self.labels[patient_id]['File Name'].str.strip()

        # Check if the filename exists in the 'File Name' column
        if filename not in self.labels[patient_id]['File Name'].values:
            raise KeyError(f"File '{filename}' not found in annotations for patient '{patient_id}'. "
                       f"Available files: {self.labels[patient_id]['File Name'].values}")

        # Get the corresponding label
        label_row = self.labels[patient_id][self.labels[patient_id]['File Name'] == filename]
        if label_row.empty:
            raise ValueError(f"No label found for file '{filename}' in patient '{patient_id}'.")
        label = label_row['Annotation'].values[0]

        # Load the image
        image_path = os.path.join(self.image_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image.astype('float32') / 255.0  # Normalize to [0, 1]

        if self.transform:
            image = self.transform(image)

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert to (C, H, W)
        label = torch.tensor(label, dtype=torch.long)  # Convert label to tensor

        return image, label

class ClassificationDataLoader:
    """
    Utility class to load and split images and annotations for classification tasks.
    """

    @staticmethod
    def load_annotations(images_folder):
        labels_file = None
        # Find the labels file in the dataset folder
        for file in os.listdir(images_folder):
            if file.endswith(".xlsx"):
                labels_file = os.path.join(images_folder, file)
                break
        if not labels_file:
            raise FileNotFoundError("No .xlsx file found in the dataset folder.")
        
        all_patient_data = {}

        # Open the Excel file
        with pd.ExcelFile(labels_file) as xls:
        # Get all sheet names dynamically
            sheet_names = xls.sheet_names
            print(f"🔍 Found sheet names: {sheet_names}")  # Debugging print

            # Ensure only relevant sheets are processed (ignore unrelated ones)
            relevant_sheets = [sheet for sheet in sheet_names if "PACIENTE" in sheet and "PEI TIFF" in sheet]

            if not relevant_sheets:
                raise ValueError("❌ No valid sheets found in the annotations file. Check sheet names!")

            # Read and store all relevant sheets
            all_patient_data = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in relevant_sheets}
        return all_patient_data


    @staticmethod
    def train_val_test_split(images_folder, annotations, train_patients, val_patients, test_patients, batch_size=8, transform=None):
        processed_images_folder = os.path.join(os.path.dirname(images_folder), "PEI_processed_data")

        if not os.path.exists(processed_images_folder) or len(os.listdir(processed_images_folder)) == 0:
            print("\n🔄 Preprocessing images...\n")
            preprocess_all_images(images_folder, processed_images_folder)
            print("✅ Preprocessing complete.")
        else:
            print("⚠️ Using existing preprocessed images.")

        def get_image_files(patient_folders):
            image_files = []
            for folder_name in patient_folders:
                patient_folder = os.path.join(processed_images_folder, folder_name)
                if not os.path.exists(patient_folder):
                    print(f"⚠️ WARNING: Folder {folder_name} not found in {processed_images_folder}")
                    continue
                patient_images = [
                    os.path.join(folder_name, file)
                    for file in os.listdir(patient_folder)
                    if file.endswith('.tif') and "(1)" not in file and "(2)" not in file
                ]
                image_files.extend(patient_images)
            return image_files

        train_files = get_image_files(train_patients)
        val_files = get_image_files(val_patients)
        test_files = get_image_files(test_patients)

        print(f"🔢 Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

        train_dataset = ClassificationDataset(train_files, processed_images_folder, annotations, transform)
        val_dataset = ClassificationDataset(val_files, processed_images_folder, annotations, transform)
        test_dataset = ClassificationDataset(test_files, processed_images_folder, annotations, transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

class InferenceDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_paths = [
            os.path.join(image_folder, fname)
            for fname in sorted(os.listdir(image_folder))
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)

        pil_image = Image.open(image_path).convert("F")  # "F" = 32-bit float grayscale
        image = np.array(pil_image, dtype=np.float32) / 255.0

        # --- Expand grayscale image to (H, W, 1) ---
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)  # shape: (H, W, 3)


        # --- Apply preprocessing ---
        image = preprocess_pei_image(image)

        # --- Convert to torch.Tensor with shape (C, H, W) ---
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return {
            "image": image,
            "filename": filename
        }


def load_inference_dataloader(image_folder, batch_size=16, transform=None):
    dataset = InferenceDataset(image_folder, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

