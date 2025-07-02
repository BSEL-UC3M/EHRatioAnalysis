# ==============================================================================
# File: dataloader_PEI_classificator.py
# Description: DataLoader for PEI TIFF images and annotations for classification tasks.
# Author: @claudiacastrillon
# Modified: 02/07/2025 by @ChatGPT for dynamic patient split and silent loading
# ==============================================================================

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from PIL import Image
import numpy as np

from utils.preprocessing_all_images import preprocess_all_images
from pipeline_scripts.utils import preprocess_pei_image


class ClassificationDataset(Dataset):
    def __init__(self, image_files, image_folder, labels, transform=None):
        self.image_files = image_files
        self.image_folder = image_folder
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        patient_id = os.path.dirname(image_file).split(os.sep)[-1]
        filename = os.path.basename(image_file).strip()

        self.labels[patient_id]['File Name'] = self.labels[patient_id]['File Name'].str.strip()

        if filename not in self.labels[patient_id]['File Name'].values:
            raise KeyError(f"File '{filename}' not found in annotations for patient '{patient_id}'.")

        label_row = self.labels[patient_id][self.labels[patient_id]['File Name'] == filename]
        if label_row.empty:
            raise ValueError(f"No label found for file '{filename}' in patient '{patient_id}'.")

        label = label_row['Annotation'].values[0]

        image_path = os.path.join(self.image_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.0

        if self.transform:
            image = self.transform(image)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.tensor(label, dtype=torch.long)

        return image, label


class ClassificationDataLoader:
    @staticmethod
    def load_annotations(images_folder):
        labels_file = None
        for file in os.listdir(images_folder):
            if file.endswith(".xlsx"):
                labels_file = os.path.join(images_folder, file)
                break
        if not labels_file:
            raise FileNotFoundError("No .xlsx file found in the dataset folder.")

        with pd.ExcelFile(labels_file) as xls:
            sheet_names = xls.sheet_names
            relevant_sheets = [sheet for sheet in sheet_names if "PACIENTE" in sheet and "PEI TIFF" in sheet]
            if not relevant_sheets:
                raise ValueError("❌ No valid sheets found in the annotations file. Check sheet names!")
            all_patient_data = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in relevant_sheets}
        return all_patient_data

    @staticmethod
    def train_val_test_split(images_folder, annotations, train_patients, val_patients, test_patients, batch_size=8, transform=None):
        processed_images_folder = images_folder  # Use the exact folder provided

        def get_image_files(patient_folders):
            image_files = []
            for folder_name in patient_folders:
                patient_folder = os.path.join(processed_images_folder, folder_name)
                if not os.path.exists(patient_folder):
                    continue  # skip silently
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

        pil_image = Image.open(image_path).convert("F")
        image = np.array(pil_image, dtype=np.float32) / 255.0

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        image = preprocess_pei_image(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return {"image": image, "filename": filename}


def load_inference_dataloader(image_folder, batch_size=16, transform=None):
    dataset = InferenceDataset(image_folder, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
