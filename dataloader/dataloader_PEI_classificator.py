# ==============================================================================
# File: dataloader_PEI_classificator.py
# Description: DataLoader for PEI TIFF images and annotations with k-fold support.
# Author: @claudiacastrillon
# Modified: 02/07/2025
# ==============================================================================

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    def split_train_val_test_patients(annotations, test_ratio=0.1, seed=42):
        """Split patient-level stratified test set and return train/test patient ids."""
        all_patient_ids = list(annotations.keys())
        labels = []

        for pid in all_patient_ids:
            df = annotations[pid]
            label_counts = df['Annotation'].value_counts()
            if label_counts.empty:
                labels.append(0)  # fallback
            else:
                labels.append(label_counts.idxmax())  # majority label

        train_ids, test_ids = train_test_split(
            all_patient_ids,
            test_size=test_ratio,
            stratify=labels,
            random_state=seed
        )

        return train_ids, test_ids

    @staticmethod
    def get_kfold_dataloaders(images_folder, annotations, patient_ids, k=5, batch_size=8, transform=None, seed=42):
        processed_images_folder = images_folder

        def get_image_files_and_labels(patients):
            image_files, image_labels = [], []
            for pid in patients:
                folder = os.path.join(processed_images_folder, pid)
                if not os.path.exists(folder):
                    continue
                valid_files = [
                    f for f in os.listdir(folder)
                    if f.endswith(".tif") and "(1)" not in f and "(2)" not in f
                ]
                for file in valid_files:
                    full_path = os.path.join(pid, file)
                    row = annotations[pid][annotations[pid]["File Name"].str.strip() == file.strip()]
                    if not row.empty:
                        label = row["Annotation"].values[0]
                        image_files.append(full_path)
                        image_labels.append(label)
            return np.array(image_files), np.array(image_labels)

        all_files, all_labels = get_image_files_and_labels(patient_ids)

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_files, all_labels)):
            train_files = all_files[train_idx]
            val_files = all_files[val_idx]

            train_dataset = ClassificationDataset(train_files, processed_images_folder, annotations, transform)
            val_dataset = ClassificationDataset(val_files, processed_images_folder, annotations, transform)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            folds.append((fold_idx, train_loader, val_loader))

        return folds

    @staticmethod
    def get_test_dataloader(images_folder, annotations, patient_ids, batch_size=8, transform=None):
        processed_images_folder = images_folder
        test_files = []

        for pid in patient_ids:
            folder = os.path.join(processed_images_folder, pid)
            if not os.path.exists(folder):
                continue
            valid_files = [
                f for f in os.listdir(folder)
                if f.endswith(".tif") and "(1)" not in f and "(2)" not in f
            ]
            for file in valid_files:
                test_files.append(os.path.join(pid, file))

        test_dataset = ClassificationDataset(test_files, processed_images_folder, annotations, transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader


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

