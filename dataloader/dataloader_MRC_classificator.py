# ==============================================================================
# File: dataloader_classificator.py
# Description: DataLoader for MRC TIFF images and annotations for classification tasks.
# Author: [Your Name]
# Creation Date: [Date]
# ==============================================================================

import os
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import cv2


class ClassificationDataset(Dataset):
    """
    PyTorch Dataset for loading MRC TIFF images and their corresponding labels
    from an xlsx file with each file containing one patient.
    """

    def __init__(self, image_files, image_folder, labels, transform=None):
        """
        Parameters:
        - image_files: List of image filenames.
        - image_folder: Path to the folder containing image files.
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
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)

        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image.astype('float32') / 255.0  # Normalize to [0, 1]

        # Get the label
        label = self.labels[image_file]

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
                labels_file= images_folder + "/" +file
                break
        if not labels_file:
            raise FileNotFoundError("No .xlsx file found in the dataset folder.")
        
        # Get the list of patient folders
        patient_folders = [folder for folder in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, folder))]

        all_patient_data = {}
        with pd.ExcelFile(labels_file) as xls:
            for patient_folder in patient_folders:
                try:
                    # Assuming the sheet name matches the patient folder name
                    sheet_data = pd.read_excel(xls, sheet_name=patient_folder)
                    all_patient_data[patient_folder] = sheet_data
                except ValueError as e:
                    print(f"WARNING: No sheet found for patient folder: {patient_folder}")
        # TODO: Extract the labels from the Excel file
        labels = {}
        return labels

    @staticmethod
    def train_val_test_split(images_folder, annotations_file, splits=(0.7, 0.15, 0.15), batch_size=8, shuffle=True, transform=None):
        """
        Splits the data into training, validation, and testing sets by patient ID.

        Parameters:
        - images_folder: Path to the folder containing patient folders with images.
        - annotations_file: Path to the Excel file containing annotations.
        - splits: Tuple indicating the train, validation, and test split ratios.
        - batch_size: Batch size for the DataLoader.
        - shuffle: Whether to shuffle the patients before splitting.
        - transform: Optional transformations to apply to the images.

        Returns:
        - train_loader: DataLoader for training data.
        - val_loader: DataLoader for validation data.
        - test_loader: DataLoader for testing data.
        """
        assert sum(splits) == 1.0, "Splits must sum to 1.0."

        # Load annotations
        labels = ClassificationDataLoader.load_annotations(annotations_file)

        # Get list of patient folders
        patient_folders = [folder for folder in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, folder))]

        # Shuffle patients
        if shuffle:
            random.shuffle(patient_folders)

        # Split patients into train, val, and test
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
                image_files.extend([os.path.join(patient, file) for file in os.listdir(patient_folder) if file.endswith('.tif')])
            return image_files

        # Get image files for each split
        train_files = get_image_files(train_patients)
        val_files = get_image_files(val_patients)
        test_files = get_image_files(test_patients)

        # Ensure no patient overlap
        assert len(set(train_files) & set(val_files)) == 0, "Train and Val sets overlap!"
        assert len(set(train_files) & set(test_files)) == 0, "Train and Test sets overlap!"
        assert len(set(val_files) & set(test_files)) == 0, "Val and Test sets overlap!"

        # Filter labels for the splits
        train_labels = {file: labels[os.path.basename(file)] for file in train_files}
        val_labels = {file: labels[os.path.basename(file)] for file in val_files}
        test_labels = {file: labels[os.path.basename(file)] for file in test_files}

        # Create datasets
        train_dataset = ClassificationDataset(train_files, images_folder, train_labels, transform)
        val_dataset = ClassificationDataset(val_files, images_folder, val_labels, transform)
        test_dataset = ClassificationDataset(test_files, images_folder, test_labels, transform)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

