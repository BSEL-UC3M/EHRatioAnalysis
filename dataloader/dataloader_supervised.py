# ==============================================================================
# Description: File to load and split images and labels by patient IDs
# Author: Gloria del Rocío Delicado Correa
# Maintainer: Caterina Fuster-Barceló
# Creation date: 29/08/2023
# ==============================================================================

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class PatientDataset(Dataset):
    def __init__(self, image_files, label_files, images_folder, labels_folder, transform=None):
        """
        Args:
            image_files (list): List of image filenames.
            label_files (list): List of label filenames.
            images_folder (str): Path to the folder containing image files.
            labels_folder (str): Path to the folder containing label files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_files = image_files
        self.label_files = label_files
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_folder, self.image_files[idx])
        label_path = os.path.join(self.labels_folder, self.label_files[idx])

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = np.expand_dims(label, axis=-1)  # Add channel dimension

        if self.transform:
            image, label = self.transform(image, label)

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Change to (C, H, W)
        label = torch.from_numpy(label).permute(2, 0, 1).float()  # Change to (C, H, W)

        return image, label

class DataLoaderByPatient:
    @staticmethod
    def get_patient_id(filename):
        """
        Extracts the patient ID from a given filename.
        """
        return filename.split("_")[0]

    @staticmethod
    def train_test_split_bypatient(images_folder, labels_folder, test_size=0.2, batch_size=8, shuffle=True, transform=None):
        """
        Splits the data into training and testing sets by patient ID and returns PyTorch DataLoaders.
        """
        # Get the list of all image and label files in the folders
        image_files = os.listdir(images_folder)
        label_files = os.listdir(labels_folder)

        # Sort both lists to ensure they are in the same order
        image_files.sort()
        label_files.sort()

        # Shuffle the file lists in the same order
        combined = list(zip(image_files, label_files))
        random.shuffle(combined)
        shuffled_image_files, shuffled_label_files = zip(*combined)

        # Group image and label files by patient ID
        image_groups = {}
        for image_file in shuffled_image_files:
            patient_id = DataLoaderByPatient.get_patient_id(image_file)
            if patient_id not in image_groups:
                image_groups[patient_id] = []
            image_groups[patient_id].append(image_file)
        label_groups = {}
        for label_file in shuffled_label_files:
            patient_id = DataLoaderByPatient.get_patient_id(label_file)
            if patient_id not in label_groups:
                label_groups[patient_id] = []
            label_groups[patient_id].append(label_file)

        # Shuffle the groups of patients
        patients = list(image_groups.keys())
        random.shuffle(patients)

        # Split the patients into train and test sets
        train_patients, test_patients = train_test_split(patients, test_size=test_size)

        # Create the final lists of image and label files for train and test sets
        train_image_files = [file for patient in train_patients for file in image_groups[patient]]
        test_image_files = [file for patient in test_patients for file in image_groups[patient]]

        train_label_files = [file for patient in train_patients for file in label_groups[patient]]
        test_label_files = [file for patient in test_patients for file in label_groups[patient]]

        # Create PyTorch datasets
        train_dataset = PatientDataset(train_image_files, train_label_files, images_folder, labels_folder, transform)
        test_dataset = PatientDataset(test_image_files, test_label_files, images_folder, labels_folder, transform)

        # Create PyTorch DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader


# Example usage: ========================================
# images_folder = '/path/to/images'
# labels_folder = '/path/to/labels'
# data_loader = DataLoaderByPatient()
# train_loader, test_loader = data_loader.train_test_split_bypatient(images_folder, labels_folder, test_size=0.2, batch_size=8)
