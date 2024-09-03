# ==============================================================================
# Description: File to load and split images and labels by patient IDs with Pytorch
# Author: Gloria del Rocío Delicado Correa
# Maintainer: Caterina Fuster-Barceló
# Creation date: 29/08/2023
# ==============================================================================

import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
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

        # Make images between 0 and 1
        image = np.clip(image, 0, 1)
        label = np.clip(label, 0, 1)

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Change to (C, H, W)
        label = torch.from_numpy(label).permute(2, 0, 1).float()  # Change to (C, H, W)

        assert image.min() >= 0 and image.max() <= 1, "WARNING: Input values should be between 0 and 1"
        return image, label

class DataLoaderByPatient:
    @staticmethod
    def get_patient_id(filename):
        """
        Extracts the patient ID from a given filename.
        """
        return filename.split("_")[0]

    @staticmethod
    def train_val_test_split_bypatient(images_folder, labels_folder, splits=(0.7, 0.15, 0.15), batch_size=8, shuffle=True, transform=None):
        """
        Splits the data into training, validation, and testing sets by patient ID and returns PyTorch DataLoaders.
        
        Parameters:
        - images_folder: str
            Path to the folder containing image files.
        - labels_folder: str
            Path to the folder containing label files.
        - splits: tuple of three floats
            Proportions for train, validation, and test splits, respectively. They must sum to 1.0.
        - batch_size: int, optional
            How many samples per batch to load.
        - shuffle: bool, optional
            Whether to shuffle the data after splitting.
        - transform: callable, optional
            A function/transform to apply to the data.
        
        Returns:
        - train_loader: DataLoader
            DataLoader for the training set.
        - val_loader: DataLoader
            DataLoader for the validation set.
        - test_loader: DataLoader
            DataLoader for the testing set.
        """
        # Ensure that splits sum to 1.0
        assert sum(splits) == 1.0, "Splits must sum to 1.0"

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

        # Calculate the number of patients for each split
        num_patients = len(patients)
        num_train = int(splits[0] * num_patients)
        num_val = int(splits[1] * num_patients)
        
        train_patients = patients[:num_train]
        val_patients = patients[num_train:num_train + num_val]
        test_patients = patients[num_train + num_val:]

        # Assert that there are no patients in more than one split
        assert len(set(train_patients) & set(val_patients)) == 0, "WARNING: Patients in both train and val sets detected"
        assert len(set(train_patients) & set(test_patients)) == 0, "WARNING: Patients in both train and test sets detected"
        assert len(set(val_patients) & set(test_patients)) == 0, "WARNING: Patients in both val and test sets detected"

        # Create the final lists of image and label files for train, val, and test sets
        train_image_files = [file for patient in train_patients for file in image_groups[patient]]
        val_image_files = [file for patient in val_patients for file in image_groups[patient]]
        test_image_files = [file for patient in test_patients for file in image_groups[patient]]

        train_label_files = [file for patient in train_patients for file in label_groups[patient]]
        val_label_files = [file for patient in val_patients for file in label_groups[patient]]
        test_label_files = [file for patient in test_patients for file in label_groups[patient]]

        # Create PyTorch datasets
        train_dataset = PatientDataset(train_image_files, train_label_files, images_folder, labels_folder, transform)
        val_dataset = PatientDataset(val_image_files, val_label_files, images_folder, labels_folder, transform)
        test_dataset = PatientDataset(test_image_files, test_label_files, images_folder, labels_folder, transform)

        # Create PyTorch DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


# Example usage: ========================================
# images_folder = '/path/to/images'
# labels_folder = '/path/to/labels'
# data_loader = DataLoaderByPatient()
# train_loader, val_loader, test_loader = data_loader.train_val_test_split_bypatient(
#    images_folder='path/to/images',
#    labels_folder='path/to/labels',
#    splits=(0.7, 0.15, 0.15),
#    batch_size=8,
#    shuffle=True,
#    transform=None
#)
# For Windows folder
# images_folder = "..\\toydataset\\toydataset\\MRC\\images"

# Show two images from data_loader 
# for i, (image, label) in enumerate(train_loader):
#     if i == 2:
#         break
#     print(image.shape, label.shape)
#     print(image.dtype, label.dtype)
#     print(image.min(), image.max())
#     print(label.min(), label.max())
#     image = image.permute(0, 2, 3, 1).numpy()
#     label = label.permute(0, 2, 3, 1).numpy()
#     plt.imshow(image[0])
