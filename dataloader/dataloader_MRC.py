# ==============================================================================
# Description: File to load and split images and labels by patient IDs with Pytorch
# Author: @laurarodmu
# Creation date: 29/03/2025
# ==============================================================================

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import cv2
import torch

class New_PatientDataset(Dataset):
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = np.expand_dims(label, axis=-1)  

        if self.transform:
            image, label = self.transform(image, label)

        image = np.array(image)
        image_max = image.max()
        image_min = image.min()
        image = (image - image_min) / (image_max - image_min)

        label = np.clip(label, 0, 1)

        image = torch.from_numpy(image).permute(2, 0, 1).float()  
        label = torch.from_numpy(label).permute(2, 0, 1).float() 

        assert image.min() >= 0 and image.max() <= 1, "WARNING: Input values should be between 0 and 1"


        image_name = self.image_files[idx] 
        return image, label, image_name


class DataLoaderByPatientSpecific:
    @staticmethod
    def get_patient_id_new(filename):
        """
        Extracts the patient ID from a given filename (after 'PAC').
        """
        return filename.split("_")[0]

    @staticmethod
    def train_val_test_split_bypatient(images_folder, labels_folder, train_patients, val_patients, test_patients, batch_size=8, shuffle=True, transform=None):
        """
        Splits the data into training, validation, and testing sets by predefined patient IDs
        and returns PyTorch DataLoaders.
        
        Parameters:
        - images_folder: str
            Path to the folder containing image files.
        - labels_folder: str
            Path to the folder containing label files.
        - train_patients: list
            List of patient IDs for the training set.
        - val_patients: list
            List of patient IDs for the validation set.
        - test_patients: list
            List of patient IDs for the test set.
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

        image_files = os.listdir(images_folder)
        label_files = os.listdir(labels_folder)

        image_files.sort()
        label_files.sort()

        image_groups = {}
        for image_file in image_files:
            patient_id = DataLoaderByPatientSpecific.get_patient_id_new(image_file)
            if patient_id not in image_groups:
                image_groups[patient_id] = []
            image_groups[patient_id].append(image_file)
        label_groups = {}
        for label_file in label_files:
            patient_id = DataLoaderByPatientSpecific.get_patient_id_new(label_file)
            if patient_id not in label_groups:
                label_groups[patient_id] = []
            label_groups[patient_id].append(label_file)


        train_image_files = []
        train_label_files = []
        for patient in train_patients:
            if patient in image_groups:
                train_image_files.extend(image_groups[patient])
                train_label_files.extend(label_groups[patient])

        val_image_files = []
        val_label_files = []
        for patient in val_patients:
            if patient in image_groups:
                val_image_files.extend(image_groups[patient])
                val_label_files.extend(label_groups[patient])

        test_image_files = []
        test_label_files = []
        for patient in test_patients:
            if patient in image_groups:
                test_image_files.extend(image_groups[patient])
                test_label_files.extend(label_groups[patient])



        train_dataset = New_PatientDataset(train_image_files, train_label_files, images_folder, labels_folder, transform)
        val_dataset = New_PatientDataset(val_image_files, val_label_files, images_folder, labels_folder, transform)
        test_dataset = New_PatientDataset(test_image_files, test_label_files, images_folder, labels_folder, transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
