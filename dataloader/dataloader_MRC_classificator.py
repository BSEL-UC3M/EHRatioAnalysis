# ==============================================================================
# File: dataloader_classificator.py
# Description: DataLoader for MRC TIFF images and annotations for classification tasks.
# Author: @cfusterbarcelo
# Creation Date: 05/01/2025
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
            print(f"Sheet names found in the file: {sheet_names}")

            # Read and store all sheets while inside the `with` block
            all_patient_data = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in sheet_names}
        return all_patient_data

    @staticmethod
    def train_val_test_split(images_folder, annotations, splits=(0.7, 0.15, 0.15), batch_size=8, shuffle=True, transform=None):
    
        """
        Splits the data into training, validation, and testing sets by patient ID.

        Parameters:
        - images_folder: Path to the folder containing patient folders with images.
        - annotations: Dictionary containing patient folder names as keys and their corresponding
                    labels as values (from `load_annotations`).
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

        # Set seed for reproducibility
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        # Get the list of patient folders
        patient_folders = list(annotations.keys())

        # Shuffle patients if required
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

            image_files=[]
            for patient in patients:
                patient_folder = os.path.join(images_folder, patient)
                if not os.path.exists(patient_folder):
                    print(f"WARNING: Folder {patient_folder} not found!")
                    continue
        
                patient_images = [
                    os.path.join(patient, file)
                    for file in os.listdir(patient_folder)
                    if file.endswith('.tif') and "(1)" not in file and "(2)" not in file  # Ignore duplicates
                ]
        
                if len(patient_images) == 0:
                    print(f"WARNING: No valid .tif images found in {patient_folder}")
        
                image_files.extend(patient_images)
    
            return image_files

        # Get image files for each split
        train_files = get_image_files(train_patients)
        val_files = get_image_files(val_patients)
        test_files = get_image_files(test_patients)

        # Ensure no patient overlap
        assert len(set(train_files) & set(val_files)) == 0, "Train and Val sets overlap!"
        assert len(set(train_files) & set(test_files)) == 0, "Train and Test sets overlap!"
        assert len(set(val_files) & set(test_files)) == 0, "Val and Test sets overlap!"

        # Create label dictionaries for the splits
        train_labels = {patient: annotations[patient] for patient in train_patients}
        val_labels = {patient: annotations[patient] for patient in val_patients}
        test_labels = {patient: annotations[patient] for patient in test_patients}

        # Create datasets
        train_dataset = ClassificationDataset(train_files, images_folder, train_labels, transform)
        val_dataset = ClassificationDataset(val_files, images_folder, val_labels, transform)
        test_dataset = ClassificationDataset(test_files, images_folder, test_labels, transform)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


