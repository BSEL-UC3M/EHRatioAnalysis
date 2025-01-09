# ==============================================================================
# File: main_classificator.py
# Description: Main script for training and evaluating the classification model.
# Author: @cfusterbarcelo
# Created: 09/01/2025
# ==============================================================================

import os
import sys
import torch
from dataloader.dataloader_MRC_classificator import ClassificationDataLoader
from models.classificator import SimpleCNN
from trainers.classificator.toy_classificator import train_model, evaluate_model

# Add the current directory to Python's module search path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ==============================================================================

# Configuration Parameters
SAVE_RESULTS = False  # Toggle to save results
NUM_EPOCHS = 5  # Number of training epochs
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 8  # Batch size for training
DATA_SPLITS = (0.34, 0.33, 0.33)  # Train, validation, test splits
IMAGES_FOLDER = "toydataset/classification/"  # Path to the folder containing images

# Full dataset for training (uncomment when needed)
# IMAGES_FOLDER = "D:/Data/VolumetricHydrops/images/MRC"

# ==============================================================================

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load annotations
annotations = ClassificationDataLoader.load_annotations(IMAGES_FOLDER)

# Create train, val, and test DataLoaders
train_loader, val_loader, test_loader = ClassificationDataLoader.train_val_test_split(
    images_folder=IMAGES_FOLDER,
    annotations=annotations,
    splits=DATA_SPLITS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    transform=None
)

# ==============================================================================

# Print information about the dataset
print("DataLoader Information:")
print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of validation samples: {len(val_loader.dataset)}")
print(f"Number of test samples: {len(test_loader.dataset)}")

# Extract unique class labels from annotations
num_classes = len(set(
    annotation 
    for patient_data in annotations.values() 
    for annotation in patient_data['Annotation']
))
# Initialize the model
model = SimpleCNN(num_classes=num_classes).to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
print("Starting training...")
trained_model = train_model(model, train_loader, criterion, optimizer, device, NUM_EPOCHS)

# Evaluate the model
print("Evaluating model...")
evaluate_model(trained_model, test_loader, device)