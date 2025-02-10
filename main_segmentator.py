# ==============================================================================
# File: main_segmentator.py
# Description: Main script for training and evaluating the segmentation model.
# Author: @cfusterbarcelo
# Creation Date: 03/09/2024
# ==============================================================================

import os
import torch
import torch.optim as optim
from datetime import datetime
from losses import losses
from dataloader.dataloader_MRC import DataLoaderByPatient
from trainers.segmentator.pretrained_trainers import train_model, evaluate_model
from models.segmentator import Segmentator

# ==============================================================================

# Configuration Parameters
SAVE_RESULTS = False  # Toggle to save results
NUM_EPOCHS = 5  # Number of training epochs
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 8  # Batch size for training
DATA_SPLITS = (0.34, 0.33, 0.33)  # Train, validation, test splits

# Dataset Paths
# Toy dataset for testing
#IMAGES_FOLDER = "toydataset/segmentation/MRC/images"
#LABELS_FOLDER = "toydataset/segmentation/MRC/labels"

# Verificar si estamos en Kaggle o en local
if os.path.exists('/kaggle/input'):
    # Si estamos en Kaggle, usar la ruta de Kaggle
    IMAGES_FOLDER = '/kaggle/input/cropped-dataset/CROPPED_DATASET/images/MRC'
    LABELS_FOLDER = '/kaggle/input/cropped-dataset/CROPPED_DATASET/labels/MRC'
else:
    # Si estamos en local, usar la ruta local
    IMAGES_FOLDER = 'toydataset/segmentation/PEI/images/'
    LABELS_FOLDER = "toydataset/segmentation/PEI/labels"

# Full dataset for training (uncomment when needed)
# IMAGES_FOLDER = "D:/Data/VolumetricHydrops/images/MRC"
# LABELS_FOLDER = "D:/Data/VolumetricHydrops/labels/MRC"

# ================================================================================

# Initialize the segmentation model
segmentator = Segmentator()

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentator = segmentator.to(device)

# Define the loss function (combined BCE and Dice loss)
criterion = losses.BCE_and_Dice_loss(
    bce_kwargs={},  # Default settings for BCELoss
    dice_class=losses.SimpleDiceLoss,  # Simple Dice loss class
    weight_ce=1,  # Weight for BCE loss
    weight_dice=1  # Weight for Dice loss
)

# Define the optimizer (Adam optimizer)
optimizer = optim.Adam(segmentator.parameters(), lr=LEARNING_RATE)

# ==============================================================================

# Create results directory if needed
if SAVE_RESULTS:
    results_folder = "./results/results_segmentator/MRC"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(results_folder, timestamp)
    os.makedirs(results_dir, exist_ok=True)
else:
    results_dir = None

# ================================================================================

# Initialize the data loader with your custom DataLoader class
data_loader = DataLoaderByPatient()
train_loader, val_loader, test_loader= data_loader.train_val_test_split_bypatient(
    images_folder=IMAGES_FOLDER,
    labels_folder=LABELS_FOLDER,
    splits=DATA_SPLITS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    transform=None
)
# ==============================================================================

# Check shape of data: Obtener un batch del train_loader
for images, labels in train_loader:
    print(f"Dimensiones de las imágenes: {images.shape}")
    print(f"Dimensiones de las etiquetas: {labels.shape}")
    break  

# =================================

# Train the model
print("Starting training...")
trained_model = train_model(segmentator, train_loader, criterion, optimizer, device, results_dir, NUM_EPOCHS)

# Evaluate the model
print("Evaluating model...")
avg_loss, mean_dice, mean_iou = evaluate_model(trained_model, test_loader, device, criterion, results_dir)

# Save the trained model if results are being saved
if SAVE_RESULTS:
    model_save_path = os.path.join(results_dir, 'unet_brain_segmentation.pth')
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# ==============================================================================

