# ==============================================================================
# Description: Main
# Author: Caterina Fuster-Barceló
# Creation date: 03/09/2024
# ==============================================================================

import torch
import torch.optim as optim
from losses import losses
from dataloader.dataloader_MRC import DataLoaderByPatient
from trainers.segmentator.pretrained_trainers import train_model, evaluate_model
from datetime import datetime
import os
from models.segmentator import Segmentator

segmentator = Segmentator()

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentator = segmentator.to(device)

# Define the loss function (Binary Cross-Entropy for segmentation)
criterion = losses.BCE_and_Dice_loss(
    bce_kwargs={},  # Default settings for BCELoss
    dice_class=losses.SimpleDiceLoss,  # Using the simple Dice loss defined above
    weight_ce=1,  # Weight for BCE loss
    weight_dice=1  # Weight for Dice loss
)

# Define the optimizer (Adam optimizer with a learning rate of 1e-4)
optimizer = optim.Adam(segmentator.parameters(), lr=1e-4)

results_folder = "./results/"

# TOY DATASET
images_folder = "toydataset\\toydataset\\MRC\\images"
labels_folder = "toydataset\\toydataset\\MRC\\labels"

# CAT's DATASET
# images_folder = "D:\\Data\\VolumetricHydrops\\images\\MRC"
# labels_folder = "D:\\Data\\VolumetricHydrops\\labels\\MRC"

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join(results_folder, timestamp)
os.makedirs(results_dir, exist_ok=True)

# Initialize the data loader with your custom class
data_loader = DataLoaderByPatient()
train_loader, val_loader, test_loader = data_loader.train_val_test_split_bypatient(
    images_folder=images_folder,
    labels_folder=labels_folder,
    splits=(0.7, 0.15, 0.15),
    batch_size=8,
    shuffle=True,
    transform=None
)

# Train the model
num_epochs = 5
trained_model = train_model(segmentator, train_loader, criterion, optimizer, device, num_epochs, results_dir)

# Evaluate the model on the test set
avg_loss, mean_dice, mean_iou, results_dir = evaluate_model(trained_model, val_loader, device, criterion, results_dir)

# Save the trained model
torch.save(trained_model.state_dict(), results_dir +'unet_brain_segmentation.pth')
