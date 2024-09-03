# ==============================================================================
# Description: Main
# Author: Caterina Fuster-Barceló
# Creation date: 03/09/2024
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from losses import losses
from dataloader.dataloader_supervised import DataLoaderByPatient
from trainers.pretrained_trainers import train_model, evaluate_model

# Ensure reproducibility
torch.manual_seed(42)

# Load the pre-trained U-Net model from torch.hub
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                       in_channels=3, out_channels=1, init_features=32, pretrained=True)

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function (Binary Cross-Entropy for segmentation)
criterion = losses.BCE_and_Dice_loss(
    bce_kwargs={},  # Default settings for BCELoss
    dice_class=losses.SimpleDiceLoss,  # Using the simple Dice loss defined above
    weight_ce=1,  # Weight for BCE loss
    weight_dice=1  # Weight for Dice loss
)

# Define the optimizer (Adam optimizer with a learning rate of 1e-4)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Set up the data directories - toy dataset
# images_folder = "toydataset\\toydataset\\MRC\\images"
# labels_folder = "toydataset\\toydataset\\MRC\\labels"

images_folder = "D:\\Data\\VolumetricHydrops\\images\\MRC"
labels_folder = "D:\\Data\\VolumetricHydrops\\labels\\MRC"

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

# print(f'Number of patients in the training set: {len(train_loader.dataset)}')
# print(f'Number of patients in the validation set: {len(val_loader.dataset)}')
# print(f'Number of patients in the test set: {len(test_loader.dataset)}')
# print(f'Total number of patients: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}')

# Train the model
num_epochs = 10
trained_model = train_model(model, train_loader, criterion, optimizer, device, num_epochs)

# Evaluate the model on the test set
evaluate_model(trained_model, test_loader, device, criterion)

# Save the trained model
torch.save(trained_model.state_dict(), 'unet_brain_segmentation.pth')
