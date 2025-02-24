# ==============================================================================
# File: main_objectdetector.py
# Description: Main script for training and evaluating the YOLO object detector.
# Author: @cfusterbarcelo
# Creation Date: 05/01/2025
# ==============================================================================

import torch
from dataloader.dataloader_MRC_object_detector import ObjectDetectionDataLoader

# Define paths
base_dataset_folder = "./toydataset/object_detection/"
images_folder = base_dataset_folder  # This contains patient subfolders with images
annotations_folder = base_dataset_folder  # The YOLO annotations will be inside patient folders

SEED = 42  

# Step 1: Load Train, Validation, and Test Data
print("Loading dataset...")
train_loader, val_loader, test_loader = ObjectDetectionDataLoader.train_val_test_split(
    images_folder=images_folder,
    annotations_folder=annotations_folder,
    splits=(0.7, 0.15, 0.15),  # 70% train, 15% validation, 15% test
    batch_size=8,
    shuffle=True,
    seed=SEED,
    transform=None  # Apply any data augmentation if needed
)

# Get total number of images in the dataset
total_train_images = sum(len(batch[0]) for batch in train_loader)
total_batches = len(train_loader)

print(f"✅ Dataset Summary:")
print(f"   - Total training images: {total_train_images}")
print(f"   - Total batches: {total_batches} (Batch size: {train_loader.batch_size})")

# Show one sample batch for verification
for images, annotations in train_loader:
    print(f"Sample batch shape: {images.shape}")  # (batch_size, 3, H, W)
    break  # Stop after one batch

# Step 3: Proceed with Training the YOLO Model...
# Example:
# model = YOLOModel()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# for epoch in range(num_epochs):
#     train_one_epoch(model, train_loader, optimizer)
