from dataloader.dataloader_MRC_classificator import ClassificationDataLoader
import os

# ==============================================================================

# Configuration Parameters
SAVE_RESULTS = False  # Toggle to save results
NUM_EPOCHS = 5  # Number of training epochs
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 8  # Batch size for training
DATA_SPLITS = (0.34, 0.33, 0.33)  # Train, validation, test splits
IMAGES_FOLDER = "toydataset/classification/" # Path to the folder containing images

# Full dataset for training (uncomment when needed)
# IMAGES_FOLDER = "D:/Data/VolumetricHydrops/images/MRC"
# LABELS_FOLDER = "D:/Data/VolumetricHydrops/labels/MRC"

# ==============================================================================


# Initialize the DataLoader
dataloader = ClassificationDataLoader()

# Create train, val, and test DataLoaders
train_loader, val_loader, test_loader = dataloader.train_val_test_split(
    images_folder=IMAGES_FOLDER,
    annotations_file=LABELS_FILE,
    splits=(0.7, 0.15, 0.15),
    batch_size=8,
    shuffle=True,
    transform=None
)

# Print some information
print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of validation samples: {len(val_loader.dataset)}")
print(f"Number of test samples: {len(test_loader.dataset)}")

# Display a sample
for images, labels in train_loader:
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    break