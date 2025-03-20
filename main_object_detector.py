# ==============================================================================
# File: main_object_detector.py
# Description: Main script for training and evaluating the YOLO object detector.
# Author: @cfusterbarcelo
# Creation Date: 24/02/2025
# Last Update: 25/02/2025
# ==============================================================================

import os
from dataloader.dataloader_MRC_object_detector import ObjectDetectionDataLoader
from trainers.object_detector.yolo_trainer import train_yolo

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SAVE_RESULTS = True  # Toggle this flag to enable/disable result saving
EPOCHS = 50  # Number of training epochs
BATCH_SIZE = 8  # Training batch size
DATASET_YAML = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/YOLO_toydataset_annotations/dataset.yaml"  # Path to dataset.yaml

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":  # ✅ Prevent multiprocessing issues on Windows

    # Ensure dataset.yaml exists before proceeding
    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(f"ERROR: Dataset YAML file not found at {DATASET_YAML}. "
                                "Please run `generate_yolo_annotations.py` first.")

    # Load dataset
    print("Loading dataset...")
    train_loader, val_loader, test_loader = ObjectDetectionDataLoader.load_from_existing_split(
        dataset_yaml=DATASET_YAML,
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=None
    )

    # Check if data was successfully loaded
    if not train_loader or len(train_loader) == 0:
        print("No valid training data found. Skipping training process.")
        exit()

    # Dataset summary
    total_train_images = sum(len(batch[0]) for batch in train_loader)
    total_batches = len(train_loader)

    print(f"Dataset Summary:")
    print(f"   - Total training images: {total_train_images}")
    print(f"   - Total batches: {total_batches} (Batch size: {BATCH_SIZE})")

    # Show one batch for verification
    for images, annotations in train_loader:
        print(f"Sample batch shape: {images.shape}")  # (batch_size, 3, H, W)
        break  # Stop after one batch

    # Train and evaluate YOLO
    print("Starting YOLO training and evaluation...")
    train_yolo(
        dataset_yaml=DATASET_YAML,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_name="yolov5s",
        save_results=SAVE_RESULTS,
        verbose=False  # Reduce printed logs if needed
    )

    print("Training and evaluation completed!")
