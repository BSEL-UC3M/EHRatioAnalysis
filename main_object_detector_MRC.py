# ===============================================================================
# File: main_object_detector_MRC.py
# Description: Main script for loading, training, and evaluating the YOLO object detector on MRC images.
# Author: @claudiacastrillon
# Creation Date: 24/02/2025
# Last Update: 25/02/2025
# ===============================================================================

import os
import datetime
from dataloader.dataloader_MRC_object_detector_MRC import MRCObjectDetectionDataLoader
from models.object_detector.yolo_MRC import YOLOv5
from pathlib import Path

# ===============================================================================
# CONFIGURATION
# ===============================================================================
SAVE_RESULTS = input("Do you want to save detection results? (yes/no): ").strip().lower() == "yes"
EPOCHS = 50  # Number of training epochs
BATCH_SIZE = 8  # Training batch size
MRC_IMAGES_DIR = "/Users/claudiacastrillonalvarez/Desktop/IMAGES_YOLO_toydataset/MRC_YOLO_toydataset/MRC"
MRC_CSV_FILE = "/Users/claudiacastrillonalvarez/Desktop/IMAGES_YOLO_toydataset/MRC_YOLO_toydataset/MRC_coordinates_toydataset.csv"

# Create results directory with a timestamped subfolder
BASE_RESULTS_DIR = Path("results/object_detector/MRC_object_detector")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MRC_RESULTS_DIR = BASE_RESULTS_DIR / f"yolo_{TIMESTAMP}"
if SAVE_RESULTS:
    MRC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================
if __name__ == "__main__":  # ✅ Prevent multiprocessing issues on Windows

    # Load dataset
    print("Loading dataset...")
    train_loader = MRCObjectDetectionDataLoader.load_from_existing_split(
        image_root_dir=MRC_IMAGES_DIR,
        csv_file=MRC_CSV_FILE,
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

    # Initialize YOLO model
    print("Initializing YOLO model...")
    yolo_model = YOLOv5(model_name="yolov5s", pretrained=True)

    # Perform object detection and evaluation
    print("Starting YOLO detection and evaluation...")
    yolo_model.detect_and_save(MRC_IMAGES_DIR, MRC_CSV_FILE, MRC_RESULTS_DIR, save_results=SAVE_RESULTS)

    print("Detection process completed!")
    print(f"Results saved in: {MRC_RESULTS_DIR}")