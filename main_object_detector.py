# ==============================================================================
# File: main_objectdetector.py
# Description: Main script for training and evaluating an object detector.
# Author: @cfusterbarcelo
# Creation Date: 24/02/2025
# ==============================================================================

import os
from dataloader.dataloader_MRC_object_detector import ObjectDetectionDataLoader
from trainers.object_detector.yolo_trainer import train_yolo

SAVE_RESULTS = True  # Toggle this flag to enable/disable result saving

if __name__ == "__main__":  # ✅ Prevent multiprocessing issues on Windows
    # ✅ Define the correct dataset path (inside YOLO/)
    dataset_yaml = "./toydataset/object_detection/YOLO/dataset.yaml"

    # ✅ Ensure `dataset.yaml` exists before proceeding
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"❌ ERROR: Dataset YAML file not found at {dataset_yaml}. "
                                "Please run `generate_yolo_annotations.py` first.")

    # ✅ Load Train, Validation, and Test Data
    print("📂 Loading dataset...")
    train_loader, val_loader, test_loader = ObjectDetectionDataLoader.load_from_existing_split(
        images_folder="./toydataset/object_detection/YOLO/",
        annotations_folder="./toydataset/object_detection/YOLO/",
        batch_size=8,
        shuffle=True,
        transform=None  # Apply any data augmentation if needed
    )

    # ✅ Check if data was successfully loaded
    if train_loader is None or len(train_loader) == 0:
        print("⚠️ No valid training data found. Skipping training process.")
        exit()

    # ✅ Get dataset statistics
    total_train_images = sum(len(batch[0]) for batch in train_loader)
    total_batches = len(train_loader)

    print(f"✅ Dataset Summary:")
    print(f"   - Total training images: {total_train_images}")
    print(f"   - Total batches: {total_batches} (Batch size: {train_loader.batch_size})")

    # ✅ Show one batch for verification
    for images, annotations in train_loader:
        print(f"Sample batch shape: {images.shape}")  # (batch_size, 3, H, W)
        break  # Stop after one batch

    # ✅ Train and evaluate YOLO (with reduced verbosity)
    print("🚀 Starting YOLO training and evaluation...")

    train_yolo(
    dataset_yaml=dataset_yaml,
    epochs=50,
    batch_size=8,
    model_name="yolov5s",
    save_results=SAVE_RESULTS,
    verbose=False  # Reduce printed logs if needed
    )
    print("✅ Training and evaluation completed!")
