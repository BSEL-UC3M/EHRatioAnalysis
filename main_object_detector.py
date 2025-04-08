# ==============================================================================
# File: main_object_detector.py
# Description: Main script for training and evaluating the YOLO object detector.
# Author: @cfusterbarcelo
# Creation Date: 24/02/2025
# Last Update: 25/02/2025
# ==============================================================================

import os
from dataloader.dataloader_object_detector import ObjectDetectionDataLoader
from trainers.object_detector.yolo_trainer import train_yolo
from torchvision import transforms
from utils.convert_dataset_to_uint8 import convert_yolo_dataset_to_uint8
# ==============================================================================
# CONFIGURATION
# ==============================================================================
SAVE_RESULTS = True  # Toggle this flag to enable/disable result saving
EPOCHS = 50 # Number of training epochs
BATCH_SIZE = 8  # Training batch size
# Convertir el dataset original a TIFF uint8 en carpeta temporal
# ORIGINAL_YAML = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/YOLO_annotations/dataset.yaml" # MRC
ORIGINAL_YAML = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/YOLO_annotations_PEI/dataset.yaml" # PEI
# TEMP_UINT8_DIR = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/YOLO_annotations_MRC_uint8"
TEMP_UINT8_DIR = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/YOLO_annotations_PEI_uint8" # PEI

DATASET_YAML = convert_yolo_dataset_to_uint8(
    original_yaml_path=ORIGINAL_YAML,
    output_base_dir=TEMP_UINT8_DIR
)

# OUTPUT_DIR="results/results_object_detector/MRC"
OUTPUT_DIR="results/results_object_detector/PEI" # PEI
# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":

    # Validación de existencia del dataset.yaml
    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(f"❌ Dataset YAML file not found at {DATASET_YAML}")

    # Puedes definir un transform personalizado si lo necesitas
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((384, 324))  # Solo si tus imágenes lo necesitan
    ])

    # Cargar dataset
    print("🔍 Loading dataset...")
    train_loader, val_loader, test_loader = ObjectDetectionDataLoader.load_from_existing_split(
        dataset_yaml=DATASET_YAML,
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=custom_transform,
        debug=True
    )

    if not train_loader or len(train_loader) == 0:
        print("❌ No valid training data found. Skipping training process.")
        exit()

    total_train_images = sum(len(batch[0]) for batch in train_loader)
    total_batches = len(train_loader)

    print(f"📊 Dataset Summary:")
    print(f"   - Total training images: {total_train_images}")
    print(f"   - Total batches: {total_batches} (Batch size: {BATCH_SIZE})")

    for images, annotations, paths in train_loader:
        print(f"🖼 Sample batch shape: {images.shape}")
        print(f"📁 First image file names: {paths[:3]}")
        break

    print("🚀 Starting YOLO training and evaluation...")
    train_yolo(
        dataset_yaml=DATASET_YAML,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_name="yolov5su",
        save_results=SAVE_RESULTS,
        verbose=False,
        conf=0.35,
        patience=20,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.0,
        augment=False,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=OUTPUT_DIR
    )

    print("✅ Training and evaluation completed!")