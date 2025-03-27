# ==============================================================================
# File: main_object_detector.py
# Description: Main script for training and evaluating the YOLO object detector.
# Author: @cfusterbarcelo
# Creation Date: 24/02/2025
# Last Update: 25/02/2025
# ==============================================================================

import os
from dataloader.dataloader_PEI_object_detector import ObjectDetectionDataLoader
from trainers.object_detector.yolo_trainer import train_yolo
from torchvision import transforms
# ==============================================================================
# CONFIGURATION
# ==============================================================================
SAVE_RESULTS = True  # Toggle this flag to enable/disable result saving
EPOCHS = 50 # Number of training epochs
BATCH_SIZE = 8  # Training batch size
DATASET_YAML = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/YOLO_annotations_toydataset_PEI/dataset.yaml"  # Path to dataset.yaml
# main_object_detector.py
proxy_yaml="/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/YOLO_annotations_proxy/dataset.yaml"

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":  # ✅ Prevent multiprocessing issues on Windows

    from torchvision import transforms

    # Ruta directa al YAML proxy (ya generada previamente)
    PROXY_YAML = "/Users/claudiacastrillonalvarez/Desktop/github/EHRatioAnalysis/YOLO_annotations_proxy/dataset.yaml"

    # Detect dataset type from path
    is_pei_dataset = "PEI" in DATASET_YAML.upper()

    # Si es PEI, cambiamos a proxy
    if is_pei_dataset:
        if not os.path.exists(PROXY_YAML):
            raise FileNotFoundError(f"❌ PEI detected, but proxy dataset.yaml not found at {PROXY_YAML}.\n"
                                    f"Please run `prepare_yolo_proxy_images()` first.")
        print(f"📂 PEI dataset detected. Switching to proxy dataset: {PROXY_YAML}")
        DATASET_YAML = PROXY_YAML  # ⬅️ Reemplazamos el original por el proxy

    # Validación
    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(f"ERROR: Dataset YAML file not found at {DATASET_YAML}.\n"
                                "Please run `generate_yolo_annotations.py` or proxy generator.")

    # Define transform solo para PEI
    if is_pei_dataset:
        print("🛠 Aplicando transform CenterCrop para PEI dataset...")
        custom_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((384, 324))  # ⬅️ tamaño estándar de PEI
        ])
    else:
        print("✅ No transform necesario para MRC dataset.")
        custom_transform = None

    # Load dataset
    print("Loading dataset...")
    train_loader, val_loader, test_loader = ObjectDetectionDataLoader.load_from_existing_split(
        dataset_yaml=DATASET_YAML,
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=custom_transform,
        debug=True
    )

    if not train_loader or len(train_loader) == 0:
        print("No valid training data found. Skipping training process.")
        exit()

    total_train_images = sum(len(batch[0]) for batch in train_loader)
    total_batches = len(train_loader)

    print(f"Dataset Summary:")
    print(f"   - Total training images: {total_train_images}")
    print(f"   - Total batches: {total_batches} (Batch size: {BATCH_SIZE})")

    for images, annotations, paths in train_loader:
        print(f"Sample batch shape: {images.shape}")
        print(f"First image file names: {paths[:3]}")
        break

    print("Starting YOLO training and evaluation...")
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
        val_loader=val_loader
    )

    print("Training and evaluation completed!")


