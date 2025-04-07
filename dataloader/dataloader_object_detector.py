# ==============================================================================
# File: dataloader_object_detector.py
# Description: DataLoader for MRC/PEI TIFF images and YOLO object detection annotations.
# Author: @cfusterbarcelo
# Creation Date: 05/01/2025
# Last Update: 25/02/2025
# ==============================================================================

import os
import torch
import cv2
import yaml
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tifffile import imread
print("✅ tifffile.imread importado correctamente")

class YoloObjectDetectorDataset(Dataset):
    """
    PyTorch Dataset for loading MRC/PEI TIFF images and YOLO-compatible bounding box annotations.
    """

    def __init__(self, image_files, dataset_yaml, transform=None, debug=False):
        """
        Parameters:
        - image_files: List of image filenames.
        - dataset_yaml: Path to dataset.yaml (containing train/val/test paths).
        - transform: Optional transformations.
        """
        self.image_files = image_files
        self.transform = transform
        self.debug=debug

        # Load dataset.yaml to get the correct paths
        yaml_path = Path(dataset_yaml)
        if not yaml_path.exists():
            raise FileNotFoundError(f"❌ dataset.yaml file not found at {yaml_path}")

        with open(yaml_path, "r") as f:
            dataset_config = yaml.safe_load(f)

        # Extract correct paths for images and annotations
        self.train_path = Path(dataset_config["train"])
        self.val_path = Path(dataset_config["val"])
        self.test_path = Path(dataset_config["test"])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx].strip()

        # Buscar imagen en las tres carpetas posibles
        possible_paths = [
            self.train_path / filename,
            self.val_path / filename,
            self.test_path / filename,
        ]
        image_path = next((p for p in possible_paths if p.exists()), None)

        if image_path is None:
            raise FileNotFoundError(f"❌ Image file {filename} not found in dataset.")

        # ✅ Leer imagen TIFF (cualquier tipo)
        image = imread(image_path)

        # 🔄 Convertir a float32 y normalizar a [0, 1]
        image = image.astype("float32")
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val > 0:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = image * 0  # imagen constante

        # 🎯 Escalar a [0, 255] y convertir a uint8
        image = (image * 255).astype("uint8")

        # Convertir a RGB si es necesario
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)  # (C,H,W) → (H,W,C)

        # 🛠 DEBUG opcional
        if self.debug:
            print(f"🧠 {filename} | shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
            try:
                cv2.imshow("Debug TIFF Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            except Exception as e:
                print(f"⚠️ Could not display image: {e}")

        # Cargar anotaciones YOLO
        annotation_path = image_path.with_suffix('.txt')
        yolo_annotations = []

        if annotation_path.exists():
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    yolo_annotations.append([class_id, x_center, y_center, width, height])
        else:
            print(f"❌ Warning: Annotation file {annotation_path} not found.")

        # Aplicar transformaciones si hay
        if self.transform:
            image = self.transform(image)

        # Convertir a tensor si no se hizo aún
        if isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        if self.debug:
            print(f"📊 Tensor: shape={image_tensor.shape}, mean={image_tensor.mean():.4f}, std={image_tensor.std():.4f}")

        yolo_annotations = torch.tensor(yolo_annotations, dtype=torch.float32) if yolo_annotations else torch.zeros((0, 5))

        return image_tensor, yolo_annotations, str(image_path.name)



class ObjectDetectionDataLoader:
    """
    Utility class to load object detection datasets.
    """

    @staticmethod
    def load_from_existing_split(dataset_yaml, batch_size=8, shuffle=True, transform=None, debug=False):
        """
        Loads pre-split YOLO train, val, and test datasets using dataset.yaml.

        Parameters:
        - dataset_yaml: Path to dataset.yaml file.
        - batch_size: Batch size for the DataLoader.
        - shuffle: Whether to shuffle the dataset.
        - transform: Optional image transformations.

        Returns:
        - train_loader, val_loader, test_loader
        """
        # Load dataset.yaml to get paths
        yaml_path = Path(dataset_yaml)
        if not yaml_path.exists():
            raise FileNotFoundError(f"❌ dataset.yaml file not found at {yaml_path}")

        with open(yaml_path, "r") as f:
            dataset_config = yaml.safe_load(f)

        # Extract paths
        train_path = Path(dataset_config["train"])
        val_path = Path(dataset_config["val"])
        test_path = Path(dataset_config["test"])

        def get_image_files(folder):
            """ Get list of image files in a given folder """
            return [file.name for file in folder.glob("*.tif")]

        train_files = get_image_files(train_path)
        val_files = get_image_files(val_path)
        test_files = get_image_files(test_path)

        train_dataset = YoloObjectDetectorDataset(train_files, dataset_yaml, transform=transform, debug=debug)
        val_dataset = YoloObjectDetectorDataset(val_files, dataset_yaml, transform=transform, debug=debug)
        test_dataset = YoloObjectDetectorDataset(test_files, dataset_yaml, transform=transform, debug=debug)

        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), \
               DataLoader(val_dataset, batch_size=batch_size, shuffle=False), \
               DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
