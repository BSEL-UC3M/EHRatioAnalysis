# ===============================================================================
# File: dataloader_object_detector.py
# Description: DataLoader for MRC/PEI TIFF images and YOLO object detection annotations.
# Author: @cfusterbarcelo
# Creation Date: 05/01/2025
# Last Update: 26/03/2025
# ==============================================================================

import os
import torch
import cv2
import yaml
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tifffile import imread
print("✅ tifffile.imread importado correctamente")

class YoloObjectDetectorDataset(Dataset):
    """
    PyTorch Dataset for loading MRC/PEI TIFF images and YOLO-compatible bounding box annotations.
    """

    def __init__(self, image_files, dataset_yaml, transform=None, debug=False):
        self.image_files = image_files
        self.transform = transform
        self.debug = debug

        yaml_path = Path(dataset_yaml)
        if not yaml_path.exists():
            raise FileNotFoundError(f"❌ dataset.yaml file not found at {yaml_path}")

        with open(yaml_path, "r") as f:
            dataset_config = yaml.safe_load(f)

        self.train_path = Path(dataset_config["train"])
        self.val_path = Path(dataset_config["val"])
        self.test_path = Path(dataset_config["test"])

        # Detect dataset type (MRC or PEI)
        self.dataset_type = "PEI" if "PEI" in str(self.train_path) else "MRC"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx].strip()

        possible_paths = [
            self.train_path / filename,
            self.val_path / filename,
            self.test_path / filename,
        ]
        image_path = next((p for p in possible_paths if p.exists()), None)

        if image_path is None:
            raise FileNotFoundError(f"❌ Image file {filename} not found in dataset.")

        # ✅ Use appropriate loader based on file extension
        if image_path.suffix.lower() == ".tif":
            image = imread(str(image_path))
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)

        image = image.astype('float32')
        max_val = image.max() if image.max() != 0 else 1.0
        image /= max_val
        # Resize a tamaño fijo
        target_size = (384, 324)
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

        # Convert to uint8 for YOLO compatibility (if needed)
        if "PEI" in str(image_path):
            image = (image * 255).clip(0, 255).astype('uint8')

        if self.debug:
            print(f"🧠 {filename} | shape: {image.shape}, dtype: {image.dtype}, min: {image.min():.4f}, max: {image.max():.4f}")

        annotation_path = image_path.with_suffix('.txt')
        yolo_annotations = []

        if annotation_path.exists():
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    yolo_annotations.append([class_id, x_center, y_center, width, height])
        else:
            print(f"❌ Warning: Annotation file {annotation_path} not found.")

        if self.transform:
            image = self.transform(image)

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image_tensor = image  # Already a tensor from transform

        if self.debug:
            print(f"📊 Tensor: shape={image_tensor.shape}, mean={image_tensor.mean():.4f}, std={image_tensor.std():.4f}")

        yolo_annotations = torch.tensor(yolo_annotations, dtype=torch.float32) if yolo_annotations else torch.zeros((0, 5))

        return image_tensor, yolo_annotations, str(image_path.name)

class ObjectDetectionDataLoader:
    @staticmethod
    def load_from_existing_split(dataset_yaml, batch_size=8, shuffle=True, transform=None, debug=False):
        yaml_path = Path(dataset_yaml)
        if not yaml_path.exists():
            raise FileNotFoundError(f"❌ dataset.yaml file not found at {yaml_path}")

        with open(yaml_path, "r") as f:
            dataset_config = yaml.safe_load(f)

        train_path = Path(dataset_config["train"])
        val_path = Path(dataset_config["val"])
        test_path = Path(dataset_config["test"])

        #def get_image_files(folder):
        #    return [file.name for file in folder.glob("*.tif")]
        def get_image_files(folder):
            """ Get list of image files (.tif and .png) in a given folder """
            return [file.name for file in folder.glob("*.png")] + \
                [file.name for file in folder.glob("*.tif")]

        train_files = get_image_files(train_path)
        val_files = get_image_files(val_path)
        test_files = get_image_files(test_path)

        train_dataset = YoloObjectDetectorDataset(train_files, dataset_yaml, transform=transform, debug=debug)
        val_dataset = YoloObjectDetectorDataset(val_files, dataset_yaml, transform=transform, debug=debug)
        test_dataset = YoloObjectDetectorDataset(test_files, dataset_yaml, transform=transform, debug=debug)

        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), \
               DataLoader(val_dataset, batch_size=batch_size, shuffle=False), \
               DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
