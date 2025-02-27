# ==============================================================================
# File: models/object_detector.py
# Description: Defines the YOLO object detection model.
# Author: @cfusterbarcelo
# Creation Date: 24/02/2025
# ==============================================================================

import torch
from ultralytics import YOLO

class YOLOv5:
    """
    Wrapper class for loading and using YOLOv5.
    """

    def __init__(self, model_name="yolov5s", pretrained=True, device=None):
        """
        Initializes the YOLOv5 model.

        Parameters:
        - model_name: (str) Name of the YOLOv5 model (e.g., 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x').
        - pretrained: (bool) Whether to load pretrained weights.
        - device: (str) 'cuda' or 'cpu' (automatically detected if None).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(f"{model_name}.pt") if pretrained else YOLO()
        self.model.to(self.device)

    def train(self, data_yaml, epochs=50, batch_size=8):
        """
        Trains YOLOv5 on a custom dataset.

        Parameters:
        - data_yaml: (str) Path to dataset configuration YAML file.
        - epochs: (int) Number of training epochs.
        - batch_size: (int) Batch size for training.
        """
        self.model.train(data=data_yaml, epochs=epochs, batch=batch_size)

    def save_model(self, path="yolov5_trained.pt"):
        """ Saves the trained YOLOv5 model. """
        self.model.save(path)

    def load_model(self, path="yolov5_trained.pt"):
        """ Loads a trained YOLOv5 model. """
        self.model = YOLO(path)
