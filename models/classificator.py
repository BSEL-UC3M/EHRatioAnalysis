# ==============================================================================
# File: classificator.py
# Description: Simple CNN model for image classification.
# Author: @cfusterbarcelo
# Created: 09/01/2025
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for image classification.
    """

    def __init__(self, num_classes=2):
        """
        Initializes the SimpleCNN model.

        Parameters:
        - num_classes: The number of output classes for classification.
        """
        super(SimpleCNN, self).__init__()

        # Define layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass through the model.
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
