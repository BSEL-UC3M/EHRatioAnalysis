import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F

class FiveLayerCNN(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        """
        CNN model adapted for PEI images.
        
        Args:
        - num_classes (int): Number of output classes.
        - dropout_prob (float): Dropout probability for regularization.
        """
        super(FiveLayerCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # **🔹 Manually set FC input size to match the output**
        self.fc_input_size = 512 * 12 * 10  # ✅ Fixed to 61440 based on your logs

        # Commented for mismatch in inference
        # self.fc_layers = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.fc_input_size, 1024),  # Adjusted to 61440 input
        #     nn.ReLU(),
        #     nn.Dropout(dropout_prob),
        #     nn.Linear(1024, num_classes)
        # )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.fc_input_size, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the CNN.
        """
        x = self.conv_layers(x)
        if not hasattr(self, '_printed_shape'):
            self._printed_shape = True
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x