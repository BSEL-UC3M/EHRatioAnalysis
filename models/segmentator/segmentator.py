# ==============================================================================
# File: segmentator.py
# Description: This file defines the baseline segmentation model using a pre-trained
#              U-Net architecture from TorchHub. The model is designed for binary
#              segmentation tasks (e.g., medical image segmentation).
#              It includes a custom Segmentator class that wraps a optimized U-Net model,
#              allowing for easy integration and usage. 
# Author: @laurarodrmu
# Created: 30/12/2024
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ==============================================================================
# Explanation:
# This script initializes a pre-trained U-Net model from TorchHub, which is designed
# for binary segmentation tasks. The model has the following features:
# - Takes 3-channel (RGB) input images.
# - Outputs a single-channel prediction map.
# - Uses 32 initial convolutional features.
# - Leverages pre-trained weights for fast convergence and strong performance.
#
# The U-Net architecture is well-suited for segmentation tasks because of its
# encoder-decoder structure with skip connections, allowing it to capture both
# high-level context and fine-grained details.
#
# How to Use:
# - Import the `Segmentator` class from this file.
# - Initialize the model using `segmentator = Segmentator()`.
# - Pass the input images through the model to get segmentation predictions.
# ==============================================================================

class Segmentator(nn.Module):
    """
    Segmentator Class: A wrapper for the pre-trained U-Net model from TorchHub.

    Attributes
    ----------
    model : nn.Module
        The pre-trained U-Net model loaded from TorchHub.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Performs a forward pass on the input tensor and returns the output predictions.
    """

    def __init__(self, in_channels=3, out_channels=1, init_features=32, pretrained=True):
        """
        Initializes the Segmentator with a pre-trained U-Net model.

        Parameters
        ----------
        in_channels : int, optional
            Number of input channels. Default is 3 (RGB images).
        out_channels : int, optional
            Number of output channels. Default is 1 (binary segmentation).
        init_features : int, optional
            Number of features in the first convolutional layer. Default is 32.
        pretrained : bool, optional
            Whether to use pre-trained weights. Default is True.
        """
        super(Segmentator, self).__init__()

        # Ensure reproducibility
        torch.manual_seed(42)

        # Load the pre-trained U-Net model from TorchHub
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                    in_channels=in_channels, out_channels=out_channels,
                                    init_features=init_features, pretrained=pretrained)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width),
            containing the predicted segmentation maps.
        """
        return self.model(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import os

class CustomUNet(nn.Module):
    """
    CustomUNet is a customized U-Net architecture for image segmentation tasks, 
    incorporating several enhancements for improved performance and regularization.

    Key Features:
    - Encoder-decoder structure with skip connections, following the traditional U-Net design.
    - Uses Group Normalization instead of Batch Normalization to improve training stability,
      especially on smaller batch sizes.
    - Incorporates Dropout layers throughout the network to mitigate overfitting.
    - Instance Normalization and additional dropout in the bottleneck layer for stronger regularization.
    - Weight initialization is handled via Kaiming He initialization (suitable for LeakyReLU activations).

    Parameters:
        in_channels (int): Number of input channels. Default is 3 (e.g., RGB images).
        out_channels (int): Number of output channels. Default is 1 (e.g., binary segmentation).
        init_features (int): Number of feature maps in the first layer. Default is 32.

    Example usage:
        model = CustomUNet(in_channels=3, out_channels=1)
    """

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(CustomUNet, self).__init__()

        features = init_features
        self.encoder1 = CustomUNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = CustomUNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = CustomUNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = CustomUNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            CustomUNet._block(features * 8, features * 16, name="bottleneck"),
            nn.InstanceNorm2d(features * 16),
            nn.Dropout2d(0.3)  
        )

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = CustomUNet._block((features * 8) * 2, features * 8, name="dec4")
        self.dropout4 = nn.Dropout2d(0.2)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = CustomUNet._block((features * 4) * 2, features * 4, name="dec3")
        self.dropout3 = nn.Dropout2d(0.2)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = CustomUNet._block((features * 2) * 2, features * 2, name="dec2")
        self.dropout2 = nn.Dropout2d(0.2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = CustomUNet._block(features * 2, features, name="dec1")
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        self._initialize_weights()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.dropout4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec3 = self.dropout3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec2 = self.dropout2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.dropout1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=features)),  
                    (name + "relu1", nn.LeakyReLU(0.1, inplace=True)),
                    (name + "dropout", nn.Dropout2d(0.2)),  
                    (name + "conv2", nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm2", nn.GroupNorm(num_groups=8, num_channels=features)),  
                    (name + "relu2", nn.LeakyReLU(0.1, inplace=True)),
                ]
            )
        )

    def _initialize_weights(self):
        """Applies Xavier/He initialization to convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def new_visualize_segmentation(self, data_loader, device='cpu', results_dir=None, save_results=False):
        """
        Visualizes segmentation results from a given data loader.
        """
        self.eval() 
        with torch.no_grad():  
            for i, (image, label) in enumerate(data_loader):
                if i == 1:  
                    break

                image = image.to(device)
                label = label.to(device)
                prediction = self(image)
                image_np = image[1].permute(1, 2, 0).cpu().numpy()  
                label_np = label[1].squeeze().cpu().numpy()  
                pred_np = prediction[1].squeeze().cpu().numpy() 

                pred_np = (pred_np - np.min(pred_np)) / (np.max(pred_np) - np.min(pred_np) + 1e-8)
                overlay_label = np.zeros((label_np.shape[0], label_np.shape[1], 4))  
                overlay_label[label_np > 0] = [0, 0, 1, 0.5]  
                
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))

                ax[0].imshow(image_np)
                ax[0].imshow(overlay_label)
                ax[0].set_title("Original Image + Ground Truth")
                ax[0].axis("off")

                ax[1].imshow(image_np)
                ax[1].imshow(pred_np, cmap='Reds', alpha=0.75)
                ax[1].set_title("Prediction")
                ax[1].axis("off")
                
                masked_pred = np.ma.masked_where(pred_np <= 0.1, pred_np) 
                cmap = plt.cm.RdYlGn
                norm = plt.Normalize(vmin=0.1, vmax=1)  
                im = ax[2].imshow(masked_pred, cmap=cmap, norm=norm)
                ax[2].set_title("Probability Map of the Prediction")
                ax[2].axis("off")
                
                cbar = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
                cbar.set_label("Prediction Confidence")

                if save_results and results_dir is not None:
                    os.makedirs(results_dir, exist_ok=True)
                    save_path = os.path.join(results_dir, "segmentation_result.png")
                    plt.savefig(save_path, bbox_inches='tight')
                    print(f"Imagen guardada en: {save_path}")

                plt.show()



# Ensure classes are properly exposed
__all__ = ["Segmentator", "CustomUNet" ]