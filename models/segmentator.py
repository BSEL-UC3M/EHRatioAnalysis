# ==============================================================================
# File: segmentator.py
# Description: This file defines the baseline segmentation model using a pre-trained
#              U-Net architecture from TorchHub. The model is designed for binary
#              segmentation tasks (e.g., medical image segmentation).
# Author: @cfusterbarcelo
# Created: 30/12/2024
# ==============================================================================

import torch
import torch.nn as nn

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


# Example usage:
if __name__ == "__main__":
    # Instantiate the model
    segmentator = Segmentator()

    # Print the model architecture
    print(segmentator)

    # Example input tensor
    example_input = torch.rand(1, 3, 256, 256)  # Batch size = 1, RGB image of 256x256

    # Perform a forward pass
    example_output = segmentator(example_input)

    # Print output shape
    print(f"Output shape: {example_output.shape}")
