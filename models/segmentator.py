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
import torch.nn.functional as F


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


class first_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(first_UNet, self).__init__()

        # Encoder (Downsampling)
        self.enc1 = self.conv_block(in_channels, features[0])
        self.enc2 = self.conv_block(features[0], features[1])
        self.enc3 = self.conv_block(features[1], features[2])
        self.enc4 = self.conv_block(features[2], features[3])

        # Bottleneck
        self.bottleneck = self.conv_block(features[3], features[3] * 2)

        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = self.conv_block(features[3] * 2, features[3])

        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = self.conv_block(features[2] * 2, features[2])

        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = self.conv_block(features[1] * 2, features[1])

        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = self.conv_block(features[0] * 2, features[0])

        # Final Output Layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """ Helper function to create a two-layer convolutional block. """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        x = F.max_pool2d(enc1, kernel_size=2, stride=2)

        enc2 = self.enc2(x)
        x = F.max_pool2d(enc2, kernel_size=2, stride=2)

        enc3 = self.enc3(x)
        x = F.max_pool2d(enc3, kernel_size=2, stride=2)

        enc4 = self.enc4(x)
        x = F.max_pool2d(enc4, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        x = self.upconv4(x)
        x = torch.cat((x, enc4), dim=1)
        x = self.dec4(x)

        x = self.upconv3(x)
        x = torch.cat((x, enc3), dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.dec1(x)
        x = self.final_conv(x)

        return torch.sigmoid(x) #Apply sigmoid to ensure output values are in range [0,1]

# Example usage
if __name__ == "__main__":
    model = first_UNet()
    x = torch.randn(1, 3, 256, 256)  # Example input (Batch, Channels, Height, Width)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Expected: (1, 1, 256, 256)


class DoubleConv(nn.Module):
    """(Convolution => [BN] => LeakyReLU) * 2 + Dropout"""
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Residual shortcut
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.conv(x) + self.residual(x)  # Add residual connection
        return self.dropout(out)


class UNet_new(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_features=64, depth=4, dropout=0.2):
        """
        Optimized U-Net Model.

        Args:
        - in_channels (int): Number of input channels (default: 3 for RGB).
        - out_channels (int): Number of output channels (default: 1 for binary segmentation).
        - base_features (int): Initial feature map count (default: 64).
        - depth (int): Number of encoder-decoder blocks (default: 4).
        - dropout (float): Dropout probability (default: 0.2).
        """
        super(UNet_new, self).__init__()

        self.depth = depth
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder Blocks
        for i in range(depth):
            in_ch = in_channels if i == 0 else base_features * (2 ** (i - 1))
            out_ch = base_features * (2 ** i)
            self.encoders.append(DoubleConv(in_ch, out_ch, dropout))

        # Bottleneck
        self.bottleneck = DoubleConv(base_features * (2 ** (depth - 1)), base_features * (2 ** depth), dropout)

        # Decoder Blocks
        for i in range(depth - 1, -1, -1):
            in_ch = base_features * (2 ** (i + 1))
            out_ch = base_features * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(in_ch, out_ch, dropout))

        # Final Convolution
        self.final_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

        # Initialize Weights
        self._initialize_weights()

    def forward(self, x):
        enc_feats = []

        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            enc_feats.append(x)
            x = F.max_pool2d(x, kernel_size=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(self.depth):
            x = self.upconvs[i](x)
            x = torch.cat((enc_feats[-(i + 1)], x), dim=1)  # Skip connection
            x = self.decoders[i](x)
        x = self.final_conv(x)

        return torch.sigmoid(x)

    def _initialize_weights(self):
        """Applies Xavier/He initialization to convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

from collections import OrderedDict


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class UNetOptimized(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNetOptimized, self).__init__()

        features = init_features
        self.encoder1 = UNetOptimized._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetOptimized._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNetOptimized._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNetOptimized._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            UNetOptimized._block(features * 8, features * 16, name="bottleneck"),
            nn.InstanceNorm2d(features * 16)  # InstanceNorm for better generalization
        )

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNetOptimized._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNetOptimized._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNetOptimized._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNetOptimized._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

        self._initialize_weights()  # Initialize weights

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=features)),  # Fixed GroupNorm
                    (name + "relu1", nn.LeakyReLU(0.1, inplace=True)),
                    (name + "dropout", nn.Dropout2d(0.2)),  # Dropout for regularization
                    (name + "conv2", nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm2", nn.GroupNorm(num_groups=8, num_channels=features)),  # Fixed GroupNorm
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


# Ensure classes are properly exposed
__all__ = ["Segmentator", "first_UNet", "UNet_new", "UNet", "UNetOptimized"]



