# Segmentation Results using UNet

## Overview

This repository contains segmentation results obtained using a UNet model applied to different datasets and preprocessing techniques. The main goal is to compare the performance of different approaches using Mean Dice Score and Mean Intersection over Union (IoU).

## Results

The following table summarizes the results:

| Approach                  | Mean Dice Score | Mean IoU |
|---------------------------|----------------|----------|
| MRC                       | 0.8560         | 0.7546   |
| PEI with preprocessing    | 0.7134         | 0.4761   |
| PEI without preprocessing | 0.6612         | 0.5239   |
| PEI inverted              | 0.6612         | 0.5239   |

- **MRC**: One segmentation technique was applied.
- **PEI**: Three different approaches were tested:
  - Using raw images.
  - Using preprocessed images.
  - Using inverted raw images.

## Model

- The segmentation model used is **UNet**.
[Model README](MRC/BEST/20250405%20MRC%20NEW%20TRAINING%20MODIEFIED%20LABELS/20250405-170559/README.md)



