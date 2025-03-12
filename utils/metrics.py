from sklearn.metrics import jaccard_score

# Define a function to calculate the Dice score
def dice_score(y_pred, y_true, threshold=0.5):
    smooth = 1e-6
    y_pred = (y_pred > threshold).float()
    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return dice.item()

# Define a function to calculate the IoU score
def iou_score(y_pred, y_true, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    y_pred_np = y_pred.cpu().numpy().flatten()
    y_true_np = y_true.cpu().numpy().flatten()
    iou = jaccard_score(y_true_np, y_pred_np)
    return iou

import torch

# Define a function to calculate the Local Dice score
def local_dice_score(y_pred, y_true, threshold=0.5, margin=10):
    """
    Calculate the Local Dice score within a window surrounding the area of interest.
    
    Args:
    - y_pred: Predicted segmentation tensor (binary mask).
    - y_true: Ground truth segmentation tensor (binary mask).
    - threshold: Threshold to binarize predictions.
    - margin: Additional margin around the bounding box for local region extraction.
    
    Returns:
    - Local Dice score (float).
    """
    # Binarize predictions
    y_pred = (y_pred > threshold).float()

    # Find the bounding box of the true mask (y_true)
    coords = torch.nonzero(y_true)  # Get coordinates of all non-zero pixels
    if coords.shape[0] == 0:  # If there's no object, return Dice of 0
        return 0.0

    # Calculate bounding box
    min_coords = coords.min(dim=0).values - margin
    max_coords = coords.max(dim=0).values + margin

    # Ensure the bounding box stays within the image boundaries
    min_coords = torch.clamp(min_coords, min=0)
    max_coords = torch.clamp(max_coords, max=torch.tensor(y_true.shape))

    # Extract the local region from both y_true and y_pred
    y_true_local = y_true[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1]]
    y_pred_local = y_pred[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1]]

    # Calculate Dice score for the local region
    smooth = 1e-6
    intersection = (y_pred_local * y_true_local).sum()
    dice = (2. * intersection + smooth) / (y_pred_local.sum() + y_true_local.sum() + smooth)
    
    return dice.item()

import torch

def new_local_dice_score(y_pred, y_true, threshold=0.5, margin=10):
    """
    Calculate the Local Dice score within a window surrounding the area of interest.
    
    Args:
    - y_pred: Predicted segmentation tensor (binary mask).
    - y_true: Ground truth segmentation tensor (binary mask).
    - threshold: Threshold to binarize predictions.
    - margin: Additional margin around the bounding box for local region extraction.
    
    Returns:
    - Local Dice score (float).
    """
    # Step 1: Binarize predictions
    y_pred = (y_pred > threshold).float()

    # Step 2: Find the bounding box of the true mask (y_true)
    coords = torch.nonzero(y_true, as_tuple=True)  # Faster than returning a full tensor
    if len(coords[0]) == 0:  # If there's no object, return Dice of 0
        return 0.0

    # Step 3: Get bounding box coordinates
    min_coords = torch.tensor([coords[0].min(), coords[1].min()]) - margin
    max_coords = torch.tensor([coords[0].max(), coords[1].max()]) + margin

    # Ensure the bounding box stays within the image boundaries (height and width only)
    min_coords = torch.clamp(min_coords, min=0)
    max_coords[0] = torch.clamp(max_coords[0], max=y_true.shape[2])  # Height dimension
    max_coords[1] = torch.clamp(max_coords[1], max=y_true.shape[3])  # Width dimension

    # Step 4: Extract the local region from both y_true and y_pred
    y_true_local = y_true[:, min_coords[0]:max_coords[0], min_coords[1]:max_coords[1]]
    y_pred_local = y_pred[:, min_coords[0]:max_coords[0], min_coords[1]:max_coords[1]]

    # Step 5: Calculate Dice score for the local region
    smooth = 1e-6  # Smoothing to avoid division by zero
    intersection = (y_pred_local * y_true_local).sum()
    dice = (2. * intersection + smooth) / (y_pred_local.sum() + y_true_local.sum() + smooth)

    return dice.item()


