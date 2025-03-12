# ==============================================================================
# Description: PyTorch custom loss functions for segmentation tasks
# Author: Alejandro Guerrero-López, Yichun Sun
# Maintainer: Caterina Fuster-Barceló
# Creation date: 30/08/2024
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# FocalLossForProbabilities class definition
class FocalLossForProbabilities(nn.Module):
    """
    Implementation of Focal Loss for probabilities in PyTorch.

    Parameters
    ----------
    gamma : float, optional
        Focusing parameter, by default 2.0.
    alpha : float, optional
        Balancing factor, by default 0.25.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLossForProbabilities, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, probabilities, targets):
        # Clipping probabilities to avoid log(0) errors
        epsilon = 1e-6
        probabilities = torch.clamp(probabilities, epsilon, 1. - epsilon)

        # Calculate the focal loss
        pt = torch.where(targets == 1, probabilities, 1 - probabilities)
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt)
        return loss.mean()


# Helper function to identify the axis for 2D or 3D images
def identify_axis(shape):
    """
    Identifies the axis for 2D or 3D segmentation tasks.
    
    Parameters
    ----------
    shape : tuple
        Shape of the input tensor.
    
    Returns
    -------
    list
        List of axes corresponding to spatial dimensions.
    
    Raises
    ------
    ValueError
        If the shape does not correspond to a 2D or 3D tensor.
    """
    if len(shape) == 5:  # 3D
        return [2, 3, 4]
    elif len(shape) == 4:  # 2D
        return [2, 3]
    else:
        raise ValueError("Metric: Shape of tensor is neither 2D nor 3D.")


# SymmetricFocalLoss class definition
class SymmetricFocalLoss(nn.Module):
    """
    Symmetric Focal Loss for segmentation tasks.
    
    Parameters
    ----------
    delta : float, optional
        Weighting factor between false positives and false negatives, by default 0.7.
    gamma : float, optional
        Focal parameter, by default 2.0.
    epsilon : float, optional
        Small constant to avoid division by zero, by default 1e-07.
    """
    def __init__(self, delta=0.7, gamma=2.0, epsilon=1e-07):
        super(SymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # Loss calculations for each class
        back_ce = torch.pow(1 - y_pred[:, 0, :, :], self.gamma) * cross_entropy[:, 0, :, :]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = torch.pow(1 - y_pred[:, 1, :, :], self.gamma) * cross_entropy[:, 1, :, :]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))
        return loss


# AsymmetricFocalLoss class definition
class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for imbalanced datasets.
    
    Parameters
    ----------
    delta : float, optional
        Weighting factor between false positives and false negatives, by default 0.7.
    gamma : float, optional
        Focal parameter, by default 2.0.
    epsilon : float, optional
        Small constant to avoid division by zero, by default 1e-07.
    """
    def __init__(self, delta=0.7, gamma=2.0, epsilon=1e-07):
        super(AsymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # Loss calculations for each class, suppressing background class only
        back_ce = torch.pow(1 - y_pred[:, 0, :, :], self.gamma) * cross_entropy[:, 0, :, :]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:, 1, :, :]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))
        return loss


# SymmetricFocalTverskyLoss class definition
class SymmetricFocalTverskyLoss(nn.Module):
    """
    Symmetric Focal Tversky Loss for binary segmentation tasks.
    
    Parameters
    ----------
    delta : float, optional
        Weighting factor between false positives and false negatives, by default 0.7.
    gamma : float, optional
        Focal parameter, by default 0.75.
    epsilon : float, optional
        Small constant to avoid division by zero, by default 1e-07.
    """
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super(SymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives, false negatives, and false positives
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1 - y_pred), axis=axis)
        fp = torch.sum((1 - y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        # Loss calculations for each class
        back_dice = (1 - dice_class[:, 0]) * torch.pow(1 - dice_class[:, 0], -self.gamma)
        fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -self.gamma)

        loss = torch.mean(torch.stack([back_dice, fore_dice], axis=-1))
        return loss


# AsymmetricFocalTverskyLoss class definition
class AsymmetricFocalTverskyLoss(nn.Module):
    """
    Asymmetric Focal Tversky Loss for binary segmentation tasks.
    
    Parameters
    ----------
    delta : float, optional
        Weighting factor between false positives and false negatives, by default 0.7.
    gamma : float, optional
        Focal parameter, by default 0.75.
    epsilon : float, optional
        Small constant to avoid division by zero, by default 1e-07.
    """
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super(AsymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives, false negatives, and false positives
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1 - y_pred), axis=axis)
        fp = torch.sum((1 - y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        # Loss calculations, enhancing foreground class only
        back_dice = 1 - dice_class[:, 0]
        fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -self.gamma)

        loss = torch.mean(torch.stack([back_dice, fore_dice], axis=-1))
        return loss

# SymmetricUnifiedFocalLoss class definition
class SymmetricUnifiedFocalLoss(nn.Module):
    """
    Symmetric Unified Focal Loss combines Dice-based and Cross-Entropy-based loss functions.
    
    Parameters
    ----------
    weight : float, optional
        Lambda parameter controlling weight between Focal Tversky loss and Focal loss, by default 0.5.
    delta : float, optional
        Weighting factor between classes, by default 0.6.
    delta : float, optional
        Weighting factor between classes, by default 0.6.
    gamma : float, optional
        Focal parameter controlling the degree of background suppression and foreground enhancement, by default 0.5.
    epsilon : float, optional
        Small constant to avoid division by zero, by default 1e-07.
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5, epsilon=1e-07):
        super(SymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Calculate Symmetric Focal Tversky Loss
        symmetric_ftl = SymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(
            y_pred, y_true
        )
        # Calculate Symmetric Focal Loss
        symmetric_fl = SymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(
            y_pred, y_true
        )
        # Return weighted sum of Symmetric Focal Tversky Loss and Symmetric Focal Loss
        if self.weight is not None:
            return (self.weight * symmetric_ftl) + ((1 - self.weight) * symmetric_fl)
        else:
            return symmetric_ftl + symmetric_fl


# AsymmetricUnifiedFocalLoss class definition
class AsymmetricUnifiedFocalLoss(nn.Module):
    """
    Asymmetric Unified Focal Loss combines Dice-based and Cross-Entropy-based loss functions for imbalanced datasets.
    
    Parameters
    ----------
    weight : float, optional
        Lambda parameter controlling weight between Asymmetric Focal Tversky loss and Asymmetric Focal loss, by default 0.5.
    delta : float, optional
        Weighting factor between classes, by default 0.6.
    gamma : float, optional
        Focal parameter controlling the degree of background suppression and foreground enhancement, by default 0.5.
    from_logits : bool, optional
        If True, assumes that the input y_pred is not passed through a sigmoid, by default False.
    epsilon : float, optional
        Small constant to avoid division by zero, by default 1e-07.
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5, from_logits=False, epsilon=1e-07):
        super(AsymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.from_logits = from_logits
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Apply sigmoid to predictions if from_logits is True
        # if not self.from_logits:
        #     y_pred = torch.sigmoid(y_pred)

        # Adjust y_true and y_pred shapes for binary segmentation
        if y_true.size()[1] == 1:
            y_true = torch.cat((1 - y_true, y_true), dim=1)
            y_pred = torch.cat((1 - y_pred, y_pred), dim=1)

        # Calculate Asymmetric Focal Tversky Loss
        asymmetric_ftl = AsymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(
            y_pred, y_true
        )

        # Calculate Asymmetric Focal Loss
        asymmetric_fl = AsymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(
            y_pred, y_true
        )

        # Return weighted sum of Asymmetric Focal Tversky Loss and Asymmetric Focal Loss
        if self.weight is not None:
            return (self.weight * asymmetric_ftl) + ((1 - self.weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl

# DC_and_BCE_loss class definition
class DC_and_BCE_loss(nn.Module):
    """
    Combined Dice Coefficient and Binary Cross Entropy loss function.
    
    Parameters
    ----------
    bce_kwargs : dict
        Keyword arguments for BCEWithLogitsLoss.
    soft_dice_kwargs : dict
        Keyword arguments for the Dice loss function.
    weight_ce : float, optional
        Weight for the BCE loss, by default 1.
    weight_dice : float, optional
        Weight for the Dice loss, by default 1.
    use_ignore_label : bool, optional
        Whether to use the ignore label in the target, by default False.
    dice_class : class, optional
        Class implementing the Dice loss, by default MemoryEfficientSoftDiceLoss.
    """
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False, dice_class=None):
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # Invert target to create a mask where we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()

        # Compute BCE loss, applying mask if necessary
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)

        # Combine Dice and BCE losses
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class BCE_and_Dice_loss(nn.Module):
    """
    Combined Binary Cross-Entropy (BCE) and Dice loss function for binary segmentation.
    
    Parameters
    ----------
    bce_kwargs : dict
        Keyword arguments for BCELoss.
    dice_class : class, optional
        Class implementing the Dice loss, by default MemoryEfficientSoftDiceLoss.
    weight_ce : float, optional
        Weight for the BCE loss, by default 1.
    weight_dice : float, optional
        Weight for the Dice loss, by default 1.
    """
    def __init__(self, bce_kwargs=None, dice_class=None, weight_ce=1, weight_dice=1):
        super(BCE_and_Dice_loss, self).__init__()
        
        # Initialize BCE loss with the provided arguments
        if bce_kwargs is None:
            bce_kwargs = {}
        self.ce = nn.BCELoss(**bce_kwargs)

        # Initialize Dice loss with the provided Dice class
        if dice_class is None:
            raise ValueError("dice_class must be provided")
        self.dc = dice_class()

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        Forward pass to compute the combined BCE and Dice loss.
        
        Args:
        - net_output: torch.Tensor
            The predicted output from the model (probabilities).
        - target: torch.Tensor
            The ground truth labels.
        
        Returns:
        - result: torch.Tensor
            The combined BCE and Dice loss.
        """
        # Compute BCE loss
        ce_loss = self.ce(net_output, target.float())

        # Compute Dice loss
        dice_loss = self.dc(net_output, target.float())

        # Combine BCE and Dice losses with their respective weights
        result = self.weight_ce * ce_loss + self.weight_dice * dice_loss

        return result

class SimpleDiceLoss(nn.Module):
    def __init__(self):
        super(SimpleDiceLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        smooth = 1e-6  # Small constant to avoid division by zero
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()

        intersection = (y_pred * y_true).sum(dim=2).sum(dim=2)
        dice = (2. * intersection + smooth) / (y_pred.sum(dim=2).sum(dim=2) + y_true.sum(dim=2).sum(dim=2) + smooth)
        return 1 - dice.mean()