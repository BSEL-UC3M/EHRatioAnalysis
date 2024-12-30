# ==============================================================================
# Description: PyTorch training script for a U-Net model with custom dataloader
# Author: Caterina Fuster-Barceló
# Creation date: 30/08/2024
# ==============================================================================

import torch
import random
import os
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from losses import losses
from dataloader.dataloader_supervised import DataLoaderByPatient
from matplotlib import pyplot as plt
from utils.metrics import dice_score, iou_score

# Training function
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=25):
    """
    Function to train the U-Net model.
    
    Args:
    - model: The neural network model to be trained.
    - dataloader: DataLoader object providing the training data.
    - criterion: Loss function.
    - optimizer: Optimization algorithm.
    - device: Device to run the training on (CPU or GPU).
    - num_epochs: Number of training epochs.
    
    Returns:
    - model: The trained model.
    """
    model.train()  # Set the model to training mode
    
    epoch_losses = []  # Initialize list to store loss for each epoch
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # Get the inputs and labels from the dataloader
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            assert inputs.min() >= 0 and inputs.max() <= 1, "WARNING: Input values should be between 0 and 1"
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            running_loss += loss.item()
            
            if i % 10 == 9:    # Print every 10 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
        
        # Calculate average loss for the epoch and store it
        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}')
    
        # Save the losses to a text file
        with open('training_losses.txt', 'w') as f:
            for epoch, loss in enumerate(epoch_losses, 1):
                f.write(f'Epoch {epoch}: Loss = {loss:.4f}\n')
    
    print('Finished Training')
    return model


def evaluate_model(model, dataloader, device, criterion, results_folder="results"):
    """
    Function to evaluate the U-Net model on a validation/test set.
    
    Args:
    - model: The trained neural network model.
    - dataloader: DataLoader object providing the validation/test data.
    - device: Device to run the evaluation on (CPU or GPU).
    - criterion: Loss function used during training.
    
    Saves:
    - A folder with the timestamp containing numerical results in a txt file
    - Three random predictions compared with their ground truth in PNG format
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    dice_scores = []
    iou_scores = []
    predictions = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate Dice and IoU scores
            dice = dice_score(outputs, labels)
            iou = iou_score(outputs, labels)
            dice_scores.append(dice)
            iou_scores.append(iou)
            predictions.append((inputs.cpu(), outputs.cpu(), labels.cpu(), dice, iou))

    # Calculate average loss, Dice, and IoU
    avg_loss = total_loss / len(dataloader)
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    
    print(f'Average loss on the evaluation set: {avg_loss:.4f}')
    print(f'Mean Dice Score: {mean_dice:.4f}')
    print(f'Mean IoU: {mean_iou:.4f}')
    
    # Create results folder with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = os.path.join(results_folder, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    # Save numerical results
    with open(os.path.join(result_dir, 'results.txt'), 'w') as f:
        f.write(f'Average Loss: {avg_loss:.4f}\n')
        f.write(f'Mean Dice Score: {mean_dice:.4f}\n')
        f.write(f'Mean IoU: {mean_iou:.4f}\n')

    # Get three random predictions to visualize
    random_predictions = random.sample(predictions, min(3, len(predictions)))

    for idx, (input_img, output_img, true_img, dice, iou) in enumerate(random_predictions):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(input_img[0].permute(1, 2, 0))
        axes[0].set_title("Input Image")
        axes[1].imshow(output_img[0][0], cmap='gray')
        axes[1].set_title(f"Prediction (Dice: {dice:.4f}, IoU: {iou:.4f})")
        axes[2].imshow(true_img[0][0], cmap='gray')
        axes[2].set_title("Ground Truth")
        
        plt.savefig(os.path.join(result_dir, f'prediction_{idx + 1}.png'), dpi=300)
        plt.close(fig)

    # Identify best and worst predictions based on Dice score
    best_prediction = max(predictions, key=lambda x: x[3])
    worst_prediction = min(predictions, key=lambda x: x[3])

    # Save best prediction
    input_img, output_img, true_img, dice, iou = best_prediction
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_img[0].permute(1, 2, 0))
    axes[0].set_title("Best Input Image")
    axes[1].imshow(output_img[0][0], cmap='gray')
    axes[1].set_title(f"Best Prediction (Dice: {dice:.4f}, IoU: {iou:.4f})")
    axes[2].imshow(true_img[0][0], cmap='gray')
    axes[2].set_title("Best Ground Truth")
    plt.savefig(os.path.join(result_dir, 'best_prediction.png'), dpi=300)
    plt.close(fig)

    # Save worst prediction
    input_img, output_img, true_img, dice, iou = worst_prediction
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_img[0].permute(1, 2, 0))
    axes[0].set_title("Worst Input Image")
    axes[1].imshow(output_img[0][0], cmap='gray')
    axes[1].set_title(f"Worst Prediction (Dice: {dice:.4f}, IoU: {iou:.4f})")
    axes[2].imshow(true_img[0][0], cmap='gray')
    axes[2].set_title("Worst Ground Truth")
    plt.savefig(os.path.join(result_dir, 'worst_prediction.png'), dpi=300)
    plt.close(fig)

    return avg_loss, mean_dice, mean_iou

