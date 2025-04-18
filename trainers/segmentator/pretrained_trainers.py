# ==============================================================================
# Description: PyTorch training script for a U-Net model with custom dataloader
# Author: @laurarodmu 
# Creation date: 30/10/2024
# ==============================================================================

import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from losses import losses
from matplotlib import pyplot as plt
from utils.metrics import dice_score, iou_score
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time
import seaborn as sns



def train_model(model, dataloader, criterion, optimizer, device, results_dir=None, num_epochs=25, 
                val_dataloader=None, patience=15):
    """
    Train the U-Net model with early stopping.
    
    Args:
    - model: The neural network model to be trained.
    - dataloader: DataLoader object providing the training data.
    - criterion: Loss function.
    - optimizer: Optimization algorithm.
    - device: Device to run the training on (CPU or GPU).
    - num_epochs: Number of training epochs.
    - val_dataloader: Validation DataLoader for early stopping.
    - patience: Number of epochs to wait before stopping if no improvement.
    
    Returns:
    - model: The trained model.
    """
    model.train() 
    
    epoch_losses = []  
    val_losses = [] 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")  
    epochs_without_improvement = 0 
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train() 
        for i, data in enumerate(dataloader):
            inputs, labels, names = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            assert inputs.min() >= 0 and inputs.max() <= 1, "WARNING: Input values should be between 0 and 1"
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}')

        if val_dataloader is not None:
            model.eval()  
            val_loss = 0.0
            with torch.no_grad():  
                for val_data in val_dataloader:
                    val_inputs, val_labels, names = val_data
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                    val_outputs = model(val_inputs)
                    loss = criterion(val_outputs, val_labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0  
            
                if results_dir:
                    os.makedirs(results_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break

        scheduler.step()

        
        if results_dir:
            with open(os.path.join(results_dir, 'training_losses.txt'), 'w') as f:
                for e, loss in enumerate(epoch_losses, 1):
                    f.write(f'Epoch {e}: Training Loss = {loss:.4f}\n')
            
            if val_dataloader:
                with open(os.path.join(results_dir, 'validation_losses.txt'), 'w') as f:
                    for e, loss in enumerate(val_losses, 1):
                        f.write(f'Epoch {e}: Validation Loss = {loss:.4f}\n')
    

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Training Loss', marker='o')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()

    print('Finished Training')
    return model

def save_binary_mask(mask, save_dir, idx, image_name):
    """
    Function to save the binary mask as a PNG image.
    
    Args:
    - mask: The predicted binary mask.
    - save_dir: Directory to save the image.
    - idx: Index for saving the mask with a unique name.
    """
    mask = mask.squeeze().cpu().numpy()
    plt.imsave(os.path.join(save_dir, f"{image_name}.png"), mask, cmap='gray')

def complete_evaluate_model(model, dataloader, device, criterion, results_dir=None, threshold=0.20):
    """
    Function to evaluate the U-Net model on a validation/test set.
    
    Args:
    - model: The trained neural network model.
    - dataloader: DataLoader object providing the validation/test data.
    - device: Device to run the evaluation on (CPU or GPU).
    - criterion: Loss function used during training.
    - threshold: Threshold for binary mask generation (default is 0.8).
    
    Saves:
    - A folder with the timestamp containing the visualizations for each image.
    """
    model.eval() 
    total_loss = 0.0
    dice_scores = []
    iou_scores = []
    predictions = []

    print(f"Number of images: {len(dataloader)}")

    if results_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(results_dir, timestamp)
        save_mask = os.path.join(save_dir, "binary_masks")
        os.makedirs(save_mask, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():  
        for i, data in enumerate(dataloader):
            inputs, labels, names = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            binary_masks = (outputs > threshold).float()

            for j in range(inputs.shape[0]):
                mask = binary_masks[j].cpu().squeeze(0)  
                mask_label = (labels[j] > threshold).float() 
                save_binary_mask(mask, save_mask, i * inputs.shape[0] + j, names[j])

                dice = dice_score(mask, mask_label)
                iou = iou_score(mask, mask_label)

                predictions.append((inputs[j].cpu(), mask.cpu(), mask_label.cpu(), dice, iou, outputs[j].cpu(), names[j]))

                dice_scores.append(dice)
                iou_scores.append(iou)

    predictions.sort(key=lambda x: x[3], reverse=True)

    #print(f"Total predictions collected: {len(predictions)}")

    avg_loss = total_loss / len(dataloader.dataset)
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    mean_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)

    print(f'Average loss on the evaluation set: {avg_loss:.4f}')
    print(f'Mean Dice Score: {mean_dice:.4f}')
    print(f'Standard Deviation of Dice Score: {std_dice:.4f}')
    print(f'Mean IoU: {mean_iou:.4f}')
    print(f'Standard Deviation of IoU: {std_iou:.4f}')

    if results_dir:
        with open(os.path.join(save_dir, "evaluation_results.txt"), "w") as f:
            f.write(f"Avg Loss: {avg_loss}\n")
            f.write(f"Mean Dice: {mean_dice} (std: {std_dice})\n")
            f.write(f"Mean IoU: {mean_iou} (std: {std_iou})\n")

    if results_dir:
        for idx, (input_image, pred_mask, true_mask, dice, iou, raw_output,image_name) in enumerate(predictions):

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(input_image.permute(1, 2, 0)) 
            axes[0].imshow(true_mask.squeeze(0), cmap='Blues', alpha=0.35) 
            axes[0].set_title(f"Original Image + Ground Truth Label")
            axes[0].axis('off')

            axes[1].imshow(input_image.permute(1, 2, 0))
            axes[1].imshow(pred_mask.squeeze(0), cmap='Reds', alpha=0.35) 
            axes[1].set_title(f" Predicted Mask {idx+1} (Dice: {dice:.2f})")
            axes[1].axis('off')

            probability_map = raw_output.squeeze(0).cpu().numpy() 
            masked_pred = np.ma.masked_where(probability_map <= 0.1, probability_map) 
            cmap = plt.cm.RdYlGn
            norm = plt.Normalize(vmin=0.1, vmax=1) 
            im = axes[2].imshow(masked_pred, cmap=cmap, norm=norm)
            axes[2].set_title(f"Probability Map of the Prediction")
            axes[2].axis('off')
            cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label("Confianza de Predicción")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{idx}_{image_name}.png"))

    return avg_loss, mean_dice, mean_iou    


