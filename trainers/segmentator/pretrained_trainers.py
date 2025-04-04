# ==============================================================================
# Description: PyTorch training script for a U-Net model with custom dataloader
# Author: Caterina Fuster-Barceló
# Creation date: 30/08/2024
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
from dataloader.dataloader_MRC import DataLoaderByPatient
from matplotlib import pyplot as plt
from utils.metrics import dice_score, iou_score, local_dice_score, new_local_dice_score
import torch


# Training function
import torch
import numpy as np
import os

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
    model.train()  # Set the model to training mode
    
    epoch_losses = []  # Track training loss
    val_losses = []  # Track validation loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")  # Initialize best validation loss
    epochs_without_improvement = 0  # Counter for early stopping
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  # Ensure model is in training mode
        
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            assert inputs.min() >= 0 and inputs.max() <= 1, "WARNING: Input values should be between 0 and 1"
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            #if i % 10 == 9:  # Print every 10 mini-batches
                #print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                #running_loss = 0.0
        
        # Calculate average training loss for the epoch
        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}')

        # Validation Phase
        if val_dataloader is not None:
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # No need to compute gradients during validation
                for val_data in val_dataloader:
                    val_inputs, val_labels = val_data
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                    val_outputs = model(val_inputs)
                    loss = criterion(val_outputs, val_labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss:.4f}')

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0  # Reset counter
                # Save the best model
                if results_dir:
                    os.makedirs(results_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break

        scheduler.step()

        # Save training and validation losses
        if results_dir:
            with open(os.path.join(results_dir, 'training_losses.txt'), 'w') as f:
                for e, loss in enumerate(epoch_losses, 1):
                    f.write(f'Epoch {e}: Training Loss = {loss:.4f}\n')
            
            if val_dataloader:
                with open(os.path.join(results_dir, 'validation_losses.txt'), 'w') as f:
                    for e, loss in enumerate(val_losses, 1):
                        f.write(f'Epoch {e}: Validation Loss = {loss:.4f}\n')
    
    # 🔹 Plot losses at the end of training
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


import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import time
import seaborn as sns

def save_binary_mask(mask, save_dir, idx, image_name):
    """
    Function to save the binary mask as a PNG image.
    
    Args:
    - mask: The predicted binary mask.
    - save_dir: Directory to save the image.
    - idx: Index for saving the mask with a unique name.
    """
    # Convert mask to a numpy array and save as PNG
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
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    dice_scores = []
    iou_scores = []
    predictions = []

    print(f"Number of images: {len(dataloader)}")

    # Optional: Create a folder to save results
    if results_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(results_dir, timestamp)
        save_mask = os.path.join(save_dir, "binary_masks")
        os.makedirs(save_mask, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for i, data in enumerate(dataloader):
            inputs, labels, names = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Apply threshold to get binary masks
            binary_masks = (outputs > threshold).float()

            # Save each binary mask as a PNG image and calculate metrics
            for j in range(inputs.shape[0]):
                mask = binary_masks[j].cpu().squeeze(0)  # Remove the channel dimension if present
                mask_label = (labels[j] > threshold).float()  # Apply threshold to the labels as well
                save_binary_mask(mask_label, save_mask, i * inputs.shape[0] + j, names[j])

                # Calculate Dice and IoU scores
                dice = dice_score(mask, mask_label)
                iou = iou_score(mask, mask_label)

                # Store predictions and corresponding metrics
                predictions.append((inputs[j].cpu(), mask.cpu(), mask_label.cpu(), dice, iou, outputs[j].cpu(), names[j]))

                dice_scores.append(dice)
                iou_scores.append(iou)

    # Sort predictions by Dice score in descending order (optional, can be removed if not needed)
    predictions.sort(key=lambda x: x[3], reverse=True)

    # Print the total number of predictions collected
    print(f"Total predictions collected: {len(predictions)}")

    # Calculate average loss, Dice, and IoU
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

    # Save the numerical results to a text file
    if results_dir:
        with open(os.path.join(save_dir, "evaluation_results.txt"), "w") as f:
            f.write(f"Avg Loss: {avg_loss}\n")
            f.write(f"Mean Dice: {mean_dice} (std: {std_dice})\n")
            f.write(f"Mean IoU: {mean_iou} (std: {std_iou})\n")

    # Visualize and save all predictions
    if results_dir:
        for idx, (input_image, pred_mask, true_mask, dice, iou, raw_output,image_name) in enumerate(predictions):
            # Create the three subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # First subplot: Input image with label overlayed
            axes[0].imshow(input_image.permute(1, 2, 0))  # Assuming the image has (C, H, W)
            axes[0].imshow(true_mask.squeeze(0), cmap='Blues', alpha=0.35)  # Ground truth in blue
            axes[0].set_title(f"Original Image + Ground Truth Label")
            axes[0].axis('off')

            # Second subplot: Input image with predicted mask overlayed
            axes[1].imshow(input_image.permute(1, 2, 0))
            axes[1].imshow(pred_mask.squeeze(0), cmap='Reds', alpha=0.35)  # Predicted mask in red
            axes[1].set_title(f" Predicted Mask {idx+1} (Dice: {dice:.2f})")
            axes[1].axis('off')

            # Third subplot: Probability map (before thresholding)
            probability_map = raw_output.squeeze(0).cpu().numpy()  # Remove channel dimension
            masked_pred = np.ma.masked_where(probability_map <= 0.1, probability_map)  # Mask values <= 0.1
            cmap = plt.cm.RdYlGn
            norm = plt.Normalize(vmin=0.1, vmax=1) 
            im = axes[2].imshow(masked_pred, cmap=cmap, norm=norm)
            #sns.heatmap(probability_map, ax=axes[2], cmap='viridis', cbar=True)
            axes[2].set_title(f"Probability Map of the Prediction")
            axes[2].axis('off')
            # Agregar barra de colores
            cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label("Confianza de Predicción")

            # Save the visualization
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{idx}_{image_name}.png"))

    return avg_loss, mean_dice, mean_iou    



# ======= OLD VERSION FOR MRC? ============

def mrc_evaluate_model(model, dataloader, device, criterion, results_dir=None):
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
    print(f"Number of images: /{len(dataloader)}")

    if results_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(results_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            threshold = 0.97

            # Apply threshold to get binary masks
            binary_masks = (outputs > threshold).float()

            # Save each binary mask as a PNG image and calculate metrics
            for j in range(inputs.shape[0]):
                mask = binary_masks[j].cpu().squeeze(0)  # Remove the channel dimension if present
                mask_label = (labels[j] > threshold).float()  # Apply threshold to the labels as well
                save_binary_mask(mask_label, save_dir, i * inputs.shape[0] + j)

            # Calculate Dice and IoU scores
            # Iterar sobre cada imagen dentro del batch
            for j in range(inputs.shape[0]):
                dice = dice_score(outputs[j].unsqueeze(0), labels[j].unsqueeze(0))  # Métrica para una imagen
                iou = iou_score(outputs[j].unsqueeze(0), labels[j].unsqueeze(0))

                predictions.append((inputs[j].cpu(), outputs[j].cpu(), labels[j].cpu(), dice, iou))
                dice_scores.append(dice)
                iou_scores.append(iou)

            # dice = dice_score(outputs, labels)
            # iou = iou_score(outputs, labels)

            # dice_scores.append(dice)
            # iou_scores.append(iou)
            # predictions.append((inputs.cpu(), outputs.cpu(), labels.cpu(), dice, iou))

    print(f"Total predictions collected: {len(predictions)}")

    # Calculate average loss, Dice, and IoU
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

    if results_dir is not None:
        # Save numerical results
        with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
            f.write(f'Average Loss: {avg_loss:.4f}\n')
            f.write(f'Mean Dice Score: {mean_dice:.4f}\n')
            f.write(f'Mean IoU: {mean_iou:.4f}\n')

       
        # Sort predictions based on Dice score
        sorted_predictions = sorted(predictions, key=lambda x: x[3], reverse=True)  # Sort by Dice score (descending)
        

        for idx, (input_img, output_img, true_img, dice, iou) in enumerate(sorted_predictions):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            image_np = input_img.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
            label_np = true_img[0].squeeze().cpu().numpy()  # Eliminar la dimensión del canal
            pred_np = output_img[0].squeeze().cpu().numpy()  # Eliminar la dimensión del canal
            pred_np = (pred_np - np.min(pred_np)) / (np.max(pred_np) - np.min(pred_np) + 1e-8)

            # Crear máscara de superposición RGBA
            overlay_label = np.zeros((label_np.shape[0], label_np.shape[1], 4))  # (H, W, 4)
            overlay_label[label_np > 0] = [0, 0, 1, 0.5]  # Azul con 50% de transparencia

            # Imagen original con la verdad de terreno
            axes[0].imshow(image_np)
            axes[0].imshow(overlay_label)
            axes[0].set_title("Original Image + Ground Truth")
            axes[0].axis("off")

            # Imagen original con la predicción sobrepuesta
            axes[1].imshow(image_np)
            axes[1].imshow(pred_np, cmap='Reds', alpha=0.7)
            axes[1].set_title(f"Prediction {idx+1} (Dice: {dice:.4f}, IoU: {iou:.4f})")
            axes[1].axis("off")

            # Mapa de probabilidades de la predicción
            masked_pred = np.ma.masked_where(pred_np <= 0.1, pred_np)  # Ocultar valores <= 0.1 después de normalizar
            cmap = plt.cm.RdYlGn
            norm = plt.Normalize(vmin=0.1, vmax=1)  # Ajustar escala de colores
            im = axes[2].imshow(masked_pred, cmap=cmap, norm=norm)
            axes[2].set_title("Probability Map of the Prediction")
            axes[2].axis("off")

            # Agregar barra de colores
            cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label("Confianza de Predicción")

            plt.savefig(os.path.join(results_dir, f'prediction_{idx+1}.png'), dpi=300)
            plt.show()

    return avg_loss, mean_dice, mean_iou















# ======= OLD VERSIONS ============

def new_evaluate_model(model, dataloader, device, criterion, results_dir=None, threshold=0.80):
    """
    Function to evaluate the U-Net model on a validation/test set.
    
    Args:
    - model: The trained neural network model.
    - dataloader: DataLoader object providing the validation/test data.
    - device: Device to run the evaluation on (CPU or GPU).
    - criterion: Loss function used during training.
    - threshold: Threshold for binary mask generation (default is 0.2).
    
    Saves:
    - A folder with the timestamp containing the binary masks for each image.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    dice_scores = []
    iou_scores = []
    predictions = []

    print(f"Number of images: {len(dataloader)}")

    # Optional: Create a folder to save results
    if results_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(results_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Apply threshold to get binary masks (using threshold of 0.2)
            binary_masks = (outputs > threshold).float()

            # Save each binary mask as a PNG image
            for j in range(inputs.shape[0]):
                # Save the binary mask
                mask = binary_masks[j].cpu().squeeze(0)  # Remove the channel dimension if present
                save_binary_mask(mask, save_dir, i * inputs.shape[0] + j)

                # Calculate Dice and IoU scores for evaluation
                mask_label = (labels[j] > threshold).float()  # Apply threshold to the labels as well
                dice = dice_score(mask, mask_label)
                iou = iou_score(mask, mask_label)

                predictions.append((inputs[j].cpu(), mask.cpu(), mask_label.cpu(), dice, iou))
                dice_scores.append(dice)
                iou_scores.append(iou)

    print(f"Total predictions collected: {len(predictions)}")

    # Calculate average loss, Dice, and IoU
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

    # Save the numerical results to a text file
    if results_dir:
        with open(os.path.join(save_dir, "evaluation_results.txt"), "w") as f:
            f.write(f"Avg Loss: {avg_loss}\n")
            f.write(f"Mean Dice: {mean_dice} (std: {std_dice})\n")
            f.write(f"Mean IoU: {mean_iou} (std: {std_iou})\n")

    # Sort predictions based on Dice score
    sorted_predictions = sorted(predictions, key=lambda x: x[3], reverse=True)  # Sort by Dice score (descending)

    for idx, (input_img, output_img, true_img, dice, iou) in enumerate(sorted_predictions):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        image_np = input_img.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
        label_np = true_img[0].squeeze().cpu().numpy()  # Eliminar la dimensión del canal
        pred_np = output_img[0].squeeze().cpu().numpy()  # Eliminar la dimensión del canal
        pred_np = (pred_np - np.min(pred_np)) / (np.max(pred_np) - np.min(pred_np) + 1e-8)

        # Umbral para segmentación binaria después de normalizar
        th = 0.5
        binary_pred_mask = (pred_np > th).astype(np.uint8)

        # Crear máscara de superposición RGBA
        overlay_label = np.zeros((label_np.shape[0], label_np.shape[1], 4))  # (H, W, 4)
        overlay_label[label_np > 0] = [0, 0, 1, 0.5]  # Azul con 50% de transparencia

        # Imagen original con la verdad de terreno
        axes[0].imshow(image_np)
        axes[0].imshow(overlay_label)
        axes[0].set_title("Original Image + Ground Truth")
        axes[0].axis("off")

        # Imagen original con la predicción sobrepuesta
        axes[1].imshow(image_np)
        axes[1].imshow(pred_np, cmap='Reds', alpha=0.7)
        axes[1].set_title(f"Prediction {idx+1} (Dice: {dice:.4f}, IoU: {iou:.4f})")
        axes[1].axis("off")

        # Mapa de probabilidades de la predicción
        masked_pred = np.ma.masked_where(pred_np <= 0.1, pred_np)  # Ocultar valores <= 0.1 después de normalizar
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(vmin=0.1, vmax=1)  # Ajustar escala de colores
        im = axes[2].imshow(masked_pred, cmap=cmap, norm=norm)
        axes[2].set_title("Probability Map of the Prediction")
        axes[2].axis("off")

        # Agregar barra de colores
        cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label("Confianza de Predicción")

        plt.savefig(os.path.join(results_dir, f'prediction_{idx+1}.png'), dpi=300)
        plt.show()

    return avg_loss, mean_dice, mean_iou


def evaluate_model(model, dataloader, device, criterion, results_dir=None):
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
    print(f"Number of images: /{len(dataloader)}")

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate Dice and IoU scores
            # Iterar sobre cada imagen dentro del batch
            for j in range(inputs.shape[0]):
                dice = dice_score(outputs[j].unsqueeze(0), labels[j].unsqueeze(0))  # Métrica para una imagen
                iou = iou_score(outputs[j].unsqueeze(0), labels[j].unsqueeze(0))

                predictions.append((inputs[j].cpu(), outputs[j].cpu(), labels[j].cpu(), dice, iou))
                dice_scores.append(dice)
                iou_scores.append(iou)

            # dice = dice_score(outputs, labels)
            # iou = iou_score(outputs, labels)

            # dice_scores.append(dice)
            # iou_scores.append(iou)
            # predictions.append((inputs.cpu(), outputs.cpu(), labels.cpu(), dice, iou))

    print(f"Total predictions collected: {len(predictions)}")

    # Calculate average loss, Dice, and IoU
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

    if results_dir is not None:
        # Save numerical results
        with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
            f.write(f'Average Loss: {avg_loss:.4f}\n')
            f.write(f'Mean Dice Score: {mean_dice:.4f}\n')
            f.write(f'Mean IoU: {mean_iou:.4f}\n')

        # Get three random predictions to visualize
        random_predictions = random.sample(predictions, min(3, len(predictions)))

        for idx, (input_img, output_img, true_img, dice, iou) in enumerate(random_predictions):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(input_img.permute(1, 2, 0))
            axes[0].set_title("Input Image")
            axes[1].imshow(output_img[0], cmap='gray')
            axes[1].set_title(f"Prediction (Dice: {dice:.4f}, IoU: {iou:.4f})")
            axes[2].imshow(true_img[0], cmap='gray')
            axes[2].set_title("Ground Truth")
            
            plt.savefig(os.path.join(results_dir, f'prediction_{idx + 1}.png'), dpi=300)
            plt.close(fig)
        # Sort predictions based on Dice score
        sorted_predictions = sorted(predictions, key=lambda x: x[3], reverse=True)  # Sort by Dice score (descending)

        best_predictions = sorted_predictions[:10]  # Top 10 best
        worst_predictions = sorted_predictions[-10:]  # Bottom 10 worst
        

        for idx, (input_img, output_img, true_img, dice, iou) in enumerate(best_predictions):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            image_np = input_img.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
            label_np = true_img[0].squeeze().cpu().numpy()  # Remove channel dimension
            pred_np = output_img[0].squeeze().cpu().numpy()  # Remove channel dimension
            pred_np = (pred_np - np.min(pred_np)) / (np.max(pred_np) - np.min(pred_np) + 1e-8)
                
                # Umbral para segmentación binaria (después de normalizar)
            th = 0.5
            binary_pred_mask = (pred_np > th).astype(np.uint8)

            # Crear máscara de superposición RGBA
            overlay_label = np.zeros((label_np.shape[0], label_np.shape[1], 4))  # (H, W, 4)
            overlay_label[label_np > 0] = [0, 0, 1, 0.5]  # Azul con 50% de transparencia
            
            axes[0].imshow(image_np)
            axes[0].imshow(overlay_label)
            axes[0].set_title("Original Image + Ground Truth")
            axes[0].axis("off")

            # Imagen original con predicción sobrepuesta
            axes[1].imshow(image_np)
            axes[1].imshow(pred_np, cmap='Reds', alpha=0.7)
            #ax[1].imshow(overlay_label)
            axes[1].set_title(f"Best Prediction {idx+1} (Dice: {dice:.4f}, IoU: {iou:.4f})")
            axes[1].axis("off")
            

            masked_pred = np.ma.masked_where(pred_np <= 0.1, pred_np)  # Ocultar valores <= 0.3 después de normalizar
            cmap = plt.cm.RdYlGn
            norm = plt.Normalize(vmin=0.1, vmax=1)  # Ajustar escala de colores
            im = axes[2].imshow(masked_pred, cmap=cmap, norm=norm)
            axes[2].set_title("Probability Map of the Prediction")
            axes[2].axis("off")
                
            # Agregar barra de colores
            cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label("Confianza de Predicción")
            
            plt.savefig(os.path.join(results_dir, f'best_prediction_{idx+1}.png'), dpi=300)
            #plt.close(fig)
            plt.show()

        for idx, (input_img, output_img, true_img, dice, iou) in enumerate(worst_predictions):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            image_np = input_img.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
            label_np = true_img[0].squeeze().cpu().numpy()  # Remove channel dimension
            pred_np = output_img[0].squeeze().cpu().numpy()  # Remove channel dimension
            pred_np = (pred_np - np.min(pred_np)) / (np.max(pred_np) - np.min(pred_np) + 1e-8)
                
                # Umbral para segmentación binaria (después de normalizar)
            th = 0.5
            binary_pred_mask = (pred_np > th).astype(np.uint8)

            # Crear máscara de superposición RGBA
            overlay_label = np.zeros((label_np.shape[0], label_np.shape[1], 4))  # (H, W, 4)
            overlay_label[label_np > 0] = [0, 0, 1, 0.5]  # Azul con 50% de transparencia
            
            axes[0].imshow(image_np)
            axes[0].imshow(overlay_label)
            axes[0].set_title("Original Image + Ground Truth")
            axes[0].axis("off")

            # Imagen original con predicción sobrepuesta
            axes[1].imshow(image_np)
            axes[1].imshow(pred_np, cmap='Reds', alpha=0.7)
            #ax[1].imshow(overlay_label)
            axes[1].set_title(f"Worst Prediction {idx+1} (Dice: {dice:.4f}, IoU: {iou:.4f})")
            axes[1].axis("off")
            

            masked_pred = np.ma.masked_where(pred_np <= 0.1, pred_np)  # Ocultar valores <= 0.3 después de normalizar
            cmap = plt.cm.RdYlGn
            norm = plt.Normalize(vmin=0.1, vmax=1)  # Ajustar escala de colores
            im = axes[2].imshow(masked_pred, cmap=cmap, norm=norm)
            axes[2].set_title("Probability Map of the Prediction")
            axes[2].axis("off")
                
            # Agregar barra de colores
            cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label("Confianza de Predicción")
            
            plt.savefig(os.path.join(results_dir, f'worst_prediction_{idx+1}.png'), dpi=300)
            #plt.close(fig)
            plt.show()
        # Mostrar todas las predicciones en lugar de solo las mejores y peores

        for idx, (input_img, output_img, true_img, dice, iou) in enumerate(sorted_predictions):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            image_np = input_img.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
            label_np = true_img[0].squeeze().cpu().numpy()  # Eliminar la dimensión del canal
            pred_np = output_img[0].squeeze().cpu().numpy()  # Eliminar la dimensión del canal
            pred_np = (pred_np - np.min(pred_np)) / (np.max(pred_np) - np.min(pred_np) + 1e-8)

            # Umbral para segmentación binaria después de normalizar
            th = 0.5
            binary_pred_mask = (pred_np > th).astype(np.uint8)

            # Crear máscara de superposición RGBA
            overlay_label = np.zeros((label_np.shape[0], label_np.shape[1], 4))  # (H, W, 4)
            overlay_label[label_np > 0] = [0, 0, 1, 0.5]  # Azul con 50% de transparencia

            # Imagen original con la verdad de terreno
            axes[0].imshow(image_np)
            axes[0].imshow(overlay_label)
            axes[0].set_title("Original Image + Ground Truth")
            axes[0].axis("off")

            # Imagen original con la predicción sobrepuesta
            axes[1].imshow(image_np)
            axes[1].imshow(pred_np, cmap='Reds', alpha=0.7)
            axes[1].set_title(f"Prediction {idx+1} (Dice: {dice:.4f}, IoU: {iou:.4f})")
            axes[1].axis("off")

            # Mapa de probabilidades de la predicción
            masked_pred = np.ma.masked_where(pred_np <= 0.1, pred_np)  # Ocultar valores <= 0.1 después de normalizar
            cmap = plt.cm.RdYlGn
            norm = plt.Normalize(vmin=0.1, vmax=1)  # Ajustar escala de colores
            im = axes[2].imshow(masked_pred, cmap=cmap, norm=norm)
            axes[2].set_title("Probability Map of the Prediction")
            axes[2].axis("off")

            # Agregar barra de colores
            cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label("Confianza de Predicción")

            plt.savefig(os.path.join(results_dir, f'prediction_{idx+1}.png'), dpi=300)
            plt.show()

    return avg_loss, mean_dice, mean_iou