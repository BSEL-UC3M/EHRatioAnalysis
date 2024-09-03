# ==============================================================================
# Description: PyTorch training script for a U-Net model with custom dataloader
# Author: Caterina Fuster-Barceló
# Creation date: 30/08/2024
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from losses import losses
from dataloader.dataloader_supervised import DataLoaderByPatient

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


# Define the evaluation function
def evaluate_model(model, dataloader, device, criterion):
    """
    Function to evaluate the U-Net model on a validation/test set.
    
    Args:
    - model: The trained neural network model.
    - dataloader: DataLoader object providing the validation/test data.
    - device: Device to run the evaluation on (CPU or GPU).
    
    Returns:
    - avg_loss: The average loss on the validation/test set.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for i, data in enumerate(dataloader):
            inputs, labels = data
            assert inputs.min() >= 0 and inputs.max() <= 1, "WARNING: Input values should be between 0 and 1"
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            assert inputs.min() >= 0 and inputs.max() <= 1, "WARNING: Input values should be between 0 and 1"
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Average loss on the evaluation set: {avg_loss:.4f}')
    return avg_loss

