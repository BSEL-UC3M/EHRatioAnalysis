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

# Ensure reproducibility
torch.manual_seed(42)

# Load the pre-trained U-Net model from torch.hub
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                       in_channels=3, out_channels=1, init_features=32, pretrained=True)

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function (Binary Cross-Entropy for segmentation)
criterion = losses.AsymmetricUnifiedFocalLoss()

# Define the optimizer (Adam optimizer with a learning rate of 1e-4)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

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
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # Print every 10 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
    
    print('Finished Training')
    return model

# Define the evaluation function
def evaluate_model(model, dataloader, device):
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
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Average loss on the evaluation set: {avg_loss:.4f}')
    return avg_loss

# Set up the data directories
images_folder = "toydataset\\toydataset\\MRC\\images"
labels_folder = "toydataset\\toydataset\\MRC\\labels"

# Initialize the data loader with your custom class
data_loader = DataLoaderByPatient()
train_loader, test_loader = data_loader.train_test_split_bypatient(
    images_folder=images_folder, 
    labels_folder=labels_folder, 
    test_size=0.2, 
    batch_size=8
)

# Train the model
num_epochs = 1
trained_model = train_model(model, train_loader, criterion, optimizer, device, num_epochs)

# Evaluate the model on the test set
evaluate_model(trained_model, test_loader, device)

# Save the trained model
torch.save(trained_model.state_dict(), 'unet_brain_segmentation.pth')
