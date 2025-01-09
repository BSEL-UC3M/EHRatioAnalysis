# ==============================================================================
# File: toy_classificator.py
# Description: Trainer for the SimpleCNN model for image classification.
# Author: [Your Name]
# Created: [Date]
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    """
    Trains the SimpleCNN model.

    Parameters:
    - model: The CNN model to train.
    - dataloader: DataLoader providing the training data.
    - criterion: Loss function.
    - optimizer: Optimization algorithm.
    - device: Device to run the training (CPU or GPU).
    - num_epochs: Number of epochs to train.

    Returns:
    - model: The trained model.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    print("Training complete.")
    return model


def evaluate_model(model, dataloader, device):
    """
    Evaluates the SimpleCNN model.

    Parameters:
    - model: The trained CNN model.
    - dataloader: DataLoader providing the evaluation data.
    - device: Device to run the evaluation (CPU or GPU).

    Returns:
    - accuracy: Accuracy of the model on the evaluation data.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy
