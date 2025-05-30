# ==============================================================================
# File: toy_classificator.py
# Description: Trainer for the Resnet model + 5 layer CNN + cross validation for image classification.
# Author: @claudiacastrillon
# Created: 30/01/2025
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold


# Train a given model for the specified number of epochs. Use a training and validation dataset, 
# compute loss and accuracy at each epoch and use backpropagation and optimizer updates
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    Trains the model and tracks loss/accuracy over epochs.

    Parameters:
    - model: The CNN model to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - criterion: Loss function.
    - optimizer: Optimization algorithm.
    - device: Device to run the training (CPU/GPU).
    - num_epochs: Number of training epochs.

    Returns:
    - model: Trained model.
    - train_losses: List of training losses per epoch.
    - val_losses: List of validation losses per epoch.
    - train_accuracies: List of training accuracy per epoch.
    - val_accuracies: List of validation accuracy per epoch.
    """
    model.train()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train, total_train = 0, 0

        # Training loop

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation loop, compute validation loss and accuracy 
        model.eval()
        running_val_loss = 0.0
        correct_val, total_val = 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        model.train()

    print("Training complete.")
    return model, train_losses, val_losses, train_accuracies, val_accuracies

# Evaluates trained model on test/validation data, computes loss and accuracy
def evaluate_model(model, dataloader, device):
    """
    Evaluates the model.
    """
    y_true, y_pred = [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1) # get predicted labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = (np.array(y_true) == np.array(y_pred)).mean() * 100
    
    return y_true, y_pred, avg_loss, accuracy


# load ResNet18 and replaces the fc layer for the new dataset 
def fine_tune_resnet(num_classes, device, learning_rate):
    """
    Fine-tune ResNet18 on the dataset.
    """
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # load pre-trained weights 
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes) # replace fc with a new layer that matches the number of classes in the dataset
    resnet = resnet.to(device) # move to gpu 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)
    
    return resnet, criterion, optimizer # returns resnet18 model, cross entropy loss function, adam optimizer 

# 5 layer CNN_ 5 convolutional layers, batch normalization, maxpooling layers, fc layers and dropout layer to prevent overfitting 
class FiveLayerCNN(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.7):
        super(FiveLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # 🔹 Compute dynamically the input size for fc1
        self._to_linear = None
        self._compute_fc1_input_size()

        self.fc1 = nn.Linear(512 * 10 * 10, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def _compute_fc1_input_size(self):
        # Forward a dummy input to compute the size of the last conv layer output
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)  # Adjust 224x224 if your input size is different
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
            x = self.pool(F.relu(self.bn5(self.conv5(x))))
            self._to_linear = x.numel()  # Number of elements after flattening

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        # printing 
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# k-fold cross validation to find best hyperparameters
# loop through different learning rates, batch sizes, optimizers 
def cross_validate_model(model_class, train_loader, num_classes, device, k_folds=5, num_epochs=5):

    """
    Perform cross-validation to find optimal hyperparameters and train the model only once.
    """
    learning_rates = [1e-4, 1e-3]
    batch_sizes = [8, 16]
    optimizers = ['adam', 'sgd']

    best_accuracy = 0
    best_params = {"trained_during_cv": True}  # ✅ Prevents retraining after CV
    best_model = None  # Stores the best trained model

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for opt in optimizers:
                print(f"Testing params: LR={lr}, Batch Size={batch_size}, Optimizer={opt}")

                fold_acc = []
                kf = KFold(n_splits=k_folds, shuffle=True)

                for fold, (train_idx, val_idx) in enumerate(kf.split(train_loader.dataset)):
                    print(f"Processing Fold {fold+1}/{k_folds}...")

                    model = model_class(num_classes).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr) if opt == 'adam' else optim.SGD(model.parameters(), lr=lr, momentum=0.9)

                    train_fold = torch.utils.data.Subset(train_loader.dataset, train_idx)
                    val_fold = torch.utils.data.Subset(train_loader.dataset, val_idx)

                    train_loader_fold = torch.utils.data.DataLoader(train_fold, batch_size=batch_size, shuffle=True)
                    val_loader_fold = torch.utils.data.DataLoader(val_fold, batch_size=batch_size, shuffle=False)

                    print(f"Training model for Fold {fold+1}/{k_folds}...")
                    train_model(model, train_loader_fold, val_loader_fold, criterion, optimizer, device, num_epochs)  # ✅ Only training here

                    _, _, _, acc = evaluate_model(model, val_loader_fold, device)
                    fold_acc.append(acc)

                mean_acc = np.mean(fold_acc)
                print(f"Mean Accuracy for LR={lr}, Batch Size={batch_size}, Optimizer={opt}: {mean_acc:.2f}%")

                if mean_acc > best_accuracy:
                    best_accuracy = mean_acc
                    best_params.update({'lr': lr, 'batch_size': batch_size, 'optimizer': opt, "accuracy": mean_acc})
                    best_model = model  # ✅ Store the best trained model

    print(f"Best Hyperparameters: {best_params} with Accuracy: {best_accuracy:.2f}%")
    return best_params, best_model 
