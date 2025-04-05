import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

def fine_tune_resnet(num_classes, device, learning_rate=0.0001, model_type='resnet50', weight_decay=1e-4):
    """
    Fine-tune ResNet model (ResNet18 or ResNet50) with dropout and L2 regularization.
    """
    if model_type == 'resnet50':
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    num_ftrs = resnet.fc.in_features
    
    # Modify fully connected layer: add dropout (0.5 initially)
    resnet.fc_layers = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    resnet = resnet.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)  # L2 Regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # Removed verbose=True

    
    return resnet, criterion, optimizer, scheduler

from tqdm import tqdm
import sys

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20, early_stop_patience=5):
    """
    Train the ResNet model with a progress bar and regularization techniques.
    """
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    print(f"\n🚀 Training started for {num_epochs} epochs...\n")  # ✅ Startup message

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train, total_train = 0, 0

        # ✅ Create Progress Bar for Training Batches
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=True)

        # ✅ Adjust Dropout after epoch 10
        if epoch == 10:  # Only modify once at epoch 10
            model.fc_layers = nn.Sequential(
                nn.Dropout(0.6),  # ✅ Correct way to modify dropout dynamically
                nn.Linear(model.fc_layers[1].in_features, 2)
            ).to(device)

        model.train()

        for batch_idx, (inputs, labels) in enumerate(progress_bar):
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

            # ✅ Update tqdm Progress Bar with Live Metrics
            progress_bar.set_postfix({
                "Loss": f"{running_loss / (batch_idx + 1):.4f}",
                "Acc": f"{(100 * correct_train / total_train):.2f}%"
            })

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # ✅ Validation Step
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

        # ✅ Print Summary per Epoch
        print(f"\n✅ Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%\n")

        scheduler.step(avg_val_loss)

        # ✅ Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0  # Reset counter
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("🛑 Early stopping triggered.")
                break

    print("🎉 Training complete!")
    return model, train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data and return predictions for confusion matrix.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * total_correct / total_samples
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    conf_matrix=confusion_matrix(y_true, y_pred)
    return avg_loss, accuracy, conf_matrix
