# ==============================================================================
# File: trainer.py
# Description: Shared trainer and evaluator for all classification models (ResNet, custom CNN)
# Author: Caterina Fuster-Barceló (refactored by ChatGPT)
# Created: 02/04/2025
# ==============================================================================

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                num_epochs=10, update_freq=10, early_stop_patience=10, model_type="custom"):
    """
    Generic training function for any classification model with early stopping and learning rate scheduling.
    """
    model.train()
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_loss = float('inf')
    early_stop_counter = 0

    print(f"\n🚀 Training {model_type} started for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train, total_train = 0, 0

        progress_bar = tqdm(total=len(train_loader),
                            desc=f"Epoch {epoch+1}/{num_epochs}",
                            unit="batch",
                            leave=True,
                            dynamic_ncols=True)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
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

            if (batch_idx + 1) % update_freq == 0 or batch_idx == len(train_loader) - 1:
                progress_bar.update(update_freq)
                progress_bar.set_postfix({
                    "Loss": f"{running_loss / (batch_idx + 1):.4f}",
                    "Acc": f"{(100 * correct_train / total_train):.2f}%"
                })

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation
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

        print(f"\n✅ Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%\n")

        scheduler.step(avg_val_loss)
        model.train()

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("\n🛑 Early stopping triggered.")
                break

    print("🎉 Training complete!")
    return model, train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Evaluate a classification model and return predictions, average loss, and accuracy.
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

            probs = torch.softmax(outputs, dim=1)
            biased_preds = (probs[:, 1] > threshold).long()  # ⚠️ Biased prediction in favour of class 1
            
            # _, predicted = torch.max(outputs, 1)
            # total_correct += (predicted == labels).sum().item()
            # total_samples += labels.size(0)
            total_samples += labels.size(0)
            total_correct += (biased_preds == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            # y_pred.extend(predicted.cpu().numpy())
            y_pred.extend(biased_preds.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * total_correct / total_samples
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}% (Threshold={threshold})")


    return y_true, y_pred, avg_loss, accuracy