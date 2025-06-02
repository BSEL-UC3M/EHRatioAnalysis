import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F

class FiveLayerCNN(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(FiveLayerCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_prob),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_prob),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_prob),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Automatically determine fc input size
        self.fc_input_size = self._compute_fc_input_size()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, num_classes)
        )

    def _compute_fc_input_size(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 320, 320) # Claudia's version: (1, 3, 224, 224) - why
            x = self.conv_layers(x)
            flattened_size = x.view(1, -1).size(1)
            print(f"🧩 FC input size: {flattened_size}")
            return flattened_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x


from tqdm.auto import tqdm
import sys

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                num_epochs=30, update_freq=10, early_stop_patience=5):
    """
    Train the CNN model on MRC images and track performance with early stopping.
    Stops training if validation loss doesn't improve for 'early_stop_patience' epochs.
    """
    model.train()
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    early_stop_counter = 0

    print(f"\n🚀 Training started for {num_epochs} epochs...\n")

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

        print(f"\n✅ Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%\n")

        scheduler.step(avg_val_loss)
        model.train()

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs.")
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
    
    return y_true, y_pred, avg_loss, accuracy

