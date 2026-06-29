import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Training Configuration
DATA_DIR = r"c:\Users\LAPIFY\Desktop\PBL_AI\dataset"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
FREEZE_BACKBONE = True
DRY_RUN = False        # Set to True for a quick 1-epoch dry-run
EVAL_ONLY = True       # Set to True to skip training and only run test evaluation
SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _run_evaluation(model, test_loader, class_names, device):
    """Run test-set evaluation, save classification report and confusion matrix."""
    print("Evaluating best model on test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nTest Set Classification Report:")
    print(report)

    with open('classification_report.txt', 'w') as f:
        f.write("=== EfficientNet-B0 Kidney Tumor Classification Report ===\n\n")
        f.write(report)
    print("Saved classification report to 'classification_report.txt'.")

    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title('Confusion Matrix on Test Set')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved confusion matrix to 'confusion_matrix.png'.")


def main():
    set_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Data Transforms
    # EfficientNet standard normalization
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    # 2. Data Loaders
    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_test_transform)
    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_test_transform)
    
    # num_workers=0 is required on Windows to avoid DataLoader multiprocessing hangs
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    class_names = train_set.classes
    print(f"Classes found: {class_names}")
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}, Test size: {len(test_set)}")
    
    # 3. Model Setup
    print("Loading pretrained EfficientNet-B0...")
    # Load modern weights structure
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    if FREEZE_BACKBONE:
        print("Freezing EfficientNet-B0 backbone parameters...")
        for param in model.parameters():
            param.requires_grad = False
            
    # Replace the classification head
    # EfficientNet-B0 has model.classifier as Sequence: (Dropout, Linear)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    model = model.to(device)
    
    # 4. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Load existing best model if EVAL_ONLY is set
    if EVAL_ONLY and os.path.exists('best_model.pth'):
        print("EVAL_ONLY mode: Loading best_model.pth and skipping training...")
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        # Jump directly to test evaluation
        _run_evaluation(model, test_loader, class_names, device)
        return
    
    # 5. Training Loop
    epochs_to_run = 1 if DRY_RUN else EPOCHS
    print(f"Starting training for {epochs_to_run} epochs...")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    early_stop_patience = 3
    epochs_no_improve = 0
    
    for epoch in range(1, epochs_to_run + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_set)
        epoch_acc = (running_corrects.double() / len(train_set)).item()
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / len(val_set)
        epoch_val_acc = (val_corrects.double() / len(val_set)).item()
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch}/{epochs_to_run} - "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        # Checkpoint Saving & Early Stopping
        torch.save(model.state_dict(), 'latest_model.pth')
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("  --> Validation loss decreased. Saved as best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience and not DRY_RUN:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    print("Training finished.")
    
    # 6. Plot Loss and Accuracy Curves
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Train')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['train_acc']) + 1), history['train_acc'], label='Train')
    plt.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_accuracy_curves.png', dpi=300)
    plt.close()
    print("Saved training curves to 'loss_accuracy_curves.png'.")
    
    # 7. Final Evaluation on Test Set using Best Model
    print("Evaluating best model on test set...")
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nTest Set Classification Report:")
    print(report)
    
    with open('classification_report.txt', 'w') as f:
        f.write("=== EfficientNet-B0 Kidney Tumor Classification Report ===\n\n")
        f.write(report)
    print("Saved classification report to 'classification_report.txt'.")
    
    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title('Confusion Matrix on Test Set')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved confusion matrix to 'confusion_matrix.png'.")

if __name__ == "__main__":
    main()
