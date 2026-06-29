"""
classification_train.py — Task 2: Medical Classification Pipeline
====================================================================
Trains an EfficientNet-B0 on the segmented ROI outputs.
Implements Two-Phase Training, AMP, Weighted Sampler, and TensorBoard.
"""

import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from config import (
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, EVAL_DIR, TENSORBOARD_DIR,
    CLASS_MODEL_PATH, CLASS_DROPOUT_RATE, CLASS_WEIGHT_DECAY, CLASS_BATCH_SIZE,
    PHASE1_EPOCHS, PHASE1_LR, PHASE2_EPOCHS, PHASE2_LR,
    EARLY_STOP_PATIENCE, GRAD_CLIP_NORM, NUM_CLASSES,
    CLASS_TRAIN_TRANSFORM, CLASS_EVAL_TRANSFORM, get_device
)

def create_weighted_sampler(dataset):
    """Calculates weights to balance the classes and returns a WeightedRandomSampler."""
    class_counts = np.bincount(dataset.targets)
    # Weight per class is inversely proportional to its frequency
    class_weights = 1.0 / class_counts
    # Weight for each sample
    sample_weights = [class_weights[t] for t in dataset.targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler, class_weights

def build_model(device):
    """Loads EfficientNet-B0, replaces the head, and moves to device."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=CLASS_DROPOUT_RATE, inplace=True),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model.to(device)

def train_phase(model, train_loader, val_loader, criterion, optimizer, scaler, scheduler, writer, device, phase, epochs, best_val_acc, patience_counter):
    
    for epoch in range(1, epochs + 1):
        print(f"\n[Phase {phase}] Epoch {epoch}/{epochs}")
        
        # --- Training ---
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # tqdm progress bar
        train_pbar = tqdm(train_loader, desc=f"Training")
        
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # AMP (Mixed Precision)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # TensorBoard Logging
        global_epoch = epoch if phase == 1 else epoch + PHASE1_EPOCHS
        writer.add_scalar('Loss/Train', avg_train_loss, global_epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, global_epoch)
        writer.add_scalar('Accuracy/Train', train_acc, global_epoch)
        writer.add_scalar('Accuracy/Val', val_acc, global_epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_epoch)
        
        # Learning Rate Scheduler Step
        scheduler.step(avg_val_loss)
        
        # Early Stopping & Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), CLASS_MODEL_PATH)
            print(f"  [*] Saved new best model to {CLASS_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  [!] Early stopping triggered (no improvement in {EARLY_STOP_PATIENCE} epochs).")
                break
                
    return best_val_acc, patience_counter


def run_classification_pipeline():
    print("="*60)
    print("PHASE 3: EFFICIENTNET-B0 ROI CLASSIFICATION")
    print("="*60)
    
    device = get_device()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    
    # 1. Load Data
    train_dir = os.path.join(PROCESSED_DATA_DIR, "train", "images")
    val_dir = os.path.join(PROCESSED_DATA_DIR, "val", "images")
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found. Run prepare_dataset_v2.py first.")
        return
        
    train_ds = ImageFolder(train_dir, transform=CLASS_TRAIN_TRANSFORM)
    val_ds = ImageFolder(val_dir, transform=CLASS_EVAL_TRANSFORM)
    
    # 2. Weighted Sampler & Weighted Loss
    sampler, class_weights = create_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=CLASS_BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CLASS_BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Convert weights to tensor for CrossEntropyLoss
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # 3. Model & AMP setup
    model = build_model(device)
    scaler = GradScaler()
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # -------------------------------------------------------------
    # PHASE 1: FROZEN BACKBONE
    # -------------------------------------------------------------
    print("\n--- PHASE 1: Training Classifier Head Only (Frozen Backbone) ---")
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False
            
    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=PHASE1_LR, weight_decay=CLASS_WEIGHT_DECAY)
    scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_val_acc, patience_counter = train_phase(
        model, train_loader, val_loader, criterion, optimizer1, scaler, scheduler1, writer, 
        device, phase=1, epochs=PHASE1_EPOCHS, best_val_acc=best_val_acc, patience_counter=patience_counter
    )
    
    # -------------------------------------------------------------
    # PHASE 2: FINE-TUNING ENTIRE MODEL
    # -------------------------------------------------------------
    print("\n--- PHASE 2: Fine-Tuning Entire Model ---")
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer2 = torch.optim.Adam(model.parameters(), lr=PHASE2_LR, weight_decay=CLASS_WEIGHT_DECAY)
    scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Reset patience for new phase
    patience_counter = 0 
    
    train_phase(
        model, train_loader, val_loader, criterion, optimizer2, scaler, scheduler2, writer, 
        device, phase=2, epochs=PHASE2_EPOCHS, best_val_acc=best_val_acc, patience_counter=patience_counter
    )
    
    writer.close()
    print("\nClassification Training Complete.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_classification_pipeline()
