"""
segmentation_train.py — Task 1: Medical Segmentation Pipeline (MONAI)
======================================================================
Trains a MONAI U-Net on the Kidney CT Dataset to isolate the Kidney ROI.
Outputs the best model weights and evaluation metrics.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.data import DataLoader, Dataset
from config import (
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, EVAL_DIR, 
    SEG_MODEL_PATH, SEG_EPOCHS, SEG_LR, SEG_BATCH_SIZE,
    get_seg_train_transforms, get_seg_eval_transforms, get_device
)

def get_data_dicts(split="train"):
    """Prepare list of dicts for MONAI Dataset (image + label paths)."""
    img_dir = os.path.join(PROCESSED_DATA_DIR, split, "images")
    mask_dir = os.path.join(PROCESSED_DATA_DIR, split, "masks")
    
    if not os.path.exists(img_dir):
        return []
        
    data_dicts = []
    for f in os.listdir(img_dir):
        img_path = os.path.join(img_dir, f)
        mask_path = os.path.join(mask_dir, f)
        if os.path.exists(mask_path):
            data_dicts.append({"image": img_path, "label": mask_path})
    return data_dicts

def train_segmentation():
    print("="*60)
    print("PHASE 2: U-NET SEGMENTATION TRAINING (MONAI)")
    print("="*60)
    
    device = get_device()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    # 1. Load Datasets
    train_files = get_data_dicts("train")
    val_files = get_data_dicts("val")
    
    if len(train_files) == 0:
        print("Error: No training data found. Please run prepare_dataset_v2.py first.")
        return
        
    print(f"Training Samples: {len(train_files)} | Validation Samples: {len(val_files)}")
    
    train_ds = Dataset(data=train_files, transform=get_seg_train_transforms())
    train_loader = DataLoader(train_ds, batch_size=SEG_BATCH_SIZE, shuffle=True, num_workers=4)
    
    val_ds = Dataset(data=val_files, transform=get_seg_eval_transforms())
    val_loader = DataLoader(val_ds, batch_size=SEG_BATCH_SIZE, num_workers=4)
    
    # 2. Define Model, Loss, Optimizer
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=SEG_LR)
    
    # Metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)
    
    # 3. Training Loop
    best_metric = -1
    
    for epoch in range(1, SEG_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{SEG_EPOCHS}")
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in tqdm(train_loader, desc="Training"):
            step += 1
            inputs = batch_data["image"].to(device)
            # pseudo masks are 0-255 initially, ensure 0-1
            labels = batch_data["label"].to(device)
            labels = (labels > 0).float() 
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= step
        print(f"  Train Dice Loss: {epoch_loss:.4f}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            for val_data in tqdm(val_loader, desc="Validation"):
                val_inputs = val_data["image"].to(device)
                val_labels = (val_data["label"].to(device) > 0).float()
                
                val_outputs = model(val_inputs)
                
                # Binarize output for metric calculation
                val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()
                
                dice_metric(y_pred=val_outputs, y=val_labels)
                iou_metric(y_pred=val_outputs, y=val_labels)
                # Hausdorff requires batched processing
                hausdorff_metric(y_pred=val_outputs, y=val_labels)
                
            # Aggregate metrics
            metric_dice = dice_metric.aggregate().item()
            metric_iou = iou_metric.aggregate().item()
            metric_hd = hausdorff_metric.aggregate().item()
            
            dice_metric.reset()
            iou_metric.reset()
            hausdorff_metric.reset()
            
            print(f"  Val Dice: {metric_dice:.4f} | Val IoU: {metric_iou:.4f} | Val HD95: {metric_hd:.4f}")
            
            if metric_dice > best_metric:
                best_metric = metric_dice
                torch.save(model.state_dict(), SEG_MODEL_PATH)
                print(f"  [*] Saved new best model to {SEG_MODEL_PATH}")

    print("\nSegmentation Training Complete.")

if __name__ == "__main__":
    # Ensure multiprocessing works on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    train_segmentation()
