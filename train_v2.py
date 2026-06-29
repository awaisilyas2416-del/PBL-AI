"""
train_v2.py — Production-grade training pipeline for Kidney CT Classification.
===============================================================================
Fixes all bugs from the original train.py:

  ✅ 2-phase training: frozen backbone → full fine-tuning
  ✅ Mixed precision training (AMP) — auto-disabled on CPU
  ✅ Gradient clipping for stable fine-tuning
  ✅ CosineAnnealingWarmRestarts scheduler
  ✅ Class-weighted CrossEntropyLoss
  ✅ WeightedRandomSampler for balanced batches
  ✅ Proper medical-imaging augmentations (train only)
  ✅ Deterministic eval transforms (Resize+CenterCrop)
  ✅ Early stopping (patience=7, monitor val_loss)
  ✅ Model checkpoint with preprocessing metadata
  ✅ TensorBoard logging
  ✅ Grad-CAM verification after training
  ✅ Temperature scaling (post-training calibration)
  ✅ Full metrics: accuracy, precision, recall, F1, specificity,
     sensitivity, balanced accuracy, AUC-ROC, confusion matrix

Framework: PyTorch 2.x
Model: EfficientNet-B0 (upgrade path to B2/B3)
Dataset: dataset_v2/ (output of prepare_dataset_v2.py)

Usage:
    py -3 train_v2.py
"""

import os
import sys
import time
import json
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize

sys.stdout.reconfigure(encoding="utf-8")

# Import shared config — single source of truth
from config import (
    DATASET_DIR, CHECKPOINT_DIR, EVAL_DIR, TENSORBOARD_DIR,
    CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE,
    NORM_MEAN, NORM_STD,
    MODEL_NAME, DROPOUT_RATE,
    PHASE1_EPOCHS, PHASE1_LR, PHASE2_EPOCHS, PHASE2_LR,
    WEIGHT_DECAY, BATCH_SIZE, GRAD_CLIP_NORM,
    EARLY_STOP_PATIENCE, SEED,
    TRAIN_TRANSFORM, EVAL_TRANSFORM, BEST_MODEL_PATH,
    get_preprocessing_metadata, save_preprocessing_metadata, get_device,
)
from gradcam import save_gradcam_grid

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_model(freeze_backbone: bool = True) -> nn.Module:
    """
    Build EfficientNet-B0 with a custom classification head.

    Phase 1 (freeze_backbone=True):
        - All backbone parameters frozen
        - Only classifier head trains
        - Use higher LR (1e-3) since head is randomly initialized

    Phase 2 (freeze_backbone=False):
        - All parameters unfrozen
        - Use lower LR (1e-4) for fine-tuning pretrained features
        - The model learns CT-specific features instead of relying on ImageNet
    """
    # Load with pretrained ImageNet weights
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # Replace classifier head with custom head (dropout + linear)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE, inplace=True),
        nn.Linear(num_features, NUM_CLASSES),
    )

    # Freeze/unfreeze backbone
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = not freeze_backbone

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    return model


def unfreeze_backbone(model: nn.Module):
    """Unfreeze all parameters for Phase 2 fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone UNFROZEN — {trainable:,} trainable parameters")


# ─────────────────────────────────────────────────────────────────────────────
# Data Loaders with WeightedRandomSampler
# ─────────────────────────────────────────────────────────────────────────────
def create_data_loaders() -> tuple:
    """
    Create train/val/test data loaders from dataset_v2/.

    Training uses WeightedRandomSampler to handle class imbalance.
    """
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    test_dir = os.path.join(DATASET_DIR, "test")

    # Verify directories exist
    for d, name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Dataset split '{name}' not found at {d}.\n"
                f"Run prepare_dataset_v2.py first!"
            )

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORM)
    val_dataset = datasets.ImageFolder(val_dir, transform=EVAL_TRANSFORM)
    test_dataset = datasets.ImageFolder(test_dir, transform=EVAL_TRANSFORM)

    # Verify class ordering
    print(f"  Class mapping: {train_dataset.class_to_idx}")
    assert list(train_dataset.class_to_idx.keys()) == CLASS_NAMES, (
        f"Class order mismatch! Expected {CLASS_NAMES}, "
        f"got {list(train_dataset.class_to_idx.keys())}"
    )

    # ── WeightedRandomSampler for balanced training ──
    # Compute inverse-frequency weights for each sample
    class_counts = np.zeros(NUM_CLASSES, dtype=int)
    targets = np.array([s[1] for s in train_dataset.samples])
    for c in range(NUM_CLASSES):
        class_counts[c] = (targets == c).sum()

    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Also compute class weights for the loss function
    total = class_counts.sum()
    loss_weights = torch.FloatTensor(total / (NUM_CLASSES * class_counts))

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"  Class counts (train): {dict(zip(CLASS_NAMES, class_counts))}")
    print(f"  Loss weights: {dict(zip(CLASS_NAMES, loss_weights.numpy().round(3)))}")

    # Create loaders
    # num_workers=0 on Windows to avoid multiprocessing hangs
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader, loss_weights, test_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Training & Evaluation Core
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model, loader, criterion, optimizer, device, scaler, use_amp
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backward with gradient scaling (for AMP stability)
        scaler.scale(loss).backward()

        # Gradient clipping BEFORE optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:
    """
    Evaluate model on a data loader.
    Returns: (avg_loss, accuracy, all_preds, all_labels, all_probs)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return (
        avg_loss,
        accuracy,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Scaling (post-hoc calibration)
# ─────────────────────────────────────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling to calibrate model confidence.
    Medical models are often overconfident — this fixes that.

    After training, we learn a single scalar T such that:
        calibrated_probs = softmax(logits / T)

    T > 1 → softens probabilities (reduces overconfidence)
    T < 1 → sharpens probabilities
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(self, model, val_loader, device, max_iter=50):
        """Optimize temperature on validation set using NLL loss."""
        print("\n  Calibrating temperature...")
        model.eval()
        self.to(device)

        # Collect all logits and labels
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                logits_list.append(logits)
                labels_list.append(labels.to(device))

        logits_all = torch.cat(logits_list)
        labels_all = torch.cat(labels_list)

        # Optimize temperature
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits_all)
            loss = criterion(scaled, labels_all)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"  Optimal temperature: {self.temperature.item():.4f}")
        return self.temperature.item()


# ─────────────────────────────────────────────────────────────────────────────
# Save Checkpoint with Metadata
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(
    model, optimizer, scheduler, epoch, val_loss, val_acc,
    temperature, path, metadata_path
):
    """Save model checkpoint with all training state and preprocessing metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "temperature": temperature,
        "preprocessing": get_preprocessing_metadata(),
    }
    torch.save(checkpoint, path)
    save_preprocessing_metadata(metadata_path)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: dict, output_dir: str):
    """Plot loss and accuracy curves for both phases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
    if history.get("phase2_start"):
        axes[0].axvline(x=history["phase2_start"], color="g", linestyle="--",
                        label="Phase 2 Start", alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val", linewidth=2)
    if history.get("phase2_start"):
        axes[1].axvline(x=history["phase2_start"], color="g", linestyle="--",
                        label="Phase 2 Start", alpha=0.7)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("EfficientNet-B0 Training — 2-Phase Fine-Tuning",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves_v2.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(preds, labels, output_dir: str, prefix="test"):
    """Plot raw + normalized confusion matrix."""
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=axes[0], cmap="Blues", values_format="d", colorbar=False
    )
    axes[0].set_title("Raw Counts", fontweight="bold")

    ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES).plot(
        ax=axes[1], cmap="Blues", values_format=".2f", colorbar=False
    )
    axes[1].set_title("Normalized (Recall)", fontweight="bold")

    plt.suptitle(f"Confusion Matrix — {prefix.title()} Set", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, f"confusion_matrix_{prefix}_v2.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_all_metrics(preds, labels, probs, split_name="test") -> dict:
    """Compute medical-grade metrics and print a full report."""
    print(f"\n  {'='*55}")
    print(f"  METRICS — {split_name.upper()} SET")
    print(f"  {'='*55}")

    # Classification report
    report = classification_report(labels, preds, target_names=CLASS_NAMES)
    print(f"\n{report}")

    # Per-class sensitivity and specificity
    cm = confusion_matrix(labels, preds)
    print(f"  {'Class':<10} {'Sensitivity':>12} {'Specificity':>12} {'F1':>8}")
    print(f"  {'-'*44}")

    sensitivities = []
    specificities = []
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        print(f"  {cls:<10} {sensitivity:>11.4f} {specificity:>11.4f} {f1:>7.4f}")

    # Aggregate metrics
    acc = (preds == labels).mean()
    bal_acc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    macro_precision = precision_score(labels, preds, average="macro")
    macro_recall = recall_score(labels, preds, average="macro")

    # AUC-ROC
    try:
        labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
        auc = roc_auc_score(labels_bin, probs, average="macro", multi_class="ovr")
    except Exception:
        auc = 0.0

    print(f"\n  Overall Accuracy      : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Balanced Accuracy     : {bal_acc:.4f}")
    print(f"  Macro Precision       : {macro_precision:.4f}")
    print(f"  Macro Recall          : {macro_recall:.4f}")
    print(f"  Macro F1              : {macro_f1:.4f}")
    print(f"  Mean Sensitivity      : {np.mean(sensitivities):.4f}")
    print(f"  Mean Specificity      : {np.mean(specificities):.4f}")
    print(f"  Macro AUC-ROC         : {auc:.4f}")

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "mean_sensitivity": float(np.mean(sensitivities)),
        "mean_specificity": float(np.mean(specificities)),
        "auc_roc": float(auc),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("KIDNEY CT CLASSIFICATION — TRAINING PIPELINE v2")
    print("=" * 60)

    set_seed(SEED)
    device = get_device()
    use_amp = device.type == "cuda"
    if use_amp:
        print("  Mixed precision (AMP): ENABLED")
    else:
        print("  Mixed precision (AMP): disabled (CPU mode)")

    # Create output directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    train_loader, val_loader, test_loader, loss_weights, test_dataset = (
        create_data_loaders()
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[2] Building model...")
    print(f"  Architecture: {MODEL_NAME}")
    print(f"  Dropout: {DROPOUT_RATE}")

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

    # ── Training History ──────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "phase2_start": None,
    }

    # Track best model
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    global_epoch = 0
    temperature = 1.0

    # GradScaler for AMP
    scaler = torch.GradScaler(device=device.type, enabled=use_amp)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Train classifier head only (backbone frozen)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 1: Head-Only Training ({PHASE1_EPOCHS} epochs, LR={PHASE1_LR})")
    print(f"{'='*60}")

    model = build_model(freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, PHASE1_EPOCHS), T_mult=1
    )

    for epoch in range(1, PHASE1_EPOCHS + 1):
        global_epoch += 1
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"  P1 Epoch {epoch}/{PHASE1_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.0f}s"
        )

        # Log to TensorBoard
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, global_epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, global_epoch)
        writer.add_scalar("LR", lr, global_epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"    → New best (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Full fine-tuning (backbone unfrozen)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 2: Full Fine-Tuning ({PHASE2_EPOCHS} epochs, LR={PHASE2_LR})")
    print(f"{'='*60}")

    # Load best phase-1 weights before unfreezing
    if best_model_state:
        model.load_state_dict(best_model_state)

    unfreeze_backbone(model)
    history["phase2_start"] = global_epoch + 1

    # New optimizer with lower LR for fine-tuning
    optimizer = optim.AdamW(
        model.parameters(),
        lr=PHASE2_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, PHASE2_EPOCHS // 3), T_mult=2
    )

    # Reset early stopping for phase 2
    epochs_no_improve = 0

    for epoch in range(1, PHASE2_EPOCHS + 1):
        global_epoch += 1
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"  P2 Epoch {epoch}/{PHASE2_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.0f}s"
        )

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, global_epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, global_epoch)
        writer.add_scalar("LR", lr, global_epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"    → New best (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"  ⛔ Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    # ── Load best model ───────────────────────────────────────────────────
    print(f"\n  Loading best model (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")
    model.load_state_dict(best_model_state)

    # ── Temperature Scaling ───────────────────────────────────────────────
    print("\n[3] Post-training calibration...")
    temp_scaler = TemperatureScaler()
    temperature = temp_scaler.calibrate(model, val_loader, device)

    # ── Save Checkpoint ───────────────────────────────────────────────────
    print("\n[4] Saving checkpoint...")
    save_checkpoint(
        model, optimizer, scheduler, global_epoch,
        best_val_loss, best_val_acc, temperature,
        BEST_MODEL_PATH,
        os.path.join(CHECKPOINT_DIR, "preprocessing_metadata.json"),
    )
    print(f"  Saved: {BEST_MODEL_PATH}")

    # Also save a clean state_dict for easy loading in Streamlit
    clean_path = os.path.join(CHECKPOINT_DIR, "best_model_v2_statedict.pth")
    torch.save(model.state_dict(), clean_path)
    print(f"  Saved (state_dict only): {clean_path}")

    # ── Test Evaluation ───────────────────────────────────────────────────
    print("\n[5] Final evaluation on TEST set...")
    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device
    )

    metrics = compute_all_metrics(test_preds, test_labels, test_probs, "test")

    # Save metrics
    metrics_path = os.path.join(EVAL_DIR, "test_metrics_v2.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[6] Generating plots...")
    plot_training_curves(history, EVAL_DIR)
    plot_confusion_matrix(test_preds, test_labels, EVAL_DIR, "test")

    # ── Grad-CAM Verification ─────────────────────────────────────────────
    print("\n[7] Grad-CAM verification...")
    from PIL import Image as PILImage
    import random as rng

    rng.seed(SEED)
    gradcam_samples = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(DATASET_DIR, "test", cls_name)
        files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        if files:
            chosen = rng.sample(files, min(2, len(files)))
            for fname in chosen:
                img = PILImage.open(os.path.join(cls_dir, fname))
                gradcam_samples.append((img, fname, cls_idx))

    if gradcam_samples:
        os.makedirs(os.path.join(EVAL_DIR, "gradcam"), exist_ok=True)
        save_gradcam_grid(
            model, gradcam_samples, EVAL_TRANSFORM, CLASS_NAMES,
            os.path.join(EVAL_DIR, "gradcam", "test_samples_v2.png"),
            title="Grad-CAM — Does the model focus on kidney regions?",
        )

    # ── Save classification report ────────────────────────────────────────
    report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES)
    report_path = os.path.join(EVAL_DIR, "classification_report_v2.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== EfficientNet-B0 Kidney CT — v2 Evaluation Report ===\n\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Temperature: {temperature:.4f}\n\n")
        f.write(report)
    print(f"  Saved: {report_path}")

    # ── Save training history ─────────────────────────────────────────────
    history_path = os.path.join(EVAL_DIR, "training_history_v2.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer.close()

    # ── Final Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE — SUMMARY")
    print(f"{'='*60}")
    print(f"  Total epochs       : {global_epoch}")
    print(f"  Best val loss      : {best_val_loss:.4f}")
    print(f"  Best val accuracy  : {best_val_acc:.4f}")
    print(f"  Test accuracy      : {test_acc:.4f}")
    print(f"  Balanced accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1           : {metrics['macro_f1']:.4f}")
    print(f"  AUC-ROC            : {metrics['auc_roc']:.4f}")
    print(f"  Temperature        : {temperature:.4f}")
    print(f"  Checkpoint         : {BEST_MODEL_PATH}")
    print(f"  TensorBoard        : tensorboard --logdir {TENSORBOARD_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
