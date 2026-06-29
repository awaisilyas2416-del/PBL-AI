"""
evaluate.py -- Comprehensive Diagnostic & Cross-Validation for EfficientNet-B0
Checks:
  1. Overfitting/Underfitting: Compare accuracy on Train vs Val vs Test
  2. Detailed Confusion Matrix: Normalized + raw, per-class analysis + AUC-ROC
  3. 5-Fold Stratified Cross-Validation: Retrain 5 times to verify accuracy
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

# --- Config ------------------------------------------------------------------
DATA_DIR   = r"c:\Users\LAPIFY\Desktop\PBL_AI\dataset"
MODEL_PATH = r"c:\Users\LAPIFY\Desktop\PBL_AI\best_model.pth"
OUT_DIR    = r"c:\Users\LAPIFY\Desktop\PBL_AI\evaluation"
BATCH_SIZE = 32
SEED       = 42
N_FOLDS    = 5
CV_EPOCHS  = 5
LR         = 0.001
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
N_CLASSES   = len(CLASS_NAMES)
# -----------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_tf, eval_tf

def build_model(device, load_path=None):
    weights = models.EfficientNet_B0_Weights.DEFAULT if load_path is None else None
    model = models.efficientnet_b0(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, N_CLASSES)
    if load_path and os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"  Loaded weights from {load_path}")
    return model.to(device)

def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def acc(preds, labels):
    return (preds == labels).mean() * 100

# =============================================================================
# PART 1: Overfitting / Underfitting Check
# =============================================================================
def check_overfit(device):
    print("\n" + "="*60)
    print("PART 1: OVERFITTING / UNDERFITTING CHECK")
    print("="*60)
    _, eval_tf = get_transforms()

    splits = {
        'Train': os.path.join(DATA_DIR, 'train'),
        'Val':   os.path.join(DATA_DIR, 'val'),
        'Test':  os.path.join(DATA_DIR, 'test'),
    }
    loaders = {
        k: DataLoader(datasets.ImageFolder(v, transform=eval_tf),
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        for k, v in splits.items()
    }

    model = build_model(device, MODEL_PATH)

    results = {}
    for split, loader in loaders.items():
        preds, labels, probs = get_predictions(model, loader, device)
        results[split] = dict(preds=preds, labels=labels, probs=probs,
                               acc=acc(preds, labels))
        print(f"  {split:<6}: {results[split]['acc']:.2f}%")

    train_acc = results['Train']['acc']
    val_acc   = results['Val']['acc']
    test_acc  = results['Test']['acc']
    gap_tv    = train_acc - val_acc
    gap_tt    = train_acc - test_acc

    print(f"\n  Train - Val  gap : {gap_tv:+.2f}%")
    print(f"  Train - Test gap : {gap_tt:+.2f}%")

    if gap_tv < 3:
        verdict = "[OK] NO OVERFITTING -- Train and Val accuracy are very close."
    elif gap_tv < 8:
        verdict = "[WARN] MILD OVERFITTING -- Small gap, still acceptable."
    else:
        verdict = "[ERR] OVERFITTING DETECTED -- Large Train-Val gap."

    if max(train_acc, val_acc) < 70:
        verdict += " [WARN] UNDERFITTING -- Accuracy too low."

    print(f"\n  VERDICT: {verdict}")

    # Confidence distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (split, r) in zip(axes, results.items()):
        max_conf = r['probs'].max(axis=1)
        correct  = (r['probs'].argmax(axis=1) == r['labels'])
        ax.hist(max_conf[correct],  bins=20, alpha=0.75, color='steelblue', label='Correct')
        ax.hist(max_conf[~correct], bins=20, alpha=0.75, color='tomato',   label='Wrong')
        ax.set_title(f'{split} Confidence\n(Acc={r["acc"]:.1f}%)', fontsize=11)
        ax.set_xlabel('Max Softmax Confidence')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Prediction Confidence: Correct vs Wrong predictions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'confidence_distribution.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\n  Saved: {out_path}")

    return (train_acc, val_acc, test_acc,
            results['Test']['preds'], results['Test']['labels'], results['Test']['probs'])


# =============================================================================
# PART 2: Detailed Confusion Matrix Analysis
# =============================================================================
def detailed_confusion_matrix(test_preds, test_labels, test_probs):
    print("\n" + "="*60)
    print("PART 2: DETAILED CONFUSION MATRIX ANALYSIS")
    print("="*60)

    cm      = confusion_matrix(test_labels, test_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=axes[0], cmap='Blues', values_format='d', colorbar=False)
    axes[0].set_title('Raw Counts', fontsize=12, fontweight='bold')

    ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES).plot(
        ax=axes[1], cmap='Blues', values_format='.2f', colorbar=False)
    axes[1].set_title('Normalized (Recall per class)', fontsize=12, fontweight='bold')

    plt.suptitle('EfficientNet-B0 -- Test Set Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'confusion_matrix_detailed.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")

    # Per-class TP/FP/FN
    print(f"\n  {'Class':<10} {'TP':>5} {'FP':>5} {'FN':>5} {'Recall':>9} {'Prec':>9} {'F1':>9}")
    print(f"  {'-'*55}")
    f1_scores = []
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)
        print(f"  {cls:<10} {tp:>5} {fp:>5} {fn:>5} {rec:>8.1%} {prec:>8.1%} {f1:>8.1%}")

    print(f"\n  Macro F1: {np.mean(f1_scores):.4f}")

    # AUC-ROC
    try:
        labels_bin = label_binarize(test_labels, classes=list(range(N_CLASSES)))
        auc = roc_auc_score(labels_bin, test_probs, average='macro', multi_class='ovr')
        print(f"  Macro AUC-ROC: {auc:.4f}")
    except Exception as e:
        print(f"  AUC-ROC skipped: {e}")

    report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES)
    print(f"\n  Full Classification Report:\n{report}")

    # Save report
    with open(os.path.join(OUT_DIR, 'classification_report_detailed.txt'), 'w', encoding='utf-8') as f:
        f.write("=== EfficientNet-B0 Kidney Tumor -- Detailed Evaluation ===\n\n")
        f.write(f"Raw Confusion Matrix:\n{cm}\n\n")
        f.write(f"Normalized:\n{np.round(cm_norm, 4)}\n\n")
        f.write(report)


# =============================================================================
# PART 3: 5-Fold Stratified Cross-Validation
# =============================================================================
def cross_validate(device):
    print("\n" + "="*60)
    print(f"PART 3: {N_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION")
    print("="*60)
    print(f"  Training {N_FOLDS} folds x {CV_EPOCHS} epochs each (frozen backbone)...")

    _, eval_tf = get_transforms()

    full_train = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=eval_tf)
    full_val   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   transform=eval_tf)
    full_test  = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),  transform=eval_tf)
    full_set   = ConcatDataset([full_train, full_val, full_test])

    all_labels = np.array(
        [s[1] for s in full_train.samples] +
        [s[1] for s in full_val.samples]   +
        [s[1] for s in full_test.samples]
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), 1):
        print(f"\n  -- Fold {fold}/{N_FOLDS} ----------------------------------------")
        t0 = time.time()

        train_loader = DataLoader(Subset(full_set, train_idx),
                                  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(Subset(full_set, val_idx),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model     = build_model(device, load_path=None)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        criterion = nn.CrossEntropyLoss()

        best_val_acc    = 0.0
        best_preds      = None
        best_labels_val = None

        for epoch in range(1, CV_EPOCHS + 1):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                criterion(model(inputs), labels).backward()
                optimizer.step()

            preds, labels_f, _ = get_predictions(model, val_loader, device)
            val_a = acc(preds, labels_f)
            print(f"    Epoch {epoch}/{CV_EPOCHS}  Val Acc: {val_a:.2f}%")

            if val_a > best_val_acc:
                best_val_acc    = val_a
                best_preds      = preds
                best_labels_val = labels_f

        elapsed = time.time() - t0
        fold_results.append({'fold': fold, 'val_acc': best_val_acc, 'time': elapsed})
        print(f"  Fold {fold} best Val Acc: {best_val_acc:.2f}%  ({elapsed/60:.1f} min)")

    # Summary
    accs = [r['val_acc'] for r in fold_results]
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(f"\n  {'Fold':<8} {'Val Acc':>10}")
    print(f"  {'-'*20}")
    for r in fold_results:
        print(f"  {r['fold']:<8} {r['val_acc']:>9.2f}%")
    print(f"  {'-'*20}")
    print(f"  {'Mean':<8} {np.mean(accs):>9.2f}%")
    print(f"  {'Std':<8} {np.std(accs):>9.2f}%")
    print(f"  {'Min':<8} {np.min(accs):>9.2f}%")
    print(f"  {'Max':<8} {np.max(accs):>9.2f}%")

    std = np.std(accs)
    mean = np.mean(accs)
    stability = "[OK] STABLE (std < 3%)" if std < 3 else "[WARN] HIGH VARIANCE (std >= 3%)"
    confidence = "[OK] CV confirms high accuracy -- result is genuine." if mean > 90 \
        else ("[MODERATE] CV shows moderate accuracy." if mean > 75 \
        else "[WARN] CV accuracy is LOW -- test result may be inflated.")

    print(f"\n  Stability : {stability}")
    print(f"  Verdict   : {confidence}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = [r['fold'] for r in fold_results]
    ax.bar(folds, accs, color='steelblue', alpha=0.85, edgecolor='navy', linewidth=0.8)
    ax.axhline(mean, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {mean:.2f}%')
    ax.fill_between([0.5, N_FOLDS + 0.5], mean - std, mean + std,
                    alpha=0.15, color='red', label=f'+/-1 Std = {std:.2f}%')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title(f'{N_FOLDS}-Fold Cross-Validation -- EfficientNet-B0',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(folds)
    ax.set_ylim(max(0, min(accs) - 5), 101)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'cross_validation_results.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\n  Saved: {out_path}")

    with open(os.path.join(OUT_DIR, 'cross_validation_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"=== {N_FOLDS}-Fold Cross-Validation Report ===\n\n")
        for r in fold_results:
            f.write(f"Fold {r['fold']}: Val Acc = {r['val_acc']:.2f}%  ({r['time']/60:.1f} min)\n")
        f.write(f"\nMean: {mean:.2f}%\nStd:  {std:.2f}%\n"
                f"Min:  {np.min(accs):.2f}%\nMax:  {np.max(accs):.2f}%\n")

    return mean, std


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output dir: {OUT_DIR}")

    train_acc, val_acc, test_acc, test_preds, test_labels, test_probs = check_overfit(device)
    detailed_confusion_matrix(test_preds, test_labels, test_probs)
    cv_mean, cv_std = cross_validate(device)

    print("\n" + "="*60)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"  Train Accuracy  : {train_acc:.2f}%")
    print(f"  Val   Accuracy  : {val_acc:.2f}%")
    print(f"  Test  Accuracy  : {test_acc:.2f}%")
    print(f"  CV Mean Acc     : {cv_mean:.2f}% +/- {cv_std:.2f}%")
    gap = train_acc - test_acc
    verdict = "[OK] Model generalizes well -- no overfitting." if abs(gap) < 5 \
        else "[WARN] Possible overfitting -- train/test gap is large."
    print(f"  Overfit Gap     : {gap:.2f}%  {verdict}")
    print(f"\nAll outputs saved to: {OUT_DIR}")
