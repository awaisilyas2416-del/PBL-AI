"""
evaluate_v2.py — Task 7: Medical Evaluation
================================================
Evaluates the fine-tuned classification model on the test set.
Computes FPR, FNR, Sensitivity, Specificity, and Calibration metrics.
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from classification_train import build_model
from config import (
    PROCESSED_DATA_DIR, CLASS_MODEL_PATH, EVAL_DIR,
    CLASS_BATCH_SIZE, CLASS_NAMES, NUM_CLASSES,
    CLASS_EVAL_TRANSFORM, get_device
)

def evaluate_model():
    print("="*60)
    print("PHASE 4: MEDICAL EVALUATION (CLASSIFICATION)")
    print("="*60)
    
    device = get_device()
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    test_dir = os.path.join(PROCESSED_DATA_DIR, "test", "images")
    if not os.path.exists(test_dir):
        print("Error: Test directory not found. Run prepare_dataset_v2.py first.")
        return
        
    test_ds = ImageFolder(test_dir, transform=CLASS_EVAL_TRANSFORM)
    test_loader = DataLoader(test_ds, batch_size=CLASS_BATCH_SIZE, shuffle=False)
    
    model = build_model(device)
    if os.path.exists(CLASS_MODEL_PATH):
        model.load_state_dict(torch.load(CLASS_MODEL_PATH, map_location=device))
        print("Loaded best model weights.")
    else:
        print("Warning: No trained model weights found! Using random initialization.")
        
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Calculate Base Metrics
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds)
    
    # Medical Metrics (Sensitivity, Specificity, FPR, FNR)
    medical_metrics = {}
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        medical_metrics[cls] = {
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "FPR": fpr,
            "FNR": fnr
        }
        
    # Calibration & AUC
    y_true_onehot = np.eye(NUM_CLASSES)[all_targets]
    all_probs = np.array(all_probs)
    auc_scores = {}
    for i, cls in enumerate(CLASS_NAMES):
        try:
            auc = roc_auc_score(y_true_onehot[:, i], all_probs[:, i])
            auc_scores[cls] = auc
        except ValueError:
            auc_scores[cls] = 0.5
            
    # Save results
    results = {
        "classification_report": report,
        "medical_metrics": medical_metrics,
        "auc_scores": auc_scores,
        "confusion_matrix": cm.tolist()
    }
    
    results_path = os.path.join(EVAL_DIR, "medical_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nEvaluation complete. Results saved to {results_path}")
    
    for cls in CLASS_NAMES:
        print(f"\n--- {cls} ---")
        print(f"  Sensitivity (Recall): {medical_metrics[cls]['Sensitivity']:.4f}")
        print(f"  Specificity:          {medical_metrics[cls]['Specificity']:.4f}")
        print(f"  False Negative Rate:  {medical_metrics[cls]['FNR']:.4f}")
        print(f"  ROC-AUC:              {auc_scores[cls]:.4f}")

if __name__ == "__main__":
    evaluate_model()
