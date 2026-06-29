"""
config.py — Central Configuration for Medical-Grade Kidney CT Pipeline
========================================================================
This module acts as the single source of truth for the entire architecture.
It manages paths, hyperparameters, model definitions, and preprocessing transforms.
"""

import os
import torch
from torchvision import transforms
try:
    import monai.transforms as mt
except ImportError:
    pass # Will be installed in requirements

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & DIRECTORY STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Datasets
RAW_DATA_DIR = os.path.join(
    PROJECT_ROOT, "archive",
    "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
    "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
)
# The deduplicated, hash-checked, StratifiedGroupKFold dataset
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset_v3")

# Model Outputs
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_v3")
SEG_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_unet_seg.pth")
CLASS_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_effnet_class.pth")

# Evaluation & Logs
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation_v3")
TENSORBOARD_DIR = os.path.join(PROJECT_ROOT, "runs_v3")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET & LEAKAGE PREVENTION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]
NUM_CLASSES = len(CLASS_NAMES)

# Hashing Settings
PHASH_HASH_SIZE = 8
PHASH_HIGHFREQ_FACTOR = 4
SIMILARITY_THRESHOLD = 5 # Hamming distance threshold for near-duplicates

# Validation Splitting
N_SPLITS = 5 # For StratifiedGroupKFold

# ─────────────────────────────────────────────────────────────────────────────
# 3. SEGMENTATION SETTINGS (MONAI U-NET)
# ─────────────────────────────────────────────────────────────────────────────
SEG_IMAGE_SIZE = (224, 224)
SEG_EPOCHS = 20
SEG_LR = 1e-3
SEG_BATCH_SIZE = 16

# MONAI Segmentation Transforms (Train)
def get_seg_train_transforms():
    return mt.Compose([
        mt.LoadImaged(keys=["image", "label"]),
        mt.EnsureChannelFirstd(keys=["image", "label"]),
        mt.Resized(keys=["image", "label"], spatial_size=SEG_IMAGE_SIZE),
        mt.ScaleIntensityd(keys=["image"]),
        mt.RandRotated(keys=["image", "label"], range_x=0.2, prob=0.5, keep_size=True),
        mt.RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
        mt.EnsureTyped(keys=["image", "label"]),
    ])

# MONAI Segmentation Transforms (Eval)
def get_seg_eval_transforms():
    return mt.Compose([
        mt.LoadImaged(keys=["image", "label"]),
        mt.EnsureChannelFirstd(keys=["image", "label"]),
        mt.Resized(keys=["image", "label"], spatial_size=SEG_IMAGE_SIZE),
        mt.ScaleIntensityd(keys=["image"]),
        mt.EnsureTyped(keys=["image", "label"]),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASSIFICATION SETTINGS (EFFICIENTNET-B0)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_IMAGE_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

CLASS_MODEL_NAME = "efficientnet_b0"
CLASS_DROPOUT_RATE = 0.4
CLASS_WEIGHT_DECAY = 1e-4
CLASS_BATCH_SIZE = 32

# Two-Phase Training
PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 20
PHASE2_LR = 1e-4

EARLY_STOP_PATIENCE = 7
GRAD_CLIP_NORM = 1.0

# Canonical Classification Transforms
CLASS_TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CLASS_IMAGE_SIZE + 32, CLASS_IMAGE_SIZE + 32)),
    transforms.RandomResizedCrop(CLASS_IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

CLASS_EVAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CLASS_IMAGE_SIZE + 32, CLASS_IMAGE_SIZE + 32)),
    transforms.CenterCrop(CLASS_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# 5. DEVICE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
