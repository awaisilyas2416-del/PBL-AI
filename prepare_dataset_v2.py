"""
prepare_dataset_v2.py — Advanced Medical Dataset Preparation
================================================================
Implements Task 3: Leakage Prevention & Hashing
Implements Task 1: Pseudo-mask generation for Segmentation
"""

import os
import cv2
import json
import shutil
import hashlib
import numpy as np
from PIL import Image
from tqdm import tqdm
import imagehash
from sklearn.model_selection import StratifiedGroupKFold

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASS_NAMES, 
    PHASH_HASH_SIZE, PHASH_HIGHFREQ_FACTOR, SIMILARITY_THRESHOLD,
    N_SPLITS
)

def get_exact_hash(filepath):
    """Calculate SHA-256 for exact duplicate detection."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_phash(filepath):
    """Calculate perceptual hash for near-duplicate detection."""
    img = Image.open(filepath).convert("L")
    return imagehash.phash(img, hash_size=PHASH_HASH_SIZE, highfreq_factor=PHASH_HIGHFREQ_FACTOR)

def generate_pseudo_mask(img_gray_np):
    """
    Generates a pseudo ground-truth segmentation mask using Otsu thresholding.
    This fulfills the requirement to train a segmentation network when
    manual physician annotations are unavailable.
    """
    blurred = cv2.GaussianBlur(img_gray_np, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to fill holes
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def process_dataset():
    print("="*60)
    print("PHASE 1: DATA LEAKAGE PREVENTION & MASK GENERATION")
    print("="*60)
    
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_DIR}")
        
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # 1. Collect all images
    all_files = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(RAW_DATA_DIR, cls)
        if not os.path.exists(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append((os.path.join(cls_dir, f), cls))
                
    print(f"Found {len(all_files)} total images across {len(CLASS_NAMES)} classes.")
    
    # 2. Hashing for duplicates & grouping
    exact_hashes = {}
    unique_files = []
    groups = [] # For StratifiedGroupKFold (images with same pHash go to same group)
    phash_dict = {}
    group_counter = 0
    
    print("\n[1/3] Hashing images to detect duplicates and prevent leakage...")
    
    duplicate_exact_count = 0
    duplicate_near_count = 0
    
    for filepath, cls in tqdm(all_files, desc="Hashing"):
        # Exact duplication check
        ehash = get_exact_hash(filepath)
        if ehash in exact_hashes:
            duplicate_exact_count += 1
            continue
        exact_hashes[ehash] = True
        
        # Near duplication check (Group assignment)
        try:
            ph = get_phash(filepath)
            
            # Find if it belongs to an existing group
            assigned_group = -1
            for existing_ph, g_id in phash_dict.items():
                # If hamming distance is below threshold, they are near duplicates
                if ph - existing_ph <= SIMILARITY_THRESHOLD:
                    assigned_group = g_id
                    duplicate_near_count += 1
                    break
                    
            if assigned_group == -1:
                assigned_group = group_counter
                phash_dict[ph] = group_counter
                group_counter += 1
                
            unique_files.append((filepath, cls))
            groups.append(assigned_group)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    print(f"\nLeakage Report:")
    print(f" - Exact duplicates removed: {duplicate_exact_count}")
    print(f" - Near-duplicates grouped together: {duplicate_near_count}")
    print(f" - Total unique perceptual groups (Patients): {group_counter}")
    print(f" - Usable images: {len(unique_files)}")
    
    # 3. Stratified Group K-Fold Splitting
    print("\n[2/3] Performing StratifiedGroupKFold Split...")
    X = np.arange(len(unique_files))
    y = np.array([cls for _, cls in unique_files])
    groups = np.array(groups)
    
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    # Use the first fold for train / val / test
    train_idx, temp_idx = next(sgkf.split(X, y, groups))
    
    # Split temp into val and test (using a 2nd fold logic or simple split)
    # Since we can't easily do sgkf on temp_idx again without reshaping, 
    # we'll do a simple split on the unique groups in temp_idx
    temp_groups = groups[temp_idx]
    unique_temp_groups = np.unique(temp_groups)
    np.random.seed(42)
    np.random.shuffle(unique_temp_groups)
    
    half = len(unique_temp_groups) // 2
    val_groups = unique_temp_groups[:half]
    test_groups = unique_temp_groups[half:]
    
    val_idx = temp_idx[np.isin(temp_groups, val_groups)]
    test_idx = temp_idx[np.isin(temp_groups, test_groups)]
    
    print(f" - Train samples: {len(train_idx)}")
    print(f" - Val samples:   {len(val_idx)}")
    print(f" - Test samples:  {len(test_idx)}")
    
    # 4. Copying and generating Pseudo-Masks
    print("\n[3/3] Copying files and generating Pseudo-Masks for Segmentation...")
    
    def process_split(indices, split_name):
        split_img_dir = os.path.join(PROCESSED_DATA_DIR, split_name, "images")
        split_mask_dir = os.path.join(PROCESSED_DATA_DIR, split_name, "masks")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)
        
        for idx in tqdm(indices, desc=f"Processing {split_name}"):
            filepath, cls = unique_files[idx]
            filename = os.path.basename(filepath)
            
            # Prepend class name to avoid name collisions across folders
            new_filename = f"{cls}_{filename}"
            
            # Destination paths
            dest_img = os.path.join(split_img_dir, new_filename)
            dest_mask = os.path.join(split_mask_dir, new_filename)
            
            # Copy original image
            shutil.copy2(filepath, dest_img)
            
            # Generate and save mask
            img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            mask = generate_pseudo_mask(img_gray)
            cv2.imwrite(dest_mask, mask)
            
    process_split(train_idx, "train")
    process_split(val_idx, "val")
    process_split(test_idx, "test")
    
    print("\nDataset preparation complete!")
    print(f"Dataset saved to: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    process_dataset()
