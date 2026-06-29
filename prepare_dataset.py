import os
import shutil
import random
from collections import Counter

def prepare_dataset(src_dir, dest_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    random.seed(seed)
    
    # Validation of ratios
    if not (0.99 <= train_ratio + val_ratio + test_ratio <= 1.01):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Locate classes
    classes = ["Normal", "Cyst", "Stone", "Tumor"]
    class_dirs = {}
    for c in classes:
        # Check standard casing or check whatever subdirectories exist matching name case-insensitively
        found_dir = None
        for name in os.listdir(src_dir):
            if name.lower() == c.lower() and os.path.isdir(os.path.join(src_dir, name)):
                found_dir = os.path.join(src_dir, name)
                break
        if not found_dir:
            raise FileNotFoundError(f"Could not find directory for class '{c}' in '{src_dir}'")
        class_dirs[c] = found_dir

    # Gather all file paths for each class
    class_files = {}
    for c, d in class_dirs.items():
        files = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        # Filter for common image extensions
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.dcm'}
        files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]
        class_files[c] = files
        print(f"Class '{c}': found {len(files)} files.")

    # Determine minimum count for balancing
    min_files = min(len(files) for files in class_files.values())
    print(f"\nBalancing classes to match the smallest class size: {min_files} images per class.")

    # Sample and split
    split_summary = {c: {"train": 0, "val": 0, "test": 0} for c in classes}
    
    for c in classes:
        files = class_files[c]
        # Shuffle randomly using the set seed
        random.shuffle(files)
        # Select the balanced subset
        balanced_subset = files[:min_files]
        
        # Calculate split sizes
        n_total = len(balanced_subset)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val # Ensure all files are accounted for
        
        train_files = balanced_subset[:n_train]
        val_files = balanced_subset[n_train:n_train+n_val]
        test_files = balanced_subset[n_train+n_val:]
        
        # Create output directories and copy files
        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        
        for split_name, split_list in splits.items():
            split_dir = os.path.join(dest_dir, split_name, c)
            os.makedirs(split_dir, exist_ok=True)
            for f in split_list:
                shutil.copy2(f, split_dir)
            split_summary[c][split_name] = len(split_list)

    # Print summary report
    print("\n" + "="*50)
    print("DATASET SPLIT AND BALANCE SUMMARY")
    print("="*50)
    print(f"{'Class':<12} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-"*50)
    for c in classes:
        tr = split_summary[c]["train"]
        va = split_summary[c]["val"]
        te = split_summary[c]["test"]
        tot = tr + va + te
        print(f"{c:<12} | {tr:<8} | {va:<8} | {te:<8} | {tot:<8}")
    print("="*50)
    print(f"Dataset successfully prepared at: {os.path.abspath(dest_dir)}")

if __name__ == "__main__":
    SRC = r"c:\Users\LAPIFY\Desktop\PBL_AI\archive\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    DEST = r"c:\Users\LAPIFY\Desktop\PBL_AI\dataset"
    prepare_dataset(SRC, DEST)
