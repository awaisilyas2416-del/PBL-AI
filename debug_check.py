"""
debug_check.py -- One-shot diagnostic script.
Checks for data leakage, duplicates, label issues, and preprocessing mismatches.
"""
import os
import sys
import hashlib
from collections import defaultdict
from PIL import Image
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"c:\Users\LAPIFY\Desktop\PBL_AI\dataset"


def check_duplicates_and_leakage():
    """Hash every image and find cross-split duplicates (= data leakage)."""
    print("=" * 60)
    print("CHECK 1: DUPLICATE & DATA-LEAKAGE DETECTION")
    print("=" * 60)

    hash_to_files = defaultdict(list)
    split_hashes = defaultdict(set)

    for split in ["train", "val", "test"]:
        sp = os.path.join(BASE, split)
        for cls in sorted(os.listdir(sp)):
            cp = os.path.join(sp, cls)
            if not os.path.isdir(cp):
                continue
            for f in os.listdir(cp):
                fp = os.path.join(cp, f)
                if not os.path.isfile(fp):
                    continue
                h = hashlib.md5(open(fp, "rb").read()).hexdigest()
                hash_to_files[h].append(f"{split}/{cls}/{f}")
                split_hashes[split].add(h)

    total = len(hash_to_files)
    print(f"  Total unique images : {total}")
    for s in ["train", "val", "test"]:
        print(f"  {s:5s} unique images  : {len(split_hashes[s])}")

    train_test = split_hashes["train"] & split_hashes["test"]
    train_val = split_hashes["train"] & split_hashes["val"]
    val_test = split_hashes["val"] & split_hashes["test"]

    print()
    print(f"  LEAKAGE train<->test : {len(train_test)} duplicates")
    print(f"  LEAKAGE train<->val  : {len(train_val)} duplicates")
    print(f"  LEAKAGE val<->test   : {len(val_test)} duplicates")

    if train_test:
        print("\n  Sample train<->test leaked files:")
        for h in list(train_test)[:3]:
            for path in hash_to_files[h]:
                print(f"    {path}")
            print()

    # Within-split duplicates
    for split in ["train", "val", "test"]:
        sp = os.path.join(BASE, split)
        seen = set()
        dups = 0
        for cls in sorted(os.listdir(sp)):
            cp = os.path.join(sp, cls)
            if not os.path.isdir(cp):
                continue
            for f in os.listdir(cp):
                fp = os.path.join(cp, f)
                if not os.path.isfile(fp):
                    continue
                h = hashlib.md5(open(fp, "rb").read()).hexdigest()
                if h in seen:
                    dups += 1
                seen.add(h)
        if dups > 0:
            print(f"  WARNING: {dups} within-{split} duplicates found!")

    return len(train_test), len(train_val), len(val_test)


def check_label_consistency():
    """Verify ImageFolder label ordering and check for mislabeled dirs."""
    print("\n" + "=" * 60)
    print("CHECK 2: LABEL ORDERING & DIRECTORY VERIFICATION")
    print("=" * 60)

    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    for split in ["train", "val", "test"]:
        sp = os.path.join(BASE, split)
        ds = datasets.ImageFolder(sp, transform=tf)
        print(f"  {split:5s} class_to_idx: {ds.class_to_idx}")

    print("\n  If class_to_idx differs across splits, labels are MISALIGNED!")


def check_image_properties():
    """Sample images from each split and check mode, size, value range."""
    print("\n" + "=" * 60)
    print("CHECK 3: IMAGE PROPERTIES (mode, size, value range)")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        sp = os.path.join(BASE, split)
        classes = sorted(os.listdir(sp))
        for cls in classes[:1]:  # Check first class only for brevity
            cp = os.path.join(sp, cls)
            files = [f for f in os.listdir(cp) if os.path.isfile(os.path.join(cp, f))][:3]
            for f in files:
                fp = os.path.join(cp, f)
                img = Image.open(fp)
                arr = np.array(img)
                print(f"  {split}/{cls}/{f}: mode={img.mode}, size={img.size}, "
                      f"dtype={arr.dtype}, range=[{arr.min()}, {arr.max()}], shape={arr.shape}")


def check_filename_patterns():
    """Look for filename patterns that could cause leakage (e.g. same patient across splits)."""
    print("\n" + "=" * 60)
    print("CHECK 4: FILENAME PATTERN ANALYSIS")
    print("=" * 60)

    split_filenames = {}
    for split in ["train", "val", "test"]:
        sp = os.path.join(BASE, split)
        names = set()
        for cls in sorted(os.listdir(sp)):
            cp = os.path.join(sp, cls)
            if not os.path.isdir(cp):
                continue
            for f in os.listdir(cp):
                names.add(f)
        split_filenames[split] = names

    # Same filenames across splits
    train_test_names = split_filenames["train"] & split_filenames["test"]
    train_val_names = split_filenames["train"] & split_filenames["val"]
    print(f"  Same FILENAMES in train & test: {len(train_test_names)}")
    print(f"  Same FILENAMES in train & val:  {len(train_val_names)}")

    if train_test_names:
        print(f"  Examples: {list(train_test_names)[:5]}")

    # Show filename patterns
    for split in ["train", "test"]:
        sp = os.path.join(BASE, split)
        cls = sorted(os.listdir(sp))[0]
        cp = os.path.join(sp, cls)
        files = sorted(os.listdir(cp))[:5]
        print(f"\n  {split}/{cls} sample names: {files}")


def check_source_dataset():
    """Check the original archive directory for total image count."""
    print("\n" + "=" * 60)
    print("CHECK 5: SOURCE DATASET STATISTICS")
    print("=" * 60)

    src = r"c:\Users\LAPIFY\Desktop\PBL_AI\archive"
    if not os.path.exists(src):
        print("  Archive directory not found, skipping.")
        return

    for root, dirs, files in os.walk(src):
        img_count = sum(1 for f in files if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if img_count > 0:
            rel = os.path.relpath(root, src)
            print(f"  {rel}: {img_count} images")


if __name__ == "__main__":
    check_duplicates_and_leakage()
    check_label_consistency()
    check_image_properties()
    check_filename_patterns()
    check_source_dataset()
    print("\n" + "=" * 60)
    print("DIAGNOSTIC CHECKS COMPLETE")
    print("=" * 60)
