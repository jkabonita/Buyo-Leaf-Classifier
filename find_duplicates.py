"""
Quick Duplicate Finder
======================
Finds exact duplicate images across classes in dataset
Much faster than full diagnostic
"""

import os
import hashlib
from collections import defaultdict

CLASS_NAMES = ['class_a', 'class_b', 'class_c', 'class_d', 'class_e']
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/val'

def file_hash(filepath):
    """Compute MD5 hash of file"""
    try:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def find_duplicates_in_split(split_dir, split_name):
    """Find duplicates in train or val split"""
    print(f"\n{'='*80}")
    print(f"  {split_name.upper()} SET")
    print(f"{'='*80}")
    
    file_hashes = defaultdict(list)
    
    # Scan all images
    for class_name in CLASS_NAMES:
        class_path = os.path.join(split_dir, class_name)
        if not os.path.exists(class_path):
            continue
        
        files = [f for f in os.listdir(class_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"\n  Scanning {class_name.upper()}: {len(files)} images...", end=" ")
        
        for fname in files:
            fpath = os.path.join(class_path, fname)
            fhash = file_hash(fpath)
            if fhash:
                file_hashes[fhash].append((class_name, fname, fpath))
        
        print("✓")
    
    # Find duplicates
    duplicates = {h: files for h, files in file_hashes.items() if len(files) > 1}
    
    if not duplicates:
        print(f"\n  ✅ No duplicates found!")
        return 0
    
    print(f"\n  {'─'*76}")
    print(f"  🔴 FOUND {len(duplicates)} DUPLICATE GROUPS")
    print(f"  {'─'*76}\n")
    
    # Categorize duplicates
    within_class = []
    cross_class = []
    
    for dup_hash, files in duplicates.items():
        classes = [f[0] for f in files]
        if len(set(classes)) == 1:
            within_class.append((dup_hash, files))
        else:
            cross_class.append((dup_hash, files))
    
    # Report within-class duplicates
    if within_class:
        print(f"  📁 WITHIN-CLASS DUPLICATES: {len(within_class)} groups")
        print(f"     (Same image multiple times in one class)")
        print()
        for idx, (dup_hash, files) in enumerate(within_class, 1):
            class_name = files[0][0]
            print(f"  {idx}. {class_name.upper()} ({len(files)} copies):")
            for cn, fname, fpath in files:
                print(f"       {fname}")
            print()
    
    # Report cross-class duplicates (CRITICAL)
    if cross_class:
        print(f"  🚨 CROSS-CLASS DUPLICATES: {len(cross_class)} groups")
        print(f"     (Same image in different classes - CRITICAL ISSUE!)")
        print()
        for idx, (dup_hash, files) in enumerate(cross_class, 1):
            classes = [f[0] for f in files]
            print(f"  {idx}. Classes: {', '.join(set(classes))}")
            for cn, fname, fpath in files:
                print(f"       {cn.upper()}: {fname}")
            print(f"       Full paths:")
            for cn, fname, fpath in files:
                print(f"         {fpath}")
            print()
    
    # Summary
    print(f"  {'─'*76}")
    print(f"  SUMMARY:")
    print(f"    Within-class: {len(within_class)} groups (minor issue)")
    print(f"    Cross-class:  {len(cross_class)} groups (CRITICAL - breaks training!)")
    print(f"  {'─'*76}\n")
    
    return len(cross_class)

def main():
    print("="*80)
    print("  DUPLICATE IMAGE FINDER")
    print("="*80)
    print("\n  Searching for duplicate images in dataset...")
    print("  This will take 1-2 minutes...")
    
    train_cross = find_duplicates_in_split(TRAIN_DIR, "Training")
    val_cross = find_duplicates_in_split(VAL_DIR, "Validation")
    
    print("\n" + "="*80)
    print("  OVERALL SUMMARY")
    print("="*80)
    print(f"  Training set: {train_cross} cross-class duplicate groups")
    print(f"  Validation set: {val_cross} cross-class duplicate groups")
    
    if train_cross > 0 or val_cross > 0:
        print("\n  🚨 ACTION REQUIRED:")
        print("     1. Review the duplicate groups above")
        print("     2. For each group, decide which class is CORRECT")
        print("     3. Delete or move the incorrect copies")
        print("     4. Re-run training after fixing duplicates")
        print("\n  Cross-class duplicates are the #1 cause of low accuracy!")
    else:
        print("\n  ✅ No critical issues found!")
        print("     Your dataset is clean - proceed with training")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
    except Exception as e:
        print(f"\n\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
