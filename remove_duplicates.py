"""
Automatic Duplicate Remover
============================
Automatically removes duplicate images across classes.
Strategy: Keep the copy in the FIRST alphabetically-sorted class, delete others.

BACKUP WARNING: This script DELETES files! Make a backup first if unsure.
"""

import os
import hashlib
from collections import defaultdict
import shutil

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

def remove_duplicates_in_split(split_dir, split_name, dry_run=True):
    """Remove duplicates in train or val split"""
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
        
        print(f"  Scanning {class_name.upper()}: {len(files)} images...", end=" ")
        
        for fname in files:
            fpath = os.path.join(class_path, fname)
            fhash = file_hash(fpath)
            if fhash:
                file_hashes[fhash].append((class_name, fname, fpath))
        
        print("✓")
    
    # Find cross-class duplicates
    cross_dupes = []
    for fhash, files in file_hashes.items():
        if len(files) > 1:
            classes = set([f[0] for f in files])
            if len(classes) > 1:  # Cross-class duplicate
                cross_dupes.append((fhash, files))
    
    if not cross_dupes:
        print(f"\n  ✅ No cross-class duplicates found!")
        return 0
    
    print(f"\n  Found {len(cross_dupes)} cross-class duplicate groups")
    
    files_to_remove = []
    files_to_keep = []
    
    # For each duplicate group, keep first alphabetically, delete rest
    for dup_hash, files in cross_dupes:
        # Sort by class name (alphabetically)
        sorted_files = sorted(files, key=lambda x: x[0])
        keep = sorted_files[0]
        remove = sorted_files[1:]
        
        files_to_keep.append(keep)
        files_to_remove.extend(remove)
        
        print(f"\n  Group: {len(files)} copies")
        print(f"    KEEP:   {keep[0].upper()}/{keep[1]}")
        for r in remove:
            print(f"    DELETE: {r[0].upper()}/{r[1]}")
    
    # Execute removal
    print(f"\n  {'─'*76}")
    if dry_run:
        print(f"  DRY RUN MODE - No files will be deleted")
        print(f"  Would delete {len(files_to_remove)} duplicate files")
        print(f"  Would keep {len(files_to_keep)} original files")
    else:
        print(f"  DELETING {len(files_to_remove)} duplicate files...")
        deleted = 0
        failed = 0
        
        for class_name, fname, fpath in files_to_remove:
            try:
                os.remove(fpath)
                deleted += 1
                print(f"    ✓ Deleted: {class_name}/{fname}")
            except Exception as e:
                failed += 1
                print(f"    ✗ Failed: {class_name}/{fname} - {e}")
        
        print(f"\n  Summary:")
        print(f"    ✓ Deleted: {deleted} files")
        if failed > 0:
            print(f"    ✗ Failed: {failed} files")
    
    print(f"  {'─'*76}")
    return len(files_to_remove)

def main():
    import sys
    
    print("="*80)
    print("  DUPLICATE IMAGE REMOVER")
    print("="*80)
    
    # Check if user wants to do actual deletion
    dry_run = True
    if len(sys.argv) > 1 and sys.argv[1] == '--delete':
        dry_run = False
        print("\n  ⚠️  DELETION MODE ENABLED")
        print("  Files will be PERMANENTLY deleted!")
        response = input("\n  Type 'DELETE' to confirm: ")
        if response != 'DELETE':
            print("\n  Aborted. No files deleted.")
            return
    else:
        print("\n  Running in DRY RUN mode (no files will be deleted)")
        print("  To actually delete files, run: python remove_duplicates.py --delete")
    
    # Process both splits
    train_removed = remove_duplicates_in_split(TRAIN_DIR, "Training", dry_run)
    val_removed = remove_duplicates_in_split(VAL_DIR, "Validation", dry_run)
    
    total = train_removed + val_removed
    
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    
    if dry_run:
        print(f"  Would remove {total} duplicate files total")
        print(f"    Training: {train_removed}")
        print(f"    Validation: {val_removed}")
        print("\n  To actually delete these files:")
        print("  python remove_duplicates.py --delete")
    else:
        print(f"  Removed {total} duplicate files total")
        print(f"    Training: {train_removed}")
        print(f"    Validation: {val_removed}")
        print("\n  ✅ Duplicates removed successfully!")
        print("  You can now train with: python train_ensemble.py")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. No files deleted.")
    except Exception as e:
        print(f"\n\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
