"""
inspect_dataset.py - Analyze dataset for quality issues
"""
import os
from PIL import Image
from pathlib import Path
from collections import Counter

def check_images(directory):
    """Check for corrupted or problematic images"""
    issues = []
    sizes = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.verify()  # Verify it's a valid image
                    img = Image.open(path)  # Reopen after verify
                    sizes.append(img.size)
                    
                    # Check if image is too small
                    if img.size[0] < 100 or img.size[1] < 100:
                        issues.append(f"Too small: {path} ({img.size})")
                    
                    # Check aspect ratio
                    aspect = max(img.size) / min(img.size)
                    if aspect > 5:
                        issues.append(f"Extreme aspect ratio: {path} ({img.size})")
                        
                except Exception as e:
                    issues.append(f"Corrupted: {path} - {e}")
    
    return issues, sizes

print("=" * 70)
print("  Dataset Inspection Report")
print("=" * 70)

# Check training data
print("\nChecking TRAIN dataset...")
train_issues, train_sizes = check_images("dataset/train")
print(f"  Total training images scanned: {len(train_sizes)}")
print(f"  Issues found: {len(train_issues)}")
if train_issues:
    print("\n  Issues:")
    for issue in train_issues[:10]:  # Show first 10
        print(f"    - {issue}")
    if len(train_issues) > 10:
        print(f"    ... and {len(train_issues) - 10} more")

# Check validation data
print("\nChecking VAL dataset...")
val_issues, val_sizes = check_images("dataset/val")
print(f"  Total validation images scanned: {len(val_sizes)}")
print(f"  Issues found: {len(val_issues)}")
if val_issues:
    print("\n  Issues:")
    for issue in val_issues[:10]:
        print(f"    - {issue}")
    if len(val_issues) > 10:
        print(f"    ... and {len(val_issues) - 10} more")

# Size distribution
if train_sizes:
    widths = [s[0] for s in train_sizes]
    heights = [s[1] for s in train_sizes]
    print(f"\nImage size statistics:")
    print(f"  Width  - min: {min(widths)}, max: {max(widths)}, avg: {sum(widths)//len(widths)}")
    print(f"  Height - min: {min(heights)}, max: {max(heights)}, avg: {sum(heights)//len(heights)}")

# Per-class distribution
print("\n" + "=" * 70)
print("  Per-class distribution")
print("=" * 70)
for split in ['train', 'val']:
    print(f"\n{split.upper()}:")
    for cls in ['class_a', 'class_b', 'class_c', 'class_d', 'class_e']:
        path = Path(f"dataset/{split}/{cls}")
        if path.exists():
            count = len(list(path.glob('*.jpg'))) + len(list(path.glob('*.jpeg'))) + len(list(path.glob('*.png')))
            print(f"  {cls.upper()}: {count:>3} images")

print("\n" + "=" * 70)
if train_issues or val_issues:
    print("⚠ Issues detected! Consider cleaning the dataset.")
else:
    print("✓ No obvious issues detected.")
print("=" * 70)
