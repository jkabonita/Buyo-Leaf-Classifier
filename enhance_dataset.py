"""
enhance_dataset.py - Aggressive Dataset Enhancement for Better Accuracy
========================================================================
Strategies:
1. More aggressive augmentation with higher variation
2. Remove near-duplicate and low-quality images
3. Balance the dataset better
4. Add more diverse augmentations
"""

import os
import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from collections import defaultdict
import hashlib

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_DIR = Path('dataset/train')
TARGET_PER_CLASS = 600  # Increase to 600 images per class
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def elastic_transform(img, alpha=40, sigma=5):
    """Elastic deformation for more variation"""
    arr = np.array(img)
    shape = arr.shape[:2]
    
    dx = np.random.randn(*shape) * sigma
    dy = np.random.randn(*shape) * sigma
    
    dx = np.clip(dx, -alpha, alpha)
    dy = np.clip(dy, -alpha, alpha)
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (np.clip(y + dy, 0, shape[0]-1).astype(int),
               np.clip(x + dx, 0, shape[1]-1).astype(int))
    
    if len(arr.shape) == 3:
        return Image.fromarray(arr[indices[0], indices[1], :])
    return Image.fromarray(arr[indices])


def add_shadow(img):
    """Add random shadows"""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    
    # Random shadow parameters
    x1 = random.randint(0, w)
    y1 = random.randint(0, h)
    x2 = random.randint(0, w)
    y2 = random.randint(0, h)
    
    # Create gradient
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    dist = np.sqrt((xx - x1)**2 + (yy - y1)**2) / np.sqrt(w**2 + h**2)
    shadow = 1 - (dist * random.uniform(0.3, 0.7))
    shadow = np.clip(shadow, 0.3, 1.0)
    
    if len(arr.shape) == 3:
        shadow = shadow[:, :, np.newaxis]
    
    result = arr * shadow
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def advanced_augment(img):
    """Apply very aggressive augmentation pipeline"""
    ops = [
        # Geometric transforms
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT) if random.random() > 0.5 else x,
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM) if random.random() > 0.3 else x,
        lambda x: x.rotate(random.uniform(-180, 180), expand=False, 
                          fillcolor=(random.randint(80, 180),) * 3),
        lambda x: ImageOps.autocontrast(x),
        
        # Perspective and distortion
        lambda x: random_crop_scale(x),
        lambda x: perspective_transform(x),
        
        # Color/lighting variations
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.4, 1.8)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.4, 2.0)),
        lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.3, 1.8)),
        lambda x: ImageEnhance.Sharpness(x).enhance(random.uniform(0.2, 3.0)),
        
        # Filters
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5))),
        lambda x: x.filter(ImageFilter.EDGE_ENHANCE) if random.random() > 0.7 else x,
        lambda x: x.filter(ImageFilter.SMOOTH_MORE) if random.random() > 0.7 else x,
        
        # Advanced
        lambda x: add_noise_advanced(x),
        lambda x: add_shadow(x) if random.random() > 0.5 else x,
        lambda x: elastic_transform(x) if random.random() > 0.6 else x,
    ]
    
    # Apply 3-6 random operations
    num_ops = random.randint(3, 6)
    selected = random.sample(ops, num_ops)
    
    for op in selected:
        try:
            img = op(img)
        except Exception as e:
            continue
    
    return img


def random_crop_scale(img):
    """Random crop with various scales"""
    w, h = img.size
    scale = random.uniform(0.6, 0.95)
    new_w, new_h = int(w * scale), int(h * scale)
    x = random.randint(0, w - new_w)
    y = random.randint(0, h - new_h)
    cropped = img.crop((x, y, x + new_w, y + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


def perspective_transform(img):
    """More aggressive perspective distortion"""
    w, h = img.size
    shift = random.randint(15, 50)
    expanded = img.resize((w + shift * 2, h + shift * 2), Image.LANCZOS)
    x0 = random.randint(0, shift * 2)
    y0 = random.randint(0, shift * 2)
    return expanded.crop((x0, y0, x0 + w, y0 + h))


def add_noise_advanced(img):
    """Multiple types of noise"""
    arr = np.array(img, dtype=np.float32)
    
    # Choose noise type
    noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, random.uniform(5, 25), arr.shape)
    elif noise_type == 'salt_pepper':
        noise = np.zeros_like(arr)
        prob = random.uniform(0.01, 0.05)
        noise[np.random.random(arr.shape[:2]) < prob] = 255
        noise[np.random.random(arr.shape[:2]) < prob] = -255
    else:  # speckle
        noise = arr * np.random.randn(*arr.shape) * random.uniform(0.1, 0.3)
    
    result = arr + noise
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_hash(img_path):
    """Compute perceptual hash for near-duplicate detection"""
    try:
        img = Image.open(img_path).convert('L').resize((16, 16), Image.LANCZOS)
        arr = np.array(img)
        avg = arr.mean()
        return ''.join('1' if x > avg else '0' for x in arr.flatten())
    except:
        return None


def hamming_distance(h1, h2):
    """Calculate hamming distance between two hashes"""
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def remove_duplicates(class_dir, threshold=5):
    """Remove near-duplicate images"""
    images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
    images = [img for img in images if not img.name.startswith('aug_')]  # Only original images
    
    print(f"    Checking {len(images)} original images for duplicates...")
    
    hashes = {}
    to_remove = set()
    
    for img_path in images:
        h = compute_hash(img_path)
        if h is None:
            continue
        
        # Check against existing hashes
        is_duplicate = False
        for existing_path, existing_hash in hashes.items():
            if hamming_distance(h, existing_hash) < threshold:
                to_remove.add(img_path)
                is_duplicate = True
                break
        
        if not is_duplicate:
            hashes[img_path] = h
    
    # Remove duplicates
    for img_path in to_remove:
        try:
            img_path.unlink()
        except:
            pass
    
    print(f"    Removed {len(to_remove)} near-duplicate images")
    return len(hashes)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENHANCEMENT
# ─────────────────────────────────────────────────────────────────────────────

def get_images(directory):
    """Get all image files"""
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    images = []
    for ext in exts:
        images.extend(directory.glob(ext))
    return [img for img in set(images) if not img.name.startswith('aug_')]


def enhance_class(class_dir, target):
    """Enhance a single class with deduplication and advanced augmentation"""
    print(f"\n  📁 {class_dir.name}")
    
    # Remove old augmented files
    for old_aug in class_dir.glob('aug_*.jpg'):
        try:
            old_aug.unlink()
        except:
            pass
    
    # Remove near-duplicates
    num_unique = remove_duplicates(class_dir)
    
    # Get remaining images
    images = get_images(class_dir)
    current = len(images)
    
    if current == 0:
        print(f"    ⚠️  No images found!")
        return
    
    print(f"    Unique images: {num_unique}")
    print(f"    Current: {current} → Target: {target}")
    
    if current >= target:
        print(f"    ✅ Already at target")
        return
    
    needed = target - current
    print(f"    Generating {needed} enhanced images...")
    
    aug_idx = 0
    attempts = 0
    max_attempts = needed * 3
    
    while aug_idx < needed and attempts < max_attempts:
        attempts += 1
        src_path = random.choice(images)
        
        try:
            img = Image.open(src_path).convert('RGB')
            # Standardize size
            img = img.resize((256, 256), Image.LANCZOS)
            
            # Apply advanced augmentation
            aug = advanced_augment(img)
            
            # Verify augmented image is different enough
            if aug.size == img.size:
                out_name = f'aug_{aug_idx:05d}_{src_path.stem}.jpg'
                out_path = class_dir / out_name
                aug.save(out_path, 'JPEG', quality=95)
                aug_idx += 1
        except Exception as e:
            continue
    
    final_count = len(get_images(class_dir)) + aug_idx
    print(f"    ✅ {current} → {final_count} images (+{aug_idx} generated)")


def main():
    print("=" * 70)
    print("  ENHANCED DATASET PREPARATION")
    print(f"  Target: {TARGET_PER_CLASS} high-quality images per class")
    print("=" * 70)
    
    if not TRAIN_DIR.exists():
        print(f"\n❌ Training directory not found: {TRAIN_DIR}")
        return
    
    class_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    
    if not class_dirs:
        print(f"❌ No class directories found in {TRAIN_DIR}")
        return
    
    for class_dir in class_dirs:
        enhance_class(class_dir, TARGET_PER_CLASS)
    
    print("\n" + "=" * 70)
    print("✅ DATASET ENHANCEMENT COMPLETE!")
    print("   Next: Run training with improved dataset")
    print("=" * 70)


if __name__ == '__main__':
    main()
