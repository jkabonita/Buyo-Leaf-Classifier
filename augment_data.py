"""
augment_data.py — Offline Data Augmentation for Buyo Leaves Dataset
====================================================================
Expands each class in dataset/train to at least MIN_IMAGES_PER_CLASS
images by generating augmented variants of existing photos.

Run this BEFORE train_gpu.py to boost accuracy on small classes (D, E).

Usage:
    python augment_data.py
"""

import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_DIR            = Path('dataset/train')
MIN_IMAGES_PER_CLASS = 200    # Minimum images each class should have
SEED                 = 42

random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _random_crop(img: Image.Image) -> Image.Image:
    """Randomly crop a portion and resize back."""
    w, h = img.size
    scale = random.uniform(0.72, 0.95)
    new_w, new_h = int(w * scale), int(h * scale)
    x = random.randint(0, w - new_w)
    y = random.randint(0, h - new_h)
    return img.crop((x, y, x + new_w, y + new_h)).resize((w, h), Image.BILINEAR)


def _add_gaussian_noise(img: Image.Image) -> Image.Image:
    """Add mild Gaussian noise."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(4, 18), arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def _perspective_warp(img: Image.Image) -> Image.Image:
    """Slight perspective distortion via affine approximation."""
    w, h = img.size
    shift = random.randint(10, 30)
    # simple perspective-like by resizing then cropping
    expanded = img.resize((w + shift * 2, h + shift * 2), Image.BILINEAR)
    x0 = random.randint(0, shift * 2)
    y0 = random.randint(0, shift * 2)
    return expanded.crop((x0, y0, x0 + w, y0 + h))


def augment_image(img: Image.Image) -> Image.Image:
    """
    Apply a random combination of 2–4 augmentations drawn from a diverse pool.
    Ensures augmented images look visually distinct from the original.
    """
    augmentation_pool = [
        # Geometric
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
        lambda x: x.rotate(random.uniform(-50, 50), expand=False,
                           fillcolor=(random.randint(100, 160),) * 3),
        lambda x: _random_crop(x),
        lambda x: _perspective_warp(x),

        # Photometric
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.55, 1.5)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.55, 1.6)),
        lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.5, 1.6)),
        lambda x: ImageEnhance.Sharpness(x).enhance(random.uniform(0.3, 2.5)),

        # Filters
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.8))),
        lambda x: x.filter(ImageFilter.SMOOTH),

        # Noise
        lambda x: _add_gaussian_noise(x),
    ]

    num_ops = random.randint(2, 4)
    chosen_ops = random.sample(augmentation_pool, num_ops)

    for op in chosen_ops:
        try:
            img = op(img)
        except Exception:
            pass  # Skip any op that fails (e.g., edge case image sizes)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AUGMENTATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def get_images(directory: Path):
    """Collect all image files from a directory."""
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    images = []
    for ext in exts:
        images.extend(directory.glob(ext))
    return list(set(images))


def augment_class(class_dir: Path, target: int) -> None:
    """Generate augmented images for a single class until target count is met."""
    images = get_images(class_dir)
    current = len(images)

    if current == 0:
        print(f'  ⚠ {class_dir.name}: No images found — skipping.')
        return

    if current >= target:
        print(f'  ✓ {class_dir.name}: {current} images — already at target.')
        return

    needed = target - current
    print(f'  {class_dir.name}: {current} images → generating {needed} augmented copies...')

    aug_idx = 0
    attempts = 0
    max_attempts = needed * 5  # safety limit

    while aug_idx < needed and attempts < max_attempts:
        attempts += 1
        src_path = random.choice(images)
        try:
            img = Image.open(src_path).convert('RGB')
            # Resize to standard size before augmenting
            img = img.resize((256, 256), Image.BILINEAR)
            aug = augment_image(img)
            out_name = f'aug_{aug_idx:05d}_{src_path.stem}.jpg'
            out_path = class_dir / out_name
            aug.save(out_path, 'JPEG', quality=92)
            aug_idx += 1
        except Exception as e:
            print(f'    Error with {src_path.name}: {e}')

    final_count = len(get_images(class_dir))
    print(f'  + {class_dir.name}: {current} -> {final_count} images (+{aug_idx} generated)')


def main():
    print('=' * 60)
    print('  Buyo Leaves — Offline Data Augmentation')
    print(f'  Target: >={MIN_IMAGES_PER_CLASS} images per class\n')
    print('=' * 60)

    if not TRAIN_DIR.exists():
        print(f'\nX Training directory not found: {TRAIN_DIR}')
        print('   Run: python prepare_dataset.py  first.')
        return

    class_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f'❌ No class sub-folders found in {TRAIN_DIR}')
        return

    print(f'\nFound {len(class_dirs)} classes:\n')
    for class_dir in class_dirs:
        augment_class(class_dir, MIN_IMAGES_PER_CLASS)

    print('\n' + '=' * 60)
    print('✅ Augmentation complete!')
    print('   Next step: python train_gpu.py')
    print('=' * 60)


if __name__ == '__main__':
    main()
