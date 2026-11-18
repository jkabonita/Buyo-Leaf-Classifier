"""
Dataset Quality Diagnostic Tool
================================
Analyzes the buyo leaf dataset to identify potential issues:
- Class imbalance
- Image quality problems
- Duplicate or corrupted images
- Visual similarity between classes
- Dataset statistics
"""

import os
import numpy as np
from PIL import Image
import cv2
from collections import Counter, defaultdict
import hashlib

# Optional progress bar (tqdm not required)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", leave=False):
        """Fallback when tqdm not available"""
        print(f"  {desc}...")
        return iterable

CLASS_NAMES = ['class_a', 'class_b', 'class_c', 'class_d', 'class_e']
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/val'

def compute_image_hash(img_path):
    """Compute perceptual hash of image"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (16, 16))
        avg = img.mean()
        return ''.join(['1' if p > avg else '0' for p in img.flatten()])
    except:
        return None

def analyze_image_quality(img_path):
    """Analyze image quality metrics"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute metrics
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()  # Blur detection
        brightness = gray.mean()
        contrast = gray.std()
        
        # Histogram entropy (image complexity)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return {
            'blur': blur_score,
            'brightness': brightness,
            'contrast': contrast,
            'entropy': entropy,
            'resolution': img.shape[:2]
        }
    except Exception as e:
        return None

def analyze_dataset():
    print("=" * 80)
    print("  BUYO LEAF DATASET DIAGNOSTIC TOOL")
    print("=" * 80)
    
    for split_name, split_dir in [("TRAINING", TRAIN_DIR), ("VALIDATION", VAL_DIR)]:
        print(f"\n{'='*80}")
        print(f"  {split_name} SET ANALYSIS")
        print(f"{'='*80}")
        
        class_stats = {}
        all_hashes = defaultdict(list)
        quality_issues = []
        
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_path = os.path.join(split_dir, class_name)
            if not os.path.exists(class_path):
                print(f"\n  ⚠️  {class_name.upper()} directory not found!")
                continue
            
            files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"\n  {class_name.upper()}: {len(files)} images")
            
            # Analyze all images in this class
            qualities = []
            corrupted = []
            duplicates = []
            hashes_in_class = set()
            
            for fname in tqdm(files, desc=f"    Analyzing {class_name}", leave=False):
                fpath = os.path.join(class_path, fname)
                
                # Check if image can be loaded
                try:
                    img = Image.open(fpath)
                    img.verify()
                except Exception as e:
                    corrupted.append((fname, str(e)))
                    continue
                
                # Compute hash for duplicate detection
                img_hash = compute_image_hash(fpath)
                if img_hash:
                    if img_hash in hashes_in_class:
                        duplicates.append(fname)
                    hashes_in_class.add(img_hash)
                    all_hashes[img_hash].append((class_name, fname))
                
                # Analyze quality
                quality = analyze_image_quality(fpath)
                if quality:
                    qualities.append(quality)
                    
                    # Flag low quality images
                    if quality['blur'] < 50:
                        quality_issues.append((class_name, fname, 'blurry', quality['blur']))
                    if quality['brightness'] < 30 or quality['brightness'] > 225:
                        quality_issues.append((class_name, fname, 'bad_brightness', quality['brightness']))
                    if quality['contrast'] < 20:
                        quality_issues.append((class_name, fname, 'low_contrast', quality['contrast']))
            
            # Compute statistics
            if qualities:
                avg_blur = np.mean([q['blur'] for q in qualities])
                avg_brightness = np.mean([q['brightness'] for q in qualities])
                avg_contrast = np.mean([q['contrast'] for q in qualities])
                avg_entropy = np.mean([q['entropy'] for q in qualities])
                resolutions = [q['resolution'] for q in qualities]
                
                print(f"    ✓ Images loaded: {len(qualities)}/{len(files)}")
                print(f"    📊 Avg blur score: {avg_blur:.1f} (higher=sharper)")
                print(f"    💡 Avg brightness: {avg_brightness:.1f} (0-255)")
                print(f"    📈 Avg contrast: {avg_contrast:.1f}")
                print(f"    🔀 Avg entropy: {avg_entropy:.2f} (complexity)")
                
                # Resolution analysis
                res_counter = Counter([f"{h}x{w}" for h, w in resolutions])
                if len(res_counter) > 1:
                    print(f"    ⚠️  Mixed resolutions: {dict(res_counter)}")
                
                class_stats[class_name] = {
                    'count': len(files),
                    'blur': avg_blur,
                    'brightness': avg_brightness,
                    'contrast': avg_contrast,
                    'entropy': avg_entropy,
                    'qualities': qualities
                }
            
            if corrupted:
                print(f"    ❌ Corrupted images: {len(corrupted)}")
                for fname, err in corrupted[:3]:
                    print(f"       - {fname}: {err}")
            
            if duplicates:
                print(f"    🔄 Duplicate images within class: {len(duplicates)}")
                for fname in duplicates[:3]:
                    print(f"       - {fname}")
        
        # Cross-class duplicate detection
        print(f"\n  {'─'*76}")
        print(f"  CROSS-CLASS DUPLICATE DETECTION")
        print(f"  {'─'*76}")
        cross_dupes = {h: paths for h, paths in all_hashes.items() if len(paths) > 1}
        if cross_dupes:
            print(f"    ⚠️  Found {len(cross_dupes)} images duplicated across classes!")
            for idx, (img_hash, paths) in enumerate(list(cross_dupes.items())[:5], 1):
                classes = [p[0] for p in paths]
                files = [p[1] for p in paths]
                print(f"    {idx}. Classes: {classes}")
                print(f"       Files: {[f[:30] for f in files]}")
        else:
            print(f"    ✓ No cross-class duplicates found")
        
        # Quality issues summary
        if quality_issues:
            print(f"\n  {'─'*76}")
            print(f"  QUALITY ISSUES SUMMARY")
            print(f"  {'─'*76}")
            issue_types = Counter([issue[2] for issue in quality_issues])
            for issue_type, count in issue_types.items():
                print(f"    {issue_type}: {count} images")
            
            # Show worst offenders per class
            by_class = defaultdict(list)
            for class_name, fname, issue_type, value in quality_issues:
                by_class[class_name].append((fname, issue_type, value))
            
            for class_name, issues in by_class.items():
                if len(issues) > 5:  # Only show if significant
                    print(f"\n    {class_name.upper()}: {len(issues)} issues")
                    for fname, itype, val in issues[:3]:
                        print(f"       - {fname}: {itype} ({val:.1f})")
        
        # Inter-class comparison
        if len(class_stats) >= 2:
            print(f"\n  {'─'*76}")
            print(f"  INTER-CLASS COMPARISON")
            print(f"  {'─'*76}")
            
            # Compare blur scores
            blur_scores = {k: v['blur'] for k, v in class_stats.items()}
            blur_sorted = sorted(blur_scores.items(), key=lambda x: x[1])
            print(f"    Blur ranking (higher=sharper):")
            for class_name, blur in blur_sorted:
                bar = "█" * int(blur / 10) + "░" * max(0, 20 - int(blur / 10))
                print(f"      {class_name}: {blur:6.1f} |{bar}|")
            
            # Check if any class is significantly different
            blur_values = list(blur_scores.values())
            blur_std = np.std(blur_values)
            if blur_std > 50:
                print(f"    ⚠️  High variation in blur scores (std={blur_std:.1f})")
                print(f"       Classes may have inconsistent image quality!")
            
            # Compare brightness
            brightness_scores = {k: v['brightness'] for k, v in class_stats.items()}
            bright_values = list(brightness_scores.values())
            bright_std = np.std(bright_values)
            if bright_std > 20:
                print(f"\n    ⚠️  High variation in brightness (std={bright_std:.1f})")
                for class_name, bright in brightness_scores.items():
                    print(f"      {class_name}: {bright:.1f}")
            
            # Compare entropy (complexity)
            entropy_scores = {k: v['entropy'] for k, v in class_stats.items()}
            print(f"\n    Image complexity (entropy):")
            for class_name, entropy in sorted(entropy_scores.items(), key=lambda x: x[1]):
                bar = "█" * int(entropy * 3) + "░" * max(0, 20 - int(entropy * 3))
                print(f"      {class_name}: {entropy:.3f} |{bar}|")
    
    print(f"\n{'='*80}")
    print(f"  RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if quality_issues:
        print(f"  1. Review and possibly remove {len(quality_issues)} low-quality images")
    if cross_dupes:
        print(f"  2. Remove or re-label {len(cross_dupes)} duplicate images across classes")
    
    print(f"  3. Consider re-capturing images for classes with:")
    print(f"     - Low blur scores (< 100)")
    print(f"     - Extreme brightness (< 50 or > 200)")
    print(f"     - Low contrast (< 30)")
    
    print(f"\n  💡 Training Tips:")
    print(f"     - Use stronger augmentation for low-quality classes")
    print(f"     - Apply focal loss to focus on hard examples")
    print(f"     - Consider data augmentation to balance quality differences")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    try:
        analyze_dataset()
    except KeyboardInterrupt:
        print("\n\n  Analysis interrupted by user.")
    except Exception as e:
        print(f"\n\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
