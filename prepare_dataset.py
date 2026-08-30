import os
import shutil
import random
from pathlib import Path

# Paths
BASE_DIR = Path("c:/Users/aboni/Desktop/BUYO LEAVES")
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

classes = ["CLASS A", "CLASS B", "CLASS C", "CLASS D", "CLASS E"]
SPLIT_RATIO = 0.8  # 80% train, 20% val

def prepare():
    # Setup directories
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    
    for cls in classes:
        cls_name = cls.lower().replace(" ", "_")
        (TRAIN_DIR / cls_name).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / cls_name).mkdir(parents=True, exist_ok=True)

    print("Splitting datasets:")
    for cls in classes:
        cls_path = BASE_DIR / cls
        cls_name = cls.lower().replace(" ", "_")
        
        # Get all images including subdirectories (like CLASS E/CLASS A-C)
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(list(cls_path.rglob(ext)))
            images.extend(list(cls_path.rglob(ext.upper())))
            
        # Remove duplicates
        images = list(set(images))
        
        # Shuffle deterministically
        random.seed(42)
        random.shuffle(images)
        
        train_count = int(len(images) * SPLIT_RATIO)
        train_images = images[:train_count]
        val_images = images[train_count:]
        
        print(f"{cls}: {len(images)} images -> {len(train_images)} train, {len(val_images)} val")
        
        for img in train_images:
            shutil.copy2(img, TRAIN_DIR / cls_name / img.name)
            
        for img in val_images:
            shutil.copy2(img, VAL_DIR / cls_name / img.name)

    print("\nDataset preparation completed!")

if __name__ == "__main__":
    prepare()
