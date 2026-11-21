"""
train_single_model_max_accuracy.py -- Single Model Maximum Accuracy Training
=============================================================================
Trains ONE highly optimized ConvNeXt-Tiny model using ALL 5 classes (A-E)
with maximum CUDA optimization to replace buyo_best.pth

TARGET: 95%+ validation accuracy
GPU: Full CUDA utilization with mixed precision training

Key Features:
- ConvNeXt-Tiny backbone (best performance/efficiency balance)
- Mixed precision training (FP16 + FP32)
- Advanced augmentation (Mixup, CutMix, RandAugment)
- Focal Loss for hard examples
- Class-weighted sampling
- Gradient accumulation for larger effective batch size
- Cosine annealing with warmup
- Stochastic Weight Averaging (SWA)
- Test-Time Augmentation (TTA)
- Early stopping with best model restoration
"""

import os, sys, copy, random, time, gc
import numpy as np
from collections import Counter
from pathlib import Path

# Force CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY CUDA
# ═══════════════════════════════════════════════════════════════════════════════
if not torch.cuda.is_available():
    print("❌ ERROR: CUDA not available!")
    print("   Please install CUDA-enabled PyTorch:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

print("=" * 70)
print("  🍃 BUYO LEAF SINGLE MODEL TRAINER -- Maximum Accuracy Mode")
print("=" * 70)
print(f"  GPU      : {torch.cuda.get_device_name(0)}")
print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
print(f"  CUDA     : {torch.version.cuda}")
print(f"  PyTorch  : {torch.__version__}")
print(f"  cuDNN    : {torch.backends.cudnn.version()}")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS - OPTIMIZED FOR MAXIMUM ACCURACY
# ═══════════════════════════════════════════════════════════════════════════════
IMG_SIZE       = 256           # Larger images for better detail
BATCH_SIZE     = 28            # Optimal for RTX 3050 6GB
ACCUM_STEPS    = 2             # Effective batch = 56
NUM_WORKERS    = 0             # Windows compatibility
PIN_MEMORY     = True

PHASE1_EPOCHS  = 25            # Head warmup
PHASE2_EPOCHS  = 120           # Full training
SWA_START      = 100           # Start SWA after this epoch
SWA_LR         = 1e-5          # SWA learning rate

LR_MAX         = 2e-3          # Maximum learning rate
LR_MIN         = 1e-6          # Minimum learning rate
WARMUP_EPOCHS  = 8             # Learning rate warmup
WEIGHT_DECAY   = 1e-4          # L2 regularization

# Focal Loss for hard examples
USE_FOCAL_LOSS = True
FOCAL_GAMMA    = 2.5
FOCAL_ALPHA    = 0.25

# Augmentation
LABEL_SMOOTH   = 0.1
MIXUP_ALPHA    = 0.4
CUTMIX_ALPHA   = 1.0
CUTMIX_PROB    = 0.5

# Training
NUM_CLASSES    = 5
TARGET_ACC     = 0.95
PATIENCE       = 35
SEED           = 42
TTA_AUGS       = 8             # Number of TTA augmentations

# Paths - Use CLASS A-E folders directly
CLASS_FOLDERS  = ["CLASS A", "CLASS B", "CLASS C", "CLASS D", "CLASS E"]
SAVE_PATH      = "buyo_best.pth"

# Seed everything
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET PREPARATION - CREATE TRAIN/VAL SPLIT FROM CLASS FOLDERS
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_dataset(class_folders, train_split=0.85):
    """Create train/val split from CLASS A-E folders."""
    dataset_root = Path("dataset_temp")
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    
    # Clean up old temp dataset
    if dataset_root.exists():
        import shutil
        shutil.rmtree(dataset_root)
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Preparing dataset from {len(class_folders)} class folders...")
    
    total_train = 0
    total_val = 0
    
    for class_folder in class_folders:
        if not Path(class_folder).exists():
            print(f"   ⚠️  {class_folder} not found, skipping...")
            continue
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(class_folder).glob(ext))
        
        if not image_files:
            print(f"   ⚠️  No images in {class_folder}, skipping...")
            continue
        
        # Shuffle and split
        random.shuffle(image_files)
        n_train = int(len(image_files) * train_split)
        train_files = image_files[:n_train]
        val_files = image_files[n_train:]
        
        # Create class subdirectories
        class_name = class_folder.split()[-1]  # "CLASS A" -> "A"
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        
        # Copy files (using symlinks on Windows if possible, else copy)
        for src in train_files:
            dst = train_dir / class_name / src.name
            try:
                dst.symlink_to(src.absolute())
            except:
                import shutil
                shutil.copy2(src, dst)
        
        for src in val_files:
            dst = val_dir / class_name / src.name
            try:
                dst.symlink_to(src.absolute())
            except:
                import shutil
                shutil.copy2(src, dst)
        
        total_train += len(train_files)
        total_val += len(val_files)
        
        print(f"   ✓ {class_name}: {len(train_files)} train, {len(val_files)} val")
    
    print(f"   📊 Total: {total_train} train, {total_val} val")
    return str(train_dir), str(val_dir)

# ═══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS - FOCUS ON HARD EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """Focal Loss - focuses training on hard misclassified examples."""
    def __init__(self, gamma=2.0, alpha=0.25, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.weight, 
            label_smoothing=self.label_smoothing, 
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ═══════════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION - AGGRESSIVE
# ═══════════════════════════════════════════════════════════════════════════════
def get_train_transform():
    """Training augmentation - aggressive for generalization."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 48, IMG_SIZE + 48)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.2, 0.2), 
            scale=(0.8, 1.2), 
            shear=15
        ),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.ColorJitter(
            brightness=0.6, 
            contrast=0.6, 
            saturation=0.5, 
            hue=0.15
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
    ])

def get_val_transform():
    """Validation transform - center crop only."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_tta_transforms(n_aug=8):
    """Generate TTA transforms for test-time augmentation."""
    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    
    tta_list = [transforms.Compose(base)]  # Original
    
    if n_aug >= 2:
        tta_list.append(transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))
    
    if n_aug >= 4:
        tta_list.append(transforms.Compose([
            transforms.RandomVerticalFlip(p=1.0),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))
        tta_list.append(transforms.Compose([
            transforms.RandomRotation((90, 90)),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))
    
    if n_aug >= 8:
        for _ in range(n_aug - 4):
            tta_list.append(transforms.Compose([
                transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
                transforms.RandomCrop(IMG_SIZE),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))
    
    return tta_list

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════
def make_loaders(train_dir, val_dir):
    """Create training and validation data loaders."""
    train_ds = datasets.ImageFolder(train_dir, transform=get_train_transform())
    val_ds = datasets.ImageFolder(val_dir, transform=get_val_transform())
    
    # Class-balanced sampling
    counts = Counter(train_ds.targets)
    weights = [1.0 / counts[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY, 
        drop_last=True,
        persistent_workers=False
    )
    
    val_dl = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        persistent_workers=False
    )
    
    return train_dl, val_dl, train_ds.classes, counts

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE - ConvNeXt-Tiny
# ═══════════════════════════════════════════════════════════════════════════════
def build_model(num_classes=5, freeze_backbone=True):
    """Build ConvNeXt-Tiny model with custom classifier."""
    print(f"\n🏗️  Building ConvNeXt-Tiny model...")
    
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    in_features = model.classifier[2].in_features
    
    # Custom classifier with dropout and layer norm
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.LayerNorm(in_features, eps=1e-6),
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.LayerNorm(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# MIXUP & CUTMIX AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
def mixup(x, y, alpha=MIXUP_ALPHA):
    """Mixup augmentation."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix(x, y, alpha=CUTMIX_ALPHA):
    """CutMix augmentation."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=device)
    W, H = x.size(3), x.size(2)
    rw = int(W * (1 - lam) ** 0.5)
    rh = int(H * (1 - lam) ** 0.5)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1 = max(cx - rw // 2, 0)
    x2 = min(cx + rw // 2, W)
    y1 = max(cy - rh // 2, 0)
    y2 = min(cy + rh // 2, H)
    
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, y, y[idx], lam

def mixed_loss(criterion, output, target_a, target_b, lam):
    """Compute mixed loss for mixup/cutmix."""
    return lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion, scaler, epoch, use_augmentation=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply mixup/cutmix after warmup
        if use_augmentation:
            if random.random() < CUTMIX_PROB:
                images, labels_a, labels_b, lam = cutmix(images, labels)
            else:
                images, labels_a, labels_b, lam = mixup(images, labels)
        
        with autocast("cuda"):
            outputs = model(images)
            
            if use_augmentation:
                loss = mixed_loss(criterion, outputs, labels_a, labels_b, lam)
                predictions = outputs.argmax(1)
                correct = (lam * (predictions == labels_a).float() + 
                          (1 - lam) * (predictions == labels_b).float()).sum().item()
            else:
                loss = criterion(outputs, labels)
                predictions = outputs.argmax(1)
                correct = (predictions == labels).sum().item()
            
            # Gradient accumulation
            loss = loss / ACCUM_STEPS
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * ACCUM_STEPS * images.size(0)
        total_correct += correct
        total_samples += images.size(0)
    
    return total_loss / total_samples, total_correct / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        predictions = outputs.argmax(1)
        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)
        
        all_preds.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    return total_loss / total_samples, total_correct / total_samples, all_preds, all_labels

@torch.no_grad()
def evaluate_tta(model, val_dir, class_names, n_aug=8):
    """Evaluate with Test-Time Augmentation."""
    print(f"\n🔄 Running TTA evaluation with {n_aug} augmentations...")
    
    model.eval()
    tta_transforms = get_tta_transforms(n_aug)
    
    val_dataset = datasets.ImageFolder(val_dir, transform=None)
    total_correct = 0
    total_samples = len(val_dataset)
    all_preds = []
    all_labels = []
    
    for img_path, label in val_dataset.imgs:
        image = Image.open(img_path).convert('RGB')
        
        # Accumulate predictions from all TTA augmentations
        logits_sum = torch.zeros(NUM_CLASSES, device=device)
        
        for tf in tta_transforms:
            img_tensor = tf(image).unsqueeze(0).to(device)
            with autocast("cuda"):
                logits = model(img_tensor)
            logits_sum += F.softmax(logits, dim=1).squeeze(0)
        
        # Average predictions
        avg_probs = logits_sum / len(tta_transforms)
        pred = avg_probs.argmax().item()
        
        all_preds.append(pred)
        all_labels.append(label)
        
        if pred == label:
            total_correct += 1
    
    accuracy = total_correct / total_samples
    return accuracy, all_preds, all_labels

# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════
def print_class_report(predictions, labels, class_names, title=""):
    """Print per-class accuracy report."""
    n_classes = len(class_names)
    confusion = [[0] * n_classes for _ in range(n_classes)]
    
    for pred, label in zip(predictions, labels):
        confusion[label][pred] += 1
    
    correct_counts = Counter()
    total_counts = Counter()
    
    for pred, label in zip(predictions, labels):
        total_counts[label] += 1
        if pred == label:
            correct_counts[label] += 1
    
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    
    for i, name in enumerate(class_names):
        acc = 100.0 * correct_counts[i] / max(1, total_counts[i])
        bar_len = int(acc / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {name:>8}: {correct_counts[i]:>3}/{total_counts[i]:<3}  {acc:5.1f}%  {bar}")
    
    overall = sum(correct_counts.values()) / max(1, sum(total_counts.values()))
    print(f"{'-'*70}")
    print(f"  {'OVERALL':>8}: {overall*100:.2f}%")
    print(f"{'='*70}")
    
    # Confusion matrix
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    header = "          " + "".join(f" {cn:>4}" for cn in class_names)
    print(header)
    for i, cn in enumerate(class_names):
        row = f"  {cn:>6} → " + "".join(f" {confusion[i][j]:>4}" for j in range(n_classes))
        print(row)
    print()
    
    return overall

def save_checkpoint(model, class_names, accuracy, path=SAVE_PATH):
    """Save model checkpoint."""
    torch.save({
        "state_dict": model.state_dict(),
        "classes": class_names,
        "backbone": "convnext_tiny",
        "val_acc": accuracy,
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
    }, path)
    print(f"   💾 Saved: {path} (val_acc={accuracy*100:.2f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════
def get_lr(epoch, max_epochs, lr_max, lr_min, warmup_epochs):
    """Cosine annealing with warmup."""
    if epoch < warmup_epochs:
        return lr_max * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(np.pi * progress))

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Prepare dataset from CLASS folders
    train_dir, val_dir = prepare_dataset(CLASS_FOLDERS)
    
    # Create data loaders
    train_dl, val_dl, class_names, counts = make_loaders(train_dir, val_dir)
    n_train = len(train_dl.dataset)
    n_val = len(val_dl.dataset)
    
    print(f"\n📊 Dataset Information:")
    print(f"   Classes: {class_names}")
    print(f"   Train samples: {n_train}")
    print(f"   Val samples: {n_val}")
    print(f"   Class distribution: {dict(counts)}")
    print(f"   Physical batch size: {BATCH_SIZE}")
    print(f"   Effective batch size: {BATCH_SIZE * ACCUM_STEPS}")
    print(f"   Steps per epoch: {len(train_dl)}")
    
    # Build model
    model = build_model(NUM_CLASSES, freeze_backbone=True)
    
    # Loss function with class weights
    class_weights = torch.tensor(
        [n_train / max(1, counts[i]) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
        device=device
    )
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(
            gamma=FOCAL_GAMMA,
            alpha=FOCAL_ALPHA,
            weight=class_weights,
            label_smoothing=LABEL_SMOOTH
        )
        criterion_eval = nn.CrossEntropyLoss(weight=class_weights)
        print(f"   Using Focal Loss (gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
        criterion_eval = criterion
        print(f"   Using Cross-Entropy Loss")
    
    # Mixed precision scaler
    scaler = GradScaler("cuda")
    
    # Best model tracking
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    start_time = time.time()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: WARM UP HEAD ONLY
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  🔥 PHASE 1: Head Warmup ({PHASE1_EPOCHS} epochs)")
    print(f"{'─'*70}")
    
    optimizer_phase1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_MAX * 0.5,
        weight_decay=WEIGHT_DECAY
    )
    
    for epoch in range(PHASE1_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_dl, optimizer_phase1, criterion, scaler, epoch, use_augmentation=False
        )
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_dl, criterion_eval)
        
        flag = " ⭐ BEST" if val_acc > best_acc else ""
        elapsed = (time.time() - start_time) / 60
        
        print(f"  [{epoch+1:>2}/{PHASE1_EPOCHS}]  "
              f"train={train_acc:.4f}({train_loss:.3f})  "
              f"val={val_acc:.4f}({val_loss:.3f})  "
              f"[{elapsed:.1f}m]{flag}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            save_checkpoint(model, class_names, val_acc)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: FULL TRAINING WITH AUGMENTATION
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  🚀 PHASE 2: Full Training ({PHASE2_EPOCHS} epochs, Mixup+CutMix)")
    print(f"{'─'*70}")
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Separate learning rates for backbone and head
    head_keywords = ["classifier"]
    backbone_params = [p for n, p in model.named_parameters() 
                      if not any(k in n for k in head_keywords)]
    head_params = [p for n, p in model.named_parameters() 
                  if any(k in n for k in head_keywords)]
    
    optimizer_phase2 = optim.AdamW([
        {"params": backbone_params, "lr": LR_MAX * 0.1},
        {"params": head_params, "lr": LR_MAX},
    ], weight_decay=WEIGHT_DECAY)
    
    # SWA model
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer_phase2, swa_lr=SWA_LR)
    swa_started = False
    
    for epoch in range(PHASE2_EPOCHS):
        # Update learning rate
        if epoch < SWA_START:
            lr = get_lr(epoch, SWA_START, LR_MAX, LR_MIN, WARMUP_EPOCHS)
            for i, param_group in enumerate(optimizer_phase2.param_groups):
                param_group['lr'] = lr * (0.1 if i == 0 else 1.0)
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_dl, optimizer_phase2, criterion, scaler, epoch, 
            use_augmentation=(epoch >= WARMUP_EPOCHS)
        )
        
        # SWA update
        if epoch >= SWA_START:
            if not swa_started:
                print(f"   🔄 Starting Stochastic Weight Averaging...")
                swa_started = True
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # Validation
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_dl, criterion_eval)
        
        flag = " ⭐ BEST" if val_acc > best_acc else ""
        elapsed = (time.time() - start_time) / 60
        
        print(f"  [{epoch+1:>3}/{PHASE2_EPOCHS}]  "
              f"lr={optimizer_phase2.param_groups[-1]['lr']:.2e}  "
              f"train={train_acc:.4f}({train_loss:.3f})  "
              f"val={val_acc:.4f}({val_loss:.3f})  "
              f"[{elapsed:.1f}m]{flag}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            save_checkpoint(model, class_names, val_acc)
            
            # Print per-class accuracy when improving and above 70%
            if val_acc > 0.70:
                print_class_report(val_preds, val_labels, class_names, 
                                 f"Per-Class Accuracy (Epoch {epoch+1})")
        else:
            patience_counter += 1
        
        # Check early stopping
        if val_acc >= TARGET_ACC:
            print(f"\n   🎯 TARGET ACCURACY {TARGET_ACC*100:.0f}% REACHED!")
            break
        
        if patience_counter >= PATIENCE:
            print(f"\n   ⏸️  Early stopping: no improvement for {PATIENCE} epochs")
            break
    
    # Update SWA batch normalization
    if swa_started:
        print(f"\n   🔄 Updating SWA batch normalization...")
        torch.optim.swa_utils.update_bn(train_dl, swa_model, device=device)
        
        # Evaluate SWA model
        _, swa_acc, swa_preds, swa_labels = evaluate(swa_model.module, val_dl, criterion_eval)
        print(f"   SWA validation accuracy: {swa_acc*100:.2f}%")
        
        if swa_acc > best_acc:
            print(f"   ✓ SWA model better than best checkpoint!")
            best_acc = swa_acc
            best_weights = copy.deepcopy(swa_model.module.state_dict())
            model = swa_model.module
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL EVALUATION WITH TTA
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  📊 FINAL EVALUATION")
    print(f"{'='*70}")
    
    # Load best weights
    model.load_state_dict(best_weights)
    
    # Standard evaluation
    _, final_acc, final_preds, final_labels = evaluate(model, val_dl, criterion_eval)
    print(f"\n   Standard evaluation: {final_acc*100:.2f}%")
    print_class_report(final_preds, final_labels, class_names, "Standard Evaluation")
    
    # TTA evaluation
    tta_acc, tta_preds, tta_labels = evaluate_tta(model, val_dir, class_names, n_aug=TTA_AUGS)
    print(f"\n   TTA evaluation ({TTA_AUGS} augmentations): {tta_acc*100:.2f}%")
    print_class_report(tta_preds, tta_labels, class_names, "TTA Evaluation")
    
    # Save final model
    final_best_acc = max(final_acc, tta_acc)
    save_checkpoint(model, class_names, final_best_acc)
    
    # Training summary
    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"   Total time: {total_time:.1f} minutes")
    print(f"   Best validation accuracy: {final_best_acc*100:.2f}%")
    print(f"   Model saved to: {SAVE_PATH}")
    
    if final_best_acc >= TARGET_ACC:
        print(f"\n   🎉 SUCCESS! Target accuracy {TARGET_ACC*100:.0f}% achieved!")
        print(f"   Ready for deployment with camera_classifier.py")
    else:
        print(f"\n   💡 Reached {final_best_acc*100:.2f}%")
        print(f"   Consider: more training epochs, more data, or different augmentation")
    
    # Cleanup temp dataset
    print(f"\n   🧹 Cleaning up temporary dataset...")
    try:
        import shutil
        shutil.rmtree("dataset_temp")
        print(f"   ✓ Cleanup complete")
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n   ⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
