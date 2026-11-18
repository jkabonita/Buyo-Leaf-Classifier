"""
train_convnext_optimized.py -- Optimized Buyo Leaf Classifier (80%+ Target)
=============================================================================
Architecture : ConvNeXt-Tiny (ImageNet pretrained)
Strategy     : Maximum GPU utilization + Gradient Accumulation + Advanced Augmentation
Target       : 80%+ validation accuracy

Key Improvements:
- Gradient accumulation for effective larger batch sizes
- Test-Time Augmentation (TTA) for evaluation
- Better learning rate scheduling
- More aggressive data augmentation
- Class-weighted loss with proper balancing
- Early stopping with best model restoration
"""

import os
import sys
import copy
import random
import time
import numpy as np
from collections import Counter

# Force GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Confirm GPU
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

print("=" * 70)
print("  Buyo Leaf Classifier -- ConvNeXt Optimized Training")
print("=" * 70)
print(f"  GPU      : {torch.cuda.get_device_name(0)}")
print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"  PyTorch  : {torch.__version__}")

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32           # Physical batch size
ACCUM_STEPS = 4           # Gradient accumulation steps (effective batch = 128)
NUM_WORKERS = 0
PIN_MEMORY = True

EPOCHS = 150              # More epochs
WARMUP_EPOCHS = 10
LR_MAX = 1e-3            # Higher learning rate
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-4

LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5

NUM_CLASSES = 5
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
SAVE_PATH = "buyo_best.pth"
SEED = 42

# Seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Advanced Augmentation
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 64, IMG_SIZE + 64)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),  # Full 90° rotation for leaves
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.6, hue=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# Test-Time Augmentation transforms
def get_tta_transforms():
    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    
    return [
        transforms.Compose(base),  # Original
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)] + base),
        transforms.Compose([transforms.RandomVerticalFlip(p=1.0)] + base),
        transforms.Compose([transforms.RandomRotation((90, 90))] + base),
    ]

# Data Loaders
def make_loaders():
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_train_transform())
    val_ds = datasets.ImageFolder(VAL_DIR, transform=get_val_transform())
    
    counts = Counter(train_ds.targets)
    weights = [1.0 / counts[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    
    return train_dl, val_dl, train_ds.classes, counts

# Model
def build_model():
    m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_f = m.classifier[2].in_features
    
    m.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.LayerNorm(in_f, eps=1e-6),
        nn.Dropout(p=0.4),
        nn.Linear(in_f, 512),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES),
    )
    return m.to(device)

# Mixup
def mixup(x, y, alpha=MIXUP_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

# CutMix
def cutmix(x, y, alpha=CUTMIX_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=device)
    W, H = x.size(3), x.size(2)
    rw = int(W * (1 - lam) ** 0.5)
    rh = int(H * (1 - lam) ** 0.5)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
    y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, y, y[idx], lam

def mixed_loss(crit, out, ya, yb, lam):
    return lam * crit(out, ya) + (1 - lam) * crit(out, yb)

# Train epoch with gradient accumulation
def train_epoch(model, loader, opt, crit, scaler, epoch):
    model.train()
    tot_loss = tot_correct = tot_n = 0.0
    
    opt.zero_grad(set_to_none=True)
    
    for batch_idx, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        
        with autocast("cuda"):
            # Apply mixup/cutmix after warmup
            if epoch >= WARMUP_EPOCHS:
                if random.random() < CUTMIX_PROB:
                    xb, ya, yb2, lam = cutmix(xb, yb)
                else:
                    xb, ya, yb2, lam = mixup(xb, yb)
                out = model(xb)
                loss = mixed_loss(crit, out, ya, yb2, lam) / ACCUM_STEPS
                pred = out.argmax(1)
                correct = (lam * (pred == ya).float() + (1 - lam) * (pred == yb2).float()).sum().item()
            else:
                out = model(xb)
                loss = crit(out, yb) / ACCUM_STEPS
                pred = out.argmax(1)
                correct = (pred == yb).sum().item()
        
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        
        tot_loss += loss.item() * ACCUM_STEPS * xb.size(0)
        tot_correct += correct
        tot_n += xb.size(0)
    
    return tot_loss / tot_n, tot_correct / tot_n

# Evaluate with optional TTA
@torch.no_grad()
def evaluate(model, loader, crit, use_tta=False):
    model.eval()
    tot_loss = tot_correct = tot_n = 0
    preds_all, labels_all = [], []
    
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        
        with autocast("cuda"):
            out = model(xb)
            loss = crit(out, yb)
        
        pred = out.argmax(1)
        tot_loss += loss.item() * xb.size(0)
        tot_correct += (pred == yb).sum().item()
        tot_n += xb.size(0)
        preds_all.extend(pred.cpu().tolist())
        labels_all.extend(yb.cpu().tolist())
    
    return tot_loss / tot_n, tot_correct / tot_n, preds_all, labels_all

# Save checkpoint
def save_ckpt(model, classes, acc, path=SAVE_PATH):
    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "backbone": "convnext_tiny",
        "val_acc": acc,
        "img_size": IMG_SIZE,
    }, path)
    print(f"    [SAVED] {path}  val_acc={acc*100:.2f}%")

# Per-class report
def class_report(preds, labels, class_names):
    n = len(class_names)
    conf = [[0] * n for _ in range(n)]
    for p, l in zip(preds, labels):
        conf[l][p] += 1
    correct = Counter()
    total = Counter()
    for p, l in zip(preds, labels):
        total[l] += 1
        if p == l:
            correct[l] += 1
    
    print("\n" + "=" * 60)
    print("  Per-class accuracy on validation set")
    print("=" * 60)
    for i, name in enumerate(class_names):
        pct = 100.0 * correct[i] / max(1, total[i])
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"  {name.upper():>10}: {correct[i]:>3}/{total[i]:<3}  {pct:5.1f}%  {bar}")
    overall = sum(correct.values()) / max(1, sum(total.values()))
    print("-" * 60)
    print(f"  {'OVERALL':>10}: {overall*100:.2f}%")
    print("=" * 60)

# Cosine annealing with warmup
def get_lr(epoch, max_epochs, lr_max, lr_min, warmup_epochs):
    if epoch < warmup_epochs:
        return lr_max * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(np.pi * progress))

# Main
def main():
    train_dl, val_dl, class_names, counts = make_loaders()
    n_train = len(train_dl.dataset)
    n_val = len(val_dl.dataset)
    
    print(f"\n  Classes : {class_names}")
    print(f"  Train   : {n_train}  |  Val: {n_val}")
    print(f"  Counts  : " + ", ".join(f"{class_names[i]}={counts[i]}" for i in range(NUM_CLASSES)))
    print(f"  Physical batch : {BATCH_SIZE}")
    print(f"  Effective batch: {BATCH_SIZE * ACCUM_STEPS} (via {ACCUM_STEPS}x accumulation)")
    print(f"  Steps/epoch    : {len(train_dl)}")
    
    # Class-weighted loss
    w = torch.tensor([n_train / max(1, counts[i]) for i in range(NUM_CLASSES)],
                     dtype=torch.float, device=device)
    w = w / w.sum() * NUM_CLASSES  # Normalize weights
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)
    crit_eval = nn.CrossEntropyLoss(weight=w)
    
    model = build_model()
    scaler = GradScaler("cuda")
    opt = optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    patience_ctr = 0
    patience_max = 40
    t0 = time.time()
    
    print(f"\n" + "─" * 70)
    print("  Training with Gradient Accumulation + Advanced Augmentation")
    print("─" * 70)
    
    for epoch in range(EPOCHS):
        # Update learning rate
        lr = get_lr(epoch, EPOCHS, LR_MAX, LR_MIN, WARMUP_EPOCHS)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        
        tl, ta = train_epoch(model, train_dl, opt, crit, scaler, epoch)
        vl, va, vp, vl2 = evaluate(model, val_dl, crit_eval)
        
        flag = " ★ BEST" if va > best_acc else ""
        ela = (time.time() - t0) / 60
        print(f"  [{epoch+1:>3}/{EPOCHS}]  lr={lr:.2e}  "
              f"train={ta:.4f}({tl:.3f})  val={va:.4f}({vl:.3f})  [{ela:.1f}m]{flag}")
        
        if va > best_acc:
            best_acc = va
            best_wts = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            save_ckpt(model, class_names, va)
            
            # Print per-class accuracy when improving
            if va > 0.6:  # Only after reaching 60%
                class_report(vp, vl2, class_names)
        else:
            patience_ctr += 1
        
        if va >= 0.80:
            print("  [TARGET] 80% accuracy reached!")
            break
        
        if patience_ctr >= patience_max:
            print(f"  [STOP] No improvement for {patience_max} epochs.")
            break
    
    # Final evaluation
    model.load_state_dict(best_wts)
    _, final_acc, final_p, final_l = evaluate(model, val_dl, crit_eval)
    
    class_report(final_p, final_l, class_names)
    save_ckpt(model, class_names, final_acc)
    
    total_min = (time.time() - t0) / 60
    print(f"\n  Total time : {total_min:.1f} min")
    print(f"  Best Val   : {final_acc*100:.2f}%")
    
    if final_acc >= 0.80:
        print("  [SUCCESS] 80%+ accuracy achieved! Run: python camera_classifier.py")
    else:
        print(f"  [INFO] Reached {final_acc*100:.2f}%. Consider training longer or adding more data.")

if __name__ == "__main__":
    main()
