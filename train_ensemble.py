"""
train_ensemble.py -- Multi-Model Ensemble Training for Buyo Leaf Classification
=================================================================================
Trains 3 strong CNN architectures independently, then ensembles their predictions
via knowledge distillation into a single high-accuracy model for camera deployment.

MODELS TRAINED:
  1. EfficientNetV2-S  (21M params, strong on small datasets)
  2. ConvNeXt-Tiny     (28M params, modern convnet, excellent generalization)
  3. ResNet50          (25M params, robust baseline, good feature extraction)

ENSEMBLE STRATEGY:
  Phase A: Train each model independently with best practices
  Phase B: Ensemble soft-vote (average softmax) → pick best single model
  Phase C: Knowledge distillation — train final model using ensemble teacher

KEY TECHNIQUES:
  * AMP (FP16) for fast GPU training
  * Mixup + CutMix augmentation
  * Label smoothing (0.1)
  * WeightedRandomSampler for class imbalance
  * OneCycleLR (Phase 1) + CosineAnnealing (Phase 2)
  * Layer-wise learning rate decay
  * SWA (Stochastic Weight Averaging)
  * Knowledge Distillation with temperature scaling
  * Test-Time Augmentation (TTA) for final evaluation
  * Gradient clipping

GPU: NVIDIA RTX 3050 6GB — fits all 3 models one-at-a-time

Usage:
    python train_ensemble.py

Output:
    buyo_best.pth           — best single model (camera_classifier.py compatible)
    buyo_effnet.pth         — EfficientNetV2-S checkpoint
    buyo_convnext.pth       — ConvNeXt-Tiny checkpoint
    buyo_resnet.pth         — ResNet50 checkpoint
"""

import os, sys, copy, random, time, gc
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from collections import Counter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - OPTIMIZED FOR HARD CLASS LEARNING
# ═══════════════════════════════════════════════════════════════════════════════
IMG_SIZE       = 256         # LARGER images for more detail
BATCH_SIZE     = 24          # Smaller batch = more gradient updates
NUM_CLASSES    = 5

PHASE1_EPOCHS  = 35          # Longer head warm-up for better initialization
PHASE2_EPOCHS  = 150         # Extended fine-tune
SWA_EPOCHS     = 15          # More SWA epochs

LR1_HEAD       = 3e-4        # Moderate initial LR
LR2_BACKBONE   = 2e-5        # Slightly higher backbone LR for better adaptation
LR2_HEAD       = 2e-4        # Higher head LR
LR_SWA         = 8e-6        # Higher SWA LR

WEIGHT_DECAY   = 1e-4        # Less regularization to allow learning
LABEL_SMOOTH   = 0.1         # Less smoothing to preserve class boundaries
MIXUP_ALPHA    = 0.3         # Moderate mixup
CUTMIX_ALPHA   = 0.8         # Moderate cutmix
CUTMIX_PROB    = 0.5         # Balanced mix
PATIENCE       = 40          # More patience for hard classes
TARGET_ACC     = 0.99

# CLASS B FOCUS TRAINING (boost underperforming class)
USE_FOCAL_LOSS = True        # Use focal loss for hard examples
FOCAL_GAMMA    = 2.5         # Higher gamma = more focus on hard examples
FOCAL_ALPHA    = None        # Auto-balanced per class

# ADVANCED AUGMENTATION FOR HARD CLASSES
USE_CLASS_SPECIFIC_AUG = True  # Apply stronger augmentation to hard classes

# Knowledge distillation
KD_TEMPERATURE = 4.0         # Softens teacher probabilities
KD_ALPHA       = 0.7         # Weight of distillation loss vs hard label loss
KD_EPOCHS      = 40          # Distillation training epochs

TRAIN_DIR      = "dataset/train"
VAL_DIR        = "dataset/val"
SEED           = 42

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

device = torch.device("cuda:0")
torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("=" * 64)
print("  BUYO LEAF ENSEMBLE TRAINER — Maximum Accuracy Mode")
print("=" * 64)
print(f"  GPU    : {torch.cuda.get_device_name(0)}")
print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"  PyTorch: {torch.__version__}")
print(f"  Models : EfficientNetV2-S, ConvNeXt-Tiny, ResNet50")
print(f"  Strategy: Independent training -> Ensemble -> Knowledge Distillation")
print("=" * 64)

# ═══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS FOR HARD EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """Focal Loss focuses on hard-to-classify examples (like CLASS B)"""
    def __init__(self, gamma=2.0, alpha=None, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, 
                                   label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE WITH CLASS-SPECIFIC AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
class ClassAwareImageFolder(datasets.ImageFolder):
    """Custom dataset that applies stronger augmentation to hard classes"""
    def __init__(self, root, transform=None, hard_transform=None, hard_classes=None):
        super().__init__(root, transform=transform)
        self.hard_transform = hard_transform
        self.hard_classes = hard_classes or []
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        # Apply stronger augmentation to hard classes
        if self.hard_transform is not None and target in self.hard_classes:
            sample = self.hard_transform(sample)
        elif self.transform is not None:
            sample = self.transform(sample)
        
        return sample, target

def make_loaders():
    # Standard augmentation
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 48, IMG_SIZE + 48)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.4),
        transforms.RandomRotation(45),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=12),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.12),
        transforms.RandomGrayscale(0.05),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.RandomAutocontrast(p=0.25),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.30, scale=(0.02, 0.18)),
    ])
    
    # STRONGER augmentation for hard classes (CLASS B, C, D)
    hard_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 56, IMG_SIZE + 56)),  # More crop area
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(0.6),  # More flips
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(70),  # More rotation
        transforms.RandomAffine(degrees=0, translate=(0.25, 0.25), scale=(0.75, 1.25), shear=20),  # Stronger
        transforms.RandomPerspective(distortion_scale=0.4, p=0.6),  # Added perspective
        transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.6, hue=0.18),  # Stronger color
        transforms.RandomGrayscale(0.1),
        transforms.GaussianBlur(5, sigma=(0.1, 3.0)),  # Stronger blur
        transforms.RandomAutocontrast(p=0.35),
        transforms.RandomEqualize(p=0.25),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.45, scale=(0.02, 0.25)),  # More erasing
    ])
    hard_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 56, IMG_SIZE + 56)),  # More crop area
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(0.6),  # More flips
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(70),  # More rotation
        transforms.RandomAffine(degrees=0, translate=(0.25, 0.25), scale=(0.75, 1.25), shear=20),  # Stronger
        transforms.RandomPerspective(distortion_scale=0.4, p=0.6),  # Added perspective
        transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.6, hue=0.18),  # Stronger color
        transforms.RandomGrayscale(0.1),
        transforms.GaussianBlur(5, sigma=(0.1, 3.0)),  # Stronger blur
        transforms.RandomAutocontrast(p=0.35),
        transforms.RandomEqualize(p=0.25),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.45, scale=(0.02, 0.25)),  # More erasing
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Identify hard classes (CLASS B=1, CLASS C=2, CLASS D=3)
    hard_classes = [1, 2, 3]  # Indices for CLASS B, C, D
    
    if USE_CLASS_SPECIFIC_AUG:
        train_ds = ClassAwareImageFolder(TRAIN_DIR, transform=train_tf, 
                                        hard_transform=hard_tf, hard_classes=hard_classes)
    else:
        train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tf)

    counts  = Counter(train_ds.targets)
    weights = [1.0 / counts[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)
    return train_dl, val_dl, train_ds.classes, counts

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════
def build_efficientnet(nc, freeze=True):
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    if freeze:
        for p in m.parameters(): p.requires_grad = False
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.35, inplace=True),
        nn.Linear(in_f, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),  # ADDED: BatchNorm for stability
        nn.Dropout(p=0.25),
        nn.Linear(512, 256),
        nn.SiLU(),
        nn.Dropout(p=0.15),
        nn.Linear(256, nc),
    )
    return m.to(device), "efficientnet_v2_s"

def build_convnext(nc, freeze=True):
    m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    if freeze:
        for p in m.parameters(): p.requires_grad = False
    in_f = m.classifier[2].in_features
    m.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.LayerNorm(in_f, eps=1e-6),
        nn.Dropout(p=0.35),
        nn.Linear(in_f, 512),
        nn.GELU(),
        nn.LayerNorm(512),  # ADDED: LayerNorm for stability
        nn.Dropout(p=0.25),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(p=0.15),
        nn.Linear(256, nc),
    )
    return m.to(device), "convnext_tiny"

def build_resnet(nc, freeze=True):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if freeze:
        for p in m.parameters(): p.requires_grad = False
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),  # ADDED: BatchNorm for stability
        nn.Dropout(p=0.20),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.15),
        nn.Linear(256, nc),
    )
    return m.to(device), "resnet50"

MODEL_BUILDERS = [
    ("EfficientNetV2-S", build_efficientnet, "buyo_effnet.pth"),
    ("ConvNeXt-Tiny",    build_convnext,     "buyo_convnext.pth"),
    ("ResNet50",         build_resnet,       "buyo_resnet.pth"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION: MIXUP + CUTMIX
# ═══════════════════════════════════════════════════════════════════════════════
def mixup(x, y):
    lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix(x, y):
    lam = float(np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA))
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

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, opt, crit, scaler, use_mix=False):
    model.train()
    tot_loss = tot_correct = tot_n = 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast("cuda"):
            if use_mix:
                if random.random() < CUTMIX_PROB:
                    xb, ya, yb2, lam = cutmix(xb, yb)
                else:
                    xb, ya, yb2, lam = mixup(xb, yb)
                out  = model(xb)
                loss = mixed_loss(crit, out, ya, yb2, lam)
                pred = out.argmax(1)
                correct = (lam * (pred == ya).float() +
                           (1 - lam) * (pred == yb2).float()).sum().item()
            else:
                out  = model(xb)
                loss = crit(out, yb)
                pred = out.argmax(1)
                correct = (pred == yb).sum().item()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        tot_loss += loss.item() * xb.size(0)
        tot_correct += correct
        tot_n += xb.size(0)
    return tot_loss / tot_n, tot_correct / tot_n

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    tot_loss = tot_correct = tot_n = 0
    preds_all, labels_all = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with autocast("cuda"):
            out  = model(xb)
            loss = crit(out, yb)
        pred = out.argmax(1)
        tot_loss += loss.item() * xb.size(0)
        tot_correct += (pred == yb).sum().item()
        tot_n += xb.size(0)
        preds_all.extend(pred.cpu().tolist())
        labels_all.extend(yb.cpu().tolist())
    return tot_loss / tot_n, tot_correct / tot_n, preds_all, labels_all

@torch.no_grad()
def evaluate_tta(model, n_aug=5):
    """Test-Time Augmentation: average predictions over multiple augmented views."""
    model.eval()
    tta_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    base_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_ds = datasets.ImageFolder(VAL_DIR, transform=None)

    correct = total = 0
    preds_all, labels_all = [], []

    for img, label in val_ds:
        x0 = base_tf(img).unsqueeze(0).to(device)
        with autocast("cuda"):
            logits = model(x0)
        avg_probs = F.softmax(logits, dim=1)

        for _ in range(n_aug):
            xa = tta_tf(img).unsqueeze(0).to(device)
            with autocast("cuda"):
                la = model(xa)
            avg_probs += F.softmax(la, dim=1)

        avg_probs /= (n_aug + 1)
        pred = avg_probs.argmax(1).item()
        preds_all.append(pred)
        labels_all.append(label)
        if pred == label: correct += 1
        total += 1

    return correct / total, preds_all, labels_all

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE / REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def save_ckpt(model, classes, acc, backbone, path):
    torch.save({
        "state_dict": model.state_dict(),
        "classes":    classes,
        "backbone":   backbone,
        "val_acc":    acc,
        "img_size":   IMG_SIZE,
    }, path)

def class_report(preds, labels, class_names, title=""):
    n = len(class_names)
    conf = [[0] * n for _ in range(n)]
    for p, l in zip(preds, labels):
        conf[l][p] += 1
    correct = Counter()
    total   = Counter()
    for p, l in zip(preds, labels):
        total[l] += 1
        if p == l: correct[l] += 1
    if title:
        print(f"\n{'='*58}")
        print(f"  {title}")
        print(f"{'='*58}")
    for i, name in enumerate(class_names):
        pct = 100.0 * correct[i] / max(1, total[i])
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"  {name.upper():>10}: {correct[i]:>3}/{total[i]:<3} {pct:5.1f}%  {bar}")
    overall = sum(correct.values()) / max(1, sum(total.values()))
    print(f"{'-'*58}")
    print(f"  {'OVERALL':>10}: {overall*100:.2f}%")

    # Confusion matrix
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    hdr = "         " + "".join(f" {cn[-1]:>3}" for cn in class_names)
    print(hdr)
    for i, cn in enumerate(class_names):
        row = f"  {cn[-1]:>5} -> " + "".join(f" {conf[i][j]:>3}" for j in range(n))
        print(row)
    return overall

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE MODEL (full pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
def train_model(name, builder, save_path, train_dl, val_dl, class_names, counts, crit, crit_eval):
    print(f"\n{'#'*64}")
    print(f"  TRAINING: {name}")
    print(f"{'#'*64}")

    model, backbone = builder(NUM_CLASSES, freeze=True)
    scaler   = GradScaler("cuda")
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    t0       = time.time()

    # -- Phase 1: Head warm-up -------------------------------------------------
    print(f"\n  Phase 1 -- Head only ({PHASE1_EPOCHS} epochs)")
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR1_HEAD, weight_decay=WEIGHT_DECAY)
    sch1 = optim.lr_scheduler.OneCycleLR(opt1, max_lr=LR1_HEAD,
                                          steps_per_epoch=len(train_dl),
                                          epochs=PHASE1_EPOCHS, pct_start=0.3)
    for ep in range(PHASE1_EPOCHS):
        tl, ta = train_epoch(model, train_dl, opt1, crit, scaler, use_mix=False)
        vl, va, _, _ = evaluate(model, val_dl, crit_eval)
        sch1.step()
        flag = " *BEST*" if va > best_acc else ""
        print(f"    P1 {ep+1:>2}/{PHASE1_EPOCHS}  t={ta:.4f}  v={va:.4f}{flag}  [{(time.time()-t0)/60:.1f}m]")
        if va > best_acc:
            best_acc = va
            best_wts = copy.deepcopy(model.state_dict())

    # -- Phase 2: Full fine-tune -----------------------------------------------
    print(f"\n  Phase 2 -- Full fine-tune ({PHASE2_EPOCHS} epochs, Mixup+CutMix)")
    for p in model.parameters(): p.requires_grad = True

    head_keywords = ["classifier", "fc", "head"]
    backbone_p = [p for n, p in model.named_parameters()
                  if not any(k in n for k in head_keywords)]
    head_p     = [p for n, p in model.named_parameters()
                  if any(k in n for k in head_keywords)]

    opt2 = optim.AdamW([
        {"params": backbone_p, "lr": LR2_BACKBONE},
        {"params": head_p,     "lr": LR2_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    sch2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=20, T_mult=2, eta_min=1e-7)

    patience_ctr = 0
    best_class_b_acc = 0.0  # Track CLASS B accuracy
    
    for ep in range(PHASE2_EPOCHS):
        tl, ta = train_epoch(model, train_dl, opt2, crit, scaler, use_mix=True)
        vl, va, vp, vl2 = evaluate(model, val_dl, crit_eval)
        
        # Calculate per-class accuracy
        class_correct = Counter()
        class_total = Counter()
        for pred, label in zip(vp, vl2):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
        
        class_b_acc = class_correct[1] / max(1, class_total[1])  # CLASS B index = 1
        
        sch2.step()
        flag = " *BEST*" if va > best_acc else ""
        if class_b_acc > best_class_b_acc:
            flag += f" [B↑{class_b_acc*100:.1f}%]"
            best_class_b_acc = class_b_acc
        
        print(f"    P2 {ep+1:>2}/{PHASE2_EPOCHS}  t={ta:.4f}  v={va:.4f} B={class_b_acc*100:.1f}%{flag}  [{(time.time()-t0)/60:.1f}m]")
        
        if va > best_acc:
            best_acc = va; best_wts = copy.deepcopy(model.state_dict()); patience_ctr = 0
        else:
            patience_ctr += 1
        if va >= TARGET_ACC:
            print(f"    [TARGET] {TARGET_ACC*100:.0f}% reached!"); break
        if patience_ctr >= PATIENCE:
            print(f"    [STOP] No improvement for {PATIENCE} epochs."); break

    # -- Phase 3: SWA ----------------------------------------------------------
    print(f"\n  Phase 3 -- SWA ({SWA_EPOCHS} epochs)")
    model.load_state_dict(best_wts)
    swa_model = AveragedModel(model)
    swa_sch = SWALR(opt2, swa_lr=LR_SWA, anneal_epochs=max(1, SWA_EPOCHS // 3))
    for ep in range(SWA_EPOCHS):
        tl, ta = train_epoch(model, train_dl, opt2, crit, scaler, use_mix=True)
        swa_model.update_parameters(model)
        swa_sch.step()
        print(f"    SWA {ep+1:>2}/{SWA_EPOCHS}  t={ta:.4f}  [{(time.time()-t0)/60:.1f}m]")
    torch.optim.swa_utils.update_bn(train_dl, swa_model, device=device)

    _, swa_acc, swa_p, swa_l = evaluate(swa_model, val_dl, crit_eval)
    print(f"    SWA val={swa_acc*100:.2f}%  |  Best ckpt={best_acc*100:.2f}%")

    if swa_acc >= best_acc:
        final_model = swa_model.module; final_acc = swa_acc
    else:
        model.load_state_dict(best_wts); final_model = model; final_acc = best_acc

    # TTA evaluation
    tta_acc, tta_p, tta_l = evaluate_tta(final_model, n_aug=7)
    print(f"    TTA val={tta_acc*100:.2f}%")
    if tta_acc > final_acc:
        final_acc = tta_acc

    save_ckpt(final_model, class_names, final_acc, backbone, save_path)
    print(f"    [SAVED] {save_path}  ({backbone}, {final_acc*100:.2f}%)")

    elapsed = (time.time() - t0) / 60
    print(f"    Training time: {elapsed:.1f} min")

    # Free GPU memory
    del swa_model, opt1, opt2, scaler
    torch.cuda.empty_cache()
    gc.collect()

    return final_model, backbone, final_acc

# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def ensemble_evaluate(model_list, val_dl, class_names):
    """Average softmax predictions across all models."""
    for m in model_list: m.eval()
    correct = total = 0
    preds_all, labels_all = [], []

    for xb, yb in val_dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        avg_probs = torch.zeros(xb.size(0), NUM_CLASSES, device=device)
        for m in model_list:
            with autocast("cuda"):
                logits = m(xb)
            avg_probs += F.softmax(logits, dim=1)
        avg_probs /= len(model_list)
        pred = avg_probs.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        preds_all.extend(pred.cpu().tolist())
        labels_all.extend(yb.cpu().tolist())

    acc = correct / total
    return acc, preds_all, labels_all

# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE DISTILLATION
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def get_teacher_logits(teacher_models, xb):
    """Average logits from all teacher models."""
    avg = torch.zeros(xb.size(0), NUM_CLASSES, device=device)
    for m in teacher_models:
        m.eval()
        with autocast("cuda"):
            avg += m(xb)
    return avg / len(teacher_models)


def distillation_loss(student_logits, teacher_logits, labels, crit_hard, T, alpha):
    """Combined KD loss = alpha * soft_loss + (1 - alpha) * hard_loss."""
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss    = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    hard_loss    = crit_hard(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


def train_distilled(teacher_models, student_builder, train_dl, val_dl,
                    class_names, counts, crit_eval, save_path):
    """Train a student model using knowledge distillation from ensemble teachers."""
    print(f"\n{'#'*64}")
    print(f"  KNOWLEDGE DISTILLATION -- Ensemble -> Single Model")
    print(f"{'#'*64}")

    student, backbone = student_builder(NUM_CLASSES, freeze=False)
    n_train = len(train_dl.dataset)
    w = torch.tensor([n_train / max(1, counts[i]) for i in range(NUM_CLASSES)],
                     dtype=torch.float, device=device)
    crit_hard = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)
    scaler = GradScaler("cuda")

    head_keywords = ["classifier", "fc", "head"]
    backbone_p = [p for n, p in student.named_parameters()
                  if not any(k in n for k in head_keywords)]
    head_p     = [p for n, p in student.named_parameters()
                  if any(k in n for k in head_keywords)]

    opt = optim.AdamW([
        {"params": backbone_p, "lr": LR2_BACKBONE * 2},
        {"params": head_p,     "lr": LR2_HEAD * 2},
    ], weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=KD_EPOCHS, eta_min=1e-7)

    best_acc = 0.0
    best_wts = copy.deepcopy(student.state_dict())
    t0 = time.time()

    for ep in range(KD_EPOCHS):
        student.train()
        tot_loss = tot_correct = tot_n = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            teacher_logits = get_teacher_logits(teacher_models, xb)

            with autocast("cuda"):
                student_logits = student(xb)
                loss = distillation_loss(student_logits, teacher_logits, yb,
                                         crit_hard, KD_TEMPERATURE, KD_ALPHA)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            pred = student_logits.argmax(1)
            tot_loss += loss.item() * xb.size(0)
            tot_correct += (pred == yb).sum().item()
            tot_n += xb.size(0)

        sch.step()
        vl, va, vp, vl2 = evaluate(student, val_dl, crit_eval)
        flag = " *BEST*" if va > best_acc else ""
        print(f"    KD {ep+1:>2}/{KD_EPOCHS}  t={tot_correct/tot_n:.4f}  v={va:.4f}{flag}  [{(time.time()-t0)/60:.1f}m]")
        if va > best_acc:
            best_acc = va
            best_wts = copy.deepcopy(student.state_dict())
        if va >= TARGET_ACC:
            print(f"    [TARGET] {TARGET_ACC*100:.0f}% reached via distillation!")
            break

    student.load_state_dict(best_wts)

    # TTA on distilled model
    tta_acc, tta_p, tta_l = evaluate_tta(student, n_aug=7)
    print(f"    KD TTA val={tta_acc*100:.2f}%")
    final_acc = max(best_acc, tta_acc)

    save_ckpt(student, class_names, final_acc, backbone, save_path)
    print(f"    [SAVED] {save_path}  ({backbone}, {final_acc*100:.2f}%)")

    return student, backbone, final_acc

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    train_dl, val_dl, class_names, counts = make_loaders()
    n_train = len(train_dl.dataset)
    n_val   = len(val_dl.dataset)

    print(f"\n  Dataset  : {n_train} train, {n_val} val")
    print(f"  Classes  : {class_names}")
    print(f"  Counts   : " + ", ".join(f"{class_names[i]}={counts[i]}" for i in range(NUM_CLASSES)))
    print(f"  Batch    : {BATCH_SIZE}  |  Steps/epoch: {len(train_dl)}")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Focal Loss: {USE_FOCAL_LOSS}")
    print(f"  Class-Specific Aug: {USE_CLASS_SPECIFIC_AUG}")

    # Weighted loss with Focal Loss for hard examples
    w = torch.tensor([n_train / max(1, counts[i]) for i in range(NUM_CLASSES)],
                     dtype=torch.float, device=device)
    
    if USE_FOCAL_LOSS:
        crit = FocalLoss(gamma=FOCAL_GAMMA, alpha=None, weight=w, label_smoothing=LABEL_SMOOTH)
        crit_eval = FocalLoss(gamma=FOCAL_GAMMA, alpha=None, weight=w, label_smoothing=0.0)
        print(f"  Using Focal Loss (gamma={FOCAL_GAMMA}) to focus on hard examples")
    else:
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)
        crit_eval = nn.CrossEntropyLoss(weight=w)

    t_start = time.time()

    # == PHASE A: Train each model independently ==============================
    trained_models = []
    results = []

    for name, builder, save_path in MODEL_BUILDERS:
        model, backbone, acc = train_model(
            name, builder, save_path,
            train_dl, val_dl, class_names, counts, crit, crit_eval
        )
        trained_models.append(model)
        results.append((name, backbone, acc, save_path))
        _, _, preds, labels = evaluate(model, val_dl, crit_eval)
        class_report(preds, labels, class_names, title=f"{name} -- Per-Class Accuracy")

    # == PHASE B: Ensemble evaluation =========================================
    print(f"\n{'#'*64}")
    print(f"  ENSEMBLE EVALUATION (3-Model Soft Vote)")
    print(f"{'#'*64}")

    ens_acc, ens_p, ens_l = ensemble_evaluate(trained_models, val_dl, class_names)
    class_report(ens_p, ens_l, class_names,
                 title="Ensemble (3-Model Average) -- Per-Class Accuracy")
    print(f"\n  Ensemble accuracy: {ens_acc*100:.2f}%")

    # == PHASE C: Knowledge distillation ======================================
    best_idx = max(range(len(results)), key=lambda i: results[i][2])
    best_name, best_bb, best_single_acc, _ = results[best_idx]
    print(f"\n  Best individual model: {best_name} ({best_single_acc*100:.2f}%)")
    print(f"  Using {best_name} architecture as distillation student...")

    student_builder = MODEL_BUILDERS[best_idx][1]
    kd_model, kd_bb, kd_acc = train_distilled(
        trained_models, student_builder,
        train_dl, val_dl, class_names, counts, crit_eval,
        "buyo_distilled.pth"
    )

    # == FINAL: Pick overall winner ===========================================
    print(f"\n{'='*64}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*64}")

    all_results = results + [("KD-Distilled", kd_bb, kd_acc, "buyo_distilled.pth")]
    all_results.append(("Ensemble(3)", "ensemble", ens_acc, None))

    for name, bb, acc, path in all_results:
        pstr = path if path else "(soft-vote, no single .pth)"
        print(f"    {name:25s}  {acc*100:6.2f}%  {pstr}")

    # Determine which single-model checkpoint to save as buyo_best.pth
    single_results = [(n, b, a, p) for n, b, a, p in all_results if p]
    winner = max(single_results, key=lambda x: x[2])
    winner_name, winner_bb, winner_acc, winner_path = winner

    print(f"\n  >>> WINNER: {winner_name} at {winner_acc*100:.2f}% <<<")

    import shutil
    shutil.copy2(winner_path, "buyo_best.pth")
    print(f"  Copied {winner_path} -> buyo_best.pth")
    print(f"  camera_classifier.py will load: buyo_best.pth ({winner_bb})")

    # Final per-class report on the winner
    if winner_path == "buyo_distilled.pth":
        final_model = kd_model
    else:
        final_model = None
        for m, (n, b, a, p) in zip(trained_models, results):
            if p == winner_path:
                final_model = m; break
    if final_model is not None:
        _, _, wp, wl = evaluate(final_model, val_dl, crit_eval)
        class_report(wp, wl, class_names, title=f"FINAL: {winner_name} -- Per-Class")

    total_min = (time.time() - t_start) / 60
    print(f"\n  Total training time: {total_min:.1f} min")
    print(f"  Run: python camera_classifier.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
