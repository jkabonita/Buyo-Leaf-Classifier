"""
train_gpu.py -- Optimized Buyo Leaves Classifier Training Script
===============================================================
Architecture : EfficientNetV2-S (ImageNet pretrained)
Strategy     : Two-phase fine-tuning
  Phase 1    : 20 epochs -- frozen backbone, train classifier head only (LR=1e-3)
  Phase 2    : 80 epochs -- full unfreeze, end-to-end fine-tune (LR=1e-5)
Augmentation : Heavy online augmentation + Mixup in Phase 2
Balance      : WeightedRandomSampler + weighted CrossEntropyLoss
Checkpointing: Saves best val-accuracy checkpoint as buyo_best.pth
Goal         : >=99% validation accuracy

Usage:
    python train_gpu.py
"""

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import datasets, transforms, models
from collections import Counter

# -----------------------------------------------------------------------------
# HYPERPARAMETERS
# -----------------------------------------------------------------------------
IMG_SIZE        = 224
BATCH_SIZE      = 16        # Small batch for small dataset -- better gradient estimates
PHASE1_EPOCHS   = 20        # Warm-up: head only
PHASE2_EPOCHS   = 80        # Full fine-tune
LR_PHASE1       = 1e-3      # Higher LR for head training
LR_PHASE2       = 1e-5      # Low LR for full fine-tune to avoid catastrophic forgetting
WEIGHT_DECAY    = 1e-4
NUM_CLASSES     = 5
TRAIN_DIR       = 'dataset/train'
VAL_DIR         = 'dataset/val'
SAVE_PATH       = 'buyo_best.pth'
TARGET_ACC      = 0.99      # Stop early if this val accuracy is achieved
PATIENCE        = 20        # Early stopping patience in Phase 2
MIXUP_ALPHA     = 0.4       # Mixup interpolation strength
SEED            = 42

# -----------------------------------------------------------------------------
# REPRODUCIBILITY
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# MIXUP AUGMENTATION
# -----------------------------------------------------------------------------
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Interpolates two random samples in a mini-batch (Mixup)."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# -----------------------------------------------------------------------------
# DATA PIPELINE
# -----------------------------------------------------------------------------
def make_loaders(train_dir: str, val_dir: str, img_size: int, batch_size: int):
    """
    Constructs balanced DataLoaders with heavy online augmentation for training
    and standard normalization for validation.
    """
    train_transforms = transforms.Compose([
        # Slightly oversized resize then random crop -- simulates different zoom levels
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transforms)

   
    class_counts   = Counter(train_dataset.targets)
    sample_weights = [1.0 / class_counts[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = 0,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
    )

    return train_loader, val_loader, train_dataset.classes, class_counts


# -----------------------------------------------------------------------------
# MODEL BUILDER
# -----------------------------------------------------------------------------
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """
    EfficientNetV2-S pretrained on ImageNet.
    Classifier head is replaced for num_classes output.
    All layers are frozen initially; Phase 2 unfreezes everything.
    """
    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    # Freeze entire backbone
    for p in model.parameters():
        p.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


# -----------------------------------------------------------------------------
# TRAIN / EVALUATE LOOPS
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device,
                    use_mixup: bool = False):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0.0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup:
            inputs, y_a, y_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
            outputs = model(inputs)
            loss    = mixup_loss(criterion, outputs, y_a, y_b, lam)
            # Accuracy: weighted contribution of both labels
            preds   = outputs.argmax(dim=1)
            correct = (lam * (preds == y_a).float()
                       + (1 - lam) * (preds == y_b).float()).sum().item()
        else:
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            preds   = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()

        loss.backward()
        # Gradient clipping -- prevents exploding gradients during full fine-tune
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item() * inputs.size(0)
        total_correct += correct
        total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        preds   = outputs.argmax(dim=1)

        total_loss    += loss.item() * inputs.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += inputs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return (total_loss / total_samples,
            total_correct / total_samples,
            all_preds,
            all_labels)


# -----------------------------------------------------------------------------
# PER-CLASS REPORT
# -----------------------------------------------------------------------------
def print_per_class_accuracy(all_preds, all_labels, class_names):
    print('\n' + '-' * 50)
    print('  Per-Class Accuracy on Validation Set')
    print('-' * 50)
    class_correct = Counter()
    class_total   = Counter()
    for p, l in zip(all_preds, all_labels):
        class_total[l]   += 1
        if p == l:
            class_correct[l] += 1

    overall = sum(class_correct.values()) / max(1, sum(class_total.values()))
    for i, name in enumerate(class_names):
        n_correct = class_correct[i]
        n_total   = class_total[i]
        acc_pct   = 100.0 * n_correct / max(1, n_total)
        bar       = '#' * int(acc_pct / 5)
        print(f'  {name.upper():>10s}: {n_correct:>3}/{n_total:<3} '
              f'{acc_pct:>6.1f}%  {bar}')
    print('-' * 50)
    print(f'  {"OVERALL":>10s}: {100*overall:.2f}%')
    print('-' * 50)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 60)
    print('  Buyo Leaves Classifier -- Optimized Training')
    print('=' * 60)
    print(f'  Device  : {device}')
    if torch.cuda.is_available():
        print(f'  GPU     : {torch.cuda.get_device_name(0)}')
    print(f'  Model   : EfficientNetV2-S (ImageNet pretrained)')
    print(f'  Phase 1 : {PHASE1_EPOCHS} epochs | LR={LR_PHASE1} | Frozen backbone')
    print(f'  Phase 2 : {PHASE2_EPOCHS} epochs | LR={LR_PHASE2} | Full fine-tune + Mixup')
    print(f'  Target  : {TARGET_ACC*100:.0f}% val accuracy')
    print('=' * 60)

    # Data
    train_l, val_l, class_names, class_counts = make_loaders(
        TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE
    )
    n_train = len(train_l.dataset)
    n_val   = len(val_l.dataset)
    print(f'\n  Classes : {class_names}')
    print(f'  Train   : {n_train} images  |  Val: {n_val} images')
    print(f'  Class distribution (train): ' +
          ', '.join(f'{class_names[i]}={class_counts[i]}' for i in range(NUM_CLASSES)))

    # Weighted loss for class imbalance
    loss_weights = torch.tensor(
        [n_train / max(1, class_counts[i]) for i in range(NUM_CLASSES)],
        dtype=torch.float,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    model     = build_model(NUM_CLASSES, device)
    best_acc  = 0.0
    best_wts  = copy.deepcopy(model.state_dict())

    # --- PHASE 1: Classifier head training -----------------------------------
    print(f'\n{"-"*60}')
    print(f'  PHASE 1 -- Training Classifier Head Only')
    print(f'{"-"*60}')

    optimizer1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=PHASE1_EPOCHS, eta_min=1e-5
    )

    for epoch in range(PHASE1_EPOCHS):
        t_loss, t_acc = train_one_epoch(
            model, train_l, optimizer1, criterion, device, use_mixup=False
        )
        v_loss, v_acc, _, _ = evaluate(model, val_l, criterion, device)
        scheduler1.step()

        flag = ' [BEST] BEST' if v_acc > best_acc else ''
        print(f'  [P1] {epoch+1:>3}/{PHASE1_EPOCHS} | '
              f'Train: {t_acc:.4f} ({t_loss:.4f}) | '
              f'Val: {v_acc:.4f} ({v_loss:.4f}){flag}')

        if v_acc > best_acc:
            best_acc = v_acc
            best_wts = copy.deepcopy(model.state_dict())

        if v_acc >= TARGET_ACC:
            print(f'\n  [TARGET] Target {TARGET_ACC*100:.0f}% reached in Phase 1!')
            break

    # --- PHASE 2: Full fine-tuning --------------------------------------------
    print(f'\n{"-"*60}')
    print(f'  PHASE 2 -- Full Model Fine-Tuning + Mixup')
    print(f'{"-"*60}')

    # Unfreeze ALL parameters
    for p in model.parameters():
        p.requires_grad = True

    optimizer2 = optim.Adam(
        model.parameters(),
        lr=LR_PHASE2,
        weight_decay=WEIGHT_DECAY,
    )
    # Cosine warm restarts: revisit high LR briefly to escape local minima
    scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer2, T_0=20, T_mult=2, eta_min=1e-7
    )

    patience_counter = 0

    for epoch in range(PHASE2_EPOCHS):
        t_loss, t_acc = train_one_epoch(
            model, train_l, optimizer2, criterion, device, use_mixup=True
        )
        v_loss, v_acc, all_preds, all_labels = evaluate(
            model, val_l, criterion, device
        )
        scheduler2.step()

        improved = ' [BEST] BEST' if v_acc > best_acc else ''
        print(f'  [P2] {epoch+1:>3}/{PHASE2_EPOCHS} | '
              f'Train: {t_acc:.4f} ({t_loss:.4f}) | '
              f'Val: {v_acc:.4f} ({v_loss:.4f}){improved}')

        if v_acc > best_acc:
            best_acc         = v_acc
            best_wts         = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if v_acc >= TARGET_ACC:
            print(f'\n  [TARGET] Target {TARGET_ACC*100:.0f}% reached!')
            break

        if patience_counter >= PATIENCE:
            print(f'\n  [STOP] Early stopping -- no val improvement for {PATIENCE} epochs.')
            break

    # --- Final evaluation ----------------------------------------------------
    model.load_state_dict(best_wts)
    _, final_acc, all_preds, all_labels = evaluate(model, val_l, criterion, device)

    print_per_class_accuracy(all_preds, all_labels, class_names)

    # --- Save checkpoint -----------------------------------------------------
    torch.save({
        'state_dict': model.state_dict(),
        'classes':    class_names,
        'backbone':   'efficientnet_v2_s',
        'val_acc':    final_acc,
        'img_size':   IMG_SIZE,
    }, SAVE_PATH)
    print(f'\n  [OK] Model saved: {SAVE_PATH}')
    print(f'  Best Val Accuracy: {final_acc*100:.2f}%')

    # ONNX export for deployment
    try:
        model.eval()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        torch.onnx.export(
            model, dummy, 'buyo_best.onnx',
            input_names=['input'],
            output_names=['output'],
            opset_version=11,
        )
        print('  [OK] ONNX exported: buyo_best.onnx')
    except Exception as e:
        print(f'  [WARN] ONNX export failed: {e}')

    if final_acc < TARGET_ACC:
        print(f'\n  [HINT] Val accuracy {final_acc*100:.2f}% < {TARGET_ACC*100:.0f}%.')
        print('  Try: python augment_data.py  (then re-run train_gpu.py)')


if __name__ == '__main__':
    main()
