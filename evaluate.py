"""
evaluate.py -- Model Evaluation for Buyo Leaves Classifier
==========================================================
Loads buyo_best.pth (ResNet50 / EfficientNetV2-S / MobileNetV2).
Generates:
  - Overall accuracy on validation set
  - Per-class precision / recall / F1 report
  - Confusion matrix (saved as confusion_matrix.png)

Usage:
    python evaluate.py
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_PATH  = 'buyo_best.pth'
FALLBACK    = 'buyo_pytorch.pth'
VAL_DIR     = 'dataset/val'
IMG_SIZE    = 224
NUM_CLASSES = 5
BATCH_SIZE  = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# -----------------------------------------------------------------------------
# MODEL LOADER
# -----------------------------------------------------------------------------
def load_model(path: str):
    ckpt     = torch.load(path, map_location=device)
    backbone = ckpt.get('backbone', 'resnet50')
    classes  = ckpt.get('classes', [f'class_{c}' for c in 'abcde'])

    if backbone == 'resnet50':
        model = models.resnet50(weights=None)
        in_f  = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, NUM_CLASSES)
        )
    elif backbone == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=None)
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.35, inplace=True),
            nn.Linear(in_f, NUM_CLASSES),
        )
    else:
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, NUM_CLASSES),
        )

    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    saved_acc = ckpt.get('val_acc', None)
    return model, classes, backbone, saved_acc

if os.path.exists(MODEL_PATH):
    model, class_names, backbone, saved_acc = load_model(MODEL_PATH)
    print(f'Loaded : {MODEL_PATH} ({backbone})')
elif os.path.exists(FALLBACK):
    model, class_names, backbone, saved_acc = load_model(FALLBACK)
    print(f'Loaded : {FALLBACK} ({backbone}) [fallback]')
else:
    raise FileNotFoundError('No model found. Run train_gpu.py first.')

if saved_acc is not None:
    print(f'Saved validation accuracy: {saved_acc*100:.2f}%')

# -----------------------------------------------------------------------------
# VALIDATION DATASET
# -----------------------------------------------------------------------------
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
folder_names = val_dataset.classes

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())

overall = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
print('=' * 60)
print(f'  Overall Accuracy : {overall*100:.2f}%')
print('=' * 60)

print('\n--- Classification Report ---')
print(classification_report(y_true, y_pred, target_names=folder_names, digits=4))

# Per-class breakdown
print('--- Per-Class Breakdown ---')
correct_map = Counter()
total_map   = Counter()
for p, t in zip(y_pred, y_true):
    total_map[t] += 1
    if p == t:
        correct_map[t] += 1

for i, name in enumerate(folder_names):
    n_c = correct_map[i]
    n_t = total_map[i]
    pct = 100.0 * n_c / max(1, n_t)
    bar = '#' * int(pct / 5) + '-' * (20 - int(pct / 5))
    print(f'  {name.upper():>10s}: {n_c:>2}/{n_t:<2} ({pct:>5.1f}%) | {bar}')

# -----------------------------------------------------------------------------
# CONFUSION MATRIX PLOT
# -----------------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[n.upper() for n in folder_names],
    yticklabels=[n.upper() for n in folder_names],
    linewidths=0.5,
    linecolor='#cccccc',
)
plt.title(
    f'Confusion Matrix -- Buyo Leaves ({backbone.upper()})\n'
    f'Val Accuracy: {overall*100:.2f}%',
    fontsize=14, fontweight='bold',
)
plt.ylabel('Actual Label',    fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print('\nSaved: confusion_matrix.png', flush=True)
