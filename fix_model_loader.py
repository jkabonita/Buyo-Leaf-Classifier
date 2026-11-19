#!/usr/bin/env python3
"""
Diagnostic script to check model architecture and fix the loader
Run this with: python fix_model_loader.py
"""
import torch
import torch.nn as nn
from torchvision import models

print("="*70)
print("CHECKING MODEL ARCHITECTURE")
print("="*70)

# Load checkpoint
ckpt = torch.load('buyo_best.pth', map_location='cpu')
print(f"\nBackbone: {ckpt.get('backbone', 'unknown')}")

# Get classifier keys
classifier_keys = sorted([k for k in ckpt['state_dict'].keys() if 'classifier' in k])

print("\nClassifier layers in saved model:")
print("-"*70)
for k in classifier_keys:
    shape = ckpt['state_dict'][k].shape
    print(f"  {k:40s} {str(shape):20s}")

print("\n" + "="*70)
print("TESTING ARCHITECTURE VARIANTS")
print("="*70)

# Test variant 1: With BatchNorm1d
print("\nVariant 1: With BatchNorm1d at position 3")
try:
    model = models.efficientnet_v2_s(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35, inplace=True),  # 0
        nn.Linear(in_f, 512),                # 1
        nn.SiLU(),                           # 2
        nn.BatchNorm1d(512),                 # 3
        nn.Dropout(p=0.25),                  # 4
        nn.Linear(512, 256),                 # 5
        nn.SiLU(),                           # 6
        nn.Dropout(p=0.15),                  # 7
        nn.Linear(256, 5),                   # 8
    )
    model.load_state_dict(ckpt['state_dict'])
    print("✓ SUCCESS! This architecture matches!")
    print("\nCORRECT ARCHITECTURE:")
    print(model.classifier)
except Exception as e:
    print(f"✗ Failed: {e}")

# Test variant 2: Without BatchNorm1d  
print("\n" + "-"*70)
print("Variant 2: Without BatchNorm1d")
try:
    model = models.efficientnet_v2_s(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35, inplace=True),
        nn.Linear(in_f, 512),
        nn.SiLU(),
        nn.Dropout(p=0.25),
        nn.Linear(512, 256),
        nn.SiLU(),
        nn.Dropout(p=0.15),
        nn.Linear(256, 5),
    )
    model.load_state_dict(ckpt['state_dict'])
    print("✓ SUCCESS! This architecture matches!")
    print("\nCORRECT ARCHITECTURE:")
    print(model.classifier)
except Exception as e:
    print(f"✗ Failed: {e}")

# Test variant 3: Old training architecture
print("\n" + "-"*70)
print("Variant 3: Checking what's actually in the checkpoint")
print("\nLet's count the Linear layers:")

linear_count = 0
batchnorm_count = 0
for k in classifier_keys:
    if 'weight' in k and 'bias' not in k:
        shape = ckpt['state_dict'][k].shape
        if len(shape) == 2:  # Linear layer
            linear_count += 1
            print(f"  Linear layer {linear_count}: {k} → {shape}")
        elif len(shape) == 1 and 'running_mean' in classifier_keys:  # BatchNorm
            batchnorm_count += 1
            print(f"  BatchNorm layer {batchnorm_count}: {k} → {shape}")

print(f"\nTotal Linear layers: {linear_count}")
print(f"Total BatchNorm layers: {batchnorm_count}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nIf Variant 1 succeeded: Use architecture WITH BatchNorm1d")
print("If Variant 2 succeeded: Use architecture WITHOUT BatchNorm1d")
print("\nThe correct code should be copy-pasted to laptop_client_gui.py")
