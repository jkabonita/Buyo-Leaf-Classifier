#!/usr/bin/env python3
"""Check model architecture in saved checkpoint"""
import torch

ckpt = torch.load('buyo_best.pth', map_location='cpu')

print("="*70)
print("MODEL CHECKPOINT INFO")
print("="*70)
print(f"Backbone: {ckpt.get('backbone', 'unknown')}")
print(f"Epoch: {ckpt.get('epoch', 'unknown')}")
print(f"Best Val Acc: {ckpt.get('best_val_acc', 'unknown')}")

print("\n" + "="*70)
print("CLASSIFIER LAYERS IN STATE_DICT")
print("="*70)

classifier_keys = sorted([k for k in ckpt['state_dict'].keys() if 'classifier' in k])

for k in classifier_keys:
    shape = ckpt['state_dict'][k].shape
    print(f"{k:40s} {shape}")

print("\n" + "="*70)
print("EXPECTED ARCHITECTURE")
print("="*70)

print("""
nn.Sequential(
    0: nn.Dropout(p=0.35, inplace=True),
    1: nn.Linear(in_f, 512),
    2: nn.SiLU(),
    3: nn.BatchNorm1d(512),  ← has weight, bias, running_mean, running_var
    4: nn.Dropout(p=0.25),
    5: nn.Linear(512, 256),  ← has weight, bias
    6: nn.SiLU(),
    7: nn.Dropout(p=0.15),
    8: nn.Linear(256, 5),    ← has weight, bias
)
""")
