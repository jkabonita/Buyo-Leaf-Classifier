#!/usr/bin/env python3
"""
Test Installation - Buyo Classifier
===================================
Verify all dependencies are installed correctly
"""

import sys

def test_import(package_name, import_name=None):
    """Test if a package can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'installed')
        print(f"  ✓ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {package_name}: NOT INSTALLED - {e}")
        return False

print("="*70)
print("  Buyo Classifier - Installation Test")
print("="*70)

print("\nTesting required packages...")
print("-"*70)

required = [
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('opencv-python', 'cv2'),
    ('pillow', 'PIL'),
    ('numpy', 'numpy'),
    ('flask', 'flask'),
    ('requests', 'requests'),
    ('tkinter', 'tkinter'),
]

optional = [
    ('RPi.GPIO', 'RPi.GPIO'),
    ('adafruit-circuitpython-servokit', 'adafruit_servokit'),
    ('picamera2', 'picamera2'),
]

all_ok = True

print("\n📦 Required Packages:")
for pkg_name, import_name in required:
    if not test_import(pkg_name, import_name):
        all_ok = False

print("\n📦 Optional Packages (Raspberry Pi only):")
for pkg_name, import_name in optional:
    test_import(pkg_name, import_name)

print("\n" + "="*70)
if all_ok:
    print("  ✓ ALL REQUIRED PACKAGES INSTALLED!")
    print("="*70)
    print("\nYou can now run:")
    print("  python laptop_client\\laptop_client_gui.py")
    print("  python raspberry_pi_server\\pi_server_gui.py")
else:
    print("  ✗ SOME REQUIRED PACKAGES MISSING")
    print("="*70)
    print("\nPlease run:")
    print("  install_all_dependencies.bat")
    sys.exit(1)

print("\n" + "="*70)
print("  Testing PyTorch GPU Support")
print("="*70)

import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU (CUDA not available)")

print("\n" + "="*70)
print("  Testing OpenCV Camera")
print("="*70)

import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("\n✓ Camera 0 accessible")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Resolution: {width}x{height}")
    cap.release()
else:
    print("\n⚠ Camera 0 not accessible (may not have local camera)")

print("\n" + "="*70)
print("  Installation test complete!")
print("="*70)
