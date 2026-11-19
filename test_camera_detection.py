#!/usr/bin/env python3
"""
Camera Detection Test Script
=============================
Run this to diagnose why the Raspberry Pi Camera 3 (imx708) is not working.
"""

import sys
import os

print("="*70)
print("RASPBERRY PI CAMERA DETECTION TEST")
print("="*70)
print()

# Test 1: Python version
print("Test 1: Python Version")
print("-" * 70)
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print()

# Test 2: Virtual environment check
print("Test 2: Virtual Environment Check")
print("-" * 70)
in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
print(f"In virtual environment: {in_venv}")
if in_venv:
    print(f"Virtual env path: {sys.prefix}")
print()

# Test 3: Check system-site-packages
print("Test 3: System Site Packages Access")
print("-" * 70)
pyvenv_cfg = os.path.join(sys.prefix, 'pyvenv.cfg')
if os.path.exists(pyvenv_cfg):
    with open(pyvenv_cfg, 'r') as f:
        cfg_content = f.read()
        if 'include-system-site-packages = true' in cfg_content:
            print("✓ System site packages: ENABLED")
        else:
            print("✗ System site packages: DISABLED")
            print("  This is why picamera2 cannot be imported!")
            print("  Run: bash fix_picamera2.sh")
else:
    print("⚠ pyvenv.cfg not found (not in venv or system Python)")
print()

# Test 4: Try importing picamera2
print("Test 4: Picamera2 Import Test")
print("-" * 70)
try:
    from picamera2 import Picamera2
    print("✓ SUCCESS: picamera2 can be imported!")
    
    # Try to get camera info
    try:
        cameras = Picamera2.global_camera_info()
        print(f"✓ Found {len(cameras)} camera(s):")
        for i, cam_info in enumerate(cameras):
            print(f"   Camera {i}:")
            for key, value in cam_info.items():
                print(f"      {key}: {value}")
    except Exception as e:
        print(f"✗ Error getting camera info: {e}")
        
except ImportError as e:
    print(f"✗ FAILED: Cannot import picamera2")
    print(f"   Error: {e}")
    print()
    print("SOLUTION:")
    print("  1. Make sure picamera2 is installed:")
    print("     sudo apt install python3-picamera2")
    print()
    print("  2. Recreate venv with system packages access:")
    print("     rm -rf venv")
    print("     python3 -m venv venv --system-site-packages")
    print("     source venv/bin/activate")
    print("     pip install -r raspberry_pi_server/requirements.txt")
    print()
    print("  3. Or run the automated fix:")
    print("     bash fix_picamera2.sh")
print()

# Test 5: Check for camera devices
print("Test 5: Camera Device Files")
print("-" * 70)
video_devices = [f"/dev/video{i}" for i in range(5)]
found_devices = [dev for dev in video_devices if os.path.exists(dev)]
if found_devices:
    print(f"✓ Found {len(found_devices)} video device(s):")
    for dev in found_devices:
        print(f"   {dev}")
else:
    print("✗ No /dev/video* devices found")
    print("  Camera may not be connected or enabled")
print()

# Test 6: OpenCV test
print("Test 6: OpenCV Camera Access")
print("-" * 70)
try:
    import cv2
    print(f"✓ OpenCV version: {cv2.__version__}")
    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ OpenCV can capture from /dev/video0: {frame.shape}")
        else:
            print("✗ OpenCV opened device but capture failed")
        cap.release()
    else:
        print("✗ OpenCV cannot open /dev/video0")
except Exception as e:
    print(f"✗ OpenCV test failed: {e}")
print()

# Test 7: Check if camera is enabled in config
print("Test 7: Camera Configuration")
print("-" * 70)
try:
    import subprocess
    result = subprocess.run(['vcgencmd', 'get_camera'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Camera status: {result.stdout.strip()}")
    else:
        print("⚠ Could not check camera status with vcgencmd")
except Exception as e:
    print(f"⚠ vcgencmd not available: {e}")
print()

# Summary
print("="*70)
print("SUMMARY")
print("="*70)
print()
print("If you see '✗ System site packages: DISABLED' above,")
print("that's the root cause. Run this to fix:")
print()
print("    bash fix_picamera2.sh")
print()
print("If picamera2 imports successfully but no cameras found,")
print("check the camera cable connection and run:")
print()
print("    libcamera-hello")
print()
print("="*70)
