#!/usr/bin/env python3
"""
Quick camera test for Raspberry Pi
Tests both V4L2 and libcamera methods
"""

import cv2
import subprocess
import os

print("="*60)
print("  RASPBERRY PI CAMERA TEST")
print("="*60)

# Test 1: Check libcamera-hello (Pi Camera Module 3)
print("\n1. Testing libcamera-hello...")
try:
    result = subprocess.run(['libcamera-hello', '--list-cameras'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ libcamera detected:")
        print(result.stdout)
    else:
        print("✗ libcamera not available")
except Exception as e:
    print(f"✗ libcamera test failed: {e}")

# Test 2: Check V4L2 devices
print("\n2. Testing V4L2 devices...")
try:
    result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ V4L2 devices:")
        print(result.stdout)
    else:
        print("✗ No V4L2 devices")
except Exception as e:
    print(f"✗ V4L2 test failed: {e}")

# Test 3: Try OpenCV with libcamera
print("\n3. Testing OpenCV with libcamera...")
try:
    # Use rpicam-vid compatible pipeline
    pipeline = (
        'libcamerasrc ! '
        'video/x-raw,width=640,height=480,framerate=10/1 ! '
        'videoconvert ! '
        'appsink'
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ OpenCV + libcamera works!")
            print(f"  Frame shape: {frame.shape}")
            cap.release()
        else:
            print("✗ Can't read frame from libcamera")
    else:
        print("✗ Can't open libcamera with OpenCV")
except Exception as e:
    print(f"✗ libcamera OpenCV test failed: {e}")

# Test 4: Try OpenCV V4L2 indices
print("\n4. Testing OpenCV V4L2 (0-5)...")
for idx in range(6):
    try:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {idx} works! Frame: {frame.shape}")
                cap.release()
                break
            cap.release()
    except Exception as e:
        pass

print("\n" + "="*60)
print("RECOMMENDATION:")
print("="*60)

if os.path.exists('/dev/video0'):
    print("Use: CAMERA_INDEX = 0")
elif os.path.exists('/usr/bin/libcamera-hello'):
    print("Use libcamera pipeline (Pi Camera Module 3)")
else:
    print("Camera not detected. Check connections.")
print("="*60)
