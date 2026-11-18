#!/bin/bash
# Ultimate Camera Fix Script for Raspberry Pi
# Tries multiple methods to get camera working

echo "======================================"
echo "  ULTIMATE CAMERA FIX"
echo "======================================"
echo ""

cd ~/Desktop/Buyo-Leaf-Classifier

# Method 1: Install OpenCV with GStreamer support
echo "[1/4] Installing OpenCV with GStreamer..."
source .venv/bin/activate
pip uninstall -y opencv-python 2>/dev/null
pip install opencv-contrib-python
echo "✓ Done"
echo ""

# Method 2: Install missing GStreamer plugins
echo "[2/4] Installing GStreamer plugins..."
sudo apt-get update
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev
echo "✓ Done"
echo ""

# Method 3: Test camera
echo "[3/4] Testing camera..."
rpicam-vid --list-cameras
echo ""

# Method 4: Run classifier
echo "[4/4] Starting classifier..."
echo ""
echo "If this doesn't work, press Ctrl+C and run:"
echo "  deactivate"
echo "  sudo apt-get install -y python3-picamera2"
echo "  python3 raspberry_pi_classifier_picamera2.py"
echo ""
echo "Press Enter to continue..."
read

python raspberry_pi_classifier.py
