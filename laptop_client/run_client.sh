#!/bin/bash
# Laptop Client Launcher for Linux/Mac
# Run this: bash run_client.sh

echo "=================================="
echo "  Buyo Classifier - Laptop Client"
echo "=================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found!"
    exit 1
fi

echo "✓ Python 3 found"

# Check dependencies
echo "Checking dependencies..."
python3 -c "import requests" 2>/dev/null && echo "✓ Requests installed" || echo "❌ Requests missing - run: pip install requests"
python3 -c "import cv2" 2>/dev/null && echo "✓ OpenCV installed" || echo "❌ OpenCV missing - run: pip install opencv-python"
python3 -c "import torch" 2>/dev/null && echo "✓ PyTorch installed" || echo "❌ PyTorch missing - run: pip install torch torchvision"
python3 -c "from PIL import Image" 2>/dev/null && echo "✓ Pillow installed" || echo "❌ Pillow missing - run: pip install pillow"

# Check model file
if [ ! -f "../buyo_best.pth" ]; then
    echo "⚠️  Model file not found: ../buyo_best.pth"
    echo "   Detection will be disabled"
fi

echo ""
echo "Starting client GUI..."
echo ""

# Run client
python3 laptop_client_gui.py
