#!/bin/bash
# Raspberry Pi Server Launcher
# Run this on Raspberry Pi: bash run_server.sh

echo "=================================="
echo "  Buyo Classifier - Pi Server"
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
python3 -c "import flask" 2>/dev/null && echo "✓ Flask installed" || echo "❌ Flask missing - run: pip3 install flask"
python3 -c "import cv2" 2>/dev/null && echo "✓ OpenCV installed" || echo "❌ OpenCV missing - run: pip3 install opencv-python"
python3 -c "from PIL import Image" 2>/dev/null && echo "✓ Pillow installed" || echo "❌ Pillow missing - run: pip3 install pillow"

echo ""
echo "Starting server GUI..."
echo ""

# Run server
python3 pi_server_gui.py
