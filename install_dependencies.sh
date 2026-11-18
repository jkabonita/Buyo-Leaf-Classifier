#!/bin/bash
# ============================================================================
# Buyo Leaf Classifier - Linux/Raspberry Pi Installation Script
# ============================================================================

echo ""
echo "========================================"
echo "  Buyo Leaf Classifier Installation"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Install with: sudo apt-get install python3 python3-pip"
    exit 1
fi

echo "[1/6] Checking Python installation..."
python3 --version

# Check if running on Raspberry Pi
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    IS_RASPBERRY_PI=true
    echo "✓ Detected Raspberry Pi"
else
    IS_RASPBERRY_PI=false
    echo "✓ Detected Linux system"
fi

echo ""
echo "[2/6] Updating system packages..."
sudo apt-get update

echo ""
echo "[3/6] Installing system dependencies..."
sudo apt-get install -y python3-pip python3-dev libopencv-dev python3-opencv

if [ "$IS_RASPBERRY_PI" = true ]; then
    echo ""
    echo "[4/6] Installing additional Raspberry Pi dependencies..."
    sudo apt-get install -y libatlas-base-dev libjasper-dev libqt4-test libqtgui4 libhdf5-dev
fi

echo ""
echo "[5/6] Upgrading pip..."
python3 -m pip install --upgrade pip

echo ""
echo "[6/6] Installing Python packages..."
if [ "$IS_RASPBERRY_PI" = true ]; then
    echo "Installing Raspberry Pi optimized packages..."
    pip3 install -r requirements_raspberry_pi.txt
    
    echo ""
    echo "Installing PyTorch for Raspberry Pi..."
    # Install OpenBLAS dependencies first (critical for PyTorch)
    sudo apt-get install -y libopenblas-dev libblas-dev
    
    # Try Method 1: Official CPU wheels
    echo "Attempting official PyTorch CPU wheels..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    # Check if installation succeeded
    if ! python3 -c "import torch" 2>/dev/null; then
        echo "Trying alternative source (piwheels)..."
        pip3 install torch torchvision --index-url https://www.piwheels.org/simple
    fi
    
    # If still failing, try older stable version
    if ! python3 -c "import torch" 2>/dev/null; then
        echo "Trying PyTorch 1.13.0 (stable for Raspberry Pi)..."
        pip3 install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "Installing standard packages..."
    pip3 install -r requirements.txt
fi

echo ""
echo "Verifying installation..."
python3 -c "import torch; import cv2; import PIL; import numpy; print('✓ All packages imported successfully!')"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  Installation Complete!"
    echo "========================================"
    echo ""
    echo "You can now run the classifier:"
    echo "  python3 raspberry_pi_classifier.py"
    echo ""
else
    echo ""
    echo "========================================"
    echo "  Installation Failed!"
    echo "========================================"
    echo "Please check the error messages above"
    echo ""
    exit 1
fi
