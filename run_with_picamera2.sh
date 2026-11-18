#!/bin/bash
# Run classifier with system picamera2 library
# This script creates a venv with access to system packages

echo "==============================================="
echo "  Buyo Classifier - System Package Runner"
echo "==============================================="
echo ""

# Check if system has picamera2
if ! python3 -c "import picamera2" 2>/dev/null; then
    echo "❌ ERROR: picamera2 not installed on system!"
    echo ""
    echo "Install with:"
    echo "  sudo apt-get install -y python3-picamera2"
    echo ""
    exit 1
fi

echo "✓ picamera2 found on system"

# Create or use venv with system packages
VENV_DIR=".venv_system"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating new venv with system packages..."
    python3 -m venv --system-site-packages "$VENV_DIR"
    
    # Install PyTorch in this venv
    echo "Installing PyTorch..."
    source "$VENV_DIR/bin/activate"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install opencv-python pillow numpy
    deactivate
fi

echo "✓ Activating venv with system packages"
source "$VENV_DIR/bin/activate"

echo "✓ Running classifier..."
echo ""

python raspberry_pi_classifier_picamera2.py

deactivate
