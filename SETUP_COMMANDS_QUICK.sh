#!/bin/bash
################################################################################
# Buyo Leaf Classifier - Raspberry Pi Quick Setup Script
# Copy and paste these commands into your Raspberry Pi terminal
################################################################################

echo "╔════════════════════════════════════════════════╗"
echo "║  Buyo Leaf Classifier - Raspberry Pi Setup    ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Step 1: Create project directory
echo "📁 Creating project directory..."
mkdir -p ~/buyo
cd ~/buyo

# Step 2: Update system
echo "🔄 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Step 3: Enable hardware interfaces
echo "🎥 Enabling camera and I2C..."
sudo raspi-config nonint do_camera 0
sudo raspi-config nonint do_i2c 0

# Step 4: Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y python3-pip python3-dev python3-opencv
sudo apt-get install -y libatlas-base-dev libjasper-dev libqtgui4 libhdf5-dev
sudo apt-get install -y i2c-tools

# Step 5: Upgrade pip
echo "⬆️  Upgrading pip..."
python3 -m pip install --upgrade pip

# Step 6: Install PyTorch (CPU version for Raspberry Pi)
echo "🔥 Installing PyTorch..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 7: Install Python packages
echo "🐍 Installing Python packages..."
pip3 install opencv-python-headless
pip3 install Pillow
pip3 install numpy

# Step 8: Install robotic arm libraries (optional - will fail gracefully if hardware not needed)
echo "🤖 Installing robotic arm libraries..."
pip3 install RPi.GPIO adafruit-circuitpython-servokit || echo "Note: Robotic arm libraries optional"

# Step 9: Add user to hardware groups
echo "👤 Adding user to hardware groups..."
sudo usermod -a -G video $USER
sudo usermod -a -G i2c $USER
sudo usermod -a -G gpio $USER

# Step 10: Verify installation
echo ""
echo "✅ Verifying installation..."
python3 -c "import torch; print('✓ PyTorch:', torch.__version__)"
python3 -c "import cv2; print('✓ OpenCV:', cv2.__version__)"
python3 -c "import numpy; print('✓ NumPy:', numpy.__version__)"
python3 -c "import PIL; print('✓ Pillow: OK')"

echo ""
echo "📋 Checking hardware..."
vcgencmd get_camera
echo ""
sudo i2cdetect -y 1

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  Installation Complete!                        ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "⚠️  IMPORTANT: Reboot required to apply group changes"
echo ""
echo "Next steps:"
echo "  1. Copy your model file (buyo_best.pth) to ~/buyo/"
echo "  2. Copy raspberry_pi_classifier.py to ~/buyo/"
echo "  3. Reboot: sudo reboot"
echo "  4. Run: cd ~/buyo && python3 raspberry_pi_classifier.py"
echo ""
