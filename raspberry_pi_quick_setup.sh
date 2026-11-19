#!/bin/bash
################################################################################
# Raspberry Pi Quick Setup - Buyo Leaf Classifier
################################################################################
# Fast one-command setup for Raspberry Pi
# Usage: bash raspberry_pi_quick_setup.sh
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "  Buyo Leaf Classifier - Raspberry Pi Quick Setup"
echo "========================================================================"
echo ""

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo -e "${YELLOW}⚠ Warning: This doesn't appear to be a Raspberry Pi${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
echo -e "${BLUE}→ Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found!${NC}"
    echo "  Install with: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Update package list (quick)
echo ""
echo -e "${BLUE}→ Updating package list...${NC}"
sudo apt update -qq

# Install system dependencies
echo ""
echo -e "${BLUE}→ Installing system dependencies...${NC}"
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libopencv-dev \
    python3-opencv \
    libjpeg-dev \
    zlib1g-dev \
    libatlas-base-dev

echo -e "${GREEN}✓ System dependencies installed${NC}"

# Install Pi Camera support (optional but recommended)
echo ""
echo -e "${BLUE}→ Installing Pi Camera support (picamera2)...${NC}"
if sudo apt list --installed 2>/dev/null | grep -q python3-picamera2; then
    echo -e "${GREEN}✓ picamera2 already installed${NC}"
else
    sudo apt install -y python3-picamera2 || echo -e "${YELLOW}⚠ picamera2 install failed (will use USB webcam)${NC}"
fi

# Install GPIO and I2C tools
echo ""
echo -e "${BLUE}→ Installing GPIO and I2C support...${NC}"
sudo apt install -y python3-rpi.gpio i2c-tools
echo -e "${GREEN}✓ GPIO support installed${NC}"

# Enable I2C and Camera interfaces
echo ""
echo -e "${BLUE}→ Enabling I2C and Camera...${NC}"
sudo raspi-config nonint do_i2c 0  # Enable I2C
sudo raspi-config nonint do_camera 0  # Enable Camera
echo -e "${GREEN}✓ I2C and Camera enabled${NC}"

# Create virtual environment with system packages access
echo ""
echo -e "${BLUE}→ Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ venv exists, removing...${NC}"
    rm -rf venv
fi
python3 -m venv venv --system-site-packages
echo -e "${GREEN}✓ Virtual environment created (with system packages access)${NC}"

# Activate venv
source venv/bin/activate

# Upgrade pip
echo ""
echo -e "${BLUE}→ Upgrading pip...${NC}"
pip install --upgrade pip -q
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install Python packages
echo ""
echo -e "${BLUE}→ Installing Python packages...${NC}"
echo "  This may take 5-10 minutes..."

# Install from requirements.txt
if [ -f "raspberry_pi_server/requirements.txt" ]; then
    pip install -r raspberry_pi_server/requirements.txt -q
    echo -e "${GREEN}✓ Core packages installed${NC}"
else
    echo -e "${YELLOW}⚠ requirements.txt not found, installing manually${NC}"
    pip install flask opencv-python pillow numpy -q
fi

# Install PyTorch (CPU version for Pi)
echo ""
echo -e "${BLUE}→ Installing PyTorch (CPU version)...${NC}"
echo "  This will take ~5 minutes..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
echo -e "${GREEN}✓ PyTorch installed${NC}"

# Install robotic arm libraries
echo ""
echo -e "${BLUE}→ Installing robotic arm libraries...${NC}"
pip install RPi.GPIO adafruit-circuitpython-servokit -q || echo -e "${YELLOW}⚠ Arm libraries had issues (will use simulation mode)${NC}"

# Verify installation
echo ""
echo "========================================================================"
echo "  Verifying Installation"
echo "========================================================================"

python3 << 'EOF'
import sys
print(f"✓ Python: {sys.version.split()[0]}")

try:
    import flask
    print(f"✓ Flask: {flask.__version__}")
except:
    print("✗ Flask not installed")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except:
    print("✗ OpenCV not installed")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except:
    print("✗ NumPy not installed")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except:
    print("✗ PyTorch not installed")

try:
    from PIL import Image
    print(f"✓ Pillow: OK")
except:
    print("✗ Pillow not installed")

try:
    import RPi.GPIO
    print(f"✓ RPi.GPIO: OK")
except:
    print("⚠ RPi.GPIO not installed (optional)")

try:
    from adafruit_servokit import ServoKit
    print(f"✓ ServoKit: OK")
except:
    print("⚠ ServoKit not installed (optional)")

try:
    from picamera2 import Picamera2
    print(f"✓ Picamera2: OK")
except:
    print("⚠ Picamera2 not available (will use USB webcam)")
EOF

echo ""
echo "========================================================================"
echo "  Setup Complete!"
echo "========================================================================"
echo ""
echo -e "${GREEN}✓ Raspberry Pi is ready to run!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   ${BLUE}source venv/bin/activate${NC}"
echo ""
echo "2. Run the server:"
echo "   ${BLUE}python3 raspberry_pi_server/pi_server_gui.py${NC}"
echo ""
echo "3. Copy the IP address shown in the GUI"
echo ""
echo "4. Use that IP on your laptop client to connect"
echo ""
echo "========================================================================"
echo ""

# Optional: Create desktop shortcut
read -p "Create desktop shortcut to launch server? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    DESKTOP_FILE="$HOME/Desktop/BuyoServer.desktop"
    cat > "$DESKTOP_FILE" << 'SHORTCUT'
[Desktop Entry]
Name=Buyo Server
Comment=Start Buyo Leaf Classifier Server
Exec=bash -c "cd ~/Desktop/Buyo-Leaf-Classifier && source venv/bin/activate && python3 raspberry_pi_server/pi_server_gui.py"
Icon=applications-science
Terminal=false
Type=Application
Categories=Application;
SHORTCUT
    chmod +x "$DESKTOP_FILE"
    echo -e "${GREEN}✓ Desktop shortcut created${NC}"
fi

echo ""
echo -e "${GREEN}${BOLD}🎉 All done! Your Raspberry Pi is ready!${NC}"
echo ""
