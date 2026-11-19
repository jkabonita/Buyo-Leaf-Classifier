#!/bin/bash
################################################################################
# Quick Fix for Picamera2 Access in Virtual Environment
################################################################################
# This script recreates the venv with system-site-packages enabled
# so that picamera2 (installed as system package) can be accessed
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================================================"
echo "  Picamera2 Quick Fix"
echo "========================================================================"
echo ""

# Check if picamera2 is installed as system package
echo -e "${BLUE}→ Checking for picamera2 system package...${NC}"
if dpkg -l | grep -q python3-picamera2; then
    echo -e "${GREEN}✓ picamera2 is installed as system package${NC}"
else
    echo -e "${YELLOW}⚠ picamera2 not found in system packages${NC}"
    echo "  Installing it now..."
    sudo apt update -qq
    sudo apt install -y python3-picamera2
    echo -e "${GREEN}✓ picamera2 installed${NC}"
fi

# Backup and remove old venv
echo ""
echo -e "${BLUE}→ Removing old virtual environment...${NC}"
if [ -d "venv" ]; then
    rm -rf venv
    echo -e "${GREEN}✓ Old venv removed${NC}"
else
    echo -e "${YELLOW}⚠ No existing venv found${NC}"
fi

# Create new venv with system-site-packages
echo ""
echo -e "${BLUE}→ Creating new virtual environment with system packages access...${NC}"
python3 -m venv venv --system-site-packages
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Activate and verify
source venv/bin/activate

echo ""
echo -e "${BLUE}→ Upgrading pip...${NC}"
pip install --upgrade pip -q
echo -e "${GREEN}✓ pip upgraded${NC}"

echo ""
echo -e "${BLUE}→ Reinstalling project packages...${NC}"
if [ -f "raspberry_pi_server/requirements.txt" ]; then
    pip install -r raspberry_pi_server/requirements.txt -q
    echo -e "${GREEN}✓ Packages installed${NC}"
fi

# Verify picamera2 is accessible
echo ""
echo "========================================================================"
echo "  Testing Picamera2 Access"
echo "========================================================================"

python3 << 'EOF'
try:
    from picamera2 import Picamera2
    print("✓ SUCCESS: Picamera2 can be imported!")
    
    # Try to get camera info
    cameras = Picamera2.global_camera_info()
    if cameras:
        print(f"✓ Found {len(cameras)} camera(s):")
        for i, cam in enumerate(cameras):
            model = cam.get('Model', 'Unknown')
            print(f"   Camera {i}: {model}")
    else:
        print("⚠ No cameras detected (check camera cable connection)")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)
EOF

echo ""
echo "========================================================================"
echo "  Fix Complete!"
echo "========================================================================"
echo ""
echo -e "${GREEN}✓ Picamera2 is now accessible in venv!${NC}"
echo ""
echo "Run the server now:"
echo "  ${BLUE}source venv/bin/activate${NC}"
echo "  ${BLUE}python3 raspberry_pi_server/pi_server_gui.py${NC}"
echo ""

