#!/bin/bash
################################################################################
# PyTorch Installation Script for Raspberry Pi
# Run this if PyTorch installation fails
################################################################################

echo "╔════════════════════════════════════════════════╗"
echo "║  PyTorch Installation for Raspberry Pi        ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Check system info
echo "📌 System Information:"
echo "   Python version: $(python3 --version)"
echo "   Architecture: $(uname -m)"
echo "   OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo ""

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y libopenblas-dev libblas-dev libatlas-base-dev
sudo apt-get install -y python3-dev python3-pip
sudo apt-get install -y libjpeg-dev zlib1g-dev

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
python3 -m pip install --upgrade pip

# Uninstall any existing torch (clean slate)
echo ""
echo "🧹 Cleaning up old PyTorch installations..."
pip3 uninstall -y torch torchvision torchaudio

# Method 1: Official CPU wheels
echo ""
echo "🔥 Method 1: Trying official PyTorch CPU wheels..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Check if it worked
if python3 -c "import torch" 2>/dev/null; then
    echo ""
    echo "✅ SUCCESS! PyTorch installed via official wheels"
    python3 -c "import torch; print('   PyTorch version:', torch.__version__)"
    exit 0
fi

# Method 2: piwheels
echo ""
echo "🔥 Method 2: Trying piwheels (Raspberry Pi optimized)..."
pip3 install torch torchvision --index-url https://www.piwheels.org/simple

# Check again
if python3 -c "import torch" 2>/dev/null; then
    echo ""
    echo "✅ SUCCESS! PyTorch installed via piwheels"
    python3 -c "import torch; print('   PyTorch version:', torch.__version__)"
    exit 0
fi

# Method 3: Older stable version
echo ""
echo "🔥 Method 3: Trying PyTorch 1.13.0 (stable for Pi)..."
pip3 install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cpu

# Check again
if python3 -c "import torch" 2>/dev/null; then
    echo ""
    echo "✅ SUCCESS! PyTorch 1.13.0 installed"
    python3 -c "import torch; print('   PyTorch version:', torch.__version__)"
    exit 0
fi

# Method 4: Even older version
echo ""
echo "🔥 Method 4: Trying PyTorch 1.11.0..."
pip3 install torch==1.11.0 torchvision==0.12.0 --index-url https://download.pytorch.org/whl/cpu

# Final check
if python3 -c "import torch" 2>/dev/null; then
    echo ""
    echo "✅ SUCCESS! PyTorch 1.11.0 installed"
    python3 -c "import torch; print('   PyTorch version:', torch.__version__)"
    exit 0
fi

# All methods failed
echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  ❌ PyTorch Installation Failed                ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Possible solutions:"
echo ""
echo "1️⃣  Check available disk space (need ~500MB):"
echo "   df -h"
echo ""
echo "2️⃣  Try manual wheel download:"
echo "   Visit: https://download.pytorch.org/whl/cpu/torch_stable.html"
echo "   Download wheel matching your Python version"
echo "   Install: pip3 install /path/to/wheel.whl"
echo ""
echo "3️⃣  Use TensorFlow Lite instead (lighter, works better on Pi):"
echo "   pip3 install tflite-runtime"
echo "   (I can convert your model to TFLite)"
echo ""
echo "4️⃣  Use ONNX Runtime (also works well):"
echo "   pip3 install onnxruntime"
echo "   (I can convert your model to ONNX)"
echo ""
echo "5️⃣  Provide more info for troubleshooting:"
echo "   python3 --version"
echo "   uname -m"
echo "   cat /etc/os-release"
echo ""

exit 1
