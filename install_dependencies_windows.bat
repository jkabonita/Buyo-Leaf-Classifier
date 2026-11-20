@echo off
REM Buyo Leaf Classifier - Windows Dependency Installation Script
REM This script installs all required Python packages globally

echo ========================================
echo  Buyo Leaf Classifier Setup
echo  Installing Dependencies...
echo ========================================
echo.

echo Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Installing core dependencies...
python -m pip install requests pillow opencv-python numpy
echo.

echo Installing PyTorch and torchvision...
python -m pip install torch torchvision
echo.

echo Installing ML utilities...
python -m pip install matplotlib seaborn scikit-learn
echo.

echo Installing Flask for Raspberry Pi server (optional on laptop)...
python -m pip install flask
echo.

echo ========================================
echo  Verifying Installation...
echo ========================================
python -c "import requests, torch, torchvision, PIL, cv2, numpy, matplotlib, seaborn, sklearn; print('✓ All libraries installed successfully!')"
echo.

echo Installation complete!
echo.
echo NOTE: PyTorch CPU version is installed by default.
echo For NVIDIA GPU support, you need Python 3.11 or 3.12.
echo PyTorch with CUDA is not yet available for Python 3.14.
echo.
pause
