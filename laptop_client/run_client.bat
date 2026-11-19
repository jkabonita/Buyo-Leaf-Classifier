@echo off
REM Laptop Client Launcher for Windows
REM Double-click to run

echo ==================================
echo   Buyo Classifier - Laptop Client
echo ==================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python not found!
    echo Install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found

REM Check if in correct directory
if not exist "laptop_client_gui.py" (
    echo X Not in laptop_client directory!
    echo Please run this script from laptop_client folder
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
python -c "import requests" 2>nul && echo [OK] Requests installed || echo [X] Requests missing - run: pip install requests
python -c "import cv2" 2>nul && echo [OK] OpenCV installed || echo [X] OpenCV missing - run: pip install opencv-python
python -c "import torch" 2>nul && echo [OK] PyTorch installed || echo [X] PyTorch missing - run: pip install torch torchvision
python -c "from PIL import Image" 2>nul && echo [OK] Pillow installed || echo [X] Pillow missing - run: pip install pillow

echo.
echo Starting client GUI...
echo.

REM Run client
python laptop_client_gui.py

pause
