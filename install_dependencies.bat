@echo off
REM ============================================================================
REM Buyo Leaf Classifier - Windows Installation Script
REM ============================================================================

echo.
echo ========================================
echo   Buyo Leaf Classifier Installation
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking Python installation...
python --version

echo.
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [3/4] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo [4/4] Verifying installation...
python -c "import torch; import cv2; import PIL; import numpy; print('✓ All packages imported successfully!')"

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   Installation Complete!
    echo ========================================
    echo.
    echo You can now run the classifier:
    echo   python raspberry_pi_classifier.py
    echo.
) else (
    echo.
    echo ========================================
    echo   Installation Failed!
    echo ========================================
    echo Please check the error messages above
    echo.
)

pause
