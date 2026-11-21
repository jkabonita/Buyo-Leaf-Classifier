@echo off
echo ============================================================
echo Python 3.12 Installation Helper
echo ============================================================
echo.
echo This script will help you install Python 3.12 for CUDA support
echo.
echo MANUAL STEPS REQUIRED:
echo ============================================================
echo.
echo 1. UNINSTALL Python 3.14:
echo    - Open "Add or Remove Programs" (Windows Settings)
echo    - Search for "Python 3.14"
echo    - Click "Uninstall"
echo.
echo 2. DOWNLOAD Python 3.12.8:
echo    - Opening browser to download page...
echo    - Download: "Windows installer (64-bit)"
echo.
start https://www.python.org/downloads/release/python-3128/
echo.
echo 3. INSTALL Python 3.12:
echo    - Run the downloaded installer
echo    - ✓ CHECK "Add Python 3.12 to PATH"
echo    - ✓ CHECK "Install pip"
echo    - Click "Install Now"
echo.
echo 4. After installation, close this window and run:
echo    install_cuda_pytorch.bat
echo.
echo ============================================================
pause
