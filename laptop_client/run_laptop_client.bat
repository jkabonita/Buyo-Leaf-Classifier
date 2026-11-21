@echo off
REM Batch script to run the Laptop Client GUI
REM This ensures all dependencies are available

echo ========================================
echo   Buyo Leaf Classifier - Laptop Client
echo ========================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Set Python path
set PYTHON_PATH=C:\Users\aboni\AppData\Local\Programs\Python\Python312\python.exe

REM Check if Python exists
if not exist "%PYTHON_PATH%" (
    echo ERROR: Python not found at %PYTHON_PATH%
    echo Please update the PYTHON_PATH variable in this script
    pause
    exit /b 1
)

echo Starting Laptop Client GUI...
echo.

REM Run the laptop client
"%PYTHON_PATH%" laptop_client_gui.py

REM If there's an error, keep window open
if errorlevel 1 (
    echo.
    echo ERROR: Failed to run laptop_client_gui.py
    echo.
    pause
)
