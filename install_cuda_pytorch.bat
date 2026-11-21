@echo off
echo ============================================================
echo Installing CUDA-enabled PyTorch for Python 3.12
echo ============================================================
echo.

echo Checking Python version...
python --version
echo.

echo Installing PyTorch with CUDA 11.8 support...
echo This may take several minutes...
echo.

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo ============================================================
echo Verifying CUDA installation...
echo ============================================================
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo ============================================================
echo Installation complete!
echo.
echo If CUDA is available, you can now run:
echo    python train_single_model_max_accuracy.py
echo.
echo If CUDA is NOT available, check:
echo    1. NVIDIA GPU drivers are installed
echo    2. CUDA Toolkit is installed
echo    3. GPU is compatible (GTX/RTX series)
echo ============================================================
pause
