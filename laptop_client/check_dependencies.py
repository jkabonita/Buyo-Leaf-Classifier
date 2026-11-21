#!/usr/bin/env python3
"""
Dependency Checker for Buyo Leaf Classifier - Laptop Client
Checks all required dependencies and provides installation instructions
"""

import sys
import subprocess

def check_module(module_name, package_name=None, import_name=None):
    """Check if a module is installed"""
    if import_name is None:
        import_name = module_name
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(import_name)
        print(f"✅ {module_name}: Installed")
        return True
    except ImportError:
        print(f"❌ {module_name}: NOT installed")
        print(f"   Install with: pip install {package_name}")
        return False

def main():
    print("=" * 60)
    print("  Buyo Leaf Classifier - Dependency Check")
    print("=" * 60)
    print()
    
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print()
    
    print("Checking dependencies...")
    print("-" * 60)
    
    all_ok = True
    
    # Check core dependencies
    all_ok &= check_module("requests", "requests")
    all_ok &= check_module("Pillow", "pillow", "PIL")
    all_ok &= check_module("OpenCV", "opencv-python", "cv2")
    all_ok &= check_module("NumPy", "numpy", "numpy")
    
    # Check PyTorch
    torch_ok = check_module("PyTorch", "torch", "torch")
    all_ok &= torch_ok
    
    if torch_ok:
        import torch
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️  WARNING: CUDA not available. GPU acceleration disabled!")
    
    print("-" * 60)
    print()
    
    if all_ok:
        print("✅ All dependencies are installed!")
        print()
        print("You can now run: python laptop_client_gui.py")
    else:
        print("❌ Some dependencies are missing!")
        print()
        print("Install all dependencies with:")
        print("  pip install -r requirements.txt")
        print()
        print("For PyTorch with CUDA 11.8:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
