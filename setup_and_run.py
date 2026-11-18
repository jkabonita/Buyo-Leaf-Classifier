#!/usr/bin/env python3
"""
Setup and Run Script for Raspberry Pi Classifier
=================================================
Automatically activates venv and runs the classifier
"""

import os
import sys
import subprocess

def run_command(cmd, description=""):
    """Run a shell command and print output"""
    if description:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    return result.returncode == 0

def main():
    print("="*60)
    print("  BUYO LEAF CLASSIFIER - SETUP & RUN")
    print("="*60)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"\nWorking directory: {script_dir}")
    
    # Check if venv exists
    venv_python = os.path.join(script_dir, "venv", "bin", "python")
    venv_activate = os.path.join(script_dir, "venv", "bin", "activate")
    
    if not os.path.exists(venv_python):
        print("\n⚠️  Virtual environment not found!")
        print("Creating virtual environment...")
        
        if not run_command("python3 -m venv venv", "Creating venv"):
            print("❌ Failed to create venv")
            return 1
        
        print("✅ Virtual environment created")
    
    # Check if PyTorch is installed in venv
    print("\n🔍 Checking PyTorch installation...")
    
    check_torch = f"{venv_python} -c 'import torch; print(torch.__version__)'"
    result = subprocess.run(check_torch, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("\n📦 Installing dependencies (this may take 10-15 minutes)...")
        
        # Install PyTorch
        print("\n1️⃣  Installing PyTorch (CPU version)...")
        install_torch = f"{venv_python} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        if not run_command(install_torch):
            print("❌ Failed to install PyTorch")
            return 1
        
        # Install other dependencies
        print("\n2️⃣  Installing OpenCV, Pillow, NumPy...")
        install_deps = f"{venv_python} -m pip install opencv-python pillow numpy"
        if not run_command(install_deps):
            print("❌ Failed to install dependencies")
            return 1
        
        # Install robotic arm libraries
        print("\n3️⃣  Installing robotic arm libraries...")
        install_arm = f"{venv_python} -m pip install adafruit-circuitpython-servokit RPi.GPIO"
        run_command(install_arm)  # Don't fail if this doesn't install (may not be on Pi)
        
        print("\n✅ All dependencies installed!")
    else:
        torch_version = result.stdout.strip()
        print(f"✅ PyTorch already installed: {torch_version}")
    
    # Check if model file exists
    print("\n🔍 Checking for model files...")
    pth_files = [f for f in os.listdir(script_dir) if f.endswith('.pth')]
    
    if not pth_files:
        print("\n⚠️  WARNING: No .pth model files found!")
        print("Please make sure you have a trained model file (buyo_best.pth)")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    else:
        print(f"✅ Found model files: {', '.join(pth_files)}")
    
    # Run the classifier
    print("\n" + "="*60)
    print("  🚀 STARTING CLASSIFIER")
    print("="*60)
    print("\nKeyboard controls:")
    print("  Q - Quit")
    print("  U - Upload image")
    print("  B - Batch upload")
    print("  S - Save screenshot")
    print("  R - Reset smoothing")
    print("\nPress Ctrl+C to stop\n")
    
    # Run with venv python
    run_cmd = f"{venv_python} raspberry_pi_classifier.py"
    
    try:
        subprocess.run(run_cmd, shell=True)
    except KeyboardInterrupt:
        print("\n\n✅ Classifier stopped by user")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
