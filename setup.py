#!/usr/bin/env python3
"""
Buyo Leaf Classifier - Automated Setup Script
==============================================
One-click setup that:
1. Detects device type (Raspberry Pi or Laptop/Desktop)
2. Creates virtual environment
3. Installs all required dependencies
4. Verifies installation
5. Provides next steps

Usage:
    python setup.py              # Interactive setup
    python setup.py --pi         # Force Raspberry Pi setup
    python setup.py --laptop     # Force Laptop setup
    python setup.py --no-venv    # Skip venv creation
"""

import os
import sys
import platform
import subprocess
import argparse
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}→ {text}{Colors.ENDC}")

def run_command(cmd, description="", check=True, shell=False):
    """Run shell command and return output"""
    try:
        print_info(f"{description}...")
        result = subprocess.run(
            cmd if not shell else cmd,
            shell=shell,
            check=check,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for large downloads
        )
        print_success(f"{description} - Done")
        return result.returncode == 0, result.stdout
    except subprocess.CalledProcessError as e:
        print_error(f"{description} - Failed")
        if e.stderr:
            print(f"   Error details: {e.stderr[:200]}")
        return False, e.stderr
    except subprocess.TimeoutExpired:
        print_error(f"{description} - Timeout (>10 minutes)")
        return False, "Timeout"
    except Exception as e:
        print_error(f"{description} - Error: {e}")
        return False, str(e)

def detect_device_type():
    """Detect if running on Raspberry Pi or Laptop/Desktop"""
    try:
        # Check for Raspberry Pi
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                if 'raspberry pi' in model:
                    return 'raspberry_pi'
        
        # Check platform
        machine = platform.machine().lower()
        if 'arm' in machine or 'aarch64' in machine:
            # Likely Raspberry Pi or ARM device
            return 'raspberry_pi'
        
        # Default to laptop/desktop
        return 'laptop'
    except:
        return 'laptop'

def check_python_version():
    """Check if Python version is compatible"""
    print_info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print_error(f"Python {version.major}.{version.minor} detected - Need Python 3.7+")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def create_venv(venv_path):
    """Create virtual environment"""
    print_header("Creating Virtual Environment")
    
    if venv_path.exists():
        print_warning(f"Virtual environment already exists at {venv_path}")
        response = input("  Recreate? (y/N): ").strip().lower()
        if response == 'y':
            print_info("Removing existing venv...")
            import shutil
            shutil.rmtree(venv_path)
        else:
            print_info("Using existing virtual environment")
            return True
    
    success, _ = run_command(
        [sys.executable, '-m', 'venv', str(venv_path)],
        f"Creating venv at {venv_path}"
    )
    return success

def get_venv_python(venv_path):
    """Get path to Python in venv"""
    if platform.system() == 'Windows':
        return venv_path / 'Scripts' / 'python.exe'
    else:
        return venv_path / 'bin' / 'python'

def get_venv_pip(venv_path):
    """Get path to pip in venv"""
    if platform.system() == 'Windows':
        return venv_path / 'Scripts' / 'pip.exe'
    else:
        return venv_path / 'bin' / 'pip'

def upgrade_pip(pip_path):
    """Upgrade pip to latest version"""
    print_info("Upgrading pip...")
    try:
        # Use python -m pip to avoid Windows file locking issues
        python_path = pip_path.parent / 'python.exe' if platform.system() == 'Windows' else pip_path.parent / 'python'
        result = subprocess.run(
            [str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print_success("Pip upgraded successfully")
            return True
        else:
            print_warning(f"Pip upgrade had issues (continuing anyway)")
            return True  # Continue even if upgrade fails
    except Exception as e:
        print_warning(f"Could not upgrade pip: {e} (continuing anyway)")
        return True  # Continue even if upgrade fails

def install_requirements(pip_path, requirements_file):
    """Install requirements from file"""
    if not requirements_file.exists():
        print_error(f"Requirements file not found: {requirements_file}")
        return False
    
    print_info(f"Installing from {requirements_file.name}...")
    success, _ = run_command(
        [str(pip_path), 'install', '-r', str(requirements_file)],
        f"Installing packages from {requirements_file.name}"
    )
    return success

def detect_gpu():
    """Detect if NVIDIA GPU is available"""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print_success("NVIDIA GPU detected")
            # Try to get CUDA version
            if 'CUDA Version:' in result.stdout:
                import re
                match = re.search(r'CUDA Version:\s+(\d+\.\d+)', result.stdout)
                if match:
                    cuda_version = match.group(1)
                    print_info(f"CUDA Version: {cuda_version}")
                    return True, cuda_version
            return True, None
        return False, None
    except:
        return False, None

def install_pytorch(pip_path, device_type):
    """Install PyTorch with appropriate backend"""
    print_header("Installing PyTorch")
    
    # Get python executable for better compatibility on Windows
    python_path = pip_path.parent / 'python.exe' if platform.system() == 'Windows' else pip_path.parent / 'python'
    
    if device_type == 'raspberry_pi':
        print_info("Installing PyTorch for Raspberry Pi (CPU only)...")
        print_info("This may take 5-10 minutes...")
        # For Raspberry Pi, use CPU version
        try:
            result = subprocess.run(
                [str(python_path), '-m', 'pip', 'install', 'torch', 'torchvision', 
                 '--index-url', 'https://download.pytorch.org/whl/cpu'],
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes for large download
            )
            if result.returncode == 0:
                print_success("PyTorch CPU installed successfully")
                return True
            else:
                print_error("PyTorch installation failed")
                print(f"   Try manually: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
                return False
        except Exception as e:
            print_error(f"PyTorch installation error: {e}")
            return False
    
    else:
        # For laptop/desktop, check for GPU
        has_gpu, cuda_version = detect_gpu()
        
        if has_gpu:
            print_info("GPU detected - Installing PyTorch with CUDA support...")
            
            # Determine CUDA version for PyTorch
            if cuda_version:
                if cuda_version.startswith('12.'):
                    cuda_tag = 'cu121'
                elif cuda_version.startswith('11.8'):
                    cuda_tag = 'cu118'
                elif cuda_version.startswith('11.'):
                    cuda_tag = 'cu118'  # Default to 11.8
                else:
                    cuda_tag = 'cu118'  # Safe default
                
                print_info(f"Installing PyTorch for CUDA {cuda_tag}...")
            else:
                cuda_tag = 'cu118'  # Safe default
                print_warning(f"Could not detect CUDA version, using default: {cuda_tag}")
            
            print_info("This may take 5-10 minutes for large download...")
            
            try:
                result = subprocess.run(
                    [str(python_path), '-m', 'pip', 'install', 'torch', 'torchvision', 
                     '--index-url', f'https://download.pytorch.org/whl/{cuda_tag}'],
                    capture_output=True,
                    text=True,
                    timeout=900  # 15 minutes for large download
                )
                if result.returncode == 0:
                    print_success(f"PyTorch with {cuda_tag} installed successfully")
                    return True
                else:
                    print_error(f"PyTorch installation failed")
                    print_warning("Trying CPU version as fallback...")
                    # Fallback to CPU version
                    result = subprocess.run(
                        [str(python_path), '-m', 'pip', 'install', 'torch', 'torchvision'],
                        capture_output=True,
                        text=True,
                        timeout=900
                    )
                    if result.returncode == 0:
                        print_success("PyTorch CPU installed (will be slower)")
                        return True
                    else:
                        print_error("PyTorch CPU installation also failed")
                        print(f"   Try manually: pip install torch torchvision")
                        return False
            except Exception as e:
                print_error(f"PyTorch installation error: {e}")
                return False
        else:
            print_warning("No GPU detected - Installing PyTorch CPU version")
            print_warning("This will be slower for inference!")
            print_info("This may take 5-10 minutes...")
            try:
                result = subprocess.run(
                    [str(python_path), '-m', 'pip', 'install', 'torch', 'torchvision'],
                    capture_output=True,
                    text=True,
                    timeout=900
                )
                if result.returncode == 0:
                    print_success("PyTorch CPU installed successfully")
                    return True
                else:
                    print_error("PyTorch installation failed")
                    return False
            except Exception as e:
                print_error(f"PyTorch installation error: {e}")
                return False

def install_pi_specific_packages(pip_path):
    """Install Raspberry Pi specific packages"""
    print_header("Installing Raspberry Pi Specific Packages")
    
    # Picamera2
    print_info("Attempting to install picamera2...")
    print_warning("Note: picamera2 may need to be installed via apt on some systems")
    success, _ = run_command(
        [str(pip_path), 'install', 'picamera2'],
        "Installing picamera2",
        check=False  # Don't fail if this doesn't work
    )
    
    if not success:
        print_warning("picamera2 installation via pip failed")
        print_info("If using Pi Camera Module, install with:")
        print_info("  sudo apt install -y python3-picamera2")
    
    # GPIO and ServoKit
    print_info("Installing robotic arm libraries...")
    success1, _ = run_command(
        [str(pip_path), 'install', 'RPi.GPIO'],
        "Installing RPi.GPIO",
        check=False
    )
    
    success2, _ = run_command(
        [str(pip_path), 'install', 'adafruit-circuitpython-servokit'],
        "Installing ServoKit",
        check=False
    )
    
    if not success1 or not success2:
        print_warning("Some arm control libraries failed to install")
        print_info("Server will run in simulation mode")
    
    return True

def verify_installation(python_path, device_type):
    """Verify that key packages are installed"""
    print_header("Verifying Installation")
    
    packages_to_check = [
        ('numpy', 'import numpy'),
        ('opencv', 'import cv2'),
        ('pillow', 'from PIL import Image'),
    ]
    
    if device_type == 'raspberry_pi':
        packages_to_check.extend([
            ('flask', 'import flask'),
            ('torch', 'import torch'),
        ])
    else:
        packages_to_check.extend([
            ('requests', 'import requests'),
            ('torch', 'import torch'),
        ])
    
    all_ok = True
    for name, import_stmt in packages_to_check:
        success, output = run_command(
            [str(python_path), '-c', import_stmt],
            f"Checking {name}",
            check=False
        )
        if not success:
            print_error(f"{name} not installed correctly")
            all_ok = False
    
    if all_ok:
        print_success("All core packages verified!")
    else:
        print_warning("Some packages failed verification")
    
    return all_ok

def create_activation_instructions(venv_path, device_type):
    """Create instructions for activating venv and running the application"""
    print_header("Setup Complete!")
    
    if platform.system() == 'Windows':
        activate_cmd = f"{venv_path}\\Scripts\\activate"
        python_cmd = f"{venv_path}\\Scripts\\python"
    else:
        activate_cmd = f"source {venv_path}/bin/activate"
        python_cmd = f"{venv_path}/bin/python"
    
    print(f"{Colors.OKGREEN}{Colors.BOLD}Next Steps:{Colors.ENDC}\n")
    
    print(f"{Colors.OKCYAN}1. Activate virtual environment:{Colors.ENDC}")
    print(f"   {activate_cmd}\n")
    
    if device_type == 'raspberry_pi':
        print(f"{Colors.OKCYAN}2. Run Raspberry Pi Server:{Colors.ENDC}")
        print(f"   cd raspberry_pi_server")
        print(f"   {python_cmd} pi_server_gui.py\n")
        
        print(f"{Colors.OKCYAN}3. Optional - System packages:{Colors.ENDC}")
        print(f"   sudo apt install -y python3-picamera2  # For Pi Camera Module")
        print(f"   sudo raspi-config  # Enable Camera and I2C\n")
    
    else:
        print(f"{Colors.OKCYAN}2. Make sure buyo_best.pth model file is in project root{Colors.ENDC}\n")
        
        print(f"{Colors.OKCYAN}3. Run Laptop Client:{Colors.ENDC}")
        print(f"   cd laptop_client")
        print(f"   {python_cmd} laptop_client_gui.py\n")
    
    print(f"{Colors.OKGREEN}{'='*70}{Colors.ENDC}")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Buyo Leaf Classifier Setup')
    parser.add_argument('--pi', action='store_true', help='Force Raspberry Pi setup')
    parser.add_argument('--laptop', action='store_true', help='Force Laptop setup')
    parser.add_argument('--no-venv', action='store_true', help='Skip virtual environment creation')
    args = parser.parse_args()
    
    print_header("Buyo Leaf Classifier - Automated Setup")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Detect device type
    if args.pi:
        device_type = 'raspberry_pi'
        print_info("Device type: Raspberry Pi (forced)")
    elif args.laptop:
        device_type = 'laptop'
        print_info("Device type: Laptop/Desktop (forced)")
    else:
        device_type = detect_device_type()
        print_info(f"Device type detected: {device_type.replace('_', ' ').title()}")
    
    # Confirm with user
    print(f"\n{Colors.BOLD}Setup will configure this system as: {device_type.replace('_', ' ').upper()}{Colors.ENDC}")
    response = input("Continue? (Y/n): ").strip().lower()
    if response == 'n':
        print("Setup cancelled.")
        sys.exit(0)
    
    # Get project root
    project_root = Path(__file__).parent.absolute()
    print_info(f"Project root: {project_root}")
    
    # Virtual environment path
    venv_path = project_root / 'venv'
    
    # Create virtual environment (unless skipped)
    if not args.no_venv:
        if not create_venv(venv_path):
            print_error("Failed to create virtual environment")
            sys.exit(1)
        
        python_path = get_venv_python(venv_path)
        pip_path = get_venv_pip(venv_path)
    else:
        print_warning("Skipping venv creation - using system Python")
        python_path = sys.executable
        pip_path = 'pip' if platform.system() == 'Windows' else 'pip3'
    
    # Upgrade pip
    print_header("Upgrading pip")
    upgrade_pip(pip_path)
    
    # Install base packages
    print_header("Installing Core Dependencies")
    
    if device_type == 'raspberry_pi':
        requirements_file = project_root / 'raspberry_pi_server' / 'requirements.txt'
    else:
        requirements_file = project_root / 'laptop_client' / 'requirements.txt'
    
    if not install_requirements(pip_path, requirements_file):
        print_error("Failed to install requirements")
        sys.exit(1)
    
    # Install PyTorch
    pytorch_success = install_pytorch(pip_path, device_type)
    if not pytorch_success:
        print_error("PyTorch installation failed")
        print_warning("\nYou can install PyTorch manually after setup:")
        if device_type == 'laptop':
            print_info("  Activate venv, then run:")
            if platform.system() == 'Windows':
                print_info("    venv\\Scripts\\activate")
            else:
                print_info("    source venv/bin/activate")
            print_info("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        else:
            print_info("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        
        response = input("\nContinue without PyTorch? (y/N): ").strip().lower()
        if response != 'y':
            print_error("Setup cancelled")
            sys.exit(1)
        print_warning("Continuing without PyTorch...")
    
    # Install device-specific packages
    if device_type == 'raspberry_pi':
        install_pi_specific_packages(pip_path)
    
    # Verify installation
    if pytorch_success:
        verify_installation(python_path, device_type)
    else:
        print_warning("Skipping verification due to PyTorch installation failure")
    
    # Show next steps
    create_activation_instructions(venv_path, device_type)
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ Setup completed successfully!{Colors.ENDC}\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}Setup failed with error: {e}{Colors.ENDC}")
        sys.exit(1)
