#!/usr/bin/env python3
"""
Raspberry Pi Peripheral Detection Tool
======================================
Detects all connected peripherals and hardware components:
- Cameras (Pi Camera Module & USB webcams)
- GPIO pins and I2C devices
- Servo controllers (PCA9685)
- USB devices
- Network interfaces
- Storage devices
"""

import os
import sys
import subprocess
import platform

def print_header(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def run_command(cmd, shell=False):
    """Run a shell command and return output"""
    try:
        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()
        result = subprocess.run(cmd, capture_output=True, text=True, shell=shell)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def check_system_info():
    """Check basic system information"""
    print_header("SYSTEM INFORMATION")
    
    print(f"Platform: {platform.system()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Check if running on Raspberry Pi
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip('\x00')
            print(f"Device Model: {model}")
    except:
        print("Device Model: Not a Raspberry Pi or model file not found")
    
    # Kernel version
    print(f"Kernel: {platform.release()}")
    
    # Python version
    print(f"Python: {sys.version.split()[0]}")

def check_cameras():
    """Detect all connected cameras"""
    print_header("CAMERAS")
    
    # Check for libcamera devices
    print("\n📷 Libcamera Devices (Pi Camera Module):")
    libcam_output = run_command("libcamera-hello --list-cameras")
    if libcam_output and "Available cameras" in libcam_output:
        print(libcam_output)
    else:
        print("  ✗ No libcamera devices found")
        print("  Run: sudo raspi-config → Interface Options → Legacy Camera (disable)")
        print("  Then: sudo raspi-config → Interface Options → Camera (enable)")
    
    # Check for V4L2 devices (USB webcams)
    print("\n📹 V4L2 Devices (USB Webcams):")
    v4l2_output = run_command("v4l2-ctl --list-devices")
    if v4l2_output and v4l2_output != "Error":
        print(v4l2_output)
    else:
        print("  ✗ No V4L2 devices found")
    
    # List /dev/video* devices
    print("\n🎥 Video Device Files:")
    video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
    if video_devices:
        for device in sorted(video_devices):
            device_path = f"/dev/{device}"
            print(f"  ✓ {device_path}")
            # Get device info
            info = run_command(f"v4l2-ctl --device={device_path} --info")
            if "Card type" in info:
                for line in info.split('\n'):
                    if 'Card type' in line or 'Driver name' in line:
                        print(f"      {line.strip()}")
    else:
        print("  ✗ No /dev/video* devices found")
    
    # Test Python camera libraries
    print("\n🐍 Python Camera Libraries:")
    
    # Check picamera2
    try:
        from picamera2 import Picamera2
        cameras = Picamera2.global_camera_info()
        print(f"  ✓ picamera2: Available ({len(cameras)} camera(s))")
        for i, cam in enumerate(cameras):
            model = cam.get('Model', 'Unknown')
            location = cam.get('Location', '')
            print(f"      Camera {i}: {model} ({location})")
    except ImportError:
        print("  ✗ picamera2: Not installed")
        print("      Install: sudo apt install python3-picamera2")
    except Exception as e:
        print(f"  ⚠ picamera2: Error - {e}")
    
    # Check OpenCV
    try:
        import cv2
        print(f"  ✓ opencv-python: {cv2.__version__}")
        # Try to open camera 0
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("      Can open /dev/video0")
            cap.release()
        else:
            print("      Cannot open /dev/video0")
    except ImportError:
        print("  ✗ opencv-python: Not installed")

def check_gpio_i2c():
    """Check GPIO and I2C devices"""
    print_header("GPIO & I2C")
    
    # Check if GPIO is accessible
    print("\n🔌 GPIO:")
    if os.path.exists('/dev/gpiomem'):
        print("  ✓ /dev/gpiomem exists (GPIO accessible)")
    else:
        print("  ✗ /dev/gpiomem not found")
    
    # Check RPi.GPIO library
    try:
        import RPi.GPIO as GPIO
        print(f"  ✓ RPi.GPIO: Available")
    except ImportError:
        print("  ✗ RPi.GPIO: Not installed")
        print("      Install: pip install RPi.GPIO")
    
    # Check I2C devices
    print("\n🔗 I2C Devices:")
    i2c_output = run_command("i2cdetect -y 1")
    if i2c_output and i2c_output != "Error":
        print(i2c_output)
        
        # Parse for detected devices
        lines = i2c_output.split('\n')[1:]  # Skip header
        devices_found = []
        for line in lines:
            parts = line.split()
            if len(parts) > 1:
                for addr in parts[1:]:
                    if addr not in ['--', 'UU'] and len(addr) == 2:
                        devices_found.append(addr)
        
        if devices_found:
            print(f"\n  ✓ Found {len(devices_found)} I2C device(s):")
            for addr in devices_found:
                addr_int = int(addr, 16)
                device_type = "Unknown"
                if addr_int == 0x40:
                    device_type = "PCA9685 PWM/Servo Controller"
                elif addr_int == 0x70:
                    device_type = "PCA9685 PWM/Servo Controller (Alt address)"
                print(f"      0x{addr} ({addr_int}) - {device_type}")
        else:
            print("\n  ✗ No I2C devices detected")
    else:
        print("  ✗ i2cdetect not available or I2C not enabled")
        print("      Enable: sudo raspi-config → Interface Options → I2C")
    
    # Check if I2C is enabled
    print("\n🔧 I2C Status:")
    if os.path.exists('/dev/i2c-1'):
        print("  ✓ /dev/i2c-1 exists (I2C enabled)")
    else:
        print("  ✗ /dev/i2c-1 not found (I2C disabled?)")
        print("      Enable: sudo raspi-config → Interface Options → I2C")

def check_servo_controllers():
    """Check for servo controller libraries and hardware"""
    print_header("SERVO CONTROLLERS")
    
    # Check adafruit-circuitpython-servokit
    try:
        from adafruit_servokit import ServoKit
        print("  ✓ adafruit-circuitpython-servokit: Available")
        
        # Try to initialize ServoKit
        try:
            kit = ServoKit(channels=16)
            print("  ✓ ServoKit initialized (PCA9685 detected)")
            print(f"      16 channels available")
        except Exception as e:
            print(f"  ⚠ ServoKit initialization failed: {e}")
            print("      Check if PCA9685 is connected via I2C")
    except ImportError:
        print("  ✗ adafruit-circuitpython-servokit: Not installed")
        print("      Install: pip install adafruit-circuitpython-servokit")
    
    # Check adafruit-pca9685
    try:
        import board
        import busio
        from adafruit_pca9685 import PCA9685
        print("  ✓ adafruit-pca9685: Available")
        
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            pca = PCA9685(i2c)
            print("  ✓ PCA9685 initialized successfully")
            pca.deinit()
        except Exception as e:
            print(f"  ⚠ PCA9685 initialization failed: {e}")
    except ImportError:
        print("  ✗ adafruit-pca9685: Not installed")

def check_usb_devices():
    """List all USB devices"""
    print_header("USB DEVICES")
    
    lsusb_output = run_command("lsusb")
    if lsusb_output and lsusb_output != "Error":
        print(lsusb_output)
    else:
        print("  ✗ lsusb not available")
    
    # Detailed USB device info
    print("\n📋 Detailed USB Info:")
    usb_devices_output = run_command("ls -l /dev/bus/usb/*/* 2>/dev/null", shell=True)
    if usb_devices_output:
        device_count = len(usb_devices_output.split('\n'))
        print(f"  Found {device_count} USB device file(s)")

def check_storage():
    """Check storage devices"""
    print_header("STORAGE DEVICES")
    
    # List block devices
    lsblk_output = run_command("lsblk")
    if lsblk_output:
        print(lsblk_output)
    
    # Disk usage
    print("\n💾 Disk Usage:")
    df_output = run_command("df -h")
    if df_output:
        print(df_output)

def check_network():
    """Check network interfaces"""
    print_header("NETWORK INTERFACES")
    
    # List network interfaces
    ip_output = run_command("ip addr")
    if ip_output:
        print(ip_output)
    else:
        ifconfig_output = run_command("ifconfig")
        if ifconfig_output:
            print(ifconfig_output)
    
    # Get hostname and IP
    print("\n🌐 Network Info:")
    hostname = run_command("hostname")
    print(f"  Hostname: {hostname}")
    
    hostname_i = run_command("hostname -I")
    if hostname_i:
        print(f"  IP Addresses: {hostname_i}")

def check_installed_packages():
    """Check relevant Python packages"""
    print_header("PYTHON PACKAGES")
    
    packages = [
        'picamera2',
        'opencv-python',
        'cv2',
        'numpy',
        'PIL',
        'flask',
        'torch',
        'torchvision',
        'RPi.GPIO',
        'adafruit-circuitpython-servokit',
        'requests',
    ]
    
    for package in packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"  ✓ opencv-python: {cv2.__version__}")
            elif package == 'PIL':
                from PIL import Image
                print(f"  ✓ Pillow: {Image.__version__}")
            else:
                mod = __import__(package.replace('-', '_'))
                version = getattr(mod, '__version__', 'installed')
                print(f"  ✓ {package}: {version}")
        except ImportError:
            print(f"  ✗ {package}: Not installed")

def main():
    """Main detection routine"""
    print("\n" + "="*70)
    print("  RASPBERRY PI PERIPHERAL DETECTION TOOL")
    print("="*70)
    print("\nScanning all connected peripherals...\n")
    
    check_system_info()
    check_cameras()
    check_gpio_i2c()
    check_servo_controllers()
    check_usb_devices()
    check_storage()
    check_network()
    check_installed_packages()
    
    print("\n" + "="*70)
    print("  SCAN COMPLETE")
    print("="*70)
    print("\n✓ Peripheral detection finished!")
    print("\nTo run specific checks:")
    print("  - Camera test: libcamera-hello")
    print("  - I2C scan: i2cdetect -y 1")
    print("  - GPIO test: gpio readall")
    print("  - USB devices: lsusb -v")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Scan interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Error during scan: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
