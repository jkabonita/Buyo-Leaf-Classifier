#!/usr/bin/env python3
"""
Raspberry Pi Camera Client - Remote Processing
===============================================
Runs on Raspberry Pi: captures frames, sends to laptop for GPU processing,
receives trigger commands, and controls robotic arm.

Setup:
1. Install: sudo apt-get install -y python3-picamera2
2. Install: pip3 install opencv-python numpy requests pillow
3. Install arm libraries: pip3 install adafruit-circuitpython-servokit RPi.GPIO
4. Get laptop IP from server startup
5. Run: python3 raspberry_pi_camera_client.py <LAPTOP_IP>

Example:
  python3 raspberry_pi_camera_client.py 192.168.1.100
"""

import sys
import os
import time
import cv2
import numpy as np
import requests
import base64
from collections import deque

# Import picamera2
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    site_pkg = '/usr/lib/python3/dist-packages'
    if os.path.exists(site_pkg) and site_pkg not in sys.path:
        sys.path.insert(0, site_pkg)
        try:
            from picamera2 import Picamera2
            PICAMERA2_AVAILABLE = True
        except ImportError:
            PICAMERA2_AVAILABLE = False
    else:
        PICAMERA2_AVAILABLE = False

if not PICAMERA2_AVAILABLE:
    print("="*64)
    print("  ❌ ERROR: picamera2 not available!")
    print("="*64)
    print("\n  Install: sudo apt-get install -y python3-picamera2\n")
    sys.exit(1)

# Import robotic arm libraries
ENABLE_ROBOTIC_ARM = False
try:
    import RPi.GPIO as GPIO
    from adafruit_servokit import ServoKit
    ENABLE_ROBOTIC_ARM = True
except ImportError:
    print("[INFO] Robotic arm libraries not available")
    print("      Running in camera-only mode")

# ============================================================================
#  CONFIGURATION
# ============================================================================
ARM_CHANNELS = {1: 3, 2: 2, 3: 1, 4: 0, 5: 8, 6: 7, 7: 6, 8: 5, 9: 4}
ARM_DROP_ANGLE = 60
ARM_HOME_ANGLE = 0
CURRENT_ARM = 1
TRIGGER_COOLDOWN = 5.0  # seconds

CLASS_NAMES = ['CLASS A', 'CLASS B', 'CLASS C', 'CLASS D', 'CLASS E']

kit = None
if ENABLE_ROBOTIC_ARM:
    try:
        kit = ServoKit(channels=16, address=0x40)
        print("[ARM] ✓ ServoKit initialized")
    except Exception as e:
        print(f"[ARM] ✗ Init failed: {e}")
        ENABLE_ROBOTIC_ARM = False

# ============================================================================
#  ROBOTIC ARM CONTROL
# ============================================================================
last_trigger_time = 0

def trigger_arm_drop(arm_num, class_name):
    """Trigger robotic arm to drop leaf"""
    global kit, last_trigger_time
    
    if not ENABLE_ROBOTIC_ARM or kit is None:
        print(f"[ARM] Simulated: Arm #{arm_num} → {class_name}")
        return
    
    channel = ARM_CHANNELS.get(arm_num)
    if channel is None:
        print(f"[ARM] ✗ Invalid arm #{arm_num}")
        return
    
    try:
        print(f"[ARM] ⬇️  Arm #{arm_num} dropping {class_name}")
        kit.servo[channel].angle = ARM_DROP_ANGLE
        time.sleep(0.6)
        kit.servo[channel].angle = ARM_HOME_ANGLE
        time.sleep(0.4)
        last_trigger_time = time.time()
        print(f"[ARM] ✓ Drop complete")
    except Exception as e:
        print(f"[ARM] ✗ Error: {e}")
        try:
            kit.servo[channel].angle = ARM_HOME_ANGLE
        except:
            pass

# ============================================================================
#  NETWORK COMMUNICATION
# ============================================================================
def send_frame_for_classification(frame_bgr, server_url):
    """Send frame to laptop server and get classification result"""
    try:
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send to server
        response = requests.post(
            f'http://{server_url}:5000/classify',
            json={'frame': frame_b64},
            timeout=5.0
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[ERROR] Server returned {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("[ERROR] Request timeout")
        return None
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to server")
        return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

# ============================================================================
#  MAIN
# ============================================================================
if len(sys.argv) < 2:
    print("="*64)
    print("  ❌ ERROR: Missing laptop IP address")
    print("="*64)
    print("\n  Usage: python3 raspberry_pi_camera_client.py <LAPTOP_IP>")
    print("\n  Example:")
    print("    python3 raspberry_pi_camera_client.py 192.168.1.100")
    print("\n" + "="*64 + "\n")
    sys.exit(1)

LAPTOP_IP = sys.argv[1]

print("="*64)
print("  Buyo Camera Client - Raspberry Pi")
print("="*64)
print(f"  Server: {LAPTOP_IP}:5000")
print(f"  🤖 Robotic arm: {'ENABLED' if ENABLE_ROBOTIC_ARM else 'DISABLED'}")
print("="*64)

# Test connection
print("\n[INIT] Testing connection to laptop...")
try:
    response = requests.get(f'http://{LAPTOP_IP}:5000/status', timeout=5)
    if response.status_code == 200:
        status = response.json()
        print(f"[INIT] ✓ Connected to {status['model']} on {status['device']}")
    else:
        print("[INIT] ✗ Server not responding properly")
        sys.exit(1)
except:
    print(f"[INIT] ✗ Cannot connect to {LAPTOP_IP}:5000")
    print("       Make sure laptop_classifier_server.py is running")
    sys.exit(1)

# Initialize camera
print("\n[INIT] Starting camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    buffer_count=2
)
picam2.configure(config)
picam2.start()
time.sleep(2)
print("[INIT] ✓ Camera ready")

print("\n" + "="*64)
print("  🚀 CLIENT RUNNING")
print("="*64)
print("  Capturing frames → Laptop GPU → Arm trigger")
print("  Press Ctrl+C to stop")
print("="*64 + "\n")

# Stats
frame_count = 0
fps_counter = deque(maxlen=30)
trigger_count = 0

try:
    while True:
        t0 = time.time()
        
        # Capture frame
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Send for classification
        result = send_frame_for_classification(frame_bgr, LAPTOP_IP)
        
        if result:
            pred_class = result['class']
            confidence = result['confidence']
            is_confident = result['is_confident']
            is_buyo = result['is_buyo']
            should_trigger = result['trigger_arm']
            
            # Trigger arm if commanded by laptop
            if should_trigger:
                current_time = time.time()
                if current_time - last_trigger_time >= TRIGGER_COOLDOWN:
                    trigger_arm_drop(CURRENT_ARM, pred_class)
                    trigger_count += 1
                else:
                    cooldown_left = TRIGGER_COOLDOWN - (current_time - last_trigger_time)
                    print(f"[ARM] ⏳ Cooldown {cooldown_left:.1f}s remaining")
            
            # Status
            status = "✓" if should_trigger else ("✗" if not is_buyo else "?")
            fps_counter.append(1.0 / max(0.001, time.time() - t0))
            fps = np.mean(fps_counter)
            
            if frame_count % 10 == 0:
                print(f"[{status}] {pred_class} ({confidence*100:.0f}%) | FPS: {fps:.1f} | Triggers: {trigger_count}")
        
        frame_count += 1
        
        # Small delay
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\n[STOP] Interrupted by user")

finally:
    print("\n[CLEANUP] Shutting down...")
    picam2.stop()
    if ENABLE_ROBOTIC_ARM and kit:
        try:
            GPIO.cleanup()
        except:
            pass
    print("[CLEANUP] ✓ Done\n")
