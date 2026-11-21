# 🚀 Quick Setup Guide - Buyo Classifier GUI System

## 📋 What You Need

- ✅ Raspberry Pi with camera (Pi Camera Module or USB webcam)
- ✅ Laptop with GPU (NVIDIA recommended)
- ✅ Both devices on same Wi-Fi network
- ✅ Trained model file: `buyo_best.pth` (in project root)

---

## 🔧 Step 1: Setup Raspberry Pi Server

### Install Python Packages
```bash
cd raspberry_pi_server
pip3 install -r requirements.txt
```

### Optional: Pi Camera Module Support
```bash
pip3 install picamera2
```

### Optional: Robotic Arm Control
```bash
pip3 install RPi.GPIO adafruit-circuitpython-servokit
```

### Run Server
```bash
python3 pi_server_gui.py
```

### ✅ What You Should See
- GUI window opens
- Large IP address displayed at top (e.g., `192.168.1.105`)
- "📋 Copy IP" button available
- Click "▶ Start Server"
- Camera preview appears (if camera connected)
- Status shows "● RUNNING"

### 📝 Note the IP Address
**Copy this IP address** - you'll need it for the laptop client!

---

## 💻 Step 2: Setup Laptop Client

### Install Python Packages
```bash
cd laptop_client
pip install -r requirements.txt
```

### Install PyTorch with GPU Support
```bash
# For CUDA 11.8 (check your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (not recommended - slow)
pip install torch torchvision
```

### Verify Model File
Make sure `buyo_best.pth` exists in the **project root** (not inside laptop_client folder):
```
BUYO LEAVES/
├── buyo_best.pth          ← Should be here
├── laptop_client/
│   └── laptop_client_gui.py
└── raspberry_pi_server/
    └── pi_server_gui.py
```

### Run Client
```bash
# From project root directory
python laptop_client/laptop_client_gui.py
```

### ✅ What You Should See
- GUI window opens
- IP input field at top
- Status shows "● DISCONNECTED"
- Video feed area (black screen until connected)
- Manual control buttons panel
- Activity log panel

---

## 🔗 Step 3: Connect & Test

### On Laptop GUI:
1. **Enter Pi IP** in the input field (the one from Step 1)
2. **Click "🔍 Test"** - should show "✓ Connection successful!"
3. **Click "🔌 Connect"**
4. Video feed should appear showing Pi camera view
5. Status changes to "● CONNECTED" (green)

### Test Manual Controls:
1. Click "🏠 Home All" - arm should move to home position
2. Check Pi server GUI - command should appear in log
3. Click "🦾 Test Arm #1" - single arm should move
4. Commands have 2-second cooldown (prevents spam)

### Test Vision Detection:
1. Check "🎯 Enable Vision Detection" checkbox
2. Point camera at buyo leaf
3. Detection annotations appear on video
4. Activity log shows detection results
5. Auto-trigger (optional) - edit code to enable

---

## 🎯 Usage Patterns

### Manual Control Mode
- Detection OFF
- Use buttons to manually trigger arms
- Good for testing, calibration, batch processing

### Auto-Detection Mode
- Detection ON
- Model automatically detects buyo leaves
- Can auto-trigger arm (code customization needed)
- Good for autonomous operation

### Monitoring Mode
- Detection ON
- Just monitor classifications
- No arm triggers
- Good for quality checking

---

## 🔧 Customization Quick Reference

### Change Arm Control Code
File: `raspberry_pi_server/pi_server_gui.py`  
Function: `execute_arm_action()`
```python
# Line ~130 - Add your servo code here
if action == 'drop':
    kit.servo[channel].angle = ARM_DROP_ANGLE
    time.sleep(0.6)
    kit.servo[channel].angle = ARM_HOME_ANGLE
```

### Change Detection Model
File: `laptop_client/laptop_client_gui.py`  
Function: `detect_target()`
```python
# Line ~130 - Replace with your YOLO/custom model
result = detect_target(frame)
# Returns: detected, class, confidence, bbox, annotated_frame
```

### Change Network Settings
Both files have config section at top:
```python
DEFAULT_PORT = 5000
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
```

### Change Detection Threshold
File: `laptop_client/laptop_client_gui.py`
```python
DETECTION_THRESHOLD = 0.60    # Minimum confidence
BUYO_CONFIDENCE_MIN = 0.40    # Minimum for buyo
```

### Change Command Cooldown
File: `laptop_client/laptop_client_gui.py`
```python
COMMAND_COOLDOWN = 2.0  # seconds between commands
```

---

## ❌ Troubleshooting

### Pi Server Won't Start
- Check port 5000 not already in use: `sudo lsof -i :5000`
- Try different port in code
- Check camera permissions

### Camera Not Detected
```bash
# Enable camera
sudo raspi-config
# → Interface Options → Camera → Enable

# Check devices
ls /dev/video*

# Test picamera2
python3 -c "from picamera2 import Picamera2; print('OK')"
```

### Laptop Can't Connect
1. Ping Pi: `ping 192.168.1.105`
2. Check firewall on Pi: `sudo ufw status`
3. Try from browser: `http://192.168.1.105:5000/status`
4. Check both on same Wi-Fi

### Video Laggy
- Reduce resolution in Pi server code
- Lower JPEG quality (line ~60 in pi_server_gui.py)
- Check Wi-Fi signal strength

### Arm Not Moving
- Check `[SIMULATION]` in Pi log = libraries not installed
- Install: `pip3 install RPi.GPIO adafruit-circuitpython-servokit`
- Check servo power supply connected
- Verify I2C enabled: `sudo raspi-config` → Interface → I2C

### Model Not Loading on Laptop
- Check `buyo_best.pth` in project root
- Check terminal for loading errors
- Verify PyTorch installed: `python -c "import torch; print(torch.__version__)"`
- Check GPU detected: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 🎓 Next Steps

### Add New Commands
1. Edit `pi_server_gui.py` → add case in `execute_arm_action()`
2. Edit `laptop_client_gui.py` → add button in button grid
3. Button calls `self.send_command('new_action', {'param': 'value'})`

### Integrate YOLO Object Detection
Replace `detect_target()` in laptop client with:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model(frame)
# Draw bounding boxes on frame
```

### Enable Auto-Trigger
In `laptop_client_gui.py`, uncomment line ~520:
```python
if result['detected']:
    self.send_command('drop', {'arm': 1, 'class': result['class']})
```

### Log Detections to File
Add to detection code:
```python
with open('detections.csv', 'a') as f:
    f.write(f"{timestamp},{class_name},{confidence}\n")
```

---

## 📞 Support

Issues? Check:
1. ✅ Both devices on same network
2. ✅ Model file exists and loads
3. ✅ Camera detected and working
4. ✅ Firewall allows port 5000
5. ✅ All dependencies installed

Still stuck? Check terminal output for error messages!

---

**🎉 Happy Classifying!**
