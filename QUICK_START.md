# Buyo Leaf Classifier - Quick Start Guide

## System Architecture

```
┌─────────────────────┐         Network         ┌──────────────────────┐
│   Laptop Client     │ ◄─────────────────────► │   Raspberry Pi       │
│   (Windows/Mac)     │      HTTP/MJPEG         │   Server             │
├─────────────────────┤                         ├──────────────────────┤
│ - ML Classification │                         │ - Camera Streaming   │
│ - Image Saving      │    Commands & Video     │ - Robotic Arm        │
│ - GUI Interface     │    ──────────────────►  │ - Servo Control      │
│                     │                         │                      │
└─────────────────────┘                         └──────────────────────┘
        Local Storage                               GPIO Hardware
```

## Installation

### Step 1: Install Dependencies

Run the installation script:
```batch
install_all_dependencies.bat
```

This installs:
- ✅ PyTorch (ML framework)
- ✅ OpenCV (Camera & image processing)
- ✅ Flask (Web server for Pi)
- ✅ Pillow (Image handling)
- ✅ NumPy (Numerical operations)
- ✅ Requests (HTTP client)

### Step 2: Verify Installation

```batch
python test_installation.py
```

Should show all packages with ✓ checkmarks.

### Step 3: Download Model

Place your trained model file:
```
buyo_best.pth   ← Put this in project root
```

## Running the System

### Option 1: On Laptop Only (Testing ML)

```batch
python laptop_client\laptop_client_gui.py
```

Features:
- ✅ ML classification on laptop
- ✅ Save images locally
- ❌ No camera streaming (requires Pi connection)
- ❌ No robotic arm control (requires Pi connection)

### Option 2: Full System (Laptop + Raspberry Pi)

#### On Raspberry Pi:
```bash
python raspberry_pi_server/pi_server_gui.py
```

Note the IP address shown (e.g., `10.116.37.31`)

#### On Laptop:
```batch
python laptop_client\laptop_client_gui.py
```

1. Enter Pi IP address
2. Click "🔌 Connect"
3. Enable "🔍 Enable Detection"

## Features

### Laptop Client (laptop_client_gui.py)

#### Buttons:
- **💾 Save Single** - Save current frame with classification to `saved_images/`
- **📦 Save Batch** - Save batch of frames and show review dialog
- **🦾 Test Arm #1/#2** - Test individual robotic arm on Pi
- **🏠 Home All** - Return all arms to home position
- **💧 Drop Arm #1/#2** - Trigger drop action for specific class

#### Storage:
All images saved locally on laptop:
```
saved_images/
├── CLASS A/
│   ├── CLASS_A_20240115_101530_0.856.jpg
│   └── batch_20240115_101600/
│       ├── CLASS_A_0.856_001.jpg
│       └── CLASS_A_0.892_002.jpg
├── CLASS B/
└── CLASS C/
```

#### After Batch Save:
- Shows popup dialog with thumbnails
- Displays classification for each image (max 100)
- Shows confidence scores with color coding:
  - 🟢 Green: > 60% confidence
  - 🟡 Yellow: 40-60% confidence
  - 🔴 Red: < 40% confidence

### Raspberry Pi Server (pi_server_gui.py)

#### Functions:
- **Camera Streaming** - MJPEG stream to laptop client
- **Robotic Arm Control** - Servo control via PCA9685
- **Command Logging** - Shows all commands received from laptop

#### Hardware Support:
- Pi Camera Module (via picamera2)
- USB Webcams (via V4L2)
- PCA9685 Servo Controller (16 channels)
- GPIO pins for alternative control

## Peripheral Detection

### On Laptop:
Automatically detected at startup:
- System info (OS, Python version)
- GPU/CUDA availability
- Camera devices
- Storage space
- Network interfaces
- Installed packages

### On Raspberry Pi:
Run diagnostic tool:
```bash
python detect_peripherals.py
```

Shows:
- Connected cameras (Pi Camera, USB)
- I2C devices (PCA9685 servo controller)
- GPIO availability
- USB devices
- Network interfaces
- Python packages

## Communication Protocol

### Client → Server Commands:

```json
{
  "action": "test",
  "params": {"arm": 1}
}
```

Supported actions:
- `test` - Test arm movement
- `drop` - Drop action with class
- `home` - Return to home position

### Server → Client:

- **Video Stream**: MJPEG at `/video_feed`
- **Status**: JSON at `/status`
- **Response**: JSON with `status` and `result`

## Troubleshooting

### "Model not loaded"
- Check `buyo_best.pth` is in project root
- Check console for detailed error
- Model architecture must match checkpoint (5-layer EfficientNet)

### "No camera detected"
- On Pi: Enable camera in `raspi-config`
- On Pi: Install picamera2: `sudo apt install python3-picamera2`
- Test: `libcamera-hello` (Pi) or check Device Manager (Windows)

### "Connection failed"
- Verify Pi IP address (shown in Pi server GUI)
- Check firewall allows port 5000
- Ensure both devices on same network
- Test ping: `ping <pi_ip>`

### "I2C device not found"
- Enable I2C: `sudo raspi-config` → Interface Options → I2C
- Check wiring: `i2cdetect -y 1`
- PCA9685 should appear at 0x40 or 0x70

### "HTTP 404 on upload"
- This is expected! Uploads are now local only
- Images save to `saved_images/` on laptop
- Server only handles arm commands and camera

## Configuration

### Change Detection Threshold:
Edit `laptop_client_gui.py`:
```python
DETECTION_THRESHOLD = 0.60  # 60% minimum confidence
```

### Change Arm Mappings:
Edit `pi_server_gui.py`:
```python
ARM_CHANNELS = {
    1: 3,  # Arm 1 → Channel 3
    2: 2,  # Arm 2 → Channel 2
    # ... etc
}
```

### Change Camera Resolution:
```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

## File Structure

```
Buyo-Leaf-Classifier/
├── buyo_best.pth                    # ML model (required)
├── laptop_client/
│   ├── laptop_client_gui.py         # Main laptop GUI
│   └── requirements.txt
├── raspberry_pi_server/
│   ├── pi_server_gui.py             # Main Pi GUI
│   └── requirements.txt
├── saved_images/                    # Local image storage
│   ├── CLASS A/
│   ├── CLASS B/
│   └── ...
├── detect_peripherals.py            # Hardware detection
├── install_all_dependencies.bat     # Windows installer
├── test_installation.py             # Verify installation
└── QUICK_START.md                   # This file
```

## Next Steps

1. ✅ Install dependencies: `install_all_dependencies.bat`
2. ✅ Test installation: `python test_installation.py`
3. ✅ Run laptop client: `python laptop_client\laptop_client_gui.py`
4. 🚀 Connect Raspberry Pi (optional)
5. 📸 Start classifying buyo leaves!

## Support

- Check console output for detailed errors
- Run `detect_peripherals.py` to diagnose hardware
- Check `saved_images/` for output files
- Review logs in the Activity Log panel

---

**Ready to classify!** 🌿🔍
