# 📊 Buyo Leaf Classifier - Complete Project Overview

## 🎯 Project Purpose

Automated grading system for Buyo leaves using:
- **Computer Vision**: Deep learning classification (5 classes: A, B, C, D, E)
- **Remote GPU Processing**: Raspberry Pi camera + Laptop GPU inference
- **Robotic Automation**: Automatic arm triggering based on classification
- **User-Friendly GUI**: Full graphical interface on both devices

---

## 📁 Project Structure

```
BUYO LEAVES/
│
├── 🖥️ GUI SYSTEM (NEW - RECOMMENDED)
│   ├── raspberry_pi_server/
│   │   ├── pi_server_gui.py           ← Pi GUI application
│   │   ├── requirements.txt           ← Pi dependencies
│   │   └── run_server.sh              ← Quick launcher (Linux/Mac)
│   │
│   └── laptop_client/
│       ├── laptop_client_gui.py       ← Laptop GUI application
│       ├── requirements.txt           ← Laptop dependencies
│       ├── run_client.bat             ← Quick launcher (Windows)
│       └── run_client.sh              ← Quick launcher (Linux/Mac)
│
├── 📝 DOCUMENTATION
│   ├── README.md                      ← Main documentation
│   ├── QUICK_SETUP_GUIDE.md          ← Step-by-step setup
│   ├── GUI_SYSTEM_SUMMARY.md         ← Implementation details
│   └── PROJECT_OVERVIEW.md           ← This file
│
├── 🧠 MACHINE LEARNING
│   ├── buyo_best.pth                 ← Trained model (required!)
│   ├── train_ensemble.py             ← Training script
│   ├── train_convnext_optimized.py   ← Alternative trainer
│   └── split_model.py                ← Model utilities
│
├── 🔧 UTILITIES
│   ├── augment_data.py               ← Data augmentation
│   ├── diagnose_dataset.py           ← Dataset analysis
│   ├── find_duplicates.py            ← Find duplicate images
│   └── remove_duplicates.py          ← Clean dataset
│
├── 📷 LEGACY COMMAND-LINE SYSTEM
│   ├── laptop_classifier_server.py   ← Old laptop server
│   ├── raspberry_pi_camera_client.py ← Old Pi client
│   └── raspberry_pi_classifier_picamera2.py ← Standalone version
│
└── 📊 DATA (NOT IN GIT - LOCAL ONLY)
    ├── CLASS A/                      ← Training images
    ├── CLASS B/
    ├── CLASS C/
    ├── CLASS D/
    ├── CLASS E/
    └── dataset/
        ├── train/
        └── val/
```

---

## 🎨 System Options

### Option 1: GUI System (✅ RECOMMENDED)

**Use When:**
- You want visual feedback and monitoring
- You need manual control options
- You're setting up the system for the first time
- You want to test and debug easily

**Advantages:**
- ✅ Easy to use
- ✅ Visual status monitoring
- ✅ Quick IP address copy
- ✅ Manual override buttons
- ✅ Real-time logging
- ✅ Detection visualization

**Files:**
- `raspberry_pi_server/pi_server_gui.py`
- `laptop_client/laptop_client_gui.py`

---

### Option 2: Command-Line System

**Use When:**
- Deploying in production (headless)
- Running as system service
- Maximum performance needed
- No display available

**Advantages:**
- ✅ Lightweight
- ✅ Can run as service
- ✅ Automatic operation
- ✅ Lower resource usage

**Files:**
- `laptop_classifier_server.py`
- `raspberry_pi_camera_client.py`

---

## 🚀 Quick Start Paths

### Path A: First-Time Setup (GUI)

1. **Install on Raspberry Pi:**
   ```bash
   cd raspberry_pi_server
   pip3 install -r requirements.txt
   python3 pi_server_gui.py
   ```

2. **Install on Laptop:**
   ```bash
   cd laptop_client
   pip install -r requirements.txt
   python laptop_client_gui.py
   ```

3. **Connect & Test:**
   - Copy IP from Pi GUI
   - Enter IP in Laptop GUI
   - Click Connect
   - Test manual controls

**Time**: ~15 minutes

---

### Path B: Production Deployment (CLI)

1. **On Laptop:**
   ```bash
   python laptop_classifier_server.py
   ```
   Note the IP address

2. **On Raspberry Pi:**
   ```bash
   python3 raspberry_pi_camera_client.py <LAPTOP_IP>
   ```

3. **System runs automatically:**
   - Camera scans continuously
   - GPU classifies in real-time
   - Arm triggers on detection

**Time**: ~5 minutes

---

## 📊 System Flow

### GUI System Flow
```
┌─────────────────┐
│  Raspberry Pi   │
│  Server GUI     │
└────────┬────────┘
         │ Starts Flask Server
         │ (Background Thread)
         ↓
┌─────────────────┐
│ Camera Capture  │
│ MJPEG Streaming │
└────────┬────────┘
         │
         │ HTTP Stream
         │ (Port 5000)
         ↓
┌─────────────────┐
│  Laptop Client  │
│     GUI         │
└────────┬────────┘
         │ Display Video
         │ Optional: Run Detection
         ↓
┌─────────────────┐
│   ML Model      │
│ (GPU - PyTorch) │
└────────┬────────┘
         │ Classification Result
         ↓
┌─────────────────┐
│ Manual Control  │
│   or Auto       │
└────────┬────────┘
         │ Send Command
         │ (HTTP POST)
         ↓
┌─────────────────┐
│  Pi Server      │
│ execute_arm_    │
│   action()      │
└────────┬────────┘
         │ Servo Control
         ↓
┌─────────────────┐
│  Robotic Arm    │
│  Movement       │
└─────────────────┘
```

---

## 🎓 Training & Development

### Train New Model

**Requirements:**
- Dataset in `dataset/train/` and `dataset/val/`
- GPU with CUDA support (NVIDIA RTX 3050 or better)
- ~30-60 minutes training time

**Command:**
```bash
python train_ensemble.py
```

**Output:**
- `buyo_best.pth` - Best performing model
- `buyo_effnet.pth` - EfficientNetV2-S checkpoint
- `buyo_convnext.pth` - ConvNeXt checkpoint
- `buyo_resnet.pth` - ResNet50 checkpoint

### Current Model Performance
- **Accuracy**: ~68-78% (after duplicate removal)
- **Classes**: 5 (CLASS A, B, C, D, E)
- **Model**: EfficientNetV2-S / ConvNeXt / ResNet50
- **Image Size**: 224x224 or 256x256
- **Training**: Ensemble + Knowledge Distillation

---

## 🔧 Customization Guide

### 1. Add New Robotic Arm Command

**File**: `raspberry_pi_server/pi_server_gui.py`

```python
def execute_arm_action(action, params=None):
    # Add new case
    if action == 'your_action':
        # Your code here
        return "✓ Action completed"
```

**File**: `laptop_client/laptop_client_gui.py`

```python
# Add button in setup_ui()
buttons = [
    # ... existing buttons
    ("🎯 New Action", lambda: self.send_command('your_action'), row, col, span),
]
```

---

### 2. Replace Detection Model

**File**: `laptop_client/laptop_client_gui.py`

Replace `detect_target()` function:

```python
# For YOLO
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

def detect_target(frame):
    results = model(frame)
    # Process results
    return detection_dict
```

```python
# For custom PyTorch model
model = YourCustomModel()
model.load_state_dict(torch.load('your_model.pth'))
model.eval()

def detect_target(frame):
    # Your inference code
    return detection_dict
```

---

### 3. Change Network Settings

Both `pi_server_gui.py` and `laptop_client_gui.py`:

```python
# At top of file
SERVER_PORT = 5000          # Change if port conflict
VIDEO_WIDTH = 640           # Reduce for lower bandwidth
VIDEO_HEIGHT = 480
COMMAND_COOLDOWN = 2.0      # Adjust command rate
```

---

### 4. Adjust Detection Thresholds

**File**: `laptop_client/laptop_client_gui.py`

```python
DETECTION_THRESHOLD = 0.60    # Minimum confidence for detection
BUYO_CONFIDENCE_MIN = 0.40    # Minimum to consider as buyo
IMG_SIZE = 224                # Model input size
```

---

## 🐛 Common Issues & Solutions

### Issue: "Connection Refused"
**Cause**: Pi server not running or firewall blocking
**Solution**:
```bash
# On Pi, check server is running
ps aux | grep pi_server_gui

# Check firewall
sudo ufw status
sudo ufw allow 5000/tcp
```

---

### Issue: "Camera Not Found"
**Cause**: Camera not enabled or wrong driver
**Solution**:
```bash
# Enable camera
sudo raspi-config
# → Interface Options → Camera → Enable

# Test camera
libcamera-hello  # For Pi Camera Module
ffplay /dev/video0  # For USB webcam
```

---

### Issue: "Model Not Loading"
**Cause**: `buyo_best.pth` missing or PyTorch not installed
**Solution**:
```bash
# Check file exists
ls -lh buyo_best.pth

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Reinstall if needed
pip install torch torchvision
```

---

### Issue: "Arm Not Moving"
**Cause**: Libraries not installed or wrong I2C address
**Solution**:
```bash
# Install libraries
pip3 install RPi.GPIO adafruit-circuitpython-servokit

# Enable I2C
sudo raspi-config
# → Interface Options → I2C → Enable

# Detect I2C devices
sudo i2cdetect -y 1

# Check servo connections and power supply
```

---

### Issue: "Video Lag"
**Cause**: Slow network or high resolution
**Solution**:
- Reduce video resolution in `pi_server_gui.py`
- Lower JPEG quality (line ~60)
- Check Wi-Fi signal strength
- Move devices closer to router

---

## 📈 Performance Metrics

### Network Bandwidth
- **Video Stream**: ~1-3 Mbps (depends on resolution/quality)
- **Commands**: <1 KB per command
- **Recommended**: 5 Mbps+ Wi-Fi connection

### Processing Speed
- **GPU Inference**: ~30-60 FPS (NVIDIA RTX 3050)
- **CPU Inference**: ~5-10 FPS (not recommended)
- **Video Streaming**: 20-30 FPS typical

### Resource Usage
**Raspberry Pi:**
- CPU: ~30-50% (camera + Flask server)
- RAM: ~200-400 MB
- Storage: <100 MB (code only)

**Laptop:**
- CPU: ~10-20% (GUI + networking)
- GPU: ~30-60% (when detection enabled)
- RAM: ~500-800 MB (with model loaded)
- Storage: ~2-5 GB (PyTorch + model)

---

## 🔒 Security Notes

### Network Security
- System uses unencrypted HTTP (local network only)
- Do NOT expose to public internet without HTTPS
- Use firewall to restrict port 5000 to local network
- Consider VPN for remote access

### Best Practices
- ✅ Keep devices on isolated network segment
- ✅ Use strong Wi-Fi password
- ✅ Update software regularly
- ✅ Don't hardcode credentials in code
- ✅ Use environment variables for sensitive config

---

## 📦 Deployment Checklist

### Pre-Deployment
- [ ] Train and test model thoroughly
- [ ] Verify arm control code works
- [ ] Test on same hardware as production
- [ ] Document custom configurations
- [ ] Backup model and code

### Installation
- [ ] Install all dependencies
- [ ] Copy model file to correct location
- [ ] Configure network settings
- [ ] Test manual controls
- [ ] Verify auto-detection (if used)

### Production
- [ ] Set up automatic startup (systemd/cron)
- [ ] Configure logging to files
- [ ] Set up error monitoring
- [ ] Create backup procedures
- [ ] Document troubleshooting steps

---

## 🎓 Learning Resources

### Technologies Used
- **Python 3.7+**: Main programming language
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library
- **Flask**: HTTP server framework
- **Tkinter**: GUI framework
- **Pillow**: Image processing
- **NumPy**: Numerical computing

### Recommended Reading
1. PyTorch documentation: https://pytorch.org/docs/
2. OpenCV tutorials: https://docs.opencv.org/
3. Flask quickstart: https://flask.palletsprojects.com/
4. Tkinter reference: https://docs.python.org/3/library/tkinter.html
5. Raspberry Pi camera: https://www.raspberrypi.com/documentation/

---

## 🤝 Contributing

### Adding Features
1. Create feature branch
2. Test thoroughly on both devices
3. Update documentation
4. Submit with clear description

### Reporting Issues
Include:
- Device type (Pi/Laptop)
- OS and Python version
- Full error message
- Steps to reproduce
- Configuration used

---

## 📄 License

MIT License - Free to use and modify for any purpose.

---

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- EfficientNet, ConvNeXt, ResNet architectures
- Raspberry Pi Foundation
- OpenCV community
- Flask developers

---

## 📞 Support

For issues or questions:
1. Check `QUICK_SETUP_GUIDE.md`
2. Review troubleshooting in `README.md`
3. Check terminal output for error messages
4. Verify all dependencies installed
5. Test with simple examples first

---

**Last Updated**: November 19, 2026  
**Version**: 2.0 (GUI System)  
**Status**: ✅ Production Ready

---

**🎉 Happy Classifying!**
