# Buyo Leaf Classifier

Remote GPU Processing System - Laptop + Raspberry Pi

## 🏗️ System Architecture

```
Raspberry Pi Camera → Network → Laptop GPU → Classification → 
Trigger Command → Network → Raspberry Pi Robotic Arm
```

**Raspberry Pi**: Camera capture + Robotic arm control  
**Laptop**: GPU-accelerated ML inference

## 🚀 Quick Start

### 1️⃣ On Laptop (GPU Server)

```bash
# Install dependencies
pip install torch torchvision opencv-python pillow numpy flask

# Run server
python laptop_classifier_server.py
```

**Note the IP address shown** (e.g., `192.168.1.100`)

### 2️⃣ On Raspberry Pi (Camera Client)

```bash
# Install dependencies
sudo apt-get install -y python3-picamera2
pip3 install opencv-python numpy requests pillow

# Install robotic arm libraries
pip3 install adafruit-circuitpython-servokit RPi.GPIO

# Run client (replace IP with your laptop IP)
python3 raspberry_pi_camera_client.py 192.168.1.100
```

## ✨ Features

- 🎥 Real-time camera streaming from Pi to Laptop
- 🧠 GPU-accelerated inference on laptop
- 🤖 Automatic robotic arm triggering
- 📊 Live GUI on laptop showing classification
- 🔄 5-second cooldown between triggers
- 🎯 Buyo detection filter (rejects hands/objects)

## 📋 How It Works

1. **Raspberry Pi** captures frame with camera
2. Frame sent to **Laptop** over network
3. **Laptop GPU** runs classification model
4. If confident + buyo detected → send trigger command
5. **Raspberry Pi** receives command → triggers robotic arm

## ⚙️ Configuration

### Enable Robotic Arm on Pi
The arm libraries are auto-detected. Make sure they're installed:
```bash
pip3 install adafruit-circuitpython-servokit RPi.GPIO
```

### Change Arm Number
On Raspberry Pi, edit `raspberry_pi_camera_client.py`:
```python
CURRENT_ARM = 1  # Change to 1-9
```

### Adjust Trigger Threshold
On Laptop, edit `laptop_classifier_server.py`:
```python
CONF_THRESH = 0.40  # 40% confidence minimum
```

## 🔍 Troubleshooting

### Cannot connect to server
- Check both devices on same network
- Check firewall allows port 5000
- Verify laptop IP address is correct

### Camera not working
```bash
sudo raspi-config
# Enable camera in Interface Options
```

### Robotic arm not triggering
- Check `[ARM]` messages in Raspberry Pi terminal
- Verify ServoKit initialized successfully
- Check 5-second cooldown hasn't passed

## 📁 Files

- `laptop_classifier_server.py` - Laptop GPU server
- `raspberry_pi_camera_client.py` - Pi camera client
- `raspberry_pi_classifier_picamera2.py` - Standalone Pi version (no remote)
- `train_ensemble.py` - Model training script

## 🎓 Training

Train new model on laptop:
```bash
python train_ensemble.py
```

Model saved as `buyo_best.pth` - keep in same folder as server script.
