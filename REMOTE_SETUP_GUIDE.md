# 🚀 Remote GPU Classification Setup Guide

Complete step-by-step guide to run GPU classification on laptop and trigger robotic arm on Raspberry Pi.

---

## 📋 What You Need

### On Laptop:
- Python 3.8+
- NVIDIA GPU (for faster inference)
- `laptop_classifier_server.py`
- `buyo_best.pth` model file

### On Raspberry Pi:
- Pi Camera Module 3 (IMX708) connected
- `raspberry_pi_camera_client.py`
- Robotic arm hardware (optional, works in simulation mode too)

---

## 🔧 Step 1: Setup Laptop Server

### 1.1 Install Dependencies
```bash
# On your laptop (Windows/Mac/Linux)
pip install torch torchvision opencv-python pillow numpy flask
```

### 1.2 Place Files
```
your_laptop_folder/
├── laptop_classifier_server.py
└── buyo_best.pth  ← Must be in same folder!
```

### 1.3 Run Server
```bash
cd your_laptop_folder
python laptop_classifier_server.py
```

**Expected Output:**
```
================================================================
  Buyo Classifier Server - Laptop GPU
================================================================
  Device: cuda
  Model: buyo_best.pth
  Backbone: efficientnet_v2_s
  Classes: 5
================================================================

  🚀 SERVER READY
================================================================
  Laptop IP: 192.168.1.100  ← COPY THIS IP!
  Port: 5000
  Device: cuda
================================================================

  On Raspberry Pi, run:
    python3 raspberry_pi_camera_client.py 192.168.1.100

================================================================
```

**⚠️ IMPORTANT: Copy your laptop IP address (e.g., `192.168.1.100`)**

---

## 🔧 Step 2: Setup Raspberry Pi Client

### 2.1 Install Dependencies
```bash
# On Raspberry Pi terminal
sudo apt-get update
sudo apt-get install -y python3-picamera2
pip3 install opencv-python numpy requests pillow
```

### 2.2 Install Robotic Arm Libraries (Optional)
```bash
# Only if you have robotic arm hardware
pip3 install adafruit-circuitpython-servokit RPi.GPIO
```

### 2.3 Enable Camera
```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
# Reboot if prompted
```

### 2.4 Run Client
```bash
# Replace 192.168.1.100 with YOUR laptop IP from Step 1.3
python3 raspberry_pi_camera_client.py 192.168.1.100
```

**Expected Output:**
```
================================================================
  Buyo Camera Client - Raspberry Pi
================================================================
  Server: 192.168.1.100:5000
  🤖 Robotic arm: ENABLED
================================================================

[INIT] Testing connection to laptop...
[INIT] ✓ Connected to buyo_best.pth on cuda

[INIT] Starting camera...
[INIT] ✓ Camera ready

================================================================
  🚀 CLIENT RUNNING
================================================================
  Capturing frames → Laptop GPU → Arm trigger
  Press Ctrl+C to stop
================================================================

[✓] CLASS A (87%) | FPS: 18.3 | Triggers: 0
[✗] NOT BUYO | FPS: 19.1 | Triggers: 0
[✓] CLASS B (56%) | FPS: 18.7 | Triggers: 1
[ARM] ⬇️  Arm #1 dropping CLASS B
[ARM] ✓ Drop complete
```

---

## ✅ How It Works

### Signal Flow:
1. **Pi Camera** captures frame
2. **Pi sends** frame to laptop via network (HTTP POST)
3. **Laptop GPU** classifies the frame
4. **Laptop determines** if should trigger: `confident (≥40%) AND is_buyo`
5. **Laptop sends back** response: `{'trigger_arm': True, 'class': 'CLASS A', ...}`
6. **Pi receives** the response
7. **Pi triggers** robotic arm if `trigger_arm == True`

### Code Snippet (Already Implemented):
```python
# In raspberry_pi_camera_client.py (lines 140-148)
should_trigger = result['trigger_arm']  # ← Signal from laptop

if should_trigger:
    current_time = time.time()
    if current_time - last_trigger_time >= TRIGGER_COOLDOWN:
        trigger_arm_drop(CURRENT_ARM, pred_class)  # ← Triggers arm!
        trigger_count += 1
```

---

## ⚙️ Configuration

### Change Arm Number
**On Raspberry Pi**, edit `raspberry_pi_camera_client.py`:
```python
CURRENT_ARM = 1  # Change to 1-9 (line 75)
```

### Change Confidence Threshold
**On Laptop**, edit `laptop_classifier_server.py`:
```python
CONF_THRESH = 0.40  # 40% minimum (line 112)
```

### Change Trigger Cooldown
**On Raspberry Pi**, edit `raspberry_pi_camera_client.py`:
```python
TRIGGER_COOLDOWN = 5.0  # seconds (line 77)
```

### Disable Robotic Arm (Simulation Mode)
If libraries not installed, it automatically runs in simulation mode:
```
[ARM] Simulated: Arm #1 → CLASS A
```

---

## 🔍 Troubleshooting

### "Cannot connect to server"
**Problem:** Pi can't reach laptop

**Solutions:**
1. Check both devices on **same WiFi network**
2. Check laptop firewall allows port 5000:
   ```bash
   # Windows: Allow Python through firewall
   # Linux: sudo ufw allow 5000
   ```
3. Verify laptop IP is correct (not `127.0.0.1`)
4. Ping laptop from Pi:
   ```bash
   ping 192.168.1.100
   ```

### "No webcam available"
**Problem:** Camera not detected

**Solutions:**
1. Enable camera in `raspi-config`
2. Test camera:
   ```bash
   libcamera-still -o test.jpg
   ```
3. Check connection cable

### "Robotic arm not triggering"
**Problem:** Arm doesn't move

**Solutions:**
1. Check arm libraries installed:
   ```bash
   pip3 list | grep -i servo
   pip3 list | grep -i gpio
   ```
2. Check terminal output for `[ARM]` messages
3. Verify 5-second cooldown hasn't blocked trigger
4. Check servo wiring to correct channels

### Laptop shows "Waiting for connection"
**Problem:** No frames arriving from Pi

**Solutions:**
1. Make sure Pi client is running
2. Check network connectivity
3. Verify laptop IP is correct in Pi command

---

## 📊 Monitoring

### On Laptop:
- GUI window shows live camera feed
- Terminal shows: `[✓] CLASS A (87.3%) | Trigger: True`

### On Raspberry Pi:
- Terminal shows: `[✓] CLASS A (87%) | FPS: 18.3 | Triggers: 1`
- Arm trigger: `[ARM] ⬇️  Arm #1 dropping CLASS A`

---

## 🛑 Stopping

### Stop Raspberry Pi Client:
Press `Ctrl+C` in terminal

### Stop Laptop Server:
Press `Ctrl+C` in terminal or close GUI window

---

## 🎯 Quick Reference Commands

### Laptop:
```bash
python laptop_classifier_server.py
```

### Raspberry Pi:
```bash
python3 raspberry_pi_camera_client.py 192.168.1.100
```

### Test Connection (on Pi):
```bash
curl http://192.168.1.100:5000/status
```

Expected response:
```json
{"status": "running", "device": "cuda", "model": "buyo_best.pth"}
```

---

## 📝 File Locations

### Laptop:
- Server script: `laptop_classifier_server.py`
- Model file: `buyo_best.pth` (80MB - transfer manually)

### Raspberry Pi:
- Client script: `raspberry_pi_camera_client.py`
- No model needed! (All processing on laptop)

---

## 💡 Tips

1. **Keep laptop plugged in** - GPU uses more power
2. **Use wired ethernet** for more stable connection (optional)
3. **Test in simulation mode first** before connecting arm
4. **Monitor FPS** - should be 15-25 FPS for good performance
5. **Check cooldown** if arm not triggering repeatedly

---

## 🎓 Advanced

### Run Server on Different Port:
Edit `laptop_classifier_server.py` line 522:
```python
app.run(host='0.0.0.0', port=5001, debug=False)  # Change 5000 to 5001
```

Then on Pi:
```bash
python3 raspberry_pi_camera_client.py 192.168.1.100:5001
```

### Multiple Arms:
Change arm number on-the-fly by editing client script or add keyboard control.

---

Need help? Check terminal output for error messages!
