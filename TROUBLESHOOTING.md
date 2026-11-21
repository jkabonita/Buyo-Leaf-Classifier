# 🔧 Troubleshooting Guide - Buyo Leaf Classifier

## Common Issues and Solutions

### ❌ Issue: `ModuleNotFoundError: No module named 'requests'`

**Cause:** Required Python packages are not installed in your environment.

**Solution:**

1. **Install all dependencies at once:**
   ```bash
   pip install -r laptop_client/requirements.txt
   ```

2. **Or install individually:**
   ```bash
   pip install requests pillow opencv-python numpy
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Check dependencies:**
   ```bash
   python laptop_client/check_dependencies.py
   ```

---

### ❌ Issue: Dependencies keep getting "uninstalled"

**Cause:** You might have multiple Python installations or virtual environments.

**Solution:**

1. **Check which Python you're using:**
   ```bash
   python --version
   where python
   ```

2. **Install dependencies with the specific Python path:**
   ```bash
   C:/Users/aboni/AppData/Local/Programs/Python/Python312/python.exe -m pip install -r laptop_client/requirements.txt
   ```

3. **Use the batch file to run the client:**
   ```bash
   laptop_client/run_laptop_client.bat
   ```

---

### ❌ Issue: `CUDA not available` or GPU not detected

**Cause:** PyTorch was installed without CUDA support, or NVIDIA drivers are missing.

**Solution:**

1. **Check CUDA availability:**
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

2. **Reinstall PyTorch with CUDA 11.8:**
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Update NVIDIA drivers:**
   - Download from: https://www.nvidia.com/Download/index.aspx
   - Or use GeForce Experience

4. **Verify CUDA installation:**
   ```bash
   nvidia-smi
   ```

---

### ❌ Issue: Laptop client GUI won't start

**Cause:** Missing dependencies, wrong Python version, or tkinter not installed.

**Solution:**

1. **Run dependency checker:**
   ```bash
   python laptop_client/check_dependencies.py
   ```

2. **Check Python version (must be 3.7+):**
   ```bash
   python --version
   ```

3. **If tkinter is missing (rare on Windows):**
   - Reinstall Python with "tcl/tk and IDLE" option checked

4. **Use the batch launcher:**
   ```bash
   laptop_client\run_laptop_client.bat
   ```

---

### ❌ Issue: Model file not found (`buyo_best.pth`)

**Cause:** Model file is missing or in wrong location.

**Solution:**

1. **Check if model exists:**
   ```bash
   dir buyo_best.pth
   ```

2. **Model should be in project root:**
   ```
   BUYO LEAVES/
   ├── buyo_best.pth          ← Should be here
   ├── laptop_client/
   │   └── laptop_client_gui.py
   ```

3. **Train a model if missing:**
   ```bash
   python train_ensemble.py
   ```

4. **Or download pre-trained model** (if available from repository)

---

### ❌ Issue: Cannot connect to Raspberry Pi

**Cause:** Wrong IP address, network issues, or Pi server not running.

**Solution:**

1. **Verify Pi IP address:**
   - On Pi, run: `hostname -I`
   - Or check Pi server GUI for displayed IP

2. **Test network connectivity:**
   ```bash
   ping 192.168.1.XXX
   ```

3. **Check if Pi server is running:**
   - Open browser and visit: `http://192.168.1.XXX:5000/status`
   - Should see JSON response

4. **Ensure both devices on same network:**
   - Check Wi-Fi SSID on both devices
   - Disable VPN if active

5. **Check firewall settings:**
   - Allow port 5000 on Pi
   - Allow Python through Windows Firewall

---

### ❌ Issue: Video feed is laggy or freezing

**Cause:** Network bandwidth, camera issues, or processing bottleneck.

**Solution:**

1. **Reduce video resolution in Pi server:**
   Edit `raspberry_pi_server/pi_server_gui.py`:
   ```python
   VIDEO_WIDTH = 320   # Reduce from 640
   VIDEO_HEIGHT = 240  # Reduce from 480
   ```

2. **Lower JPEG quality:**
   ```python
   encode_param = [cv2.IMWRITE_JPEG_QUALITY, 60]  # Reduce from 80
   ```

3. **Use 5GHz Wi-Fi instead of 2.4GHz**

4. **Move devices closer to router**

5. **Disable other bandwidth-heavy applications**

---

### ❌ Issue: Low classification accuracy

**Cause:** Model not properly trained, poor dataset quality, or wrong model loaded.

**Solution:**

1. **Check dataset quality:**
   ```bash
   python diagnose_dataset.py
   ```

2. **Remove duplicates:**
   ```bash
   python find_duplicates.py
   python remove_duplicates.py
   ```

3. **Augment dataset:**
   ```bash
   python augment_data.py
   ```

4. **Retrain with ensemble:**
   ```bash
   python train_ensemble.py
   ```

5. **Verify correct model is loaded:**
   ```bash
   python check_model.py
   ```

---

### ❌ Issue: Robotic arm not responding

**Cause:** Hardware not connected, libraries missing, or code not configured.

**Solution:**

1. **Check servo libraries:**
   ```bash
   pip3 list | grep servo
   pip3 install adafruit-circuitpython-servokit
   ```

2. **Enable I2C on Raspberry Pi:**
   ```bash
   sudo raspi-config
   # Interface Options → I2C → Enable
   ```

3. **Detect I2C devices:**
   ```bash
   sudo i2cdetect -y 1
   ```

4. **Check power supply:**
   - Servos need external 5-6V power
   - DO NOT power servos from Pi GPIO pins

5. **Configure servo code:**
   - Edit `execute_arm_action()` in `pi_server_gui.py`
   - Set correct GPIO pins or servo channels

---

### ❌ Issue: High memory usage or crashes

**Cause:** Model too large for available RAM, memory leak, or batch size too high.

**Solution:**

1. **Monitor memory usage:**
   ```bash
   # Windows Task Manager
   taskmgr
   
   # Pi
   free -h
   ```

2. **Reduce batch size in training:**
   Edit training script:
   ```python
   BATCH_SIZE = 8  # Reduce from 16 or 32
   ```

3. **Enable gradient accumulation:**
   ```python
   GRADIENT_ACCUMULATION_STEPS = 4
   ```

4. **Use mixed precision training:**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

5. **Close other applications**

6. **Restart the application**

---

### ❌ Issue: Permission denied errors

**Cause:** File/folder permissions, or camera access denied.

**Solution:**

1. **Run with appropriate permissions:**
   ```bash
   # Windows: Run as Administrator (if needed)
   # Linux/Pi: Use sudo for GPIO/camera access
   sudo python3 pi_server_gui.py
   ```

2. **Check camera permissions:**
   ```bash
   # Pi
   sudo usermod -a -G video $USER
   # Logout and login again
   ```

3. **Fix file permissions:**
   ```bash
   chmod +x script_name.py
   ```

---

## 🛠️ Diagnostic Tools

### Check All Dependencies
```bash
python laptop_client/check_dependencies.py
```

### Verify GPU and CUDA
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Test Camera (Pi)
```bash
# Pi Camera Module
libcamera-hello

# USB Webcam
ffplay /dev/video0
```

### Test Model Loading
```bash
python check_model.py
```

### Network Connectivity Test
```bash
# Test Pi server from laptop
curl http://192.168.1.XXX:5000/status

# Or in Python
python -c "import requests; print(requests.get('http://192.168.1.XXX:5000/status').json())"
```

---

## 📞 Getting More Help

If none of these solutions work:

1. **Check documentation:**
   - [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
   - [QUICK_SETUP_GUIDE.md](QUICK_SETUP_GUIDE.md)
   - [README.md](README.md)

2. **Enable debug logging:**
   - Add `print()` statements to trace execution
   - Check console output for error messages

3. **Search GitHub Issues:**
   - Look for similar problems in project issues

4. **Create a new issue:**
   - Include error messages
   - Specify OS, Python version, and hardware
   - Describe steps to reproduce

---

## ✅ Quick Checklist

Before running the system, ensure:

- [ ] Python 3.7+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] PyTorch with CUDA support (for GPU)
- [ ] Model file (`buyo_best.pth`) in project root
- [ ] Both devices on same Wi-Fi network
- [ ] Pi server running and IP address noted
- [ ] Camera connected and working
- [ ] Firewall allows port 5000

---

**Last Updated:** November 21, 2026
