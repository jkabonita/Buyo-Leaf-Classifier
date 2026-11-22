# 🌿 Buyo Leaf Classifier - Complete Operation Guide

## 📋 Table of Contents
1. [System Setup](#system-setup)
2. [Starting the System](#starting-the-system)
3. [Button Guide](#button-guide)
4. [Step-by-Step Operation](#step-by-step-operation)
5. [Troubleshooting](#troubleshooting)

---

## 🔧 System Setup

### **Equipment Needed:**
- ✅ Laptop (Windows) with GUI application
- ✅ Raspberry Pi with server running
- ✅ Both connected to **same WiFi network**
- ✅ Camera (laptop camera or Pi camera)
- ✅ Robotic arm with claw servo
- ✅ Stepper motor for carousel
- ✅ MG996R servo for camera rotation

### **Before You Start:**
1. Power on Raspberry Pi
2. Start Pi server: `python3 raspberry_pi_server/pi_server_gui.py`
3. Note the Pi's IP address shown on screen
4. Have a Buyo leaf ready for testing

---

## 🚀 Starting the System

### **Step 1: Launch Laptop Application**
```bash
cd "C:\Users\aboni\Desktop\BUYO LEAVES"
python laptop_client/laptop_client_gui.py
```

### **Step 2: Connect to Raspberry Pi**

**Option A: Auto-Discovery (Recommended)**
1. Click **"🔎 Find Pi"** button (top bar)
2. Wait for network scan (10-30 seconds)
3. Dialog shows found servers: `1. 10.125.22.31:5000`
4. Click **"✓ Connect"** button
5. Status changes to: **"Connected ✓"**

**Option B: Manual Connection**
1. Enter Pi IP in text box (e.g., `192.168.1.100`)
2. Click **"🔌 Connect"** button
3. Status changes to: **"Connected ✓"**

### **Step 3: Verify Camera**
- Local camera should start automatically
- You'll see live video feed on left side
- Detection status shows: **"Scanning for Buyo leaves..."**

---

## 🎛️ Button Guide

### **📊 Classification Controls (Green Buttons)**

| Button | Function | When to Use |
|--------|----------|-------------|
| **📷 Classify Single Image** | Upload and classify 1 image file | Testing with saved photos |
| **📦 Classify Batch Images** | Upload and classify multiple images | Analyzing many leaves at once |
| **💾 Save Current Frame** | Save current camera frame | Capturing good examples |

---

### **🦾 Remote Control - Robotic Arms (Blue Buttons)**

| Button | Function | What It Does |
|--------|----------|--------------|
| **🦾 Test Arm #1** | Test first arm | Moves arm to drop angle and back (quick test) |
| **🦾 Test Arm #2** | Test second arm | Same test for arm 2 |
| **🏠 Home All** | Reset all arms | Returns all 9 arms to home position |
| **💧 Drop Arm #1** | Drop from arm 1 | Triggers drop sequence for arm 1 |
| **💧 Drop Arm #2** | Drop from arm 2 | Triggers drop sequence for arm 2 |

---

### **✊ Claw Operations (Purple Buttons)**

| Button | Function | What It Does |
|--------|----------|--------------|
| **✋ Open Claw Arm #1** | Open claw manually | Opens claw to 90° (wide open) |
| **✊ Close Claw Arm #1** | Close claw manually | Closes claw to 0° (fully closed/grabs) |
| **🤖 Auto Grab & Drop Arm #1** | **MAIN BUTTON** | **Full automated sequence** (see below) |

---

### **⚙️ Carousel & Camera Controls**

#### **Stepper Motor (Purple Buttons)**
| Button | Function | What It Does |
|--------|----------|--------------|
| **⚙️ Rotate to Class A** | Move to bin A | Carousel rotates to Class A position |
| **⚙️ Rotate to Class B** | Move to bin B | Carousel rotates to Class B position |
| **⚙️ Rotate to Class C** | Move to bin C | Carousel rotates to Class C position |
| **⚙️ Rotate to Class D** | Move to bin D | Carousel rotates to Class D position |
| **⚙️ Rotate to Class E** | Move to bin E | Carousel rotates to Class E position |

#### **Camera Servo (Orange Buttons)**
| Button | Function | What It Does |
|--------|----------|--------------|
| **📹 Camera to Arm #1** | Point camera at arm 1 | Rotates camera to 0° (arm 1 position) |
| **📹 Camera to Arm #2** | Point camera at arm 2 | Rotates camera to 2° (arm 2 position) |
| **📹 Camera Sweep 9 Arms** | Scan all positions | Sweeps through all 9 arm positions |
| **🛑 STOP CAMERA SERVO** | **EMERGENCY STOP** | Immediately stops camera servo (if stuck) |

---

### **🔌 System Controls (Teal Button)**

| Button | Function | What It Does |
|--------|----------|--------------|
| **🔌 Detect Peripherals** | Check hardware | Shows status of all connected devices |

---

## 📖 Step-by-Step Operation

### **🎯 MAIN WORKFLOW: Auto Grab & Drop**

This is the **primary button** for sorting Buyo leaves!

#### **Step 1: Prepare**
```
✓ Pi connected
✓ Camera working
✓ Robotic arm at home position
✓ Buyo leaf ready in hand
```

#### **Step 2: Click "🤖 Auto Grab & Drop Arm #1"**

A confirmation dialog appears:
```
Start automatic grab-and-drop sequence?

WORKFLOW:
1. Claw opens (3 seconds)
2. >> USER: Place Buyo leaf in claw <<
3. Claw closes (grabs leaf)
4. Camera detects leaf class
5. Arm moves down to bin
6. Claw opens (drops into bin)
7. Arm returns home

Ready to start?
```

Click **"Yes"**

#### **Step 3: Place Leaf (3 SECONDS WINDOW)**

**Status Dialog Shows:**
```
🤖 Grab & Drop Sequence
Step 1: Opening claw...
Please place Buyo leaf in claw!

⏰ 3 seconds to place leaf...
⏰ 2 seconds to place leaf...
⏰ 1 seconds to place leaf...
```

**👉 ACTION: Quickly place Buyo leaf in the open claw!**

#### **Step 4: Claw Closes (Automatic)**

```
Step 2: Closing claw (grabbing leaf)...
```

The claw automatically closes and grabs the leaf. No action needed.

#### **Step 5: Camera Detection (Automatic)**

```
Step 3: Detecting Buyo leaf class...
Point camera at the leaf!
```

**👉 ACTION: Point the camera at the leaf in the claw**

The system will:
- Analyze the leaf
- Detect the class (A, B, C, D, or E)
- Show confidence percentage

Wait up to 10 seconds for detection.

#### **Step 6: Drop into Bin (Automatic)**

```
Step 4: Detected CLASS C (85.3%)
Moving to bin...

Step 5: Dropping into CLASS C bin...
```

The arm:
- Moves down to bin position
- Opens claw
- Drops leaf into correct bin
- Returns to home

#### **Step 7: Complete! ✅**

```
✅ SUCCESS!
Leaf dropped into CLASS C bin
Arm returned to home position

Sequence complete!
```

Dialog closes automatically after 3 seconds.

---

## 🧪 Testing Individual Components

### **Test 1: Claw Movement**
1. Click **"✋ Open Claw Arm #1"** → Claw opens wide
2. Manually place a leaf
3. Click **"✊ Close Claw Arm #1"** → Claw closes/grabs
4. Check if leaf is held securely

### **Test 2: Camera Detection**
1. Hold a Buyo leaf in front of camera
2. Watch the **"Auto Detection: ACTIVE"** status
3. When detected, you'll see:
   ```
   ✓ Detected: CLASS C (47.4%)
   ```
4. Activity log shows: `🍃 Buyo leaf detected: CLASS C`

### **Test 3: Carousel Rotation**
1. Click **"⚙️ Rotate to Class A"** → Carousel moves
2. Click **"⚙️ Rotate to Class B"** → Moves to next position
3. Verify bins align correctly with arms

### **Test 4: Camera Servo**
1. Click **"📹 Camera to Arm #1"** → Camera points at arm 1
2. Click **"📹 Camera to Arm #2"** → Camera points at arm 2
3. Verify camera rotates smoothly (only 2° difference)

---

## 🔥 Quick Start for First-Time Users

### **Fastest Way to Test (5 Steps):**

1. **Start laptop app** → `python laptop_client/laptop_client_gui.py`

2. **Connect to Pi** → Click **"🔎 Find Pi"** → Click **"✓ Connect"**

3. **Test claw** → Click **"✋ Open Claw"** → Click **"✊ Close Claw"**

4. **Get a Buyo leaf** → Hold it in your hand

5. **Run auto sequence** → Click **"🤖 Auto Grab & Drop Arm #1"**
   - When claw opens, **place leaf quickly (3 seconds)**
   - Point camera at leaf
   - Watch it drop into correct bin! 🎉

---

## ⚠️ Troubleshooting

### **Problem: Can't Connect to Pi**
**Solution:**
1. Check Pi server is running: `python3 raspberry_pi_server/pi_server_gui.py`
2. Verify both on same WiFi
3. Click **"🔎 Find Pi"** for auto-discovery
4. Check firewall allows port 5000

### **Problem: Camera Servo Keeps Rotating**
**Solution:**
1. **IMMEDIATELY** click **"🛑 STOP CAMERA SERVO"**
2. This kills the PWM signal
3. Servo should stop within 1 second

### **Problem: No Detection Happening**
**Solution:**
1. Check camera feed is working (shows live video)
2. Point camera directly at leaf
3. Ensure good lighting
4. Leaf should be green and visible
5. Check Activity Log for errors

### **Problem: Claw Doesn't Grab Leaf**
**Solution:**
1. Adjust `CLAW_OPEN_ANGLE` in pi_server_gui.py (currently 90°)
2. Adjust `CLAW_CLOSE_ANGLE` (currently 0°)
3. Test with **"✋ Open"** and **"✊ Close"** buttons first
4. Verify servo is connected to correct channel

### **Problem: Arm Drops in Wrong Bin**
**Solution:**
1. Check detection confidence (should be >40%)
2. Improve lighting for better detection
3. Point camera closer to leaf
4. Retrain model if accuracy is poor

### **Problem: "Cooldown active" Message**
**Solution:**
- Wait 2 seconds between commands
- This prevents servo overload
- Countdown shows in status bar

---

## 📊 Understanding the Interface

### **Top Bar:**
- **Raspberry Pi IP**: Enter manually or use auto-discovery
- **🔌 Connect**: Connect to Pi server
- **🔍 Test**: Test connection
- **🔎 Find Pi**: Auto-discover Pi on network
- **📷 LOCAL CAMERA**: Start/stop local camera
- **🧪 Test**: Test camera feed

### **Left Panel: Live Camera Feed**
- Shows real-time video
- Detection boxes appear on Buyo leaves
- Class labels shown in center
- Green box = High confidence (>70%)
- Orange box = Medium confidence (30-70%)

### **Right Panel: Manual Controls**
- All buttons organized by function
- Color-coded for easy identification
- Activity log at bottom shows all actions

### **Bottom Status Bar:**
- **Model**: Shows if ML model is loaded
- **Status**: Current system status
- **FPS**: Camera frame rate

---

## 🎓 Pro Tips

1. **Always test manually first**
   - Use individual buttons before auto sequence
   - Verify claw opens/closes properly
   - Check camera detection works

2. **Watch the Activity Log**
   - Shows all actions in real-time
   - Green = Success
   - Orange = Warning
   - Red = Error

3. **Use Peripheral Detection**
   - Click **"🔌 Detect Peripherals"** to check hardware
   - Verify all devices are online before starting

4. **Camera Detection Tips**
   - Good lighting helps accuracy
   - Point camera directly at leaf
   - Wait for green detection box
   - Confidence >70% is ideal

5. **Emergency Controls**
   - **🛑 STOP CAMERA SERVO** for stuck servo
   - Close application to stop all operations
   - Power cycle Pi if needed

---

## 📞 Need Help?

**Check These First:**
1. Activity Log (bottom right panel)
2. Status bar messages (bottom)
3. This operation guide

**Common Success Messages:**
- `✓ Connected to Pi`
- `✓ Model loaded successfully`
- `🍃 Buyo leaf detected: CLASS C`
- `✓ Grab-and-drop complete: CLASS C`

**System is Working When:**
- ✅ Status shows "Connected ✓"
- ✅ Camera feed is live
- ✅ Detection boxes appear on leaves
- ✅ Activity log shows green messages

---

## 🎉 Success Checklist

After first successful run, you should see:

```
[Activity Log]
[00:00:00] ✓ Connected to 192.168.1.100:5000
[00:00:05] 🤖 Starting grab-and-drop sequence...
[00:00:08] 🍃 Buyo leaf detected: CLASS C (85.3%)
[00:00:12] ✅ Grab-and-drop complete: CLASS C
```

**Congratulations! 🎉 Your system is working!**

---

## 📝 Quick Reference Card

### **Essential Buttons Only:**
1. **🔎 Find Pi** - Connect to Raspberry Pi
2. **🤖 Auto Grab & Drop** - Main sorting button
3. **✋ Open Claw** - Manual claw control
4. **✊ Close Claw** - Manual claw control
5. **🛑 STOP CAMERA** - Emergency stop

### **Essential Steps:**
1. Connect → 2. Place leaf in claw → 3. Point camera → 4. Watch it drop!

---

**Last Updated:** 2024
**Version:** 1.0
**Support:** Check Activity Log and TROUBLESHOOTING.md
