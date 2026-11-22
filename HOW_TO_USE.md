# How to Use laptop_client_gui.py

Simple step-by-step guide for operating the Buyo Leaf Classifier.

---

## Installation

### Install Python Dependencies

```bash
cd "C:\Users\aboni\Desktop\BUYO LEAVES"
pip install scipy scikit-image
```

If already installed, skip this step.

---

## Starting the Application

### Step 1: Start Raspberry Pi Server (if using Pi)

On Raspberry Pi:
```bash
python3 raspberry_pi_server/pi_server_gui.py
```

Note the IP address shown on screen (e.g., 192.168.1.100)

### Step 2: Start Laptop Client

On your laptop:
```bash
cd "C:\Users\aboni\Desktop\BUYO LEAVES"
python laptop_client/laptop_client_gui.py
```

Application window opens.

---

## Connecting to Raspberry Pi

### Method 1: Auto-Discovery (Recommended)

1. Click "Find Pi" button (top bar)
2. Wait 10-30 seconds for scan
3. Dialog shows found servers
4. Click "Connect" button
5. Status changes to "Connected"

### Method 2: Manual Connection

1. Enter Pi IP in text box (e.g., 192.168.1.100)
2. Click "Connect" button
3. Status changes to "Connected"

### Note:
- Both devices must be on same WiFi network
- Camera feed starts automatically after connection

---

## Understanding the Interface

### Left Side: Live Camera Feed
- Shows real-time video
- Detection boxes appear on Buyo leaves
- Analysis panel shows on detected leaves

### Right Side: Control Panel
- Top section: Classification buttons (green)
- Middle section: Remote controls (blue/purple/orange)
- Bottom section: Activity log

### Bottom Status Bar
- Model status (left)
- System status (middle)
- FPS counter (right)

---

## Main Features

### 1. Auto Grab & Drop (Main Function)

Purpose: Automatically grab Buyo leaf, detect class, and drop in correct bin

Steps:
1. Click "Auto Grab & Drop Arm #1" button
2. Click "Yes" to confirm
3. Claw opens - you have 3 seconds
4. Place Buyo leaf in claw
5. Claw closes automatically (grabs leaf)
6. Point camera at the leaf in claw
7. System detects class (e.g., CLASS C)
8. Arm moves down to bin
9. Claw opens (drops leaf)
10. Arm returns to home
11. Done

Important:
- Must place leaf within 3 seconds when claw opens
- Camera must see the leaf for detection
- Wait for "SUCCESS" message

### 2. Manual Claw Control

Open Claw:
- Click "Open Claw Arm #1"
- Claw opens to 90 degrees

Close Claw:
- Click "Close Claw Arm #1"
- Claw closes to 0 degrees (grab position)

Use for:
- Testing claw movement
- Manual leaf placement
- Adjusting grip

### 3. Classification (Local)

Classify Single Image:
1. Click "Classify Single Image"
2. Select image file
3. Results show in dialog with quality analysis

Classify Batch Images:
1. Click "Classify Batch Images"
2. Select multiple images
3. Results table shows all classifications
4. Can export to CSV

### 4. Carousel Control (Stepper Motor)

Purpose: Rotate sorting bins to different positions

Buttons:
- "Rotate to Class A" - moves to bin A
- "Rotate to Class B" - moves to bin B
- "Rotate to Class C" - moves to bin C
- "Rotate to Class D" - moves to bin D
- "Rotate to Class E" - moves to bin E

Use when:
- Setting up bins
- Testing carousel movement
- Manual bin positioning

### 5. Camera Servo Control (MG996R)

Purpose: Point camera at different arm positions

Buttons:
- "Camera to Arm #1" - points at arm 1 (0 degrees)
- "Camera to Arm #2" - points at arm 2 (2 degrees)
- "Camera Sweep 9 Arms" - automatically scans all 9 positions

Emergency:
- "STOP CAMERA SERVO" - immediately stops servo if stuck

Use when:
- Need to see specific arm position
- Scanning multiple leaves
- Camera stuck or rotating incorrectly

### 6. Peripheral Detection

Purpose: Check what hardware is connected

Steps:
1. Click "Detect Peripherals"
2. Dialog shows status of:
   - Camera module
   - Stepper motor
   - MG996R servo
   - Robotic arms (PCA9685 boards)

Use when:
- Troubleshooting hardware issues
- Verifying connections before starting
- Checking which devices are online

---

## Understanding Detection Display

When camera points at Buyo leaf, analysis panel appears:

```
BUYO LEAF ANALYSIS
Class: CLASS B              <- ML classification
Quality: Grade B            <- Quality assessment
⭐⭐⭐⭐ 82/100             <- Star rating and score
Green: 78.5%                <- Healthy green percentage
Uniformity: 85%             <- Color consistency
Defects: 3.2%               <- Total defect percentage
  Yellow: 2.1%              <- Aging/chlorosis
  Brown: 1.1%               <- Disease spots
█ #2d7a4f                   <- Dominant color
```

### Quality Grades

Grade A (⭐⭐⭐⭐⭐):
- Score 90+
- Defects less than 2%
- Premium quality, export grade

Grade B (⭐⭐⭐⭐):
- Score 75-89
- Defects 2-5%
- High quality, commercial grade

Grade C (⭐⭐⭐):
- Score 50-74
- Defects 5-15%
- Standard quality, local market

Grade D (⭐⭐):
- Score 25-49
- Defects 15-30%
- Below standard, processing

Grade E (⭐):
- Score below 25
- Defects over 30%
- Reject, discard

### Detection Colors

Green box/text:
- High confidence (over 70%)
- Low defects (under 5%)
- Good quality leaf

Orange box/text:
- Medium confidence (30-70%)
- Moderate defects (5-15%)
- Acceptable quality

Red box/text:
- Low confidence (under 30%)
- High defects (over 15%)
- Poor quality leaf

---

## Activity Log

Bottom right panel shows all actions:

Green messages = Success
- "Connected to Pi"
- "Buyo leaf detected: CLASS C"
- "Grab-and-drop complete"

Orange messages = Warnings
- "No Buyo leaf detected yet"
- "Cooldown active"

Red messages = Errors
- "Connection failed"
- "Grab-and-drop error"

Use log to:
- Track what system is doing
- Diagnose problems
- See detection results in real-time

---

## Common Workflows

### Workflow 1: Sort Single Leaf

1. Connect to Pi
2. Click "Auto Grab & Drop Arm #1"
3. Place leaf in claw (3 seconds)
4. Point camera at leaf
5. Watch it drop in correct bin
6. Done

Time: About 10 seconds per leaf

### Workflow 2: Test System

1. Connect to Pi
2. Click "Detect Peripherals"
3. Check all devices online
4. Click "Open Claw" - verify opens
5. Click "Close Claw" - verify closes
6. Click "Camera to Arm #1" - verify points correctly
7. Hold Buyo leaf to camera - verify detection
8. System ready to use

### Workflow 3: Classify Images

1. No Pi connection needed
2. Click "Classify Single Image" or "Classify Batch Images"
3. Select files
4. Review results
5. Export to CSV if needed

### Workflow 4: Manual Operation

1. Connect to Pi
2. Click "Open Claw"
3. Manually place leaf
4. Click "Close Claw"
5. Point camera at leaf (watch for detection)
6. Note detected class
7. Click appropriate "Rotate to Class X" button
8. Click "Drop Arm #1"
9. Click "Open Claw" to release
10. Click "Home All" to reset

---

## Timing and Cooldowns

Command Cooldown:
- 2 seconds between commands
- Prevents servo overload
- Countdown shows in status bar
- Just wait, then click again

Auto Grab & Drop Timing:
- Claw open: 3 seconds (for placement)
- Claw close: 1 second
- Detection: up to 10 seconds
- Move to bin: 1 second
- Drop: 0.8 seconds
- Return home: 0.5 seconds
- Total: About 8-10 seconds

Camera Detection:
- Runs every 0.5 seconds
- No manual trigger needed
- Always scanning when camera active

---

## Troubleshooting

### Can't Connect to Pi

Check:
1. Pi server is running
2. Both on same WiFi
3. Try "Find Pi" button
4. Check firewall allows port 5000

### No Camera Feed

Check:
1. Camera plugged in
2. Permissions granted
3. No other app using camera
4. Restart application

### No Detection

Check:
1. Camera feed working
2. Good lighting
3. Leaf is green and visible
4. Point camera directly at leaf
5. Model loaded (check status bar)

### Claw Not Working

Check:
1. Connected to Pi
2. Hardware connected
3. Try "Detect Peripherals"
4. Check Activity Log for errors

### Camera Servo Stuck

Solution:
1. Click "STOP CAMERA SERVO" immediately
2. Waits 1 second
3. Check if stopped
4. If still rotating, restart Pi

### Detection Wrong

Check:
1. Better lighting
2. Camera closer to leaf
3. White background preferred
4. Leaf takes up most of frame
5. Wait for high confidence (over 70%)

---

## Tips

1. Always test manually first
   - Open/close claw
   - Move camera
   - Check detection
   - Then use auto mode

2. Watch Activity Log
   - Shows what's happening
   - Helps diagnose issues
   - Green = good, red = problem

3. Use good lighting
   - Natural daylight best
   - Avoid harsh shadows
   - Not too dark or bright

4. Camera positioning
   - 20-30 cm from leaf
   - Perpendicular angle
   - Leaf fills most of frame
   - White/neutral background

5. Leaf placement
   - Center of claw
   - Not folded or bent
   - Visible to camera
   - Place quickly (3 seconds)

6. Quality over speed
   - Wait for good detection
   - High confidence better
   - Rush = wrong bin

7. Regular checks
   - "Detect Peripherals" daily
   - Test claw movement
   - Verify camera servo
   - Check Activity Log

---

## Important Notes

Model Accuracy:
- Current model has moderate accuracy
- Confidence under 50% is common
- For best results, retrain model with more data
- See MODEL_PERFORMANCE_ISSUE.md for details

Safety:
- Emergency stop available for camera servo
- 2-second cooldown prevents overload
- Always supervise auto operation
- Keep hands clear of moving parts

Limitations:
- Detects one leaf at a time
- Requires good lighting
- WiFi connection needed for Pi
- 3-second placement window is firm

Best Practices:
- Test before production use
- Calibrate regularly
- Monitor for errors
- Keep system clean
- Update software regularly

---

## Button Reference

Green Buttons (Classification):
- Classify Single Image
- Classify Batch Images
- Save Current Frame

Blue Buttons (Robotic Arms):
- Test Arm #1
- Test Arm #2
- Home All
- Drop Arm #1
- Drop Arm #2

Purple Buttons (Claw):
- Open Claw Arm #1
- Close Claw Arm #1
- Auto Grab & Drop Arm #1 (MAIN)

Purple Buttons (Carousel):
- Rotate to Class A/B/C/D/E

Orange Buttons (Camera):
- Camera to Arm #1
- Camera to Arm #2
- Camera Sweep 9 Arms

Red Button (Emergency):
- STOP CAMERA SERVO

Teal Button (System):
- Detect Peripherals

---

## Summary

Basic Operation:
1. Start application
2. Connect to Pi
3. Click "Auto Grab & Drop"
4. Place leaf (3 seconds)
5. Point camera at leaf
6. Watch it sort automatically

That's it. Simple as that.

---

For detailed information, see:
- OPERATION_GUIDE.md - Full guide with images
- BUYO_LEAF_IDENTIFICATION.md - Classification criteria
- INSTALL_ADVANCED_ANALYSIS.md - Feature details
- TROUBLESHOOTING.md - Problem solving

Version: 1.0
Last Updated: 2024
