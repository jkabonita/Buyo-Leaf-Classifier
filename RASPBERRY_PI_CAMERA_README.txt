================================================================================
RASPBERRY PI CAMERA 3 (imx708) - QUICK REFERENCE
================================================================================

CURRENT ISSUE:
--------------
Camera shows BLACK SCREEN in GUI because picamera2 cannot be imported in venv.

Root cause: Virtual environment was created without --system-site-packages flag,
so it cannot access the system-installed picamera2 library.


QUICK FIX (1 COMMAND):
----------------------
On your Raspberry Pi, run:

    bash fix_picamera2.sh

This will:
✓ Verify picamera2 is installed
✓ Recreate venv with correct settings
✓ Reinstall all packages
✓ Test camera detection
✓ Show which camera is detected

Takes ~2 minutes.


VERIFY IT'S FIXED:
------------------
After running fix_picamera2.sh, test with:

    source venv/bin/activate
    python3 test_camera_detection.py

You should see:
  ✓ System site packages: ENABLED
  ✓ SUCCESS: picamera2 can be imported!
  ✓ Found 1 camera(s):
     Camera 0: Model: imx708


RUN THE SERVER:
---------------
    source venv/bin/activate
    python3 raspberry_pi_server/pi_server_gui.py

Expected terminal output:
  ✓ Picamera2 library found
  📸 Found 1 Pi Camera(s)
     Camera 0: {'Model': 'imx708', ...}
  ✅ SUCCESS! Camera initialized: (480, 640, 3)
  📷 Camera type: Pi Camera (imx708)

Expected GUI:
  - Preview shows LIVE VIDEO (not black)
  - Camera status: "Camera: Pi Camera (imx708)" in GREEN


ALTERNATIVE COMMANDS:
---------------------
If you prefer to do it manually instead of using the script:

# Delete old venv
rm -rf venv

# Install picamera2 (if not already)
sudo apt install python3-picamera2

# Create NEW venv with system packages access
python3 -m venv venv --system-site-packages

# Activate and install packages
source venv/bin/activate
pip install --upgrade pip
pip install -r raspberry_pi_server/requirements.txt

# Test
python3 test_camera_detection.py


TROUBLESHOOTING:
----------------

Problem: Camera still shows black after fix
Solution: 
  1. Check camera cable is firmly connected
  2. Reboot Raspberry Pi: sudo reboot
  3. Test camera: libcamera-hello
  4. Share terminal output from test_camera_detection.py

Problem: "No cameras detected"
Solution:
  1. Enable camera in raspi-config:
     sudo raspi-config → Interface Options → Camera → Enable
  2. Reboot: sudo reboot
  3. Test: libcamera-hello

Problem: GUI won't close
Solution: 
  - Press Ctrl+C in terminal
  - Or use Stop Server button first, then close window
  - This has been fixed in latest code


FILES CREATED:
--------------
✓ fix_picamera2.sh - Automated fix script
✓ test_camera_detection.py - Diagnostic tool
✓ CAMERA_FIX_INSTRUCTIONS.txt - Detailed instructions
✓ raspberry_pi_quick_setup.sh - Updated (fixed for future setups)
✓ raspberry_pi_server/pi_server_gui.py - Improved (better error messages)


WHAT WAS CHANGED:
-----------------
1. raspberry_pi_quick_setup.sh
   - Changed: python3 -m venv venv --system-site-packages
   - Before it was: python3 -m venv venv (isolated)

2. raspberry_pi_server/pi_server_gui.py
   - Added comprehensive logging for camera detection
   - Added proper window close handler
   - Better error messages showing solutions

3. Created diagnostic and fix tools


NEXT STEPS:
-----------
1. Run: bash fix_picamera2.sh
2. Run: python3 test_camera_detection.py (to verify)
3. Run: python3 raspberry_pi_server/pi_server_gui.py
4. Verify camera preview shows live video
5. Connect from laptop client


================================================================================
