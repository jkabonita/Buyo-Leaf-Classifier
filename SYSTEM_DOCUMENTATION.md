# BUYO LEAF CLASSIFICATION SYSTEM DOCUMENTATION

## OVERVIEW

This is an automated Buyo (betel) leaf classification and sorting system using computer vision and machine learning. The system classifies leaves into 5 quality grades (A through E) and controls a robotic arm for automated sorting.

## MODEL INFORMATION

Model File: buyo_best.pth
Architecture: ResNet-50 (Deep Convolutional Neural Network)
Framework: PyTorch
Input Size: 224x224 pixels RGB images
Output: 5 classes (CLASS A, CLASS B, CLASS C, CLASS D, CLASS E)

## DATASET INFORMATION

Total Training Images: 875 images

Breakdown by Class:
- CLASS A: 225 images (25.7%)
- CLASS B: 170 images (19.4%)
- CLASS C: 169 images (19.3%)
- CLASS D: 176 images (20.1%)
- CLASS E: 135 images (15.4%)

Image Formats: JPG, JPEG, PNG
Image Source: Real photos of Buyo leaves taken with smartphone camera
Training Date: May 2026 (based on image timestamps)

## CLASS DEFINITIONS

CLASS A - Highest Quality
Description: Fresh, uniform green leaves with minimal defects
Green Content: Above 70%
Defects: Less than 2%
Use Case: Premium grade for direct sale

CLASS B - High Quality
Description: Good green leaves with minor imperfections
Green Content: 60-70%
Defects: 2-5%
Use Case: Commercial grade

CLASS C - Medium Quality
Description: Acceptable leaves with moderate defects or color variation
Green Content: 40-60%
Defects: 5-15%
Use Case: Processing grade

CLASS D - Low Quality
Description: Leaves with significant yellowing or brown spots
Green Content: 20-40%
Defects: 15-30%
Use Case: Limited processing use

CLASS E - Poor Quality
Description: Heavily damaged, brown, or diseased leaves
Green Content: Below 20%
Defects: Above 30%
Use Case: Reject or compost

## DEFECT IDENTIFICATION METHOD

The system identifies defects by analyzing leaf colors:

Yellow Spots (Chlorosis/Aging):
- HSV Color Range: Hue 40-70, Saturation 40-255, Value 0-255
- Indicates: Nutrient deficiency, aging, or early disease
- Impact: Reduces quality score

Brown Spots (Disease/Damage):
- HSV Color Range: Hue 15-35, Saturation 30-255, Value 0-50
- Indicates: Fungal infection, physical damage, or decay
- Impact: Major quality reduction

Black Spots (Severe Damage):
- Grayscale Threshold: Below 20 intensity
- Indicates: Necrosis, severe disease, or rot
- Impact: Critical quality failure

Color Uniformity:
- Measures: Standard deviation of hue across leaf
- Lower STD = More uniform = Higher quality
- Higher STD = Patchy colors = Lower quality

Green Content Analysis:
- Target Range: Hue 80-140 (healthy green)
- Saturation: Above 30 (vibrant color)
- Percentage: Ratio of green pixels to total pixels
- Higher green % = Healthier leaf

Quality Scoring Formula:
- Color Health Score: 40 points max (green ratio + uniformity + hue consistency)
- Defect Penalty: 35 points max (subtracted based on yellow/brown/black spots)
- ML Confidence: 25 points max (model prediction confidence)
- Total: 0-100 points
- Grade Assignment: A (90+), B (75+), C (50+), D (25+), E (below 25)

## PYTHON DEPENDENCIES

Core Libraries:
- Python 3.8 or higher

Machine Learning:
- torch 2.0.0 or higher (PyTorch for neural network)
- torchvision 0.15.0 or higher (Pre-trained models and transforms)

Image Processing:
- opencv-python 4.8.0 or higher (Computer vision and image manipulation)
- pillow 10.0.0 or higher (Image file I/O)
- numpy 1.24.0 or higher (Array operations and math)

Advanced Analysis:
- scikit-image 0.21.0 or higher (Image analysis algorithms)
- scipy 1.11.0 or higher (Scientific computing for color analysis)
- colormath 3.0.0 or higher (Color space conversions)

GUI and Networking:
- tkinter (Built-in with Python - GUI framework)
- requests 2.31.0 or higher (HTTP communication with Raspberry Pi)

Utilities:
- python-dateutil 2.8.2 or higher (Date and time utilities)
- datetime (Built-in - Timestamp management)
- threading (Built-in - Concurrent operations)
- queue (Built-in - Thread-safe communication)

Installation:
Run: pip install -r laptop_client/requirements_advanced.txt

## SYSTEM ARCHITECTURE

Laptop Client (laptop_client_gui.py):
- Runs the ML model for leaf classification
- Displays live video feed from Pi or local camera
- Shows real-time detection info panel
- Sends control commands to Raspberry Pi
- Performs single and batch image classification

Raspberry Pi Server (pi_server_gui.py):
- Streams video via MJPEG (Motion JPEG)
- Controls hardware peripherals
- Receives commands via HTTP REST API
- Operates robotic arms and stepper motor
- Manages camera servo positioning

Communication Protocol: HTTP REST API over local network

## RASPBERRY PI CONNECTION

Default Port: 5000

Connection Methods:

1. Manual IP Entry:
   - Enter Raspberry Pi IP address (e.g., 192.168.1.100)
   - Click "Connect" button
   - System will test connection and establish link

2. Auto Discovery:
   - Click "Find Pi" button
   - Scans local network starting from last known IP
   - Searches sequentially (e.g., if last was .100, tries .101, .99, .102, .98)
   - Shows list of found servers
   - Select server and auto-connects

3. Saved IP History:
   - System saves last successful IP to buyo_pi_ip.txt
   - Auto-loads on next startup
   - Discovery prioritizes this IP first

Connection Indicators:
- Green Badge: "CONNECTED" with IP address
- Red Badge: "DISCONNECTED"
- Status updates in real-time

Network Requirements:
- Laptop and Pi must be on same local network
- No firewall blocking port 5000
- Raspberry Pi server must be running

## RASPBERRY PI CONTROL COMMANDS

All commands sent via HTTP POST to: http://PI_IP:5000/command

Command Cooldown: 2 seconds between commands (prevents command flooding)

Robotic Arm Commands:

/command with action: "test", params: {"arm": 1 or 2}
- Tests specified robotic arm by moving through sequence
- Arm moves down, opens claw, closes claw, returns home

/command with action: "home", params: {}
- Returns all robotic arms to home position
- Resets servos to default angles

/command with action: "drop", params: {"arm": 1 or 2, "class": "CLASS A"}
- Moves arm to position for specified class bin
- Opens claw to drop leaf
- Returns to home position

/command with action: "claw_open", params: {"arm": 1}
- Opens claw on specified arm
- Waits for user to place leaf

/command with action: "claw_close", params: {"arm": 1}
- Closes claw on specified arm
- Grips leaf securely

/command with action: "grab_and_drop", params: {"arm": 1, "class": "CLASS A"}
- Full automated sequence:
- Opens claw for 3 seconds (user places leaf)
- Closes claw
- Waits for camera detection
- Drops leaf into appropriate bin
- Returns home

Carousel Stepper Motor Commands:

/command with action: "rotate", params: {"class": "A"}
- Rotates stepper motor to position for specified class
- Class positions: A, B, C, D, E (evenly spaced around carousel)

Camera Servo Commands:

/command with action: "camera", params: {"position": 1}
- Moves camera servo to view specified arm (1-9)
- Uses MG996R servo in standard 180-degree mode
- Angle range: 0-16 degrees (2-degree steps for 9 arms)
- PWM automatically killed after movement (prevents wire damage)

/command with action: "camera_sweep", params: {}
- Sweeps camera across all 9 arm positions
- Useful for inspection or calibration

/command with action: "camera_stop", params: {}
- Emergency stop for camera servo
- Kills PWM signal immediately
- Use if servo behaves erratically

Peripheral Detection:

/peripherals endpoint (GET request)
- Returns JSON with connected hardware status
- Shows: Stepper motor, servos, camera, PCA9685 board
- Used for diagnostics

## HARDWARE COMPONENTS

On Raspberry Pi:

PCA9685 Servo Controller:
- 16-channel PWM driver
- Controls up to 16 servos simultaneously
- I2C communication with Raspberry Pi

Robotic Arms:
- 2 arms (expandable to 9)
- Each arm has shoulder and claw servos
- Servo angles: Shoulder (0-180), Claw (open/close positions)

Stepper Motor:
- Controls carousel rotation
- 5-position system (one per class)
- Rotates to align bin with drop zone

MG996R Camera Servo:
- High-torque servo for camera positioning
- 180-degree standard mode (NOT continuous)
- Views 9 different arm positions
- 2-degree step size between positions
- Safety: PWM killed after each move to prevent wire damage

Camera:
- USB webcam or Pi Camera module
- Streams MJPEG video at ~30 FPS
- Resolution: Configurable (default 640x480)

## DETECTION THRESHOLDS

Green Content Filter:
- Enabled: True (rejects non-leaf objects like human faces)
- Minimum Green Ratio: 15% of pixels must be green
- Hue Range: 35-85 degrees (HSV color space)
- Minimum Saturation: 40 (ensures vibrant color)

ML Model Thresholds:
- Detection Threshold: 40% confidence (high confidence detection)
- Buyo Confidence Minimum: 20% (low threshold to show predictions)
- These are very low due to current model accuracy issues

Stabilization Settings:
- Enabled: True (prevents blinking detections)
- Frames Required: 5 consecutive frames must agree
- Agreement Threshold: 60% of frames must match same class
- Hold Time: 2 seconds (displays same result before changing)

## REAL-TIME DETECTION INFO PANEL

Located below video feed, displays:

Row 1: Classification and Confidence
- Shows: CLASS A/B/C/D/E or NO BUYO
- Confidence percentage with color coding
- Green: High confidence (above 40%)
- Orange: Medium confidence (20-40%)
- Red: Low confidence (below 20%)

Row 2: Quality Assessment
- Quality Score: 0-100 points with star rating (1-5 stars)
- Quality Grade: A through E with color coding

Row 3: Color Health
- Green Content: Percentage of healthy green pixels
- Uniformity: Color consistency score (0-100%)

Row 4: Defect Analysis
- Total Defects: Combined defect percentage
- Breakdown: Y (Yellow), B (Brown), BL (Black) spot percentages

Updates: Every 0.5 seconds during active detection
Cleared: When no Buyo leaf detected

## SINGLE IMAGE CLASSIFICATION

Feature: Classify individual image files

Steps:
1. Click "Classify Single Image" button
2. Select image file (JPG, PNG, JPEG, BMP)
3. System runs ML model and color analysis
4. Shows results in scrollable dialog

Result Dialog Shows:
- Image preview (resized to fit)
- Classification section: Class name and confidence
- Quality analysis: Score, grade, stars
- Color health: Green content, uniformity, defects
- All class probabilities: Bar chart showing all 5 classes
- Save option: Copy image to classified folder

Analysis Time: Typically 50-200ms per image (GPU) or 500-1000ms (CPU)

## BATCH IMAGE CLASSIFICATION

Feature: Classify multiple images at once

Steps:
1. Click "Classify Batch Images" button
2. Select multiple image files (Ctrl+Click or Shift+Click)
3. Progress bar shows processing status
4. Results displayed in grid view

Results Window Shows:

Performance Metrics:
- Time Complexity: O(n) - Linear scaling
- Total processing time
- Throughput: Images per second
- Average time per image
- GPU/CPU indicator

Classification Distribution:
- Bar chart showing class counts
- Percentage breakdown
- Visual progress bars

Confidence Analysis:
- High confidence count (above 70%)
- Medium confidence count (40-70%)
- Low confidence count (below 40%)
- Average confidence across all images

Image Grid:
- 3 columns of thumbnails
- Each shows: Image preview, class, confidence, grade, quality score
- Click any image to see full detailed analysis
- Scrollable for large batches

Export Options:
- Save All: Copies images to classified_images/CLASS_X/ folders
- Export CSV: Creates spreadsheet with all results and probabilities

## CAMERA MODES

Local Camera Mode:
- Uses laptop's built-in webcam
- Starts automatically if not connected to Pi
- Full ML detection and analysis
- Save frames with classification
- No hardware control (requires Pi)

Pi Remote Camera Mode:
- Streams video from Raspberry Pi camera
- MJPEG stream over HTTP
- Full ML detection and analysis
- Hardware control enabled
- Can trigger robotic arms based on detection

Frame Rate: Target 30 FPS (actual may vary based on CPU/GPU)
Resolution: 640x480 displayed (configurable in code)

## AUTO GRAB AND DROP WORKFLOW

Automated leaf handling sequence:

Step 1: Open Claw (3 seconds)
- Robotic arm claw opens
- User has 3 seconds to place Buyo leaf in claw
- Countdown displayed

Step 2: Close Claw
- Claw closes to grip leaf
- Waits 1 second for grip to secure

Step 3: Camera Detection (10 second timeout)
- Moves leaf into camera view
- ML model runs detection
- Waits for stable detection
- Identifies leaf class

Step 4: Navigate to Bin
- Moves arm to position for detected class
- Lowers to drop height

Step 5: Drop Leaf
- Opens claw
- Leaf falls into appropriate class bin
- Waits 1 second

Step 6: Return Home
- Arm returns to home position
- Ready for next leaf

Error Handling:
- If no detection within 10 seconds: Drops to UNKNOWN bin
- If grip fails: Aborts and returns home
- If Pi connection lost: Shows error message

## STATUS INDICATORS

Model Status:
- "Model: READY" (green) - Model loaded successfully
- "Model: Loading..." (orange) - Model initializing
- "Model: ERROR" (red) - Model failed to load

Connection Status:
- "CONNECTED" (green) - Pi connection active
- "DISCONNECTED" (red) - No Pi connection
- "LOCAL CAMERA" (blue) - Using laptop camera

Detection Status:
- "Auto-Detection: ACTIVE" (green) - Always running
- "Detected: CLASS X (Y%)" (green) - High confidence detection
- "Buyo Leaf: CLASS X (Y%)" (orange) - Low confidence detection
- "Scanning for Buyo leaves..." (gray) - No detection
- "NO BUYO" (gray) - Non-leaf object detected

FPS Indicator:
- Shows current frame rate
- Located in bottom-right status bar
- Target: 30 FPS

## FILE LOCATIONS

Model Files:
- buyo_best.pth - Main trained model (ResNet-50)
- buyo_convnext.pth - Alternative model (ConvNeXt)
- buyo_effnet.pth - Alternative model (EfficientNet)
- buyo_resnet50.pth - Alternative model (ResNet-50 variant)

Training Data:
- CLASS A/ - 225 images
- CLASS B/ - 170 images
- CLASS C/ - 169 images
- CLASS D/ - 176 images
- CLASS E/ - 135 images

Code Files:
- laptop_client/laptop_client_gui.py - Main laptop application
- raspberry_pi_server/pi_server_gui.py - Raspberry Pi server
- laptop_client/requirements_advanced.txt - Python dependencies

Output Folders (Created Automatically):
- saved_images/ - Frames saved from live camera
- classified_images/ - Single image classifications
- saved_images/CLASS_X/ - Organized by detected class

Configuration Files:
- buyo_pi_ip.txt - Saves last successful Pi IP address

Documentation:
- SYSTEM_DOCUMENTATION.md - This file
- HOW_TO_USE.md - User guide
- OPERATION_GUIDE.md - Detailed operations
- DETECTION_DISPLAY_FIX.md - Technical fix details
- DETECTION_STABILIZATION_FIX.md - Technical fix details
- MODEL_PERFORMANCE_ISSUE.md - Model accuracy notes

## KEYBOARD SHORTCUTS

None currently implemented (all GUI button-based)

## TROUBLESHOOTING

Model Loading Issues:
- Check buyo_best.pth is in same folder as script
- Verify PyTorch installed: python -c "import torch; print(torch.__version__)"
- Check GPU availability: torch.cuda.is_available()

Pi Connection Issues:
- Verify both devices on same network
- Check Pi server is running
- Try manual IP instead of auto-discovery
- Check firewall not blocking port 5000
- Ping Pi: ping RASPBERRY_PI_IP

Detection Not Working:
- Check camera is working (test with local camera)
- Verify model loaded (check status indicator)
- Ensure adequate lighting
- Check green filter settings if detecting non-leaves

Human Face Detected as Leaf:
- This is expected with current model (trained only on leaves)
- Green filter should reject it now (enabled by default)
- For complete fix: Retrain model with negative examples

Servo Rotating Continuously:
- Use "STOP CAMERA SERVO" emergency button
- Camera servo must use 180-degree mode, not continuous
- PWM should kill after each movement

Blinking Detection:
- Stabilization is now enabled (requires 5 consistent frames)
- Increase STABILIZATION_FRAMES for more stability
- Increase DETECTION_HOLD_TIME for longer display

Low Confidence Scores:
- Current model has accuracy issues (see MODEL_PERFORMANCE_ISSUE.md)
- Thresholds set very low (40% detection, 20% minimum)
- Consider retraining model with more data

## SYSTEM LIMITATIONS

Current Known Issues:

Model Accuracy:
- Current accuracy below 50% on some classes
- Trained without negative examples (non-leaf images)
- Small dataset (875 images total)
- Imbalanced classes (CLASS E has only 135 images)

Detection:
- Very low thresholds required (20% minimum)
- Green filter is workaround for non-leaf rejection
- Stabilization needed to prevent blinking

Hardware:
- Limited to 2 active robotic arms (expandable to 9)
- Stepper motor single-direction only
- Camera servo must not rotate continuously (wire damage risk)

Network:
- Requires local network connection
- No internet/cloud features
- Limited to same subnet

Recommended Improvements:

1. Retrain model with:
   - More images per class (1000+ recommended)
   - Balanced dataset across all classes
   - Negative examples (walls, hands, non-leaves)
   - Data augmentation for variety

2. Add features:
   - Automatic bin switching when full
   - Logging and analytics dashboard
   - Multi-leaf batch processing
   - Quality trend analysis

3. Hardware upgrades:
   - Add weight sensors for bin monitoring
   - LED indicators for class bins
   - Emergency stop button
   - Dust/moisture sensors

## VERSION HISTORY

Current Version: 1.0 (November 2026)

Features:
- 5-class Buyo leaf classification
- Real-time detection with stabilization
- Color analysis and defect detection
- Quality scoring and grading
- Single and batch image classification
- Raspberry Pi hardware control
- 2 robotic arms with auto grab-and-drop
- Stepper motor carousel
- Camera servo positioning
- Green content filter
- Detection info panel
- Network auto-discovery

Training Date: May 2026 (based on dataset timestamps)
Model Architecture: ResNet-50
Dataset: 875 images across 5 classes
