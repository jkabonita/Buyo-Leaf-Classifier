<div align="center">

# Buyo Leaf Classification System

### AI-Powered Automated Sorting and Quality Grading for Betel Leaves

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## Overview

An automated Buyo (betel) leaf classification and sorting system using deep learning and computer vision. The system analyzes leaves in real-time, classifies them into 5 quality grades (A through E), detects defects, and controls robotic hardware for automated sorting.

**Key Capabilities:**
- Real-time leaf detection and classification using ResNet-50
- Color analysis and defect identification (yellow, brown, black spots)
- Quality scoring from 0-100 with letter grades
- Automated robotic sorting with 9-arm capacity
- Network-based control via Raspberry Pi

---

## Features

### ![ML](https://img.shields.io/badge/-Machine_Learning-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)

- **5-Class Classification**: Grades leaves from CLASS A (premium) to CLASS E (reject)
- **ResNet-50 Architecture**: Pre-trained deep convolutional neural network
- **Real-time Inference**: 20-200ms per image depending on hardware
- **Detection Stabilization**: 5-frame consensus to prevent flickering results
- **Batch Processing**: Classify multiple images simultaneously

### ![CV](https://img.shields.io/badge/-Computer_Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

- **Color Health Analysis**: Green content percentage and uniformity
- **Defect Detection**: Identifies yellow, brown, and black spots via HSV analysis
- **Quality Scoring**: 0-100 point system with star ratings
- **Green Content Filter**: Rejects non-leaf objects (humans, walls, etc.)
- **MJPEG Video Streaming**: 30 FPS live feed from Raspberry Pi

### ![Hardware](https://img.shields.io/badge/-Hardware_Control-A22846?style=flat-square&logo=raspberrypi&logoColor=white)

- **Robotic Arms**: 9 arms with shoulder + claw servos (2 active, expandable)
- **Stepper Motor**: NEMA 17 carousel for 5-position bin rotation
- **Camera Servo**: MG996R for multi-angle viewing
- **Network Control**: HTTP REST API over local network
- **Auto Grab & Drop**: Fully automated leaf handling workflow

### ![GUI](https://img.shields.io/badge/-User_Interface-0078D4?style=flat-square&logo=windows&logoColor=white)

- **Live Detection Panel**: Real-time display of classification, quality, and defects
- **Single/Batch Classification**: Process individual or multiple images
- **Network Discovery**: Auto-detect Raspberry Pi on local network
- **Activity Logging**: Track all operations and detections
- **Manual Controls**: Override automation for testing

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   LAPTOP CLIENT STATION                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Python GUI (Tkinter)                              │     │
│  │  - PyTorch ML Model (ResNet-50)                    │     │
│  │  - OpenCV Image Processing                         │     │
│  │  - Detection Stabilization (5-frame)               │     │
│  │  - Color Analysis & Defect Detection               │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                HTTP/REST API (Port 5000)
                            │
┌─────────────────────────────────────────────────────────────┐
│                 RASPBERRY PI 4 CONTROLLER                   │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Flask Server                                      │     │
│  │  - MJPEG Video Streaming                           │     │
│  │  - Hardware Control (I2C, GPIO)                    │     │
│  │  - Command Processing                              │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌─────────┐      ┌──────────────┐    ┌─────────────┐
   │ PCA9685 │      │   Stepper    │    │   Camera    │
   │ 18 Servos│      │   Driver     │    │  USB/CSI    │
   │ 9 Arms  │      │   NEMA 17    │    │             │
   └─────────┘      └──────────────┘    └─────────────┘
```

**Power System:**
- 12V DC PSU → Buck Converter → 5V for Pi & Servos
- Stepper motor runs on 12V directly

---

## Quick Start

### Prerequisites

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

**Hardware:**
- Laptop (Windows/Linux/Mac) with 4GB+ RAM
- Optional: NVIDIA GPU with CUDA for faster inference
- Optional: Raspberry Pi 4 for hardware control

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/buyo-leaves.git
cd buyo-leaves
```

2. **Install Python dependencies**
```bash
pip install -r laptop_client/requirements_advanced.txt
```

Required packages:
```
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
torch>=2.0.0
torchvision>=0.15.0
scikit-image>=0.21.0
scipy>=1.11.0
requests>=2.31.0
```

3. **Download the trained model**

Place `buyo_best.pth` in the project root directory.

Model: ResNet-50 trained on 875 Buyo leaf images
Size: ~100MB
Classes: 5 (CLASS A, B, C, D, E)

4. **Run the laptop client**
```bash
python laptop_client/laptop_client_gui.py
```

---

## Usage

### Laptop Client GUI

**Launch Application:**
```bash
python laptop_client/laptop_client_gui.py
```

**Features:**
- **Live Camera Feed**: View real-time video with detection overlay
- **Detection Info Panel**: Shows classification, confidence, quality score, grade, defects
- **Single Image Classification**: Upload and analyze individual images
- **Batch Processing**: Classify multiple images with detailed reports
- **Remote Control**: Connect to Raspberry Pi for hardware operation

**Connect to Raspberry Pi:**
1. Enter Pi IP address (or use "Find Pi" auto-discovery)
2. Click "Connect"
3. Control robotic arms, stepper motor, and camera servo

### Raspberry Pi Server

**Setup:**
```bash
# On Raspberry Pi
cd raspberry_pi_server
python pi_server_gui.py
```

**Hardware Control:**
- Streams video via MJPEG (port 5000)
- Receives commands via HTTP REST API
- Controls servos via PCA9685 I2C driver
- Manages stepper motor via GPIO pins

### Classification Only (No Hardware)

Use local camera without Raspberry Pi:
1. Launch `laptop_client_gui.py`
2. Click "LOCAL CAMERA" if Pi not connected
3. System uses laptop webcam for detection
4. Full ML analysis available, no hardware control

---

## Dataset

**Training Data:**
- **Total Images**: 875
- **CLASS A**: 225 images (25.7%) - Premium quality
- **CLASS B**: 170 images (19.4%) - High quality
- **CLASS C**: 169 images (19.3%) - Medium quality
- **CLASS D**: 176 images (20.1%) - Low quality
- **CLASS E**: 135 images (15.4%) - Reject quality

**Image Characteristics:**
- Format: JPG, JPEG, PNG
- Source: Smartphone camera photos
- Date: May 2026
- Resolution: Variable (resized to 224x224 for training)

---

## Model Architecture

**Network**: ResNet-50 (Residual Network)
**Layers**: 50 convolutional layers with skip connections
**Input**: 224x224 RGB images
**Output**: 5-class probabilities (softmax)
**Pre-training**: ImageNet weights
**Fine-tuning**: 875 Buyo leaf images

**Training Configuration:**
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 50-100
- Data Augmentation: Rotation, flip, brightness, contrast

**Model Files:**
- `buyo_best.pth` - Main production model (ResNet-50)
- `buyo_convnext.pth` - Alternative ConvNeXt model
- `buyo_effnet.pth` - Alternative EfficientNet model
- `buyo_resnet50.pth` - Alternative ResNet-50 variant

---

## Defect Detection

**How It Works:**

The system analyzes leaf images in HSV color space to identify defects:

**Yellow Spots (Chlorosis/Aging):**
- HSV Range: Hue 40-70°, Saturation 40-255, Value 0-255
- Indicates: Nutrient deficiency, aging, early disease
- Impact: Reduces quality score

**Brown Spots (Disease/Damage):**
- HSV Range: Hue 15-35°, Saturation 30-255, Value 0-50
- Indicates: Fungal infection, physical damage, decay
- Impact: Major quality reduction

**Black Spots (Severe Damage):**
- Grayscale Threshold: Intensity < 20
- Indicates: Necrosis, severe disease, rot
- Impact: Critical quality failure

**Green Content:**
- Healthy Range: Hue 80-140°, Saturation 30-80, Value 40-80
- Percentage: Ratio of green pixels to total pixels
- Uniformity: Standard deviation of hue (lower = more uniform)

**Quality Scoring (0-100 points):**
- Color Health: 40 points (green ratio + uniformity + hue)
- Defect Score: 35 points (penalty for yellow/brown/black)
- ML Confidence: 25 points (model prediction confidence)

**Grade Assignment:**
- Grade A: 90+ points, <2% defects
- Grade B: 75+ points, <5% defects
- Grade C: 50+ points, <15% defects
- Grade D: 25+ points, <30% defects
- Grade E: <25 points or >30% defects

---

## Hardware Requirements

### Laptop Client
- **OS**: Windows 10/11, Linux, or macOS
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional NVIDIA GPU with CUDA support
- **Storage**: 500MB for dependencies, 100MB for model

### Raspberry Pi Server
- **Model**: Raspberry Pi 4 (4GB or 8GB RAM)
- **OS**: Raspberry Pi OS (Bullseye or later)
- **Storage**: 32GB microSD card
- **Camera**: USB webcam or Pi Camera Module
- **Network**: WiFi or Ethernet

### Physical Hardware
- **Power Supply**: 12V DC, 5A (60W)
- **Buck Converter**: 12V to 5V, 5A output
- **PCA9685**: 16-channel I2C PWM servo driver
- **Stepper Driver**: A4988 or DRV8825
- **NEMA 17 Stepper**: 1.8° step angle, 12V
- **Servo Motors**: 18x SG90 (9 arms × 2 servos)
- **Camera Servo**: 1x MG996R high-torque servo

---

## Performance

**Inference Speed:**
- GPU (CUDA): 20-50ms per image
- CPU: 200-500ms per image
- Real-time: 15-30 FPS video processing

**Accuracy:**
- Current: ~40-60% (small dataset, needs improvement)
- Confidence Thresholds: 40% detection, 20% minimum

**System Throughput:**
- Single Leaf: 15-20 seconds (automated grab-and-drop)
- Batch Processing: 1-2 images/sec (CPU), 5-10 images/sec (GPU)
- Video Stream: 30 FPS target, 15-25 FPS typical

**Hardware Timing:**
- Servo Movement: 500ms per 90° rotation
- Stepper Rotation: 1-2 seconds per position
- Camera Positioning: 500ms
- Full Sorting Cycle: 15-20 seconds per leaf

---

## Project Structure

```
buyo-leaves/
├── README.md                          # This file
├── SYSTEM_DOCUMENTATION.md            # Complete system documentation
├── buyo_best.pth                      # Trained ResNet-50 model
├── laptop_client/
│   ├── laptop_client_gui.py           # Main laptop application
│   └── requirements_advanced.txt      # Python dependencies
├── raspberry_pi_server/
│   └── pi_server_gui.py               # Raspberry Pi server
├── CLASS A/                           # Training data - Premium quality
├── CLASS B/                           # Training data - High quality
├── CLASS C/                           # Training data - Medium quality
├── CLASS D/                           # Training data - Low quality
├── CLASS E/                           # Training data - Reject quality
├── train_convnext_optimized.py        # Training script (ConvNeXt)
├── train_ensemble.py                  # Ensemble training script
└── install_dependencies_windows.bat   # Windows installer
```

---

## Configuration

### Detection Thresholds

Edit `laptop_client_gui.py`:

```python
# ML Model Configuration
DETECTION_THRESHOLD = 0.40      # 40% confidence for detection
BUYO_CONFIDENCE_MIN = 0.20      # 20% minimum to show prediction

# Green Content Filter
USE_COLOR_FILTER = True         # Enable to reject non-leaves
MIN_GREEN_RATIO = 0.15          # 15% green pixels required
GREEN_HUE_MIN = 35              # Green hue range 35-85°
GREEN_HUE_MAX = 85

# Detection Stabilization
DETECTION_STABILIZATION = True   # Enable stable detection
STABILIZATION_FRAMES = 5         # Frames to agree (60% threshold)
DETECTION_HOLD_TIME = 2.0        # Hold result for 2 seconds
```

### Hardware Settings

Edit `pi_server_gui.py`:

```python
# Servo Configuration
ARM_CHANNELS = {1: 3, 2: 2, 3: 1, 4: 0, ...}  # PCA9685 channel mapping
ARM_DROP_ANGLE = 60                            # Drop position angle
ARM_HOME_ANGLE = 0                             # Home position angle

# Stepper Motor
STEPS_PER_POSITION = 40           # 72° per position (5 bins)
MICROSTEPPING = 16                # Driver microstepping mode

# Camera Servo
CAMERA_CHANNEL = 15               # PCA9685 channel for camera
CAMERA_POSITIONS = 9              # Number of arm viewing positions
CAMERA_STEP = 2                   # Degrees per position (0-16°)
```

---

## Troubleshooting

### Model Not Loading
```
Error: FileNotFoundError: buyo_best.pth
```
**Solution**: Ensure `buyo_best.pth` is in the project root directory

### Low Accuracy
```
Issue: Model showing <50% confidence
```
**Solution**: Current model has limited training data. Consider:
- Collecting more images (2000+ per class recommended)
- Adding negative examples (non-leaf images)
- Balancing dataset across all classes
- Using data augmentation

### GPU Not Detected
```
Warning: No CUDA GPU detected, using CPU
```
**Solution**: 
- Install NVIDIA drivers
- Install CUDA toolkit
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Raspberry Pi Connection Failed
```
Error: Connection refused at 192.168.1.100:5000
```
**Solution**:
- Verify Pi is on same network
- Check Pi server is running: `python pi_server_gui.py`
- Test with ping: `ping 192.168.1.100`
- Use "Find Pi" auto-discovery button

### Human Face Detected as Leaf
```
Issue: System classifies person as CLASS E
```
**Solution**: Green content filter is now enabled by default. If still occurring:
- Verify `USE_COLOR_FILTER = True`
- Adjust `MIN_GREEN_RATIO` threshold
- Consider retraining model with negative examples

---

## Contributing

Contributions are welcome! Areas for improvement:

**Model Enhancement:**
- Collect more training images (target: 2000+ per class)
- Add negative examples (non-leaf objects)
- Implement ensemble methods
- Try different architectures (Vision Transformer, etc.)

**Features:**
- Web-based dashboard
- Historical analytics and reporting
- Automated model retraining
- Mobile app control
- Cloud integration

**Hardware:**
- Activate remaining 7 robotic arms
- Add bin full sensors
- Implement automatic leaf feeder
- Multi-leaf batch processing

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Documentation

For detailed information, see:
- **SYSTEM_DOCUMENTATION.md** - Complete technical documentation
  - Model architecture and training details
  - Hardware connection diagrams and pinouts
  - Defect detection algorithms
  - Quality scoring methodology
  - API command reference
  - Troubleshooting guide

---

## Contact

For questions, issues, or collaborations, please open an issue on GitHub.

---

<div align="center">

**Built with** ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) ![Raspberry Pi](https://img.shields.io/badge/-Raspberry_Pi-A22846?style=flat-square&logo=raspberrypi&logoColor=white)

</div>
