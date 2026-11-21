#!/usr/bin/env python3
"""
Laptop Client GUI - Buyo Leaf Classifier
========================================
Full GUI client application for laptop with:
- Connect to Raspberry Pi server (with auto-discovery)
- Live MJPEG video feed display
- Vision detection with ML model
- Manual control buttons
- Activity logging
- Non-blocking networking
- Network auto-discovery and IP scanning
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import requests
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageTk
import io
import queue
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import os
import platform
import socket
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_PI_IP = "192.168.1.100"
DEFAULT_PORT = 5000
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
COMMAND_COOLDOWN = 2.0  # seconds between commands

# Network discovery settings
SAVED_IP_FILE = "last_pi_ip.txt"
NETWORK_SCAN_TIMEOUT = 0.5  # seconds per IP
COMMON_PI_PORTS = [5000, 8080, 80]

# ============================================================================
# NETWORK DISCOVERY FUNCTIONS
# ============================================================================

def load_saved_ip():
    """Load last successful Pi IP from file"""
    try:
        if os.path.exists(SAVED_IP_FILE):
            with open(SAVED_IP_FILE, 'r') as f:
                return f.read().strip()
    except:
        pass
    return DEFAULT_PI_IP

def save_pi_ip(ip_address):
    """Save successful Pi IP to file"""
    try:
        with open(SAVED_IP_FILE, 'w') as f:
            f.write(ip_address)
    except:
        pass

def get_local_ip():
    """Get local machine IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

def get_network_prefix():
    """Get network prefix from local IP (e.g., 192.168.1)"""
    local_ip = get_local_ip()
    parts = local_ip.split('.')
    return '.'.join(parts[:3])

def check_pi_server(ip, port=DEFAULT_PORT, timeout=NETWORK_SCAN_TIMEOUT):
    """Check if Pi server is running at given IP"""
    try:
        url = f"http://{ip}:{port}/status"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            # Verify it's our Pi server
            if 'camera' in data or 'status' in data:
                return True
    except:
        pass
    return False

def scan_network_for_pi(progress_callback=None):
    """
    Scan local network for Raspberry Pi server
    Returns list of (ip, port) tuples where server was found
    
    Scanning strategy:
    1. Check last successful IP first
    2. Scan nearby IPs (±10 from last IP) sequentially
    3. Scan full range sequentially if not found
    """
    results = []
    network_prefix = get_network_prefix()
    
    print(f"🔍 Scanning network {network_prefix}.x for Pi server...")
    
    # Get last successful IP
    last_ip = load_saved_ip()
    last_host_num = None
    
    # Extract host number from last IP
    try:
        if last_ip and last_ip.startswith(network_prefix):
            last_host_num = int(last_ip.split('.')[-1])
    except:
        pass
    
    # Priority scan list based on connection history
    scanned = set()
    priority_ips = []
    
    if last_host_num is not None:
        # Check last successful IP first
        priority_ips.append(f"{network_prefix}.{last_host_num}")
        
        # Then check nearby IPs sequentially (±10 range)
        # Order: last+1, last-1, last+2, last-2, ...
        for offset in range(1, 11):
            if 1 <= last_host_num + offset <= 254:
                priority_ips.append(f"{network_prefix}.{last_host_num + offset}")
            if 1 <= last_host_num - offset <= 254:
                priority_ips.append(f"{network_prefix}.{last_host_num - offset}")
    else:
        # No history - check common IPs
        priority_ips = [
            f"{network_prefix}.100",
            f"{network_prefix}.101",
            f"{network_prefix}.102",
            f"{network_prefix}.10",
            f"{network_prefix}.11",
            f"{network_prefix}.12",
        ]
    
    # Check priority IPs first
    for ip in priority_ips:
        if progress_callback:
            progress_callback(f"Checking {ip}...")
        
        scanned.add(ip)
        
        for port in COMMON_PI_PORTS:
            if check_pi_server(ip, port):
                print(f"  ✓ Found Pi server at {ip}:{port}")
                results.append((ip, port))
                return results  # Return immediately on first find
    
    # If not found, scan full range sequentially
    if progress_callback:
        progress_callback("Scanning full network range...")
    
    for i in range(1, 255):
        ip = f"{network_prefix}.{i}"
        
        if ip in scanned:
            continue  # Already checked
        
        if progress_callback and i % 5 == 0:
            progress_callback(f"Scanning {ip}...")
        
        for port in COMMON_PI_PORTS:
            if check_pi_server(ip, port, timeout=0.2):  # Faster timeout for full scan
                print(f"  ✓ Found Pi server at {ip}:{port}")
                results.append((ip, port))
                return results  # Return on first find
    
    return results

# ML Model Configuration
import os
# Try to find model in current dir or parent dir
if os.path.exists("buyo_best.pth"):
    MODEL_PATH = "buyo_best.pth"
elif os.path.exists("../buyo_best.pth"):
    MODEL_PATH = "../buyo_best.pth"
else:
    MODEL_PATH = "buyo_best.pth"  # Will fail but show clear error

CLASS_NAMES = ['CLASS A', 'CLASS B', 'CLASS C', 'CLASS D', 'CLASS E']
IMG_SIZE = 224
DETECTION_THRESHOLD = 0.40  # Lowered to catch more detections
BUYO_CONFIDENCE_MIN = 0.20  # Very low threshold to see all predictions
AUTO_DETECTION = True  # Always run detection automatically

# Additional filters for Buyo leaf detection
USE_COLOR_FILTER = False  # DISABLED - model needs retraining, filter hiding valid detections
GREEN_HUE_MIN = 35  # Minimum hue for green (HSV)
GREEN_HUE_MAX = 85  # Maximum hue for green (HSV)
GREEN_SATURATION_MIN = 40  # Minimum saturation
MIN_GREEN_RATIO = 0.15  # Minimum ratio of green pixels

# ============================================================================
# ML MODEL LOADING
# ============================================================================

# Auto-detect best available device
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(0)
    print(f"🎮 GPU detected: {device_name}")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    device = torch.device('cpu')
    print(f"💻 No CUDA GPU detected, using CPU")
    print(f"   To enable GPU acceleration:")
    print(f"   1. Install NVIDIA drivers")
    print(f"   2. Run: install_pytorch_cuda.bat")

model_loaded = False
model = None
preprocess = None

def load_model():
    """Load ML model for vision detection"""
    global model, model_loaded, preprocess
    
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"✗ Model file not found: {MODEL_PATH}")
            print(f"   Current directory: {os.getcwd()}")
            print(f"   Looking for: {os.path.abspath(MODEL_PATH)}")
            print(f"\n   SOLUTION: Run the client from the project root directory:")
            print(f"   cd ~/Desktop/BUYO-LEAVES")
            print(f"   python laptop_client/laptop_client_gui.py")
            return False
        
        print(f"⏳ Loading model from {MODEL_PATH}...")
        print(f"   File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")
        
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        backbone = ckpt.get('backbone', 'efficientnet_v2_s')
        print(f"   Backbone: {backbone}")
        
        # Build model architecture
        if 'convnext' in backbone:
            print(f"   Building ConvNeXt-Tiny architecture...")
            model = models.convnext_tiny(weights=None)
            in_f = model.classifier[2].in_features
            model.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
                nn.LayerNorm(in_f, eps=1e-6), nn.Dropout(p=0.35),
                nn.Linear(in_f, 512), nn.GELU(), nn.LayerNorm(512),
                nn.Dropout(p=0.25), nn.Linear(512, 256), nn.GELU(),
                nn.Dropout(p=0.15), nn.Linear(256, 5),
            )
        elif 'resnet' in backbone:
            print(f"   Building ResNet50 architecture...")
            model = models.resnet50(weights=None)
            in_f = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.35), nn.Linear(in_f, 512), nn.ReLU(),
                nn.BatchNorm1d(512), nn.Dropout(p=0.20),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.15),
                nn.Linear(256, 5),
            )
        else:  # efficientnet
            print(f"   Building EfficientNetV2-S architecture...")
            model = models.efficientnet_v2_s(weights=None)
            in_f = model.classifier[1].in_features
            # Simple architecture matching the saved checkpoint
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.35, inplace=True),  # 0
                nn.Linear(in_f, 512),                # 1 (1280 -> 512)
                nn.SiLU(),                           # 2
                nn.Dropout(p=0.25),                  # 3
                nn.Linear(512, 5),                   # 4 (512 -> 5 classes)
            )
        
        print(f"   Loading state dict...")
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(device)
        model.eval()
        print(f"   Model moved to {device}")
        
        # Setup preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        model_loaded = True
        print(f"\n✅ MODEL LOADED SUCCESSFULLY!")
        print(f"   Classes: {CLASS_NAMES}")
        print(f"   Detection Threshold: {DETECTION_THRESHOLD*100}%")
        print(f"   Buyo Confidence Min: {BUYO_CONFIDENCE_MIN*100}%")
        print(f"   Auto-Detection: {'ENABLED' if AUTO_DETECTION else 'DISABLED'}")
        return True
        
    except Exception as e:
        print(f"\n❌ MODEL LOADING FAILED!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_if_green_leaf(frame):
    """
    Check if frame contains green leaf-like colors
    Returns True if significant green content detected
    """
    if not USE_COLOR_FILTER:
        return True
    
    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define green color range in HSV
        lower_green = np.array([GREEN_HUE_MIN, GREEN_SATURATION_MIN, 40])
        upper_green = np.array([GREEN_HUE_MAX, 255, 255])
        
        # Create mask for green colors
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate ratio of green pixels
        green_pixels = np.sum(green_mask > 0)
        total_pixels = frame.shape[0] * frame.shape[1]
        green_ratio = green_pixels / total_pixels
        
        # Return True if enough green content
        return green_ratio >= MIN_GREEN_RATIO
        
    except Exception as e:
        print(f"Color filter error: {e}")
        return True  # Default to allowing detection if filter fails


def detect_target(frame):
    """
    Vision detection - detect Buyo leaves in frame
    
    Returns only valid Buyo leaf detections with confidence above threshold
    
    Args:
        frame: OpenCV BGR image (numpy array)
    
    Returns:
        dict with detection results:
        {
            'detected': bool,  # True if Buyo leaf detected
            'is_buyo': bool,   # True if this is likely a Buyo leaf
            'class': str,
            'confidence': float,
            'bbox': (x1, y1, x2, y2) or None,
            'annotated_frame': numpy array
        }
    """
    if not model_loaded or model is None:
        return {
            'detected': False,
            'is_buyo': False,
            'class': 'MODEL NOT LOADED',
            'confidence': 0.0,
            'bbox': None,
            'annotated_frame': frame
        }
    
    try:
        # Check if frame contains green leaf-like colors (when filter is enabled)
        has_green = check_if_green_leaf(frame)
        
        # If green filter is enabled and no green detected, skip inference
        if USE_COLOR_FILTER and not has_green:
            # No green content - likely not a Buyo leaf
            annotated = frame.copy()
            h, w = frame.shape[:2]
            cv2.putText(annotated, "🔍 SCANNING...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            cv2.putText(annotated, "No green leaf detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
            
            return {
                'detected': False,
                'is_buyo': False,
                'class': 'NO GREEN',
                'class_idx': -1,
                'confidence': 0.0,
                'probabilities': [0.0] * len(CLASS_NAMES),
                'bbox': None,
                'annotated_frame': annotated
            }
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Preprocess and infer
        tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = CLASS_NAMES[pred_idx]
        
        # Check if this is a Buyo leaf (confidence above minimum threshold)
        is_buyo = confidence >= BUYO_CONFIDENCE_MIN
        
        # Check if confident enough for positive detection
        detected = confidence >= DETECTION_THRESHOLD
        
        # Annotate frame
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Only show detection info if it's likely a Buyo leaf
        if is_buyo:
            # Color coding based on confidence
            if confidence >= DETECTION_THRESHOLD:
                color = (0, 255, 0)  # Green - High confidence
                status = "🍃 BUYO DETECTED"
            elif confidence >= BUYO_CONFIDENCE_MIN:
                color = (0, 165, 255)  # Orange - Medium confidence
                status = "🍃 BUYO LEAF"
            else:
                color = (100, 100, 255)  # Red - Low confidence
                status = "SCANNING"
            
            # Draw detection box
            padding = 20
            cv2.rectangle(annotated, (padding, padding), (w-padding, h-padding), color, 3)
            
            # Main class label (large, center)
            class_text = f"{pred_class}"
            conf_text = f"{confidence*100:.1f}%"
            
            # Get text size for background
            (class_w, class_h), _ = cv2.getTextSize(class_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            
            # Center position
            center_x = w // 2
            center_y = h // 2
            
            # Draw background for class
            cv2.rectangle(annotated, 
                         (center_x - class_w//2 - 10, center_y - 60),
                         (center_x + class_w//2 + 10, center_y - 10),
                         (0, 0, 0), -1)
            cv2.rectangle(annotated, 
                         (center_x - class_w//2 - 10, center_y - 60),
                         (center_x + class_w//2 + 10, center_y - 10),
                         color, 3)
            
            # Draw class text
            cv2.putText(annotated, class_text, 
                       (center_x - class_w//2, center_y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Draw background for confidence
            cv2.rectangle(annotated,
                         (center_x - conf_w//2 - 10, center_y + 10),
                         (center_x + conf_w//2 + 10, center_y + 50),
                         (0, 0, 0), -1)
            
            # Draw confidence text
            cv2.putText(annotated, conf_text,
                       (center_x - conf_w//2, center_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Status at top
            cv2.putText(annotated, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw probability bars (compact, top-right corner)
            bar_x = w - 220
            bar_y = 50
            bar_height = 15
            
            for i, cls in enumerate(CLASS_NAMES):
                prob = probs[i]
                bar_w = int(150 * prob)
                
                # Background bar
                cv2.rectangle(annotated, (bar_x, bar_y + i*20), 
                            (bar_x + 150, bar_y + i*20 + bar_height),
                            (50, 50, 50), -1)
                
                # Probability bar
                bar_color = color if i == pred_idx else (100, 100, 100)
                cv2.rectangle(annotated, (bar_x, bar_y + i*20), 
                            (bar_x + bar_w, bar_y + i*20 + bar_height),
                            bar_color, -1)
                
                # Class label
                cv2.putText(annotated, f"{cls}: {prob*100:.0f}%", 
                           (bar_x + 155, bar_y + i*20 + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        else:
            # Not a Buyo leaf or very low confidence
            cv2.putText(annotated, "🔍 SCANNING...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            cv2.putText(annotated, "No Buyo Leaf Detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        
        return {
            'detected': detected,
            'is_buyo': is_buyo,
            'class': pred_class if is_buyo else 'NO BUYO',
            'class_idx': pred_idx if is_buyo else -1,
            'confidence': confidence,
            'probabilities': probs.tolist(),
            'bbox': None,
            'annotated_frame': annotated
        }
        
    except Exception as e:
        print(f"Detection error: {e}")
        return {
            'detected': False,
            'is_buyo': False,
            'class': 'ERROR',
            'confidence': 0.0,
            'bbox': None,
            'annotated_frame': frame
        }

# ============================================================================
# PROGRESS DIALOG
# ============================================================================
class ProgressDialog:
    """Progress dialog for batch operations"""
    def __init__(self, parent, title, total_items):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("500x200")
        self.dialog.configure(bg='#2b2b2b')
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center window
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (200 // 2)
        self.dialog.geometry(f"500x200+{x}+{y}")
        
        self.total_items = total_items
        self.current_item = 0
        
        # Title
        tk.Label(self.dialog, text=title, font=('Arial', 14, 'bold'),
                bg='#2b2b2b', fg='#ffffff').pack(pady=10)
        
        # Status text
        self.status_label = tk.Label(self.dialog, text="Initializing...",
                                     font=('Arial', 10), bg='#2b2b2b', fg='#ffffff')
        self.status_label.pack(pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.dialog, variable=self.progress_var,
                                           maximum=100, length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        
        # Progress text
        self.progress_label = tk.Label(self.dialog, text="0 / 0 (0%)",
                                       font=('Arial', 9), bg='#2b2b2b', fg='#888888')
        self.progress_label.pack(pady=5)
        
        # Time estimate
        self.time_label = tk.Label(self.dialog, text="Estimated time: Calculating...",
                                   font=('Arial', 9), bg='#2b2b2b', fg='#888888')
        self.time_label.pack(pady=5)
        
        self.start_time = time.time()
    
    def update(self, current, status_text="Processing..."):
        """Update progress"""
        self.current_item = current
        progress_pct = (current / self.total_items * 100) if self.total_items > 0 else 0
        
        self.progress_var.set(progress_pct)
        self.status_label.config(text=status_text)
        self.progress_label.config(text=f"{current} / {self.total_items} ({progress_pct:.1f}%)")
        
        # Calculate time estimate
        if current > 0:
            elapsed = time.time() - self.start_time
            avg_time = elapsed / current
            remaining = (self.total_items - current) * avg_time
            self.time_label.config(text=f"Estimated time remaining: {remaining:.1f}s")
        
        self.dialog.update()
    
    def close(self):
        """Close the dialog"""
        self.dialog.destroy()


# ============================================================================
# GUI APPLICATION
# ============================================================================
class LaptopClientGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Buyo Classifier - Laptop Client")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Connection state
        self.pi_ip = tk.StringVar(value=load_saved_ip())  # Load last successful IP
        self.pi_port = DEFAULT_PORT
        self.connected = False
        self.video_running = False
        self.detection_enabled = True  # Always enabled - no toggle needed
        
        # Local camera support
        self.local_camera = None
        self.use_local_camera = False
        
        # Cooldown tracking
        self.last_command_time = 0
        
        # Threading
        self.video_thread = None
        self.video_stop_event = threading.Event()
        self.command_queue = queue.Queue()
        
        # Current frame and detection
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.last_detection = None  # Store last detection result
        
        self.setup_ui()
        self.start_command_processor()
        
        # Auto-start local camera if Pi not connected
        self.root.after(1000, self.check_and_start_local_camera)
        
    def setup_ui(self):
        """Setup GUI layout"""
        
        # ===== TOP BAR - CONNECTION =====
        top_frame = tk.Frame(self.root, bg='#1e1e1e', relief=tk.RAISED, bd=2)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(top_frame, text="Raspberry Pi IP:", font=('Arial', 10, 'bold'),
                bg='#1e1e1e', fg='#ffffff').pack(side=tk.LEFT, padx=5)
        
        ip_entry = tk.Entry(top_frame, textvariable=self.pi_ip, font=('Arial', 12),
                           width=15, bg='#2b2b2b', fg='#ffffff', insertbackground='white')
        ip_entry.pack(side=tk.LEFT, padx=5)
        
        self.connect_btn = tk.Button(top_frame, text="🔌 Connect", command=self.toggle_connection,
                                     font=('Arial', 10, 'bold'), bg='#4a90e2', fg='white',
                                     padx=20, pady=5)
        self.connect_btn.pack(side=tk.LEFT, padx=10)
        
        self.status_badge = tk.Label(top_frame, text="● DISCONNECTED",
                                     font=('Arial', 10, 'bold'),
                                     bg='#1e1e1e', fg='#ff4444')
        self.status_badge.pack(side=tk.LEFT, padx=10)
        
        # Test connection button
        tk.Button(top_frame, text="🔍 Test", command=self.test_connection,
                 font=('Arial', 9), bg='#555555', fg='white', padx=10).pack(side=tk.LEFT, padx=5)
        
        # Auto-discover button
        tk.Button(top_frame, text="🔎 Find Pi", command=self.auto_discover_pi,
                 font=('Arial', 9), bg='#9c27b0', fg='white', padx=10).pack(side=tk.LEFT, padx=5)
        
        # ===== MAIN CONTENT =====
        content_frame = tk.Frame(self.root, bg='#2b2b2b')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left Panel - Video Feed
        left_panel = tk.Frame(content_frame, bg='#2b2b2b')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        video_frame = tk.LabelFrame(left_panel, text="Live Camera Feed",
                                   font=('Arial', 10, 'bold'),
                                   bg='#1e1e1e', fg='#ffffff')
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_frame, bg='#000000',
                                    width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        self.video_label.pack(padx=5, pady=5)
        
        # Detection status (always on - no toggle)
        detection_frame = tk.Frame(video_frame, bg='#1e1e1e')
        detection_frame.pack(pady=5)
        
        tk.Label(detection_frame, text="🎯 Auto-Detection: ACTIVE",
                font=('Arial', 10, 'bold'),
                bg='#1e1e1e', fg='#4caf50').pack()
        
        self.detection_status = tk.Label(detection_frame, text="Scanning for Buyo leaves...",
                                        font=('Arial', 9), bg='#1e1e1e', fg='#888888')
        self.detection_status.pack()
        
        # Right Panel - Controls & Log
        right_panel = tk.Frame(content_frame, bg='#2b2b2b', width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)
        
        # Manual Controls
        control_frame = tk.LabelFrame(right_panel, text="Manual Controls",
                                     font=('Arial', 10, 'bold'),
                                     bg='#1e1e1e', fg='#ffffff')
        control_frame.pack(fill=tk.X, pady=5)
        
        # Grid layout for buttons
        btn_grid = tk.Frame(control_frame, bg='#1e1e1e')
        btn_grid.pack(padx=10, pady=10)
        
        # Classification buttons (always available)
        classify_buttons = [
            ("📷 Classify Single Image", self.classify_single_image, 0, 0, 2),
            ("📦 Classify Batch Images", self.classify_batch_images, 1, 0, 2),
            ("💾 Save Current Frame", self.save_current_frame, 2, 0, 2),
        ]
        
        for text, cmd, row, col, span in classify_buttons:
            btn = tk.Button(btn_grid, text=text, command=cmd,
                          font=('Arial', 9, 'bold'), bg='#4caf50', fg='white',
                          padx=10, pady=10, width=30)
            btn.grid(row=row, column=col, columnspan=span, padx=2, pady=2, sticky='ew')
        
        # Separator
        separator = tk.Label(btn_grid, text="━━━ Remote Control (Pi Only) ━━━",
                            font=('Arial', 8), bg='#1e1e1e', fg='#888888')
        separator.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Remote control buttons (Pi only) - Robotic Arms
        remote_buttons = [
            ("🦾 Test Arm #1", lambda: self.send_command('test', {'arm': 1}), 4, 0, 1),
            ("🦾 Test Arm #2", lambda: self.send_command('test', {'arm': 2}), 4, 1, 1),
            ("🏠 Home All", lambda: self.send_command('home'), 5, 0, 2),
            ("💧 Drop Arm #1", lambda: self.send_command('drop', {'arm': 1, 'class': 'CLASS A'}), 6, 0, 1),
            ("💧 Drop Arm #2", lambda: self.send_command('drop', {'arm': 2, 'class': 'CLASS B'}), 6, 1, 1),
        ]
        
        for text, cmd, row, col, span in remote_buttons:
            btn = tk.Button(btn_grid, text=text, command=cmd,
                          font=('Arial', 9), bg='#4a90e2', fg='white',
                          padx=10, pady=8, width=14 if span == 1 else 30)
            btn.grid(row=row, column=col, columnspan=span, padx=2, pady=2, sticky='ew')
        
        # Separator for Stepper & Servo Controls
        separator2 = tk.Label(btn_grid, text="━━━ Carousel & Camera Controls ━━━",
                             font=('Arial', 8), bg='#1e1e1e', fg='#888888')
        separator2.grid(row=7, column=0, columnspan=2, pady=5)
        
        # Stepper Motor Controls
        stepper_buttons = [
            ("⚙️ Rotate to Class A", lambda: self.send_command('rotate', {'class': 'A'}), 8, 0, 1),
            ("⚙️ Rotate to Class B", lambda: self.send_command('rotate', {'class': 'B'}), 8, 1, 1),
            ("⚙️ Rotate to Class C", lambda: self.send_command('rotate', {'class': 'C'}), 9, 0, 1),
            ("⚙️ Rotate to Class D", lambda: self.send_command('rotate', {'class': 'D'}), 9, 1, 1),
            ("⚙️ Rotate to Class E", lambda: self.send_command('rotate', {'class': 'E'}), 10, 0, 2),
        ]
        
        for text, cmd, row, col, span in stepper_buttons:
            btn = tk.Button(btn_grid, text=text, command=cmd,
                          font=('Arial', 9), bg='#9c27b0', fg='white',
                          padx=10, pady=8, width=14 if span == 1 else 30)
            btn.grid(row=row, column=col, columnspan=span, padx=2, pady=2, sticky='ew')
        
        # Camera Servo (MG996R) Controls
        servo_buttons = [
            ("📹 Camera to Arm #1", lambda: self.send_command('camera', {'position': 1}), 11, 0, 1),
            ("📹 Camera to Arm #2", lambda: self.send_command('camera', {'position': 2}), 11, 1, 1),
            ("📹 Camera Sweep 9 Arms", lambda: self.send_command('camera_sweep'), 12, 0, 2),
        ]
        
        for text, cmd, row, col, span in servo_buttons:
            btn = tk.Button(btn_grid, text=text, command=cmd,
                          font=('Arial', 9), bg='#ff6f00', fg='white',
                          padx=10, pady=8, width=14 if span == 1 else 30)
            btn.grid(row=row, column=col, columnspan=span, padx=2, pady=2, sticky='ew')
        
        # Peripheral Detection Button
        peripheral_btn = tk.Button(btn_grid, text="🔌 Detect Peripherals", 
                                   command=self.detect_peripherals,
                                   font=('Arial', 9, 'bold'), bg='#00796b', fg='white',
                                   padx=10, pady=8, width=30)
        peripheral_btn.grid(row=13, column=0, columnspan=2, padx=2, pady=5, sticky='ew')
        
        # Cooldown indicator
        self.cooldown_label = tk.Label(control_frame, text="",
                                      font=('Arial', 8), bg='#1e1e1e', fg='#888888')
        self.cooldown_label.pack(pady=5)
        
        # Activity Log
        log_frame = tk.LabelFrame(right_panel, text="Activity Log",
                                 font=('Arial', 10, 'bold'),
                                 bg='#1e1e1e', fg='#ffffff')
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame,
                                                  font=('Consolas', 9),
                                                  bg='#1a1a1a', fg='#00ff00',
                                                  insertbackground='white',
                                                  height=20, width=40)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status Bar
        status_bar = tk.Frame(self.root, bg='#1e1e1e', relief=tk.SUNKEN, bd=1)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.model_status_label = tk.Label(status_bar, text="Model: ⏳ Loading...",
                                          font=('Arial', 9, 'bold'), bg='#1e1e1e', fg='#ffaa00')
        self.model_status_label.pack(side=tk.LEFT, padx=10, pady=2)
        
        self.status_text = tk.Label(status_bar, text="Ready",
                                   font=('Arial', 9), bg='#1e1e1e', fg='#888888',
                                   anchor=tk.W)
        self.status_text.pack(side=tk.LEFT, padx=10, pady=2)
        
        self.fps_label = tk.Label(status_bar, text="FPS: 0.0",
                                 font=('Arial', 9), bg='#1e1e1e', fg='#888888')
        self.fps_label.pack(side=tk.RIGHT, padx=10, pady=2)
    
    def log(self, message, level='INFO'):
        """Add message to activity log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        color_tag = {
            'INFO': 'info',
            'SUCCESS': 'success',
            'WARNING': 'warning',
            'ERROR': 'error'
        }.get(level, 'info')
        
        # Configure tags if not already done
        self.log_text.tag_config('info', foreground='#00ff00')
        self.log_text.tag_config('success', foreground='#00ff00')
        self.log_text.tag_config('warning', foreground='#ffaa00')
        self.log_text.tag_config('error', foreground='#ff4444')
        
        log_msg = f"[{timestamp}] {message}\n"
        
        def update_log():
            self.log_text.insert(tk.END, log_msg, color_tag)
            self.log_text.see(tk.END)
        
        if threading.current_thread() is threading.main_thread():
            update_log()
        else:
            self.root.after(0, update_log)
    
    def auto_discover_pi(self):
        """Auto-discover Raspberry Pi on network"""
        self.log("🔎 Searching for Raspberry Pi on network...")
        
        # Create progress dialog
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Discovering Raspberry Pi")
        progress_dialog.geometry("400x150")
        progress_dialog.configure(bg='#2b2b2b')
        progress_dialog.transient(self.root)
        progress_dialog.grab_set()
        
        # Center dialog
        progress_dialog.update_idletasks()
        x = (progress_dialog.winfo_screenwidth() // 2) - 200
        y = (progress_dialog.winfo_screenheight() // 2) - 75
        progress_dialog.geometry(f"400x150+{x}+{y}")
        
        status_label = tk.Label(progress_dialog, text="Scanning network...",
                               font=('Arial', 10), bg='#2b2b2b', fg='#ffffff')
        status_label.pack(pady=20)
        
        progress_bar = ttk.Progressbar(progress_dialog, mode='indeterminate', length=300)
        progress_bar.pack(pady=10)
        progress_bar.start()
        
        def update_status(text):
            status_label.config(text=text)
            progress_dialog.update()
        
        def discover():
            try:
                results = scan_network_for_pi(update_status)
                
                progress_bar.stop()
                progress_dialog.destroy()
                
                if results:
                    ip, port = results[0]
                    self.pi_ip.set(ip)
                    self.pi_port = port
                    save_pi_ip(ip)
                    
                    self.log(f"✓ Found Pi server at {ip}:{port}", 'SUCCESS')
                    messagebox.showinfo("Pi Found", 
                                       f"Raspberry Pi server found at:\n{ip}:{port}\n\nClick Connect to establish connection.")
                else:
                    self.log("✗ No Pi server found on network", 'WARNING')
                    messagebox.showwarning("Not Found", 
                                          "Could not find Raspberry Pi server on local network.\n\n"
                                          "Please ensure:\n"
                                          "1. Pi server is running\n"
                                          "2. Both devices on same WiFi\n"
                                          "3. Pi firewall allows port 5000")
            except Exception as e:
                progress_bar.stop()
                progress_dialog.destroy()
                self.log(f"✗ Discovery error: {str(e)}", 'ERROR')
        
        threading.Thread(target=discover, daemon=True).start()
    
    def test_connection(self):
        """Test connection to Raspberry Pi"""
        ip = self.pi_ip.get()
        self.log(f"Testing connection to {ip}...")
        
        def test():
            try:
                response = requests.get(f"http://{ip}:{self.pi_port}/status", timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    self.log(f"✓ Connection successful! Camera: {data.get('camera', 'unknown')}", 'SUCCESS')
                else:
                    self.log(f"✗ Connection failed: HTTP {response.status_code}", 'ERROR')
            except Exception as e:
                self.log(f"✗ Connection failed: {str(e)}", 'ERROR')
        
        threading.Thread(target=test, daemon=True).start()
    
    def check_and_start_local_camera(self):
        """Start local camera if not connected to Pi"""
        if not self.connected and not self.video_running:
            self.log("📷 No Pi connection - starting local camera...")
            self.start_local_camera()
    
    def start_local_camera(self):
        """Start video feed from local laptop camera"""
        self.log("Starting local camera...")
        
        # Find available camera
        camera_index = None
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_index = i
                cap.release()
                break
        
        if camera_index is None:
            self.log("✗ No local camera detected", 'ERROR')
            messagebox.showerror("Camera Error", "No camera detected on this device")
            return
        
        self.use_local_camera = True
        self.local_camera = cv2.VideoCapture(camera_index)
        
        if not self.local_camera.isOpened():
            self.log("✗ Failed to open camera", 'ERROR')
            return
        
        # Start local video stream
        self.video_stop_event.clear()
        self.video_thread = threading.Thread(target=self.local_camera_worker, daemon=True)
        self.video_thread.start()
        
        # Update UI
        self.status_badge.config(text="● LOCAL CAMERA", fg='#ffaa00')
        self.status_text.config(text=f"Using local camera {camera_index}")
        self.log(f"✓ Local camera started (Camera {camera_index})", 'SUCCESS')
    
    def stop_local_camera(self):
        """Stop local camera feed"""
        if self.local_camera:
            self.video_stop_event.set()
            if self.video_thread:
                self.video_thread.join(timeout=2)
            
            self.local_camera.release()
            self.local_camera = None
            self.use_local_camera = False
            self.video_running = False
            
            # Clear display
            self.video_label.config(image='')
            
            self.log("✓ Local camera stopped", 'SUCCESS')
    
    def local_camera_worker(self):
        """Background worker for local camera feed with auto-detection"""
        self.log("Local camera worker started with auto-detection...")
        fps_counter = []
        last_detection_time = 0
        detection_cooldown = 0.5  # Run detection every 0.5 seconds
        
        self.video_running = True
        
        try:
            while not self.video_stop_event.is_set():
                t0 = time.time()
                
                ret, frame = self.local_camera.read()
                if not ret:
                    self.log("✗ Failed to read from camera", 'ERROR')
                    break
                
                # Store current frame
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Always run detection
                display_frame = frame
                current_time = time.time()
                if current_time - last_detection_time >= detection_cooldown:
                    result = detect_target(frame)
                    display_frame = result['annotated_frame']
                    
                    # Update detection status based on Buyo detection
                    if result['is_buyo']:
                        if result['detected']:
                            status_text = f"✓ Detected: {result['class']} ({result['confidence']*100:.1f}%)"
                            status_color = '#4caf50'
                            self.log(f"🍃 Buyo leaf detected: {result['class']} ({result['confidence']*100:.1f}%)", 'SUCCESS')
                        else:
                            status_text = f"Buyo Leaf: {result['class']} ({result['confidence']*100:.1f}%)"
                            status_color = '#ffaa00'
                    else:
                        status_text = "Scanning for Buyo leaves..."
                        status_color = '#888888'
                    
                    self.root.after(0, lambda t=status_text, c=status_color: 
                                  self.detection_status.config(text=t, fg=c))
                    
                    # Store last detection result
                    if result['is_buyo']:
                        self.last_detection = {
                            'class': result['class'],
                            'confidence': result['confidence']
                        }
                    
                    last_detection_time = current_time
                
                # Resize for display
                display_frame = cv2.resize(display_frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update display
                self.video_label.config(image=img_tk)
                self.video_label.image = img_tk
                
                # Calculate FPS
                fps_counter.append(1.0 / max(0.001, time.time() - t0))
                if len(fps_counter) > 30:
                    fps_counter.pop(0)
                fps = np.mean(fps_counter)
                self.root.after(0, lambda f=fps: self.fps_label.config(text=f"FPS: {f:.1f}"))
                
                # Limit framerate
                time.sleep(0.03)  # ~30 FPS
        
        except Exception as e:
            self.log(f"✗ Local camera error: {str(e)}", 'ERROR')
        
        finally:
            self.video_running = False
            self.log("Local camera worker stopped")
    
    def toggle_connection(self):
        """Connect or disconnect from Raspberry Pi"""
        if not self.connected:
            self.connect()
        else:
            self.disconnect()
    
    def connect(self):
        """Connect to Raspberry Pi and start video stream with retry logic"""
        ip = self.pi_ip.get()
        self.log(f"Connecting to {ip}:{self.pi_port}...")
        
        # Stop local camera if running
        if self.use_local_camera:
            self.log("Stopping local camera...")
            self.stop_local_camera()
        
        # Test connection first with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.log(f"Connection attempt {attempt + 1}/{max_retries}...")
                response = requests.get(f"http://{ip}:{self.pi_port}/status", timeout=5)
                if response.status_code == 200:
                    # Connection successful
                    self.log(f"✓ Connection test successful", 'SUCCESS')
                    break
            except requests.exceptions.Timeout:
                self.log(f"✗ Attempt {attempt + 1} timed out", 'WARNING')
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    self.log("✗ Connection failed: Server not responding", 'ERROR')
                    messagebox.showerror("Connection Error", 
                                        f"Cannot reach Raspberry Pi at {ip}:{self.pi_port}\n\n"
                                        f"Server is not responding.\n\n"
                                        f"Solutions:\n"
                                        f"1. Click 'Find Pi' to auto-discover\n"
                                        f"2. Verify Pi server is running\n"
                                        f"3. Check both devices on same WiFi\n"
                                        f"4. Try manual IP entry")
                    self.root.after(1000, self.check_and_start_local_camera)
                    return
            except Exception as e:
                self.log(f"✗ Attempt {attempt + 1} failed: {str(e)}", 'WARNING')
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    self.log(f"✗ Connection failed: {str(e)}", 'ERROR')
                    messagebox.showerror("Connection Error", 
                                        f"Cannot reach Raspberry Pi:\n{str(e)}\n\n"
                                        f"Click 'Find Pi' to auto-discover on network")
                    self.root.after(1000, self.check_and_start_local_camera)
                    return
        
        # Save successful IP
        save_pi_ip(ip)
        
        # Start video stream
        self.connected = True
        self.video_stop_event.clear()
        self.video_thread = threading.Thread(target=self.video_stream_worker, daemon=True)
        self.video_thread.start()
        
        # Update UI
        self.connect_btn.config(text="🔌 Disconnect", bg='#f44336')
        self.status_badge.config(text="● CONNECTED", fg='#4caf50')
        self.status_text.config(text=f"Connected to {ip}:{self.pi_port}")
        
        self.log(f"✓ Connected to {ip}", 'SUCCESS')
    
    def disconnect(self):
        """Disconnect from Raspberry Pi"""
        self.log("Disconnecting...")
        
        # Stop video stream
        self.video_stop_event.set()
        if self.video_thread:
            self.video_thread.join(timeout=2)
        
        self.connected = False
        self.video_running = False
        
        # Clear video display
        self.video_label.config(image='')
        
        # Update UI
        self.connect_btn.config(text="🔌 Connect", bg='#4a90e2')
        self.status_badge.config(text="● DISCONNECTED", fg='#ff4444')
        self.status_text.config(text="Disconnected")
        
        self.log("✓ Disconnected", 'SUCCESS')
        
        # Restart local camera after disconnect
        self.root.after(1000, self.check_and_start_local_camera)
    
    def video_stream_worker(self):
        """Background thread for video streaming with auto-detection"""
        ip = self.pi_ip.get()
        stream_url = f"http://{ip}:{self.pi_port}/video_feed"
        
        self.log("Starting video stream with auto-detection...")
        fps_counter = []
        last_detection_time = 0
        detection_cooldown = 0.5  # Run detection every 0.5 seconds
        
        try:
            stream = requests.get(stream_url, stream=True, timeout=5)
            bytes_data = bytes()
            
            self.video_running = True
            
            for chunk in stream.iter_content(chunk_size=4096):
                if self.video_stop_event.is_set():
                    break
                
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                b = bytes_data.find(b'\xff\xd9')  # JPEG end
                
                if a != -1 and b != -1:
                    t0 = time.time()
                    
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    # Decode frame
                    nparr = np.frombuffer(jpg, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Store current frame
                        with self.frame_lock:
                            self.current_frame = frame.copy()
                        
                        # Always run detection
                        display_frame = frame
                        current_time = time.time()
                        if current_time - last_detection_time >= detection_cooldown:
                            result = detect_target(frame)
                            display_frame = result['annotated_frame']
                            
                            # Update detection status based on Buyo detection
                            if result['is_buyo']:
                                if result['detected']:
                                    status_text = f"✓ Detected: {result['class']} ({result['confidence']*100:.1f}%)"
                                    status_color = '#4caf50'
                                    self.log(f"🍃 Buyo leaf detected: {result['class']} ({result['confidence']*100:.1f}%)", 'SUCCESS')
                                else:
                                    status_text = f"Buyo Leaf: {result['class']} ({result['confidence']*100:.1f}%)"
                                    status_color = '#ffaa00'
                            else:
                                status_text = "Scanning for Buyo leaves..."
                                status_color = '#888888'
                            
                            self.root.after(0, lambda t=status_text, c=status_color: 
                                          self.detection_status.config(text=t, fg=c))
                            
                            # Store last detection result
                            if result['is_buyo']:
                                self.last_detection = {
                                    'class': result['class'],
                                    'confidence': result['confidence']
                                }
                                # Optional: Auto-trigger arm if configured
                                # if result['detected']:
                                #     self.send_command('drop', {'arm': 1, 'class': result['class']})
                            
                            last_detection_time = current_time
                        
                        # Resize for display
                        display_frame = cv2.resize(display_frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        
                        # Convert to PIL Image
                        img = Image.fromarray(frame_rgb)
                        img_tk = ImageTk.PhotoImage(image=img)
                        
                        # Update display
                        self.video_label.config(image=img_tk)
                        self.video_label.image = img_tk
                        
                        # Calculate FPS
                        fps_counter.append(1.0 / max(0.001, time.time() - t0))
                        if len(fps_counter) > 30:
                            fps_counter.pop(0)
                        fps = np.mean(fps_counter)
                        self.root.after(0, lambda f=fps: self.fps_label.config(text=f"FPS: {f:.1f}"))
            
            self.video_running = False
            
        except Exception as e:
            self.log(f"✗ Video stream error: {str(e)}", 'ERROR')
            self.video_running = False
    
    def classify_single_image(self):
        """Open file dialog to classify a single image"""
        from tkinter import filedialog
        
        if not model_loaded:
            self.log("✗ Model not loaded", 'ERROR')
            messagebox.showerror("Model Error", "ML model is not loaded. Please check console for errors.")
            return
        
        # Open file dialog
        filepath = filedialog.askopenfilename(
            title="Select Image to Classify",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return  # User cancelled
        
        self.log(f"📂 Loading image: {os.path.basename(filepath)}")
        
        def classify():
            try:
                # Load image
                img = cv2.imread(filepath)
                if img is None:
                    self.log(f"✗ Failed to load image", 'ERROR')
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to load image file"))
                    return
                
                # Run detection
                result = detect_target(img)
                
                # Display result
                pred_class = result['class']
                confidence = result['confidence']
                
                self.log(f"🎯 Classification: {pred_class} ({confidence*100:.1f}%)", 'SUCCESS')
                
                # Show result dialog with image
                self.root.after(0, lambda: self.show_classification_result(filepath, result))
                
            except Exception as e:
                self.log(f"✗ Classification error: {str(e)}", 'ERROR')
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=classify, daemon=True).start()
    
    def classify_batch_images(self):
        """Open file dialog to classify multiple images"""
        from tkinter import filedialog
        
        if not model_loaded:
            self.log("✗ Model not loaded", 'ERROR')
            messagebox.showerror("Model Error", "ML model is not loaded. Please check console for errors.")
            return
        
        # Open file dialog for multiple files
        filepaths = filedialog.askopenfilenames(
            title="Select Images to Classify",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if not filepaths:
            return  # User cancelled
        
        self.log(f"📦 Classifying {len(filepaths)} images...")
        
        # Create progress dialog
        progress = ProgressDialog(self.root, "Batch Classification", len(filepaths))
        
        def classify_batch():
            try:
                results = []
                start_time = time.time()  # Track processing time
                
                for idx, filepath in enumerate(filepaths):
                    filename = os.path.basename(filepath)
                    progress.update(idx, f"Processing: {filename}")
                    self.log(f"[{idx+1}/{len(filepaths)}] Processing: {filename}")
                    
                    # Load image
                    img = cv2.imread(filepath)
                    if img is None:
                        self.log(f"  ✗ Failed to load", 'WARNING')
                        continue
                    
                    # Run detection
                    img_start = time.time()
                    result = detect_target(img)
                    img_time = time.time() - img_start
                    
                    pred_class = result['class']
                    confidence = result['confidence']
                    
                    self.log(f"  → {pred_class} ({confidence*100:.1f}%) [{img_time*1000:.1f}ms]")
                    
                    # Store result
                    results.append({
                        'filepath': filepath,
                        'filename': filename,
                        'class': pred_class,
                        'confidence': confidence,
                        'probabilities': result.get('probabilities', []),
                        'image': img,
                        'processing_time': img_time,
                        'is_buyo': result.get('is_buyo', False)
                    })
                
                # Final update
                progress.update(len(filepaths), "Complete!")
                time.sleep(0.5)  # Show completion briefly
                progress.close()
                
                total_time = time.time() - start_time
                self.log(f"✓ Batch classification complete: {len(results)}/{len(filepaths)} images in {total_time:.2f}s", 'SUCCESS')
                self.log(f"  Throughput: {len(results)/total_time:.1f} images/sec", 'INFO')
                
                # Show batch results dialog with timing info
                self.root.after(0, lambda: self.show_batch_classification_results(results, total_time))
                
            except Exception as e:
                progress.close()
                self.log(f"✗ Batch classification error: {str(e)}", 'ERROR')
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=classify_batch, daemon=True).start()
    
    def save_current_frame(self):
        """Save the current camera frame with classification"""
        if self.current_frame is None:
            self.log("✗ No frame to save", 'WARNING')
            messagebox.showwarning("No Frame", "No camera frame available. Please start camera first.")
            return
        
        if self.last_detection is None:
            self.log("⚠ No detection yet, saving as UNKNOWN", 'WARNING')
            class_name = 'UNKNOWN'
            confidence = 0.0
        else:
            class_name = self.last_detection['class']
            confidence = self.last_detection['confidence']
        
        self.log(f"💾 Saving frame as {class_name} ({confidence*100:.1f}%)")
        
        def save():
            try:
                # Create local directory for saved images
                save_dir = f"saved_images/{class_name}"
                os.makedirs(save_dir, exist_ok=True)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{class_name}_{timestamp}_{confidence:.3f}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                # Save frame
                cv2.imwrite(filepath, self.current_frame)
                
                self.log(f"✓ Saved: {filepath}", 'SUCCESS')
                messagebox.showinfo("Saved", f"Frame saved to:\n{filepath}")
            
            except Exception as e:
                self.log(f"✗ Save error: {str(e)}", 'ERROR')
        
        threading.Thread(target=save, daemon=True).start()
    
    def show_classification_result(self, filepath, result):
        """Show classification result in a dialog with enhanced feedback"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Classification Result - {os.path.basename(filepath)}")
        dialog.geometry("700x650")
        dialog.configure(bg='#2b2b2b')
        
        # Title
        title_text = f"🎯 Classification Result"
        title_label = tk.Label(dialog, text=title_text,
                              font=('Arial', 16, 'bold'), bg='#2b2b2b', fg='#ffffff')
        title_label.pack(pady=10)
        
        # Image preview
        try:
            img = cv2.imread(filepath)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to fit dialog
            h, w = img_rgb.shape[:2]
            max_size = 400
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            
            img_resized = cv2.resize(img_rgb, (new_w, new_h))
            pil_img = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(image=pil_img)
            
            img_label = tk.Label(dialog, image=img_tk, bg='#1e1e1e')
            img_label.image = img_tk  # Keep reference
            img_label.pack(pady=10)
        except Exception as e:
            self.log(f"Failed to load image preview: {e}", 'WARNING')
        
        # Result frame
        result_frame = tk.Frame(dialog, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
        result_frame.pack(fill=tk.X, padx=20, pady=10)
        
        pred_class = result['class']
        confidence = result['confidence']
        is_buyo = result.get('is_buyo', True)  # Default to True if not specified
        
        # Determine display based on confidence
        # Always show the predicted class unless it's explicitly "NO GREEN"
        if pred_class == 'NO GREEN':
            display_class = "NO BUYO"
            display_color = '#888888'
            status_msg = "No green leaf detected in image"
        elif confidence >= DETECTION_THRESHOLD:
            display_class = pred_class
            display_color = '#4caf50'
            status_msg = f"High Confidence - Buyo leaf: {pred_class}"
        elif confidence >= 0.30:
            display_class = pred_class
            display_color = '#ffaa00'
            status_msg = f"Medium Confidence - Likely: {pred_class}"
        else:
            display_class = pred_class
            display_color = '#ff8888'
            status_msg = f"Low Confidence - Uncertain: {pred_class}"
        
        # Status message
        status_label = tk.Label(result_frame, text=status_msg,
                               font=('Arial', 10), bg='#1e1e1e', fg='#ffffff',
                               wraplength=600)
        status_label.pack(pady=5)
        
        # Predicted class
        class_label = tk.Label(result_frame, text=f"Predicted Class: {display_class}",
                              font=('Arial', 14, 'bold'), bg='#1e1e1e', fg=display_color)
        class_label.pack(pady=5)
        
        # Confidence
        conf_label = tk.Label(result_frame, text=f"Confidence: {confidence*100:.2f}%",
                             font=('Arial', 12), bg='#1e1e1e', fg='#ffffff')
        conf_label.pack(pady=5)
        
        # Confidence interpretation
        if confidence >= 0.70:
            interp = "Very High - Strong detection"
        elif confidence >= 0.50:
            interp = "High - Good detection"
        elif confidence >= 0.30:
            interp = "Medium - Uncertain"
        else:
            interp = "Low - Weak detection"
        
        interp_label = tk.Label(result_frame, text=f"Interpretation: {interp}",
                               font=('Arial', 9), bg='#1e1e1e', fg='#888888')
        interp_label.pack(pady=2)
        
        # Probability bars
        if 'probabilities' in result:
            probs_frame = tk.Frame(result_frame, bg='#1e1e1e')
            probs_frame.pack(pady=10)
            
            tk.Label(probs_frame, text="All Class Probabilities:",
                    font=('Arial', 10, 'bold'), bg='#1e1e1e', fg='#ffffff').pack()
            
            for i, cls in enumerate(CLASS_NAMES):
                prob = result['probabilities'][i]
                
                bar_frame = tk.Frame(probs_frame, bg='#1e1e1e')
                bar_frame.pack(fill=tk.X, pady=2)
                
                tk.Label(bar_frame, text=f"{cls}:", width=10,
                        font=('Arial', 9), bg='#1e1e1e', fg='#ffffff', anchor='w').pack(side=tk.LEFT)
                
                canvas = tk.Canvas(bar_frame, width=300, height=20, bg='#2b2b2b', highlightthickness=0)
                canvas.pack(side=tk.LEFT, padx=5)
                
                bar_width = int(280 * prob)
                bar_color = display_color if i == result.get('class_idx', -1) else '#555555'
                canvas.create_rectangle(2, 2, bar_width + 2, 18, fill=bar_color, outline='')
                
                tk.Label(bar_frame, text=f"{prob*100:.1f}%",
                        font=('Arial', 9), bg='#1e1e1e', fg='#ffffff').pack(side=tk.LEFT)
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg='#2b2b2b')
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="✓ Close", command=dialog.destroy,
                 font=('Arial', 10, 'bold'), bg='#4a90e2', fg='white',
                 padx=30, pady=10).pack(side=tk.LEFT, padx=5)
        
        # Option to save with classification
        def save_with_class():
            try:
                save_dir = f"classified_images/{pred_class}"
                os.makedirs(save_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{pred_class}_{confidence:.3f}_{timestamp}_{os.path.basename(filepath)}"
                save_path = os.path.join(save_dir, filename)
                
                import shutil
                shutil.copy2(filepath, save_path)
                
                messagebox.showinfo("Saved", f"Image saved to:\n{save_path}")
                self.log(f"✓ Saved classified image: {save_path}", 'SUCCESS')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
        
        tk.Button(btn_frame, text="💾 Save to Classified Folder", command=save_with_class,
                 font=('Arial', 10), bg='#4caf50', fg='white',
                 padx=20, pady=10).pack(side=tk.LEFT, padx=5)
    
    def show_batch_classification_results(self, results, processing_time=0, metrics=None):
        """Show batch classification results in a dialog with enhanced metrics and time complexity"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Batch Classification Results - {len(results)} Images")
        dialog.geometry("1024x768")
        dialog.configure(bg='#2b2b2b')
        
        # Title
        title_label = tk.Label(dialog, text=f"📦 Batch Classification Complete - {len(results)} Images",
                              font=('Arial', 16, 'bold'), bg='#2b2b2b', fg='#ffffff')
        title_label.pack(pady=10)
        
        # ===== ENHANCED METRICS PANEL =====
        metrics_container = tk.Frame(dialog, bg='#2b2b2b')
        metrics_container.pack(fill=tk.X, padx=10, pady=5)
        
        # Left metrics - Performance
        left_metrics = tk.LabelFrame(metrics_container, text="⚡ Performance Metrics",
                                     font=('Arial', 10, 'bold'), bg='#1e1e1e', fg='#4caf50')
        left_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Calculate metrics
        total_images = len(results)
        avg_conf = sum(r['confidence'] for r in results) / total_images if results else 0
        high_conf = sum(1 for r in results if r['confidence'] > 0.7)
        medium_conf = sum(1 for r in results if 0.4 <= r['confidence'] <= 0.7)
        low_conf = sum(1 for r in results if r['confidence'] < 0.4)
        
        # Time complexity metrics
        if processing_time > 0:
            images_per_sec = total_images / processing_time
            ms_per_image = (processing_time * 1000) / total_images
            
            # Big O notation display
            tk.Label(left_metrics, 
                    text=f"Time Complexity: O(n) - Linear",
                    font=('Consolas', 9, 'bold'), bg='#1e1e1e', fg='#4caf50').pack(anchor='w', padx=10, pady=2)
            
            tk.Label(left_metrics,
                    text=f"Total Time: {processing_time:.2f}s | Throughput: {images_per_sec:.1f} img/s",
                    font=('Arial', 9), bg='#1e1e1e', fg='#ffffff').pack(anchor='w', padx=10, pady=2)
            
            tk.Label(left_metrics,
                    text=f"Avg Time/Image: {ms_per_image:.1f}ms",
                    font=('Arial', 9), bg='#1e1e1e', fg='#ffffff').pack(anchor='w', padx=10, pady=2)
        else:
            tk.Label(left_metrics,
                    text="Processing time not recorded",
                    font=('Arial', 9), bg='#1e1e1e', fg='#888888').pack(anchor='w', padx=10, pady=2)
        
        # Device information
        device_info = f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}"
        if torch.cuda.is_available():
            device_info += f" - {torch.cuda.get_device_name(0)}"
        tk.Label(left_metrics,
                text=device_info,
                font=('Arial', 9), bg='#1e1e1e', fg='#ffaa00').pack(anchor='w', padx=10, pady=2)
        
        # Right metrics - Classification Distribution
        right_metrics = tk.LabelFrame(metrics_container, text="📊 Classification Distribution",
                                      font=('Arial', 10, 'bold'), bg='#1e1e1e', fg='#4caf50')
        right_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Count by class with visual bars
        class_counts = {}
        for item in results:
            class_name = item['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Display class distribution with progress bars
        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            percentage = (count / total_images) * 100
            
            class_frame = tk.Frame(right_metrics, bg='#1e1e1e')
            class_frame.pack(fill=tk.X, padx=10, pady=2)
            
            tk.Label(class_frame, text=f"{cls}:",
                    font=('Arial', 9, 'bold'), bg='#1e1e1e', fg='#ffffff',
                    width=10, anchor='w').pack(side=tk.LEFT)
            
            # Visual bar
            bar_width = int((percentage / 100) * 150)
            bar_canvas = tk.Canvas(class_frame, width=150, height=15, bg='#2b2b2b', highlightthickness=0)
            bar_canvas.pack(side=tk.LEFT, padx=5)
            bar_canvas.create_rectangle(0, 0, bar_width, 15, fill='#4caf50', outline='')
            
            tk.Label(class_frame, text=f"{count} ({percentage:.1f}%)",
                    font=('Arial', 9), bg='#1e1e1e', fg='#ffffff').pack(side=tk.LEFT)
        
        # ===== CONFIDENCE BREAKDOWN =====
        confidence_frame = tk.LabelFrame(dialog, text="🎯 Confidence Analysis",
                                        font=('Arial', 10, 'bold'), bg='#1e1e1e', fg='#4caf50')
        confidence_frame.pack(fill=tk.X, padx=10, pady=5)
        
        conf_grid = tk.Frame(confidence_frame, bg='#1e1e1e')
        conf_grid.pack(padx=10, pady=5)
        
        # High confidence
        high_frame = tk.Frame(conf_grid, bg='#1e1e1e')
        high_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(high_frame, text="🟢 High (>70%)",
                font=('Arial', 9, 'bold'), bg='#1e1e1e', fg='#4caf50').pack()
        tk.Label(high_frame, text=f"{high_conf} images",
                font=('Arial', 11, 'bold'), bg='#1e1e1e', fg='#ffffff').pack()
        tk.Label(high_frame, text=f"{(high_conf/total_images*100):.1f}%",
                font=('Arial', 8), bg='#1e1e1e', fg='#888888').pack()
        
        # Medium confidence
        med_frame = tk.Frame(conf_grid, bg='#1e1e1e')
        med_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(med_frame, text="🟡 Medium (40-70%)",
                font=('Arial', 9, 'bold'), bg='#1e1e1e', fg='#ffaa00').pack()
        tk.Label(med_frame, text=f"{medium_conf} images",
                font=('Arial', 11, 'bold'), bg='#1e1e1e', fg='#ffffff').pack()
        tk.Label(med_frame, text=f"{(medium_conf/total_images*100):.1f}%",
                font=('Arial', 8), bg='#1e1e1e', fg='#888888').pack()
        
        # Low confidence
        low_frame = tk.Frame(conf_grid, bg='#1e1e1e')
        low_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(low_frame, text="🔴 Low (<40%)",
                font=('Arial', 9, 'bold'), bg='#1e1e1e', fg='#ff4444').pack()
        tk.Label(low_frame, text=f"{low_conf} images",
                font=('Arial', 11, 'bold'), bg='#1e1e1e', fg='#ffffff').pack()
        tk.Label(low_frame, text=f"{(low_conf/total_images*100):.1f}%",
                font=('Arial', 8), bg='#1e1e1e', fg='#888888').pack()
        
        # Average confidence
        avg_frame = tk.Frame(conf_grid, bg='#1e1e1e')
        avg_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(avg_frame, text="📈 Average",
                font=('Arial', 9, 'bold'), bg='#1e1e1e', fg='#ffffff').pack()
        tk.Label(avg_frame, text=f"{avg_conf*100:.1f}%",
                font=('Arial', 11, 'bold'), bg='#1e1e1e', 
                fg='#4caf50' if avg_conf > 0.7 else '#ffaa00' if avg_conf > 0.4 else '#ff4444').pack()
        tk.Label(avg_frame, text="confidence",
                font=('Arial', 8), bg='#1e1e1e', fg='#888888').pack()
        
        # Scrollable frame for results
        canvas = tk.Canvas(dialog, bg='#2b2b2b', highlightthickness=0)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#2b2b2b')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Grid of results (3 columns)
        for idx, item in enumerate(results):
            row = idx // 3
            col = idx % 3
            
            item_frame = tk.Frame(scrollable_frame, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
            item_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            # Thumbnail
            try:
                img_resized = cv2.resize(item['image'], (150, 150))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                thumbnail_tk = ImageTk.PhotoImage(pil_img)
                
                img_label = tk.Label(item_frame, image=thumbnail_tk, bg='#1e1e1e')
                img_label.image = thumbnail_tk  # Keep reference
                img_label.pack(padx=5, pady=5)
            except:
                pass
            
            # Class label
            class_color = '#4caf50' if item['confidence'] > 0.6 else '#ffaa00' if item['confidence'] > 0.4 else '#ff4444'
            class_label = tk.Label(item_frame, 
                                   text=f"{item['class']}\n{item['confidence']*100:.1f}%",
                                   font=('Arial', 10, 'bold'), bg='#1e1e1e', fg=class_color)
            class_label.pack(pady=2)
            
            # Filename
            filename_label = tk.Label(item_frame, text=item['filename'][:25], 
                                     font=('Arial', 8), bg='#1e1e1e', fg='#888888')
            filename_label.pack(pady=2)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg='#2b2b2b')
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="✓ Close", command=dialog.destroy,
                 font=('Arial', 10, 'bold'), bg='#4a90e2', fg='white',
                 padx=30, pady=10).pack(side=tk.LEFT, padx=5)
        
        # Save all to classified folders
        def save_all_classified():
            try:
                saved_count = 0
                for item in results:
                    save_dir = f"classified_images/{item['class']}"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{item['class']}_{item['confidence']:.3f}_{timestamp}_{item['filename']}"
                    save_path = os.path.join(save_dir, filename)
                    
                    import shutil
                    shutil.copy2(item['filepath'], save_path)
                    saved_count += 1
                
                messagebox.showinfo("Saved", f"Saved {saved_count} images to classified_images/ folder")
                self.log(f"✓ Saved {saved_count} classified images", 'SUCCESS')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
        
        tk.Button(btn_frame, text="💾 Save All to Classified Folders", command=save_all_classified,
                 font=('Arial', 10), bg='#4caf50', fg='white',
                 padx=20, pady=10).pack(side=tk.LEFT, padx=5)
        
        # Export to CSV
        def export_csv():
            from tkinter import filedialog
            csv_path = filedialog.asksaveasfilename(
                title="Export Results to CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if csv_path:
                try:
                    import csv
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Filename', 'Filepath', 'Predicted Class', 'Confidence'] + 
                                      [f'Prob_{cls}' for cls in CLASS_NAMES])
                        
                        for item in results:
                            row = [item['filename'], item['filepath'], item['class'], 
                                  f"{item['confidence']:.4f}"]
                            row.extend([f"{p:.4f}" for p in item['probabilities']])
                            writer.writerow(row)
                    
                    messagebox.showinfo("Exported", f"Results exported to:\n{csv_path}")
                    self.log(f"✓ Exported results to CSV: {csv_path}", 'SUCCESS')
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export: {str(e)}")
        
        tk.Button(btn_frame, text="📊 Export to CSV", command=export_csv,
                 font=('Arial', 10), bg='#ff9800', fg='white',
                 padx=20, pady=10).pack(side=tk.LEFT, padx=5)
    
    def send_command(self, action, params=None):
        """Send command to Raspberry Pi with cooldown"""
        if not self.connected:
            self.log("✗ Not connected to Raspberry Pi", 'WARNING')
            messagebox.showwarning("Not Connected", "Remote control requires Raspberry Pi connection")
            return
        
        # Check cooldown
        current_time = time.time()
        time_since_last = current_time - self.last_command_time
        if time_since_last < COMMAND_COOLDOWN:
            remaining = COMMAND_COOLDOWN - time_since_last
            self.log(f"⏳ Cooldown active: {remaining:.1f}s remaining", 'WARNING')
            self.show_cooldown(remaining)
            return
        
        self.last_command_time = current_time
        
        # Queue command for background processing
        self.command_queue.put({'action': action, 'params': params or {}})
    
    def detect_peripherals(self):
        """Detect and display connected peripherals on Raspberry Pi"""
        if not self.connected:
            self.log("✗ Not connected to Raspberry Pi", 'WARNING')
            messagebox.showwarning("Not Connected", "Peripheral detection requires Raspberry Pi connection")
            return
        
        self.log("🔌 Detecting peripherals on Raspberry Pi...")
        
        def fetch_peripherals():
            try:
                url = f"http://{self.pi_ip.get()}:{self.pi_port.get()}/peripherals"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display results in a dialog
                    self.root.after(0, lambda: self.show_peripherals_dialog(data))
                    self.log("✓ Peripheral detection complete", 'SUCCESS')
                else:
                    self.log(f"✗ Failed to detect peripherals: {response.status_code}", 'ERROR')
                    
            except requests.exceptions.Timeout:
                self.log("✗ Peripheral detection timeout", 'ERROR')
            except Exception as e:
                self.log(f"✗ Peripheral detection error: {str(e)}", 'ERROR')
        
        threading.Thread(target=fetch_peripherals, daemon=True).start()
    
    def show_peripherals_dialog(self, data):
        """Display peripheral detection results in a dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("🔌 Peripheral Detection Results")
        dialog.geometry("600x500")
        dialog.configure(bg='#2b2b2b')
        dialog.transient(self.root)
        
        # Title
        title_frame = tk.Frame(dialog, bg='#1e1e1e', pady=10)
        title_frame.pack(fill=tk.X)
        
        tk.Label(title_frame, text="🔌 Connected Peripherals",
                font=('Arial', 14, 'bold'), bg='#1e1e1e', fg='#4a90e2').pack()
        
        # Scrollable content
        canvas = tk.Canvas(dialog, bg='#2b2b2b', highlightthickness=0)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#2b2b2b')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Peripheral status items
        peripherals = [
            ("📷 Camera Module", data.get('camera', {}).get('status', 'Unknown'), 
             data.get('camera', {}).get('device', 'N/A')),
            ("⚙️ Stepper Motor", data.get('stepper', {}).get('status', 'Unknown'),
             data.get('stepper', {}).get('pins', 'N/A')),
            ("📹 MG996R Servo (Camera)", data.get('servo_camera', {}).get('status', 'Unknown'),
             data.get('servo_camera', {}).get('channel', 'N/A')),
            ("🦾 Robotic Arm Bus (0x40)", data.get('pca9685_0x40', {}).get('status', 'Unknown'),
             data.get('pca9685_0x40', {}).get('channels', 'N/A')),
            ("🦾 Robotic Arm Bus (0x41)", data.get('pca9685_0x41', {}).get('status', 'Unknown'),
             data.get('pca9685_0x41', {}).get('channels', 'N/A')),
        ]
        
        for name, status, details in peripherals:
            # Create peripheral card
            card = tk.Frame(scrollable_frame, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
            card.pack(fill=tk.X, padx=15, pady=5)
            
            # Header with status indicator
            header = tk.Frame(card, bg='#1e1e1e')
            header.pack(fill=tk.X, padx=10, pady=5)
            
            tk.Label(header, text=name, font=('Arial', 11, 'bold'),
                    bg='#1e1e1e', fg='#ffffff').pack(side=tk.LEFT)
            
            # Status color coding
            if 'online' in status.lower() or 'ready' in status.lower() or 'detected' in status.lower():
                status_color = '#4caf50'
                status_icon = '✓'
            elif 'offline' in status.lower() or 'not found' in status.lower() or 'error' in status.lower():
                status_color = '#ff4444'
                status_icon = '✗'
            else:
                status_color = '#ffaa00'
                status_icon = '⚠'
            
            tk.Label(header, text=f"{status_icon} {status}", font=('Arial', 10, 'bold'),
                    bg='#1e1e1e', fg=status_color).pack(side=tk.RIGHT)
            
            # Details
            if details != 'N/A':
                tk.Label(card, text=f"Details: {details}", font=('Arial', 9),
                        bg='#1e1e1e', fg='#888888').pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        # Summary
        summary_frame = tk.Frame(scrollable_frame, bg='#282A36', relief=tk.RIDGE, bd=2)
        summary_frame.pack(fill=tk.X, padx=15, pady=10)
        
        online_count = sum(1 for _, status, _ in peripherals 
                          if 'online' in status.lower() or 'ready' in status.lower() or 'detected' in status.lower())
        total_count = len(peripherals)
        
        tk.Label(summary_frame, text=f"📊 System Status: {online_count}/{total_count} Peripherals Online",
                font=('Arial', 11, 'bold'), bg='#282A36', fg='#4a90e2').pack(pady=10)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Close button
        btn_frame = tk.Frame(dialog, bg='#2b2b2b')
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="✓ Close", command=dialog.destroy,
                 font=('Arial', 10, 'bold'), bg='#4a90e2', fg='white',
                 padx=30, pady=10).pack()
    
    def upload_current_frame(self):
        """Save the current frame locally with detection class"""
        if self.current_frame is None:
            self.log("✗ No frame to save", 'WARNING')
            return
        
        if self.last_detection is None:
            self.log("⚠ No detection yet, saving as UNKNOWN", 'WARNING')
            class_name = 'UNKNOWN'
            confidence = 0.0
        else:
            class_name = self.last_detection['class']
            confidence = self.last_detection['confidence']
        
        self.log(f"� Saving frame as {class_name} ({confidence*100:.1f}%)")
        
        def save():
            try:
                # Create local directory for saved images
                save_dir = f"saved_images/{class_name}"
                os.makedirs(save_dir, exist_ok=True)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{class_name}_{timestamp}_{confidence:.3f}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                # Convert and save frame
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                pil_img.save(filepath, format='JPEG', quality=95)
                
                self.log(f"✓ Saved locally: {filepath}", 'SUCCESS')
            
            except Exception as e:
                self.log(f"✗ Save error: {str(e)}", 'ERROR')
        
        threading.Thread(target=save, daemon=True).start()
    
    def upload_batch_frames(self):
        """Save batch of recent frames locally with classification review"""
        self.log("💾 Starting batch save...")
        
        if self.current_frame is None:
            self.log("✗ No frames to save", 'WARNING')
            return
        
        # Collect frames to save (currently just the current frame, but can be expanded to buffer)
        frames_to_save = []
        
        # If we have detection, use it; otherwise classify the frame now
        if self.last_detection is None and model_loaded:
            self.log("⏳ Running detection for batch...", 'INFO')
            # Run detection on current frame
            result = detect_target(self.current_frame)
            if result['detected']:
                frames_to_save.append({
                    'frame': self.current_frame.copy(),
                    'class': result['class'],
                    'confidence': result['confidence']
                })
            else:
                frames_to_save.append({
                    'frame': self.current_frame.copy(),
                    'class': 'UNKNOWN',
                    'confidence': 0.0
                })
        elif self.last_detection:
            frames_to_save.append({
                'frame': self.current_frame.copy(),
                'class': self.last_detection['class'],
                'confidence': self.last_detection['confidence']
            })
        else:
            frames_to_save.append({
                'frame': self.current_frame.copy(),
                'class': 'UNKNOWN',
                'confidence': 0.0
            })
        
        def save_batch():
            try:
                # Create local directory
                batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
                saved_items = []
                
                for idx, item in enumerate(frames_to_save[:100]):  # Max 100 photos
                    class_name = item['class']
                    confidence = item['confidence']
                    
                    save_dir = f"saved_images/{class_name}/batch_{batch_id}"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Save frame
                    frame_rgb = cv2.cvtColor(item['frame'], cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    
                    filename = f"{class_name}_{confidence:.3f}_{idx:03d}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    pil_img.save(filepath, format='JPEG', quality=95)
                    
                    saved_items.append({
                        'filename': filename,
                        'class': class_name,
                        'confidence': confidence,
                        'path': filepath,
                        'thumbnail': pil_img.resize((100, 100), Image.Resampling.LANCZOS)
                    })
                
                self.log(f"✓ Batch saved: {len(saved_items)} image(s) to {save_dir}/", 'SUCCESS')
                
                # Show review dialog on main thread
                self.root.after(0, lambda: self.show_batch_review_dialog(saved_items, batch_id))
            
            except Exception as e:
                self.log(f"✗ Batch save error: {str(e)}", 'ERROR')
        
        threading.Thread(target=save_batch, daemon=True).start()
    
    def show_batch_review_dialog(self, saved_items, batch_id):
        """Show a dialog reviewing the batch upload with thumbnails and classifications"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Batch Upload Review - {batch_id}")
        dialog.geometry("800x600")
        dialog.configure(bg='#2b2b2b')
        
        # Title
        title_label = tk.Label(dialog, text=f"📦 Batch Upload Complete - {len(saved_items)} Images",
                              font=('Arial', 14, 'bold'), bg='#2b2b2b', fg='#ffffff')
        title_label.pack(pady=10)
        
        # Stats summary
        stats_frame = tk.Frame(dialog, bg='#1e1e1e')
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Count by class
        class_counts = {}
        for item in saved_items:
            class_name = item['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        stats_text = "Distribution: " + " | ".join([f"{cls}: {cnt}" for cls, cnt in sorted(class_counts.items())])
        tk.Label(stats_frame, text=stats_text, font=('Arial', 10), 
                bg='#1e1e1e', fg='#4caf50').pack(pady=5)
        
        # Scrollable frame for images
        canvas = tk.Canvas(dialog, bg='#2b2b2b', highlightthickness=0)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#2b2b2b')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Grid of images (4 columns)
        for idx, item in enumerate(saved_items):
            row = idx // 4
            col = idx % 4
            
            item_frame = tk.Frame(scrollable_frame, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
            item_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            # Thumbnail
            thumbnail_tk = ImageTk.PhotoImage(item['thumbnail'])
            img_label = tk.Label(item_frame, image=thumbnail_tk, bg='#1e1e1e')
            img_label.image = thumbnail_tk  # Keep reference
            img_label.pack(padx=5, pady=5)
            
            # Class label
            class_color = '#4caf50' if item['confidence'] > 0.6 else '#ffaa00' if item['confidence'] > 0.4 else '#ff4444'
            class_label = tk.Label(item_frame, 
                                   text=f"{item['class']}\n{item['confidence']*100:.1f}%",
                                   font=('Arial', 8, 'bold'), bg='#1e1e1e', fg=class_color)
            class_label.pack(pady=2)
            
            # Filename
            filename_label = tk.Label(item_frame, text=item['filename'][:20], 
                                     font=('Arial', 7), bg='#1e1e1e', fg='#888888')
            filename_label.pack(pady=2)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Close button
        close_btn = tk.Button(dialog, text="✓ Close", command=dialog.destroy,
                             font=('Arial', 10, 'bold'), bg='#4a90e2', fg='white',
                             padx=30, pady=10)
        close_btn.pack(pady=10)
    
    def start_command_processor(self):
        """Background thread for processing commands"""
        def processor():
            while True:
                try:
                    cmd = self.command_queue.get(timeout=0.5)
                    self.execute_command(cmd['action'], cmd['params'])
                except queue.Empty:
                    pass
        
        threading.Thread(target=processor, daemon=True).start()
    
    def execute_command(self, action, params):
        """Execute command (runs in background thread)"""
        ip = self.pi_ip.get()
        url = f"http://{ip}:{self.pi_port}/command"
        
        self.log(f"→ Sending: {action} {params}")
        
        try:
            response = requests.post(url, json={'action': action, 'params': params}, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                result_msg = data.get('result', 'OK')
                self.log(f"✓ Response: {result_msg}", 'SUCCESS')
            else:
                self.log(f"✗ Command failed: HTTP {response.status_code}", 'ERROR')
        
        except Exception as e:
            self.log(f"✗ Command error: {str(e)}", 'ERROR')
    
    def show_cooldown(self, remaining):
        """Show cooldown indicator"""
        self.cooldown_label.config(text=f"⏳ Cooldown: {remaining:.1f}s", fg='#ffaa00')
        
        def update_cooldown(time_left):
            if time_left > 0:
                self.cooldown_label.config(text=f"⏳ Cooldown: {time_left:.1f}s")
                self.root.after(100, lambda: update_cooldown(time_left - 0.1))
            else:
                self.cooldown_label.config(text="✓ Ready", fg='#4caf50')
                self.root.after(2000, lambda: self.cooldown_label.config(text=""))
        
        update_cooldown(remaining)

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("="*70)
    print("  Buyo Classifier - Laptop Client")
    print("="*70)
    print(f"  Device: {device}")
    print("="*70)
    
    # Detect connected peripherals
    print("\n" + "="*70)
    print("  CONNECTED PERIPHERALS")
    print("="*70)
    
    # System info
    print("\n📋 System Information:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Machine: {platform.machine()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  PyTorch: {torch.__version__}")
    
    # GPU/CUDA Detection
    print(f"\n🎮 GPU & Acceleration:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  ✓ Using GPU acceleration")
    else:
        print(f"  Device: CPU only")
        print(f"  ⚠ GPU acceleration not available")
        if 'cpu' in torch.__version__:
            print(f"  📝 You have CPU-only PyTorch installed")
            print(f"  💡 To enable GPU: Run install_pytorch_cuda.bat")
    
    # Camera detection
    print("\n📷 Camera Detection:")
    available_cameras = []
    
    # Suppress OpenCV warnings during camera detection
    import warnings
    import sys
    import os
    
    # Redirect stderr temporarily to suppress OpenCV camera detection warnings
    stderr_backup = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            available_cameras.append(i)
            cap.release()
    
    # Restore stderr
    sys.stderr.close()
    sys.stderr = stderr_backup
    
    # Now print results
    if available_cameras:
        for cam_id in available_cameras:
            cap = cv2.VideoCapture(cam_id)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(f"  ✓ Camera {cam_id}: {width}x{height} @ {fps}fps")
            cap.release()
        print(f"  Total: {len(available_cameras)} camera(s) available")
    else:
        print("  ✗ No local cameras detected")
    
    # Storage info
    print("\n💾 Storage Information:")
    try:
        import shutil
        total, used, free = shutil.disk_usage(os.getcwd())
        print(f"  Working directory: {os.getcwd()}")
        print(f"  Total: {total // (2**30)} GB")
        print(f"  Used: {used // (2**30)} GB")
        print(f"  Free: {free // (2**30)} GB")
        
        # Create save directory
        os.makedirs("saved_images", exist_ok=True)
        print(f"  ✓ Image save directory: saved_images/")
    except Exception as e:
        print(f"  ⚠ Could not get storage info: {e}")
    
    # Network info
    print("\n🌐 Network Information:")
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"  Hostname: {hostname}")
        print(f"  Local IP: {local_ip}")
    except Exception as e:
        print(f"  ⚠ Could not get network info: {e}")
    
    # Python packages
    print("\n📦 Key Python Packages:")
    packages_to_check = {
        'torch': torch,
        'cv2': cv2,
        'PIL': Image,
        'numpy': np,
        'requests': requests,
    }
    for name, module in packages_to_check.items():
        version = getattr(module, '__version__', 'installed')
        print(f"  ✓ {name}: {version}")
    
    # Model file check
    print("\n🤖 ML Model:")
    if os.path.exists(MODEL_PATH):
        model_size = os.path.getsize(MODEL_PATH) / (1024*1024)
        print(f"  ✓ Model file found: {MODEL_PATH}")
        print(f"  Size: {model_size:.1f} MB")
    else:
        print(f"  ✗ Model file not found: {MODEL_PATH}")
        print(f"  Current directory: {os.getcwd()}")
    
    print("\n" + "="*70)
    print("  INITIALIZATION")
    print("="*70)
    
    print("\n🚀 Starting GUI...")
    # Start GUI
    root = tk.Tk()
    app = LaptopClientGUI(root)
    
    # Load model in background AFTER app is created
    def load_model_async():
        print("\n" + "="*70)
        print("  LOADING ML MODEL")
        print("="*70)
        
        if load_model():
            print("\n✅ ML MODEL READY - AUTO-DETECTION ACTIVE")
            app.root.after(0, lambda: app.model_status_label.config(
                text="Model: ✅ READY", fg='#4caf50'))
            app.root.after(0, lambda: app.status_text.config(
                text="Model ready - Auto-detection active"))
            app.root.after(0, lambda: app.log("Model loaded successfully - Auto-detection active", 'SUCCESS'))
        else:
            print("\n❌ ML MODEL FAILED - DETECTION UNAVAILABLE")
            app.root.after(0, lambda: app.model_status_label.config(
                text="Model: ❌ FAILED", fg='#ff4444'))
            app.root.after(0, lambda: app.status_text.config(
                text="Model failed to load - check console for errors"))
            app.root.after(0, lambda: app.log("Model loading failed - check console", 'ERROR'))
    
    threading.Thread(target=load_model_async, daemon=True).start()
    
    root.mainloop()
