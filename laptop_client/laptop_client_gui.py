#!/usr/bin/env python3
"""
Laptop Client GUI - Buyo Leaf Classifier
========================================
Full GUI client application for laptop with:
- Connect to Raspberry Pi server
- Live MJPEG video feed display
- Vision detection with ML model
- Manual control buttons
- Activity logging
- Non-blocking networking
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

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_PI_IP = "192.168.1.100"
DEFAULT_PORT = 5000
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
COMMAND_COOLDOWN = 2.0  # seconds between commands

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
DETECTION_THRESHOLD = 0.60
BUYO_CONFIDENCE_MIN = 0.40

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
            print(f"   cd ~/Desktop/Buyo-Leaf-Classifier")
            print(f"   python3 laptop_client/laptop_client_gui.py")
            return False
        
        print(f"Loading model from {MODEL_PATH}...")
        print(f"  File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")
        
        ckpt = torch.load(MODEL_PATH, map_location=device)
        backbone = ckpt.get('backbone', 'efficientnet_v2_s')
        print(f"  Backbone: {backbone}")
        
        # Build model architecture
        if 'convnext' in backbone:
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
            model = models.resnet50(weights=None)
            in_f = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.35), nn.Linear(in_f, 512), nn.ReLU(),
                nn.BatchNorm1d(512), nn.Dropout(p=0.20),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.15),
                nn.Linear(256, 5),
            )
        else:  # efficientnet
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
        
        print(f"  Loading state dict...")
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(device)
        model.eval()
        print(f"  Model moved to {device}")
        
        # Setup preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        model_loaded = True
        print(f"\n✓ Model loaded successfully!")
        print(f"  Classes: {CLASS_NAMES}")
        print(f"  Threshold: {DETECTION_THRESHOLD*100}%")
        return True
        
    except Exception as e:
        print(f"\n✗ Model loading failed!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def detect_target(frame):
    """
    Vision detection placeholder - detect buyo leaves in frame
    
    ⚠️ PLUG IN YOUR AI MODEL (YOLO/OpenCV) HERE ⚠️
    
    Args:
        frame: OpenCV BGR image (numpy array)
    
    Returns:
        dict with detection results:
        {
            'detected': bool,
            'class': str,
            'confidence': float,
            'bbox': (x1, y1, x2, y2) or None,
            'annotated_frame': numpy array
        }
    """
    if not model_loaded or model is None:
        return {
            'detected': False,
            'class': 'MODEL NOT LOADED',
            'confidence': 0.0,
            'bbox': None,
            'annotated_frame': frame
        }
    
    try:
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
        
        # Check if confident detection
        detected = confidence >= DETECTION_THRESHOLD
        
        # Annotate frame
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw detection info
        color = (0, 255, 0) if detected else (100, 100, 255)
        status = "DETECTED" if detected else "SCANNING"
        
        cv2.putText(annotated, f"{status}: {pred_class}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(annotated, f"Confidence: {confidence*100:.1f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw probability bars
        bar_y = 90
        for i, cls in enumerate(CLASS_NAMES):
            prob = probs[i]
            bar_w = int(200 * prob)
            cv2.rectangle(annotated, (10, bar_y + i*25), (10 + bar_w, bar_y + i*25 + 15),
                         color if i == pred_idx else (200, 200, 200), -1)
            cv2.putText(annotated, f"{cls}: {prob*100:.0f}%", (220, bar_y + i*25 + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return {
            'detected': detected,
            'class': pred_class,
            'class_idx': pred_idx,
            'confidence': confidence,
            'probabilities': probs.tolist(),
            'bbox': None,  # Add bounding box if using object detection
            'annotated_frame': annotated
        }
        
    except Exception as e:
        print(f"Detection error: {e}")
        return {
            'detected': False,
            'class': 'ERROR',
            'confidence': 0.0,
            'bbox': None,
            'annotated_frame': frame
        }

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
        self.pi_ip = tk.StringVar(value=DEFAULT_PI_IP)
        self.pi_port = DEFAULT_PORT
        self.connected = False
        self.video_running = False
        self.detection_enabled = tk.BooleanVar(value=False)
        
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
        
        # Detection toggle
        detection_frame = tk.Frame(video_frame, bg='#1e1e1e')
        detection_frame.pack(pady=5)
        
        self.detection_check = tk.Checkbutton(detection_frame, text="🎯 Enable Vision Detection",
                                             variable=self.detection_enabled,
                                             font=('Arial', 10, 'bold'),
                                             bg='#1e1e1e', fg='#ffffff',
                                             selectcolor='#2b2b2b',
                                             activebackground='#1e1e1e',
                                             activeforeground='#ffffff')
        self.detection_check.pack()
        
        self.detection_status = tk.Label(detection_frame, text="Detection: Disabled",
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
        
        # Remote control buttons (Pi only)
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
        
        self.model_status_label = tk.Label(status_bar, text="Model: Loading...",
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
        """Background worker for local camera feed"""
        self.log("Local camera worker started...")
        fps_counter = []
        last_detection_time = 0
        detection_cooldown = 1.0
        
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
                
                # Run detection if enabled
                display_frame = frame
                if self.detection_enabled.get():
                    current_time = time.time()
                    if current_time - last_detection_time >= detection_cooldown:
                        result = detect_target(frame)
                        display_frame = result['annotated_frame']
                        
                        # Update detection status
                        status_text = f"Detection: {result['class']} ({result['confidence']*100:.1f}%)"
                        status_color = '#00ff00' if result['detected'] else '#888888'
                        self.root.after(0, lambda: self.detection_status.config(
                            text=status_text, fg=status_color))
                        
                        # Store last detection result
                        if result['detected']:
                            self.last_detection = {
                                'class': result['class'],
                                'confidence': result['confidence']
                            }
                            self.log(f"🎯 Detected: {result['class']} ({result['confidence']*100:.1f}%)", 'SUCCESS')
                        
                        last_detection_time = current_time
                else:
                    self.root.after(0, lambda: self.detection_status.config(
                        text="Detection: Disabled", fg='#888888'))
                
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
        """Connect to Raspberry Pi and start video stream"""
        ip = self.pi_ip.get()
        self.log(f"Connecting to {ip}:{self.pi_port}...")
        
        # Stop local camera if running
        if self.use_local_camera:
            self.log("Stopping local camera...")
            self.stop_local_camera()
        
        # Test connection first
        try:
            response = requests.get(f"http://{ip}:{self.pi_port}/status", timeout=3)
            if response.status_code != 200:
                self.log("✗ Connection failed: Server not responding", 'ERROR')
                messagebox.showerror("Connection Error", "Cannot reach Raspberry Pi server")
                # Restart local camera
                self.root.after(1000, self.check_and_start_local_camera)
                return
        except Exception as e:
            self.log(f"✗ Connection failed: {str(e)}", 'ERROR')
            messagebox.showerror("Connection Error", f"Cannot reach Raspberry Pi:\n{str(e)}")
            # Restart local camera
            self.root.after(1000, self.check_and_start_local_camera)
            return
        
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
        """Background thread for video streaming with optional detection"""
        ip = self.pi_ip.get()
        stream_url = f"http://{ip}:{self.pi_port}/video_feed"
        
        self.log("Starting video stream...")
        fps_counter = []
        last_detection_time = 0
        detection_cooldown = 1.0  # Run detection every 1 second
        
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
                        
                        # Run detection if enabled
                        display_frame = frame
                        if self.detection_enabled.get():
                            current_time = time.time()
                            if current_time - last_detection_time >= detection_cooldown:
                                result = detect_target(frame)
                                display_frame = result['annotated_frame']
                                
                                # Update detection status
                                status_text = f"Detection: {result['class']} ({result['confidence']*100:.1f}%)"
                                status_color = '#00ff00' if result['detected'] else '#888888'
                                self.root.after(0, lambda: self.detection_status.config(
                                    text=status_text, fg=status_color))
                                
                                # Store last detection result
                                if result['detected']:
                                    self.last_detection = {
                                        'class': result['class'],
                                        'confidence': result['confidence']
                                    }
                                    self.log(f"🎯 Detected: {result['class']} ({result['confidence']*100:.1f}%)", 'SUCCESS')
                                    # Auto-trigger arm if configured
                                    # self.send_command('drop', {'arm': 1, 'class': result['class']})
                                
                                last_detection_time = current_time
                        else:
                            self.root.after(0, lambda: self.detection_status.config(
                                text="Detection: Disabled", fg='#888888'))
                        
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
        
        def classify_batch():
            try:
                results = []
                
                for idx, filepath in enumerate(filepaths):
                    self.log(f"[{idx+1}/{len(filepaths)}] Processing: {os.path.basename(filepath)}")
                    
                    # Load image
                    img = cv2.imread(filepath)
                    if img is None:
                        self.log(f"  ✗ Failed to load", 'WARNING')
                        continue
                    
                    # Run detection
                    result = detect_target(img)
                    
                    pred_class = result['class']
                    confidence = result['confidence']
                    
                    self.log(f"  → {pred_class} ({confidence*100:.1f}%)")
                    
                    # Store result
                    results.append({
                        'filepath': filepath,
                        'filename': os.path.basename(filepath),
                        'class': pred_class,
                        'confidence': confidence,
                        'probabilities': result.get('probabilities', []),
                        'image': img
                    })
                
                self.log(f"✓ Batch classification complete: {len(results)}/{len(filepaths)} images", 'SUCCESS')
                
                # Show batch results dialog
                self.root.after(0, lambda: self.show_batch_classification_results(results))
                
            except Exception as e:
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
        """Show classification result in a dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Classification Result - {os.path.basename(filepath)}")
        dialog.geometry("700x600")
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
        
        # Predicted class
        color = '#4caf50' if confidence > 0.6 else '#ffaa00' if confidence > 0.4 else '#ff4444'
        class_label = tk.Label(result_frame, text=f"Predicted Class: {pred_class}",
                              font=('Arial', 14, 'bold'), bg='#1e1e1e', fg=color)
        class_label.pack(pady=5)
        
        # Confidence
        conf_label = tk.Label(result_frame, text=f"Confidence: {confidence*100:.2f}%",
                             font=('Arial', 12), bg='#1e1e1e', fg='#ffffff')
        conf_label.pack(pady=5)
        
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
                bar_color = color if i == result.get('class_idx', -1) else '#555555'
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
    
    def show_batch_classification_results(self, results):
        """Show batch classification results in a dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Batch Classification Results - {len(results)} Images")
        dialog.geometry("900x700")
        dialog.configure(bg='#2b2b2b')
        
        # Title
        title_label = tk.Label(dialog, text=f"📦 Batch Classification Complete - {len(results)} Images",
                              font=('Arial', 14, 'bold'), bg='#2b2b2b', fg='#ffffff')
        title_label.pack(pady=10)
        
        # Stats summary
        stats_frame = tk.Frame(dialog, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Count by class
        class_counts = {}
        for item in results:
            class_name = item['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        stats_text = "Distribution: " + " | ".join([f"{cls}: {cnt}" for cls, cnt in sorted(class_counts.items())])
        tk.Label(stats_frame, text=stats_text, font=('Arial', 10), 
                bg='#1e1e1e', fg='#4caf50').pack(pady=5)
        
        # Average confidence
        avg_conf = sum(r['confidence'] for r in results) / len(results) if results else 0
        tk.Label(stats_frame, text=f"Average Confidence: {avg_conf*100:.1f}%",
                font=('Arial', 10), bg='#1e1e1e', fg='#ffffff').pack(pady=5)
        
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
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            available_cameras.append(i)
            print(f"  ✓ Camera {i}: {width}x{height} @ {fps}fps")
            cap.release()
    
    if not available_cameras:
        print("  ✗ No local cameras detected")
    else:
        print(f"  Total: {len(available_cameras)} camera(s) available")
    
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
    
    # Load model in background
    def load_model_async():
        global app
        if load_model():
            print("✓ ML model loaded - detection ready")
            if 'app' in globals():
                app.root.after(0, lambda: app.model_status_label.config(
                    text="Model: ✓ LOADED", fg='#4caf50'))
                app.root.after(0, lambda: app.status_text.config(
                    text="Model loaded - detection ready"))
        else:
            print("⚠ ML model not loaded - detection unavailable")
            if 'app' in globals():
                app.root.after(0, lambda: app.model_status_label.config(
                    text="Model: ✗ NOT LOADED", fg='#ff4444'))
                app.root.after(0, lambda: app.status_text.config(
                    text="Model failed to load - check console for errors"))
    
    threading.Thread(target=load_model_async, daemon=True).start()
    
    print("\n🚀 Starting GUI...")
    # Start GUI
    root = tk.Tk()
    app = LaptopClientGUI(root)
    root.mainloop()
