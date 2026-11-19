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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.35, inplace=True),  # 0
                nn.Linear(in_f, 512),                # 1
                nn.SiLU(),                           # 2
                nn.BatchNorm1d(512),                 # 3
                nn.Dropout(p=0.25),                  # 4
                nn.Linear(512, 256),                 # 5
                nn.SiLU(),                           # 6
                nn.Dropout(p=0.15),                  # 7
                nn.Linear(256, 5),                   # 8
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
        
        # Cooldown tracking
        self.last_command_time = 0
        
        # Threading
        self.video_thread = None
        self.video_stop_event = threading.Event()
        self.command_queue = queue.Queue()
        
        # Current frame
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        self.setup_ui()
        self.start_command_processor()
        
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
        
        buttons = [
            ("📤 Batch Upload", lambda: self.send_command('batch_upload'), 0, 0, 2),
            ("📷 Single Upload", lambda: self.send_command('single_upload'), 1, 0, 2),
            ("🦾 Test Arm #1", lambda: self.send_command('test', {'arm': 1}), 2, 0, 1),
            ("🦾 Test Arm #2", lambda: self.send_command('test', {'arm': 2}), 2, 1, 1),
            ("🏠 Home All", lambda: self.send_command('home'), 3, 0, 2),
            ("💧 Drop Arm #1", lambda: self.send_command('drop', {'arm': 1, 'class': 'CLASS A'}), 4, 0, 1),
            ("💧 Drop Arm #2", lambda: self.send_command('drop', {'arm': 2, 'class': 'CLASS B'}), 4, 1, 1),
        ]
        
        for text, cmd, row, col, span in buttons:
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
        
        # Test connection first
        try:
            response = requests.get(f"http://{ip}:{self.pi_port}/status", timeout=3)
            if response.status_code != 200:
                self.log("✗ Connection failed: Server not responding", 'ERROR')
                messagebox.showerror("Connection Error", "Cannot reach Raspberry Pi server")
                return
        except Exception as e:
            self.log(f"✗ Connection failed: {str(e)}", 'ERROR')
            messagebox.showerror("Connection Error", f"Cannot reach Raspberry Pi:\n{str(e)}")
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
                                
                                # Log detection
                                if result['detected']:
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
    
    def send_command(self, action, params=None):
        """Send command to Raspberry Pi with cooldown"""
        if not self.connected:
            self.log("✗ Not connected to Raspberry Pi", 'WARNING')
            messagebox.showwarning("Not Connected", "Please connect to Raspberry Pi first")
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
    print("="*64)
    print("  Buyo Classifier - Laptop Client")
    print("="*64)
    print(f"  Device: {device}")
    print("="*64)
    
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
    
    # Start GUI
    root = tk.Tk()
    app = LaptopClientGUI(root)
    root.mainloop()
