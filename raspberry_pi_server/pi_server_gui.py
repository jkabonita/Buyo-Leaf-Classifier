#!/usr/bin/env python3
"""
Raspberry Pi Server GUI - Buyo Leaf Classifier
==============================================
Full GUI server application for Raspberry Pi with:
- Live camera streaming
- Robotic arm control
- Command logging
- Easy IP display and connection management
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import socket
import time
from datetime import datetime
from flask import Flask, Response, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageTk
import io
import queue

# Try to import picamera2 for Pi Camera Module support
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

# Try to import robotic arm libraries
try:
    import RPi.GPIO as GPIO
    from adafruit_servokit import ServoKit
    ARM_AVAILABLE = True
except ImportError:
    ARM_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
SERVER_PORT = 5000
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 240

# Robotic Arm Configuration
ARM_CHANNELS = {1: 3, 2: 2, 3: 1, 4: 0, 5: 8, 6: 7, 7: 6, 8: 5, 9: 4}
ARM_DROP_ANGLE = 60
ARM_HOME_ANGLE = 0
CURRENT_ARM = 1

# ============================================================================
# FLASK SERVER
# ============================================================================
app = Flask(__name__)
camera = None
camera_lock = threading.Lock()
command_queue = queue.Queue()
kit = None

def detect_available_cameras():
    """Detect which cameras are available"""
    available = []
    
    # Check for Picamera2 (libcamera) - Pi Camera Module
    if PICAMERA2_AVAILABLE:
        try:
            from picamera2 import Picamera2
            cameras = Picamera2.global_camera_info()
            if cameras:
                for i, cam_info in enumerate(cameras):
                    model = cam_info.get('Model', 'Unknown')
                    location = cam_info.get('Location', '')
                    available.append(f"Pi Camera #{i}: {model} ({location})")
                return available  # Return early if Pi cameras found
        except Exception as e:
            print(f"Picamera2 detection error: {e}")
    
    # Only check for USB webcams if no Pi Camera found
    # Check only /dev/video0 to avoid spam
    try:
        import os
        if os.path.exists('/dev/video0'):
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if cap.isOpened():
                available.append("USB Webcam /dev/video0")
                cap.release()
    except Exception as e:
        print(f"USB webcam check error: {e}")
    
    return available

def init_camera():
    """Initialize camera (picamera2 or USB webcam)"""
    global camera
    
    print("🔍 Detecting available cameras...")
    available_cameras = detect_available_cameras()
    
    if available_cameras:
        print(f"📸 Found {len(available_cameras)} camera(s):")
        for cam in available_cameras:
            print(f"   - {cam}")
    else:
        print("⚠️  No cameras detected!")
        return None
    
    # Try Picamera2 first (Pi Camera Module)
    if PICAMERA2_AVAILABLE:
        try:
            print("🔄 Initializing Picamera2...")
            from picamera2 import Picamera2
            
            # Get available cameras
            cameras = Picamera2.global_camera_info()
            if not cameras:
                print("✗ No Picamera2 cameras found!")
            else:
                print(f"📸 Found {len(cameras)} Pi Camera(s)")
                for i, cam_info in enumerate(cameras):
                    print(f"   Camera {i}: {cam_info}")
                
                # Initialize first camera
                camera = Picamera2()
                cam_model = cameras[0].get('Model', 'Unknown')
                
                # Configure for streaming
                config = camera.create_video_configuration(
                    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
                )
                camera.configure(config)
                camera.start()
                
                print("⏳ Waiting for camera to stabilize...")
                time.sleep(2)
                
                # Test capture
                test_frame = camera.capture_array()
                if test_frame is not None and test_frame.size > 0:
                    print(f"✓ Camera ready: {test_frame.shape}")
                    return f"Pi Camera ({cam_model})"
                else:
                    print("✗ Camera test capture failed")
                    camera.stop()
                    camera.close()
                    camera = None
        except Exception as e:
            print(f"✗ Picamera2 initialization failed: {e}")
            import traceback
            traceback.print_exc()
            if camera:
                try:
                    camera.stop()
                    camera.close()
                except:
                    pass
            camera = None
    
    # Fallback to USB webcam
    print("🔄 Trying USB webcam...")
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("✗ Could not open /dev/video0")
            return None
            
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        time.sleep(1)
        
        # Test capture
        ret, test_frame = camera.read()
        if ret and test_frame is not None:
            print(f"✓ USB webcam ready: {test_frame.shape}")
            return "USB Webcam"
        else:
            print("✗ Could not capture from USB webcam")
            return None
    except Exception as e:
        print(f"✗ USB webcam failed: {e}")
        return None

def get_frame():
    """Capture frame from camera"""
    global camera
    
    with camera_lock:
        if camera is None:
            return None
        
        try:
            if PICAMERA2_AVAILABLE and isinstance(camera, Picamera2):
                frame_rgb = camera.capture_array()
                return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = camera.read()
                return frame if ret else None
        except Exception as e:
            print(f"Frame capture error: {e}")
            return None

def generate_frames():
    """Generate MJPEG stream"""
    while True:
        frame = get_frame()
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint"""
    return jsonify({
        'status': 'running',
        'camera': 'active' if camera else 'inactive',
        'arm': 'available' if ARM_AVAILABLE else 'unavailable'
    })

@app.route('/command', methods=['POST'])
def command():
    """Command endpoint - receives commands from laptop"""
    try:
        data = request.json
        action = data.get('action', '')
        params = data.get('params', {})
        
        # Add to command queue for GUI logging
        command_queue.put({
            'time': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'params': params
        })
        
        # Execute arm action
        result = execute_arm_action(action, params)
        
        return jsonify({'status': 'success', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# ============================================================================
# ROBOTIC ARM CONTROL
# ============================================================================
def init_arm():
    """Initialize robotic arm"""
    global kit
    
    if not ARM_AVAILABLE:
        return False
    
    try:
        kit = ServoKit(channels=16, address=0x40)
        return True
    except Exception as e:
        print(f"Arm init failed: {e}")
        return False

def execute_arm_action(action, params=None):
    """
    Execute robotic arm action
    
    ⚠️ PLUG IN YOUR SERVO/MOTOR LIBRARY CODE HERE ⚠️
    
    Args:
        action: Action command (e.g., 'grab', 'drop', 'home')
        params: Additional parameters (e.g., {'arm': 1, 'class': 'CLASS A'})
    
    Returns:
        Result message
    """
    global kit
    
    if params is None:
        params = {}
    
    arm_num = params.get('arm', CURRENT_ARM)
    buyo_class = params.get('class', 'UNKNOWN')
    
    # Simulation mode if no hardware
    if not ARM_AVAILABLE or kit is None:
        return f"[SIMULATION] {action.upper()} - Arm #{arm_num} → {buyo_class}"
    
    # ===== YOUR ROBOTIC ARM CODE GOES HERE =====
    try:
        if action == 'drop':
            channel = ARM_CHANNELS.get(arm_num)
            if channel is not None:
                # Drop sequence
                kit.servo[channel].angle = ARM_DROP_ANGLE
                time.sleep(0.6)
                kit.servo[channel].angle = ARM_HOME_ANGLE
                time.sleep(0.4)
                return f"✓ Arm #{arm_num} dropped {buyo_class}"
            else:
                return f"✗ Invalid arm #{arm_num}"
        
        elif action == 'home':
            # Return all arms to home position
            for channel in ARM_CHANNELS.values():
                kit.servo[channel].angle = ARM_HOME_ANGLE
            return "✓ All arms homed"
        
        elif action == 'test':
            # Test single arm
            channel = ARM_CHANNELS.get(arm_num)
            if channel is not None:
                kit.servo[channel].angle = ARM_DROP_ANGLE
                time.sleep(0.3)
                kit.servo[channel].angle = ARM_HOME_ANGLE
                return f"✓ Arm #{arm_num} test complete"
        
        else:
            return f"⚠ Unknown action: {action}"
    
    except Exception as e:
        return f"✗ Error: {str(e)}"

# ============================================================================
# GUI APPLICATION
# ============================================================================
class PiServerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Buyo Classifier - Raspberry Pi Server")
        self.root.geometry("800x600")
        self.root.configure(bg='#2b2b2b')
        
        self.server_thread = None
        self.server_running = False
        self.preview_running = False
        self.camera_type = None
        
        self.setup_ui()
        self.start_command_logger()
        
    def setup_ui(self):
        """Setup GUI layout"""
        
        # ===== IP ADDRESS SECTION =====
        ip_frame = tk.Frame(self.root, bg='#1e1e1e', relief=tk.RAISED, bd=2)
        ip_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(ip_frame, text="Raspberry Pi IP Address:", 
                font=('Arial', 12, 'bold'), bg='#1e1e1e', fg='#ffffff').pack(pady=5)
        
        self.ip_label = tk.Label(ip_frame, text=self.get_local_ip(), 
                                font=('Arial', 24, 'bold'), bg='#1e1e1e', fg='#00ff00')
        self.ip_label.pack(pady=5)
        
        tk.Button(ip_frame, text="📋 Copy IP", command=self.copy_ip,
                 font=('Arial', 10), bg='#4a90e2', fg='white', 
                 padx=20, pady=5).pack(pady=5)
        
        # ===== MAIN CONTENT =====
        content_frame = tk.Frame(self.root, bg='#2b2b2b')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left Panel - Controls & Preview
        left_panel = tk.Frame(content_frame, bg='#2b2b2b')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Server Controls
        control_frame = tk.LabelFrame(left_panel, text="Server Controls", 
                                     font=('Arial', 10, 'bold'),
                                     bg='#1e1e1e', fg='#ffffff')
        control_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = tk.Button(control_frame, text="▶ Start Server", 
                                   command=self.toggle_server,
                                   font=('Arial', 12, 'bold'),
                                   bg='#4caf50', fg='white', padx=30, pady=10)
        self.start_btn.pack(pady=10)
        
        self.status_label = tk.Label(control_frame, text="● STOPPED", 
                                     font=('Arial', 11, 'bold'),
                                     bg='#1e1e1e', fg='#ff4444')
        self.status_label.pack(pady=5)
        
        # Camera Preview
        preview_frame = tk.LabelFrame(left_panel, text="Camera Preview",
                                     font=('Arial', 10, 'bold'),
                                     bg='#1e1e1e', fg='#ffffff')
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.preview_label = tk.Label(preview_frame, bg='#000000',
                                      width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        self.preview_label.pack(padx=5, pady=5)
        
        self.cam_status = tk.Label(preview_frame, text="Camera: Not initialized",
                                   font=('Arial', 9), bg='#1e1e1e', fg='#888888')
        self.cam_status.pack(pady=2)
        
        # Right Panel - Command Log
        right_panel = tk.Frame(content_frame, bg='#2b2b2b')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        log_frame = tk.LabelFrame(right_panel, text="Command Log",
                                 font=('Arial', 10, 'bold'),
                                 bg='#1e1e1e', fg='#ffffff')
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, 
                                                  font=('Consolas', 9),
                                                  bg='#1a1a1a', fg='#00ff00',
                                                  insertbackground='white',
                                                  height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status Bar
        status_bar = tk.Frame(self.root, bg='#1e1e1e', relief=tk.SUNKEN, bd=1)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_text = tk.Label(status_bar, text="Ready", 
                                   font=('Arial', 9), bg='#1e1e1e', fg='#888888',
                                   anchor=tk.W)
        self.status_text.pack(side=tk.LEFT, padx=10, pady=2)
        
    def get_local_ip(self):
        """Get local network IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def copy_ip(self):
        """Copy IP to clipboard"""
        ip = self.ip_label.cget("text")
        self.root.clipboard_clear()
        self.root.clipboard_append(ip)
        self.log(f"✓ IP address {ip} copied to clipboard!")
    
    def toggle_server(self):
        """Start or stop server"""
        if not self.server_running:
            self.start_server()
        else:
            self.stop_server()
    
    def start_server(self):
        """Start Flask server and camera"""
        self.log("Starting server...")
        self.log("Detecting available cameras...")
        
        # Show available cameras
        available = detect_available_cameras()
        if available:
            self.log(f"Found {len(available)} camera(s):")
            for cam in available:
                self.log(f"  • {cam}")
        else:
            self.log("⚠ No cameras detected!")
        
        # Initialize camera
        self.camera_type = init_camera()
        if self.camera_type:
            self.log(f"✓ Camera initialized: {self.camera_type}")
            self.cam_status.config(text=f"Camera: {self.camera_type}", fg='#00ff00')
        else:
            self.log("✗ Camera initialization failed!")
            self.log("Check camera connection and enable it in raspi-config")
            self.cam_status.config(text="Camera: Failed", fg='#ff4444')
        
        # Initialize arm
        if init_arm():
            self.log("✓ Robotic arm initialized")
        else:
            self.log("⚠ Robotic arm in simulation mode")
        
        # Start Flask server in background thread
        self.server_running = True
        self.server_thread = threading.Thread(target=self.run_flask, daemon=True)
        self.server_thread.start()
        
        # Update UI
        self.start_btn.config(text="⏸ Stop Server", bg='#f44336')
        self.status_label.config(text="● RUNNING", fg='#4caf50')
        self.status_text.config(text=f"Server running on {self.get_local_ip()}:{SERVER_PORT}")
        
        # Start camera preview
        self.preview_running = True
        self.update_preview()
        
        self.log(f"✓ Server started on {self.get_local_ip()}:{SERVER_PORT}")
    
    def stop_server(self):
        """Stop server and camera"""
        self.log("Stopping server...")
        
        # Stop preview
        self.preview_running = False
        
        # Stop camera
        global camera
        if camera:
            try:
                if PICAMERA2_AVAILABLE and isinstance(camera, Picamera2):
                    camera.stop()
                else:
                    camera.release()
            except:
                pass
            camera = None
        
        self.server_running = False
        
        # Update UI
        self.start_btn.config(text="▶ Start Server", bg='#4caf50')
        self.status_label.config(text="● STOPPED", fg='#ff4444')
        self.status_text.config(text="Server stopped")
        self.cam_status.config(text="Camera: Not initialized", fg='#888888')
        
        self.log("✓ Server stopped")
    
    def run_flask(self):
        """Run Flask server"""
        app.run(host='0.0.0.0', port=SERVER_PORT, debug=False, threaded=True)
    
    def update_preview(self):
        """Update camera preview"""
        if not self.preview_running:
            return
        
        frame = get_frame()
        if frame is not None:
            # Resize for preview
            frame_resized = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.preview_label.config(image=img_tk)
            self.preview_label.image = img_tk
        
        # Schedule next update
        self.root.after(50, self.update_preview)
    
    def start_command_logger(self):
        """Start command logger thread"""
        threading.Thread(target=self.process_commands, daemon=True).start()
    
    def process_commands(self):
        """Process incoming commands from queue"""
        while True:
            try:
                cmd = command_queue.get(timeout=0.5)
                self.log(f"[{cmd['time']}] Received: {cmd['action']} {cmd.get('params', '')}")
            except queue.Empty:
                pass
    
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message}\n"
        
        def update_log():
            self.log_text.insert(tk.END, log_msg)
            self.log_text.see(tk.END)
        
        if threading.current_thread() is threading.main_thread():
            update_log()
        else:
            self.root.after(0, update_log)

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    root = tk.Tk()
    app_gui = PiServerGUI(root)
    root.mainloop()
