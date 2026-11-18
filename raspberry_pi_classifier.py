"""
Raspberry Pi Buyo Leaf Grade Classifier with Robotic Arm Integration
=====================================================================
Optimized for Raspberry Pi with automatic model detection
Classifies Buyo leaves into 5 quality grades: Class A through Class E

Features:
  - Automatic .pth model detection
  - Threaded inference for smooth performance
  - Temporal smoothing for stable predictions
  - Upload button for image classification
  - Robotic arm trigger on confident classification
  - Lightweight GUI optimized for Raspberry Pi

Controls:
  Q - quit
  S - save screenshot
  U - upload and classify a photo
  R - reset smoothing buffer
"""

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from collections import deque
import os
import time
import threading
import queue
import glob
import tkinter as tk
from tkinter import filedialog

# ============================================================================
#  ROBOTIC ARM HARDWARE INTEGRATION (Optional)
# ============================================================================
# Set to True if robotic arm hardware is connected
ENABLE_ROBOTIC_ARM = False

try:
    import RPi.GPIO as GPIO
    from adafruit_servokit import ServoKit
    ENABLE_ROBOTIC_ARM = True
except ImportError:
    print("[INFO] Robotic arm libraries not available (RPi.GPIO or ServoKit)")
    print("      Classifier will run in camera-only mode")

# Robotic Arm Configuration
ARM_CHANNELS = {
    1: 3, 2: 2, 3: 1, 4: 0, 5: 8,
    6: 7, 7: 6, 8: 5, 9: 4,
}
ARM_DROP_ANGLE = 60   # Extend/release angle
ARM_HOME_ANGLE = 0    # Retract angle
CURRENT_ARM = 1       # Which robotic arm is currently active

# Initialize hardware
kit = None
if ENABLE_ROBOTIC_ARM:
    try:
        kit = ServoKit(channels=16, address=0x40)
        print("[ROBOTIC ARM] ServoKit initialized on 0x40")
    except Exception as e:
        print(f"[ROBOTIC ARM] Initialization failed: {e}")
        ENABLE_ROBOTIC_ARM = False

# ============================================================================
#  CONFIGURATION
# ============================================================================

CLASS_NAMES = ['CLASS A', 'CLASS B', 'CLASS C', 'CLASS D', 'CLASS E']
IMG_SIZE = 224
SMOOTH_FRAMES = 7
CAMERA_INDEX = 0
CONF_THRESH = 0.40  # 40% minimum confidence

# Buyo Detection Parameters (to filter out non-buyo objects)
BUYO_DETECTION_ENABLED = True        # Enable/disable buyo detection
BUYO_MIN_CONFIDENCE = 0.30           # Minimum confidence to consider it might be buyo
BUYO_ENTROPY_THRESHOLD = 1.3         # Maximum entropy for buyo detection
                                     # Lower = more certain, Higher = more uncertain

# UI Settings
SIDEBAR_W = 290
FONT_MAIN = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

# BGR colors for each class
CLASS_COLORS = {
    'CLASS A': (50, 220, 50),     # Bright Green
    'CLASS B': (0, 200, 255),     # Orange
    'CLASS C': (255, 200, 0),     # Cyan-gold
    'CLASS D': (200, 150, 238),   # Light violet
    'CLASS E': (0, 100, 255),     # Red
}

# ============================================================================
#  AUTOMATIC MODEL DETECTION
# ============================================================================

def find_model_files():
    """Search for .pth model files in current directory"""
    pth_files = glob.glob('*.pth')
    if not pth_files:
        raise FileNotFoundError('❌ No .pth model files found in current directory!')
    
    # Prioritize files with 'best' in name, then 'convnext', then any
    priority_order = ['best', 'convnext', 'effnet', 'efficientnet', 'resnet']
    
    for keyword in priority_order:
        for pth in pth_files:
            if keyword in pth.lower():
                return pth
    
    return pth_files[0]

def load_classifier(path, num_classes, dev):
    """Load model with automatic backbone detection"""
    print(f'Loading model: {path}')
    
    try:
        ckpt = torch.load(path, map_location=dev)
    except Exception as e:
        print(f'ERROR loading checkpoint: {e}')
        raise
    
    # Detect backbone from checkpoint or filename
    bb = ckpt.get('backbone', None)
    
    if bb is None:
        # Try to infer from filename
        path_lower = path.lower()
        if 'convnext' in path_lower:
            bb = 'convnext_tiny'
        elif 'effnet' in path_lower or 'efficientnet' in path_lower:
            bb = 'efficientnet_v2_s'
        elif 'resnet' in path_lower:
            bb = 'resnet50'
        else:
            bb = 'resnet50'  # Default fallback
    
    print(f'  Detected backbone: {bb}')
    
    # Build model based on backbone
    if bb == 'convnext_tiny':
        m = models.convnext_tiny(weights=None)
        in_f = m.classifier[2].in_features
        m.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(in_f, eps=1e-6),
            nn.Dropout(p=0.4),
            nn.Linear(in_f, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
    elif bb == 'efficientnet_v2_s':
        m = models.efficientnet_v2_s(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_f, 512),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
    elif bb == 'resnet50':
        m = models.resnet50(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
    else:
        # Fallback to MobileNet for unknown backbones
        print(f'  Unknown backbone, using MobileNet V2')
        m = models.mobilenet_v2(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.last_channel, num_classes),
        )
        bb = 'mobilenet_v2'
    
    # Load state dict
    try:
        m.load_state_dict(ckpt['state_dict'])
    except KeyError:
        # Try loading direct state dict if 'state_dict' key doesn't exist
        m.load_state_dict(ckpt)
    
    return m, bb

# ============================================================================
#  INITIALIZATION
# ============================================================================

# Device setup (CPU optimized for Raspberry Pi)
device = torch.device('cpu')  # Raspberry Pi typically uses CPU
print(f'═══════════════════════════════════════════════')
print(f'  Buyo Leaf Grade Classifier - Raspberry Pi')
print(f'═══════════════════════════════════════════════')
print(f'  Device: {device}')

# Auto-detect and load model
MODEL_PATH = find_model_files()
print(f'  Model : {MODEL_PATH}')

num_classes = len(CLASS_NAMES)
model, backbone = load_classifier(MODEL_PATH, num_classes, device)
print(f'  Classes: {len(CLASS_NAMES)} (A through E)')
print(f'═══════════════════════════════════════════════')

model = model.to(device)
model.eval()

# ============================================================================
#  ROBOTIC ARM CONTROL FUNCTIONS
# ============================================================================

def trigger_arm_drop(arm_num, class_name):
    """
    Triggers the robotic arm to drop the leaf into the correct bin.
    
    Args:
        arm_num: Robotic arm number (1-9)
        class_name: Classified buyo class ('CLASS A', 'CLASS B', etc.)
    """
    global kit
    
    if not ENABLE_ROBOTIC_ARM or kit is None:
        print(f"[ROBOTIC ARM] Simulated drop: Arm #{arm_num} → {class_name}")
        return
    
    channel = ARM_CHANNELS.get(arm_num)
    if channel is None:
        print(f"[WARN] Arm #{arm_num} not in channel map")
        return
    
    try:
        print(f"[ROBOTIC ARM] Arm #{arm_num} dropping {class_name} (channel {channel})")
        
        # Extend arm to drop position
        kit.servo[channel].angle = ARM_DROP_ANGLE
        time.sleep(0.6)  # Hold while leaf drops
        
        # Retract arm to home position
        kit.servo[channel].angle = ARM_HOME_ANGLE
        time.sleep(0.4)  # Allow retraction to complete
        
        print(f"[ROBOTIC ARM] Arm #{arm_num} drop complete")
        
    except Exception as e:
        print(f"[ERROR] Arm #{arm_num} drop failed: {e}")
        try:
            kit.servo[channel].angle = ARM_HOME_ANGLE  # Safety: always retract
        except Exception:
            pass

# ============================================================================
#  IMAGE PREPROCESSING
# ============================================================================

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess_frame(bgr_frame):
    """Preprocess camera frame for model inference"""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return preprocess(pil).unsqueeze(0).to(device)

def preprocess_pil(pil_img):
    """Preprocess PIL image for model inference"""
    return preprocess(pil_img.convert('RGB')).unsqueeze(0).to(device)

# ============================================================================
#  TEMPORAL SMOOTHING
# ============================================================================

prob_history = deque(maxlen=SMOOTH_FRAMES)
_smooth_lock = threading.Lock()

def get_smoothed_probs(raw_probs):
    """Average probabilities over recent frames for stability"""
    with _smooth_lock:
        prob_history.append(raw_probs)
        return np.mean(prob_history, axis=0)

def reset_smoothing():
    """Clear smoothing buffer"""
    with _smooth_lock:
        prob_history.clear()

def is_confident(probs):
    """Check if prediction is confident enough"""
    max_prob = float(np.max(probs))
    return max_prob >= CONF_THRESH

def is_buyo_leaf(probs):
    """
    Detect if the object is actually a buyo leaf or something else.
    
    Uses entropy (uncertainty) to detect non-buyo objects:
    - Low entropy = confident prediction = likely buyo leaf
    - High entropy = uncertain prediction = likely NOT buyo leaf
    
    Returns: (is_buyo: bool, reason: str)
    """
    if not BUYO_DETECTION_ENABLED:
        return True, "Detection disabled"
    
    # Calculate entropy (measure of uncertainty)
    # Entropy formula: -sum(p * log(p))
    # Lower entropy = more confident, Higher entropy = more uncertain
    probs_safe = np.maximum(probs, 1e-10)  # Avoid log(0)
    entropy = -np.sum(probs * np.log(probs_safe))
    
    max_prob = float(np.max(probs))
    
    # Check 1: Is there any reasonable confidence?
    if max_prob < BUYO_MIN_CONFIDENCE:
        return False, f"Too low confidence ({max_prob*100:.1f}% < {BUYO_MIN_CONFIDENCE*100:.0f}%)"
    
    # Check 2: Is entropy too high (too uncertain)?
    if entropy > BUYO_ENTROPY_THRESHOLD:
        return False, f"High uncertainty (entropy={entropy:.2f} > {BUYO_ENTROPY_THRESHOLD:.1f})"
    
    # Check 3: Are probabilities too evenly distributed? (sign of non-buyo)
    # If all probabilities are similar, it's likely not buyo
    prob_std = np.std(probs)
    if prob_std < 0.08:  # Very low standard deviation = all probs similar
        return False, f"Uniform distribution (std={prob_std:.3f})"
    
    return True, f"Valid buyo (conf={max_prob*100:.1f}%, entropy={entropy:.2f})"

# ============================================================================
#  THREADED INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    """Background thread for model inference to keep UI responsive"""
    
    def __init__(self):
        self._in_q = queue.Queue(maxsize=2)
        self._result = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._last_trigger_time = 0
        self._trigger_cooldown = 5.0  # Seconds between robotic arm triggers
    
    def put_frame(self, bgr_frame):
        """Submit frame for inference (non-blocking)"""
        try:
            self._in_q.put_nowait(bgr_frame.copy())
        except queue.Full:
            pass
    
    def get_result(self):
        """Get latest inference result"""
        with self._lock:
            return self._result
    
    def stop(self):
        """Stop inference thread"""
        self._stop.set()
        try:
            self._in_q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
    
    def _worker(self):
        """Background inference loop"""
        while not self._stop.is_set():
            try:
                frame = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break
            
            tensor = preprocess_frame(frame)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            
            smooth = get_smoothed_probs(probs)
            pred_idx = int(np.argmax(smooth))
            confidence = float(smooth[pred_idx])
            is_conf = is_confident(smooth)
            is_buyo, buyo_reason = is_buyo_leaf(smooth)
            
            with self._lock:
                self._result = (smooth, pred_idx, confidence, is_conf, is_buyo, buyo_reason)
                
                # Trigger robotic arm only if:
                # 1. Confident classification
                # 2. Detected as buyo leaf
                # 3. Robotic arm enabled
                if is_conf and is_buyo and ENABLE_ROBOTIC_ARM:
                    current_time = time.time()
                    if current_time - self._last_trigger_time >= self._trigger_cooldown:
                        pred_name = CLASS_NAMES[pred_idx]
                        # Trigger in separate thread to avoid blocking inference
                        threading.Thread(
                            target=trigger_arm_drop,
                            args=(CURRENT_ARM, pred_name),
                            daemon=True
                        ).start()
                        self._last_trigger_time = current_time

# ============================================================================
#  TOAST NOTIFICATIONS
# ============================================================================

class Toast:
    """Temporary on-screen notifications"""
    
    def __init__(self):
        self._messages = []
    
    def show(self, text, duration=1.8, color=(255, 255, 255)):
        """Display toast message"""
        self._messages.append([text, time.time() + duration, color])
    
    def draw(self, frame):
        """Draw active toasts on frame"""
        now = time.time()
        active = [m for m in self._messages if m[1] > now]
        self._messages = active
        
        h = frame.shape[0]
        for idx, (text, expiry, color) in enumerate(reversed(active)):
            alpha = min(1.0, (expiry - now) / 0.4)
            y = h - 60 - idx * 28
            cv2.putText(frame, text, (12, y + 1), FONT_SMALL, 0.55,
                       (0, 0, 0), 2, cv2.LINE_AA)
            faded_color = tuple(int(c * alpha) for c in color)
            cv2.putText(frame, text, (12, y), FONT_SMALL, 0.55,
                       faded_color, 1, cv2.LINE_AA)

toast = Toast()

# ============================================================================
#  UI DRAWING FUNCTIONS
# ============================================================================

def draw_reticle(frame, is_conf, pred_name, t):
    """Draw scanning reticle on camera feed"""
    h, w = frame.shape[:2]
    cam_w = w - SIDEBAR_W
    
    cx, cy = cam_w // 2, h // 2
    box_w, box_h = int(cam_w * 0.55), int(h * 0.60)
    x1, y1 = cx - box_w // 2, cy - box_h // 2
    x2, y2 = cx + box_w // 2, cy + box_h // 2
    
    if not is_conf:
        base_color = (100, 100, 100)
    else:
        base_color = CLASS_COLORS[pred_name]
    
    pulse = 0.6 + 0.4 * abs(np.sin(t * 2.5))
    color = tuple(int(c * pulse) for c in base_color)
    
    arm, thick = 28, 2
    corners = [
        ((x1, y1), (x1 + arm, y1), (x1, y1 + arm)),
        ((x2, y1), (x2 - arm, y1), (x2, y1 + arm)),
        ((x1, y2), (x1 + arm, y2), (x1, y2 - arm)),
        ((x2, y2), (x2 - arm, y2), (x2, y2 - arm)),
    ]
    for apex, h_pt, v_pt in corners:
        cv2.line(frame, apex, h_pt, color, thick, cv2.LINE_AA)
        cv2.line(frame, apex, v_pt, color, thick, cv2.LINE_AA)
    
    # Center crosshair
    ch_len = 10
    cv2.line(frame, (cx - ch_len, cy), (cx + ch_len, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - ch_len), (cx, cy + ch_len), color, 1, cv2.LINE_AA)

def draw_hud(frame, pred_name, confidence, all_probs, is_conf, fps, mode='LIVE', t=0.0, is_buyo=True, buyo_reason=""):
    """Draw heads-up display with predictions and info"""
    h, w = frame.shape[:2]
    x0 = w - SIDEBAR_W + 14
    
    # Sidebar background
    sidebar_region = frame[:, w - SIDEBAR_W:]
    blurred = cv2.GaussianBlur(sidebar_region, (31, 31), 0)
    dark = cv2.addWeighted(blurred, 0.25, np.zeros_like(blurred), 0.75, 0)
    frame[:, w - SIDEBAR_W:] = dark
    cv2.line(frame, (w - SIDEBAR_W, 0), (w - SIDEBAR_W, h), (60, 60, 60), 1)
    
    # Header
    cv2.putText(frame, 'BUYO LEAF', (x0, 30), FONT_MAIN, 0.65,
               (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, 'GRADING', (x0, 52), FONT_MAIN, 0.65,
               (240, 240, 240), 1, cv2.LINE_AA)
    cv2.line(frame, (x0, 62), (w - 10, 62), (70, 70, 70), 1)
    
    # Model info
    cv2.putText(frame, backbone.upper(), (x0, 76),
               FONT_SMALL, 0.36, (110, 110, 110), 1, cv2.LINE_AA)
    
    # Mode badge
    if mode == 'LIVE':
        badge_col, badge_txt = (60, 220, 100), '● LIVE'
    else:
        badge_col, badge_txt = (80, 160, 255), '● PHOTO'
    cv2.putText(frame, badge_txt, (x0 + 142, 76),
               FONT_SMALL, 0.37, badge_col, 1, cv2.LINE_AA)
    
    # Buyo detection status
    if BUYO_DETECTION_ENABLED and mode == 'LIVE':
        if is_buyo:
            buyo_status = '✓ BUYO'
            buyo_color = (100, 255, 100)
        else:
            buyo_status = '✗ NOT BUYO'
            buyo_color = (100, 100, 255)
        cv2.putText(frame, buyo_status, (x0, 88),
                   FONT_SMALL, 0.32, buyo_color, 1, cv2.LINE_AA)
    
    # Robotic arm status indicator
    if ENABLE_ROBOTIC_ARM:
        arm_status = f'ARM #{CURRENT_ARM}'
        cv2.putText(frame, arm_status, (x0 + 130, 88),
                   FONT_SMALL, 0.32, (100, 180, 255), 1, cv2.LINE_AA)
    
    # Probability bars
    bar_w, bar_h, row_gap, y_start = SIDEBAR_W - 28, 14, 42, 100
    
    for i, name in enumerate(CLASS_NAMES):
        y = y_start + i * row_gap
        prob = float(all_probs[i])
        color = CLASS_COLORS[name]
        
        is_pred = (name == pred_name and is_conf)
        label_color = color if is_pred else (190, 190, 190)
        
        # Highlight active class
        if is_pred:
            roi = frame[y - 13: y + 4 + bar_h + 3, x0 - 4: w - 8]
            if roi.size > 0:
                tint = np.full_like(roi, color, dtype=np.uint8)
                frame[y - 13: y + 4 + bar_h + 3, x0 - 4: w - 8] = \
                    cv2.addWeighted(roi, 0.88, tint, 0.12, 0)
        
        cv2.putText(frame, name, (x0, y), FONT_SMALL, 0.43,
                   label_color, 1, cv2.LINE_AA)
        
        # Bar background
        cv2.rectangle(frame, (x0, y + 4), (x0 + bar_w, y + 4 + bar_h),
                     (45, 45, 45), -1)
        cv2.rectangle(frame, (x0, y + 4), (x0 + bar_w, y + 4 + bar_h),
                     (75, 75, 75), 1)
        
        # Bar fill
        fill = max(1, int(bar_w * prob))
        cv2.rectangle(frame, (x0, y + 4), (x0 + fill, y + 4 + bar_h),
                     color, -1)
        
        # Percentage
        pct_txt = f'{prob * 100:.1f}%'
        cv2.putText(frame, pct_txt, (x0 + bar_w - 48, y + 15),
                   FONT_SMALL, 0.37, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Divider
    y_div = y_start + len(CLASS_NAMES) * row_gap + 10
    cv2.line(frame, (x0, y_div), (w - 10, y_div), (70, 70, 70), 1)
    
    # Prediction result
    if pred_name == "NOT BUYO":
        pred_txt = 'NOT BUYO'
        pred_col = (100, 100, 255)  # Red for non-buyo
        conf_txt = buyo_reason if buyo_reason else 'Not a buyo leaf'
        conf_col = (150, 150, 200)
    elif not is_conf:
        pred_txt = 'UNCERTAIN'
        pred_col = (100, 100, 200)
        conf_txt = f'Low confidence ({confidence * 100:.1f}%)'
        conf_col = (130, 130, 180)
    else:
        pred_txt = pred_name
        pred_col = CLASS_COLORS[pred_name]
        conf_txt = f'{confidence * 100:.1f}% confident'
        conf_col = (200, 200, 200)
    
    # Result box
    res_y0, res_y1 = y_div + 6, y_div + 80
    res_roi = frame[res_y0:res_y1, x0 - 4: w - 8]
    if res_roi.size > 0:
        if not is_conf:
            tint = np.full_like(res_roi, (40, 40, 40), dtype=np.uint8)
            frame[res_y0:res_y1, x0 - 4: w - 8] = \
                cv2.addWeighted(res_roi, 0.5, tint, 0.5, 0)
        else:
            frame[res_y0:res_y1, x0 - 4: w - 8] = \
                (res_roi.astype(np.float32) * 0.6).astype(np.uint8)
    
    cv2.putText(frame, pred_txt, (x0, y_div + 40),
               FONT_MAIN, 0.90, pred_col, 2, cv2.LINE_AA)
    cv2.putText(frame, conf_txt, (x0, y_div + 60),
               FONT_SMALL, 0.40, conf_col, 1, cv2.LINE_AA)
    if is_conf:
        cv2.line(frame, (x0, y_div + 65), (x0 + 180, y_div + 65),
                pred_col, 1)
    
    # Footer
    cv2.line(frame, (x0, h - 68), (w - 10, h - 68), (55, 55, 55), 1)
    if mode == 'LIVE':
        fps_col = (80, 220, 80) if fps >= 20 else \
                  (80, 180, 255) if fps >= 10 else (60, 60, 220)
        cv2.putText(frame, f'FPS  {fps:.1f}', (x0, h - 52),
                   FONT_SMALL, 0.40, fps_col, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Q quit   S save   R reset',
                   (x0, h - 36), FONT_SMALL, 0.33, (85, 85, 85), 1, cv2.LINE_AA)
        cv2.putText(frame, 'U upload   B batch   A arm#',
                   (x0, h - 20), FONT_SMALL, 0.33, (85, 85, 85), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Q close   S save',
                   (x0, h - 20), FONT_SMALL, 0.35, (85, 85, 85), 1, cv2.LINE_AA)
    
    # Camera feed border
    tk2 = 3
    pulse_alpha = 0.55 + 0.45 * abs(np.sin(t * 1.8))
    if is_conf:
        bc = tuple(int(c * pulse_alpha) for c in CLASS_COLORS[pred_name])
    else:
        bc = tuple(int(c * pulse_alpha) for c in (100, 100, 100))
    cv2.rectangle(frame, (tk2, tk2), (w - SIDEBAR_W - tk2, h - tk2), bc, tk2)

# ============================================================================
#  IMAGE UPLOAD AND CLASSIFICATION
# ============================================================================

def open_file_dialog():
    """Open file selection dialog for single image upload"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title='Select a Buyo Leaf Image',
        filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.webp'),
                  ('All files', '*.*')]
    )
    root.destroy()
    return path if path else None

def open_files_dialog():
    """Open file selection dialog for multiple images (batch upload)"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    paths = filedialog.askopenfilenames(
        title='Select Buyo Leaf Images (Max 100)',
        filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.webp'),
                  ('All files', '*.*')]
    )
    root.destroy()
    return list(paths) if paths else []

def classify_image_file(path):
    """Classify uploaded image and display results"""
    try:
        pil_img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f'ERROR loading image: {e}')
        return None
    
    tensor = preprocess_pil(pil_img)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    pred_name = CLASS_NAMES[pred_idx]
    is_conf = is_confident(probs)
    
    print(f'\n─── Photo Classification ───')
    print(f'File: {os.path.basename(path)}')
    if is_conf:
        print(f'Grade: {pred_name}  ({confidence * 100:.2f}%)')
    else:
        print(f'Grade: UNCERTAIN  (low confidence {confidence * 100:.2f}%)')
    for i, name in enumerate(CLASS_NAMES):
        bar = '█' * int(probs[i] * 20)
        print(f'  {name}: {probs[i] * 100:5.1f}% |{bar:<20}|')
    
    # Build display
    target_w, target_h = 1280, 720
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    cam_w = target_w - SIDEBAR_W
    ih, iw = img_bgr.shape[:2]
    scale = min(cam_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - nh) // 2
    x_off = (cam_w - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    
    # Filename label
    fname_short = os.path.basename(path)
    if len(fname_short) > 42:
        fname_short = '...' + fname_short[-39:]
    cv2.rectangle(canvas, (0, 0), (cam_w, 30), (0, 0, 0), -1)
    cv2.putText(canvas, fname_short, (10, 20),
               FONT_SMALL, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    
    draw_hud(canvas, pred_name, confidence, probs, is_conf, 0.0, 'PHOTO', 0.0)
    draw_reticle(canvas, is_conf, pred_name, 0.0)
    
    # Save result
    result_path = os.path.splitext(path)[0] + '_result.jpg'
    cv2.imwrite(result_path, canvas)
    print(f'Result saved: {result_path}')
    
    win_name = f'Photo: {os.path.basename(path)}'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, target_w, target_h)
    cv2.imshow(win_name, canvas)
    print('Press Q (or close window) to continue.')
    
    while True:
        key = cv2.waitKey(30) & 0xFF
        try:
            visible = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
        except:
            visible = -1
        if key == ord('q') or visible < 1:
            break
        elif key == ord('s'):
            cv2.imwrite(result_path, canvas)
            print(f'Saved: {result_path}')
    
    try:
        cv2.destroyWindow(win_name)
    except:
        pass
    
    # Return classification result
    return {
        'filename': os.path.basename(path),
        'path': path,
        'class': pred_name,
        'confidence': confidence,
        'is_confident': is_conf,
        'probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    }

def classify_batch_files(paths):
    """Classify multiple images in batch (up to 100)"""
    max_files = 100
    if len(paths) > max_files:
        print(f'⚠️  Warning: Selected {len(paths)} files, processing first {max_files} only')
        paths = paths[:max_files]
    
    print(f'\n╔═══════════════════════════════════════════════╗')
    print(f'║  Batch Classification: {len(paths)} images')
    print(f'╚═══════════════════════════════════════════════╝\n')
    
    results = []
    class_counts = {name: 0 for name in CLASS_NAMES}
    confident_count = 0
    uncertain_count = 0
    
    start_time = time.time()
    
    for idx, path in enumerate(paths, 1):
        print(f'\n[{idx}/{len(paths)}] Processing: {os.path.basename(path)}')
        
        try:
            pil_img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f'❌ ERROR loading image: {e}')
            results.append({
                'filename': os.path.basename(path),
                'path': path,
                'class': 'ERROR',
                'confidence': 0.0,
                'is_confident': False,
                'error': str(e)
            })
            continue
        
        tensor = preprocess_pil(pil_img)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_name = CLASS_NAMES[pred_idx]
        is_conf = is_confident(probs)
        
        # Update statistics
        if is_conf:
            class_counts[pred_name] += 1
            confident_count += 1
            print(f'✓ Grade: {pred_name} ({confidence * 100:.1f}%)')
        else:
            uncertain_count += 1
            print(f'? Grade: UNCERTAIN (low confidence {confidence * 100:.1f}%)')
        
        result = {
            'filename': os.path.basename(path),
            'path': path,
            'class': pred_name,
            'confidence': confidence,
            'is_confident': is_conf,
            'probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        }
        results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f'\n╔═══════════════════════════════════════════════╗')
    print(f'║  Batch Classification Complete')
    print(f'╚═══════════════════════════════════════════════╝')
    print(f'\nTotal Images: {len(paths)}')
    print(f'Processing Time: {elapsed_time:.2f}s ({elapsed_time/len(paths):.2f}s per image)')
    print(f'\n📊 Results Summary:')
    print(f'  Confident: {confident_count} ({confident_count/len(paths)*100:.1f}%)')
    print(f'  Uncertain: {uncertain_count} ({uncertain_count/len(paths)*100:.1f}%)')
    print(f'\n📈 Grade Distribution:')
    for name in CLASS_NAMES:
        count = class_counts[name]
        percentage = count / len(paths) * 100 if len(paths) > 0 else 0
        bar = '█' * int(percentage / 5)
        print(f'  {name}: {count:3d} ({percentage:5.1f}%) |{bar:<20}|')
    
    # Save results to CSV
    csv_path = f'batch_results_{int(time.time())}.csv'
    try:
        import csv
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'class', 'confidence', 'is_confident'] + \
                         [f'prob_{name}' for name in CLASS_NAMES]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                if 'error' not in r:
                    row = {
                        'filename': r['filename'],
                        'class': r['class'],
                        'confidence': f"{r['confidence']:.4f}",
                        'is_confident': r['is_confident']
                    }
                    for name in CLASS_NAMES:
                        row[f'prob_{name}'] = f"{r['probabilities'][name]:.4f}"
                    writer.writerow(row)
        print(f'\n💾 Results saved to: {csv_path}')
    except Exception as e:
        print(f'\n⚠️  Could not save CSV: {e}')
    
    # Ask if user wants to view individual results
    print(f'\nPress ENTER to view individual results, or any other key to skip...')
    import sys
    import select
    
    # Wait for user input (with timeout for non-interactive environments)
    try:
        if sys.stdin in select.select([sys.stdin], [], [], 5)[0]:
            user_input = sys.stdin.readline()
            if user_input.strip() == '':
                # User pressed Enter, show results
                for result in results:
                    if 'error' not in result:
                        classify_image_file(result['path'])
    except:
        pass  # Skip interactive prompt if not supported
    
    return results

# ============================================================================
#  MAIN CAMERA LOOP
# ============================================================================

def main():
    """Main application loop"""
    
    global CURRENT_ARM
    
    print('='*64)
    print('  Buyo Leaf Grade Classifier - Raspberry Pi')
    print('='*64)
    print(f'  Device: {device}')
    print(f'  Model : {MODEL_PATH}')
    print()
    print('  Detecting camera...')
    
    # Initialize camera
    cap = None
    camera_found = False
    camera_method = ""
    
    # Method 1: Try libcamera first (for Pi Camera Module 3 / IMX708)
    print('  [1/3] Trying libcamera (Pi Camera Module 3)...')
    try:
        # GStreamer pipeline for libcamera
        pipeline = (
            'libcamerasrc ! '
            'video/x-raw,width=640,height=480,framerate=10/1 ! '
            'videoconvert ! '
            'appsink drop=1'
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print('        ✓ Pi Camera detected via libcamera')
                camera_found = True
                camera_method = "libcamera"
            else:
                cap.release()
                cap = None
        else:
            cap = None
    except Exception as e:
        print(f'        ✗ libcamera failed: {e}')
        cap = None
    
    # Method 2: Try V4L2 USB cameras
    if not camera_found:
        print('  [2/3] Trying V4L2 (USB cameras)...')
        for idx in range(6):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f'        ✓ USB Camera detected at /dev/video{idx}')
                        camera_found = True
                        camera_method = f"V4L2 index {idx}"
                        break
                    cap.release()
            except Exception:
                pass
    
    # Method 3: Try any available backend
    if not camera_found:
        print('  [3/3] Trying any available backend...')
        for idx in range(6):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f'        ✓ Camera detected at index {idx}')
                        camera_found = True
                        camera_method = f"Auto index {idx}"
                        break
                    cap.release()
            except Exception:
                pass
    
    if not camera_found or cap is None or not cap.isOpened():
        print()
        print('  ❌ ERROR: No camera available')
        print()
        print('  Please run diagnostic:')
        print('    python test_camera.py')
        print()
        print('  Or check:')
        print('    1. sudo raspi-config → Interface → Camera → Enable')
        print('    2. v4l2-ctl --list-devices')
        print('    3. libcamera-hello --list-cameras')
        print()
        return
    
    print(f'  ✓ Camera initialized using {camera_method}')
    
    # Set camera properties (optimized for Raspberry Pi + IMX708)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)     # Increased buffer for Pi Camera
    cap.set(cv2.CAP_PROP_FPS, 10)           # Lower FPS for IMX708 stability
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # Extended warm up for Pi Camera Module 3 (IMX708)
    print('  Warming up camera (IMX708)...')
    successful_reads = 0
    for i in range(30):
        ret, _ = cap.read()
        if ret:
            successful_reads += 1
        time.sleep(0.1)
    
    if successful_reads < 15:
        print(f'  WARNING: Camera unstable ({successful_reads}/30 frames)')
    else:
        print(f'  ✓ Camera ready ({successful_reads}/30 frames)')
    
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f'✓ Resolution: {int(actual_w)}x{int(actual_h)}')
    print('  Q=quit | S=save | U=upload | B=batch | R=reset | A=set arm#')
    if ENABLE_ROBOTIC_ARM:
        print(f'  🤖 Robotic arm integration: ENABLED (Current Arm: #{CURRENT_ARM})')
    else:
        print('  🤖 Robotic arm integration: DISABLED (camera-only mode)')
    print()
    
    engine = InferenceEngine()
    
    win_name = 'Buyo Leaf Grade Classifier'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    
    prev_time = time.time()
    screenshot_idx = 0
    t_start = time.time()
    
    # Placeholder result
    n_cls = len(CLASS_NAMES)
    result = (np.ones(n_cls) / n_cls, 0, 1.0 / n_cls, False, True, "Initializing...")
    
    retry_count = 0
    max_retries = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            retry_count += 1
            print(f'WARNING: Failed to grab frame (retry {retry_count}/{max_retries})')
            if retry_count >= max_retries:
                print('ERROR: Camera timeout - too many failed frames')
                break
            time.sleep(0.1)  # Wait before retry
            continue
        
        retry_count = 0  # Reset on successful read
        
        engine.put_frame(frame)
        
        fresh = engine.get_result()
        if fresh is not None:
            result = fresh
        
        smooth, pred_idx, confidence, is_conf, is_buyo, buyo_reason = result
        t_now = time.time()
        fps = 1.0 / max(t_now - prev_time, 1e-6)
        prev_time = t_now
        t = t_now - t_start
        
        pred_name = CLASS_NAMES[pred_idx]
        
        # Override display if not detected as buyo
        if not is_buyo:
            pred_name_display = "NOT BUYO"
            is_conf_display = False
        else:
            pred_name_display = pred_name
            is_conf_display = is_conf
        
        draw_hud(frame, pred_name_display, confidence, smooth, is_conf_display, fps, 'LIVE', t, is_buyo, buyo_reason)
        draw_reticle(frame, is_conf_display and is_buyo, pred_name_display, t)
        toast.draw(frame)
        
        cv2.imshow(win_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f'screenshot_{screenshot_idx:03d}.jpg'
            cv2.imwrite(fname, frame)
            toast.show(f'Saved {fname}', color=(100, 255, 100))
            print(f'Saved: {fname}')
            screenshot_idx += 1
        elif key == ord('r'):
            reset_smoothing()
            toast.show('Smoothing reset', color=(255, 200, 80))
            print('Smoothing reset')
        elif key == ord('a'):
            # Change active robotic arm
            print('\n═══ Set Active Robotic Arm ═══')
            print(f'Current: Arm #{CURRENT_ARM}')
            try:
                new_arm = int(input('Enter arm number (1-9): '))
                if 1 <= new_arm <= 9:
                    CURRENT_ARM = new_arm
                    toast.show(f'Arm #{CURRENT_ARM} active', color=(100, 180, 255))
                    print(f'✓ Active arm set to #{CURRENT_ARM}')
                else:
                    print('❌ Invalid arm number (must be 1-9)')
            except ValueError:
                print('❌ Invalid input')
        elif key == ord('u'):
            print('Opening file dialog...')
            img_path = open_file_dialog()
            if img_path:
                reset_smoothing()
                classify_image_file(img_path)
                reset_smoothing()
                toast.show('Photo classified', color=(100, 180, 255))
            else:
                print('No file selected')
        elif key == ord('b'):
            print('Opening batch file dialog...')
            img_paths = open_files_dialog()
            if img_paths:
                print(f'Selected {len(img_paths)} files')
                reset_smoothing()
                classify_batch_files(img_paths)
                reset_smoothing()
                toast.show(f'Batch: {len(img_paths)} files', color=(255, 180, 100))
            else:
                print('No files selected')
    
    engine.stop()
    cap.release()
    cv2.destroyAllWindows()
    
    # Cleanup GPIO if used
    if ENABLE_ROBOTIC_ARM:
        try:
            GPIO.cleanup()
        except:
            pass
    
    print('\n✓ Camera classifier closed')

# ============================================================================
#  ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    main()
