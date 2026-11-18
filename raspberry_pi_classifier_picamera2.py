#!/usr/bin/env python3
"""
Raspberry Pi Buyo Leaf Grade Classifier - Picamera2 + Robotic Arm
==================================================================
Full-featured classifier optimized for Pi Camera Module 3 (IMX708)
Automatically triggers robotic arm when buyo leaf is detected

Features:
  - Picamera2 support for IMX708 camera
  - Real-time classification with GUI
  - Automatic robotic arm trigger on buyo detection
  - Batch image upload and classification
  - Temporal smoothing for stable predictions
  - Buyo detection filter (rejects hands/objects)

Controls:
  Q - quit
  S - save screenshot  
  U - upload single photo
  B - batch upload multiple photos
  R - reset smoothing buffer
  A - set active arm number (1-9)

Robotic Arm Auto-Trigger Conditions:
  1. Confident classification (≥40%)
  2. Detected as buyo leaf (not hand/object)
  3. Hardware enabled (ENABLE_ROBOTIC_ARM = True)
  4. 5-second cooldown passed
"""

import sys
import os

# Import picamera2 from system packages
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    site_pkg = '/usr/lib/python3/dist-packages'
    if os.path.exists(site_pkg) and site_pkg not in sys.path:
        sys.path.insert(0, site_pkg)
        try:
            from picamera2 import Picamera2
            PICAMERA2_AVAILABLE = True
        except ImportError:
            PICAMERA2_AVAILABLE = False
    else:
        PICAMERA2_AVAILABLE = False

if not PICAMERA2_AVAILABLE:
    print("="*64)
    print("  ❌ ERROR: picamera2 not available!")
    print("="*64)
    print("\n  Install: sudo apt-get install -y python3-picamera2")
    print("  Then run: python3 raspberry_pi_classifier_picamera2.py\n")
    sys.exit(1)

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from collections import deque, Counter
import time
import threading
import queue
import glob
import tkinter as tk
from tkinter import filedialog

# ============================================================================
#  ROBOTIC ARM HARDWARE INTEGRATION
# ============================================================================
ENABLE_ROBOTIC_ARM = False

try:
    import RPi.GPIO as GPIO
    from adafruit_servokit import ServoKit
    ENABLE_ROBOTIC_ARM = True
except ImportError:
    print("[INFO] Robotic arm libraries not available")
    print("      Running in camera-only mode")

# Robotic Arm Configuration
ARM_CHANNELS = {1: 3, 2: 2, 3: 1, 4: 0, 5: 8, 6: 7, 7: 6, 8: 5, 9: 4}
ARM_DROP_ANGLE = 60
ARM_HOME_ANGLE = 0
CURRENT_ARM = 1

kit = None
if ENABLE_ROBOTIC_ARM:
    try:
        kit = ServoKit(channels=16, address=0x40)
        print("[ROBOTIC ARM] ✓ ServoKit initialized")
    except Exception as e:
        print(f"[ROBOTIC ARM] ✗ Init failed: {e}")
        ENABLE_ROBOTIC_ARM = False

# ============================================================================
#  CONFIGURATION
# ============================================================================
CLASS_NAMES = ['CLASS A', 'CLASS B', 'CLASS C', 'CLASS D', 'CLASS E']
IMG_SIZE = 224
SMOOTH_FRAMES = 7
CONF_THRESH = 0.40
BUYO_DETECTION_ENABLED = True
BUYO_MIN_CONFIDENCE = 0.30
BUYO_ENTROPY_THRESHOLD = 1.3

# UI Settings
SIDEBAR_W = 290
FONT_MAIN = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX
CLASS_COLORS = {
    'CLASS A': (50, 220, 50), 'CLASS B': (0, 200, 255),
    'CLASS C': (255, 200, 0), 'CLASS D': (200, 150, 238),
    'CLASS E': (0, 100, 255),
}

# ============================================================================
#  MODEL LOADING
# ============================================================================
device = torch.device('cpu')
print("="*64)
print("  Buyo Classifier - Picamera2 + Robotic Arm")
print("="*64)
print(f"  Device: {device}")

# Find model
pth_files = glob.glob('*.pth')
if not pth_files:
    print("  ❌ No .pth model files found!")
    sys.exit(1)

MODEL_PATH = None
for keyword in ['best', 'convnext', 'effnet']:
    for pth in pth_files:
        if keyword in pth.lower():
            MODEL_PATH = pth
            break
    if MODEL_PATH:
        break
if not MODEL_PATH:
    MODEL_PATH = pth_files[0]

print(f"  Model: {MODEL_PATH}")

# Load model
ckpt = torch.load(MODEL_PATH, map_location=device)
bb = ckpt.get('backbone', 'efficientnet_v2_s')
print(f"  Backbone: {bb}")

# Build architecture matching train_ensemble.py
if 'convnext' in bb:
    model = models.convnext_tiny(weights=None)
    in_f = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
        nn.LayerNorm(in_f, eps=1e-6), nn.Dropout(p=0.35),
        nn.Linear(in_f, 512), nn.GELU(), nn.LayerNorm(512),
        nn.Dropout(p=0.25), nn.Linear(512, 256), nn.GELU(),
        nn.Dropout(p=0.15), nn.Linear(256, 5),
    )
elif 'resnet' in bb:
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
        nn.Dropout(p=0.35, inplace=True), nn.Linear(in_f, 512),
        nn.SiLU(), nn.BatchNorm1d(512), nn.Dropout(p=0.25),
        nn.Linear(512, 256), nn.SiLU(), nn.Dropout(p=0.15),
        nn.Linear(256, 5),
    )

model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f"  Classes: {len(CLASS_NAMES)}")
print(f"  🤖 Robotic arm: {'ENABLED' if ENABLE_ROBOTIC_ARM else 'DISABLED'}")
print("="*64)

# ============================================================================
#  ROBOTIC ARM CONTROL
# ============================================================================
def trigger_arm_drop(arm_num, class_name):
    """Trigger robotic arm to drop leaf into bin"""
    global kit
    
    if not ENABLE_ROBOTIC_ARM or kit is None:
        print(f"[ARM] Simulated: Arm #{arm_num} → {class_name}")
        return
    
    channel = ARM_CHANNELS.get(arm_num)
    if channel is None:
        print(f"[ARM] ✗ Invalid arm #{arm_num}")
        return
    
    try:
        print(f"[ARM] ⬇️  Arm #{arm_num} dropping {class_name}")
        kit.servo[channel].angle = ARM_DROP_ANGLE
        time.sleep(0.6)
        kit.servo[channel].angle = ARM_HOME_ANGLE
        time.sleep(0.4)
        print(f"[ARM] ✓ Drop complete")
    except Exception as e:
        print(f"[ARM] ✗ Error: {e}")
        try:
            kit.servo[channel].angle = ARM_HOME_ANGLE
        except:
            pass

# ============================================================================
#  IMAGE PREPROCESSING & DETECTION
# ============================================================================
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

prob_history = deque(maxlen=SMOOTH_FRAMES)
_smooth_lock = threading.Lock()

def get_smoothed_probs(raw_probs):
    with _smooth_lock:
        prob_history.append(raw_probs)
        return np.mean(prob_history, axis=0)

def reset_smoothing():
    with _smooth_lock:
        prob_history.clear()

def is_confident(probs):
    return float(np.max(probs)) >= CONF_THRESH

def is_buyo_leaf(probs):
    """Detect if object is buyo leaf using entropy analysis"""
    if not BUYO_DETECTION_ENABLED:
        return True, "Detection disabled"
    
    probs_safe = np.maximum(probs, 1e-10)
    entropy = -np.sum(probs * np.log(probs_safe))
    max_prob = float(np.max(probs))
    
    if max_prob < BUYO_MIN_CONFIDENCE:
        return False, f"Low conf {max_prob*100:.0f}%"
    if entropy > BUYO_ENTROPY_THRESHOLD:
        return False, f"High entropy {entropy:.2f}"
    
    prob_std = np.std(probs)
    if prob_std < 0.08:
        return False, f"Uniform dist"
    
    return True, f"Valid buyo"

# ============================================================================
#  THREADED INFERENCE ENGINE
# ============================================================================
class InferenceEngine:
    """Background thread for real-time inference + arm triggering"""
    
    def __init__(self):
        self._in_q = queue.Queue(maxsize=2)
        self._result = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._last_trigger_time = 0
        self._trigger_cooldown = 5.0
    
    def put_frame(self, rgb_frame):
        try:
            self._in_q.put_nowait(rgb_frame.copy())
        except queue.Full:
            pass
    
    def get_result(self):
        with self._lock:
            return self._result
    
    def stop(self):
        self._stop.set()
        try:
            self._in_q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
    
    def _worker(self):
        while not self._stop.is_set():
            try:
                frame = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break
            
            # Inference
            pil_img = Image.fromarray(frame)
            tensor = preprocess(pil_img).unsqueeze(0).to(device)
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
                
                # AUTO-TRIGGER ROBOTIC ARM
                if is_conf and is_buyo and ENABLE_ROBOTIC_ARM:
                    current_time = time.time()
                    if current_time - self._last_trigger_time >= self._trigger_cooldown:
                        pred_name = CLASS_NAMES[pred_idx]
                        threading.Thread(
                            target=trigger_arm_drop,
                            args=(CURRENT_ARM, pred_name),
                            daemon=True
                        ).start()
                        self._last_trigger_time = current_time

# ============================================================================
#  GUI DRAWING FUNCTIONS
# ============================================================================
class Toast:
    def __init__(self):
        self._messages = []
    
    def show(self, text, duration=1.8, color=(255, 255, 255)):
        self._messages.append([text, time.time() + duration, color])
    
    def draw(self, frame):
        now = time.time()
        active = [m for m in self._messages if m[1] > now]
        self._messages = active
        h = frame.shape[0]
        for idx, (text, expiry, color) in enumerate(reversed(active)):
            alpha = min(1.0, (expiry - now) / 0.4)
            y = h - 60 - idx * 28
            cv2.putText(frame, text, (12, y + 1), FONT_SMALL, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            faded = tuple(int(c * alpha) for c in color)
            cv2.putText(frame, text, (12, y), FONT_SMALL, 0.55, faded, 1, cv2.LINE_AA)

toast = Toast()

def draw_reticle(frame, is_conf, pred_name, t):
    h, w = frame.shape[:2]
    cam_w = w - SIDEBAR_W
    cx, cy = cam_w // 2, h // 2
    box_w, box_h = int(cam_w * 0.55), int(h * 0.60)
    x1, y1 = cx - box_w // 2, cy - box_h // 2
    x2, y2 = cx + box_w // 2, cy + box_h // 2
    
    base_color = CLASS_COLORS[pred_name] if is_conf else (100, 100, 100)
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
    
    ch_len = 10
    cv2.line(frame, (cx - ch_len, cy), (cx + ch_len, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - ch_len), (cx, cy + ch_len), color, 1, cv2.LINE_AA)

def draw_hud(frame, pred_name, confidence, all_probs, is_conf, fps, mode='LIVE', t=0.0, is_buyo=True, buyo_reason=""):
    h, w = frame.shape[:2]
    x0 = w - SIDEBAR_W + 14
    
    # Sidebar background
    sidebar = frame[:, w - SIDEBAR_W:]
    blurred = cv2.GaussianBlur(sidebar, (31, 31), 0)
    dark = cv2.addWeighted(blurred, 0.25, np.zeros_like(blurred), 0.75, 0)
    frame[:, w - SIDEBAR_W:] = dark
    cv2.line(frame, (w - SIDEBAR_W, 0), (w - SIDEBAR_W, h), (60, 60, 60), 1)
    
    # Header
    cv2.putText(frame, 'BUYO LEAF', (x0, 30), FONT_MAIN, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, 'GRADING', (x0, 52), FONT_MAIN, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.line(frame, (x0, 62), (w - 10, 62), (70, 70, 70), 1)
    
    cv2.putText(frame, bb.upper(), (x0, 76), FONT_SMALL, 0.36, (110, 110, 110), 1, cv2.LINE_AA)
    
    badge_col, badge_txt = ((60, 220, 100), '● LIVE') if mode == 'LIVE' else ((80, 160, 255), '● PHOTO')
    cv2.putText(frame, badge_txt, (x0 + 142, 76), FONT_SMALL, 0.37, badge_col, 1, cv2.LINE_AA)
    
    # Buyo detection status
    if BUYO_DETECTION_ENABLED and mode == 'LIVE':
        if is_buyo:
            cv2.putText(frame, '✓ BUYO', (x0, 88), FONT_SMALL, 0.32, (100, 255, 100), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, '✗ NOT BUYO', (x0, 88), FONT_SMALL, 0.32, (100, 100, 255), 1, cv2.LINE_AA)
    
    # Arm status
    if ENABLE_ROBOTIC_ARM:
        cv2.putText(frame, f'🤖 ARM #{CURRENT_ARM}', (x0 + 125, 88), FONT_SMALL, 0.32, (100, 180, 255), 1, cv2.LINE_AA)
    
    # Probability bars
    bar_w, bar_h, row_gap, y_start = SIDEBAR_W - 28, 14, 42, 100
    
    for i, name in enumerate(CLASS_NAMES):
        y = y_start + i * row_gap
        prob = float(all_probs[i])
        color = CLASS_COLORS[name]
        is_pred = (name == pred_name and is_conf)
        label_color = color if is_pred else (190, 190, 190)
        
        if is_pred:
            roi = frame[y - 13: y + 4 + bar_h + 3, x0 - 4: w - 8]
            if roi.size > 0:
                tint = np.full_like(roi, color, dtype=np.uint8)
                frame[y - 13: y + 4 + bar_h + 3, x0 - 4: w - 8] = cv2.addWeighted(roi, 0.88, tint, 0.12, 0)
        
        cv2.putText(frame, name, (x0, y), FONT_SMALL, 0.43, label_color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y + 4), (x0 + bar_w, y + 4 + bar_h), (45, 45, 45), -1)
        cv2.rectangle(frame, (x0, y + 4), (x0 + bar_w, y + 4 + bar_h), (75, 75, 75), 1)
        
        fill = max(1, int(bar_w * prob))
        cv2.rectangle(frame, (x0, y + 4), (x0 + fill, y + 4 + bar_h), color, -1)
        cv2.putText(frame, f'{prob * 100:.1f}%', (x0 + bar_w - 48, y + 15), FONT_SMALL, 0.37, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Divider
    y_div = y_start + len(CLASS_NAMES) * row_gap + 10
    cv2.line(frame, (x0, y_div), (w - 10, y_div), (70, 70, 70), 1)
    
    # Prediction result
    if not is_buyo and mode == 'LIVE':
        pred_txt, pred_col = 'NOT BUYO', (100, 100, 255)
        conf_txt, conf_col = buyo_reason, (150, 150, 200)
    elif not is_conf:
        pred_txt, pred_col = 'UNCERTAIN', (100, 100, 200)
        conf_txt, conf_col = f'Low conf {confidence * 100:.0f}%', (130, 130, 180)
    else:
        pred_txt, pred_col = pred_name, CLASS_COLORS[pred_name]
        conf_txt, conf_col = f'{confidence * 100:.1f}% confident', (200, 200, 200)
    
    res_y0, res_y1 = y_div + 6, y_div + 80
    res_roi = frame[res_y0:res_y1, x0 - 4: w - 8]
    if res_roi.size > 0:
        if not is_conf:
            tint = np.full_like(res_roi, (40, 40, 40), dtype=np.uint8)
            frame[res_y0:res_y1, x0 - 4: w - 8] = cv2.addWeighted(res_roi, 0.5, tint, 0.5, 0)
        else:
            frame[res_y0:res_y1, x0 - 4: w - 8] = (res_roi.astype(np.float32) * 0.6).astype(np.uint8)
    
    cv2.putText(frame, pred_txt, (x0, y_div + 40), FONT_MAIN, 0.90, pred_col, 2, cv2.LINE_AA)
    cv2.putText(frame, conf_txt, (x0, y_div + 60), FONT_SMALL, 0.40, conf_col, 1, cv2.LINE_AA)
    if is_conf:
        cv2.line(frame, (x0, y_div + 65), (x0 + 180, y_div + 65), pred_col, 1)
    
    # Footer
    cv2.line(frame, (x0, h - 68), (w - 10, h - 68), (55, 55, 55), 1)
    if mode == 'LIVE':
        fps_col = (80, 220, 80) if fps >= 20 else (80, 180, 255) if fps >= 10 else (60, 60, 220)
        cv2.putText(frame, f'FPS {fps:.1f}', (x0, h - 52), FONT_SMALL, 0.40, fps_col, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Q quit  S save  R reset', (x0, h - 36), FONT_SMALL, 0.33, (85, 85, 85), 1, cv2.LINE_AA)
        cv2.putText(frame, 'U upload  B batch  A arm#', (x0, h - 20), FONT_SMALL, 0.33, (85, 85, 85), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Q close   S save', (x0, h - 20), FONT_SMALL, 0.35, (85, 85, 85), 1, cv2.LINE_AA)
    
    # Border
    tk2 = 3
    pulse_alpha = 0.55 + 0.45 * abs(np.sin(t * 1.8))
    bc = tuple(int(c * pulse_alpha) for c in (CLASS_COLORS[pred_name] if is_conf else (100, 100, 100)))
    cv2.rectangle(frame, (tk2, tk2), (w - SIDEBAR_W - tk2, h - tk2), bc, tk2)

# ============================================================================
#  IMAGE UPLOAD & BATCH PROCESSING
# ============================================================================
def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title='Select Buyo Leaf Image',
        filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp'), ('All', '*.*')]
    )
    root.destroy()
    return path if path else None

def open_files_dialog():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    paths = filedialog.askopenfilenames(
        title='Select Buyo Leaf Images (Max 100)',
        filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp'), ('All', '*.*')]
    )
    root.destroy()
    return list(paths)[:100] if paths else []

def classify_image(path):
    try:
        pil_img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f'ERROR: {e}')
        return None
    
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    pred_name = CLASS_NAMES[pred_idx]
    is_conf = is_confident(probs)
    
    print(f'\n─── {os.path.basename(path)} ───')
    print(f'Grade: {pred_name}  ({confidence * 100:.1f}%)')
    for i, name in enumerate(CLASS_NAMES):
        bar = '█' * int(probs[i] * 20)
        print(f'  {name}: {probs[i] * 100:5.1f}% |{bar}')
    
    # Build display
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    target_w, target_h = 1280, 720
    cam_w = target_w - SIDEBAR_W
    ih, iw = img_bgr.shape[:2]
    scale = min(cam_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - nh) // 2
    x_off = (cam_w - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    
    fname = os.path.basename(path)
    if len(fname) > 42:
        fname = '...' + fname[-39:]
    cv2.rectangle(canvas, (0, 0), (cam_w, 30), (0, 0, 0), -1)
    cv2.putText(canvas, fname, (10, 20), FONT_SMALL, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    
    draw_hud(canvas, pred_name, confidence, probs, is_conf, 0.0, 'PHOTO', 0.0)
    draw_reticle(canvas, is_conf, pred_name, 0.0)
    
    result_path = os.path.splitext(path)[0] + '_result.jpg'
    cv2.imwrite(result_path, canvas)
    
    win_name = f'Photo: {os.path.basename(path)}'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, target_w, target_h)
    cv2.imshow(win_name, canvas)
    
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
    
    cv2.destroyWindow(win_name)
    return {'file': os.path.basename(path), 'class': pred_name, 'conf': confidence}

def classify_batch(paths):
    print(f'\n╔{"="*50}╗')
    print(f'║  Batch Classification: {len(paths)} images')
    print(f'╚{"="*50}╝\n')
    
    results = []
    class_counts = Counter()
    
    for idx, path in enumerate(paths, 1):
        print(f'[{idx}/{len(paths)}] {os.path.basename(path)}')
        try:
            pil_img = Image.open(path).convert('RGB')
            tensor = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            
            pred_idx = int(np.argmax(probs))
            pred_name = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
            
            if is_confident(probs):
                class_counts[pred_name] += 1
                print(f'  ✓ {pred_name} ({confidence * 100:.1f}%)')
            else:
                print(f'  ? UNCERTAIN ({confidence * 100:.1f}%)')
            
            results.append({'file': os.path.basename(path), 'class': pred_name, 'conf': confidence})
        except Exception as e:
            print(f'  ✗ ERROR: {e}')
    
    print(f'\n{"─"*50}')
    print('  Summary:')
    for name in CLASS_NAMES:
        if class_counts[name] > 0:
            print(f'    {name}: {class_counts[name]}')
    print(f'{"─"*50}\n')
    return results

# ============================================================================
#  MAIN LOOP
# ============================================================================
print("\nInitializing camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    buffer_count=2
)
picam2.configure(config)
picam2.start()
time.sleep(2)
print("✓ Camera ready!\n")

engine = InferenceEngine()
fps_counter = deque(maxlen=30)
t0 = time.time()
frame_count = 0

win_name = 'Buyo Classifier - Picamera2'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1280, 720)

print("="*64)
print("  CLASSIFIER RUNNING")
print("="*64)
print("  Controls:")
print("    Q - quit")
print("    S - save screenshot")
print("    U - upload single photo")
print("    B - batch upload photos")
print("    R - reset smoothing")
print("    A - set arm number")
print()
if ENABLE_ROBOTIC_ARM:
    print("  🤖 ROBOTIC ARM AUTO-TRIGGER: ACTIVE")
    print("     Triggers when: buyo detected + ≥40% confidence")
    print(f"     Current arm: #{CURRENT_ARM}")
    print(f"     Cooldown: {engine._trigger_cooldown}s between triggers")
else:
    print("  🤖 Robotic arm: DISABLED (simulation mode)")
print("="*64)
print()

try:
    while True:
        frame_t0 = time.time()
        
        # Capture frame
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Submit for inference
        engine.put_frame(frame_rgb)
        
        # Get result
        result = engine.get_result()
        if result is None:
            result = (np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES), 0, 0.2, False, False, "Initializing...")
        
        probs, pred_idx, confidence, is_conf, is_buyo, buyo_reason = result
        pred_name = CLASS_NAMES[pred_idx]
        
        # Build display
        t = time.time() - t0
        fps_counter.append(1.0 / max(0.001, time.time() - frame_t0))
        fps = np.mean(fps_counter)
        
        display = frame_bgr.copy()
        draw_hud(display, pred_name, confidence, probs, is_conf, fps, 'LIVE', t, is_buyo, buyo_reason)
        draw_reticle(display, is_conf, pred_name, t)
        toast.draw(display)
        
        cv2.imshow(win_name, display)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ss_path = f'screenshot_{int(time.time())}.jpg'
            cv2.imwrite(ss_path, display)
            toast.show(f'Screenshot saved: {ss_path}', 2.0, (100, 255, 100))
            print(f'Screenshot: {ss_path}')
        elif key == ord('r'):
            reset_smoothing()
            toast.show('Smoothing buffer reset', 1.5, (255, 200, 100))
            print('[RESET] Smoothing buffer cleared')
        elif key == ord('u'):
            path = open_file_dialog()
            if path:
                print(f'\n[UPLOAD] Classifying: {os.path.basename(path)}')
                classify_image(path)
        elif key == ord('b'):
            paths = open_files_dialog()
            if paths:
                classify_batch(paths)
        elif key == ord('a'):
            print('\n[ARM] Enter arm number (1-9):')
            try:
                num = int(input('  > '))
                if 1 <= num <= 9:
                    CURRENT_ARM = num
                    toast.show(f'Arm #{CURRENT_ARM} selected', 2.0, (100, 180, 255))
                    print(f'[ARM] Switched to arm #{CURRENT_ARM}')
                else:
                    print('[ARM] Invalid (1-9 only)')
            except:
                print('[ARM] Invalid input')
        
        frame_count += 1

except KeyboardInterrupt:
    print("\n\n[STOP] Interrupted by user")

finally:
    print("\n[CLEANUP] Shutting down...")
    engine.stop()
    picam2.stop()
    cv2.destroyAllWindows()
    if ENABLE_ROBOTIC_ARM and kit:
        try:
            GPIO.cleanup()
        except:
            pass
    print("[CLEANUP] ✓ Done\n")
