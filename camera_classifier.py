"""
camera_classifier.py -- Real-Time Buyo Leaf Grade Classifier
=============================================================
Classifies Buyo leaves into 5 quality grades: Class A through Class E

BUYO LEAF CLASSES:
  CLASS A - Highest quality grade (premium leaves)
  CLASS B - High quality grade (good leaves)
  CLASS C - Medium quality grade (average leaves)
  CLASS D - Lower quality grade (below average)
  CLASS E - Lowest quality grade (poor quality)

Features:
  - Threaded inference for smooth camera feed
  - Temporal smoothing (7-frame average) for stable predictions
  - Confidence thresholding to reject uncertain predictions
  - GPU acceleration when available
  - Photo upload and classification
  - Per-class probability visualization

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
import tkinter as tk
from tkinter import filedialog

# Configuration
MODEL_PATH = 'buyo_best.pth'
FALLBACK_PATH = 'buyo_pytorch.pth'

CLASS_NAMES = ['CLASS A', 'CLASS B', 'CLASS C', 'CLASS D', 'CLASS E']
IMG_SIZE = 224
SMOOTH_FRAMES = 7
CAMERA_INDEX = 0

# Inference queue settings
INFER_QUEUE_SIZE = 2

# Confidence threshold - predictions below this are marked as "UNCERTAIN"
# Lower = more predictions shown, Higher = more "UNCERTAIN" results
CONF_THRESH = 0.40   # 40% minimum confidence

# BGR colors for each class
CLASS_COLORS = {
    'CLASS A': (50, 220, 50),     # Bright Green (highest quality)
    'CLASS B': (0, 200, 255),     # Orange (high quality)
    'CLASS C': (255, 200, 0),     # Cyan-gold (medium)
    'CLASS D': (200, 150, 238),   # Light violet (lower)
    'CLASS E': (0, 100, 255),     # Red (lowest quality)
}

# UI Settings
SIDEBAR_W = 290
FONT_MAIN = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'═══════════════════════════════════════════════')
print(f'  Buyo Leaf Grade Classifier')
print(f'═══════════════════════════════════════════════')
print(f'  Device: {device}')
if torch.cuda.is_available():
    print(f'  GPU   : {torch.cuda.get_device_name(0)}')
print(f'  Classes: {len(CLASS_NAMES)} (A through E)')
print(f'═══════════════════════════════════════════════')

# Model loader
def load_classifier(path, num_classes, dev):
    ckpt = torch.load(path, map_location=dev)
    bb = ckpt.get('backbone', 'resnet50')
    
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
        m = models.mobilenet_v2(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.last_channel, num_classes),
        )
    
    m.load_state_dict(ckpt['state_dict'])
    return m, bb

# Load model
num_classes = len(CLASS_NAMES)
if os.path.exists(MODEL_PATH):
    model, backbone = load_classifier(MODEL_PATH, num_classes, device)
    print(f'✓ Loaded: {MODEL_PATH} ({backbone})')
elif os.path.exists(FALLBACK_PATH):
    model, backbone = load_classifier(FALLBACK_PATH, num_classes, device)
    print(f'✓ Loaded: {FALLBACK_PATH} ({backbone})')
else:
    raise FileNotFoundError('❌ No model found! Train first.')

model = model.to(device)
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess_frame(bgr_frame):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return preprocess(pil).unsqueeze(0).to(device)

def preprocess_pil(pil_img):
    return preprocess(pil_img.convert('RGB')).unsqueeze(0).to(device)

# Temporal smoothing
prob_history = deque(maxlen=SMOOTH_FRAMES)
_smooth_lock = threading.Lock()

def get_smoothed_probs(raw_probs):
    with _smooth_lock:
        prob_history.append(raw_probs)
        return np.mean(prob_history, axis=0)

def reset_smoothing():
    with _smooth_lock:
        prob_history.clear()

# Confidence check
def is_confident(probs):
    """Check if the prediction is confident enough"""
    max_prob = float(np.max(probs))
    return max_prob >= CONF_THRESH

# Threaded inference
class InferenceEngine:
    def __init__(self):
        self._in_q = queue.Queue(maxsize=INFER_QUEUE_SIZE)
        self._result = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
    
    def put_frame(self, bgr_frame):
        try:
            self._in_q.put_nowait(bgr_frame.copy())
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
            
            tensor = preprocess_frame(frame)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            
            smooth = get_smoothed_probs(probs)
            pred_idx = int(np.argmax(smooth))
            confidence = float(smooth[pred_idx])
            is_conf = is_confident(smooth)
            
            with self._lock:
                self._result = (smooth, pred_idx, confidence, is_conf)

# Toast notifications
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
            cv2.putText(frame, text, (12, y + 1), FONT_SMALL, 0.55,
                       (0, 0, 0), 2, cv2.LINE_AA)
            faded_color = tuple(int(c * alpha) for c in color)
            cv2.putText(frame, text, (12, y), FONT_SMALL, 0.55,
                       faded_color, 1, cv2.LINE_AA)

toast = Toast()

# Scan-box reticle
def draw_reticle(frame, is_conf, pred_name, t):
    h, w = frame.shape[:2]
    cam_w = w - SIDEBAR_W
    
    cx, cy = cam_w // 2, h // 2
    box_w, box_h = int(cam_w * 0.55), int(h * 0.60)
    x1, y1 = cx - box_w // 2, cy - box_h // 2
    x2, y2 = cx + box_w // 2, cy + box_h // 2
    
    if not is_conf:
        base_color = (100, 100, 100)  # Gray for uncertain
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
    cv2.line(frame, (cx - ch_len, cy), (cx + ch_len, cy),
            color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - ch_len), (cx, cy + ch_len),
            color, 1, cv2.LINE_AA)

# HUD drawing
def draw_hud(frame, pred_name, confidence, all_probs, is_conf, fps, mode='LIVE', t=0.0):
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
    
    # Probability bars
    bar_w, bar_h, row_gap, y_start = SIDEBAR_W - 28, 14, 42, 92
    
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
    if not is_conf:
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
        cv2.putText(frame, 'U upload photo',
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

# File upload
def open_file_dialog():
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

def classify_image_file(path):
    try:
        pil_img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f'ERROR loading image: {e}')
        return
    
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

# Main camera loop
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f'ERROR: Cannot open camera {CAMERA_INDEX}')
        return
    
    print('\n✓ Camera started')
    print('  Q=quit | S=save | U=upload photo | R=reset smoothing\n')
    
    engine = InferenceEngine()
    
    win_name = 'Buyo Leaf Grade Classifier'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    
    prev_time = time.time()
    screenshot_idx = 0
    t_start = time.time()
    
    # Placeholder result
    n_cls = len(CLASS_NAMES)
    result = (np.ones(n_cls) / n_cls, 0, 1.0 / n_cls, False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('ERROR: Failed to grab frame')
            break
        
        engine.put_frame(frame)
        
        fresh = engine.get_result()
        if fresh is not None:
            result = fresh
        
        smooth, pred_idx, confidence, is_conf = result
        t_now = time.time()
        fps = 1.0 / max(t_now - prev_time, 1e-6)
        prev_time = t_now
        t = t_now - t_start
        
        pred_name = CLASS_NAMES[pred_idx]
        draw_hud(frame, pred_name, confidence, smooth, is_conf, fps, 'LIVE', t)
        draw_reticle(frame, is_conf, pred_name, t)
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
    
    engine.stop()
    cap.release()
    cv2.destroyAllWindows()
    print('\n✓ Camera classifier closed')

if __name__ == '__main__':
    main()
