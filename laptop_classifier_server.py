#!/usr/bin/env python3
"""
Laptop Classifier Server - GPU Processing
==========================================
Runs on your laptop with GPU to process frames from Raspberry Pi camera
and send robotic arm trigger commands back.

Setup:
1. Install: pip install torch torchvision opencv-python pillow numpy flask
2. Make sure buyo_best.pth is in same directory
3. Run: python laptop_classifier_server.py
4. Note the laptop IP address shown
5. On Raspberry Pi, run: python raspberry_pi_camera_client.py <LAPTOP_IP>

Architecture:
  Raspberry Pi Camera → Network → Laptop GPU → Classification → 
  Trigger Command → Network → Raspberry Pi Robotic Arm
"""

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from collections import deque
import time
import glob
from flask import Flask, request, jsonify
import threading
import base64
import io

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*64)
print("  Buyo Classifier Server - Laptop GPU")
print("="*64)
print(f"  Device: {device}")

# Find model
pth_files = glob.glob('*.pth')
if not pth_files:
    print("  ❌ No .pth model files found!")
    exit(1)

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

# Build architecture
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
model = model.to(device)
model.eval()
print(f"  Classes: {len(CLASS_NAMES)}")
print("="*64)

# ============================================================================
#  PREPROCESSING & DETECTION
# ============================================================================
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

prob_history = deque(maxlen=SMOOTH_FRAMES)

def get_smoothed_probs(raw_probs):
    prob_history.append(raw_probs)
    return np.mean(prob_history, axis=0)

def reset_smoothing():
    prob_history.clear()

def is_confident(probs):
    return float(np.max(probs)) >= CONF_THRESH

def is_buyo_leaf(probs):
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
#  GUI DRAWING
# ============================================================================
latest_frame = None
latest_result = None
frame_lock = threading.Lock()

def draw_hud(frame, pred_name, confidence, all_probs, is_conf, fps, is_buyo, buyo_reason):
    h, w = frame.shape[:2]
    x0 = w - SIDEBAR_W + 14
    
    # Sidebar
    sidebar = frame[:, w - SIDEBAR_W:]
    blurred = cv2.GaussianBlur(sidebar, (31, 31), 0)
    dark = cv2.addWeighted(blurred, 0.25, np.zeros_like(blurred), 0.75, 0)
    frame[:, w - SIDEBAR_W:] = dark
    cv2.line(frame, (w - SIDEBAR_W, 0), (w - SIDEBAR_W, h), (60, 60, 60), 1)
    
    # Header
    cv2.putText(frame, 'BUYO LEAF', (x0, 30), FONT_MAIN, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, 'GRADING', (x0, 52), FONT_MAIN, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.line(frame, (x0, 62), (w - 10, 62), (70, 70, 70), 1)
    
    cv2.putText(frame, 'GPU SERVER', (x0, 76), FONT_SMALL, 0.36, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(frame, '● REMOTE', (x0 + 142, 76), FONT_SMALL, 0.37, (60, 220, 100), 1, cv2.LINE_AA)
    
    # Buyo status
    if is_buyo:
        cv2.putText(frame, '✓ BUYO', (x0, 88), FONT_SMALL, 0.32, (100, 255, 100), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, '✗ NOT BUYO', (x0, 88), FONT_SMALL, 0.32, (100, 100, 255), 1, cv2.LINE_AA)
    
    # Probability bars
    bar_w, bar_h, row_gap, y_start = SIDEBAR_W - 28, 14, 42, 100
    
    for i, name in enumerate(CLASS_NAMES):
        y = y_start + i * row_gap
        prob = float(all_probs[i])
        color = CLASS_COLORS[name]
        is_pred = (name == pred_name and is_conf)
        
        cv2.putText(frame, name, (x0, y), FONT_SMALL, 0.43, color if is_pred else (190, 190, 190), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y + 4), (x0 + bar_w, y + 4 + bar_h), (45, 45, 45), -1)
        cv2.rectangle(frame, (x0, y + 4), (x0 + bar_w, y + 4 + bar_h), (75, 75, 75), 1)
        
        fill = max(1, int(bar_w * prob))
        cv2.rectangle(frame, (x0, y + 4), (x0 + fill, y + 4 + bar_h), color, -1)
        cv2.putText(frame, f'{prob * 100:.1f}%', (x0 + bar_w - 48, y + 15), FONT_SMALL, 0.37, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Result
    y_div = y_start + len(CLASS_NAMES) * row_gap + 10
    cv2.line(frame, (x0, y_div), (w - 10, y_div), (70, 70, 70), 1)
    
    if not is_buyo:
        pred_txt, pred_col = 'NOT BUYO', (100, 100, 255)
        conf_txt = buyo_reason
    elif not is_conf:
        pred_txt, pred_col = 'UNCERTAIN', (100, 100, 200)
        conf_txt = f'{confidence * 100:.0f}%'
    else:
        pred_txt, pred_col = pred_name, CLASS_COLORS[pred_name]
        conf_txt = f'{confidence * 100:.1f}%'
    
    cv2.putText(frame, pred_txt, (x0, y_div + 40), FONT_MAIN, 0.90, pred_col, 2, cv2.LINE_AA)
    cv2.putText(frame, conf_txt, (x0, y_div + 60), FONT_SMALL, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Footer
    cv2.line(frame, (x0, h - 38), (w - 10, h - 38), (55, 55, 55), 1)
    fps_col = (80, 220, 80) if fps >= 20 else (80, 180, 255) if fps >= 10 else (60, 60, 220)
    cv2.putText(frame, f'FPS {fps:.1f}', (x0, h - 22), FONT_SMALL, 0.40, fps_col, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Q quit  R reset', (x0, h - 8), FONT_SMALL, 0.33, (85, 85, 85), 1, cv2.LINE_AA)

def gui_thread():
    """Display GUI on laptop"""
    global latest_frame, latest_result
    
    win_name = 'Buyo Classifier Server (Laptop GPU)'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    
    fps_counter = deque(maxlen=30)
    
    while True:
        t0 = time.time()
        
        with frame_lock:
            if latest_frame is not None and latest_result is not None:
                frame = latest_frame.copy()
                result = latest_result.copy()
            else:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, 'Waiting for Raspberry Pi connection...', (400, 360),
                           FONT_MAIN, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
                cv2.imshow(win_name, frame)
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                continue
        
        probs = result['probs']
        pred_idx = result['pred_idx']
        confidence = result['confidence']
        is_conf = result['is_confident']
        is_buyo = result['is_buyo']
        buyo_reason = result['buyo_reason']
        pred_name = CLASS_NAMES[pred_idx]
        
        fps_counter.append(1.0 / max(0.001, time.time() - t0))
        fps = np.mean(fps_counter)
        
        display = frame.copy()
        draw_hud(display, pred_name, confidence, probs, is_conf, fps, is_buyo, buyo_reason)
        
        cv2.imshow(win_name, display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_smoothing()
            print('[RESET] Smoothing buffer cleared')
    
    cv2.destroyAllWindows()

# ============================================================================
#  FLASK API
# ============================================================================
app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    """Receive frame from Raspberry Pi, classify, return result + trigger command"""
    try:
        data = request.json
        img_data = base64.b64decode(data['frame'])
        
        # Decode image
        nparr = np.frombuffer(img_data, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Inference
        pil_img = Image.fromarray(frame_rgb)
        tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        smooth = get_smoothed_probs(probs)
        pred_idx = int(np.argmax(smooth))
        confidence = float(smooth[pred_idx])
        is_conf = is_confident(smooth)
        is_buyo, buyo_reason = is_buyo_leaf(smooth)
        pred_name = CLASS_NAMES[pred_idx]
        
        # Determine if should trigger arm
        should_trigger = is_conf and is_buyo
        
        # Update GUI
        global latest_frame, latest_result
        with frame_lock:
            latest_frame = frame_bgr
            latest_result = {
                'probs': smooth,
                'pred_idx': pred_idx,
                'confidence': confidence,
                'is_confident': is_conf,
                'is_buyo': is_buyo,
                'buyo_reason': buyo_reason,
            }
        
        # Log
        status = "✓" if should_trigger else ("✗" if not is_buyo else "?")
        print(f"[{status}] {pred_name} ({confidence*100:.1f}%) | Trigger: {should_trigger}")
        
        return jsonify({
            'class': pred_name,
            'class_idx': pred_idx,
            'confidence': confidence,
            'probabilities': smooth.tolist(),
            'is_confident': is_conf,
            'is_buyo': is_buyo,
            'buyo_reason': buyo_reason,
            'trigger_arm': should_trigger,
        })
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Reset smoothing buffer"""
    reset_smoothing()
    print('[RESET] Smoothing buffer cleared')
    return jsonify({'status': 'reset'})

@app.route('/status', methods=['GET'])
def status():
    """Health check"""
    return jsonify({
        'status': 'running',
        'device': str(device),
        'model': MODEL_PATH,
        'backbone': bb,
    })

# ============================================================================
#  MAIN
# ============================================================================
if __name__ == '__main__':
    import socket
    
    # Get laptop IP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        laptop_ip = s.getsockname()[0]
    except:
        laptop_ip = '127.0.0.1'
    finally:
        s.close()
    
    print("\n" + "="*64)
    print("  🚀 SERVER READY")
    print("="*64)
    print(f"  Laptop IP: {laptop_ip}")
    print(f"  Port: 5000")
    print(f"  Device: {device}")
    print("="*64)
    print("\n  On Raspberry Pi, run:")
    print(f"    python raspberry_pi_camera_client.py {laptop_ip}")
    print("\n" + "="*64 + "\n")
    
    # Start GUI thread
    gui = threading.Thread(target=gui_thread, daemon=True)
    gui.start()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
