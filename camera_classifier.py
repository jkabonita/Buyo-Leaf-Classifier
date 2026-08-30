"""
camera_classifier.py -- Real-Time Buyo Leaf Classifier (Camera + Photo Upload)
===============================================================================
Loads the best trained model (buyo_best.pth / ResNet50 or EfficientNetV2-S).
Falls back to buyo_pytorch.pth (MobileNetV2) if needed.

Features:
  [*] Temporal smoothing (rolling 7-frame average) -- no jitter
  [*] Confidence threshold -- shows "???" if confidence < 60%
  [*] Per-class probability bars on sidebar
  [*] GPU-accelerated inference when available
  [*] Clean HUD overlay with color-coded class labels
  [*] Upload photo from disk for classification (press U)

Controls:
  Q -- quit
  S -- save current frame as screenshot
  U -- upload a photo file and classify it

Usage:
    python camera_classifier.py
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
import sys
import tkinter as tk
from tkinter import filedialog

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_PATH    = 'buyo_best.pth'
FALLBACK_PATH = 'buyo_pytorch.pth'

CLASS_NAMES   = ['CLASS A', 'CLASS B', 'CLASS C', 'CLASS D', 'CLASS E']
IMG_SIZE      = 224
CONF_THRESH   = 0.55    # Hide label if confidence < this
SMOOTH_FRAMES = 7       # Rolling average over N frames
CAMERA_INDEX  = 0       # 0 = default webcam

# BGR color per class
CLASS_COLORS = {
    'CLASS A': (50,  200,  50),   # Green
    'CLASS B': (0,   165, 255),   # Orange
    'CLASS C': (255, 191,  0),    # Deep sky blue (BGR)
    'CLASS D': (238, 130, 238),   # Violet
    'CLASS E': (0,   100, 255),   # Red
}

# Sidebar width used by HUD
SIDEBAR_W  = 270
FONT_MAIN  = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

# -----------------------------------------------------------------------------
# DEVICE SETUP
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Inference device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# -----------------------------------------------------------------------------
# MODEL LOADER
# -----------------------------------------------------------------------------
def load_classifier(path: str, num_classes: int, dev: torch.device):
    ckpt = torch.load(path, map_location=dev)
    bb = ckpt.get('backbone', 'resnet50')

    if bb == 'resnet50':
        m = models.resnet50(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
    elif bb == 'efficientnet_v2_s':
        m = models.efficientnet_v2_s(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.35, inplace=True),
            nn.Linear(in_f, num_classes),
        )
    else:
        m = models.mobilenet_v2(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.last_channel, num_classes),
        )

    m.load_state_dict(ckpt['state_dict'])
    return m, bb

num_classes = len(CLASS_NAMES)
if os.path.exists(MODEL_PATH):
    model, backbone = load_classifier(MODEL_PATH, num_classes, device)
    print(f'Loaded model from {MODEL_PATH} ({backbone})')
elif os.path.exists(FALLBACK_PATH):
    model, backbone = load_classifier(FALLBACK_PATH, num_classes, device)
    print(f'[Fallback] Loaded model from {FALLBACK_PATH} ({backbone})')
else:
    raise FileNotFoundError('No model file found! Run train_gpu.py first.')

model = model.to(device)
model.eval()

# -----------------------------------------------------------------------------
# PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess_frame(bgr_frame: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return preprocess(pil).unsqueeze(0).to(device)

def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    return preprocess(pil_img.convert('RGB')).unsqueeze(0).to(device)

# -----------------------------------------------------------------------------
# TEMPORAL SMOOTHING
# -----------------------------------------------------------------------------
prob_history = deque(maxlen=SMOOTH_FRAMES)

def get_smoothed_probs(raw_probs: np.ndarray) -> np.ndarray:
    prob_history.append(raw_probs)
    return np.mean(prob_history, axis=0)

def reset_smoothing():
    prob_history.clear()

# -----------------------------------------------------------------------------
# HUD DRAWING
# -----------------------------------------------------------------------------
def draw_hud(frame: np.ndarray, pred_name: str, confidence: float,
             all_probs: np.ndarray, uncertain: bool, fps: float,
             mode: str = 'LIVE') -> None:
    h, w = frame.shape[:2]
    x0   = w - SIDEBAR_W + 12

    # Blurred dark sidebar background
    sidebar = frame[:, w - SIDEBAR_W:].copy()
    blurred = cv2.GaussianBlur(sidebar, (21, 21), 0)
    dark    = (blurred * 0.35).astype(np.uint8)
    frame[:, w - SIDEBAR_W:] = dark

    # Header
    cv2.putText(frame, 'BUYO LEAF',   (x0, 28),  FONT_MAIN,  0.62, (230, 230, 230), 1)
    cv2.putText(frame, 'CLASSIFIER',  (x0, 50),  FONT_MAIN,  0.62, (230, 230, 230), 1)
    cv2.line(frame, (x0, 60), (w - 10, 60), (90, 90, 90), 1)
    cv2.putText(frame, f'{backbone.upper()}', (x0, 73), FONT_SMALL, 0.35, (120, 120, 120), 1)

    # Mode badge
    badge_color = (0, 200, 100) if mode == 'LIVE' else (200, 100, 0)
    cv2.putText(frame, mode, (w - SIDEBAR_W + 160, 73), FONT_SMALL, 0.38, badge_color, 1)

    # Probability bars
    bar_w   = SIDEBAR_W - 24
    bar_h   = 13
    row_gap = 40
    y_start = 88

    for i, name in enumerate(CLASS_NAMES):
        y     = y_start + i * row_gap
        prob  = float(all_probs[i])
        color = CLASS_COLORS[name]

        label_color = color if (name == pred_name and not uncertain) else (200, 200, 200)
        cv2.putText(frame, name, (x0, y), FONT_SMALL, 0.42, label_color, 1)

        cv2.rectangle(frame, (x0, y + 4), (x0 + bar_w, y + 4 + bar_h), (55, 55, 55), -1)
        cv2.rectangle(frame, (x0, y + 4), (x0 + bar_w, y + 4 + bar_h), (80, 80, 80), 1)

        fill = max(1, int(bar_w * prob))
        cv2.rectangle(frame, (x0, y + 4), (x0 + fill, y + 4 + bar_h), color, -1)

        pct_txt = f'{prob*100:.1f}%'
        cv2.putText(frame, pct_txt, (x0 + bar_w - 46, y + 15), FONT_SMALL, 0.37, (255, 255, 255), 1)

    # Divider
    y_div = y_start + len(CLASS_NAMES) * row_gap + 8
    cv2.line(frame, (x0, y_div), (w - 10, y_div), (90, 90, 90), 1)

    # Prediction result
    if uncertain:
        pred_txt = '???'
        pred_col = (170, 170, 170)
        conf_txt = 'Low confidence'
    else:
        pred_txt = pred_name
        pred_col = CLASS_COLORS[pred_name]
        conf_txt = f'{confidence*100:.1f}% confident'

    cv2.putText(frame, pred_txt, (x0, y_div + 32), FONT_MAIN,  0.85, pred_col, 2)
    cv2.putText(frame, conf_txt, (x0, y_div + 54), FONT_SMALL, 0.40, (170, 170, 170), 1)

    # Footer controls or FPS
    if mode == 'LIVE':
        cv2.putText(frame, f'FPS: {fps:.1f}', (x0, h - 46), FONT_SMALL, 0.38, (100, 100, 100), 1)
        cv2.putText(frame, 'Q=quit  S=save', (x0, h - 30), FONT_SMALL, 0.35, (90, 90, 90), 1)
        cv2.putText(frame, 'U=upload photo', (x0, h - 14), FONT_SMALL, 0.35, (90, 90, 90), 1)
    else:
        cv2.putText(frame, 'Q=close  S=save', (x0, h - 14), FONT_SMALL, 0.35, (90, 90, 90), 1)

    # Border
    if not uncertain:
        border_color = CLASS_COLORS[pred_name]
        thick = 3
        cv2.rectangle(frame, (thick, thick), (w - SIDEBAR_W - thick, h - thick), border_color, thick)

# -----------------------------------------------------------------------------
# FILE UPLOAD CLASSIFIER
# -----------------------------------------------------------------------------
def open_file_dialog() -> str | None:
    """Open a Tkinter file dialog and return the selected image path."""
    root = tk.Tk()
    root.withdraw()          # Hide the root window
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title='Select a Buyo Leaf Image',
        filetypes=[
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.webp'),
            ('All files', '*.*'),
        ]
    )
    root.destroy()
    return path if path else None


def classify_image_file(path: str) -> None:
    """Load an image from disk, classify it, and display a result window."""
    # Load with PIL
    try:
        pil_img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f'ERROR loading image: {e}')
        return

    # Run inference
    tensor = preprocess_pil(pil_img)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    pred_name  = CLASS_NAMES[pred_idx]
    uncertain  = confidence < CONF_THRESH

    # Console output
    print(f'\n--- Photo Classification ---')
    print(f'  File      : {os.path.basename(path)}')
    print(f'  Prediction: {pred_name}  ({confidence*100:.2f}%)')
    for i, name in enumerate(CLASS_NAMES):
        bar = '#' * int(probs[i] * 20)
        print(f'  {name}: {probs[i]*100:5.1f}% |{bar:<20}|')
    print()

    # Convert PIL -> BGR for OpenCV display; target window = 1280 x 720
    target_w, target_h = 1280, 720
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Fit image into the camera area (left of sidebar)
    cam_w = target_w - SIDEBAR_W
    ih, iw = img_bgr.shape[:2]
    scale  = min(cam_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img_bgr, (nw, nh))

    # Canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off  = (target_h - nh) // 2
    x_off  = (cam_w - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized

    # Draw filename at top of image area
    fname_short = os.path.basename(path)
    if len(fname_short) > 40:
        fname_short = '...' + fname_short[-37:]
    cv2.putText(canvas, fname_short, (10, 22), FONT_SMALL, 0.5, (220, 220, 220), 1)

    # Draw HUD sidebar
    draw_hud(canvas, pred_name, confidence, probs, uncertain, fps=0.0, mode='PHOTO')

    # Show until Q or window closed
    win_name = f'Photo: {os.path.basename(path)}'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, target_w, target_h)
    cv2.imshow(win_name, canvas)

    # Save result image alongside the source photo
    result_path = os.path.splitext(path)[0] + '_result.jpg'
    cv2.imwrite(result_path, canvas)
    print(f'  Result saved: {result_path}')

    print('  Press Q to close the photo window.')
    while True:
        key = cv2.waitKey(30) & 0xFF
        # Check if window was closed by clicking X
        try:
            visible = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
        except Exception:
            visible = -1
        if key == ord('q') or visible < 1:
            break
        elif key == ord('s'):
            cv2.imwrite(result_path, canvas)
            print(f'  Saved: {result_path}')

    try:
        cv2.destroyWindow(win_name)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# CAMERA LOOP
# -----------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f'ERROR: Cannot open camera index {CAMERA_INDEX}.')
        return

    print('Camera started.')
    print('  Q = quit  |  S = save screenshot  |  U = upload & classify a photo')
    prev_time = time.time()
    screenshot_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('ERROR: Failed to grab frame.')
            break

        tensor = preprocess_frame(frame)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        smooth     = get_smoothed_probs(probs)
        pred_idx   = int(np.argmax(smooth))
        confidence = float(smooth[pred_idx])
        uncertain  = confidence < CONF_THRESH

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        draw_hud(frame, CLASS_NAMES[pred_idx], confidence, smooth, uncertain, fps, mode='LIVE')
        cv2.imshow('Buyo Leaf Classifier', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            fname = f'screenshot_{screenshot_idx:03d}.jpg'
            cv2.imwrite(fname, frame)
            print(f'Saved: {fname}')
            screenshot_idx += 1

        elif key == ord('u'):
            # Pause camera, open file dialog, classify photo
            print('Opening file dialog...')
            img_path = open_file_dialog()
            if img_path:
                reset_smoothing()          # Clear history so live feed re-warms after photo
                classify_image_file(img_path)
                reset_smoothing()
            else:
                print('No file selected.')

    cap.release()
    cv2.destroyAllWindows()
    print('Camera classifier closed.')


if __name__ == '__main__':
    main()
