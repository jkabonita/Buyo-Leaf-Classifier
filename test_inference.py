import cv2
import numpy as np
import tensorflow as tf
import os
import glob

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="buyo_classifier_5class.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['class_a', 'class_b', 'class_c', 'class_d', 'class_e']

# Find a test image
image_path = ""
for ext in ['*.jpg', '*.jpeg', '*.png']:
    files = glob.glob(f"dataset/val/*/{ext}")
    if files:
        image_path = files[0]
        break

if not image_path:
    print("No test image found in dataset/val/")
    exit()

print(f"Testing on image: {image_path}")

# Load and preprocess a test image
image = cv2.imread(image_path)
if image is None:
    print("Failed to load image")
    exit()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (224, 224))
image_data = np.expand_dims(image_resized, axis=0).astype(np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], image_data)

# Run inference
interpreter.invoke()

# Extract prediction
output_data = interpreter.get_tensor(output_details[0]['index'])
scores = output_data[0]
predicted_index = np.argmax(scores)
predicted_class = class_names[predicted_index]
confidence = scores[predicted_index] * 100

print("\n--- Predictions ---")
for i, name in enumerate(class_names):
    print(f"{name.upper()}: {scores[i]*100:.2f}%")

print(f"\nFinal Prediction: {predicted_class.upper()} | Confidence: {confidence:.2f}%")
