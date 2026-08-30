import os
import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. HYPERPARAMETERS ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

# --- 2. DATA PIPELINE WITH AUGMENTATION ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Store class names for later evaluation if needed
class_names = train_ds.class_names
print("Class names:", class_names)

# Optimize pipeline I/O
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation layers to improve robustness against lighting/angles
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# --- 3. MODEL ARCHITECTURE (MobileNetV2) ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze pre-trained weights

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
# 5 classes for Multi-class classification (Class A to E)
outputs = layers.Dense(5, activation="softmax")(x)

model = models.Model(inputs, outputs)

# --- 4. COMPILATION & TRAINING ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save standard Keras model
model.save("buyo_classifier_5class.keras")
print("[SUCCESS] Model trained and saved as 'buyo_classifier_5class.keras'")
