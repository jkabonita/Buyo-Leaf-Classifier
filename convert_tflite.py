import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model("buyo_classifier_5class.keras")

# Initialize TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply standard optimizations (reduces model size & latency)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the converted model
with open("buyo_classifier_5class.tflite", "wb") as f:
    f.write(tflite_model)

print("[SUCCESS] TFLite model generated: 'buyo_classifier_5class.tflite'")
