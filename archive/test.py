import numpy as np
import tensorflow as tf
from tensorflow import keras
import coremltools as ct

# Load your Keras model
model = keras.models.load_model("stopNetwork.keras")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the TFLite model to a file
with open("stopNetwork.tflite", "wb") as f:
    f.write(tflite_quant_model)

# Define the input shape if your model uses flexible input sizes,
# otherwise, specify the exact input shape.
input_shape = (1, 5)  # Example fixed input shape. Replace with your actual input shape.
inputs = [ct.TensorType(shape=input_shape)]  # Use ct.RangeDim() for flexible dimensions if needed.

# Convert the model to Core ML format with specified input shape
coreml_model = ct.convert(model, source="tensorflow", inputs=inputs)

# Save the Core ML model to a file
coreml_model.save("stopNetwork.mlpackage")
