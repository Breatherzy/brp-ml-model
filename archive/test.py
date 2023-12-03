import numpy as np
import tensorflow as tf
from tensorflow import keras
import coremltools as ct

# Load your Keras model
model = keras.models.load_model("newNetwork.h5")

SIZE = 5
with open('../data_categories_2/raw_data_normalized.txt', 'r') as f:
    data = f.read().splitlines()
    data = [float(value) for value in data]

with open('../data_categories_2/raw_data_no_breath2.txt', 'r') as f:
    data = f.read().splitlines()
    data = [float(value.split(" ")[-1]) for value in data]


# data = data[:1000]
#
# mono_numbers = []
# previous = 0
# for i in range(SIZE, len(data)):
#     print(f'{i} of {len(data)}')
#     window = data[i - SIZE:i]
#     amplitude = max(window) - min(window)
#     window = window + [amplitude]
#     prediction = model.predict([window])
#     print(i, window, prediction)
#     if prediction >= 0.8 and amplitude >= 0.2:
#         mono_numbers.append(1)
#         previous = 1
#     elif prediction <= -0.8 and amplitude >= 0.2:
#         mono_numbers.append(-1)
#         previous = -1
#     elif prediction <= 0.3 and prediction >= -0.3 and amplitude < 0.2:
#         mono_numbers.append(0)
#         previous = 0
#     else:
#         mono_numbers.append(previous)
#
# from plot import interactive_plot
#
# print(len(data), len(mono_numbers))
# interactive_plot(data, mono_numbers)

print(model.predict(np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.4]])))
# print(model.predict(np.array([[0.9, 0.8, 0.7, 0.6, 0.5]])))
# print(model.predict(np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])))
# print(model.predict(np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])))

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the TFLite model to a file
with open("afterRandomForest.tflite", "wb") as f:
    f.write(tflite_quant_model)

# Define the input shape if your model uses flexible input sizes,
# otherwise, specify the exact input shape.
# input_shape = (1, 5)  # Example fixed input shape. Replace with your actual input shape.
# inputs = [ct.TensorType(shape=input_shape)]  # Use ct.RangeDim() for flexible dimensions if needed.

# Convert the model to Core ML format with specified input shape
coreml_model = ct.convert(model, source="tensorflow")

# Save the Core ML model to a file
coreml_model.save("afterRandomForest.mlmodel")
