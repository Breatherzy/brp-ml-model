import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from saving import load_data

model = keras.models.load_model("networkTest.h5")
model.save("networkTest.keras")


# print(model.predict(np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])))
# print(model.predict(np.array([[0.9, 0.8, 0.7, 0.6, 0.5]])))
# print(model.predict(np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])))
# print(model.predict(np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])))
#
#
# data, result = load_data(
#     [
#         ["saved_data/wdechDebug2.txt", 1],
#         ["saved_data/wydechDebug2.txt", -1],
#         ["saved_data/bezdechDebug2.txt", 0],
#     ]
# )


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the model to a .tflite file
with open("networkTest.tflite", "wb") as f:
    f.write(tflite_quant_model)

import coremltools as ct

# Convert the model to Core ML format
coreml_model = ct.convert(model, source="tensorflow")

# Save the model to a .mlmodel file
coreml_model.save("networkTest.mlmodel")
