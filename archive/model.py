import numpy as np
import tensorflow as tf
from tensorflow import keras

# Assuming your load_data function loads the data as you want it
from saving import load_data

data, result = load_data(
    [
        ["saved_data/wdechDebug2.txt", 1],
        ["saved_data/wydechDebug2.txt", -1],
        ["saved_data/bezdechDebug2.txt", 0],
    ]
)
print(data[:10])
for segment in data:
    amp = max(segment[0]) - min(segment[0])
    for i in range(len(segment[0])):
        segment[0][i] = segment[0][i] * amp
print(data[:10])
# Convert data to numpy arrays and compute amplitude

x_train = np.array(data).reshape(-1, 5)
y_train = np.array(result)

# Create the neural network model
model = keras.Sequential(
    [
        keras.layers.Dense(
            3, input_shape=(5,), activation="tanh"
        ),  # First hidden layer with 4 neurons (or you can choose another number)
        keras.layers.Dense(1, activation="tanh"),  # Output layer with 1 neuron
    ]
)

# Compile the model with the optimizer, loss, and metrics
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.3), loss="mean_squared_error"
)

# Train the model
model.fit(x_train, y_train, epochs=5000)

# Save the model to a file
model.save("newNetwork.h5")
