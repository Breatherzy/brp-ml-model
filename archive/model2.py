import numpy as np
import tensorflow as tf
from tensorflow import keras

# Assuming your load_data function loads the data as you want it
from saving import load_data

data = []
result = []
with open('../data_categories_2/data_to_train.txt') as f:
    data_ = f.readlines()
    for line in data_:
        data.append([float(value) for value in line.strip().split(" ")[:6]])
        result.append(int(line.strip().split(" ")[-1]))
# Convert data to numpy arrays and compute amplitude

x_train = np.array(data).reshape(-1, 6)
y_train = np.array(result)

# Create the neural network model
model = keras.Sequential(
    [
        keras.layers.Dense(
            3, input_shape=(6,), activation="tanh"
        ),  # First hidden layer with 4 neurons (or you can choose another number)
        keras.layers.Dense(1, activation="tanh"),  # Output layer with 1 neuron
    ]
)

# Compile the model with the optimizer, loss, and metrics
model.compile(
    optimizer=keras.optimizers.legacy.SGD(learning_rate=0.3), loss="mean_squared_error"
)

# Train the model
model.fit(x_train, y_train, epochs=500)

# Save the model to a file
model.save("newNetwork.h5")
