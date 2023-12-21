import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from load_data import load_data_from_file

data = load_data_from_file('data_to_train.txt')

def LSTM_model(data):
    # Oddzielenie danych od etykiet
    X = data[:, :-1]  # Wszystkie cechy oprócz ostatniej kolumny
    y = data[:, -1]   # Ostatnia kolumna jako etykieta

    # Mapowanie etykiet z -1, 0, 1 do 0, 1, 2
    y = y + 1

    # Przygotowanie danych do LSTM - wymagany format [samples, time steps, features]
    X = np.expand_dims(X, axis=1)

    # Parametry modelu
    num_classes = 3  # Liczba klas (0, 1, 2)

    # Budowa modelu LSTM
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        Dense(10, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Kompilacja modelu
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Trening modelu
    model.fit(X, y, epochs=100)
    return model

model = LSTM_model(data)

X = data[:, :-1]
y = data[:, -1]
y = y + 1
X = np.expand_dims(X, axis=1)

predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

print(predicted_classes[:10])
print(y[:10])

model.save('Models/LSTM_model.h5')  # Zapisuje model w formacie HDF5
