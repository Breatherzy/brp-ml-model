import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

from load_data import load_data_from_file

data = load_data_from_file('data_to_train.txt')

# Oddzielenie danych od etykiet
X = data[:, :-1]  # Wszystkie cechy oprócz ostatniej kolumny
y = data[:, -1]   # Ostatnia kolumna jako etykieta

# Mapowanie etykiet z -1, 0, 1 do 0, 1, 2
y = y + 1

# Przygotowanie danych do formatu wymaganego przez GRU
# Wymagany format: [próbki, kroki czasowe, cechy]
# Uwaga: Może być konieczne dostosowanie kształtu danych
X = np.expand_dims(X, axis=1)

# Parametry modelu
num_classes = 3  # Liczba klas wyjściowych

# Budowa modelu GRU
model = Sequential([
    GRU(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dense(50, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trening modelu
model.fit(X, y, epochs=100, batch_size=32)

# Predykcje i ocena modelu
predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

print(predicted_classes[:10])
print(y[:10])

# Zapis modelu
model.save('Models/Gru_model.h5')
