import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

from load_data import load_data_from_file

data = load_data_from_file('data_to_train.txt')

# Oddzielenie danych od etykiet
X = data[:, :-1]  # Wszystkie cechy oprócz ostatniej kolumny
y = data[:, -1]   # Ostatnia kolumna jako etykieta

# Mapowanie etykiet z -1, 0, 1 do 0, 1, 2
y = y + 1

# Reshape X do formatu [samples, time_steps, features]
# Zakładamy, że każdy wiersz to jedna sekwencja (time_step) z sześcioma cechami
X = np.expand_dims(X, axis=1)

# Parametry modelu
num_classes = 3  # Liczba klas wyjściowych
num_features = X.shape[2]  # Liczba cech na sekwencję

# Budowa modelu 1D CNN bez MaxPooling1D
model = Sequential([
    Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, num_features)),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trening modelu
model.fit(X, y, epochs=100, batch_size=32)


predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

print(predicted_classes[:10])
print(y[:10])

model.save('Models/CONV_model.h5')  # Zapisuje model w formacie HDF5