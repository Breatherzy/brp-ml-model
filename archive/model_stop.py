import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# Assuming your load_data function loads the data as you want it
from saving import load_data, read_data_from_files_and_create_flags

# Lista ścieżek do plików
file_paths = ['reszta.txt', 'zatrzymanie.txt']

# Wywołanie funkcji i otrzymanie danych
data, result = read_data_from_files_and_create_flags(file_paths)

print(data[:10])
for segment in data:
    amp = max(segment[0]) - min(segment[0])
    for i in range(len(segment[0])):
        segment[0][i] = segment[0][i] * amp
print(data[:10])
# Convert data to numpy arrays and compute amplitude

X = np.array(data).reshape(-1, 10)
y = np.array(result)

# Definicja modelu sieci neuronowej
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=len(X[0])))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Kompilacja modelu
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Trenowanie modelu
model.fit(X, y, epochs=2000, batch_size=32)

# Save the model to a file
model.save("stopNetwork.keras")
