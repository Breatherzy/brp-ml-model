import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from load_data import load_data_from_file

data = load_data_from_file('data_to_train.txt')
X = data[:, :-1]  # Wszystkie kolumny oprócz ostatniej
y = data[:, -1]  # Ostatnia kolumna

def BN_model(data):
    # Podział danych na cechy i etykiety
    X = data[:, :-1]  # Wszystkie kolumny oprócz ostatniej
    y = data[:, -1]  # Ostatnia kolumna

    # Kodowanie etykiet (klas)
    y_encoded = tf.keras.utils.to_categorical(y, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Tworzenie modelu
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=(X.shape[1],), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(3, activation='softmax')  # Wyjście dla 3 klas
    ])

    # Kompilacja modelu
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Trenowanie modelu
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    return model

model = BN_model(data)

# Predykcje i ocena modelu
predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

print(predicted_classes[:10])
print(y[:10])

print(np.argmax(model.predict([[0.2, 0.3, 0.4, 0.5, 0.6, 0.4]]), axis=1))

# Zapis modelu
model.save('Models/BN_model.h5')
