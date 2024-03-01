import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from AbstractModel import AbstractModel


class BNModel(AbstractModel):

    def compile(self):
        if self.check_if_data_is_loaded():
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, input_shape=(
                    self.X.shape[1],), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                # Wyjście dla 3 klas
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            self.model.compile(
                optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self):
        if self.check_if_model_is_compiled():
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42)
            y_train_encoded = tf.keras.utils.to_categorical(
                self.y_train, num_classes=3)
            self.model.fit(self.X_train, y_train_encoded, epochs=100,
                           batch_size=32, validation_data=(self.X_test, self.y_test))

    def predict(self, X_test):
        if self.check_if_model_is_fitted():
            return np.argmax(self.model.predict(X_test), axis=1)

    def evaluate(self):
        y_pred = self.predict(self.X_test)
        print('Accuracy:', np.mean(y_pred == self.y_test))

    def save(self, filename):
        self.model.save(filename)
