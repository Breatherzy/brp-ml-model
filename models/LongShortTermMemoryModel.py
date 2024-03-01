import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from AbstractModel import AbstractModel


class LSTMModel(AbstractModel):
    def compile(self):
        if self.check_if_data_is_loaded():
            self.model = Sequential([
                LSTM(50, activation='relu', input_shape=(
                    self.X.shape[1], self.X.shape[2])),
                Dense(10, activation='relu'),
                Dense(3, activation='softmax')
            ])
            self.model.compile(
                optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self):
        if self.check_if_model_is_compiled():
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42)
            self.model.fit(self.X_train, self.y_train, epochs=100,
                           batch_size=32, validation_data=(self.X_test, self.y_test))

    def predict(self, X_test):
        if self.check_if_model_is_fitted():
            return np.argmax(self.model.predict(X_test), axis=1)

    def evaluate(self):
        y_pred = self.predict(self.X_test)
        print('Accuracy:', np.mean(y_pred == self.y_test))

    def save(self, filename):
        self.model.save(filename)
