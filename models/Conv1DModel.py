import numpy as np
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model

from models.AbstractModel import AbstractModel


class Conv1DModel(AbstractModel):
    def compile(self):
        if self.check_if_data_is_loaded():
            for dataset_name in ['X', 'X_train', 'X_test']:
                dataset = getattr(self, dataset_name)
                setattr(self, dataset_name, np.expand_dims(dataset, axis=1))

            self.model = Sequential([
                Conv1D(filters=64, kernel_size=1, activation='relu',
                       input_shape=(1, self.X.shape[2])),
                Flatten(),
                Dense(50, activation='relu'),
                Dense(3, activation='softmax')
            ])
            self.model.compile(
                optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self):
        if self.check_if_model_is_compiled():
            history = self.model.fit(self.X_train, self.y_train, epochs=100,
                                     batch_size=32, validation_data=(self.X_test, self.y_test))
            history = history.history
            with open('models/saves/' + self.__class__.__name__ + '.history', 'w') as file:
                file.write(str(history))

    def predict(self, X_test):
        if self.check_if_model_is_fitted():
            if len(X_test.shape) == 2:
                X_test = np.expand_dims(X_test, axis=1)
            return np.argmax(self.model.predict(X_test), axis=1)

    def evaluate(self):
        y_pred = self.predict(self.X_test)
        print('Accuracy:', np.mean(y_pred == self.y_test))

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)
        self.is_model_loaded = True
