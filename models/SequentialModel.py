from abc import ABC, ABCMeta, abstractmethod

import numpy as np
from keras.models import load_model

from models.AbstractModel import AbstractModel
from scripts.plot import interactive_plot


class SequentialModel(AbstractModel, ABC, metaclass=ABCMeta):
    @abstractmethod
    def compile(self):
        """
        Metoda do kompilowania modelu sekwencyjnego.
        """
        super().compile()

    def fit(self, epochs=100, batch_size=32):
        if self.check_if_model_is_compiled():
            history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.X_test, self.y_test),
            )
            history = history.history
            with open(
                "models/saves/" + self.__class__.__name__ + ".history", "a"
            ) as file:
                file.write(str(history) + "\n")
            self.is_model_fitted = True

    def predict(self, X_test):
        if self.check_if_model_is_fitted():
            return np.argmax(self.model.predict(X_test), axis=1)

    def plot_prediction(self, X_test, title=None) -> None:
        interactive_plot(
            self.X_test[:, -1, 0], self.predict(X_test), self.y_test, title=title)

    def evaluate(self, X_test=None, y_test=None):
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        y_pred = self.predict(X_test)
        if len(y_test.shape) > 1:
            y_test = np.argmax(self.y_test, axis=1)
        result = np.mean(y_pred == y_test)
        print("Evaluation result:", result)
        return result

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)
        self.is_model_loaded = True
