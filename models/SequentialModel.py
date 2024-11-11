from abc import ABC, ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from models.AbstractModel import AbstractModel


class SequentialModel(AbstractModel, ABC, metaclass=ABCMeta):
    @abstractmethod
    def compile(self):
        """
        Method for compiling the sequential model.
        """
        super().compile()

    def fit(self, epochs=200, batch_size=500, sensor_type="hist"):
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
                f"models/saves/{sensor_type}/{self.__class__.__name__}.history", "w"
            ) as file:
                file.write(str(history) + "\n")
            self.is_model_fitted = True

    def predict(self, X_test):
        if self.check_if_model_is_fitted():
            return np.argmax(self.model.predict(X_test), axis=1)

    def evaluate(self, X_test=None, y_test=None):
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        y_pred = self.predict(X_test)
        if len(y_test.shape) > 1:
            y_test = np.argmax(self.y_test, axis=1)
        result = np.mean(y_pred == y_test)

        predicted_breaths = self.count_breaths(y_pred)

        actual_breaths = self.count_breaths(y_test)

        if actual_breaths == 0:
            accuracy = 100.0 if predicted_breaths == 0 else 0.0
        else:
            accuracy = (
                1 - abs(predicted_breaths - actual_breaths) / actual_breaths
            ) * 100
            accuracy = max(0.0, min(100.0, accuracy))

        print("Evaluation result:", result)
        print(f"Actual breaths: {actual_breaths}")
        print(f"Predicted breaths: {predicted_breaths}")
        print(f"Accuracy: {accuracy:.2f}%")
        return result

    def count_breaths(self, results: list[float]) -> int:
        """
        Count complete breaths from a file containing numbers and predictions.
        A complete breath consists of:
        1. A breath in (prediction = 2)
        2. Followed by any number of other states (1, 3, 0)
        3. Until a breath out (0) is encountered

        Args:
        filename (str): Path to the file containing two columns: number and prediction

        Returns:
        int: Number of complete breaths
        """
        breath_count = 0
        breath_in_progress = False
        results = list(results)

        for i in range(len(results)):
            if results[i:i+5] == [2 for _ in range(5)] and not breath_in_progress:
                breath_in_progress = True

            elif results[i:i+5] == [0 for _ in range(5)] and breath_in_progress:
                breath_count += 1
                breath_in_progress = False

        return breath_count

    def save(self, filename):
        self.model.save(filename + ".keras")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter._experimental_lower_tensor_list_ops = False
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter.convert()
        with open(filename + ".tflite", "wb") as f:
            f.write(tflite_model)

    def load(self, filename):
        self.model = load_model(filename + ".keras")
        self.is_model_loaded = True
