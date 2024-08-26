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
                f"models/saves/{sensor_type}/{self.__class__.__name__}.history", "a"
            ) as file:
                file.write(str(history) + "\n")
            self.is_model_fitted = True
            self.save_misclassified_samples(
                f"data/misclassified/{sensor_type}_misclassified.txt"
            )

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
        print("Evaluation result:", result)
        return result

    def get_misclassified_samples(self):
        y_pred = self.predict(self.X_test)
        if len(self.y_test.shape) > 1:
            y_test_to_compare = np.argmax(self.y_test, axis=1)
        else:
            y_test_to_compare = self.y_test
        misclassified_indices = np.where(y_pred != y_test_to_compare)[0]
        misclassified_samples = self.X_test[misclassified_indices]
        misclassified_labels = y_test_to_compare[misclassified_indices]
        return misclassified_samples, misclassified_labels

    def save_misclassified_samples(self, filename):
        samples, labels = self.get_misclassified_samples()
        with open(filename, "w") as file:
            for sample, label in zip(samples, labels):
                flattened_sample = sample.flatten()
                combined = list(flattened_sample) + [label]
                file.write(",".join(map(str, combined)) + "\n")

    def augment_data(self, X, y, sensor_type: str = "tens"):
        self.load_data(
            filename="data/misclassified/tens_misclassified.txt",
            sensor_type=sensor_type,
        )
        xd1 = self.X_train
        yd1 = self.y_train
        self.X_train = np.vstack((self.X_train, np.array(X)))
        self.y_train = np.concatenate((self.y_train, np.array(y)))
        return xd1, yd1

    def calculate_sample_weights(self, misclassified_indices):
        sample_weights = np.ones(len(self.X_train))
        sample_weights[:misclassified_indices] = (
            10  # Increase weight for misclassified samples
        )
        return sample_weights

    def retrain_with_misclassified(
        self, epochs=50, batch_size=500, sensor_type: str = "tens"
    ):
        augmented_X_train, augmented_y_train = self.augment_data(
            self.X_train,
            self.y_train,
            sensor_type=sensor_type,
        )
        sample_weights = self.calculate_sample_weights(len(augmented_X_train))

        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            sample_weight=sample_weights,
            verbose=1,
        )
        self.save_misclassified_samples(f"data/misclassified/tens_misclassified.txt")
        history = history.history
        return history

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
