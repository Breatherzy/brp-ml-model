import numpy as np
from keras.layers import Conv1D, Flatten, Dense
from keras.models import Sequential

from models.SequentialModel import SequentialModel


class Conv1DModel(SequentialModel):
    def load_data(self, filename):
        super().load_data(filename, expand_dims=True)

    def compile(self):
        if self.check_if_data_is_loaded():
            self.model = Sequential(
                [
                    Conv1D(
                        filters=64,
                        kernel_size=1,
                        activation="relu",
                        input_shape=(1, self.X.shape[2]),
                    ),
                    Flatten(),
                    Dense(50, activation="relu"),
                    Dense(3, activation="softmax"),
                ]
            )
            self.model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

    def predict(self, X_test):
        if len(X_test.shape) == 2:
            X_test = np.expand_dims(X_test, axis=1)
        return super().predict(X_test)
