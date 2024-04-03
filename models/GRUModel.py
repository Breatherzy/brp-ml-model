import numpy as np
from keras.layers import GRU, Dense
from keras.models import Sequential

from models.AbstractModel import SensorType
from models.SequentialModel import SequentialModel


class GRUModel(SequentialModel):
    def load_data(self, filename, sensor_type=SensorType.TENSOMETER.value):
        super().load_data(filename, expand_dims=True, sensor_type=sensor_type)

    def compile(self):
        if self.check_if_data_is_loaded():
            self.model = Sequential(
                [
                    GRU(
                        50,
                        activation="relu",
                        input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                    ),
                    Dense(50, activation="tanh"),
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
