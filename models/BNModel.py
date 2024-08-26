from keras.layers import BatchNormalization, Dense
from keras.models import Sequential
from keras.src.layers import Activation

from models.AbstractModel import SensorType
from models.SequentialModel import SequentialModel


class BNModel(SequentialModel):
    def load_data(self, filename, sensor_type=SensorType.TENSOMETER.value):
        super().load_data(
            filename, convert_to_categorical=True, sensor_type=sensor_type
        )

    def compile(self):
        if self.check_if_data_is_loaded():
            self.model = Sequential(
                [
                    Dense(64, input_shape=(self.X_train.shape[1],)),
                    BatchNormalization(),
                    Activation("relu"),
                    Dense(64),
                    BatchNormalization(),
                    Activation("relu"),
                    Dense(4, activation="softmax"),
                ]
            )
            self.model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
