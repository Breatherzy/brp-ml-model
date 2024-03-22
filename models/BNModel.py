from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from models.SequentialModel import SequentialModel


class BNModel(SequentialModel):
    def load_data(self, filename):
        super().load_data(filename, convert_to_categorical=True)

    def compile(self):
        if self.check_if_data_is_loaded():
            self.model = Sequential(
                [
                    Dense(64, input_shape=(self.X.shape[1],), activation="relu"),
                    BatchNormalization(),
                    Dense(64, activation="relu"),
                    BatchNormalization(),
                    # Wyjście dla 3 klas
                    Dense(3, activation="softmax"),
                ]
            )
            self.model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
