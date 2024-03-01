import numpy as np
import tensorflow as tf

from models.AbstractModel import AbstractModel


class BNModel(AbstractModel):
    def compile(self):
        if self.check_if_data_is_loaded():
            for dataset_name in ['y', 'y_train', 'y_test']:
                dataset = getattr(self, dataset_name)
                setattr(self, dataset_name, tf.keras.utils.to_categorical(dataset, num_classes=3))

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
            history = self.model.fit(self.X_train, self.y_train, epochs=100,
                                     batch_size=32, validation_data=(self.X_test, self.y_test))
            history = history.history
            with open('models/saves/' + self.__class__.__name__ + '.history', 'w') as file:
                file.write(str(history))

    def predict(self, X_test):
        if self.check_if_model_is_fitted():
            return np.argmax(self.model.predict(X_test), axis=1)

    def evaluate(self):
        y_pred = self.predict(self.X_test)
        if self.y_test.ndim > 1:
            y_test = np.argmax(self.y_test, axis=1)
        else:
            y_test = self.y_test
        print('Accuracy:', np.mean(y_pred == y_test))

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename)
        self.is_model_loaded = True
