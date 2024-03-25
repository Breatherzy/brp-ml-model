from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np
from keras.src.utils import to_categorical

from scripts.plot import interactive_plot


class SensorType(Enum):
    TENSOMETER = "tens"
    ACCELEROMETER = "acc"


class AbstractModel(metaclass=ABCMeta):
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.model = None
        self.is_model_loaded = False
        self.is_model_fitted = False

    def load_data(
            self,
            filename,
            expand_dims=False,
            convert_to_categorical=False,
            sensor_type=SensorType.TENSOMETER.value,
    ):
        """
        Method for loading data from a file and splitting it into training and testing data.
        Each line of the file contains comma-separated data, with the last value being the label.

        :param filename: name of the data file
        :type filename: str

        :param expand_dims: whether input data should be expanded by one axis
        :type expand_dims: bool

        :param convert_to_categorical: whether labels should be transformed into categorical form
        :type convert_to_categorical: bool

        :param sensor_type: type of sensor
        """
        with open(filename, "r") as f:
            sequences = []
            for line in f.readlines():
                sequences.append([float(value) for value in line.split(",")])
            data = np.array(sequences)

        # TODO: add testing set from tens_test.txt
        with open(
                f"data/pretrained/{sensor_type}_sequence/{sensor_type}_test.txt", "r"
        ) as f:
            sequences = []
            for line in f.readlines():
                sequences.append([float(value) for value in line.split(",")])
            data_test = np.array(sequences)

        self.X_train = data[:, :-1]
        self.y_train = data[:, -1]

        zeros_column = np.zeros((self.X_train.shape[0], 1))
        self.X_train = np.concatenate((self.X_train, zeros_column), axis=1)
        # Add amplitude to the end of the sequence
        for i in range(len(self.X_train)):
            amplitude = np.max(self.X_train[i][:-1]) - np.min(self.X_train[i][:-1])
            self.X_train[i][-1] = amplitude

        self.X_test = data_test[:, :-1]
        self.y_test = data_test[:, -1]

        zeros_column = np.zeros((self.X_test.shape[0], 1))
        self.X_test = np.concatenate((self.X_test, zeros_column), axis=1)
        # Add amplitude to the end of the sequence
        for i in range(len(self.X_test)):
            amplitude = np.max(self.X_test[i][:-1]) - np.min(self.X_test[i][:-1])
            self.X_test[i][-1] = amplitude

        if expand_dims:
            self.X_train = np.expand_dims(self.X_train, axis=1)
            self.X_test = np.expand_dims(self.X_test, axis=1)
        if convert_to_categorical:
            self.y_train = to_categorical(
                self.y_train, num_classes=len(np.unique(self.y_train))
            )
            self.y_test = to_categorical(
                self.y_test, num_classes=len(np.unique(self.y_test))
            )

    @abstractmethod
    def compile(self):
        """
        Method for compiling the model.
        """
        raise NotImplementedError("The compile() method is not implemented")

    @abstractmethod
    def fit(self):
        """
        Method for fitting the model to data.
        It should save the training history of the model.
        """
        raise NotImplementedError("The fit() method is not implemented")

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Method for predicting on new data.
        """
        raise NotImplementedError("The predict() method is not implemented")

    def plot_prediction(self, X_test, title=None) -> None:
        if X_test.shape[1] == 1:
            interactive_plot(
                self.X_test[:, -1, 0], self.predict(X_test), self.y_test, title=title
            )
        else:
            interactive_plot(
                self.X_test[:, 0], self.predict(X_test), self.y_test, title=title
            )

    @abstractmethod
    def evaluate(self, X_test: np.ndarray = None, y_test: np.ndarray = None) -> float:
        """
        Method returning the evaluation results of the model.
        """
        raise NotImplementedError("The evaluate() method is not implemented")

    @abstractmethod
    def save(self, filename):
        """
        Method for saving the model to a file.
        """
        raise NotImplementedError("The save() method is not implemented")

    @abstractmethod
    def load(self, filename):
        """
        Method for loading the model from a file.
        Sets the is_model_loaded attribute to True.
        """
        raise NotImplementedError("The load() method is not implemented")

    def check_if_model_is_compiled(self):
        """
        Method for checking if the model is compiled.
        """
        if self.model is None and not self.is_model_loaded:
            raise ValueError("The model is not compiled")
        return True

    def check_if_model_is_fitted(self):
        """
        Method for checking if the model is fitted.
        """
        if not self.is_model_fitted and not self.is_model_loaded:
            raise ValueError(
                "Data is not split into training and testing sets. Use the fit() method before evaluating the model"
            )
        return True

    def check_if_data_is_loaded(self):
        """
        Method for checking if data is loaded.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Data is not loaded. Use the load_data() method before compiling the model"
            )
        return True
