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
        Metoda do ładowania danych z pliku i podziału na dane treningowe i testowe.
        Każda linia pliku zawiera dane rozdzielone przecinkiem, a ostatnia wartość to etykieta.

        :param filename: nazwa pliku z danymi
        :type filename: str

        :param expand_dims: czy dane wejściowe mają być rozszerzone o jedną osie
        :type expand_dims: bool

        :param convert_to_categorical: czy etykiety mają być przekształcone do postaci kategorycznej
        :type convert_to_categorical: bool
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
        Metoda do kompilowania modelu.
        """
        raise NotImplementedError("Metoda compile() nie jest zaimplementowana")

    @abstractmethod
    def fit(self):
        """
        Metoda do dopasowywania modelu do danych.
        Powinna zapisywać historię uczenia modelu.
        """
        raise NotImplementedError("Metoda fit() nie jest zaimplementowana")

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Metoda do przewidywania na nowych danych.
        """

        raise NotImplementedError("Metoda predict() nie jest zaimplementowana")

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
        Metoda zwrająca wyniki ewaluacji modelu.
        """
        raise NotImplementedError("Metoda evaluate() nie jest zaimplementowana")

    @abstractmethod
    def save(self, filename):
        """
        Metoda do zapisywania modelu do pliku.
        """
        raise NotImplementedError("Metoda save() nie jest zaimplementowana")

    @abstractmethod
    def load(self, filename):
        """
        Metoda do wczytywania modelu z pliku.
        Ustawia atrybut is_model_loaded na True.
        """
        raise NotImplementedError("Metoda load() nie jest zaimplementowana")

    def check_if_model_is_compiled(self):
        """
        Metoda do sprawdzania czy model jest skompilowany.
        """
        if self.model is None and not self.is_model_loaded:
            raise ValueError("Model nie jest skompilowany")
        return True

    def check_if_model_is_fitted(self):
        """
        Metoda do sprawdzania czy model jest skompilowany.
        """
        if not self.is_model_fitted and not self.is_model_loaded:
            raise ValueError(
                "Dane nie są podzielone na zbiór treningowy i testowy. Użyj metody fit() przed ewaluacją modelu"
            )
        return True

    def check_if_data_is_loaded(self):
        """
        Metoda do sprawdzania czy dane są wczytane.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Dane nie są wczytane. Użyj metody load_data() przed kompilacją modelu"
            )
        return True
