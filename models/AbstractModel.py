from abc import ABCMeta, abstractmethod

import numpy as np
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split


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

    def load_data(self, filename, expand_dims=False, convert_to_categorical=False):
        """
        Metoda do ładowania danych z pliku i podziału na dane treningowe i testowe.
        Każda linia pliku zawiera dane rozdielone spacją, a ostatnia wartość to etykieta.

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
                sequences.append([float(value) for value in line.split(" ")])
            data = np.array(sequences)

        self.X = data[:, :-1]
        self.y = data[:, -1]

        if expand_dims:
            self.X = np.expand_dims(self.X, axis=1)
        if convert_to_categorical:
            self.y = to_categorical(self.y, num_classes=len(np.unique(self.y)))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
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
        if self.X is None or self.y is None:
            raise ValueError(
                "Dane nie są wczytane. Użyj metody load_data() przed kompilacją modelu"
            )
        return True