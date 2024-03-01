from abc import ABC, abstractmethod

import numpy as np


class AbstractModel(ABC):
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self, filename):
        """
        Metoda do ładowania danych z pliku.
        Każda linia pliku zawiera dane rozdielone spacją, a ostatnia wartość to etykieta.
        """
        with open(filename, 'r') as f:
            sequences = []
            for line in f.readlines():
                sequences.append([float(value) for value in line.split(' ')])
            data = np.array(sequences)
        self.X = data[:, :-1]
        self.y = data[:, -1]

    @abstractmethod
    def compile(self):
        """
        Metoda do kompilowania modelu.
        Zawiera ewentualne przekształcenia danych wejściowych potrzebne do modelu.
        """
        raise NotImplementedError("Metoda compile() nie jest zaimplementowana")

    @abstractmethod
    def fit(self):
        """
        Metoda do dopasowywania modelu do danych.
        Dzieli dane na zbiór treningowy i testowy.
        """
        raise NotImplementedError("Metoda fit() nie jest zaimplementowana")

    @abstractmethod
    def predict(self, X_test) -> np.ndarray:
        """
        Metoda do przewidywania na nowych danych.
        """
        raise NotImplementedError("Metoda predict() nie jest zaimplementowana")

    @abstractmethod
    def evaluate(self):
        """
        Metoda drukująca wyniki ewaluacji modelu.
        """
        raise NotImplementedError(
            "Metoda evaluate() nie jest zaimplementowana")

    @abstractmethod
    def save(self, filename):
        """
        Metoda do zapisywania modelu do pliku.
        """
        raise NotImplementedError("Metoda save() nie jest zaimplementowana")

    def load(self, filename):
        """
        Metoda do wczytywania modelu z pliku.
        """
        raise NotImplementedError("Metoda load() nie jest zaimplementowana")

    def check_if_model_is_compiled(self):
        """
        Metoda do sprawdzania czy model jest skompilowany.
        """
        if self.model is None:
            raise ValueError("Model nie jest skompilowany")
        return True

    def check_if_model_is_fitted(self):
        """
        Metoda do sprawdzania czy model jest skompilowany.
        """
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError(
                "Dane nie są podzielone na zbiór treningowy i testowy. Użyj metody fit() przed ewaluacją modelu")

    def check_if_data_is_loaded(self):
        """
        Metoda do sprawdzania czy dane są wczytane.
        """
        if self.X is None or self.y is None:
            raise ValueError("Dane nie są wczytane. Użyj metody load_data() przed kompilacją modelu")
        return True
