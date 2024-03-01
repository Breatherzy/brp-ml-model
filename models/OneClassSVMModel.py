import numpy as np
from joblib import dump
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from models.AbstractModel import AbstractModel


class OneClassSVMModel(AbstractModel):
    def compile(self):
        if self.check_if_data_is_loaded():
            self.model = make_pipeline(
                StandardScaler(), OneClassSVM(gamma='auto'))

    def fit(self):
        if self.check_if_model_is_compiled():
            self.model.fit(self.X)

    def predict(self, new_data):
        return self.model.predict(new_data)

    def evaluate(self):
        if self.check_if_model_is_fitted():
            y_pred = self.model.predict(self.X_test)
            print(f'Accuracy: {np.mean(y_pred == 1)}')

    def save(self, filename):
        dump(self.model, filename)
