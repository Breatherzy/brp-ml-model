from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from models.AbstractModel import AbstractModel


class LogisticRegressionModel(AbstractModel):
    def compile(self):
        if self.check_if_data_is_loaded():
            self.model = LogisticRegression(max_iter=100)

    def fit(self):
        if self.check_if_model_is_compiled():
            self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        if self.check_if_model_is_fitted():
            return self.model.predict(X_test)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print('Test accuracy:', accuracy)

    def save(self, filename):
        dump(self.model, filename)
