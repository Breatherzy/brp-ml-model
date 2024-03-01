from joblib import dump
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from models.AbstractModel import AbstractModel


class SupportVectorMachineModel(AbstractModel):

    def compile(self):
        if self.check_if_data_is_loaded():
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
            self.model = SVC(gamma='scale')

    def fit(self):
        if self.check_if_model_is_compiled():
            self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        if self.check_if_model_is_fitted():
            return self.model.predict(X_test)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print('Accuracy:', accuracy_score(self.y_test, y_pred))

    def save(self, filename):
        dump(self.model, filename)
