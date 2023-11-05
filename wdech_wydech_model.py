import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump


# Wczytywanie danych
def load_data(filename):
    with open(filename, 'r') as f:
        sequences = []
        for line in f.readlines():
            sequences.append([float(value) for value in line.split(',')])
        return np.array(sequences)


# Wczytanie danych dla obu stanów
data_state_minus_1 = load_data('data_categorised/wydech.txt')
data_state_1 = load_data('data_categorised/wdech.txt')

# Tworzenie etykiet dla danych
labels_state_minus_1 = np.ones(len(data_state_minus_1)) * -1
labels_state_1 = np.ones(len(data_state_1))

# Łączenie danych i etykiet
data = np.vstack([data_state_minus_1, data_state_1])
labels = np.hstack([labels_state_minus_1, labels_state_1])


# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# Trenowanie modelu
clf = LogisticRegression(max_iter=100).fit(X_train, y_train)

print(clf.predict([[0, 0, 0, 0, 0]]))

# Przewidywanie na zestawie testowym
y_pred = clf.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy:', accuracy)

dump(clf, 'modele/wdech_wydech.joblib')