import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

# Wczytywanie danych profilu
def load_data(filename):
    with open(filename, 'r') as f:
        sequences = []
        for line in f.readlines():
            sequences.append([float(value) for value in line.split(' ')])
        return np.array(sequences)


# Załadowanie danych
profile_data = load_data('data_categorised/zatrzymanie.txt')

# Przygotowanie etykiet
# Wszystkie dane z profilu będą miały etykietę 1
labels = np.ones(len(profile_data))

# Przygotowanie danych i etykiet (przy założeniu, że mamy równą ilość danych "nieprofilowych")
# Tu symulujemy "negatywne" przypadki jako losowe dane
# W rzeczywistym przypadku powinieneś użyć rzeczywistych danych, które nie pasują do profilu
non_profile_data = np.random.rand(len(profile_data), profile_data.shape[1]) # symulacja danych niezwiązanych z profilem
non_profile_labels = np.zeros(len(non_profile_data)) # etykieta 0 dla niezwiązanych danych

# Łączenie danych
X = np.vstack((profile_data, non_profile_data))
y = np.hstack((labels, non_profile_labels))

# Normalizacja danych
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie modelu SVM
clf = svm.SVC(gamma='scale')

# Trenowanie modelu
clf.fit(X_train, y_train)

# Przewidywanie na zestawie testowym
y_pred = clf.predict(X_test)

# Ocena modelu
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))

dump(clf, 'modele/zatrzymanie.joblib')
