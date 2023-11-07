import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Wczytywanie danych profilu
def load_data(filename):
    with open(filename, 'r') as f:
        sequences = []
        for line in f.readlines():
            sequences.append([float(value) for value in line.split(' ')])
        return np.array(sequences)


def average_slope(x, y):
    # Oblicz różnice między kolejnymi wartościami y i x
    dy = np.diff(y)
    dx = np.diff(x)

    # Oblicz nachylenia między każdymi dwoma punktami
    slopes = dy / dx

    # Oblicz średnie nachylenie (średnią z wartości nachyleń)
    avg_slope = np.mean(slopes)

    return avg_slope

# Załadowanie danych
profile_data = load_data('data_categorised/zatrzymanie.txt')
X_train = []
x = []
for i in range(10):
    x.append(i)

for points in profile_data:
    delta = points[-1] - points[1]
    range = max(points) - min(points)
    avg_slope = average_slope(x,points)
    std = np.std(points)
    X_train.append([delta, range, avg_slope, points.size, std])


# Załaduj tylko pasujące sekwencje jako dane treningowe
#X_train =   # Właściwe cechy tylko pasujących danych

# Skalowanie cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Tworzenie modelu One-Class SVM
model = OneClassSVM(gamma='auto').fit(X_scaled)

# Po nauczeniu modelu, możesz go użyć do predykcji nowych danych
new_data = X_train  # Nowe dane do klasyfikacji, przeskalowane tak samo jak dane treningowe
new_data_scaled = scaler.transform(new_data)

# Przewidywanie nowych danych
# Model zwróci 1 dla sekwencji podobnych do tych, na których był trenowany,
# i -1 dla tych, które uzna za odstające
predicted = model.predict(new_data_scaled)

# Wyświetlenie wyników
for i, pred in enumerate(predicted):
    print(f'Sequence {i} prediction: {"Match" if pred == 1 else "No match"}')
