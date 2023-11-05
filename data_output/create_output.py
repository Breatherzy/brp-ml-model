from scripts.load_data import load_data, save_data
from scripts.normalization import normalize
from joblib import load

# Wczytanie modelu z pliku
loaded_clf = load('../modele/wdech_wydech.joblib')

data = load_data('../data_raw/raw_data.txt')
data = normalize(data)
data_output = [[0], [0], [0], [0]]


def devide_segments(dane, data_output, clf, dlugosc_segmentu=5):
    for i in range(len(dane) - dlugosc_segmentu + 1):
        segment = dane[i:i + dlugosc_segmentu]
        data_output.append(clf.predict([segment]))


devide_segments(data, data_output, loaded_clf)
save_data('wdech_wydech.txt', data_output)
