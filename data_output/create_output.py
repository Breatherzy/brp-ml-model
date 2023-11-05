from scripts.load_data import load_data, save_data
from scripts.normalization import normalize
from joblib import load

#model = 'wdech_wydech'
model = 'bezdech'
#model = 'zatrzymanie'

samples = 10

# Wczytanie modelu z pliku
loaded_clf = load('../modele/'+model+'.joblib')

data = load_data('../data_raw/raw_data.txt')
data = normalize(data)
data_output = []
for i in range(samples):
    data_output.append([0])


def devide_segments(dane, data_output, clf, dlugosc_segmentu=samples):
    for i in range(len(dane) - dlugosc_segmentu + 1):
        segment = dane[i:i + dlugosc_segmentu]
        data_output.append(clf.predict([segment]))


devide_segments(data, data_output, loaded_clf)
save_data(model+'.txt', data_output)
