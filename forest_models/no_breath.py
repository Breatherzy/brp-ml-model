import random

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
import pandas as pd

# random.seed(42)

SIZE = 5


def load_data(filename, splitter=" "):
    with open(filename, 'r') as f:
        sequences = []
        for line in f.readlines():
            sequences.append([float(value) for value in line.strip().split(splitter)])
        return sequences


# Adjust the 'not constant velocity' data to have 10 features by repeating some values
# This is a simplification for the sake of the example
profile_data3_tmp = load_data('../data_categories_2/bezdech.txt')
profile_data3 = []
for i in range(len(profile_data3_tmp)):
    amp = max(profile_data3_tmp[i]) - min(profile_data3_tmp[i])
    # if amp < 1:
    profile_data3.append(profile_data3_tmp[i])
with open('../data_categories_2/bezdech_2.txt', 'w') as f:
    for line in profile_data3:
        f.write(str(line).replace('[', '').replace(']', '') + '\n')
profile_data4 = load_data('../data_categories_2/wdech_10.txt', splitter=", ")
profile_data5 = load_data('../data_categories_2/wydech_10.txt', splitter=", ")

x1 = []
for i in range(len(profile_data3)):
    first_half = profile_data3[i][:SIZE]
    second_half = profile_data3[i][SIZE:]
    amp1 = max(first_half) - min(first_half)
    amp2 = max(second_half) - min(second_half)
    x1.append(first_half + [amp1])
    x1.append(second_half + [amp2])
    amp = max(profile_data3[i]) - min(profile_data3[i])
    profile_data3[i] = profile_data3[i]

x2 = []
for i in range(len(profile_data4)):
    first_half = profile_data4[i][:SIZE]
    second_half = profile_data4[i][SIZE:]
    amp1 = max(first_half) - min(first_half)
    amp2 = max(second_half) - min(second_half)
    x2.append(first_half + [amp1])
    x2.append(second_half + [amp2])
    amp = max(profile_data4[i]) - min(profile_data4[i])
    profile_data4[i] = profile_data4[i]

x3 = []
for i in range(len(profile_data5)):
    first_half = profile_data5[i][:SIZE]
    second_half = profile_data5[i][SIZE:]
    amp1 = max(first_half) - min(first_half)
    amp2 = max(second_half) - min(second_half)
    x3.append(first_half + [amp1])
    x3.append(second_half + [amp2])
    amp = max(profile_data5[i]) - min(profile_data5[i])
    profile_data5[i] = profile_data5[i]

X = np.vstack((x1, x2, x3))

Y = np.array([0] * len(x1) + [1] * len(x2) + [-1] * len(x3))

# Split the data for the second model
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, Y, test_size=0.2)

# Train the second Random Forest model
model2 = RandomForestClassifier()
model2.fit(X_train_2, y_train_2)

# Predict and evaluate the second model
y_pred_2 = model2.predict(X_test_2)
accuracy2 = accuracy_score(y_test_2, y_pred_2)

# pipeline = PMMLPipeline([
#     ("classifier", RandomForestClassifier())
# ])
# pipeline.fit(pd.DataFrame(X_train_2), pd.Series(y_train_2))
#
# sklearn2pmml(pipeline, "random_forest.pmml", with_repr=True)

print(f'Accuracy of the second model: {accuracy2}')
print(f'Classification report of the second model:\n{classification_report(y_test_2, y_pred_2)}')

# TESTING THE MODELS

# Load the data
# with open('../data_categories_2/raw_data_no_breath2.txt', 'r') as f:
#     data = f.read().splitlines()
#     data = [float(value.split(" ")[-1]) for value in data]

with open('../data_categories_2/raw_data_normalized.txt', 'r') as f:
    data = f.read().splitlines()
    data = [float(value) for value in data]


data = data

result = []
mono_numbers = []
previous = 0
for i in range(SIZE, len(data)):
    print(f'{i} of {len(data)}')
    window = data[i - SIZE:i]
    amplitude = max(window) - min(window)
    window = window + [amplitude]
    prediction = model2.predict([window])
    print(i, window, prediction)
    if prediction == 1 and amplitude >= 0.2:
        mono_numbers.append(1)
        previous = 1
    elif prediction == -1 and amplitude >= 0.2:
        mono_numbers.append(-1)
        previous = -1
    elif prediction == 0 and amplitude < 0.2:
        mono_numbers.append(0)
        previous = 0
    else:
        mono_numbers.append(previous)
    #window.reverse()
    result.append(window + [mono_numbers[-1]])

with open('../data_categories_2/data_to_train.txt', 'w') as f:
    for line in result:
        f.write(str(line).replace('[', '').replace(']', '') .replace(',', '') + '\n')

from plot import interactive_plot

print(len(data), len(mono_numbers))
interactive_plot(data, mono_numbers)
