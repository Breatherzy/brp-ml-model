import random
import sys

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

random.seed(42)
SIZE = 5
SIZE2 = 5
def load_data(filename, splitter=" "):
    with open(filename, 'r') as f:
        sequences = []
        for line in f.readlines():
            sequences.append([float(value) for value in line.strip().split(splitter)])
        return np.array(sequences)


# Załadowanie danych
profile_data = load_data('../data_categories_2/wdech.txt', splitter=", ")
profile_data2 = load_data('../data_categories_2/wydech.txt', splitter=", ")

for i in range(len(profile_data)):
    amp = max(profile_data[i]) - min(profile_data[i])
    profile_data[i] = profile_data[i] + [amp]

for i in range(len(profile_data2)):
    amp = max(profile_data2[i]) - min(profile_data2[i])
    profile_data2[i] = profile_data2[i] + [amp]

# Example data for the first Random Forest model
# Let's assume each row has 5 features
# Dataset for accelerating (label 1) and slowing down (label -1)
X_accelerating = profile_data
X_slowing_down = profile_data2

# Combine datasets and create labels
X = np.vstack((X_accelerating, X_slowing_down))
y = np.array([1] * len(X_accelerating) + [-1] * len(X_slowing_down))  # 1 for accelerating, -1 for slowing down

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the first Random Forest model and train it
model1 = RandomForestClassifier(random_state=42)
model1.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model1.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred)

# Adjust the 'not constant velocity' data to have 10 features by repeating some values
# This is a simplification for the sake of the example
profile_data3 = load_data('../data_categories_2/bezdech.txt')
profile_data4 = load_data('../data_categories_2/wdech_10.txt', splitter=", ")
profile_data5 = load_data('../data_categories_2/wydech_10.txt', splitter=", ")

x1 = []
for i in range(len(profile_data3)):
    first_half = profile_data3[i][:SIZE2]
    second_half = profile_data3[i][SIZE2:]
    amp1 = max(first_half) - min(first_half)
    amp2 = max(second_half) - min(second_half)
    x1.append(first_half + [amp1])
    x1.append(second_half + [amp2])
    amp = max(profile_data3[i]) - min(profile_data3[i])
    profile_data3[i] = profile_data3[i] + [amp]

x2 = []
for i in range(len(profile_data4)):
    first_half = profile_data4[i][:SIZE2]
    second_half = profile_data4[i][SIZE2:]
    amp1 = max(first_half) - min(first_half)
    amp2 = max(second_half) - min(second_half)
    x2.append(first_half + [amp1])
    x2.append(second_half + [amp2])
    amp = max(profile_data4[i]) - min(profile_data4[i])
    profile_data4[i] = profile_data4[i] + [amp]

x3 = []
for i in range(len(profile_data5)):
    first_half = profile_data5[i][:SIZE2]
    second_half = profile_data5[i][SIZE2:]
    amp1 = max(first_half) - min(first_half)
    amp2 = max(second_half) - min(second_half)
    x3.append(first_half + [amp1])
    x3.append(second_half + [amp2])
    amp = max(profile_data5[i]) - min(profile_data5[i])
    profile_data5[i] = profile_data5[i] + [amp]

X_constant_velocity = profile_data3
X_not_constant = random.sample(list(profile_data4), 75) + random.sample(list(profile_data5), 75)

# Combine datasets again for the second model
X_2 = np.vstack((X_constant_velocity, X_not_constant))
y_2 = np.array([0] * len(X_constant_velocity) + [2] * len(X_not_constant))

# Split the data for the second model
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

# Train the second Random Forest model
model2 = RandomForestClassifier(random_state=42)
model2.fit(X_train_2, y_train_2)

#sys.exit()
# Predict and evaluate the second model
y_pred_2 = model2.predict(X_test_2)
accuracy2 = accuracy_score(y_test_2, y_pred_2)

print(f'Accuracy of the first model: {accuracy1}')
print(f'Classification report of the first model:\n{classification_report(y_test, y_pred)}')
print(f'Accuracy of the second model: {accuracy2}')
print(f'Classification report of the second model:\n{classification_report(y_test_2, y_pred_2)}')

# TESTING THE MODELS

# Load the data
with open('../data_categories_2/raw_data_normalized.txt', 'r') as f:
    data = f.read().splitlines()
    data = [float(value) for value in data]


results = []
mono_numbers = []
previous = 0
for i in range(10, len(data)):
    print(f'{i} of {len(data)}')
    window = data[i - 10:i]
    amplitude = max(window) - min(window)
    window = window + [amplitude]
    prediction = model2.predict([window])
    if prediction == 2 and amplitude >= 0.2:
        prediction = model1.predict([window[-6:-1]])
        if prediction == 1:
            mono_numbers.append(1)
            previous = 1
        elif prediction == -1:
            mono_numbers.append(-1)
            previous = -1
    elif prediction == 0 and amplitude < 0.2:
        mono_numbers.append(0)
        previous = 0
    else:
        mono_numbers.append(previous)

from plot import interactive_plot

interactive_plot(data[10:], mono_numbers)

