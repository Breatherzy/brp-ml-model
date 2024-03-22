import os

import numpy as np
from scripts.load_data import load_data

from scripts.normalization import normalize

# Load and normalize data
np.random.seed(42)
numbers = []
number_string = []


def monotonicity(data, data_size=5):
    result = [0] * len(data)

    for i in range(data_size - 1, len(data), 5):
        amplitude = max(data[i - data_size + 1 : i + 1]) - min(
            data[i - data_size + 1 : i + 1]
        )
        if (
            all(data[j] < data[j + 1] for j in range(i - data_size - 1, i))
            and amplitude > 0.3
        ):
            result[i] = 1
        elif (
            all(data[j] > data[j + 1] for j in range(i - data_size - 1, i))
            and amplitude > 0.3
        ):
            result[i] = -1
        else:
            result[i] = -2

    return result


def check_monotonicity_change(subarray):
    increasing = False
    decreasing = False
    changes = 0
    x_points = np.arange(0, len(subarray))
    coefficients = np.polyfit(x_points, subarray, 1)
    a = coefficients[0]
    if max(subarray) - min(subarray) < 0.25:
        return 0
    for i in range(1, len(subarray)):
        if subarray[i] > subarray[i - 1]:
            if decreasing:
                changes += 1
                decreasing = False
            increasing = True
        elif subarray[i] < subarray[i - 1]:
            if increasing:
                changes += 1
                increasing = False
            decreasing = True

        if changes > 1:
            return 0

    return 1 if changes == 1 and -0.05 < a < 0.05 else 0


def find_monotonicity_changes(array, window_size=widnow_size):
    results = []
    for i in range(window_size, len(array)):
        subarray = array[i - window_size : i]
        result = check_monotonicity_change(subarray)
        results.append(result)

    return results


def save_sequences(data, monotonicity, value, filename, size):
    sequences = []
    for i in range(size, len(data)):
        if monotonicity[i] == value:
            sequences.append(data[i - size + 1 : i + 1])

    with open("data_output/" + filename, "w") as file:
        for seq in sequences:
            file.write(", ".join(map(str, seq)) + "\n")


directory = os.fsencode("data_set/raw/tens")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        numbers = load_data("data_set/raw/tens/" + filename)
        numbers = normalize(numbers)
        number_strings = [str(str_number) for str_number in numbers]
        mono_numbers = []
        widnow_size = 10

        mono_numbers = monotonicity(numbers, data_size=widnow_size)
        save_sequences(
            numbers, mono_numbers, 1, filename[:-4] + "_wdech_10.txt", widnow_size
        )
        save_sequences(
            numbers, mono_numbers, -1, filename[:-4] + "_wydech_10.txt", widnow_size
        )
