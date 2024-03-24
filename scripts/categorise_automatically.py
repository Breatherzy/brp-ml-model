import os

import numpy as np

from scripts.load_data import load_raw_data as load_data
from scripts.normalization import normalize

# Load and normalize data
np.random.seed(42)
numbers = []
number_string = []
WINDOW_SIZE = 5


def monotonicity(data, data_size=5) -> list[int]:
    deriv = np.gradient(data)
    result = [0] * len(data)

    for i in range(len(deriv) - data_size + 1):
        if abs(deriv[i]) < 0.0079:
            result[i] = 1
        elif deriv[i] > 0:
            result[i] = 2
        else:
            result[i] = 0
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


def find_monotonicity_changes(array, window_size=WINDOW_SIZE):
    results = []
    for i in range(window_size, len(array)):
        subarray = array[i - window_size : i]
        result = check_monotonicity_change(subarray)
        results.append(result)

    return results


def save_tagged_data(data, monotonicity, filename):
    with open("../data/pretrained/tens_point/" + filename, "w") as file:
        for i in range(len(data)):
            file.write(f"{data[i]},{monotonicity[i]}\n")


numbers = load_data("/Users/masze/Projects/brp-ml-model/data/raw/tens/tens_bezdech.txt")
numbers = normalize(numbers)
number_strings = [str(str_number) for str_number in numbers]
mono_tags = monotonicity(numbers, data_size=WINDOW_SIZE)

save_tagged_data(numbers, mono_tags, "tens_bezdech" + ".txt")

exit()

current_directory = os.getcwd()
desired_directory = (
    os.path.dirname(os.path.dirname(current_directory))
    + "/brp-ml-model/data/raw/tens_point/"
)
for file in os.listdir(desired_directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        numbers = load_data(desired_directory + filename)
        numbers = normalize(numbers)
        number_strings = [str(str_number) for str_number in numbers]
        mono_tags = monotonicity(numbers, data_size=WINDOW_SIZE)

        save_tagged_data(numbers, mono_tags, filename[:-4] + ".txt")

        # save_sequences(
        #     numbers, mono_numbers, 1, filename[:-4] + "_wdech_10.txt", WINDOW_SIZE
        # )
        # save_sequences(
        #     numbers, mono_numbers, -1, filename[:-4] + "_wydech_10.txt", WINDOW_SIZE
        # )
