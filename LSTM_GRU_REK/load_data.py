import numpy as np
def load_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            sample = [float(value) for value in line.split()]
            data.append(sample)
    return np.array(data)