import re


def load_raw_data(filename: str) -> list[float]:
    numbers = []
    # Geting data from file
    with open(filename) as file:
        data = file.read().splitlines()
        for data_line in data:
            numbers.append(float(data_line))
    return numbers

def load_tagged_data(filename: str) -> (list[float], list[float]):
    numbers = []
    tags = []
    # Geting data from file
    with open(filename) as file:
        data = file.read().splitlines()
        for data_line in data:
            numbers.append(float(data_line.split(",")[0]))
            tags.append(float(data_line.split(",")[1]))
    return numbers, tags

def load_data(filename):
    numbers = []
    # Geting data from file
    with open(filename) as file:
        data = file.read().splitlines()
        for data_line in data:
            match = re.search(r"y:(\d+\.\d+)", data_line)
            if match:
                number = float(match.group(1))
                numbers.append(number)

    return numbers[200:-200]


def save_data(filname, data):
    with open(filname, "w") as file:
        for element in data:
            file.write(f"{element[0]}\n")


def save_sequences(file_to_retrieve_sequences: str, file_to_save: str, size: int) -> None:
    with open(file_to_retrieve_sequences):
        data, tags = load_tagged_data(file_to_retrieve_sequences)
    sequences = []
    for i in range(size, len(data)):
        sequence = data[i-size: i]
        taged_sequence = tags[i-size: i]
        sequences.append(sequence + [int(sum(taged_sequence) / size)+1])

    with open(file_to_save, "w") as file:
        for seq in sequences:
            file.write(",".join(map(str, seq)) + "\n")