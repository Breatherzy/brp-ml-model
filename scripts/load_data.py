import re

from models.AbstractModel import SensorType


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


def save_sequences(
        file_to_retrieve_sequences: str, file_to_save: str, size: int
) -> None:
    with open(file_to_retrieve_sequences):
        data, tags = load_tagged_data(file_to_retrieve_sequences)
    sequences = []
    for i in range(size, len(data)):
        sequence = data[i - size: i]
        sequence.append(abs(max(sequence) - min(sequence)))
        sequences.append(sequence + [tags[i - size] + 1])

    with open(file_to_save, "w") as file:
        for seq in sequences:
            file.write(",".join(map(str, seq)) + "\n")


def save_sequences_to_concatenated(
        file_to_retrieve_sequences: str, file_to_save: str
) -> None:
    with open(file_to_retrieve_sequences) as f:
        data = f.read().splitlines()

    with open(file_to_save, "a") as file:
        for seq in data:
            file.write(seq.strip() + "\n")


def empty_file(filename: str) -> None:
    with open(filename, "w") as file:
        pass


def prepare_data_for_training(sensor: SensorType) -> None:

    input_size = sensor.value["size"] - 1
    sensor_type = sensor.value["name"]
    empty_file(f"data/labelled/{sensor_type}_sequence/{sensor_type}_concatenated.txt")
    for data in [
        "_normal.txt",
        "_bezdech_wdech.txt",
        "_bezdech_wydech.txt",
        "_hiper.txt",
        "_wydech_wstrzym.txt",
        "_wdech_wstrzym.txt",
        "_bezdech.txt",
        "_test.txt",
    ]:
        save_sequences(
            f"data/labelled/{sensor_type}_point/{sensor_type}" + data,
            f"data/labelled/{sensor_type}_sequence/{sensor_type}" + data,
            input_size,
        )

        if data != "_test.txt":
            save_sequences_to_concatenated(
                f"data/labelled/{sensor_type}_sequence/{sensor_type}" + data,
                f"data/labelled/{sensor_type}_sequence/{sensor_type}_concatenated.txt",
            )
