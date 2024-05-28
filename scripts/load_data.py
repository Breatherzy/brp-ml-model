import csv
import re
from os import mkdir
from os.path import exists


def load_raw_data(filename: str) -> tuple[list[float], list[float]]:
    numbers = []
    times = []
    # Geting data from file
    with open(filename) as file:
        data = list(csv.reader(file))[1:]
        for data_line in data:
            numbers.append(float(data_line[1]))
            times.append(float(data_line[0]))
    return times, numbers


def load_tagged_data(filename: str) -> tuple[list[float], list[float]]:
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
        sequence = data[i - size : i]
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
    folder = filename.split("/")[2]
    if not exists(f"data/pretrained/{folder}"):
        mkdir(f"data/pretrained/{folder}")

    with open(filename, "w") as file:
        pass


def prepare_data_for_training(sensor) -> None:
    input_size = sensor.value["size"] - 1
    sensor_type = sensor.value["name"]
    empty_file(f"data/pretrained/{sensor_type}_sequence/{sensor_type}_concatenated.txt")
    for data in [
        "_cough.txt",
        "_exhale_pause.txt",
        "_exhale_stop.txt",
        "_hyper.txt",
        "_inhale_pause.txt",
        "_inhale_stop.txt",
        "_normal.txt",
        "_shallow.txt",
        "_slow.txt",
        "_test.txt",
    ]:
        save_sequences(
            f"data/pretrained/{sensor_type}/{sensor_type}" + data,
            f"data/pretrained/{sensor_type}_sequence/{sensor_type}" + data,
            input_size,
        )

        if data != "_test.txt":
            save_sequences_to_concatenated(
                f"data/pretrained/{sensor_type}_sequence/{sensor_type}" + data,
                f"data/pretrained/{sensor_type}_sequence/{sensor_type}_concatenated.txt",
            )
