import re


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
