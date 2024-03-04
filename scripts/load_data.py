import re

def load_data(filename):
    numbers =[]
    # Geting data from file
    with open(filename) as file:
        data = file.read().splitlines()
        for data_line in data:
            match = re.search(r'y:(\d+\.\d+)', data_line)
            if match:
                number = float(match.group(1))
                numbers.append(number)


    return numbers[200:-200]

def load_data_no_breath(filename):
    numbers = []
    # Geting data from file
    with open(filename) as file:
        data = file.read().splitlines()
        for data_line in data:
            match = re.search(r'I  (\d+\.\d+)', data_line)
            if match:
                number = float(match.group(1))
                numbers.append(number)
    return numbers

def load_data_raw(filename):
    numbers = []
    with open(filename, 'r') as file:
        for line in file:
            numbers.append(float(line.strip()))
    return numbers