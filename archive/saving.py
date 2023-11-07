def remove_short_sequences(input_file, output_file, min_lines):
    # Wczytujemy zawartość pliku
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Tworzymy listę sekwencji
    sequences = []
    current_sequence = []

    for line in lines:
        # Jeśli linia jest pusta, dodajemy bieżącą sekwencję do listy
        if line.strip() == "":
            if len(current_sequence) >= min_lines:
                sequences.append(current_sequence)
            current_sequence = []
        else:
            # Dodajemy linię do bieżącej sekwencji
            current_sequence.append(line)

    # Dodajemy ostatnią sekwencję, jeśli jest wystarczająco długa
    if len(current_sequence) >= min_lines:
        sequences.append(current_sequence)

    # Zapisujemy sekwencje do pliku
    with open(output_file, "w") as f:
        for seq in sequences:
            for line in seq:
                f.write(line)
            f.write("\n")


def save_sequences(array, filename):
    # Sortujemy tablicę po indeksach
    array.sort(key=lambda x: x[1])

    # Tworzymy listę sekwencji
    sequences = []
    current_sequence = []

    for i in range(len(array) - 1):
        # Sprawdzamy, czy następny element jest kolejnym indeksem
        if array[i + 1][1] == array[i][1] + 1:
            current_sequence.append(array[i][0])
        else:
            # Koniec sekwencji - dodajemy ostatni element
            current_sequence.append(array[i][0])
            # Dodajemy sekwencję do listy sekwencji
            sequences.append(current_sequence)
            # Czyścimy sekwencję
            current_sequence = []

    # Dodajemy ostatni element do ostatniej sekwencji
    current_sequence.append(array[-1][0])
    sequences.append(current_sequence)

    # Zapisujemy sekwencje do pliku
    with open(filename, "w") as f:
        for seq in sequences:
            for num in seq:
                f.write(str(num) + "\n")
            f.write("\n")
    remove_short_sequences(filename, filename, 5)


def load_data(filenames):
    data = []
    result = []
    for j in filenames:
        with open(j[0]) as f:
            subset = []
            for line in f:
                if line.strip():
                    subset.append(float(line.strip()))
                else:
                    Y = []
                    for i in range(0, len(subset) - (len(subset) % 5), 5):
                        Y.append(subset[i:i + 5])
                        data.append(Y)
                        result.append([[j[1]]])
                        Y = []
                    subset = []

            if subset:
                data.append([subset])
                result.append([[j[1]]])

    return data, result

def read_data_from_files_and_create_flags(file_paths):
    combined_data = []  # Wspólna tablica na dane z wszystkich plików
    combined_flags = []  # Wspólna tablica na flagi

    for file_path in file_paths:
        # Ustalenie flagi w zależności od nazwy pliku
        flag = 1 if 'zatrzymanie.txt' in file_path else 0

        # Otwarcie pliku i czytanie danych
        with open('saved_data/'+file_path, 'r') as file:
            for line in file:
                # Przetwarzanie każdej linii i dodawanie do combined_data jako tablica
                line_data = [float(number) for number in line.strip().split(', ')]
                combined_data.append([line_data])  # Dodanie danych z linii jako tablica do głównej tablicy

                # Dodanie tablicy flag odpowiadającej danej linii
                combined_flags.append([[flag]])

    return combined_data, combined_flags