import matplotlib.pyplot as plt

# Definicja funkcji głównej
def interactive_plot(numbers, mono_numbers, window_size=150):
    # Inicjalizacja zmiennej globalnej
    current_index = 0

    # Definicja funkcji rysującej wykres
    def plot(start_index=0):
        # Resetowanie wykresu
        ax.clear()
        colors = ["red" if m == -1 else "green" if m == 1 else "blue" if m == 0 else "gray" for m in mono_numbers]
        for i, (y, color) in enumerate(
                zip(numbers[start_index:start_index + window_size], colors[start_index:start_index + window_size])):
            ax.scatter(i + start_index, y, color=color)
            ax.annotate(f'{y:.2f}', (i + start_index, y), textcoords="offset points", xytext=(0, 10), ha='center')

        ax.set_xlim(start_index, start_index + window_size)
        plt.draw()

    # Definicja funkcji obsługi zdarzeń klawiatury
    def on_key(event):
        nonlocal current_index
        if event.key == 'right':
            current_index += 10  # Przesunięcie okna w prawo
        elif event.key == 'left':
            current_index -= 10  # Przesunięcie okna w lewo

        # Zapobieganie wyjściu poza zakres
        current_index = max(0, min(len(numbers) - window_size, current_index))
        plot(current_index)

    # Tworzenie figury i osi
    fig, ax = plt.subplots()
    plt.gcf().canvas.mpl_connect('key_press_event', on_key)

    # Pierwsze rysowanie wykresu
    plot(current_index)

    # Wyświetlenie wykresu
    plt.show()

