import matplotlib.pyplot as plt


def interactive_plot(
        features, labels_predicted, labels_actual, window_size=150, title="Interactive plot"
):
    if len(labels_actual.shape) > 1 and labels_actual.shape[1] == 3:
        labels_actual = labels_actual.argmax(axis=1)
    # Inicjalizacja zmiennej globalnej
    current_index = 0

    # Definicja funkcji rysującej wykres
    def plot(ax, start_index=0, predicted=True):
        ax.clear()
        colors = [
            "red" if m == 0 else "blue" if m == 2 else "green" if m == 1 else "gray"
            for m in labels_predicted
        ]
        if not predicted:
            colors = [
                "red" if m == 0 else "blue" if m == 2 else "green" if m == 1 else "gray"
                for m in labels_actual
            ]

        ax.scatter(
            range(start_index, start_index + window_size),
            features[start_index: start_index + window_size],
            color=colors[start_index: start_index + window_size],
            picker=True,
        )
        ax.plot(
            range(start_index, start_index + window_size),
            features[start_index: start_index + window_size],
            linestyle="-",
            color="gray",
            alpha=0.5,
        )
        ax.set_xlim(start_index, start_index + window_size)

    # Definicja funkcji obsługi zdarzeń klawiatury
    def on_key(event):
        nonlocal current_index
        if event.key == "right":
            current_index += 10  # Przesunięcie okna w prawo
        elif event.key == "left":
            current_index -= 10  # Przesunięcie okna w lewo

        # Zapobieganie wyjściu poza zakres
        current_index = max(0, min(len(features) - window_size, current_index))
        plot(ax1, current_index, predicted=True)
        plot(ax2, current_index, predicted=False)
        plt.draw()

    # Tworzenie figury i osi
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.gcf().canvas.mpl_connect("key_press_event", on_key)

    # Pierwsze rysowanie wykresu
    plot(ax1, current_index, predicted=True)
    plot(ax2, current_index, predicted=False)

    # Ustawienie tytułu wykresu
    plt.suptitle(title)

    # Wyświetlenie wykresu
    plt.show()
