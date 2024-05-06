import ast

import matplotlib.pyplot as plt
import numpy as np

from scripts.load_data import load_raw_data
from scripts.normalization import moving_average, normalize


def interactive_plot(
        features, labels_predicted, labels_actual, window_size=150, title="Interactive plot"
):
    if len(labels_actual.shape) > 1 and labels_actual.shape[1] == 3:
        labels_actual = labels_actual.argmax(axis=1)
    current_index = 0

    def plot(ax, start_index=0, predicted=True):
        ax.clear()
        colors = [
            "red"
            if m == 0
            else "green"
            if m == 1
            else "blue"
            if m == 2
            else "yellow"
            if m == 3
            else "gray"
            for m in labels_predicted
        ]
        if not predicted:
            colors = [
                "red"
                if m == 0
                else "green"
                if m == 1
                else "blue"
                if m == 2
                else "yellow"
                if m == 3
                else "gray"
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
        ax.set_xlabel("" if predicted else 'Indeks próbki')
        ax.set_ylabel(f"Etykiety {'przewidywane' if predicted else 'wzorcowe'}")

    def on_key(event):
        nonlocal current_index
        if event.key == "right":
            current_index += 10
        elif event.key == "left":
            current_index -= 10

        current_index = max(0, min(len(features) - window_size, current_index))
        plot(ax1, current_index, predicted=True)
        plot(ax2, current_index, predicted=False)
        plt.draw()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.gcf().canvas.mpl_connect("key_press_event", on_key)

    plot(ax1, current_index, predicted=True)
    plot(ax2, current_index, predicted=False)

    plt.suptitle(title)

    plt.show()


def plot_raw_data(sensor_type: str):
    seconds, numbers = load_raw_data(f"data/raw/{sensor_type}/{sensor_type}_test.csv")

    features = numbers
    labels = np.zeros(len(numbers))

    interactive_plot(features, labels, labels)


def plot_test_data(sensor_type: str):
    seconds, numbers = load_raw_data(f"data/raw/{sensor_type}/{sensor_type}_test.csv")

    plt.figure(figsize=(20, 10))

    # numbers = moving_average(numbers, 11)

    # numbers = normalize(numbers, 375)

    # seconds = seconds[: len(numbers)]

    plt.plot(seconds, numbers)
    fontsize = 15
    plt.xlabel("Czas [s]", fontsize=fontsize)

    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.ylabel("Surowa wartość z akcelerometru", fontsize=fontsize)
    plt.title(f"Surowe dane z akcelerometru", fontsize=fontsize)

    avg = 0.6

    plt.axvline(x=0, color="red", linestyle="--")
    plt.text(0, avg, "norm", rotation=90, fontsize=fontsize)
    plt.axvline(x=60, color="red", linestyle="--")
    plt.text(60, avg, "shallow", rotation=90, fontsize=fontsize)
    plt.axvline(x=90, color="red", linestyle="--")
    plt.text(90, avg, "inhale_stop", rotation=90, fontsize=fontsize)
    plt.axvline(x=120, color="red", linestyle="--")
    plt.text(120, avg, "exhale_stop", rotation=90, fontsize=fontsize)
    plt.axvline(x=150, color="red", linestyle="--")
    plt.text(150, avg, "slow", rotation=90, fontsize=fontsize)
    plt.axvline(x=180, color="red", linestyle="--")
    plt.text(180, avg, "hiper", rotation=90, fontsize=fontsize)
    plt.axvline(x=210, color="red", linestyle="--")
    plt.text(210, avg, "cough", rotation=90, fontsize=fontsize)
    plt.axvline(x=240, color="red", linestyle="--")
    plt.text(240, avg, "inhale_pause", rotation=90, fontsize=fontsize)
    plt.axvline(x=270, color="red", linestyle="--")
    plt.text(270, avg, "exhale_pause", rotation=90, fontsize=fontsize)

    # increase font size
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.savefig(f"graphs/raw_data_{sensor_type}.png")

    plt.show()


def plot_tagged_data(sensor_type: str):
    data = np.loadtxt(
        f"data/labelled/{sensor_type}_point/{sensor_type}_test.txt", delimiter=","
    )

    features = data[:, :-1]
    labels = data[:, -1]

    interactive_plot(features, labels, labels)


def plot_history(filename: str):
    with open(filename, "r") as file:
        history = file.read()

    history = history.split("\n")[-2]
    history = ast.literal_eval(history)

    plt.plot(history["accuracy"], label="accuracy")
    plt.plot(history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(filename.split('/')[-1])
    plt.show()


def plot_evaluation_history(history: list[tuple[int, float]]):
    plt.plot([i for i, _ in history], [acc for _, acc in history])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
