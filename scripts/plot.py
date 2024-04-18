import matplotlib.pyplot as plt
import numpy as np
import ast


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
            features[start_index : start_index + window_size],
            color=colors[start_index : start_index + window_size],
            picker=True,
        )
        ax.plot(
            range(start_index, start_index + window_size),
            features[start_index : start_index + window_size],
            linestyle="-",
            color="gray",
            alpha=0.5,
        )
        ax.set_xlim(start_index, start_index + window_size)

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
    data = np.loadtxt(f"data/raw/{sensor_type}/{sensor_type}_test.txt")

    features = data
    labels = np.zeros(len(data))

    interactive_plot(features, labels, labels)


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

    history = ast.literal_eval(history)

    plt.plot(history["accuracy"], label="accuracy")
    plt.plot(history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_evaluation_history(history: list[tuple[int, float]]):
    plt.plot([i for i, _ in history], [acc for _, acc in history])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
