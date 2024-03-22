from multiprocessing import Process, Queue, freeze_support
from tkinter import Button, Frame, Tk

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector

matplotlib.use("TkAgg")

# Load data from txt file
file_path = "../data/pretrained/tens/tens_wydech_wstrzym.txt"
data = np.loadtxt(file_path, delimiter=",")

# Constants
COLOR_MAP = {-1: "red", 0: "green", 1: "blue"}
WINDOW_SIZE = 300
SELECTION_SIZE = 5

# Global variables
current_start_index = 0
span_selector_active = False


# Interactive plotting
def plot_data(ax):
    ax.cla()  # Clear current axes
    end_index = current_start_index + WINDOW_SIZE
    colors = [COLOR_MAP[label] for label in data[current_start_index:end_index, -1]]
    scatter = ax.scatter(
        range(current_start_index, end_index),
        data[current_start_index:end_index, 0],
        color=colors,
        picker=True,
    )
    line = ax.plot(
        range(current_start_index, end_index),
        data[current_start_index:end_index, 0],
        linestyle="-",
        color="gray",
        alpha=0.5,
    )
    ax.set_title("Interactive Plot")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(True)
    fig.canvas.draw()
    scatter.set_picker(True)


def onpick(event):
    if not span_selector_active and event.artist.get_label() == "_child0":
        point_index = int(event.ind[0] + current_start_index)
        current_color = int(data[point_index, -1])
        new_color = choose_color(current_color)
        update_point_color(point_index, point_index+1, new_color)


def update_point_color(start_index, end_index, new_color=None):
    # Update color in the data array
    for i in range(start_index, end_index):
        data[i, -1] = new_color
    plot_data(ax)

    # Save updated data to the txt file
    np.savetxt(file_path, data, fmt="%.18f", delimiter=",")


def choose_color(initial_color):
    result = {"color": None}

    def select_color(color):
        result["color"] = color
        root.destroy()

    root = Tk()
    root.title("Choose Color")

    frame = Frame(root)
    frame.pack(padx=10, pady=10)
    BREATH_STATE = {-1: "BREATH OUT", 0: "NO BREATH", 1: "BREATH IN"}
    for label, color in {0: "green", -1: "red", 1: "blue"}.items():
        button_label = BREATH_STATE[label]
        button = Button(
            frame,
            text=str(button_label),
            command=lambda c=label: select_color(c),
            bg=color,
            width=8,
            highlightbackground=color,
        )
        button.pack(side="left", padx=5)

    root.protocol("WM_DELETE_WINDOW", lambda: select_color(initial_color))
    root.wait_window()

    return result["color"]


def on_key(event):
    global current_start_index, span_selector_active
    if event.key == "right":
        current_start_index += SELECTION_SIZE
        if current_start_index > len(data) - WINDOW_SIZE:
            current_start_index = len(data) - WINDOW_SIZE
    elif event.key == "left":
        current_start_index -= SELECTION_SIZE
        if current_start_index < 0:
            current_start_index = 0
    elif event.key == "m":
        # Toggle span selector mode
        span_selector_active = not span_selector_active

        if span_selector_active:
            span_selector.set_active(True)
        else:
            span_selector.set_active(False)

    plot_data(ax)


def on_span_select(xmin, xmax):
    start_index = int(max(xmin, 0))
    end_index = int(min(xmax + 1, len(data)))
    new_color = choose_color(int(data[start_index, -1]))
    update_point_color(start_index, end_index, new_color)


def tk_process(q):
    plt.show()
    q.put("done")


# Create initial plot
fig, ax = plt.subplots(figsize=(10, 6))
# Connect the pick event to the function
fig.canvas.mpl_connect("pick_event", onpick)
fig.canvas.mpl_connect("key_press_event", on_key)
plt.gcf().set_picker(True)  # Enable picking for the figure

plot_data(ax)

# Create a SpanSelector
span_selector = SpanSelector(ax, on_span_select, "horizontal", useblit=True)
span_selector.set_active(False)

# Ensure that the script runs in the main block when using multiprocessing
if __name__ == "__main__":
    # Add freeze_support() before starting any process
    freeze_support()

    # Create a queue to communicate between the processes
    q = Queue()

    # Start the Tkinter process
    tkinter_process = Process(target=tk_process, args=(q,))
    tkinter_process.start()

    # Wait for the Tk
    q.get()
