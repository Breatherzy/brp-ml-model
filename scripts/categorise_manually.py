import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector
from scripts.load_data import load_data_no_breath

from scripts.normalization import normalize

# Load and normalize data
np.random.seed(42)
numbers = []
number_string = []
files = ['bezdech.txt', 'zatrzymanie.txt', '']
numbers = load_data_no_breath("data_set/raw_data_no_breath2.txt")
numbers = normalize(numbers)
number_strings = [str(str_number) for str_number in numbers]
mono_numbers = []

# Constants
WINDOW_SIZE = 150
SELECTION_SIZE = 10
current_start_index = 0


def plot_data(start_index=0):
    plt.cla()  # Clear current axes
    end_index = start_index + WINDOW_SIZE
    # 'picker=5' allows points to be clickable
    plt.plot(numbers[start_index:end_index], '-o', ms=4, picker=5)
    plt.title('Interactive Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    fig.canvas.draw()


def on_key(event):
    global current_start_index
    if event.key == 'right':
        current_start_index += SELECTION_SIZE
        if current_start_index > len(numbers) - WINDOW_SIZE:
            current_start_index = len(numbers) - WINDOW_SIZE
    elif event.key == 'left':
        current_start_index -= SELECTION_SIZE
        if current_start_index < 0:
            current_start_index = 0
    plot_data(current_start_index)


def onselect(xmin, xmax):
    range_ = int(xmax) - int(xmin)
    if range_ >= SELECTION_SIZE:  # If more than 10 points are selected, just pick the first 10
        selected_points = number_strings[int(
            xmin)+current_start_index:int(xmin + SELECTION_SIZE)+current_start_index]
        with open('data_output/' + files[0], "a") as file:
            file.write(f'{" ".join(selected_points)}\n')


fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.mpl_connect('key_press_event', on_key)

span = SpanSelector(
    ax,
    onselect,
    "horizontal",
    useblit=True,
    props=dict(alpha=0.5, facecolor="tab:blue"),
    interactive=True,
    drag_from_anywhere=True,
    minspan=10,
)

plot_data()
plt.show()
