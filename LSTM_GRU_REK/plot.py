from random import random
from load_data import load_data_from_file

breath_in = load_data_from_file("../../data_categories_2/data_categories_tenslog/wdech_10.txt")
breath_out = load_data_from_file("../../data_categories_2/data_categories_tenslog/wydech_10.txt")
no_breath = load_data_from_file("../../data_categories_2/data_categories_tenslog/bezdech.txt")

SIZE = 5
x1_ = []
for i in range(len(breath_in)):
    x1_.append(breath_in[i] + [max(breath_in[i]) - min(breath_in[i])])

x2_ = []
for i in range(len(breath_out)):
    x2_.append(breath_out[i] + [max(breath_out[i]) - min(breath_out[i])])

x3_ = []
for i in range(len(no_breath)):
    x3_.append(no_breath[i] + [max(no_breath[i]) - min(no_breath[i])])

X_to_test = x1_ + x2_ + x3_
Y_to_test = [1] * len(x1_) + [-1] * len(x2_) + [0] * len(x3_)


data = load_data_from_file('data_to_train.txt')
SIZE = 5

for idx, i_ in enumerate([5, 10, 25, 150]):




    #fit

    result = []
    mono_numbers = []
    previous = 0
    for i in range(SIZE, len(data)):
        print(f'{i} of {len(data)}')
        window = data[i - SIZE:i]
        amplitude = max(window) - min(window)
        window = window + [amplitude]
        prediction = model2.predict([window])
        print(i, window, prediction)
        if prediction == 1 and amplitude >= 0.2:
            mono_numbers.append(1)
            previous = 1
        elif prediction == -1 and amplitude >= 0.2:
            mono_numbers.append(-1)
            previous = -1
        elif prediction == 0 and amplitude < 0.2:
            mono_numbers.append(0)
            previous = 0
        else:
            mono_numbers.append(previous)

    # plot the data
    start_idx = 0
    current_color = 'green' if mono_numbers[0] == 0 else 'red' if mono_numbers[0] == 1 else 'blue'

    for i in range(1, len(mono_numbers)):
        if ((mono_numbers[i] == 0 and current_color != 'green') or
                (mono_numbers[i] == 1 and current_color != 'red') or
                (mono_numbers[i] == -1 and current_color != 'blue')):
            axs[idx+1].plot(range(start_idx, i+1), data[start_idx:i+1], color=current_color)
            start_idx = i
            current_color = 'green' if mono_numbers[i] == 0 else 'red' if mono_numbers[i] == 1 else 'blue'

    y_pred_ = model2.predict(X_to_test)
    accuracy = accuracy_score(Y_to_test, y_pred_)

    axs[idx+1].plot(range(start_idx, len(mono_numbers)), data[start_idx:-5], color=current_color)
    axs[idx+1].set_title(f'Number of samples: {i_}, Accuracy: {round(accuracy, 5)}')
    axs[idx+1].set_xlabel('Data point index')
    green_patch = mpatches.Patch(color='green', label='No breath')
    red_patch = mpatches.Patch(color='red', label='Breath in')
    blue_patch = mpatches.Patch(color='blue', label='Breath out')

    # Add these patches to the legend
    axs[idx+1].legend(handles=[green_patch, red_patch, blue_patch], loc='lower right')


plt.show()