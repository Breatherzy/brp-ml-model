def normalize_window(window):
    min_val = min(window)
    max_val = max(window)
    range_val = max_val - min_val
    if range_val == 0:
        return [0 for _ in window]

    normalized = [(-1 + 2 * (x - min_val) / range_val) for x in window]
    return normalized


def normalize(numbers):
    normalized_values = []

    for i in range(150, len(numbers)):
        window = numbers[i - 150: i]
        normalized_window_values = normalize_window(window)
        normalized_values.append(normalized_window_values[-1])
    return normalized_values
