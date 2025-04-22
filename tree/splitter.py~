from collections import defaultdict

def split_numeric(dataset, labels, attr_index, threshold):
    left_data, left_labels = [], []
    right_data, right_labels = [], []
    for x, y in zip(dataset, labels):
        if x[attr_index] <= threshold:
            left_data.append(x)
            left_labels.append(y)
        else:
            right_data.append(x)
            right_labels.append(y)
    return (left_data, left_labels), (right_data, right_labels)

def split_categorical(dataset, labels, attr_index):
    splits = defaultdict(lambda: ([], []))
    for x, y in zip(dataset, labels):
        value = x[attr_index]
        splits[value][0].append(x)
        splits[value][1].append(y)
    return splits

def generate_numeric_thresholds(dataset, attr_index):
    values = sorted(set(x[attr_index] for x in dataset))
    return [(v1 + v2) / 2 for v1, v2 in zip(values[:-1], values[1:])]
