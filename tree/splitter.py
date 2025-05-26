from collections import defaultdict

def split_numeric(samples, labels, attribute_index, threshold):

    left_samples, left_labels = [], []
    right_samples, right_labels = [], []

    # Перебираем объекты и их метки
    for sample, label in zip(samples, labels):
        attribute_value = sample[attribute_index]
        # Проверяем, попадает ли значение в левую или правую группу
        if attribute_value <= threshold:
            left_samples.append(sample)
            left_labels.append(label)
        else:
            right_samples.append(sample)
            right_labels.append(label)

    return (left_samples, left_labels), (right_samples, right_labels)

def split_categorical(samples, labels, attribute_index):

    # Создаем словарь, где для каждого значения атрибута хранятся данные и метки
    category_groups = defaultdict(lambda: ([], []))

    # Распределяем данные по категориям
    for sample, label in zip(samples, labels):
        category_value = sample[attribute_index]
        category_groups[category_value][0].append(sample)  # Добавляем данные
        category_groups[category_value][1].append(label)   # Добавляем метку

    return category_groups

def generate_numeric_thresholds(samples, attribute_index):

    # Извлекаем уникальные значения атрибута и сортируем их
    unique_values = sorted(set(sample[attribute_index] for sample in samples))

    # Создаем пороги как средние между соседними значениями
    thresholds = [(value1 + value2) / 2 for value1, value2 in zip(unique_values[:-1], unique_values[1:])]

    return thresholds