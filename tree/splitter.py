from collections import defaultdict

def split_numeric(samples, labels, attribute_index, threshold):
    """
    Разделяет данные по числовому атрибуту на две группы: значения ≤ порога и > порога.

    Args:
        samples: Список объектов (каждый объект — список значений атрибутов).
        labels: Метки классов для каждого объекта.
        attribute_index: Индекс атрибута, по которому выполняется разбиение.
        threshold: Пороговое значение для числового атрибута.

    Returns:
        Кортеж из двух групп: ((левая_данные, левые_метки), (правая_данные, правые_метки)).
    """
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
    """
    Разделяет данные по категориальному атрибуту на группы по каждому уникальному значению.

    Args:
        samples: Список объектов (каждый объект — список значений атрибутов).
        labels: Метки классов для каждого объекта.
        attribute_index: Индекс категориального атрибута.

    Returns:
        Словарь, где ключ — значение атрибута, значение — (список_данных, список_меток).
    """
    # Создаем словарь, где для каждого значения атрибута хранятся данные и метки
    category_groups = defaultdict(lambda: ([], []))

    # Распределяем данные по категориям
    for sample, label in zip(samples, labels):
        category_value = sample[attribute_index]
        category_groups[category_value][0].append(sample)  # Добавляем данные
        category_groups[category_value][1].append(label)   # Добавляем метку

    return category_groups

def generate_numeric_thresholds(samples, attribute_index):
    """
    Генерирует пороговые значения для числового атрибута (средние между соседними уникальными значениями).

    Args:
        samples: Список объектов (каждый объект — список значений атрибутов).
        attribute_index: Индекс числового атрибута.

    Returns:
        Список пороговых значений.
    """
    # Извлекаем уникальные значения атрибута и сортируем их
    unique_values = sorted(set(sample[attribute_index] for sample in samples))

    # Создаем пороги как средние между соседними значениями
    thresholds = [(value1 + value2) / 2 for value1, value2 in zip(unique_values[:-1], unique_values[1:])]

    return thresholds