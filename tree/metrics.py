from collections import Counter
import math

def entropy(labels):
    """
    Вычисляет энтропию для набора меток классов (мера неопределенности).

    Энтропия равна 0, если все метки одинаковы, и максимальна, если метки равномерно распределены.

    Args:
        labels: Список меток классов.

    Returns:
        Значение энтропии (число, ≥ 0).
    """
    # Если меток нет, энтропия равна 0
    total_count = len(labels)
    if total_count == 0:
        return 0

    # Подсчитываем, сколько раз встречается каждая метка
    label_counts = Counter(labels)

    # Вычисляем энтропию по формуле: -∑(p * log2(p)), где p — доля метки
    entropy_value = -sum(
        (count / total_count) * math.log2(count / total_count)
        for count in label_counts.values()
    )

    return entropy_value

def information_gain(parent_labels, splits):
    """
    Вычисляет информационный прирост от разбиения набора данных.

    Информационный прирост = энтропия до разбиения - взвешенная энтропия после разбиения.

    Args:
        parent_labels: Метки классов до разбиения.
        splits: Список подмножеств меток после разбиения.

    Returns:
        Значение информационного прироста (число, ≥ 0).
    """
    # Вычисляем энтропию исходного набора
    parent_entropy = entropy(parent_labels)

    # Общее количество меток в родительском наборе
    total_count = len(parent_labels)

    # Вычисляем взвешенную энтропию всех подмножеств
    weighted_child_entropy = sum(
        (len(subset) / total_count) * entropy(subset)
        for subset in splits
    )

    # Информационный прирост = разница между энтропией до и после
    return parent_entropy - weighted_child_entropy