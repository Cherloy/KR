from collections import Counter
import math

def entropy(labels):

    # Если меток нет, энтропия равна 0
    total_count = len(labels)
    if total_count == 0:
        return 0

    # Подсчитываем, сколько раз встречается каждая метка
    label_counts = Counter(labels)

    entropy_value = -sum(
        (count / total_count) * math.log2(count / total_count)
        for count in label_counts.values()
    )

    return entropy_value

def information_gain(parent_labels, splits):

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