import math
from collections import Counter
from tree.node import DecisionNode
from tree.splitter import split_numeric, split_categorical, generate_numeric_thresholds
from tree.metrics import entropy


def find_most_common_label(labels):
    """Возвращает наиболее частую метку в списке."""
    return Counter(labels).most_common(1)[0][0]


def calculate_gain_ratio(parent_labels, child_groups, child_weights):
    """
    Вычисляет Gain Ratio с учетом пропущенных значений.

    Args:
        parent_labels: Метки классов до разбиения.
        child_groups: Список групп меток после разбиения.
        child_weights: Веса групп (учитывают пропущенные значения).

    Returns:
        Gain Ratio (информационный прирост, нормированный на информацию о разбиении).
    """
    total_weight = sum(child_weights)

    # Взвешенная энтропия дочерних групп
    weighted_child_entropy = sum(
        (weight / total_weight) * entropy(group)
        for group, weight in zip(child_groups, child_weights)
        if weight > 0
    )

    # Информационный прирост = энтропия родителя - взвешенная энтропия детей
    information_gain = entropy(parent_labels) - weighted_child_entropy

    # Информация о разбиении (энтропия распределения весов)
    split_information = -sum(
        (weight / total_weight) * math.log2(weight / total_weight)
        for weight in child_weights
        if weight > 0
    )

    # Если split_information равно 0, разбиение бессмысленно
    if split_information == 0:
        return 0

    return information_gain / split_information


def find_best_split(data, labels, attribute_types, verbose=False):
    """
    Находит лучший атрибут и порог для разбиения данных.

    Args:
        data: Список объектов (каждый объект - список значений атрибутов).
        labels: Метки классов для каждого объекта.
        attribute_types: Список типов атрибутов ('numerical' или 'categorical').
        verbose: Если True, выводит отладочную информацию.

    Returns:
        Кортеж: (индекс атрибута, порог (для числовых), данные разбиения).
    """
    best_gain_ratio = -1
    best_attribute = None
    best_threshold = None
    best_split_data = None

    total_samples = len(data)

    # Перебираем все атрибуты
    for attr_index in range(len(data[0])):
        # Разделяем данные на известные и пропущенные
        known_samples, known_labels = [], []
        missing_samples, missing_labels = [], []

        for sample, label in zip(data, labels):
            if sample[attr_index] is None or sample[attr_index] == '?':
                missing_samples.append(sample)
                missing_labels.append(label)
            else:
                known_samples.append(sample)
                known_labels.append(label)

        # Пропускаем атрибут, если нет известных значений
        if not known_samples:
            continue

        attr_type = attribute_types[attr_index]

        # Обработка числовых атрибутов
        if attr_type == 'numerical':
            thresholds = generate_numeric_thresholds(known_samples, attr_index)
            for threshold in thresholds:
                left, right = split_numeric(known_samples, known_labels, attr_index, threshold)
                left_samples, left_labels = left
                right_samples, right_labels = right

                # Пропускаем, если одна из групп пуста
                if not left_labels or not right_labels:
                    continue

                # Вычисляем веса групп
                left_weight = len(left_labels)
                right_weight = len(right_labels)
                total_known = left_weight + right_weight

                # Распределяем пропущенные значения пропорционально
                left_labels_with_missing = left_labels + missing_labels
                right_labels_with_missing = right_labels + missing_labels
                left_weight += len(missing_labels) * (left_weight / total_known)
                right_weight += len(missing_labels) * (right_weight / total_known)

                # Вычисляем Gain Ratio
                gain_ratio = calculate_gain_ratio(
                    labels,
                    [left_labels_with_missing, right_labels_with_missing],
                    [left_weight, right_weight]
                )

                if verbose:
                    print(f"[NUM] Attribute {attr_index}, threshold {threshold:.3f}, gain ratio = {gain_ratio:.4f}")

                # Обновляем лучшее разбиение, если текущее лучше
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_attribute = attr_index
                    best_threshold = threshold
                    best_split_data = (left_samples, left_labels, right_samples, right_labels)

        # Обработка категориальных атрибутов
        else:
            splits = split_categorical(known_samples, known_labels, attr_index)
            total_known = sum(len(sub_labels) for _, sub_labels in splits.values())

            group_labels = []
            group_weights = []

            for value, (sub_samples, sub_labels) in splits.items():
                group_labels.append(sub_labels + missing_labels)
                weight = len(sub_labels) + len(missing_labels) * (len(sub_labels) / total_known)
                group_weights.append(weight)

            gain_ratio = calculate_gain_ratio(labels, group_labels, group_weights)

            if verbose:
                print(f"[CAT] Attribute {attr_index}, values = {len(splits)}, gain ratio = {gain_ratio:.4f}")

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_attribute = attr_index
                best_threshold = None
                best_split_data = splits

    if verbose:
        print(
            f"==> Selected attribute {best_attribute}, threshold {best_threshold}, gain ratio = {best_gain_ratio:.4f}\n")

    return best_attribute, best_threshold, best_split_data


def build_tree(data, labels, attribute_types, min_samples_split=2, depth=0):
    """
    Рекурсивно строит дерево решений.

    Args:
        data: Список объектов.
        labels: Метки классов.
        attribute_types: Типы атрибутов ('numerical' или 'categorical').
        min_samples_split: Минимальное количество объектов для разбиения.
        depth: Текущая глубина дерева.

    Returns:
        Узел дерева (DecisionNode).
    """
    # Если все метки одинаковы, создаем лист
    if len(set(labels)) == 1:
        return DecisionNode(label=labels[0], is_leaf=True)

    # Если слишком мало данных, создаем лист с наиболее частой меткой
    if len(data) < min_samples_split:
        return DecisionNode(label=find_most_common_label(labels), is_leaf=True)

    # Находим лучшее разбиение
    best_attribute, best_threshold, best_split_data = find_best_split(data, labels, attribute_types)

    # Если разбиение невозможно, создаем лист
    if best_attribute is None:
        return DecisionNode(label=find_most_common_label(labels), is_leaf=True)

    # Создаем узел для текущего разбиения
    node = DecisionNode(attribute=best_attribute, threshold=best_threshold)

    # Для числовых атрибутов
    if attribute_types[best_attribute] == 'numerical':
        left_samples, left_labels, right_samples, right_labels = best_split_data
        node.branches['<='] = build_tree(
            left_samples, left_labels, attribute_types, min_samples_split, depth + 1
        )
        node.branches['>'] = build_tree(
            right_samples, right_labels, attribute_types, min_samples_split, depth + 1
        )

    # Для категориальных атрибутов
    else:
        for value, (sub_samples, sub_labels) in best_split_data.items():
            node.branches[value] = build_tree(
                sub_samples, sub_labels, attribute_types, min_samples_split, depth + 1
            )

    return node