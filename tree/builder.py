import math
import random
from collections import Counter
from tree.node import DecisionNode
from tree.splitter import split_numeric, split_categorical, generate_numeric_thresholds
from tree.metrics import entropy


def find_most_common_label(labels):

    return Counter(labels).most_common(1)[0][0]


def calculate_gain_ratio(parent_labels, child_groups, child_weights):

    total_weight = sum(child_weights)

    # Взвешенная энтропия дочерних групп
    weighted_child_entropy = sum(
        (weight / total_weight) * entropy(group)
        for group, weight in zip(child_groups, child_weights)
        if weight > 0
    )

    # Информационный прирост
    information_gain = entropy(parent_labels) - weighted_child_entropy

    # Информация о разбиении
    split_information = -sum(
        (weight / total_weight) * math.log2(weight / total_weight)
        for weight in child_weights
        if weight > 0
    )

    if split_information == 0:
        return 0

    return information_gain / split_information


def find_best_split(data, labels, attribute_types, max_features=None, verbose=False):

    best_gain_ratio = -1
    best_attribute = None
    best_threshold = None
    best_split_data = None

    total_samples = len(data)
    n_features = len(data[0])

    # Определяем, какие признаки рассматривать
    if max_features is None:
        feature_indices = list(range(n_features))
    else:
        # Вычисляем количество признаков для выборки
        if max_features == 'sqrt':
            n_subset = int(math.sqrt(n_features))
        elif max_features == 'log2':
            n_subset = int(math.log2(n_features))
        elif isinstance(max_features, float):
            n_subset = int(max_features * n_features)
        else:
            n_subset = max_features
        # Случайно выбираем подмножество признаков
        n_subset = min(n_subset, n_features)
        feature_indices = random.sample(range(n_features), n_subset)

    # Перебираем только выбранные признаки
    for attr_index in feature_indices:

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

                if not left_labels or not right_labels:
                    continue

                left_weight = len(left_labels)
                right_weight = len(right_labels)
                total_known = left_weight + right_weight

                left_labels_with_missing = left_labels + missing_labels
                right_labels_with_missing = right_labels + missing_labels
                left_weight += len(missing_labels) * (left_weight / total_known)
                right_weight += len(missing_labels) * (right_weight / total_known)

                gain_ratio = calculate_gain_ratio(
                    labels,
                    [left_labels_with_missing, right_labels_with_missing],
                    [left_weight, right_weight]
                )

                if verbose:
                    print(f"[NUM] Attribute {attr_index}, threshold {threshold:.3f}, gain ratio = {gain_ratio:.4f}")

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
        print(f"==> Selected attribute {best_attribute}, threshold {best_threshold}, gain ratio = {best_gain_ratio:.4f}\n")

    return best_attribute, best_threshold, best_split_data


def build_tree(data, labels, attribute_types, min_samples_split=2, max_features=None, depth=0):

    if len(set(labels)) == 1:
        return DecisionNode(label=labels[0], is_leaf=True)

    if len(data) < min_samples_split:
        return DecisionNode(label=find_most_common_label(labels), is_leaf=True)

    # Передаем max_features в find_best_split
    best_attribute, best_threshold, best_split_data = find_best_split(data, labels, attribute_types, max_features)

    if best_attribute is None:
        return DecisionNode(label=find_most_common_label(labels), is_leaf=True)

    node = DecisionNode(attribute=best_attribute, threshold=best_threshold)

    if attribute_types[best_attribute] == 'numerical':
        left_samples, left_labels, right_samples, right_labels = best_split_data
        node.branches['<='] = build_tree(
            left_samples, left_labels, attribute_types, min_samples_split, max_features, depth + 1
        )
        node.branches['>'] = build_tree(
            right_samples, right_labels, attribute_types, min_samples_split, max_features, depth + 1
        )
    else:
        for value, (sub_samples, sub_labels) in best_split_data.items():
            node.branches[value] = build_tree(
                sub_samples, sub_labels, attribute_types, min_samples_split, max_features, depth + 1
            )

    return node