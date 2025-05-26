from collections import Counter
from tree.node import DecisionNode
from tree.predict import predict_batch
from scipy.stats import norm
import math

def majority_class(labels):

    return Counter(labels).most_common(1)[0][0]

def compute_error(predictions, true_labels):
    return sum(pred != true_label for pred, true_label in zip(predictions, true_labels) if pred is not None)

def pessimistic_error(errors, total, z_score):

    if total == 0:
        return 0

    error_rate = errors / total
    variance_term = error_rate * (1 - error_rate) / total
    correction_term = z_score**2 / (4 * total**2)

    # Формула пессимистической ошибки
    numerator = error_rate + (z_score**2 / (2 * total)) + z_score * math.sqrt(variance_term + correction_term)
    denominator = 1 + z_score**2 / total

    return numerator / denominator

def prune_tree(node, validation_samples, validation_labels, attribute_types, confidence_factor=0.25, error_tolerance =1):

    # Если узел — лист или нет валидационных данных, возвращаем его без изменений
    if node.is_leaf or not validation_samples:
        return node

    # Разделяем валидационные данные для дочерних узлов
    if node.threshold is not None:
        # Числовой атрибут: разделяем на ≤ порога и > порога
        left_samples, left_labels = [], []
        right_samples, right_labels = [], []

        for sample, label in zip(validation_samples, validation_labels):
            attribute_value = sample[node.attribute]
            if attribute_value <= node.threshold:
                left_samples.append(sample)
                left_labels.append(label)
            else:
                right_samples.append(sample)
                right_labels.append(label)

        # Рекурсивно обрезаем дочерние ветви
        node.branches['<='] = prune_tree(
            node.branches['<='], left_samples, left_labels, attribute_types, confidence_factor
        )
        node.branches['>'] = prune_tree(
            node.branches['>'], right_samples, right_labels, attribute_types, confidence_factor
        )
    else:
        # Категориальный атрибут: группируем данные по значениям атрибута
        category_groups = {}
        for sample, label in zip(validation_samples, validation_labels):
            category = sample[node.attribute]
            if category not in category_groups:
                category_groups[category] = ([], [])
            category_groups[category][0].append(sample)
            category_groups[category][1].append(label)

        # Рекурсивно обрезаем ветви для существующих категорий
        for category in node.branches:
            if category in category_groups:
                node.branches[category] = prune_tree(
                    node.branches[category],
                    category_groups[category][0],
                    category_groups[category][1],
                    attribute_types,
                    confidence_factor
                )

    # Оцениваем ошибки текущего дерева
    current_predictions = predict_batch(node, validation_samples, attribute_types)
    current_error_count = compute_error(current_predictions, validation_labels)

    # Оцениваем ошибки, если заменить узел на лист с мажоритарной меткой
    majority_label = majority_class(validation_labels)
    leaf_node = DecisionNode(label=majority_label, is_leaf=True)
    leaf_predictions = [majority_label for _ in validation_labels]
    leaf_error_count = compute_error(leaf_predictions, validation_labels)

    # Вычисляем пессимистические оценки ошибок
    z_score = norm.ppf(1 - confidence_factor)
    current_error_rate = pessimistic_error(
        current_error_count, len(validation_labels), z_score
    )
    leaf_error_rate = pessimistic_error(
        leaf_error_count, len(validation_labels), z_score
    )

    # Если лист дает меньшую или равную ошибку, заменяем узел на лист
    if leaf_error_rate <= current_error_rate * error_tolerance:
        return leaf_node
    return node