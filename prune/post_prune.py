from collections import Counter
from tree.node import DecisionNode
from tree.predict import predict_batch
from scipy.stats import norm
import math

def majority_class(labels):
    return Counter(labels).most_common(1)[0][0]

def compute_error(predictions, true_labels):
    return sum(p != y for p, y in zip(predictions, true_labels) if p is not None)

def pessimistic_error_with_CI(errors, total, z):
    if total == 0:
        return 0
    f = errors / total
    return (f + z**2 / (2 * total) + z * math.sqrt(f * (1 - f) / total + z**2 / (4 * total**2))) / (1 + z**2 / total)

def prune_tree(node, validation_data, validation_labels, attribute_types, confidence_factor=0.25):
    if node.is_leaf or not validation_data:
        return node

    if node.threshold is not None:
        left_data, left_labels, right_data, right_labels = [], [], [], []
        for x, y in zip(validation_data, validation_labels):
            if x[node.attribute] <= node.threshold:
                left_data.append(x)
                left_labels.append(y)
            else:
                right_data.append(x)
                right_labels.append(y)
        node.branches['<='] = prune_tree(node.branches['<='], left_data, left_labels, attribute_types, confidence_factor)
        node.branches['>'] = prune_tree(node.branches['>'], right_data, right_labels, attribute_types, confidence_factor)
    else:
        split_map = {}
        for x, y in zip(validation_data, validation_labels):
            key = x[node.attribute]
            if key not in split_map:
                split_map[key] = ([], [])
            split_map[key][0].append(x)
            split_map[key][1].append(y)
        for key in node.branches:
            if key in split_map:
                node.branches[key] = prune_tree(node.branches[key], split_map[key][0], split_map[key][1], attribute_types, confidence_factor)

    # Предсказания дерева до обрезки
    full_predictions = predict_batch(node, validation_data, attribute_types)
    full_errors = compute_error(full_predictions, validation_labels)

    # Предсказания после обрезки (если заменить на лист)
    majority = majority_class(validation_labels)
    temp_leaf = DecisionNode(label=majority, is_leaf=True)
    pruned_predictions = [majority for _ in validation_labels]
    pruned_errors = compute_error(pruned_predictions, validation_labels)

    # Пессимистические оценки ошибок
    z = norm.ppf(1 - confidence_factor)
    full_error_rate = pessimistic_error_with_CI(full_errors, len(validation_labels), z)
    pruned_error_rate = pessimistic_error_with_CI(pruned_errors, len(validation_labels), z)

    return temp_leaf if pruned_error_rate <= full_error_rate else node
