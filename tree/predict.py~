from collections import Counter

def majority_class_in_tree(node):
    if node is None:
        return None
    if node.is_leaf:
        return node.label
    labels = []
    for child in node.branches.values():
        label = majority_class_in_tree(child)
        if label is not None:
            labels.append(label)
    return Counter(labels).most_common(1)[0][0] if labels else None

def predict_single(node, x, attribute_types, default_class=0):
    if node.is_leaf:
        return node.label

    if attribute_types[node.attribute] == 'numerical':
        if x[node.attribute] <= node.threshold:
            next_node = node.branches['<=']
        else:
            next_node = node.branches['>']
    else:
        value = x[node.attribute]
        if value not in node.branches:
            # Вместо None возвращаем мажоритарный класс поддерева
            majority = majority_class_in_tree(node)
            return majority if majority is not None else default_class
        next_node = node.branches[value]

    return predict_single(next_node, x, attribute_types, default_class)

def predict_batch(tree, dataset, attribute_types, default_class=0):
    predictions = []
    for x in dataset:
        pred = predict_single(tree, x, attribute_types, default_class)
        if pred is None:  # На случай, если majority_class_in_tree вернул None
            pred = default_class
        predictions.append(pred)
    return predictions