from collections import Counter

def majority_class_in_tree(node):

    # Если узел отсутствует, возвращаем None
    if node is None:
        return None

    # Если узел — лист, возвращаем его метку
    if node.is_leaf:
        return node.label

    # Собираем метки из всех дочерних узлов
    child_labels = []
    for child_node in node.branches.values():
        label = majority_class_in_tree(child_node)
        if label is not None:
            child_labels.append(label)

    # Возвращаем наиболее частую метку или None, если меток нет
    if child_labels:
        return Counter(child_labels).most_common(1)[0][0]
    return None

def predict_single(node, sample, attribute_types, default_class=0):

    # Если узел — лист, возвращаем его метку
    if node.is_leaf:
        return node.label

    attribute_index = node.attribute
    attribute_value = sample[attribute_index]

    # Для числового атрибута проверяем порог
    if attribute_types[attribute_index] == 'numerical':
        if attribute_value <= node.threshold:
            next_node = node.branches['<=']
        else:
            next_node = node.branches['>']
    # Для категориального атрибута ищем ветвь по значению
    else:
        if attribute_value not in node.branches:
            # Если значения нет в ветвях, возвращаем мажоритарную метку поддерева
            majority_label = majority_class_in_tree(node)
            return majority_label if majority_label is not None else default_class
        next_node = node.branches[attribute_value]

    # Рекурсивно продолжаем с дочерним узлом
    return predict_single(next_node, sample, attribute_types, default_class)

def predict_batch(tree, samples, attribute_types, default_class=0):

    predictions = []

    # Прогнозируем метку для каждого объекта
    for sample in samples:
        predicted_label = predict_single(tree, sample, attribute_types, default_class)
        # Если предсказание не удалось, используем метку по умолчанию
        if predicted_label is None:
            predicted_label = default_class
        predictions.append(predicted_label)

    return predictions