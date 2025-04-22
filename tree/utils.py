def print_tree(node, attribute_names=None, indent="", attribute_types=None):
    """
    Выводит структуру дерева решений в читаемом виде.

    Для числовых признаков показывает пороговые условия (<= порог).
    Для категориальных признаков показывает ветви для каждого значения.

    Args:
        node: Узел дерева решений (DecisionNode).
        attribute_names: Список имен признаков (опционально, иначе 'Attr N').
        indent: Текущий отступ для форматирования (строка).
        attribute_types: Список типов признаков ('numerical' или 'categorical', опционально).
    """
    if node.is_leaf:
        # Для листа выводим метку класса
        print(f"{indent}Class: {node.label}")
        return

    # Получаем имя признака
    name = f"Attr {node.attribute}" if not attribute_names else attribute_names[node.attribute]
    attr_type = attribute_types[node.attribute] if attribute_types else 'unknown'

    if node.threshold is not None and attr_type != 'categorical':
        # Для числовых признаков выводим условие с порогом
        print(f"{indent}{name} <= {node.threshold:.2f}?")
        print(f"{indent}├── Yes:")
        print_tree(node.branches['<='], attribute_names, f"{indent}│   ", attribute_types)
        print(f"{indent}└── No:")
        print_tree(node.branches['>'], attribute_names, f"{indent}    ", attribute_types)
    else:
        # Для категориальных признаков выводим ветви для каждого значения
        values = list(node.branches.keys())
        for i, val in enumerate(values):
            is_last = i == len(values) - 1
            prefix = "└──" if is_last else "├──"
            next_indent = f"{indent}    " if is_last else f"{indent}│   "
            print(f"{indent}{prefix} {name} = {val}:")
            print_tree(node.branches[val], attribute_names, next_indent, attribute_types)

def count_nodes(node):
    """
    Подсчитывает общее количество узлов в дереве.

    Args:
        node: Узел дерева решений (DecisionNode).

    Returns:
        Количество узлов (целое число).
    """
    if node.is_leaf:
        return 1
    return 1 + sum(count_nodes(child) for child in node.branches.values())

def tree_depth(node):
    """
    Вычисляет максимальную глубину дерева.

    Args:
        node: Узел дерева решений (DecisionNode).

    Returns:
        Глубина дерева (целое число).
    """
    if node.is_leaf:
        return 1
    return 1 + max(tree_depth(child) for child in node.branches.values())