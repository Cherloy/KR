def print_tree(node, attribute_names=None, indent="", attribute_types=None):

    if node.is_leaf:
        # Для листа выводим метку класса
        print(f"{indent}Class: {node.label}")
        return

    # Получаем имя признака
    name = f"Attr {node.attribute}" if attribute_names is None else attribute_names[node.attribute]
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

    if node.is_leaf:
        return 1
    return 1 + sum(count_nodes(child) for child in node.branches.values())

def tree_depth(node):

    if node.is_leaf:
        return 1
    return 1 + max(tree_depth(child) for child in node.branches.values())