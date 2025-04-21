def print_tree(node, attribute_names=None, indent=""):
    if node.is_leaf:
        print(indent + f"🌿 Class: {node.label}")
    else:
        name = f"Attr {node.attribute}" if not attribute_names else attribute_names[node.attribute]
        if node.threshold is not None:
            print(indent + f"🔀 {name} <= {node.threshold}?")
            print(indent + "├── Yes:")
            print_tree(node.branches['<='], attribute_names, indent + "│   ")
            print(indent + "└── No:")
            print_tree(node.branches['>'], attribute_names, indent + "    ")
        else:
            for val, subnode in node.branches.items():
                print(indent + f"🔀 {name} == {val}:")
                print_tree(subnode, attribute_names, indent + "    ")

def count_nodes(node):
    if node.is_leaf:
        return 1
    return 1 + sum(count_nodes(child) for child in node.branches.values())

def tree_depth(node):
    if node.is_leaf:
        return 1
    return 1 + max(tree_depth(child) for child in node.branches.values())
