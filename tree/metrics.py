from collections import Counter
import math

def entropy(labels):
    total = len(labels)
    if total == 0:
        return 0
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

def information_gain(parent_labels, splits):
    base_entropy = entropy(parent_labels)
    total = len(parent_labels)
    weighted_entropy = sum(
        (len(subset) / total) * entropy(subset)
        for subset in splits
    )
    return base_entropy - weighted_entropy
