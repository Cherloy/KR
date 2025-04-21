from tree.predict import predict_single
class DecisionNode:
    def __init__(self, attribute=None, threshold=None, branches=None, label=None, is_leaf=False):
        self.attribute = attribute
        self.threshold = threshold
        self.branches = branches or {}
        self.label = label
        self.is_leaf = is_leaf

    def is_terminal(self):
        return self.is_leaf