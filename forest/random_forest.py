import random
from collections import Counter
from tree.builder import build_tree
from tree.predict import predict_single

class RandomForestC45:
    def __init__(self, n_estimators=10, sample_ratio=1, min_samples_split=2, max_features='sqrt', random_state=None,
                 debug= False):
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.debug = debug  # Параметр для включения отладки
        self.trees = []
        self.attribute_types = None

    def fit(self, dataset, labels, attribute_types):
        if self.random_state:
            random.seed(self.random_state)
        self.attribute_types = attribute_types
        self.trees = []
        n_samples = len(dataset)
        for i in range(self.n_estimators):
            indices = [random.randint(0, n_samples - 1) for _ in range(int(n_samples * self.sample_ratio))]
            sampled_data = [dataset[i] for i in indices]
            sampled_labels = [labels[i] for i in indices]

            # Отладка бэггинга
            if self.debug:
                unique_indices = len(set(indices))
                print(f"Tree {i + 1}: {len(indices)} samples, {unique_indices} unique indices "
                      f"({unique_indices / len(indices) * 100:.2f}% unique)")
                print(f"First 10 indices: {indices[:10]}")

            tree = build_tree(sampled_data, sampled_labels, attribute_types, self.min_samples_split, self.max_features)
            self.trees.append(tree)

    def predict(self, dataset):
        predictions = []
        for x in dataset:
            votes = []
            for tree in self.trees:
                pred = predict_single(tree, x, self.attribute_types)
                if pred is not None:
                    votes.append(pred)
            if votes:
                majority = Counter(votes).most_common(1)[0][0]
                predictions.append(majority)
            else:
                predictions.append(None)
        return predictions