from sklearn.datasets import load_wine, load_iris, load_breast_cancer, load_digits
import numpy as np

def load_iris_dataset(threshold=10):
    wine = load_iris()
    X = wine.data
    y = wine.target

    attribute_types = []
    for col in X.T:
        unique_values = np.unique(col)
        if len(unique_values) <= threshold:
            attribute_types.append('categorical')
        else:
            attribute_types.append('numerical')

    return X.tolist(), y.tolist(), attribute_types