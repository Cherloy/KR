from sklearn.datasets import load_wine, load_iris, load_breast_cancer, load_digits
import numpy as np
import numpy as np

def load_iris_dataset(threshold=10):
    iris = load_breast_cancer()
    x = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names.tolist()  # Извлекаем имена классов

    attribute_types = []
    for col in x.T:
        unique_values = np.unique(col)
        if len(unique_values) <= threshold:
            attribute_types.append('categorical')
        else:
            attribute_types.append('numerical')

    return x.tolist(), y.tolist(), attribute_types, feature_names, class_names