from sklearn.datasets import load_wine, load_iris, load_breast_cancer, load_digits
import numpy as np
import pandas as pd
def load_iris_dataset(threshold=10):
    iris = load_digits()
    x = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names.tolist()

    attribute_types = []
    for col in x.T:
        unique_values = np.unique(col)
        if len(unique_values) <= threshold:
            attribute_types.append('categorical')
        else:
            attribute_types.append('numerical')

    return x.tolist(), y.tolist(), attribute_types, feature_names, class_names

def load_wine_dataset(threshold=10):
    iris = load_wine()
    x = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names.tolist()

    attribute_types = []
    for col in x.T:
        unique_values = np.unique(col)
        if len(unique_values) <= threshold:
            attribute_types.append('categorical')
        else:
            attribute_types.append('numerical')

    return x.tolist(), y.tolist(), attribute_types, feature_names, class_names

def load_breast_cancer_dataset(threshold=10):
    iris = load_breast_cancer()
    x = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names.tolist()

    attribute_types = []
    for col in x.T:
        unique_values = np.unique(col)
        if len(unique_values) <= threshold:
            attribute_types.append('categorical')
        else:
            attribute_types.append('numerical')

    return x.tolist(), y.tolist(), attribute_types, feature_names, class_names

def load_digits_dataset(threshold=10):
    iris = load_digits()
    x = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names.tolist()

    attribute_types = []
    for col in x.T:
        unique_values = np.unique(col)
        if len(unique_values) <= threshold:
            attribute_types.append('categorical')
        else:
            attribute_types.append('numerical')

    return x.tolist(), y.tolist(), attribute_types, feature_names, class_names

def load_mushroom_dataset():

    # URL датасета Mushroom из UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

    feature_names = [
        'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]


    data = pd.read_csv(url, header=None, names=['class'] + feature_names)
    # Извлекаем метки (первый столбец: 'e' или 'p')
    labels = data['class'].tolist()

    # Извлекаем признаки (все столбцы, кроме первого)
    samples = data[feature_names].values.tolist()

    attribute_types = ['categorical' for _ in feature_names]

    # Имена классов
    class_names = ['edible', 'poisonous']

    return samples, labels, attribute_types, feature_names, class_names


def load_titanic_dataset():
    # URL датасета Titanic (используем общедоступный датасет)
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

    # Названия столбцов (основные признаки из датасета)
    feature_names = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
        'Fare', 'Embarked'
    ]

    # Загружаем датасет
    data = pd.read_csv(url)

    # Извлекаем метки (столбец Survived: 0 или 1)
    labels = data['Survived'].tolist()

    # Извлекаем признаки (выбираем только указанные столбцы)
    samples = data[feature_names].values.tolist()

    # Типы признаков: категориальные и числовые
    attribute_types = [
        'categorical',  # Pclass
        'categorical',  # Sex
        'numeric',  # Age
        'numeric',  # SibSp
        'numeric',  # Parch
        'numeric',  # Fare
        'categorical'  # Embarked
    ]

    # Имена классов
    class_names = ['not survived', 'survived']

    return samples, labels, attribute_types, feature_names, class_names