import pandas as pd
def load_iris_dataset():
    """
    Загружает датасет Mushroom с категориальными признаками в виде строк.

    Датасет содержит информацию о грибах, где все атрибуты (например, цвет шляпки, запах)
    представлены строками (например, 'n' для 'brown', 'f' для 'foul'). Целевая переменная
    указывает, съедобный гриб ('e') или ядовитый ('p').

    Returns:
        Кортеж: (список_объектов, список_меток, список_типов_атрибутов, имена_признаков, имена_классов).
    """
    # URL датасета Mushroom из UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

    # Названия столбцов (из описания датасета)
    feature_names = [
        'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]

    # Загружаем датасет
    data = pd.read_csv(url, header=None, names=['class'] + feature_names)
    # Извлекаем метки (первый столбец: 'e' или 'p')
    labels = data['class'].tolist()

    # Извлекаем признаки (все столбцы, кроме первого)
    samples = data[feature_names].values.tolist()

    # Все признаки в этом датасете категориальные (строковые)
    attribute_types = ['categorical' for _ in feature_names]

    # Имена классов
    class_names = ['edible', 'poisonous']

    return samples, labels, attribute_types, feature_names, class_names