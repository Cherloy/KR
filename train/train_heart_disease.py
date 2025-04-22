from data.load_dataset_cat import load_iris_dataset
from tree.builder import build_tree
from tree.predict import predict_batch, predict_single
from prune.post_prune import prune_tree
from forest.random_forest import RandomForestC45
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data.statistics import compute_feature_stats, is_outlier
from tree.utils import print_tree, count_nodes, tree_depth
from copy import deepcopy

def train_and_evaluate():
    """
    Обучает и оценивает дерево решений C4.5 и случайный лес на датасете.

    Returns:
        Кортеж: (обрезанное_дерево, случайный_лес, типы_признаков, статистики_признаков,
                 имена_признаков, имена_классов).
    """
    # Загружаем датасет
    samples, labels, attribute_types, feature_names, class_names = load_iris_dataset()

    # Разделяем данные на тренировочные и тестовые
    temp_samples, test_samples, temp_labels, test_labels = train_test_split(
        samples, labels, test_size=0.2, random_state=42
    )

    # Делим тренировочные данные на build и validation
    build_samples, val_samples, build_labels, val_labels = train_test_split(
        temp_samples, temp_labels, test_size=0.2, random_state=42
    )

    # Вычисляем статистики признаков
    feature_stats = compute_feature_stats(samples, attribute_types)

    # Обучение и оценка дерева решений
    print("===> C4.5 Decision Tree")
    tree = build_tree(build_samples, build_labels, attribute_types)
    print_tree(tree, attribute_types)
    print("Глубина:", tree_depth(tree))
    print("Узлов:", count_nodes(tree))

    predictions = predict_batch(tree, test_samples, attribute_types)
    accuracy = accuracy_score(test_labels, predictions)
    print("Accuracy (before pruning):", accuracy)

    # Обрезаем дерево
    pruned_tree = prune_tree(deepcopy(tree), val_samples, val_labels, attribute_types)
    print_tree(pruned_tree, attribute_types)
    print("Глубина:", tree_depth(pruned_tree))
    print("Узлов:", count_nodes(pruned_tree))

    pruned_predictions = predict_batch(pruned_tree, test_samples, attribute_types)
    pruned_accuracy = accuracy_score(test_labels, pruned_predictions)
    print("Accuracy (after pruning):", pruned_accuracy)

    # Обучение и оценка случайного леса
    print("\n===> Random Forest")
    forest = RandomForestC45(n_estimators=10, random_state=42)
    forest.fit(build_samples, build_labels, attribute_types)

    # Статистика леса до обрезки
    depths_before = [tree_depth(tree) for tree in forest.trees]
    nodes_before = [count_nodes(tree) for tree in forest.trees]
    print(f"Средняя глубина (до обрезки): {sum(depths_before) / len(depths_before):.2f}")
    print(f"Среднее число узлов (до обрезки): {sum(nodes_before) / len(nodes_before):.2f}")

    forest_predictions = forest.predict(test_samples)
    forest_accuracy = accuracy_score(test_labels, forest_predictions)
    print("Accuracy (random forest before pruning):", forest_accuracy)

    # Обрезаем все деревья в лесу
    forest.trees = [
        prune_tree(tree, val_samples, val_labels, attribute_types)
        for tree in forest.trees
    ]

    # Статистика леса после обрезки
    depths_after = [tree_depth(tree) for tree in forest.trees]
    nodes_after = [count_nodes(tree) for tree in forest.trees]
    print(f"Средняя глубина (после обрезки): {sum(depths_after) / len(depths_after):.2f}")
    print(f"Среднее число узлов (после обрезки): {sum(nodes_after) / len(nodes_after):.2f}")

    forest_predictions = forest.predict(test_samples)
    forest_accuracy = accuracy_score(test_labels, forest_predictions)
    print("Accuracy (random forest):", forest_accuracy)

    return pruned_tree, forest, attribute_types, feature_stats, feature_names, class_names

def interactive_prediction(tree, forest, attribute_types, feature_stats, feature_names, class_names):
    """
    Интерактивно запрашивает значения признаков у пользователя и делает предсказания.

    Для числовых признаков показывает диапазон (min-max), для категориальных — возможные значения.
    Обрабатывает строковые метки классов (например, 'e', 'p') и сопоставляет их с именами классов.

    Args:
        tree: Обрезанное дерево решений C4.5.
        forest: Случайный лес.
        attribute_types: Список типов признаков ('numerical' или 'categorical').
        feature_stats: Список словарей со статистиками признаков.
        feature_names: Список имен признаков.
        class_names: Список имен классов (например, ['edible', 'poisonous']).
    """
    # Проверяем согласованность данных
    if len(feature_names) != len(feature_stats) or len(feature_names) != len(attribute_types):
        raise ValueError("Несоответствие количества признаков в feature_names, feature_stats или attribute_types")

    # Создаем словарь для сопоставления меток классов с их именами
    # Предполагаем, что метки 'e' и 'p' соответствуют 'edible' и 'poisonous'
    label_to_class = {'e': 'edible', 'p': 'poisonous'}

    # Формируем описания признаков
    feature_descriptions = {}
    for i, (name, stats, attr_type) in enumerate(zip(feature_names, feature_stats, attribute_types)):
        if attr_type == 'numerical' and stats.get('min') is not None:
            # Для числовых признаков показываем диапазон
            description = f"{name} (от {stats['min']:.1f} до {stats['max']:.1f})"
        else:
            # Для категориальных признаков показываем возможные значения
            values = ', '.join(f"'{k}'" for k in stats.get('value_counts', {}).keys())
            description = f"{name} (возможные значения: {values or 'неизвестно'})"
        feature_descriptions[name] = description

    print("\n=== Интерактивный режим ===")
    print("Введите данные для предсказания последовательно по признакам:")

    user_input = []
    for i, feature in enumerate(feature_names):
        attr_type = attribute_types[i]
        while True:
            print(f"\n{feature_descriptions[feature]}")
            try:
                user_value = input("Введите значение: ")
                if attr_type == 'numerical':
                    # Для числовых признаков ожидаем число
                    value = float(user_value)
                else:
                    # Для категориальных признаков принимаем строку
                    value = user_value.strip()
                    # Проверяем, что значение допустимо
                    if value not in feature_stats[i].get('value_counts', {}):
                        print(f"Предупреждение: значение '{value}' не встречалось в данных.")
                user_input.append(value)
                break
            except ValueError:
                print("Ошибка: введите число для числового признака или корректное значение для категориального")

    # Проверяем, является ли введенный объект выбросом (только для числовых признаков)
    if is_outlier(user_input, feature_stats, attribute_types):
        print("\n⚠️ Введены значения, которые считаются выбросами. Модель может не дать корректного предсказания.")

    # Предсказание с использованием дерева решений
    tree_prediction = predict_single(tree, user_input, attribute_types)
    tree_class = label_to_class.get(tree_prediction, 'Ошибка классификации')
    print(f"\nДерево решений: {tree_class}")

    # Предсказание с использованием случайного леса
    forest_prediction = forest.predict([user_input])[0]
    forest_class = label_to_class.get(forest_prediction, 'Ошибка классификации')
    print(f"Случайный лес: {forest_class}")

if __name__ == "__main__":
    trained_tree, trained_forest, attribute_types, feature_stats, feature_names, class_names = train_and_evaluate()
    interactive_prediction(trained_tree, trained_forest, attribute_types, feature_stats, feature_names, class_names)