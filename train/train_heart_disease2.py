from data.load_dataset import load_iris_dataset
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
    X, y, attribute_types, feature_names, class_names = load_iris_dataset()  # Получаем class_names

    # Разбиваем на тренировочную и тестовую выборки
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Далее делим тренировочную выборку на build и validation
    X_build, X_val, y_build, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    feature_stats = compute_feature_stats(X, attribute_types)

    print("===> C4.5 Decision Tree")
    tree = build_tree(X_build, y_build, attribute_types)
    print_tree(tree, attribute_types)
    print("Глубина:", tree_depth(tree))
    print("Узлов:", count_nodes(tree))

    preds = predict_batch(tree, X_test, attribute_types)
    acc = accuracy_score(y_test, preds)
    print("Accuracy (before pruning):", acc)

    # Глубокая копия, чтобы обрезка не изменила исходное дерево
    tree_pruned = prune_tree(deepcopy(tree), X_val, y_val, attribute_types)
    print_tree(tree_pruned, attribute_types)
    print("Глубина:", tree_depth(tree_pruned))
    print("Узлов:", count_nodes(tree_pruned))

    pruned_preds = predict_batch(tree_pruned, X_test, attribute_types)
    pruned_acc = accuracy_score(y_test, pruned_preds)
    print("Accuracy (after pruning):", pruned_acc)

    print("\n===> Random Forest")
    forest = RandomForestC45(n_estimators=10, random_state=42)
    forest.fit(X_build, y_build, attribute_types)

    # Статистика до обрезки
    depths_before = [tree_depth(tree) for tree in forest.trees]
    nodes_before = [count_nodes(tree) for tree in forest.trees]
    print(f"Средняя глубина (до обрезки): {sum(depths_before) / len(depths_before):.2f}")
    print(f"Среднее число узлов (до обрезки): {sum(nodes_before) / len(nodes_before):.2f}")

    forest_preds = forest.predict(X_test)
    forest_acc = accuracy_score(y_test, forest_preds)
    print("Accuracy (random forest before pruning):", forest_acc)

    # Постобрезка всех деревьев случайного леса
    forest.trees = [
        prune_tree(tree, X_val, y_val, attribute_types)
        for tree in forest.trees
    ]

    # Статистика после обрезки
    depths_after = [tree_depth(tree) for tree in forest.trees]
    nodes_after = [count_nodes(tree) for tree in forest.trees]
    print(f"\nСредняя глубина (после обрезки): {sum(depths_after) / len(depths_after):.2f}")
    print(f"Среднее число узлов (после обрезки): {sum(nodes_after) / len(nodes_after):.2f}")

    forest_preds = forest.predict(X_test)
    forest_acc = accuracy_score(y_test, forest_preds)
    print("Accuracy (random forest):", forest_acc)

    return tree_pruned, forest, attribute_types, feature_stats, feature_names, class_names


from copy import deepcopy

def interactive_prediction(tree, forest, attribute_types, feature_stats, feature_names, class_names):
    # Проверяем, что количество признаков совпадает с feature_stats
    if len(feature_names) != len(feature_stats):
        raise ValueError("Количество признаков в feature_names и feature_stats не совпадает")

    # Формируем описания признаков на основе min и max из feature_stats
    feature_descriptions = {
        feature_names[i]: f"{feature_names[i]} (от {stats['min']:.1f} до {stats['max']:.1f})"
        for i, stats in enumerate(feature_stats)
    }

    print("\n=== Интерактивный режим ===")
    print("Введите данные ириса последовательно по признакам:")

    user_input = []
    for feature in feature_names:
        while True:
            print(f"\n{feature_descriptions[feature]}")
            try:
                value = float(input("Введите значение: "))
                user_input.append(value)
                break
            except ValueError:
                print("Ошибка: введите числовое значение")

    if is_outlier(user_input, feature_stats, attribute_types):
        print("\n⚠️ Введены значения, которые считаются выбросами. Модель может не дать корректного предсказания.")

    # Предсказание с использованием дерева решений
    tree_pred = predict_single(tree, user_input, attribute_types)
    print(f"\nДерево решений: {class_names[tree_pred] if tree_pred is not None else 'Ошибка классификации'}")

    # Предсказание с использованием случайного леса
    forest_pred = forest.predict([user_input])[0]
    print(f"Случайный лес: {class_names[forest_pred] if forest_pred is not None else 'Ошибка классификации'}")


if __name__ == "__main__":
    trained_tree, trained_forest, attribute_types, feature_stats, feature_names, class_names = train_and_evaluate()
    interactive_prediction(trained_tree, trained_forest, attribute_types, feature_stats, feature_names, class_names)