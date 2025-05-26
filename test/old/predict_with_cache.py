import random
from collections import Counter
from tree.builder import build_tree
from tree.predict import predict_single
from forest.random_forest import RandomForestC45

class RandomForestC45Cached(RandomForestC45):
    """
    Случайный лес C4.5 с кэшированием предсказаний для ускорения метода predict.
    """
    def predict(self, dataset):
        """
        Делает предсказания для набора данных с использованием кэширования.

        Args:
            dataset: Список объектов (каждый объект — список значений признаков).

        Returns:
            Список предсказанных меток.
        """
        # Инициализируем кэш: {tree_index: {input_tuple: prediction}}
        cache = {i: {} for i in range(len(self.trees))}
        predictions = []

        for x in dataset:
            # Преобразуем входной объект в хэшируемый кортеж
            input_tuple = tuple(x)
            votes = []

            # Проверяем предсказания для каждого дерева
            for i, tree in enumerate(self.trees):
                # Если предсказание для этого объекта и дерева уже в кэше, используем его
                if input_tuple in cache[i]:
                    pred = cache[i][input_tuple]
                else:
                    # Иначе выполняем предсказание и сохраняем в кэш
                    pred = predict_single(tree, x, self.attribute_types)
                    cache[i][input_tuple] = pred
                if pred is not None:
                    votes.append(pred)

            # Определяем мажоритарное предсказание
            if votes:
                majority = Counter(votes).most_common(1)[0][0]
                predictions.append(majority)
            else:
                predictions.append(None)

        return predictions