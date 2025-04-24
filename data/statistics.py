import numpy as np
from collections import Counter

def compute_feature_stats(samples, attribute_types):
    """
    Вычисляет статистики для каждого признака в датасете с учетом их типов.

    Для числовых признаков: Q1, Q3, IQR, минимум и максимум.
    Для категориальных признаков: частоты уникальных значений.

    Args:
        samples: Список объектов (каждый объект — список значений признаков).
        attribute_types: Список типов признаков ('numerical' или 'categorical').

    Returns:
        Список словарей со статистиками для каждого признака.
    """
    # Преобразуем датасет в массив NumPy для удобной работы со столбцами
    sample_array = np.array(samples, dtype=object)  # dtype=object для строк
    feature_stats = []

    # Перебираем столбцы и их типы
    for column, attr_type in zip(sample_array.T, attribute_types):
        if attr_type == 'numerical':
            # Для числовых признаков вычисляем Q1, Q3, IQR, min, max
            try:
                q1 = np.percentile(column, 25)
                q3 = np.percentile(column, 75)
                iqr = q3 - q1
                min_value = np.min(column)
                max_value = np.max(column)
                stats = {
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'min': min_value,
                    'max': max_value
                }
            except (TypeError, ValueError):
                # Если столбец содержит нечисловые данные, возвращаем пустые статистики
                stats = {
                    'q1': None,
                    'q3': None,
                    'iqr': None,
                    'min': None,
                    'max': None
                }
        else:
            # Для категориальных признаков вычисляем частоты уникальных значений
            value_counts = Counter(column)
            stats = {'value_counts': dict(value_counts)}

        feature_stats.append(stats)

    return feature_stats

def is_outlier(sample, feature_stats, attribute_types):
    """
    Проверяет, является ли объект выбросом на основе числовых признаков.

    Использует правило 1.5 * IQR для числовых признаков. Категориальные признаки игнорируются.

    Args:
        sample: Объект (список значений признаков).
        feature_stats: Список словарей со статистиками признаков.
        attribute_types: Список типов признаков ('numerical' или 'categorical').

    Returns:
        True, если объект — выброс по хотя бы одному числовому признаку, иначе False.
    """
    for value, stats, attr_type in zip(sample, feature_stats, attribute_types):
        # Проверяем только числовые признаки
        if attr_type == 'numerical' and stats['iqr'] is not None:
            lower_bound = stats['q1'] - 3 * stats['iqr']
            upper_bound = stats['q3'] + 3 * stats['iqr']
            if value < lower_bound or value > upper_bound:
                return True
    return False