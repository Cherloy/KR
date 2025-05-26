import numpy as np
from collections import Counter

def compute_feature_stats(samples, attribute_types):
    sample_array = np.array(samples, dtype=object)
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

    for value, stats, attr_type in zip(sample, feature_stats, attribute_types):
        if attr_type == 'numerical' and stats['iqr'] is not None:
            lower_bound = stats['q1'] - 3 * stats['iqr']
            upper_bound = stats['q3'] + 3 * stats['iqr']
            if value < lower_bound or value > upper_bound:
                return True
    return False