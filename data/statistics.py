import numpy as np

def compute_feature_stats(dataset):
    """
    Вычисляет Q1, Q3 и IQR для каждого числового признака
    """
    dataset = np.array(dataset)
    stats = []
    for i in range(dataset.shape[1]):
        column = dataset[:, i]
        q1 = np.percentile(column, 25)
        q3 = np.percentile(column, 75)
        iqr = q3 - q1
        stats.append({
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'min': column.min(),
            'max': column.max()
        })
    return stats
def is_outlier(x, feature_stats):
    for xi, stat in zip(x, feature_stats):
        lower = stat['q1'] - 1.5 * stat['iqr']
        upper = stat['q3'] + 1.5 * stat['iqr']
        if xi < lower or xi > upper:
            return True
    return False