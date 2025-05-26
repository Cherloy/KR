import numpy as np
import pandas as pd
import time
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from forest.random_forest import RandomForestC45
from data.load_dataset import load_mushroom_dataset, load_wine_dataset, load_iris_dataset, load_breast_cancer_dataset, \
    load_digits_dataset
from collections import defaultdict
from sklearn.metrics import confusion_matrix


def preprocess_data(data: List[List[float]], labels: List, attribute_types: List[str]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    data_rf = np.array(data, dtype=object)
    labels_str = np.array([str(label) for label in labels])

    le = LabelEncoder()
    labels_num = le.fit_transform(labels_str)

    data_df = pd.DataFrame(data)
    for i, attr_type in enumerate(attribute_types):
        col = data_df[i]
        if attr_type == 'numerical':
            col_numeric = np.array([float(x) if x is not None and x != '' else np.nan for x in col])
            mask = ~np.isnan(col_numeric)
            if mask.sum() > 0:
                median = np.median(col_numeric[mask])
                col_numeric[~mask] = median
            data_df[i] = col_numeric
        else:
            mask = col != None
            if mask.sum() > 0:
                mode = col[mask].mode()[0]
                data_df.loc[~mask, i] = mode

    data_sklearn = pd.get_dummies(data_df, columns=[i for i, t in enumerate(attribute_types) if t == 'categorical'])
    data_sklearn = data_sklearn.to_numpy()

    return data_rf, labels_str, labels_num, attribute_types, data_sklearn


def evaluate_model(dataset_name: str, data: List[List[float]], labels: List, attribute_types: List[str],
                   n_runs: int = 30) -> Tuple[
    dict, dict]:
    data_rf, labels_str, labels_num, attribute_types, data_sklearn = preprocess_data(data, labels, attribute_types)

    metrics_list = defaultdict(list)
    metrics_sklearn_list = defaultdict(list)

    for run in range(n_runs):
        # Разделение данных с разным random_state для каждого запуска
        X_train_rf, X_test_rf, y_train_str, y_test_str = train_test_split(data_rf, labels_str, test_size=0.2,
                                                                          random_state=42 + run)
        X_train_sklearn, X_test_sklearn, y_train_num, y_test_num = train_test_split(data_sklearn, labels_num,
                                                                                    test_size=0.2,
                                                                                    random_state=42 + run)

        # Собственная модель
        start_time = time.time()
        model = RandomForestC45(n_estimators=10, sample_ratio=1.0, min_samples_split=2, max_features='sqrt',
                                random_state=42 + run)
        model.fit(X_train_rf, y_train_str, attribute_types)
        y_pred = model.predict(X_test_rf)
        rf_time = time.time() - start_time

        # Эталонная модель
        start_time = time.time()
        sklearn_model = RandomForestClassifier(n_estimators=10, max_features='sqrt', random_state=42 + run)
        sklearn_model.fit(X_train_sklearn, y_train_num)
        y_pred_sklearn = sklearn_model.predict(X_test_sklearn)
        sklearn_time = time.time() - start_time

        # Матрица ошибок для специфичности
        cm_rf = confusion_matrix(y_test_str, y_pred)
        cm_sklearn = confusion_matrix(y_test_num, y_pred_sklearn)

        # Специфичность
        specificity_rf = 0
        specificity_sklearn = 0
        for i in range(cm_rf.shape[0]):
            tn_rf = cm_rf.sum() - (cm_rf[i, :].sum() + cm_rf[:, i].sum() - cm_rf[i, i])
            fp_rf = cm_rf[:, i].sum() - cm_rf[i, i]
            specificity_rf += tn_rf / (tn_rf + fp_rf) if (tn_rf + fp_rf) > 0 else 0

            tn_sklearn = cm_sklearn.sum() - (cm_sklearn[i, :].sum() + cm_sklearn[:, i].sum() - cm_sklearn[i, i])
            fp_sklearn = cm_sklearn[:, i].sum() - cm_sklearn[i, i]
            specificity_sklearn += tn_sklearn / (tn_sklearn + fp_sklearn) if (tn_sklearn + fp_sklearn) > 0 else 0

        specificity_rf /= cm_rf.shape[0]
        specificity_sklearn /= cm_sklearn.shape[0]

        # Сбор метрик
        metrics_list['accuracy'].append(accuracy_score(y_test_str, y_pred))
        metrics_list['precision'].append(precision_score(y_test_str, y_pred, average='weighted'))
        metrics_list['recall'].append(recall_score(y_test_str, y_pred, average='weighted'))
        metrics_list['f1_score'].append(f1_score(y_test_str, y_pred, average='weighted'))
        metrics_list['specificity'].append(specificity_rf)
        metrics_list['time'].append(rf_time)

        metrics_sklearn_list['accuracy'].append(accuracy_score(y_test_num, y_pred_sklearn))
        metrics_sklearn_list['precision'].append(precision_score(y_test_num, y_pred_sklearn, average='weighted'))
        metrics_sklearn_list['recall'].append(recall_score(y_test_num, y_pred_sklearn, average='weighted'))
        metrics_sklearn_list['f1_score'].append(f1_score(y_test_num, y_pred_sklearn, average='weighted'))
        metrics_sklearn_list['specificity'].append(specificity_sklearn)
        metrics_sklearn_list['time'].append(sklearn_time)

    # Вычисление средних значений
    metrics = {key: np.mean(values) for key, values in metrics_list.items()}
    metrics_sklearn = {key: np.mean(values) for key, values in metrics_sklearn_list.items()}

    # Вычисление дисперсий
    variances = {key: np.var(values) for key, values in metrics_list.items()}
    variances_sklearn = {key: np.var(values) for key, values in metrics_sklearn_list.items()}

    return metrics, metrics_sklearn, variances, variances_sklearn


def test_example(data: List[List[float]], labels: List, attribute_types: List[str], dataset_name: str) -> Tuple[
    str, str, str]:
    data_rf, labels_str, labels_num, attribute_types, data_sklearn = preprocess_data(data, labels, attribute_types)
    X_train_rf, X_test_rf, y_train_str, y_test_str = train_test_split(data_rf, labels_str, test_size=0.2,
                                                                      random_state=42)
    X_train_sklearn, X_test_sklearn, y_train_num, y_test_num = train_test_split(data_sklearn, labels_num, test_size=0.2,
                                                                                random_state=42)
    test_sample_rf = X_test_rf[0]
    test_sample_sklearn = X_test_sklearn[0]
    true_label = str(y_test_str[0])

    model = RandomForestC45(n_estimators=10, sample_ratio=1.0, min_samples_split=2, max_features='sqrt',
                            random_state=42)
    model.fit(X_train_rf, y_train_str, attribute_types)
    pred_label = model.predict([test_sample_rf])[0]

    sklearn_model = RandomForestClassifier(n_estimators=10, max_features='sqrt', random_state=42)
    sklearn_model.fit(X_train_sklearn, y_train_num)
    pred_label_sklearn = str(sklearn_model.predict([test_sample_sklearn])[0])

    print(f"\nТестовый пример ({dataset_name}):")
    print(f"Входные признаки: {test_sample_rf}")
    print(f"Ожидаемая метка: {true_label}")
    print(f"Предсказанная (собственная): {pred_label}")
    print(f"Предсказанная (sklearn): {pred_label_sklearn}")

    return true_label, pred_label, pred_label_sklearn


def main():
    datasets = {
        'Mushroom': load_mushroom_dataset(),
        'Wine': load_wine_dataset(),
        'Iris': load_iris_dataset(),
        'Breast Cancer': load_breast_cancer_dataset(),
        'Digits': load_digits_dataset()
    }

    results = defaultdict(list)
    variances_results = defaultdict(list)

    # Проверка на тестовом примере для каждого датасета
    for name, (data, labels, attr_types, _, _) in datasets.items():
        test_example(data, labels, attr_types, name)

    # Оценка на всех датасетах
    for name, (data, labels, attr_types, _, _) in datasets.items():
        metrics, metrics_sklearn, variances, variances_sklearn = evaluate_model(name, data, labels, attr_types,
                                                                                n_runs=30)
        results[name].append(('Собственная', metrics))
        results[name].append(('sklearn', metrics_sklearn))
        variances_results[name].append(('Собственная', variances))
        variances_results[name].append(('sklearn', variances_sklearn))

    # Вывод средних значений
    print("\nСредние значения метрик по 30 запускам:")
    print(
        "| Датасет         | Реализация  | Точность (%) | Точность (Prec) (%) | Полнота (%) | F1-мера (%) | Специфичность (%) | Время (с) |")
    print(
        "|-----------------|-------------|--------------|---------------------|-------------|-------------|-------------------|-----------|")
    for name, result in results.items():
        for impl, metrics in result:
            print(
                f"| {name:<15} | {impl:<11} | {metrics['accuracy'] * 100:>12.1f} | {metrics['precision'] * 100:>19.1f} | {metrics['recall'] * 100:>11.1f} | {metrics['f1_score'] * 100:>11.1f} | {metrics['specificity'] * 100:>17.1f} | {metrics['time']:>9.2f} |")

    # Вывод дисперсий
    print("\nДисперсии метрик по 30 запускам:")
    print(
        "| Датасет         | Реализация  | Точность     | Точность (Prec)     | Полнота     | F1-мера     | Специфичность     | Время     |")
    print(
        "|-----------------|-------------|--------------|---------------------|-------------|-------------|-------------------|-----------|")
    for name, result in variances_results.items():
        for impl, variances in result:
            print(
                f"| {name:<15} | {impl:<11} | {variances['accuracy']:>12.6f} | {variances['precision']:>19.6f} | {variances['recall']:>11.6f} | {variances['f1_score']:>11.6f} | {variances['specificity']:>17.6f} | {variances['time']:>9.6f} |")


if __name__ == "__main__":
    main()