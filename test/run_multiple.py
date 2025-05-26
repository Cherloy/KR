import numpy as np
import pandas as pd
import time
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from forest.random_forest import RandomForestC45
from data.load_dataset import load_mushroom_dataset, load_wine_dataset, load_iris_dataset, load_breast_cancer_dataset, load_digits_dataset
from collections import defaultdict

def run_experiment(dataset_name: str, data: List[List[float]], labels: List, attribute_types: List[str], n_runs: int = 30) -> List[dict]:
    results = []

    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=42 + run
        )

        if not X_train or not y_train:
            print(f"Предупреждение: пустая тренировочная выборка для датасета {dataset_name}, прогон {run + 1}")
            continue

        # Обучение модели
        start_time = time.time()
        model = RandomForestC45(
            n_estimators=10,
            sample_ratio=1.0,
            min_samples_split=2,
            max_features='sqrt',
            random_state=42 + run
        )
        model.fit(X_train, y_train, attribute_types)
        y_pred = model.predict(X_test)
        run_time = time.time() - start_time

        # Вычисление матрицы ошибок для специфичности
        cm = confusion_matrix(y_test, y_pred)
        specificity = 0
        for i in range(cm.shape[0]):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity += tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity /= cm.shape[0]

        # Метрики
        metrics = {
            'dataset': dataset_name,
            'run': run + 1,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'specificity': specificity,
            'time': run_time,
        }
        results.append(metrics)

    return results

def save_to_csv(results: List[dict], filename: str = 'experiment_results.csv'):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Результаты сохранены в {filename}")

def compute_stats(metrics: List[dict]) -> dict:
    accuracies = [m['accuracy'] for m in metrics]
    precisions = [m['precision'] for m in metrics]
    recalls = [m['recall'] for m in metrics]
    f1_scores = [m['f1_score'] for m in metrics]
    specificities = [m['specificity'] for m in metrics]
    times = [m['time'] for m in metrics]

    return {
        'mean_accuracy': np.mean(accuracies),
        'var_accuracy': np.var(accuracies, ddof=1),
        'mean_precision': np.mean(precisions),
        'var_precision': np.var(precisions, ddof=1),
        'mean_recall': np.mean(recalls),
        'var_recall': np.var(recalls, ddof=1),
        'mean_f1_score': np.mean(f1_scores),
        'var_f1_score': np.var(f1_scores, ddof=1),
        'mean_specificity': np.mean(specificities),
        'var_specificity': np.var(specificities, ddof=1),
        'mean_time': np.mean(times),
        'var_time': np.var(times, ddof=1),
    }

def main():
    datasets = {
        'Mushroom': load_mushroom_dataset(),
        'Wine': load_wine_dataset(),
        'Iris': load_iris_dataset(),
        'Breast Cancer': load_breast_cancer_dataset(),
        'Digits': load_digits_dataset()
    }

    all_results = []
    summary_results = defaultdict(list)

    # Прогон экспериментов для каждого датасета
    for name, (data, labels, attr_types, _, _) in datasets.items():
        print(f"\nТестирование датасета: {name}")
        results = run_experiment(name, data, labels, attr_types, n_runs=30)
        all_results.extend(results)

        # Вычисление статистики
        stats = compute_stats(results)
        summary_results[name].append(('Собственная', {
            'mean_accuracy': stats['mean_accuracy'],
            'var_accuracy': stats['var_accuracy'],
            'mean_precision': stats['mean_precision'],
            'var_precision': stats['var_precision'],
            'mean_recall': stats['mean_recall'],
            'var_recall': stats['var_recall'],
            'mean_f1_score': stats['mean_f1_score'],
            'var_f1_score': stats['var_f1_score'],
            'mean_specificity': stats['mean_specificity'],
            'var_specificity': stats['var_specificity'],
            'mean_time': stats['mean_time'],
            'var_time': stats['var_time'],
        }))

    save_to_csv(all_results, 'experiment_results.csv')

    # Вывод результатов
    print("\nРезультаты оценки эффективности (средние за 30 прогонов):")
    print(
        "| Датасет         | Реализация  | Точность (%) | Дисп. Точн. | Точность (Prec) (%) | Дисп. Prec | Полнота (%) | Дисп. Полн. | F1-мера (%) | Дисп. F1 | Специфичность (%) | Дисп. Спец. | Время (с) | Дисп. Вр. |"
    )
    print(
        "|-----------------|-------------|--------------|-------------|---------------------|------------|-------------|-------------|-------------|----------|-------------------|-------------|-----------|-----------|"
    )
    for name, result in summary_results.items():
        for impl, metrics in result:
            print(
                f"| {name:<15} | {impl:<11} | {metrics['mean_accuracy'] * 100:>12.1f} | {metrics['var_accuracy'] * 100:>11.3f} | {metrics['mean_precision'] * 100:>19.1f} | {metrics['var_precision'] * 100:>10.3f} | {metrics['mean_recall'] * 100:>11.1f} | {metrics['var_recall'] * 100:>11.3f} | {metrics['mean_f1_score'] * 100:>11.1f} | {metrics['var_f1_score'] * 100:>8.3f} | {metrics['mean_specificity'] * 100:>17.1f} | {metrics['var_specificity'] * 100:>11.3f} | {metrics['mean_time']:>9.2f} | {metrics['var_time']:>9.3f} |"
            )

if __name__ == "__main__":
    main()