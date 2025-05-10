import time
from data.load_dataset import load_iris_dataset
from forest.random_forest import RandomForestC45
from predict_with_cache import RandomForestC45Cached
from sklearn.model_selection import train_test_split

# Загружаем датасет
X, y, attribute_types, feature_names, class_names = load_iris_dataset()

# Разделяем данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализируем леса
forest_no_cache = RandomForestC45(n_estimators=10, sample_ratio=1, random_state=42)
forest_cached = RandomForestC45Cached(n_estimators=10, sample_ratio=1, random_state=42)

# Обучаем леса
print("=== Обучение лесов ===")
start = time.time()
forest_no_cache.fit(X_train, y_train, attribute_types)
print(f"⏱️ Обучение без кэша: {time.time() - start:.4f} сек")

start = time.time()
forest_cached.fit(X_train, y_train, attribute_types)
print(f"⚡ Обучение с кэшем: {time.time() - start:.4f} сек")

# Тестируем предсказания
print("\n=== Предсказания без кэша ===")
start = time.time()
_ = forest_no_cache.predict(X_test)
print(f"⏱️ Без кэша: {time.time() - start:.4f} сек")

print("\n=== Предсказания с кэшем ===")
start = time.time()
_ = forest_cached.predict(X_test)
print(f"⚡ С кэшем: {time.time() - start:.4f} сек")