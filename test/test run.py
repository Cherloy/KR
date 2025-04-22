import time
from data.load_dataset import load_iris_dataset
from tree.builder import build_tree as build_tree_no_cache
from builder_with_cache import build_tree as build_tree_cached

X, y, attribute_types, feature_names, class_names = load_iris_dataset()

print("=== Строим без кэша ===")
start = time.time()
_ = build_tree_no_cache(X, y, attribute_types)
print(f"⏱️ Без кэша: {time.time() - start:.4f} сек")

print("\n=== Строим с кэшем ===")
start = time.time()
_ = build_tree_cached(X, y, attribute_types)
print(f"⚡ С кэшем: {time.time() - start:.4f} сек")

