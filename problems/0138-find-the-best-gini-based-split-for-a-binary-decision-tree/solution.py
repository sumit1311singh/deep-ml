import numpy as np
from typing import Tuple

def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs**2)

def gini_split(y_left, y_right):
    n = len(y_left) + len(y_right)
    g_left = gini_impurity(y_left)
    g_right = gini_impurity(y_right)
    g_weighted = (len(y_left)/n)*g_left + (len(y_right)/n)*g_right
    return g_weighted

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    thresholds = np.unique(X)
    best_feature, best_thresh, best_gini = None, None, 1.0

    for feature in range(X.shape[1]):
        col = X[:, feature]
        thresholds = np.unique(col)

        for t in thresholds:
            mask = col <= t
            y_left, y_right = y[mask], y[~mask]
            g = gini_split(y_left, y_right)
            if g < best_gini:
                best_gini, best_thresh, best_feature = g, t, feature
    return best_feature, best_thresh