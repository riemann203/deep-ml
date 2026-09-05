import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""

    def gini_impurity(class_proportions: np.ndarray) -> float:
        return 1 - np.sum(class_proportions ** 2)

    n_rows, n_cols = X.shape
    answer = None; minimum = float("inf")
    for i in range(n_rows):
        for j in range(n_cols):
            threshold = X[i, j]
            left, right = [], []
            for k in range(n_rows):
                if X[k, j] <= threshold:
                    left.append(y[k])
                else:
                    right.append(y[k])

            if (not left) or (not right):
                continue

            left_proportions = np.array([left.count(0) / len(left), left.count(1) / len(left)])
            right_proportions = np.array([right.count(0) / len(right), right.count(1) / len(right)])

            weighted_gini_impurity = len(left) / n_rows * gini_impurity(left_proportions) + len(right) / n_rows * gini_impurity(right_proportions)
            if weighted_gini_impurity < minimum:
                minimum = weighted_gini_impurity
                answer = (j, threshold)

    return answer